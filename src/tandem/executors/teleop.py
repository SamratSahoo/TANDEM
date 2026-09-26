"""A person driving the arm: TANDEM's own executor for a human phase, and the one every profile starts with.

The work is ``tandem.teleop.child.TeleopChild``'s. It launches the DROID teleop driver, answers its
prompts and follows its events file. This module is the executor protocol around it: it launches one
driver per leg, waits for the leg to be handed back, ends the driver, and reports what reached disk.

`TeleopChild` was written against the session and reads what it needs off it (see its module
docstring). `_ChildHost` supplies the same few attributes from an `ExecutorContext` and a `LegSpec`, so
the child is unchanged and the executor needs no session.

Two behaviours are carried over from the session's hand-off on purpose:

* **A driver that cannot start does not end the leg.** The arm is already released and may already be
  in someone's hands. Taking it back would drive a robot a person is holding, so the problem is
  reported and the leg waits to be handed back like any other. The person can do the step by hand; it
  is then unrecorded, and the loop decides whether that may stand.
* **The frame count is read after the driver exits, not before.** Ending the driver only writes a line
  to its stdin. The count arrives with ``rollout_saved``, which the driver emits after muxing its videos,
  and waiting for the process is what drains it. Asking earlier gets "nothing was recorded" every time.
"""

from __future__ import annotations

import logging
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tandem.core.errors import TandemError
from tandem.executors.base import (
    CustodyError,
    ExecutorContext,
    ExecutorFactory,
    HumanPhaseRequest,
    HumanPhaseResult,
)
from tandem.planners.base import LegSpec
from tandem.teleop.child import TeleopChild

DISPLAY_NAME = "Teleoperation"
SUMMARY = "A person drives the arm with a VR controller or a SpaceMouse, through the DROID teleop driver."
REQUIREMENTS = (
    "teleop.enabled is true",
    "the teleop runtime (`tandem executors install teleop`), or teleop.python and teleop.droid_dir naming a "
    "DROID environment and checkout of your own",
    "a VR headset and controller, or a SpaceMouse (teleop.device)",
)


def unmet_requirements(settings: Any) -> list[str]:
    """What this machine lacks for teleop. Reads settings and checks two paths; starts nothing.

    The hardware itself cannot be checked without opening it, so that requirement is never reported
    here. A missing headset surfaces as the driver's own error, at the start of the leg.
    """
    teleop = settings.teleop
    unmet = []
    if not teleop.enabled:
        unmet.append("teleop is not enabled (`tandem executors install teleop` turns it on)")
    if not (teleop.python or teleop.droid_dir):
        # The runtime tandem builds. Reading its record is cheap and starts nothing.
        from tandem.teleop import recipe

        if not recipe.runtime(settings).is_ready():
            unmet.append(f"the teleop runtime is not installed (`{recipe.INSTALL_COMMAND}`)")
        return unmet
    # A DROID checkout and environment of the user's own, named by both settings.
    python = Path(teleop.python).expanduser() if teleop.python else None
    if python is None or not python.is_file():
        unmet.append(
            f"teleop.python is not an interpreter on this machine ({teleop.python or 'unset'}); "
            "set it with `tandem config set teleop.python /path/to/droid/env/bin/python`"
        )
    droid_dir = Path(teleop.droid_dir).expanduser() if teleop.droid_dir else None
    if droid_dir is None or not droid_dir.is_dir():
        unmet.append(
            f"teleop.droid_dir is not a directory on this machine ({teleop.droid_dir or 'unset'}); "
            "set it with `tandem config set teleop.droid_dir /path/to/droid`"
        )
    return unmet


class TeleopExecutor:
    """Lends the arm to a person through the teleop driver, one driver process per leg."""

    name = "teleop"
    segment_source = "teleop"
    display_name = DISPLAY_NAME
    summary = SUMMARY
    requirements = REQUIREMENTS

    # How often the wait for the leg to be handed back looks at `should_stop`.
    POLL = 0.1
    # Closing the cameras blocks for the SDK teardown, measured at about 14 s for two, and the driver
    # only exits after that. The wait for it to let go of the hardware has to be generous.
    EXIT_GRACE = 60.0
    # After a kill. A driver still alive after this holds the robot and the cameras for good.
    KILL_GRACE = 10.0

    def __init__(self, ctx: ExecutorContext) -> None:
        self.ctx = ctx
        self._lock = threading.Lock()
        self._child: TeleopChild | None = None
        self._killed = threading.Event()

    def run(
        self,
        request: HumanPhaseRequest | None,
        leg: LegSpec,
        *,
        save_root: Path,
        should_stop: Callable[[], bool],
    ) -> HumanPhaseResult:
        """Lend the arm for one leg, and return once the driver has let go of it.

        The leg is stamped from `leg`, the recording contract, and never from `request`. A mismatch
        between the two is a caller's bug that would file this leg under the wrong phase, so it is
        refused before the driver starts.
        """
        if not leg.trajectory_id:
            # The driver treats a leg with no trajectory id as a standalone episode: it is never merged,
            # and the driver blocks at a success/failure prompt nobody answers.
            raise TandemError(
                "A teleop leg needs the trajectory id it belongs to.",
                hint="Build the LegSpec with HumanPhaseRequest.leg_spec, from the id the session minted.",
            )
        if leg.segment_source != self.segment_source:
            # The driver writes "teleop" whatever it is told, so this is a caller that built the leg
            # for something else (LegSpec defaults to the planner's "tamp").
            raise TandemError(
                f"A teleop leg was asked to record as {leg.segment_source!r}.",
                hint="Build the LegSpec with segment_source=executor.segment_source.",
            )
        if request is not None and leg.phase_index is not None and (
            (leg.phase_index, leg.n_phases) != (request.phase_index, request.n_phases)
        ):
            raise TandemError(
                f"The leg is stamped as phase {leg.phase_index} of {leg.n_phases}, but the request is for "
                f"phase {request.phase_index} of {request.n_phases}.",
                hint="Build the LegSpec with HumanPhaseRequest.leg_spec so the two cannot disagree.",
            )

        # `_killed` is NOT cleared here. A forced stop can land before the leg starts -- the arm takes
        # seconds to release -- and clearing it on entry threw that kill away: the driver was launched,
        # opened the robot and the cameras, and the leg came back "done". It is cleared when the leg
        # ends instead (the `finally` below), so each kill ends exactly the leg it was aimed at.
        log = self.ctx.on_log
        if self._killed.is_set():
            self._killed.clear()
            log("tandem", "a forced stop arrived before the teleop driver started; it is not started")
            return HumanPhaseResult("aborted")
        if request is not None and request.stamped:
            log("tandem", f"handing phase {request.phase_index + 1} of {request.n_phases} to a person: "
                          f"{request.description}")
        host = _ChildHost(self.ctx, leg, Path(save_root), self._scratch_dir())
        child = self._start(host, leg)
        with self._lock:
            self._child = child
            # Checked again under the lock `kill` takes: one that landed while the driver was being
            # launched found no child to end, so it is ended here instead of driving the arm.
            killed_while_starting = self._killed.is_set()
        if killed_while_starting and child is not None:
            child.kill()

        try:
            noticed_exit = False
            while not should_stop() and not self._killed.wait(self.POLL):
                if child is not None and not noticed_exit and child.proc is not None:
                    code = child.proc.poll()
                    if code is not None:
                        # Not the end of the leg. The arm is released and may be in someone's hands, so
                        # only the person can say when it is safe to take back. A clean exit is the
                        # driver quitting when asked to from the controller. Anything else is usually a
                        # headset or a camera it could not open, and the person needs to see that now.
                        noticed_exit = True
                        if code == 0:
                            log("tandem", "the teleop driver has finished; return control to take the arm")
                        else:
                            host.report(
                                f"The teleop driver exited with code {code} (its log says why). Return "
                                "control when the arm is safe to take back."
                            )
        except BaseException:
            # Whatever unwound the wait, the caller reaches for the hardware next, and a driver left
            # running still holds it.
            if child is not None:
                child.kill()
                child.wait(timeout=self.KILL_GRACE)
            raise
        else:
            if child is not None:
                self._end(child)
        finally:
            with self._lock:
                self._child = None
                killed = self._killed.is_set()
                self._killed.clear()

        dirs = tuple(Path(directory) for directory, _ in host.saved)
        if killed:
            return HumanPhaseResult("aborted", n_frames=host.n_frames, leg_dir=host.leg_dir, leg_dirs=dirs)
        if host.n_frames:
            log("tandem", f"the teleop leg is part of this episode ({host.n_frames} frames)")
        return HumanPhaseResult("done", n_frames=host.n_frames, leg_dir=host.leg_dir, leg_dirs=dirs)

    def kill(self) -> None:
        self._killed.set()
        with self._lock:
            child = self._child
        if child is not None:
            child.kill()

    # ---- the driver --------------------------------------------------------

    def _settings(self) -> Any:
        if self.ctx.settings is not None:
            return self.ctx.settings
        from tandem.core import settings as settings_mod

        return settings_mod.load()

    def _scratch_dir(self) -> Path:
        """A directory of this leg's own, for the driver's events file.

        One per leg, never one per session. The child follows its events file from the first byte, so a
        file shared between hand-offs replays the previous leg's events into the next: its frame count
        and directory are reported as the new leg's, and its `awaiting_task` spends a start attempt.
        """
        root = Path(self.ctx.session_dir) / "teleop"
        root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="leg-", dir=root))

    def _start(self, host: _ChildHost, leg: LegSpec) -> TeleopChild | None:
        """Launch the driver, or say why not and leave the arm to the person."""
        settings = self._settings()
        if not settings.teleop.enabled:
            host.report(
                "Teleop is not configured, so nothing is driving the arm. Drive it by hand if you like, "
                "then return control."
            )
            return None
        stamp: dict[str, Any] = {}
        if leg.phase_index is not None and leg.n_phases:
            stamp = {
                "phase_index": leg.phase_index,
                "n_phases": leg.n_phases,
                "phase_description": leg.phase_description,
            }
        try:
            child = TeleopChild(host, settings, **stamp).start()
        except Exception as exc:
            host.report(f"Could not start the teleop driver: {exc}")
            return None
        self.ctx.on_log("tandem", "teleop driver started; the arm is yours")
        return child

    def _end(self, child: TeleopChild) -> None:
        """Ask the driver to save and quit, and make sure it is gone before the arm is taken back."""
        if self._killed.is_set():
            child.kill()
        else:
            child.finish()
        if child.wait(timeout=self.EXIT_GRACE):
            return
        self.ctx.on_log("tandem", "the teleop driver has not exited; killing it so the arm can be taken back")
        child.kill()
        if not child.wait(timeout=self.KILL_GRACE):
            raise CustodyError("the teleop driver will not exit, so it still holds the robot and cameras")


class _ChildHost:
    """The part of a session `TeleopChild` reads, supplied from an executor's context and one leg.

    The child takes the trajectory id, the language label, where legs are written, the cameras, and
    somewhere to report. It learns nothing else about the session, and this is all of that.
    """

    def __init__(self, ctx: ExecutorContext, leg: LegSpec, save_root: Path, scratch: Path) -> None:
        self._ctx = ctx
        self.id = leg.trajectory_id
        self._files = {"session_dir": scratch}
        self._trajectory_id = leg.trajectory_id
        # The child falls back to the current rollout's _meta.json for an id. Here the id always comes
        # from the leg, so there is nothing to fall back to.
        self.current = None
        self.instruction = leg.instruction
        self.profile = _ProfileView(ctx.profile, save_root, rig=ctx.rig)
        self._handoff_error: str | None = None
        # Every recording this hand-off saved, as (directory, frames). The child keeps only the last,
        # but the driver starts another recording whenever one ends short of quitting, and every one of
        # them is a leg of this trajectory.
        self.saved: list[tuple[str, int]] = []

    @property
    def handoff_error(self) -> str | None:
        return self._handoff_error

    @handoff_error.setter
    def handoff_error(self, message: str | None) -> None:
        # The child logs what it sets here itself, so this only passes the problem on to be shown.
        self._handoff_error = message
        if message:
            self._ctx.on_problem(message)

    def report(self, message: str) -> None:
        """A problem of the executor's own, logged and shown the way the child's are."""
        self.handoff_error = message
        self._ctx.on_log("tandem", message)

    @property
    def n_frames(self) -> int:
        return sum(frames for _, frames in self.saved)

    @property
    def leg_dir(self) -> Path | None:
        return Path(self.saved[-1][0]) if self.saved else None

    def _log(self, stream: str, text: str) -> None:
        self._ctx.on_log(stream, text)

    def _emit(self, payload: dict) -> None:
        if payload.get("type") == "teleop_event" and payload.get("event") == "rollout_saved":
            frames = int(payload.get("n_frames") or 0)
            if frames and payload.get("dir"):
                self.saved.append((str(payload["dir"]), frames))
        self._ctx.on_emit(payload)

    def _pump(self, stream, name: str) -> None:
        # Never stops before the stream closes: a drain that ends leaves the driver blocked on a full
        # pipe, holding the arm. Decoding cannot fail (the child's pipe is decoded leniently); a log sink
        # that fails is skipped past rather than allowed to end it.
        def drain() -> None:
            try:
                for raw in stream:
                    try:
                        self._ctx.on_log(name, raw.rstrip("\n"))
                    except Exception:
                        logging.getLogger(__name__).exception("the log sink failed on a teleop driver line")
            except (ValueError, OSError):
                pass  # closed under the drain: the driver is gone

        threading.Thread(target=drain, name=f"{name}:{self.id}", daemon=True).start()


class _ProfileView:
    """What the child reads off "the profile": the rig's cameras, and where this executor was told to write.

    The cameras are this machine's rig's, not the profile's: every profile on the machine records from the
    same ones. The context's rig, or this machine's when the context was built without one.
    """

    def __init__(self, profile: Any, save_root: Path, *, rig: Any = None) -> None:
        self._profile = profile
        self._save_root = save_root
        self._rig = rig

    @property
    def rig(self) -> Any:
        if self._rig is None:
            from tandem.core import rig as rig_mod

            self._rig = rig_mod.load()
        return self._rig

    @property
    def cameras(self) -> Any:
        return self.rig.cameras

    def trajectories_dir(self) -> Path:
        return self._save_root


FACTORY = ExecutorFactory(
    create=TeleopExecutor,
    display_name=DISPLAY_NAME,
    summary=SUMMARY,
    segment_source="teleop",
    requirements=REQUIREMENTS,
    check=unmet_requirements,
)
