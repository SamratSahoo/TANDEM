"""The tandem side of a teleop hand-off: the driver process, launched and answered from here.

``driver.py`` runs under the DROID environment's interpreter. This module does not. It runs in
tandem's own process and imports nothing beyond the base install. It launches the driver, answers
the prompts the driver raises over stdin, and follows the driver's events file. That is how the
teleop executor knows when a leg was recorded and how many frames it has.

A `TeleopChild` is given a host and reads from it: the scratch directory (``_files["session_dir"]``),
the trajectory id it stamps the leg with (``_trajectory_id``, falling back to ``current.dir``), where
legs are written (``profile.trajectories_dir()``), the cameras (``profile.cameras``: the rig's, which the
executor's view hands over), and the language label (``instruction``). It reports
back through ``_log``, ``_emit``, ``_pump`` and ``handoff_error``. The host was once the session; it is
now ``executors.teleop._ChildHost``, which supplies exactly those attributes from the executor's
context and one leg, with a scratch directory of its own per leg, so no hand-off reads another's
events file. ``tests/test_teleop_handoff.py`` drives the child against a stub with the same
attributes. Which phase the leg records is not read off the host; whoever launches the child passes
it in.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from tandem.core import events as events_mod
from tandem.core.errors import TandemError

if TYPE_CHECKING:
    from tandem.core.session import Session


class TeleopChild:
    """The teleop driver that holds the arm during a hand-off.

    Runs under the DROID environment's interpreter (a separate install with the VR and camera
    bindings), talks the same events-file + stdin protocol as the planner, and stamps every
    leg it records with the trajectory's lineage id.
    """

    # How many times to answer the driver's task prompt before giving up. It re-prompts after a
    # controller that never came up ("put the headset on and press Start again"), which is a real
    # recoverable state and worth retrying -- but not forever, or a headset left in a drawer spins
    # the arm's owner in a loop with no way out.
    MAX_START_ATTEMPTS = 3

    def __init__(
        self,
        session: Session,
        cfg,
        *,
        phase_index: int | None = None,
        n_phases: int | None = None,
        phase_description: str | None = None,
    ) -> None:
        self.session = session
        self.cfg = cfg
        # Which phase of a phase-planned task this leg records, for the driver to stamp into the
        # leg's _meta.json under the keys the planner's legs use. The merge copies them into
        # segments[], which is the only place a merged demonstration says which leg was which
        # phase. All None for a hand-off with no phase attached (one the operator asked for at a
        # phase boundary, or phase planning off): the keys are then left out, not guessed.
        self.phase_index = phase_index
        self.n_phases = n_phases
        self.phase_description = phase_description
        self.proc: subprocess.Popen | None = None
        self.events_file: Path | None = None
        self._tailer: events_mod.EventTailer | None = None
        self._start_attempts = 0
        self._recording = False
        self.n_frames: int | None = None
        self.leg_dir: str | None = None

    def start(self) -> TeleopChild:
        from tandem import teleop as teleop_pkg

        python, droid_dir, import_paths = resolve_driver(self.cfg)

        session_dir = self.session._files["session_dir"]
        self.events_file = session_dir / "teleop-events.jsonl"
        self.events_file.touch()

        # tandem mints this, so take it from the session rather than reading it back out of a leg's
        # _meta.json. The disk round-trip was correct only while the PLANNER minted the id; now it
        # is stale in every case where no planner leg has been written yet — a plan whose first
        # phase is the person's, an operator-requested hand-off on the first leg, and (the default
        # policy) a phase the planner could not plan, where `current` is set but nothing was ever
        # recorded into it. An unstamped leg is never merged, so the demonstration is orphaned; and
        # the driver, seeing no trajectory id, prompts for a success/failure label nobody answers
        # and holds the robot and cameras until it is killed.
        trajectory_id = self.session._trajectory_id or (
            _trajectory_id_of(Path(self.session.current.dir)) if self.session.current else None
        )

        args = [
            str(python), str(teleop_pkg.driver_path()),
            "--events-file", str(self.events_file),
            "--output-root", str(self.session.profile.trajectories_dir()),
            "--instruction", self.session.instruction,
            "--controller", self.cfg.teleop.controller,
            # The planner parked the arm mid-task on purpose; homing here would undo the whole
            # point of the hand-off and could drop whatever it is holding.
            "--keep-pose",
        ]
        if trajectory_id:
            # These episodes are LEGS of the planner's trajectory, not episodes in their own
            # right: stamped with its id, left unlabeled, and never prompting for a verdict —
            # the verdict belongs to the whole trajectory and is given once, at the end.
            args += ["--trajectory-id", str(trajectory_id)]
        # `is not None`, not truthiness: phase 0 is a real phase, and a person's step is often the
        # first one (a box that has to be opened before anything can go in it).
        if self.phase_index is not None:
            args += ["--phase-index", str(int(self.phase_index))]
        if self.n_phases is not None:
            args += ["--n-phases", str(int(self.n_phases))]
        if self.phase_description is not None:
            args += ["--phase-description", str(self.phase_description)]

        cameras = self.session.profile.cameras.configured()
        for key, flag in (
            ("hand", "--hand-camera-id"),
            ("external", "--external-camera-id"),
            ("external_2", "--external-2-camera-id"),
        ):
            if key in cameras:
                args += [flag, cameras[key].serial]

        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [*(str(p) for p in import_paths), env.get("PYTHONPATH", "")])
        )
        env["TELEOP_EVENTS_FILE"] = str(self.events_file)
        # DROID reads the NUC's address and the wrist camera's serial from droid.misc.parameters, which
        # reads these (DROID's TANDEM branch). Set from the rig, so that file is never edited by hand and
        # never disagrees with robot.host. A checkout without the override ignores them.
        rig = getattr(self.session.profile, "rig", None)
        if rig is not None:
            env["DROID_NUC_IP"] = str(rig.robot.host)
        for key, var in (
            ("hand", "TIPTOP_HAND_CAMERA_ID"),
            ("external", "TIPTOP_EXTERNAL_CAMERA_ID"),
            ("external_2", "TIPTOP_EXTERNAL_2_CAMERA_ID"),
        ):
            if key in cameras:
                env[var] = str(cameras[key].serial)

        self.session._log("tandem", "$ " + " ".join(args))
        self.proc = subprocess.Popen(
            args,
            cwd=str(droid_dir),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            # Leniently: one byte of a camera SDK's output that is not UTF-8 ended the drain, and the
            # driver then blocked on a full pipe holding the arm (planners/rpc.py says more).
            encoding="utf-8",
            errors="backslashreplace",
            bufsize=1,
            start_new_session=True,
        )
        self.session._pump(self.proc.stdout, "teleop")

        self._tailer = events_mod.EventTailer(self.events_file, self._on_event)
        self._tailer.start()
        return self

    def _on_event(self, event: events_mod.Event) -> None:
        """Answer the driver's prompts, and keep the session's view of the leg up to date.

        The driver records NOTHING until it is told to. It parks at its task prompt and drops every
        stdin line that is not ``{"cmd":"start"}``, so a hand-off that never sends one produces an
        episode with no frames in it -- and, because the same prompt swallows ``end_and_quit`` too,
        a driver that never exits and a "return control" that times out waiting for it.
        """
        self.session._emit({"type": "teleop_event", **event.to_dict()})

        if event.name == "awaiting_task":
            self._begin()
        elif event.name == "rollout_start":
            self._recording = True
            self.leg_dir = event.dir
        elif event.name == "rollout_saved":
            self._recording = False
            self.n_frames = int(event.payload.get("n_frames") or 0)
            self.leg_dir = event.dir or self.leg_dir
            self.session._log("tandem", f"teleop leg recorded: {self.n_frames} frames")
        elif event.name == "rollout_aborted":
            self._recording = False
        elif event.name == "awaiting_label":
            # A leg of a task is not a standalone episode, so the driver only prompts for a verdict
            # when it was launched WITHOUT a trajectory id — which now means something went wrong
            # upstream. Answer it anyway: unanswered, the driver blocks there holding the robot and
            # both cameras until it is killed, and a stalled hand-off is a far worse way to find
            # out than a line in the log.
            self.session._log(
                "tandem",
                "the teleop driver asked for a success/failure label, which means this leg was not "
                "stamped as part of the task — answering it so the arm comes back",
            )
            self._write_raw("n")
        elif event.name == "error":
            # The driver is back at its prompt and will accept another `start`; _begin's attempt
            # budget is what stops that becoming a loop.
            self._recording = False
            message = str(event.payload.get("message") or "the teleop driver reported an error")
            self.session.handoff_error = message
            self.session._log("tandem", f"teleop: {message}")

    def _begin(self) -> None:
        """Tell the driver to start recording, if it has not been told already."""
        if self._recording or self.proc is None or self.proc.poll() is not None:
            return
        if self._start_attempts >= self.MAX_START_ATTEMPTS:
            if self._start_attempts == self.MAX_START_ATTEMPTS:
                self._start_attempts += 1  # log the giving-up once, not on every re-prompt
                self.session.handoff_error = (
                    "The teleop driver would not start recording. Drive the arm by hand if you like, "
                    "then return control — but this leg will not be part of the episode."
                )
                self.session._log("tandem", self.session.handoff_error)
            return
        self._start_attempts += 1
        self._write({"cmd": "start"})

    def _write(self, payload: dict) -> None:
        proc = self.proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            return
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass

    def finish(self) -> None:
        """Ask the driver to save what it has and quit.

        `end_and_quit` rather than a bare `q`: mid-recording a bare `q` DISCARDS the episode the
        human just demonstrated. Note it is only honoured MID-RECORDING -- at the task prompt the
        driver drops it and goes on waiting -- so a leg that never started has to be ended with `q`
        instead, or the process never exits and `resume_from_teleop` times out on it.
        """
        if self.proc is None or self.proc.poll() is not None:
            return
        if self._recording:
            self._write({"cmd": "end_and_quit"})
        else:
            self._write_raw("q")

    def _write_raw(self, line: str) -> None:
        proc = self.proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            return
        try:
            proc.stdin.write(line + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass

    def discard(self) -> None:
        """Throw away the leg in flight, leaving the driver ready for another."""
        if self._recording:
            self._write({"cmd": "discard"})

    def wait(self, timeout: float) -> bool:
        if self.proc is None:
            return True
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        finally:
            if self._tailer is not None:
                self._tailer.stop()
        return True

    def kill(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass


def _trajectory_id_of(directory: Path) -> str | None:
    """The lineage id stamped into a rollout's metadata, if it has one."""
    meta = directory / "_meta.json"
    if not meta.is_file():
        return None
    try:
        return json.loads(meta.read_text()).get("trajectory_id")
    except (ValueError, OSError):
        return None


def resolve_driver(cfg) -> tuple[Path, Path, list[Path]]:
    """The interpreter the driver runs under, its working directory, and what goes on its PYTHONPATH.

    ``teleop.python`` and ``teleop.droid_dir``, when either is set: a DROID checkout and environment of
    the user's own, as before tandem managed one. Otherwise the teleop runtime ``tandem executors install
    teleop`` builds (``tandem.teleop.recipe``).
    """
    teleop = cfg.teleop
    if teleop.python or teleop.droid_dir:
        python = Path(teleop.python).expanduser()
        if not teleop.python or not python.is_file():
            raise TandemError(
                f"The configured teleop interpreter does not exist: {teleop.python or 'unset'}",
                hint="Set it with `tandem config set teleop.python /path/to/droid/env/bin/python`, or unset "
                "teleop.python and teleop.droid_dir to use the runtime `tandem executors install teleop` builds.",
            )
        droid_dir = Path(teleop.droid_dir).expanduser()
        if not teleop.droid_dir or not droid_dir.is_dir():
            raise TandemError(
                f"The configured DROID checkout does not exist: {teleop.droid_dir or 'unset'}",
                hint="Set it with `tandem config set teleop.droid_dir /path/to/droid`, or unset teleop.python and "
                "teleop.droid_dir to use the runtime `tandem executors install teleop` builds.",
            )
        return python, droid_dir, [droid_dir]

    from tandem.teleop import recipe

    rt = recipe.runtime(cfg)
    rt.require_ready()
    return rt.python(), rt.source_dir("droid"), recipe.pythonpath(rt)
