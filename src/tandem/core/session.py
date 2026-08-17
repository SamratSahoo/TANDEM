"""The human-in-the-loop session engine.

One long-lived driver process per session. It warms up once — cuRobo, SAM2, the cameras, the
robot client — and then loops rollouts against that warm state, which is why a session is
kept alive across a bad episode rather than restarted.

Threads and a callback bus, no asyncio in here, so the identical object drives both
`tandem collect` (synchronous, Rich Live) and `tandem ui` (FastAPI, bridged to SSE). The
state machine exists once.

    spawning → warming → rolling → awaiting_label → labeling → awaiting_task → rolling …
                            │
                        SIGUSR1
                            ↓
                     handing_off → teleop_handoff ──"resume"──→ warming → rolling

Three behaviours are load-bearing and were each learned expensively upstream:

* **Preempt is not stop.** SIGINT to the process group aborts the in-flight rollout and
  returns the driver to its task prompt with everything still warm. It sets no end reason
  and schedules no kill escalation; only `stop()` ends a session.
* **Stopping parks the arm.** The driver never homes on its own at the end of a rollout, so
  `stop()` writes `home\\n` before `q\\n` and widens its kill grace to cover the move.
* **Signals go to the process group.** `pixi run` is a wrapper; the process actually holding
  the robot is its grandchild, and a signal to the direct child never reaches it.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from tandem.core import events as events_mod
from tandem.core import paths, render, secrets
from tandem.core import settings as settings_mod
from tandem.core.errors import SessionConflict, TandemError
from tandem.core.events import Event
from tandem.core.profiles import Profile
from tandem.core.runtime import Runtime

# How long to let the arm's park-and-exit move run before escalating to SIGTERM. The
# ordinary grace is a couple of seconds, which would strand the arm part-way home.
HOME_EXIT_GRACE = 45.0
TERM_GRACE = 8.0
LOG_BUFFER = 4000
# Closing the ZEDs blocks for the SDK teardown — measured at ~14 s for two cameras — and the
# driver only exits after that, so the wait for a teleop child to release the hardware has to
# be generous.
TELEOP_EXIT_GRACE = 60.0


class State(str, Enum):
    SPAWNING = "spawning"
    WARMING = "warming"
    ROLLING = "rolling"
    AWAITING_LABEL = "awaiting_label"
    LABELING = "labeling"
    AWAITING_TASK = "awaiting_task"
    HANDING_OFF = "handing_off"
    TELEOP_HANDOFF = "teleop_handoff"
    QUITTING = "quitting"
    STOPPED = "stopped"
    FAILED = "failed"


TERMINAL = frozenset({State.STOPPED, State.FAILED})
# During a hand-off the driver is parked with its robot client released and its cameras
# closed. SIGINT there raises out of the wait and strands it; "return control" is the way out.
NO_PREEMPT = frozenset({State.HANDING_OFF, State.TELEOP_HANDOFF})


@dataclass
class LogLine:
    stream: str  # "stdout" | "stderr" | "tandem"
    text: str
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"stream": self.stream, "text": self.text, "at": self.at}


@dataclass
class RolloutRecord:
    dir: str
    started_at: float
    n_frames: int = 0
    status: str | None = None
    success: bool | None = None

    def to_dict(self) -> dict:
        return {
            "dir": self.dir,
            "id": Path(self.dir).name,
            "started_at": self.started_at,
            "n_frames": self.n_frames,
            "status": self.status,
            "success": self.success,
        }


class Session:
    """A collection session: one driver process, driven by a human."""

    def __init__(
        self,
        profile: Profile,
        runtime: Runtime,
        *,
        task: str | None = None,
        execute: bool = True,
        record: bool | None = None,
        max_episodes: int | None = None,
        session_id: str | None = None,
    ) -> None:
        self.id = session_id or uuid.uuid4().hex[:12]
        self.profile = profile
        self.runtime = runtime
        self.task = task or profile.goal_or_prompt()
        self.execute = execute
        self.record = profile.recording.enabled if record is None else record
        self.max_episodes = max_episodes

        self.state: State = State.SPAWNING
        self.error: str | None = None
        self.end_reason: str | None = None
        self.started_at = time.time()
        self.ended_at: float | None = None

        self.rollouts: list[RolloutRecord] = []
        self.current: RolloutRecord | None = None
        self.labeled_count = 0
        self.success_count = 0
        # A hand-off is armed but not yet honoured; the driver takes it at its next plan-step
        # boundary, so the arm parks at a sane place rather than mid-motion.
        self.teleop_pending = False
        self.handoff_error: str | None = None

        self._proc: subprocess.Popen | None = None
        self._pgid: int | None = None
        self._tailer: events_mod.EventTailer | None = None
        self._logs: deque[LogLine] = deque(maxlen=LOG_BUFFER)
        self._subscribers: list[Callable[[dict], None]] = []
        self._lock = threading.RLock()
        self._exit_watcher: threading.Thread | None = None
        self._files: dict = {}
        self._stopping = False
        self._teleop: TeleopChild | None = None

    # ---- lifecycle ---------------------------------------------------------

    def start(self) -> Session:
        self.runtime.require_ready()
        if not secrets.gemini_api_key():
            raise TandemError(
                "No Gemini API key is set, and perception needs one every rollout.",
                hint="Run `tandem config set-gemini-key`.",
            )

        if not self.profile.cameras.configured():
            raise TandemError(
                f"Profile {self.profile.name!r} has no cameras configured, so nothing can be "
                "perceived or recorded.",
                hint="Add them with `tandem profile edit`, or import a rig with "
                "`tandem profile create <name> --import-from <checkout>`.",
            )

        problems = render.check_assets(self.profile, runtime_dir=self.runtime.root)
        fatal = [p for p in problems if p.startswith("no camera extrinsics")]
        if fatal:
            raise TandemError(
                "\n".join(fatal),
                hint="Extrinsics are keyed by camera serial; add them before collecting.",
            )
        for problem in problems:
            self._log("tandem", f"warning: {problem}")

        self._files = render.prepare_session_files(self.profile, self.id, runtime_dir=self.runtime.root)
        env = render.render_env(
            self.profile,
            events_file=self._files["events_file"],
            task=self.task,
            runtime_dir=self.runtime.root,
        )

        args = ["tiptop-run", "--output-dir", str(self.profile.trajectories_dir())]
        args.append("--enable-recording" if self.record else "--no-enable-recording")
        if not self.execute:
            args.append("--no-execute-plan")
        overrides = self._files.get("overrides_file")
        if overrides is not None:
            args += ["--curobo-overrides", str(overrides)]

        cmd = self.runtime.command(args)
        self._log("tandem", "$ " + " ".join(cmd))

        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(self.runtime.tiptop_dir),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                # Its own process group: `pixi run` is a wrapper, and every signal we send
                # has to reach the grandchild that actually holds the robot.
                start_new_session=True,
            )
        except OSError as exc:
            self._fail(f"could not start the driver: {exc}")
            raise TandemError(
                f"Could not start the collection driver: {exc}",
                hint="Run `tandem doctor` to check the runtime.",
            ) from exc

        self._pgid = os.getpgid(self._proc.pid)
        self._set_state(State.WARMING)

        self._pump(self._proc.stdout, "stdout")
        self._pump(self._proc.stderr, "stderr")

        self._tailer = events_mod.EventTailer(self._files["events_file"], self._on_event)
        self._tailer.start()

        self._exit_watcher = threading.Thread(target=self._watch_exit, name=f"exit:{self.id}", daemon=True)
        self._exit_watcher.start()
        return self

    def _watch_exit(self) -> None:
        assert self._proc is not None
        code = self._proc.wait()
        # Let the tailer catch a session_end written just before exit.
        time.sleep(0.2)
        if self._tailer is not None:
            self._tailer.stop()
        with self._lock:
            if self.state in TERMINAL:
                return
            if code == 0 or self._stopping:
                self._set_state(State.STOPPED)
            else:
                self._fail(f"the driver exited with code {code}")

    # ---- human actions -----------------------------------------------------

    def label(self, success: bool) -> None:
        """Answer the success/failure prompt."""
        self._require(State.AWAITING_LABEL, "label a rollout")
        self._write("y" if success else "n")
        self._set_state(State.LABELING)

    def next_task(self, task: str | None = None) -> None:
        """Answer the task prompt: a new task, or blank to repeat the last one."""
        self._require(State.AWAITING_TASK, "start another rollout")
        text = (task or "").strip()
        if text:
            self.task = text
        # A bare newline repeats the previous task, which is what an operator collecting 20
        # episodes of the same thing wants.
        self._write(text)
        self._set_state(State.ROLLING)

    def preempt(self) -> None:
        """Abort the rollout in flight, keeping the session warm.

        This stops us sending further plan steps. It cannot stop the arm: the controller was
        handed a whole trajectory segment in one request and has no abort, so the motion runs
        to the end of that segment. The physical E-stop is the only instant stop.
        """
        with self._lock:
            if self.state in TERMINAL:
                raise SessionConflict("The session has already ended.")
            if self.state in NO_PREEMPT:
                raise SessionConflict(
                    "A teleop hand-off is in progress, so preempting would strand the driver "
                    "with no robot and no cameras.",
                    hint='Use "return control to TAMP" to finish the hand-off first.',
                )
        self._log("tandem", "preempt: SIGINT to the driver's process group")
        self._signal(signal.SIGINT)

    def request_teleop(self) -> None:
        """Ask for the arm. Cooperative: nothing is aborted.

        The driver honours it at the end of the current plan step, so the arm parks at a plan
        boundary still holding whatever it was holding.
        """
        with self._lock:
            if self.state in TERMINAL:
                raise SessionConflict("The session has already ended.")
            if self.state in NO_PREEMPT:
                raise SessionConflict("A hand-off is already in progress.")
            self.teleop_pending = True
        self._log("tandem", "teleop switch requested (SIGUSR1)")
        self._signal(signal.SIGUSR1)
        self._emit({"type": "teleop_requested"})

    def resume_from_teleop(self) -> None:
        """Hand the arm back.

        Waits for the teleop child's **process to exit**, not merely for a terminal state: the
        driver emits its last event from a `finally` block and releases the robot and cameras
        after it, so writing `resume` any earlier hands the planner hardware somebody still
        holds. Only then does the planner re-open the cameras, reconnect, and replan the same
        task from wherever the human left the arm — no homing, no gripper open.
        """
        self._require(State.TELEOP_HANDOFF, "return control")

        teleop = self._teleop
        if teleop is not None:
            self._log("tandem", "ending the teleop session and waiting for it to release the hardware")
            teleop.finish()
            if not teleop.wait(timeout=TELEOP_EXIT_GRACE):
                raise SessionConflict(
                    "The teleop process has not exited, so it still holds the robot and cameras.",
                    hint="Give it a moment and try again, or stop it by hand.",
                )
            self._teleop = None

        self._write("resume")
        self._set_state(State.WARMING)

    def _start_teleop(self) -> None:
        """Launch the teleop driver once the planner has released the hardware.

        Failure here is not fatal to the session: the planner stays parked at its hand-off
        wait with `handoff_error` set, which is recoverable, rather than grabbing an arm the
        operator may be holding.
        """
        cfg = settings_mod.load()
        if not cfg.teleop.enabled:
            self.handoff_error = (
                "Teleop is not configured, so nothing is driving the arm. Drive it by hand if you "
                "like, then return control."
            )
            self._log("tandem", self.handoff_error)
            return
        try:
            self._teleop = TeleopChild(self, cfg).start()
            self._log("tandem", "teleop driver started; the arm is yours")
        except Exception as exc:
            self.handoff_error = f"Could not start the teleop driver: {exc}"
            self._log("tandem", self.handoff_error)

    def stop(self, *, park: bool = True) -> None:
        """End the session gracefully, parking the arm on the way out."""
        with self._lock:
            if self.state in TERMINAL or self._stopping:
                return
            self._stopping = True
            self.end_reason = "stop"
            state = self.state

        # The driver's reset runs at the START of a rollout, so ending a session otherwise
        # leaves the arm wherever the last plan put it. stdin is a pipe: it reads `home`,
        # runs the blocking move, then reads the buffered `q`. Only legal at the task prompt
        # -- mid-rollout nothing is reading stdin, and during a hand-off the driver has
        # released its robot client. The gripper is deliberately NOT opened: nothing here can
        # know the arm is not holding something.
        grace = TERM_GRACE
        if park and state == State.AWAITING_TASK:
            self._log("tandem", "parking the arm before exit")
            self._write("home")
            grace = HOME_EXIT_GRACE

        self._set_state(State.QUITTING)
        self._write("q")
        threading.Thread(target=self._escalate, args=(grace,), name=f"stop:{self.id}", daemon=True).start()

    def force_stop(self) -> None:
        """Hard stop, for when the driver is wedged. Not the same as preempt."""
        with self._lock:
            self._stopping = True
            self.end_reason = "force-stop"
        self._log("tandem", "force stop")
        self._signal(signal.SIGTERM)
        threading.Thread(target=self._escalate, args=(3.0,), name=f"kill:{self.id}", daemon=True).start()

    def _escalate(self, grace: float) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            pass
        self._log("tandem", f"driver still alive after {grace:.0f}s — SIGTERM")
        self._signal(signal.SIGTERM)
        try:
            proc.wait(timeout=TERM_GRACE)
            return
        except subprocess.TimeoutExpired:
            pass
        self._log("tandem", "driver still alive — SIGKILL")
        self._signal(signal.SIGKILL)

    # ---- plumbing ----------------------------------------------------------

    def _require(self, expected: State, action: str) -> None:
        with self._lock:
            if self.state is expected:
                return
            if self.state in TERMINAL:
                raise SessionConflict(f"Cannot {action}: the session has ended ({self.state.value}).")
            raise SessionConflict(
                f"Cannot {action} while the session is {self.state.value}.",
                hint=f"That is only possible at {expected.value}.",
            )

    def _signal(self, sig: int) -> None:
        if self._pgid is None or self._proc is None or self._proc.poll() is not None:
            return
        try:
            os.killpg(self._pgid, sig)
        except (ProcessLookupError, PermissionError) as exc:
            self._log("tandem", f"could not signal the driver: {exc}")

    def _write(self, line: str) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            raise SessionConflict("The driver is not accepting input any more.")
        try:
            proc.stdin.write(line + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise SessionConflict(f"The driver closed its input: {exc}") from exc

    def _pump(self, stream, name: str) -> None:
        def run() -> None:
            try:
                for raw in stream:
                    self._log(name, raw.rstrip("\n"))
            except (ValueError, OSError):
                pass

        threading.Thread(target=run, name=f"{name}:{self.id}", daemon=True).start()

    def _log(self, stream: str, text: str) -> None:
        line = LogLine(stream=stream, text=secrets.redact(text))
        with self._lock:
            self._logs.append(line)
        self._emit({"type": "log", **line.to_dict()})

    # ---- state machine -----------------------------------------------------

    def _on_event(self, event: Event) -> None:
        with self._lock:
            name = event.name

            if name == "session_start":
                self._set_state(State.WARMING, locked=True)

            elif name == "rollout_start":
                self.current = RolloutRecord(dir=event.dir or "", started_at=event.at)
                self._set_state(State.ROLLING, locked=True)

            elif name == "rollout_saved":
                if self.current is not None:
                    self.current.n_frames = int(event.payload.get("n_frames") or 0)
                    if event.dir:
                        self.current.dir = event.dir

            elif name == "awaiting_label":
                if self.current is None:
                    self.current = RolloutRecord(dir=event.dir or "", started_at=event.at)
                elif event.dir:
                    self.current.dir = event.dir
                self._set_state(State.AWAITING_LABEL, locked=True)

            elif name == "labeled":
                success = bool(event.payload.get("success"))
                record = self.current or RolloutRecord(dir=event.dir or "", started_at=event.at)
                if event.dir:
                    record.dir = event.dir
                record.success = success
                record.status = "success" if success else "failure"
                self.rollouts.append(record)
                self.current = None
                self.labeled_count += 1
                self.success_count += int(success)
                # `labeled` is the only unambiguous end of a trajectory — until it arrives the
                # operator could always hand off again — so a multi-leg hand-off is joined here.
                # Fire and forget: the driver is already back at its task prompt, and a merge
                # of several GB of video must not hold up the next rollout.
                trajectory_id = event.payload.get("trajectory_id") or (
                    self.rollouts[-1].dir and _trajectory_id_of(Path(record.dir))
                )
                if trajectory_id:
                    threading.Thread(
                        target=self._merge_trajectory,
                        args=(str(trajectory_id), record.status),
                        name=f"merge:{self.id}",
                        daemon=True,
                    ).start()

            elif name in ("rollout_aborted", "rollout_discarded"):
                self.current = None
                self._set_state(State.AWAITING_TASK, locked=True)

            elif name == "awaiting_task":
                self.current = None
                self._set_state(State.AWAITING_TASK, locked=True)

            elif name == "teleop_switch_pending":
                self.teleop_pending = True
                self._set_state(State.HANDING_OFF, locked=True)

            elif name == "teleop_handoff_start":
                self._set_state(State.HANDING_OFF, locked=True)

            elif name == "teleop_handoff_warning":
                self.handoff_error = str(event.payload.get("message") or "")

            elif name == "awaiting_teleop_resume":
                # The planner has released the robot and closed its cameras; only now can
                # anything else open them.
                self.teleop_pending = False
                self._set_state(State.TELEOP_HANDOFF, locked=True)
                threading.Thread(
                    target=self._start_teleop, name=f"teleop:{self.id}", daemon=True
                ).start()

            elif name == "teleop_handoff_done":
                self.handoff_error = None
                self._set_state(State.WARMING, locked=True)

            elif name == "session_end":
                if self.state not in TERMINAL:
                    self._set_state(State.STOPPED, locked=True)

        self._emit({"type": "event", **event.to_dict()})

        if self.max_episodes and self.labeled_count >= self.max_episodes and self.state == State.AWAITING_TASK:
            self._log("tandem", f"reached {self.max_episodes} episode(s); stopping")
            self.stop()

    def _merge_trajectory(self, trajectory_id: str, status: str | None) -> None:
        """Join a hand-off's legs into one trajectory, in the background.

        Failure here must never take down a session: the legs are untouched on disk (the merge
        never partially writes) and `tandem traj merge` can retry once the cause is fixed.
        """
        from tandem.core import merge as merge_mod

        try:
            result = merge_mod.merge(
                self.profile, trajectory_id, status=status, runtime_dir=self.runtime.root
            )
        except Exception as exc:
            self._log("tandem", f"could not merge trajectory {trajectory_id}: {exc}")
            self._log("tandem", f"the legs are intact; retry with: tandem traj merge {trajectory_id}")
            return
        if result.get("merged"):
            self._log(
                "tandem",
                f"merged {result['n_legs']} legs into one trajectory "
                f"({result['n_frames']} frames) at {result['dir']}",
            )
            self._emit({"type": "merged", **result})

    def _set_state(self, state: State, *, locked: bool = False) -> None:
        if not locked:
            self._lock.acquire()
        try:
            if self.state is state or self.state in TERMINAL:
                return
            self.state = state
            if state in TERMINAL:
                self.ended_at = time.time()
        finally:
            if not locked:
                self._lock.release()
        self._emit({"type": "state", "state": state.value, **self.summary()})

    def _fail(self, message: str) -> None:
        with self._lock:
            self.error = message
            self.state = State.FAILED
            self.ended_at = time.time()
        self._log("tandem", f"session failed: {message}")
        self._emit({"type": "state", "state": State.FAILED.value, **self.summary()})

    # ---- observation -------------------------------------------------------

    def subscribe(self, callback: Callable[[dict], None]) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def _emit(self, message: dict) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(message)
            except Exception:
                # A broken subscriber (a disconnected browser) must never take down the
                # session driving a physical robot.
                pass

    def logs(self, *, limit: int | None = None) -> list[dict]:
        with self._lock:
            lines = list(self._logs)
        if limit:
            lines = lines[-limit:]
        return [line.to_dict() for line in lines]

    def summary(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "profile": self.profile.name,
                "state": self.state.value,
                "task": self.task,
                "execute": self.execute,
                "record": self.record,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "error": self.error,
                "end_reason": self.end_reason,
                "labeled": self.labeled_count,
                "success": self.success_count,
                "target": self.max_episodes or self.profile.task.target_episodes,
                "current": self.current.to_dict() if self.current else None,
                "rollouts": [r.to_dict() for r in self.rollouts],
                "teleop_pending": self.teleop_pending,
                "teleop_available": settings_mod.load().teleop.enabled,
                "handoff_error": self.handoff_error,
                "can_preempt": self.state not in TERMINAL and self.state not in NO_PREEMPT,
                "events_file": str(self._files.get("events_file", "")),
            }

    @property
    def alive(self) -> bool:
        return self.state not in TERMINAL

    def wait(self, timeout: float | None = None) -> int | None:
        if self._proc is None:
            return None
        try:
            return self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None


class TeleopChild:
    """The teleop driver that holds the arm during a hand-off.

    Runs under the DROID environment's interpreter (a separate install with the VR and camera
    bindings), talks the same events-file + stdin protocol as the planner, and stamps every
    leg it records with the trajectory's lineage id.
    """

    def __init__(self, session: Session, cfg) -> None:
        self.session = session
        self.cfg = cfg
        self.proc: subprocess.Popen | None = None
        self.events_file: Path | None = None
        self._tailer: events_mod.EventTailer | None = None

    def start(self) -> TeleopChild:
        from tandem import teleop as teleop_pkg

        python = Path(self.cfg.teleop.python).expanduser()
        if not python.is_file():
            raise TandemError(
                f"The configured teleop interpreter does not exist: {python}",
                hint="Set it with `tandem config set teleop.python /path/to/droid/env/bin/python`.",
            )
        droid_dir = Path(self.cfg.teleop.droid_dir).expanduser()
        if not droid_dir.is_dir():
            raise TandemError(
                f"The configured DROID checkout does not exist: {droid_dir}",
                hint="Set it with `tandem config set teleop.droid_dir /path/to/droid`.",
            )

        session_dir = self.session._files["session_dir"]
        self.events_file = session_dir / "teleop-events.jsonl"
        self.events_file.touch()

        trajectory_id = _trajectory_id_of(Path(self.session.current.dir)) if self.session.current else None

        args = [
            str(python), str(teleop_pkg.driver_path()),
            "--events-file", str(self.events_file),
            "--output-root", str(self.session.profile.trajectories_dir()),
            "--instruction", self.session.task,
            "--device", self.cfg.teleop.device,
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
            filter(None, [str(droid_dir), env.get("PYTHONPATH", "")])
        )
        env["TELEOP_EVENTS_FILE"] = str(self.events_file)

        self.session._log("tandem", "$ " + " ".join(args))
        self.proc = subprocess.Popen(
            args,
            cwd=str(droid_dir),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self.session._pump(self.proc.stdout, "teleop")

        self._tailer = events_mod.EventTailer(
            self.events_file, lambda event: self.session._emit({"type": "teleop_event", **event.to_dict()})
        )
        self._tailer.start()
        return self

    def finish(self) -> None:
        """Ask the driver to save what it has and quit.

        `end_and_quit` rather than a bare `q`: at the task prompt they are the same, but
        mid-recording a bare `q` DISCARDS the episode the human just demonstrated.
        """
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(json.dumps({"cmd": "end_and_quit"}) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError, AssertionError):
            pass

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


class SessionManager:
    """At most one live session per profile — two drivers would fight over the robot."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()

    def create(self, profile: Profile, runtime: Runtime, **kwargs) -> Session:
        with self._lock:
            live = self.live_for(profile.name)
            if live is not None:
                raise SessionConflict(
                    f"A session is already running for profile {profile.name!r} (state: {live.state.value}).",
                    hint="Stop it before starting another; two drivers cannot share the robot.",
                )
            session = Session(profile, runtime, **kwargs)
            self._sessions[session.id] = session
        session.start()
        return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise TandemError(f"No session {session_id!r}.")
        return session

    def live_for(self, profile_name: str) -> Session | None:
        with self._lock:
            for session in self._sessions.values():
                if session.profile.name == profile_name and session.alive:
                    return session
        return None

    def all(self) -> Iterable[Session]:
        with self._lock:
            return list(self._sessions.values())

    def shutdown(self) -> None:
        for session in self.all():
            if session.alive:
                session.stop()


_manager: SessionManager | None = None


def manager() -> SessionManager:
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager


def _trajectory_id_of(directory: Path) -> str | None:
    """The lineage id stamped into a rollout's metadata, if it has one."""
    meta = directory / "_meta.json"
    if not meta.is_file():
        return None
    try:
        return json.loads(meta.read_text()).get("trajectory_id")
    except (ValueError, OSError):
        return None


def write_session_log(session: Session) -> Path:
    """Persist a finished session for post-mortems."""
    directory = paths.log_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"session-{session.id}.json"
    path.write_text(
        json.dumps({"summary": session.summary(), "logs": session.logs()}, indent=2, default=str) + "\n"
    )
    return path
