"""The human-in-the-loop session engine.

One long-lived session per collection run. It warms a task-and-motion planner once — solvers,
perception models, the cameras, the robot client — and then loops tasks against that warm state,
which is why a session is kept alive across a bad episode rather than restarted.

**tandem drives the task; the planner only plans motion.** A task is broken into an ordered list of
phases (``tandem.planning``), and this engine walks them: a robot phase becomes one call to the
planner backend, a human phase becomes a leg of the human executor (``hitl.human_executor``, a
person at the teleop rig by default), and every leg of one task shares a trajectory
id so they merge into a single episode. The planner is reached through ``tandem.planners`` and is
never modified — it is asked for one thing, "achieve this goal in this scene, and record it".

That is the difference from what this replaced, where the phase plan lived inside a fork of the
planner's own process. tandem could see a rollout start and had to guess which phase it belonged to;
it could not reorder a phase, retry one, or hand a person a step the planner turned out not to be
able to do. Now it holds the plan.

The walk itself is ``tandem.core.phase_loop``, and so is the custody transfer of a hand-off: the
loop releases the arm, lets the human executor (``tandem.executors``) carry out the leg, and takes
the arm back. This module is everything around it: the state machine, the prompts a person
answers, what the executor is built with, and the label. Filing and merging the finished legs is
``tandem.core.episodes``.

Threads and a callback bus, no asyncio out here, so the identical object drives both
`tandem collect` (synchronous, Rich Live) and `tandem ui` (FastAPI, bridged to SSE). The state
machine exists once.

    spawning → warming → rolling → awaiting_label → labeling → awaiting_task → rolling …
                            │  │
                            │  └── excluded: filed with no label ──→ awaiting_task
                            ├── a phase only a person can do
                            ↓
                     awaiting_human_phase → handing_off → teleop_handoff ──"resume"──→ rolling

Three behaviours are load-bearing and were each learned expensively:

* **Preempt is not stop.** It abandons the task attempt and returns to the task prompt with
  everything still warm. It sets no end reason and ends no session; only `stop()` does that. It
  cannot stop the arm mid-motion: the planner is handed a whole trajectory segment in one request
  and has no abort, so the motion runs to the end of that segment. The physical E-stop is the only
  instant stop.
* **Stopping parks the arm.** Nothing homes at the end of a task, so `stop()` asks the backend to
  park before closing it. The gripper is deliberately NOT opened — nothing here can know the arm is
  not holding something.
* **A hand-off is a custody transfer.** The planner holds the robot and the cameras exclusively, so
  every human leg is bracketed by `release_hardware()` / `reacquire_hardware()`, and the executor
  (the teleop driver, by default) is not started until the release has actually completed. That
  bracket is the phase loop's (``PhaseLoop._lend_arm``); the session supplies the states it passes
  through and the "return control" that ends it (`_SessionIO.arm_lent`).
"""

from __future__ import annotations

import functools
import json
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from tandem import executors
from tandem.core import episodes, paths, render, secrets
from tandem.core import settings as settings_mod
from tandem.core.errors import SessionConflict, TandemError
from tandem.core.phase_loop import TELEOP, HumanPhase, PhaseLoop, TrialOutcome
from tandem.core.profiles import Profile
from tandem.core.runtime import Runtime
from tandem.executors.base import CustodyError, ExecutorContext

LOG_BUFFER = 4000
# How long a caller should wait for `stop()` to finish. It has to cover the session thread
# unwinding, the arm's park-and-exit move, and the planner letting go of the cameras — the backend
# bounds those at HARDWARE_TIMEOUT and the channel at EXIT_GRACE, so this is deliberately longer
# than their sum rather than a guess.
STOP_GRACE = 300.0
# How long a human phase, or a lent arm, waits for the person before the session is considered
# abandoned. Long: the whole point is that somebody is doing something with their hands.
HUMAN_PHASE_TIMEOUT = 3600.0


class State(str, Enum):
    SPAWNING = "spawning"
    WARMING = "warming"
    ROLLING = "rolling"
    AWAITING_LABEL = "awaiting_label"
    LABELING = "labeling"
    AWAITING_TASK = "awaiting_task"
    # Phase planning reached a step only a person can do. The driver is blocked on a prompt
    # showing what to do and what will be checked afterwards.
    AWAITING_HUMAN_PHASE = "awaiting_human_phase"
    HANDING_OFF = "handing_off"
    TELEOP_HANDOFF = "teleop_handoff"
    QUITTING = "quitting"
    STOPPED = "stopped"
    FAILED = "failed"


class _Preempted(Exception):
    """Raised inside the session loop to abandon one task attempt and keep everything warm."""


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
        # Two different strings, deliberately. `task` steers PLANNING (the goal), `instruction` is
        # the language label stored with the episode and exported as the LeRobot task. A profile
        # that sets `task.goal` says they must differ -- and the planner used to be handed both,
        # through $TIPTOP_TASK and $TIPTOP_INSTRUCTION. Upstream dropped the second, so tandem
        # writes the label itself now; conflating them would stamp every episode with the goal.
        self.task = task or profile.goal_or_prompt()
        self.instruction = task or profile.task.prompt or self.task
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
        # Trials the phase loop excluded (a human phase that never verified): filed under failure/
        # with their legs, never labeled, and never counted towards `max_episodes`, since they are
        # not part of the dataset the target counts.
        self.excluded_count = 0
        # How the last trial ended, for the operator: set as soon as the attempt ends (so the label
        # prompt can say why the loop stopped it) and settled by the label. None while one runs.
        self.last_trial: dict | None = None
        # A hand-off is armed but not yet honoured; the driver takes it at its next plan-step
        # boundary, so the arm parks at a sane place rather than mid-motion.
        self.teleop_pending = False
        self.handoff_error: str | None = None

        # Phase planning. `human_phase` is set only while the driver is blocked waiting for a
        # person; `phase_progress` survives across the legs of one task so the operator can see
        # how far through it is.
        self.hitl_enabled = profile.hitl.enabled
        self.human_phase: HumanPhase | None = None
        self.phase_progress: tuple[int, int] | None = None
        self.unrepresented: list[dict] = []

        self._logs: deque[LogLine] = deque(maxlen=LOG_BUFFER)
        self._subscribers: list[Callable[[dict], None]] = []
        self._lock = threading.RLock()
        self._files: dict = {}
        self._stopping = False
        self._park_on_exit = True
        # The trial algorithm, built at the first task and kept: it holds the human executors it has
        # built, and building one may start a process. `force_stop` reaches a leg in flight through it.
        self._loop: PhaseLoop | None = None
        self._legs_recorded = 0

        # The planner, and what it says it can be asked for.
        self._backend = None
        self._capabilities = None
        self._planning_cfg = None

        # The lineage id the legs of the task in progress share, and the plan they were walking.
        # Both are tandem's now; they used to live inside the planner's process, where tandem could
        # see neither. The plan itself is the phase loop's while the attempt runs; this is its last
        # plan, kept after the attempt drops it, so the audit record can still be written for a task
        # that was abandoned part-way -- which is exactly the episode whose provenance you want.
        self._last_plan = None
        # How the phase loop said the last attempt ended (`phase_loop.TrialOutcome`), read after it
        # returns or raises. Its `outcome` decides whether the operator is asked for a label at all.
        self._last_outcome: TrialOutcome | None = None
        self._trajectory_id: str = ""
        self._vlm_dir: Path | None = None

        # The session runs on its own thread and blocks on these. Each human action sets one; a
        # stop sets them all, so nothing is left waiting on somebody who has gone home.
        self._worker: threading.Thread | None = None
        self._task_ready = threading.Event()
        self._label_ready = threading.Event()
        self._human_ready = threading.Event()
        self._resume_ready = threading.Event()
        self._teleop_requested = threading.Event()
        self._preempt = threading.Event()
        self._label_answer = False
        self._human_answer = "done"

    # ---- lifecycle ---------------------------------------------------------

    def start(self) -> Session:
        """Preflight, build the planner backend, and hand the session to its own thread.

        The planner's own preflight -- whether its runtime is built, whether its assets and
        calibration are in place -- is the planner's, run by its factory and its ``require_ready``.
        The session checks only what it needs itself whichever planner is named.
        """
        if not secrets.gemini_api_key():
            raise TandemError(
                "No Gemini API key is set, and perception needs one every rollout.",
                hint="Run `tandem config set-gemini-key`.",
            )

        # The teleop legs record from these as well, so they are the session's to insist on.
        if not self.profile.cameras.configured():
            raise TandemError(
                f"Profile {self.profile.name!r} has no cameras configured, so nothing can be "
                "perceived or recorded.",
                hint="Add them with `tandem profile edit`, or import a rig with "
                "`tandem profile create <name> --import-from <checkout>`.",
            )

        if self.hitl_enabled:
            # Resolved now, before anything is warmed: a plugin that will not import, or a name two
            # installed packages claim, is otherwise found at the first human phase, with the robot
            # already part-way through the task. Only described here, not built; the loop builds it.
            executors.info(self.profile.hitl.human_executor)

        self._files = self._session_files()
        self._backend = self._build_backend()
        try:
            self._backend.require_ready()
        except BaseException:
            # Nothing is warmed yet, but a backend is allowed to have taken something in create(),
            # and a session that never starts will never reach _shutdown to give it back.
            backend, self._backend = self._backend, None
            try:
                backend.close()
            except Exception as exc:
                self._log("tandem", f"could not close the planner cleanly: {exc}")
            raise

        self._set_state(State.WARMING)
        self._worker = threading.Thread(target=self._run, name=f"session:{self.id}", daemon=True)
        self._worker.start()
        return self

    def _session_files(self) -> dict:
        """The session's scratch directory, and its events file, created before anything writes."""
        session_dir = paths.session_scratch_dir() / self.profile.name / self.id
        session_dir.mkdir(parents=True, exist_ok=True)
        events_file = session_dir / "events.jsonl"
        # Pre-created, so a tailer can attach before the first event is written.
        events_file.touch()
        return {"session_dir": session_dir, "events_file": events_file}

    def _build_backend(self):
        """The planner this profile names, built by its factory and ready to be warmed.

        Every planner is built the same way, from the same context. What any one of them needs set
        up first -- a runtime, rendered config, environment variables -- is its factory's business,
        which is what lets a planner the session has never heard of be named in a profile.
        """
        from tandem.planners import registry
        from tandem.planners.base import BackendContext

        spec = self.profile.planner
        return registry.create(
            spec.backend,
            BackendContext(
                profile=self.profile,
                session_dir=self._files["session_dir"],
                output_dir=self.profile.trajectories_dir(),
                execute=self.execute,
                record=self.record,
                on_log=self._log,
                options=dict(spec.options),
                settings=settings_mod.load(),
                session_id=self.id,
                task=self.task,
                events_file=self._files["events_file"],
                runtime_dir=self.runtime.root,
            ),
        )

    # ---- the session loop --------------------------------------------------

    def _run(self) -> None:
        """Warm the planner once, then walk one task after another until told to stop."""
        try:
            self._event("session_start")
            self._backend.warm()
            self._capabilities = self._backend.capabilities()
            while not self._stopping:
                if not self._await_task():
                    break
                try:
                    self._run_task()
                except _Preempted:
                    self._log("tandem", "the task attempt was preempted; the planner is still warm")
                    self._event("rollout_aborted")
                except CustodyError:
                    # The robot or the cameras could not be got back (the phase loop's hand-off).
                    # Deliberately not caught as one bad attempt: a session that has lost custody is
                    # not warm and cannot run the next task, so returning to the prompt would leave
                    # an operator staring at a ready-looking session that fails at every rollout --
                    # and an arm nothing will ever park.
                    raise
                except Exception as exc:  # a bad task must not end a warm session
                    self._log("tandem", f"the task attempt failed: {type(exc).__name__}: {exc}")
                    self._event("rollout_aborted", error=str(exc))
                if self.max_episodes and self.labeled_count >= self.max_episodes:
                    self._log("tandem", f"reached {self.max_episodes} episode(s); stopping")
                    break
        except Exception as exc:
            self._fail(f"{type(exc).__name__}: {exc}")
            return
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        """Park the arm and release everything. Runs on every exit path, including a failure."""
        backend = self._backend
        self._backend = None
        if backend is not None:
            # `park` alone decides this. Gating it on a preempt flag as well meant a graceful stop
            # that had just told the operator "parking the arm first" quietly did not, whenever an
            # earlier preempt had left the flag set.
            if self._park_on_exit:
                # Nothing homes at the end of a task, so without this the arm stays wherever the
                # last plan left it. The gripper is deliberately not opened.
                try:
                    self._set_state(State.QUITTING)
                    backend.home()
                except Exception as exc:
                    self._log("tandem", f"could not park the arm on the way out: {exc}")
            try:
                backend.close()
            except Exception as exc:
                self._log("tandem", f"could not close the planner cleanly: {exc}")
        self._event("session_end")
        with self._lock:
            if self.state not in TERMINAL:
                self._set_state(State.STOPPED, locked=True)

    def _run_task(self) -> None:
        """One attempt at the current task, and the label that closes it out.

        The attempt itself (planning the task into phases and walking them) is the phase loop's.
        What stays here is what a person sees around it: the lineage id the legs share, where the
        models' audit trail goes, and the verdict at the end.
        """
        self._set_state(State.ROLLING)
        # tandem mints the lineage id, because only tandem sees both the planner's legs and the
        # teleop ones. It is what joins them into a single episode.
        self._trajectory_id = uuid.uuid4().hex[:16]
        self._legs_recorded = 0
        self._last_plan = None
        self._last_outcome = None
        self.last_trial = None
        # Every image sent to a model this attempt, and what it answered, gathered in one place and
        # filed with the finished episode. Per ATTEMPT rather than per leg: the proposal happens on
        # the first leg and the verifications on later ones, and split across leg directories --
        # half of which are tidied away -- the trail is unreadable. On by default, because when a
        # run goes wrong the question is always what the model saw and what it decided.
        self._vlm_dir = None
        if self.hitl_enabled and self._planning_config().save_vlm_io:
            self._vlm_dir = self._files["session_dir"] / "vlm" / self._trajectory_id

        if self._loop is None:
            self._loop = self._phase_loop()
        loop = self._loop
        try:
            loop.run(
                task=self.task,
                instruction=self.instruction,
                trajectory_id=self._trajectory_id,
                vlm_dir=self._vlm_dir,
            )
        finally:
            # Read off the loop's outcome, which it keeps current as it goes, so it is right on
            # every exit path, including the ones that unwind out of the loop as an exception.
            outcome = loop.outcome
            self._legs_recorded = outcome.legs_recorded
            self._last_plan = outcome.plan
            self._last_outcome = outcome
            self.last_trial = self._trial_summary(None)
            # Every exit path lands here -- finished, abandoned, excluded, rebound onto a scene that
            # no longer matches, or preempted. The rule is the same for all of them: frames on disk
            # have to be filed, because filing is what ends a trajectory and merges its legs.
            # Anything else leaves legs nothing will ever join, filing as episodes of their own.
            # Filing normally waits for the operator's label. An EXCLUDED trial does not: the method
            # has already decided it is not part of the dataset, and a label prompt the operator
            # could answer "success" would put it straight back in.
            if self._legs_recorded and outcome.outcome == "excluded":
                self._file_excluded(outcome)
            elif self._legs_recorded:
                self._await_label()
            else:
                self._log("tandem", "nothing was recorded, so there is nothing to label")
                self._event("rollout_discarded", **self.last_trial)

    def _phase_loop(self) -> PhaseLoop:
        """The trial algorithm, wired to this session's planner, settings and operator."""
        io = _SessionIO(self)
        return PhaseLoop(
            self._backend,
            self._capabilities,
            self._planning_config(),
            events=io,
            operator=io,
            executor_context=self._executor_context(),
            legs=episodes.LegDirs(self.profile, self._files["session_dir"], log=io.log),
            record=self.record,
        )

    # ---- a leg for a person ------------------------------------------------

    def _executor_context(self) -> ExecutorContext:
        """What a human executor is built with: this session's profile, scratch space, log and UI.

        ``settings`` is left unset, so the machine's settings are read afresh at every leg: a
        ``tandem config set teleop.*`` between two hand-offs takes effect at the next one, as it did
        when the session launched the driver itself.
        """
        return ExecutorContext(
            profile=self.profile,
            session_dir=self._files["session_dir"],
            on_log=self._log,
            on_emit=self._emit,
            on_problem=self._handoff_problem,
        )

    def _handoff_problem(self, message: str) -> None:
        """A problem with the leg in flight that the person holding the arm has to see now.

        Shown until the arm is taken back (`_SessionIO.arm_returned`): a driver that would not start
        leaves the arm released and the session parked at the hand-off, which the operator recovers
        from by driving the arm themselves -- rather than the planner grabbing an arm they may already
        be holding.
        """
        self.handoff_error = message

    # ---- waiting for a person ----------------------------------------------

    def _await_task(self) -> bool:
        """Block at the task prompt. False means the session is ending."""
        self.current = None
        self.human_phase = None
        self.phase_progress = None
        self.unrepresented = []
        # Cleared BEFORE the prompt is shown, not after the wait returns. These flags are answers
        # to a specific question, and one set while nobody was asking must not be mistaken for an
        # answer to the next one -- which is how a stray click ended up skipping a whole hand-off.
        self._task_ready.clear()
        self._set_state(State.AWAITING_TASK)
        self._event("awaiting_task")
        while not self._stopping:
            if self._task_ready.wait(timeout=0.2):
                self._task_ready.clear()
                self._preempt.clear()
                return not self._stopping
        return False

    def _await_label(self) -> None:
        """Block at the success/failure prompt, then join the task's legs into one episode."""
        directory = str(self.current.dir) if self.current else ""
        self.current = self.current or RolloutRecord(dir=directory, started_at=time.time())
        self._label_ready.clear()
        self._set_state(State.AWAITING_LABEL)
        self._event("awaiting_label", dir=directory)

        while not self._stopping:
            if self._label_ready.wait(timeout=0.2):
                self._label_ready.clear()
                break
        else:
            return
        if self._stopping:
            return

        success = bool(self._label_answer)
        self._set_state(State.LABELING)
        record = self.current or RolloutRecord(dir=directory, started_at=time.time())
        record.success = success
        record.status = "success" if success else "failure"
        self.rollouts.append(record)
        self.current = None
        self.labeled_count += 1
        self.success_count += int(success)
        # The label settles the trial's outcome, unless the loop had already decided it (an aborted
        # trial stays aborted); the stage the loop stopped at, if it stopped one, goes with it.
        self.last_trial = self._trial_summary(record.status)
        self._event("labeled", dir=record.dir, success=success, **self.last_trial)
        self._file_episode(record.status)

    def _file_excluded(self, outcome: TrialOutcome) -> None:
        """File a trial the phase loop excluded: under failure/, marked excluded, with no label.

        The paper's rule for a human phase that never verified (``on_verification_failure:
        exclude``). The trial is not part of the dataset, but it is not thrown away either: its legs
        are merged exactly as a labeled trial's are, and ``hitl.json`` beside them says it was
        excluded, at which stage, and carries the failing verdicts -- the raw material for working
        out whether the person or the classifier got it wrong.
        """
        directory = str(self.current.dir) if self.current else ""
        record = self.current or RolloutRecord(dir=directory, started_at=time.time())
        # Not labeled, so neither success nor failure as far as the operator's tally goes.
        record.status = "excluded"
        record.success = None
        self.rollouts.append(record)
        self.current = None
        self.excluded_count += 1
        self.last_trial = self._trial_summary("failure")
        self._log(
            "tandem",
            f"this trial is excluded from the dataset ({outcome.failure_stage}): {outcome.reason}. It is "
            "not labeled; its legs are kept under failure/ with excluded: true",
        )
        self._event("trial_excluded", dir=record.dir, **self.last_trial)
        self._file_episode("failure")

    def _trial_summary(self, status: str | None) -> dict:
        """How the attempt just walked ended, as the events, the summary and ``hitl.json`` say it.

        ``status`` is where the episode is filed (``success``/``failure``), or None before it is.
        The outcome is resolved the one way ``hitl.json`` resolves it (``episodes.trial_outcome``), so
        the events file and the record on disk can never disagree about a trial.
        """
        outcome = self._last_outcome
        loop_outcome = outcome.outcome if outcome is not None else None
        return {
            "trajectory_id": self._trajectory_id,
            **episodes.trial_outcome(loop_outcome, status),
            "failure_stage": outcome.failure_stage if outcome is not None else None,
            "reason": outcome.reason if outcome is not None else None,
        }

    def _file_episode(self, status: str) -> None:
        """Move the attempt's legs under ``status``, merge them, and write the record beside them."""
        # Fire and forget: a merge of several GB of video must not hold up the next task.
        trajectory_id = self._trajectory_id
        # The LAST plan, not the one the loop was still walking: an attempt that was abandoned drops
        # its plan, and those are precisely the episodes whose provenance -- which phases ran, what
        # could not be verified, which clauses the run knowingly skipped -- is worth having.
        plan = self._last_plan
        outcome = self._last_outcome
        if trajectory_id:
            threading.Thread(
                target=episodes.merge_trajectory,
                args=(self.profile, trajectory_id, status, plan),
                kwargs={
                    "runtime_dir": self.runtime.root,
                    # This attempt's audit trail, taken now rather than when the merge finishes. A
                    # merge of several GB of video can outlast the start of the next task, which
                    # points `_vlm_dir` at that task's trail instead.
                    "vlm_dir": self._vlm_dir,
                    "log": functools.partial(self._log, "tandem"),
                    "emit": self._emit,
                    "reason": outcome.reason if outcome is not None else None,
                },
                name=f"merge:{self.id}",
                daemon=True,
            ).start()

    def _await_human(self) -> str:
        """What the person answered at a human phase: `done`, `abort`, or `teleop`."""
        # Cleared before the prompt, for the same reason as the others: an answer given to an
        # earlier question is not an answer to this one. Without it a preempt, or a second click
        # during a teardown, silently answered the NEXT human phase on the person's behalf -- so
        # they were never asked to do the step, and it was then verified and marked as not done.
        self._human_ready.clear()
        self._human_answer = "done"
        deadline = time.monotonic() + HUMAN_PHASE_TIMEOUT
        while not self._stopping and time.monotonic() < deadline:
            # Before the wait, not after: a preempt must unwind the attempt rather than be read as
            # somebody saying the step is finished.
            self._check_preempt()
            if self._teleop_requested.is_set():
                self._teleop_requested.clear()
                return "teleop"
            if self._human_ready.wait(timeout=0.2):
                self._human_ready.clear()
                return self._human_answer or "done"
        return "abort"

    def _check_preempt(self) -> None:
        if self._preempt.is_set():
            self._preempt.clear()
            raise _Preempted()

    def _planning_config(self):
        if self._planning_cfg is None:
            self._planning_cfg = self.profile.hitl.to_planning_config(
                cache_path=render.resolve_cache_path(self.profile)
            )
        return self._planning_cfg

    def _event(self, name: str, **payload) -> None:
        """Append one line to the session's own events file, and tell every subscriber.

        The file is tandem's now rather than a planner child's, but it stays a deliberately dumb
        append-only channel for the same reason it always was: when a session goes wrong the useful
        question is what it was doing and in what order, and that has to be answerable from disk
        after the process is gone.
        """
        record = {"event": name, "ts": time.time(), **payload}
        path = self._files.get("events_file")
        if path is not None:
            try:
                with Path(path).open("a") as handle:
                    handle.write(json.dumps(record, default=str) + "\n")
            except OSError:
                pass
        self._emit({"type": "event", **record, "at": record["ts"]})

    # ---- human actions -----------------------------------------------------

    def label(self, success: bool) -> None:
        """Answer the success/failure prompt."""
        self._require(State.AWAITING_LABEL, "label a rollout")
        self._label_answer = success
        self._label_ready.set()

    def next_task(self, task: str | None = None) -> None:
        """Answer the task prompt: a new task, or blank to repeat the last one."""
        self._require(State.AWAITING_TASK, "start another rollout")
        text = (task or "").strip()
        if text:
            # Typed at the prompt, it is both: there is no separate label to keep.
            self.task = text
            self.instruction = text
        self._task_ready.set()

    def complete_human_phase(self) -> None:
        """Tell the session the human step is done, so it can check it and carry on.

        This is the "I did it by hand" answer. Taking the arm through the teleop rig instead is
        `request_teleop()`, which is honoured immediately at this prompt rather than waiting for a
        plan-step boundary that will never come.

        Refused while recording, unless ``hitl.allow_unrecorded_human_phase`` is set: a step done by
        hand has no leg, and the episode would be missing exactly the demonstration the trial exists
        to capture while looking complete. The phase loop refuses it too; refusing it here as well
        tells the person why at the moment they ask, rather than as a prompt that silently comes back.
        """
        with self._lock:
            self._require(State.AWAITING_HUMAN_PHASE, "complete a human phase")
            phase = self.human_phase
            if phase is not None and not phase.by_hand:
                raise SessionConflict(
                    "This step is being recorded, so it has to be done through "
                    f"{_executor_title(phase.executor)} for the episode to have it.",
                    hint="Take the arm and do it there, or give up on the task. To accept steps done by "
                    "hand while recording, set hitl.allow_unrecorded_human_phase: true.",
                )
            self._human_answer = "done"
            self._human_ready.set()

    def abort_human_phase(self) -> None:
        """Give up on this phase, abandoning the task attempt.

        Distinct from a preempt: it is the answer the prompt asks for, so the plan is torn down
        deliberately and says why, rather than unwinding from an interrupt.
        """
        self._require(State.AWAITING_HUMAN_PHASE, "abort a human phase")
        self._human_answer = "abort"
        self._human_ready.set()

    def preempt(self) -> None:
        """Abandon the task attempt, keeping the session warm.

        It cannot stop the arm: the planner is handed a whole trajectory segment in one request and
        has no abort, so the motion runs to the end of that segment. The physical E-stop is the only
        instant stop. What this does is stop asking it for the next one.
        """
        with self._lock:
            if self.state in TERMINAL:
                raise SessionConflict("The session has already ended.")
            if self.state in NO_PREEMPT:
                raise SessionConflict(
                    "A teleop hand-off is in progress, so preempting would strand the session "
                    "with no robot and no cameras.",
                    hint='Use "return control to TAMP" to finish the hand-off first.',
                )
        self._log("tandem", "preempt: abandoning this attempt at the next step boundary")
        self._preempt.set()

    def request_teleop(self) -> None:
        """Ask for the arm. Cooperative: nothing is aborted.

        Honoured at the end of the current step, so the arm parks at a plan boundary still holding
        whatever it was holding.
        """
        with self._lock:
            if self.state in TERMINAL:
                raise SessionConflict("The session has already ended.")
            if self.state in NO_PREEMPT:
                raise SessionConflict("A hand-off is already in progress.")
            self.teleop_pending = True
        self._log("tandem", "teleop switch requested")
        self._teleop_requested.set()
        self._event("teleop_switch_pending")
        self._emit({"type": "teleop_requested"})

    def resume_from_teleop(self) -> None:
        """Hand the arm back.

        The wait for the teleop child to actually exit happens on the session thread, not here:
        the child releases the robot and the cameras after its last event, so the planner must not
        reach for them until its process is gone.
        """
        self._require(State.TELEOP_HANDOFF, "return control")
        self._resume_ready.set()

    def stop(self, *, park: bool = True) -> None:
        """End the session gracefully, parking the arm on the way out."""
        with self._lock:
            if self.state in TERMINAL or self._stopping:
                return
            self._stopping = True
            self.end_reason = "stop"
            self._park_on_exit = park
        self._log("tandem", "stopping" + (" (parking the arm first)" if park else ""))
        # Unblock whatever the session thread is waiting on.
        for flag in (self._task_ready, self._label_ready, self._human_ready, self._resume_ready):
            flag.set()

    def force_stop(self) -> None:
        """Hard stop, for when the session is wedged. Not the same as preempt."""
        # Whatever is driving the arm in a hand-off -- the teleop driver, or a policy -- is ended at
        # once rather than asked to finish, and the leg comes back as aborted. Killed FIRST: every flag
        # below also ends the leg, but as a graceful hand-back, and an executor that sees one of them
        # before the kill may already have returned "done" -- a forced stop recorded as a step carried
        # out, and checked as one.
        loop = self._loop
        if loop is not None:
            loop.kill()
        with self._lock:
            self._stopping = True
            self.end_reason = "force-stop"
        self._log("tandem", "force stop")
        self._park_on_exit = False
        self._preempt.set()
        for flag in (self._task_ready, self._label_ready, self._human_ready, self._resume_ready):
            flag.set()
        backend = self._backend
        if backend is not None:
            threading.Thread(target=backend.close, name=f"kill:{self.id}", daemon=True).start()

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
                "excluded": self.excluded_count,
                "last_trial": self.last_trial,
                "target": self.max_episodes or self.profile.task.target_episodes,
                "current": self.current.to_dict() if self.current else None,
                "rollouts": [r.to_dict() for r in self.rollouts],
                "teleop_pending": self.teleop_pending,
                # Whether a person can take the arm between phases: the teleop executor is ready on
                # this machine. `human_executor` is the same for whoever carries out a human phase.
                "teleop_available": _executor_status(TELEOP)["ready"],
                "human_executor": _executor_status(self.profile.hitl.human_executor),
                "handoff_error": self.handoff_error,
                "can_preempt": self.state not in TERMINAL and self.state not in NO_PREEMPT,
                "events_file": str(self._files.get("events_file", "")),
                "hitl_enabled": self.hitl_enabled,
                "human_phase": self.human_phase.to_dict() if self.human_phase else None,
                "phase_progress": list(self.phase_progress) if self.phase_progress else None,
                "unrepresented": self.unrepresented,
            }

    @property
    def alive(self) -> bool:
        return self.state not in TERMINAL

    def wait(self, timeout: float | None = None) -> int | None:
        """Block until the session thread has finished. None means it is still running."""
        worker = self._worker
        if worker is None:
            return None
        worker.join(timeout=timeout)
        return None if worker.is_alive() else 0


class _SessionIO:
    """The session as the phase loop sees it: an event sink, and the person at the prompts.

    An adapter rather than more methods on Session, so the session's own API stays the set of
    things a person can do (`label`, `next_task`, `complete_human_phase`, ...), and nothing a route
    or the CLI can reach blocks on a prompt meant for the session thread.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ---- EventSink ---------------------------------------------------------

    def event(self, name: str, **payload) -> None:
        self._session._event(name, **payload)

    def log(self, text: str) -> None:
        self._session._log("tandem", text)

    # ---- OperatorIO --------------------------------------------------------

    def check_preempt(self) -> None:
        self._session._check_preempt()

    def take_handoff_request(self) -> bool:
        requested = self._session._teleop_requested
        if not requested.is_set():
            return False
        requested.clear()
        return True

    def rolling(self) -> None:
        self._session._set_state(State.ROLLING)

    def show_progress(self, progress: tuple[int, int] | None) -> None:
        self._session.phase_progress = progress

    def show_unrepresented(self, clauses: list[dict]) -> None:
        self._session.unrepresented = clauses

    def show_human_phase(self, phase: HumanPhase | None) -> None:
        self._session.human_phase = phase

    def await_human_phase(self) -> str:
        self._session._set_state(State.AWAITING_HUMAN_PHASE)
        return self._session._await_human()

    def rollout_started(self, save_dir: Path) -> None:
        self._session.current = RolloutRecord(dir=str(save_dir), started_at=time.time())

    def rollout_saved(self, n_frames: int) -> None:
        if self._session.current is not None:
            self._session.current.n_frames = n_frames

    def handing_off(self) -> None:
        session = self._session
        session._set_state(State.HANDING_OFF)
        # Honoured now, whichever way it was asked for: the switch between phases, or the answer at
        # a human phase's prompt.
        session.teleop_pending = False
        session._event("teleop_handoff_start")

    def arm_lent(self) -> Callable[[], bool]:
        session = self._session
        session._event("awaiting_teleop_resume", trajectory_id=session._trajectory_id)
        # Cleared before the leg, not after it. The state stays TELEOP_HANDOFF for the whole exit
        # sequence, so a second "return control" -- which the UI keeps offering throughout -- used
        # to leave this set with nobody waiting, and the NEXT hand-off then ended instantly while
        # the person was still being told the arm was theirs.
        session._resume_ready.clear()
        session._set_state(State.TELEOP_HANDOFF)
        deadline = time.monotonic() + HUMAN_PHASE_TIMEOUT
        timed_out: list[bool] = []

        def should_stop() -> bool:
            if session._resume_ready.is_set() or session._stopping:
                return True
            if time.monotonic() < deadline:
                return False
            if not timed_out:
                timed_out.append(True)
                session._log(
                    "tandem",
                    f"nobody returned control within {HUMAN_PHASE_TIMEOUT / 60:.0f} minutes; ending the "
                    "leg and taking the arm back",
                )
            return True

        return should_stop

    def arm_returned(self) -> None:
        session = self._session
        # A "return control" that arrived after the leg had already ended must not end the next one.
        session._resume_ready.clear()
        session.handoff_error = None
        session._event("teleop_handoff_done")
        session._set_state(State.ROLLING)


def _executor_status(name: str) -> dict:
    """Whether the human executor ``name`` can run on this machine, for the prompts that offer it.

    Never raises. The summary is read on every change of state, and an executor that cannot even be
    described -- a plugin that will not import -- is simply one the operator cannot be offered, which
    is what ``ready: False`` with the ``error`` says.
    """
    try:
        info = executors.info(name)
    except Exception as exc:
        message = exc.message if isinstance(exc, TandemError) else f"{type(exc).__name__}: {exc}"
        return {"name": name, "display_name": name, "ready": False, "unmet": [], "error": message}
    return {
        "name": name,
        "display_name": info.display_name,
        "ready": info.ready,
        "unmet": list(info.unmet),
        "error": None,
    }


def _executor_title(name: str) -> str:
    """How a sentence addressed to the operator names the executor that does a human step."""
    return "the teleop rig" if name == TELEOP else f"the {name!r} executor"


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


def write_session_log(session: Session) -> Path:
    """Persist a finished session for post-mortems."""
    directory = paths.log_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"session-{session.id}.json"
    path.write_text(
        json.dumps({"summary": session.summary(), "logs": session.logs()}, indent=2, default=str) + "\n"
    )
    return path
