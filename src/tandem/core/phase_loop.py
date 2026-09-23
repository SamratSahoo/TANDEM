"""The trial algorithm: one attempt at a task, planned into phases and walked to the end.

This is the method's inner loop, and it is tandem's rather than the planner's. The plan is proposed
here, each phase goes to whoever can do it, and a person's step is checked before the next one
starts. The session around it (``tandem.core.session``) owns everything a person sees: the state
machine, the prompts, the hand-off and the label. It gives this loop only what the loop needs, and
it gives each thing explicitly:

* the planner backend and the ``Capabilities`` it declared, for the robot's phases;
* the resolved phase-planning settings (``tandem.planning.config.PlanningConfig``);
* an event sink (`EventSink`), for the session's events file and its log;
* the operator (`OperatorIO`), for the prompts a phase raises and the progress it shows;
* the human-leg seam (`HumanLeg`), which carries out a person's step. Today that is a teleop
  hand-off through ``tandem.teleop.child``;
* where legs are allocated on disk (``tandem.core.episodes.LegDirs``).

How it maps onto the paper (TANDEM, Sec. IV-D "Task Plan Generation and Execution" and Sec. IV-E
"Demonstration Generation"):

* **Task plan generation.** Phi = (phi_1 .. phi_N), with phi_k = (gamma_k, e_k). `_prepare_plan`
  passes the current perception image and the instruction to ``tandem.planning.plan.build_plan``.
  That function proposes, validates and repairs the plan. With phase planning off there is no Phi,
  and the backend's own goal translation stands in for a single robot phase.
* **Autonomous execution.** A robot phase's subgoal goes to the TAMP system with the current scene,
  and the system plans and executes it. This is `_run_robot_phase`, through ``TampBackend.plan`` and
  ``TampBackend.execute``: "an interface for specifying subgoals and executing the resulting plans",
  and nothing more. `_on_plan_failure` decides what happens when no plan is found
  (``on_robot_phase_failure``).
* **Human execution.** The operator is shown a natural-language version of the subgoal, takes
  control through teleoperation, and ends the phase. This is `_run_human_phase`, through the
  operator's prompt and the human-leg seam.
* **Re-perception and verification.** A fresh image is taken after every phase. After a human
  phase, the VLM classifiers g_psi check the phase's intended effects, and a check that still fails
  ends the trial. In this module that is the perception pass at the top of every leg, `_rebind` for
  object names that drift between passes, and `_verify` with the retry loop in `_run_human_phase`.
* **Demonstration generation**, tau = ((tau_1, phi_1), .., (tau_N, phi_N)). Every leg this loop
  runs is stamped with the one trajectory id the session minted (``LegSpec.trajectory_id``).
  `TrialOutcome.legs_recorded` tells the session whether there is anything to label. After the label,
  ``tandem.core.episodes`` merges the legs into one episode.

Where this does not match the paper yet:

* A leg that was planned but failed to EXECUTE still advances the plan.
* Perception also runs before a human leg, not only before robot legs.
* Verification checks a human phase's atoms only (no delete effects, no preconditions).
* A trial whose check still fails after its retries is labeled like any other, not excluded.
* `replan` proposes again from scratch. The planner's failure is not fed back to the proposer.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from tandem.core.episodes import LegDirs
    from tandem.planners.base import Capabilities, SceneView, TampBackend
    from tandem.planning.config import PlanningConfig
    from tandem.planning.plan import PhasePlan
    from tandem.planning.structs import Phase

# How many times one task attempt may be decomposed again after the planner failed to plan a phase
# (`on_robot_phase_failure: replan`). Bounded, because a goal the planner genuinely cannot reach
# fails identically every time, and an unbounded retry would perceive and re-propose forever with
# an operator watching an arm that never moves.
MAX_REPLANS = 2


@dataclass
class HumanPhase:
    """A step of the task the plan says only a person can do.

    `instructions` is what the operator is shown; `expected` is what the VLM will be asked
    about afterwards — the same list, so nobody is checked against a hidden standard.
    """

    description: str
    instructions: str
    expected: list[str] = field(default_factory=list)
    index: int = 0
    total: int = 0
    attempt: int = 1
    verified: bool | None = None
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "instructions": self.instructions,
            "expected": self.expected,
            "index": self.index,
            "total": self.total,
            "attempt": self.attempt,
            "verified": self.verified,
            "missing": self.missing,
        }


@dataclass
class TrialOutcome:
    """How one attempt at a task ended, as far as the loop can tell.

    The loop keeps this up to date while it runs; it does not only build it at the end. An attempt
    that is preempted, or that cannot get the arm back, leaves `PhaseLoop.run` as an exception. The
    session still has to label and merge whatever reached disk, so it reads `PhaseLoop.outcome` on
    every exit path.

    For a trial that ran to the end, the operator's label is still the verdict, so `outcome` is None
    there. The loop sets `outcome` only when it ended the attempt early itself, and sets
    `failure_stage` to the stage that ended it. `outcome` is one of "success", "failure", "excluded"
    or "aborted". `failure_stage` is one of "invention", "tamp_planning", "tamp_execution",
    "verification" or "human_policy". Nothing reads either field yet.
    """

    trajectory_id: str
    outcome: str | None = None
    failure_stage: str | None = None
    # The last plan proposed this attempt. Kept after the attempt drops it, so the audit record can
    # still be written for a task that was abandoned part-way -- which is exactly the episode whose
    # provenance you want.
    plan: PhasePlan | None = None
    legs_recorded: int = 0


class EventSink(Protocol):
    """Where the loop reports what it is doing: the session's events file and its log."""

    def event(self, name: str, **payload: Any) -> None:
        """One line in the events file, and a message to every subscriber."""

    def log(self, text: str) -> None:
        """One line of the session's log, as tandem's own voice."""


class OperatorIO(Protocol):
    """The person running the trial, as the loop sees them. The session implements it.

    Each method either puts a question to a person or shows them something. The loop never touches
    the session's state machine directly. `rolling` and `await_human_phase` are the only two
    transitions it causes, and which states those are is for the session to decide.
    """

    def check_preempt(self) -> None:
        """Raise if the operator has abandoned the attempt. Called at every step boundary."""

    def take_handoff_request(self) -> bool:
        """Whether the operator asked for the arm since the last boundary. Consumes the request."""

    def rolling(self) -> None:
        """The loop is busy: perceiving, decomposing, planning or executing."""

    def show_progress(self, progress: tuple[int, int] | None) -> None:
        """How far through the plan the attempt is, as (phase index, number of phases)."""

    def show_unrepresented(self, clauses: list[dict]) -> None:
        """The parts of the instruction that the plan knowingly leaves out."""

    def show_human_phase(self, phase: HumanPhase | None) -> None:
        """The step a person is being asked to do, or None once it is over."""

    def await_human_phase(self) -> str:
        """Wait at the step last shown. Returns "done", "abort" or "teleop"."""

    def rollout_started(self, save_dir: Path) -> None:
        """A robot leg is about to be planned and recorded into ``save_dir``."""

    def rollout_saved(self, n_frames: int) -> None:
        """The robot leg in progress recorded ``n_frames`` frames."""


class HumanLeg(Protocol):
    """The human-leg seam: how a person's part of the task is carried out.

    It is called with the phase the person is asked to do. For a hand-off the operator asked for at
    a phase boundary it is called with None, which lends them the arm with no phase attached. It
    gives the arm to a person, waits until they give it back, and then takes it back. Today the
    session implements it as a teleop hand-off (``tandem.teleop.child``).

    It calls ``recorded(n_frames)`` as soon as a leg is safely on disk, and before it takes the arm
    back. Taking the arm back can fail and end the session, and a leg that reached disk must still
    be labeled and merged on the way out. It is not called at all when nothing was recorded.
    """

    def __call__(self, phase: Phase | None, recorded: Callable[[int], None]) -> None: ...


class PhaseLoop:
    """One attempt at a task: plan it into phases, then walk them.

    `run` resets every piece of per-attempt state, so one loop can run attempt after attempt against
    the same warm backend. `outcome` always describes the most recent `run`, and is meant to be read
    after it has returned or raised.
    """

    def __init__(
        self,
        backend: TampBackend,
        caps: Capabilities,
        cfg: PlanningConfig,
        *,
        events: EventSink,
        operator: OperatorIO,
        human_leg: HumanLeg,
        legs: LegDirs,
        record: bool = True,
    ) -> None:
        self.backend = backend
        self.caps = caps
        self.cfg = cfg
        self.events = events
        self.operator = operator
        self.human_leg = human_leg
        self.legs = legs
        self.record = record

        self.outcome = TrialOutcome(trajectory_id="")
        # One attempt's working state, reset by `run`.
        self._task = ""
        self._instruction = ""
        self._trajectory_id = ""
        self._vlm_dir: Path | None = None
        self._plan: PhasePlan | None = None
        self._detected_goal: tuple = ()
        self._task_done = False
        self._replans_left = MAX_REPLANS
        self._last_verdicts: list = []
        self._human_phase: HumanPhase | None = None

    # ---- the attempt -------------------------------------------------------

    def run(
        self,
        *,
        task: str,
        instruction: str,
        trajectory_id: str,
        vlm_dir: Path | None = None,
    ) -> TrialOutcome:
        """One attempt at the current task: plan it into phases, then walk them.

        `task` steers planning (the goal). `instruction` is the language label every leg is stamped
        with. `trajectory_id` is the lineage id that joins this attempt's legs into one episode.
        `vlm_dir` is where every model call made during the attempt is recorded, or None to record
        nothing.

        A preempt, or an arm that cannot be taken back, is raised straight through. `outcome` is
        still accurate when that happens.
        """
        self.outcome = TrialOutcome(trajectory_id=trajectory_id)
        self._task = task
        self._instruction = instruction
        self._trajectory_id = trajectory_id
        self._vlm_dir = vlm_dir
        self._plan = None
        self._detected_goal = ()
        self._task_done = False
        self._replans_left = MAX_REPLANS
        self._last_verdicts = []
        self._show_human_phase(None)
        self.operator.show_progress(None)
        self.operator.show_unrepresented([])
        leg = 0

        while not self._task_done:
            self.operator.check_preempt()
            # One directory per leg, allocated ONCE and handed to every call that writes into
            # it. Perception debug output, the plan and the recording all belong to one leg.
            leg_dir = self.legs.new()
            try:
                scene = self._perceive(leg_dir, first_leg=leg == 0)
                if self._plan is None:
                    # Also re-run with phase planning OFF, where it refreshes the planner's own
                    # goal from THIS pass's object labels -- the previous pass's labels may not
                    # even exist any more.
                    if not self._prepare_plan(scene):
                        break
                elif not self._rebind(scene):
                    break

                if self._plan is not None and self._plan.finished:
                    break
                phase = self._plan.current if self._plan is not None else None

                if self.operator.take_handoff_request():
                    # An operator-asked hand-off, honoured at a phase boundary so the arm parks
                    # somewhere sane rather than mid-motion. Nothing advances: the same phase
                    # (or, with no plan, the same task) is re-perceived and planned afterwards.
                    self._human_leg(None)
                elif phase is not None and phase.is_human:
                    self._run_human_phase(phase)
                else:
                    self._run_robot_phase(scene, leg_dir)
            finally:
                self.legs.retire(leg_dir)

            leg += 1
            if self._plan is not None and not self._plan.finished:
                self.events.event(
                    "phase_complete",
                    phase_index=self._plan.index,
                    n_phases=len(self._plan.phases),
                )
        return self.outcome

    def _end(self, outcome: str, failure_stage: str | None = None) -> None:
        """Record why the loop itself ended the attempt early."""
        self.outcome.outcome = outcome
        self.outcome.failure_stage = failure_stage

    def _leg_recorded(self, n_frames: int) -> None:
        """Count a leg that reached disk. The session labels the attempt only if one did."""
        if n_frames:
            self.outcome.legs_recorded += 1

    # ---- one leg -----------------------------------------------------------

    def _perceive(self, leg_dir: Path, *, first_leg: bool) -> SceneView:
        """Look at the workspace. The arm is only parked first when nothing is mid-task.

        Resetting between phases would undo the step before it — and after a hand-off it could
        drive an arm a person just handed us, holding something, back to home.
        """
        self.operator.rolling()
        scene = self.backend.perceive(
            task_hint=self._task,
            save_dir=leg_dir,
            reset_arm=first_leg,
        )
        self.events.log(f"perceived: {', '.join(scene.object_labels) or 'nothing'}")
        return scene

    def _prepare_plan(self, scene) -> bool:
        """Decompose the task into phases, or fall through to the planner's own goal.

        With phase planning off there is nothing for a model to decompose, so the goal is the one
        the planner's own translator produced from the instruction during perception — exactly the
        behaviour a session had before any of this existed.
        """
        if not self.cfg.enabled:
            self._detected_goal = scene.detected_goal
            return True

        from tandem.planning.grounding import to_pil
        from tandem.planning.plan import build_plan
        from tandem.planning.record import recording_to

        if not scene.rgb_path or not Path(scene.rgb_path).is_file():
            self.events.log("perception saved no image, so the task cannot be decomposed")
            return False

        try:
            from PIL import Image

            image = to_pil(Image.open(scene.rgb_path).convert("RGB"))
        except Exception as exc:
            self.events.log(f"could not read the perception image: {exc}")
            return False

        import asyncio

        self.operator.rolling()
        self.events.log(f"decomposing the task with {self.cfg.proposal_model}")
        try:
            with recording_to(self._vlm_dir):
                plan, failure = asyncio.run(
                    build_plan(
                        image,
                        self._task,
                        scene.object_labels,
                        scene.table_label,
                        self.cfg,
                        self.caps,
                        self._trajectory_id,
                    )
                )
        except Exception as exc:
            self.events.log(f"could not decompose the task: {type(exc).__name__}: {exc}")
            self._end("failure", "invention")
            return False
        if plan is None:
            self.events.log(failure or "the task could not be decomposed")
            self._end("failure", "invention")
            return False

        self._plan = plan
        self.outcome.plan = plan
        self.operator.show_progress((0, len(plan.phases)))
        unrepresented = [dict(u) for u in plan.spec.unrepresented]
        self.operator.show_unrepresented(unrepresented)
        if unrepresented:
            # Loud, not merely logged: every later phase is planned against this, the dataset is
            # labeled with the whole instruction regardless, and the remedy — put the missing object
            # on the table and start again — is only available before the arm moves.
            self.events.event("instruction_not_fully_represented", unrepresented=unrepresented)
            for dropped in unrepresented:
                self.events.log(f"NOT part of the plan — {dropped['clause']}: {dropped['reason']}")
        for i, phase in enumerate(plan.phases):
            self.events.log(f"phase {i} [{phase.executor}] {phase.description}")
        return True

    def _rebind(self, scene) -> bool:
        """Point the plan at this pass's object labels, or give up on the attempt.

        Perception names objects afresh every pass and the names drift. Mid-task that is fatal if
        unhandled: the plan refers to objects this pass did not produce, so it would be thrown away
        and the task re-planned from a scene already half rearranged — asking the person to redo the
        step they just finished.
        """
        if self._plan is None:
            return True
        from tandem.planning.drift import match_drifted_names

        detected = set(scene.object_labels) | {scene.table_label}
        needed = self._plan.objects_needed_now()
        missing = sorted(needed - detected)
        if not missing:
            return True

        # Only a label the plan does not already own can be a drifted spelling of one it does;
        # offering an object a completed phase named would point this leg at something already put
        # away.
        candidates = sorted(detected - self._plan.spec.scene_types.all_names)
        mapping = match_drifted_names(missing, candidates)
        if mapping is None:
            self.events.log(f"perception no longer detects {', '.join(missing)}; abandoning the attempt")
            self._plan = None
            self._task_done = True
            return False
        self._plan.rebind(mapping)
        return True

    def _run_robot_phase(self, scene, save_dir: Path) -> None:
        """Hand one sub-goal to the planner and record what it did."""
        from tandem.planners.base import LegSpec

        if self._plan is not None:
            goal = self._plan.goal()
            surfaces = self._plan.surfaces()
            run = self._plan.robot_run()
            description = "; ".join(p.description for p in run)
            index, total = self._plan.index, len(self._plan.phases)
        else:
            goal, surfaces = list(self._detected_goal), frozenset()
            description, index, total = self._task, None, None

        if not goal:
            self.events.log("nothing to plan for: the goal is empty")
            self._plan = None
            return

        self.events.event("rollout_start", dir=str(save_dir), phase_index=index, n_phases=total)
        self.operator.rollout_started(save_dir)
        self.events.log(f"planning: {[a.to_dict() for a in goal]}")
        result = self.backend.plan(scene.scene_id, goal, surfaces=surfaces, save_dir=save_dir)

        if not result.ok:
            self._on_plan_failure(result.failure_reason or "no plan found")
            return

        self.operator.check_preempt()
        execution = self.backend.execute(
            result.plan_handle,
            LegSpec(
                trajectory_id=self._trajectory_id,
                instruction=self._instruction,
                phase_index=index,
                n_phases=total,
                phase_description=description,
                record=self.record,
            ),
            save_dir=save_dir,
        )
        self.operator.rollout_saved(execution.n_frames)
        self._leg_recorded(execution.n_frames)
        self.events.event("rollout_saved", dir=str(save_dir), n_frames=execution.n_frames)
        if not execution.ok:
            self.events.log(f"execution failed: {execution.failure_reason}")

        if self._plan is not None:
            self._plan.record_plan(self._plan.index, result)
            self._plan.advance()
            self.operator.show_progress((self._plan.index, len(self._plan.phases)))
        else:
            # No phase plan: that was the whole task, in one leg.
            self._detected_goal = ()
            self._task_done = True

    def _on_plan_failure(self, reason: str) -> None:
        """What happens when the planner cannot plan a phase.

        `teleop` is the option the old design could not express at all: who does what was decided
        at proposal time inside the planner's process, so a phase it turned out not to be able to
        plan could only end the attempt. tandem owns the split, so the person can simply do it.
        """
        policy = self.cfg.on_robot_phase_failure
        self.events.log(f"the planner could not plan this phase: {reason}")
        self.events.event("phase_plan_failed", reason=reason, policy=policy)

        if self._plan is None or policy == "abort":
            self._plan = None
            self._task_done = True
            self._end("failure", "tamp_planning")
            return
        if policy == "replan":
            # Drop the plan and go round again: the next pass perceives afresh and decomposes the
            # task against the scene as it now stands. Bounded, because a goal the planner cannot
            # reach fails the same way every time -- and `_task_done` stays False, which is the
            # whole difference from `abort`. Without that this policy was abort under another name.
            if self._replans_left <= 0:
                self.events.log("out of re-planning attempts; giving up on this task")
                self._plan = None
                self._task_done = True
                self._end("failure", "tamp_planning")
                return
            self._replans_left -= 1
            self.events.log("re-planning the task from the scene as it now stands")
            self._plan = None
            return
        self.events.log("offering this phase to you as teleop instead")
        phase = self._plan.hand_current_to_human()
        self._run_human_phase(phase)

    # ---- a phase for a person ----------------------------------------------

    def _run_human_phase(self, phase) -> None:
        """Hand the arm over, let a person do this step, and check they did.

        A failed check is not a lost demonstration: the person is told what is still missing and
        given another go, because one bad classifier call should not cost an episode.
        """
        from tandem.planning.plan import phase_summary, retry_message

        cfg = self.cfg
        attempts_left = cfg.verify_retries
        attempt = 1
        while True:
            summary = phase_summary(self._plan, phase) if self._plan is not None else {}
            previous = self._human_phase
            self._show_human_phase(
                HumanPhase(
                    description=summary.get("description", phase.description),
                    instructions=summary.get("instructions", phase.instructions),
                    expected=list(summary.get("expected", [])),
                    index=int(summary.get("phase_index", 0)),
                    total=int(summary.get("n_phases", 0)),
                    attempt=attempt,
                    missing=previous.missing if previous is not None and attempt > 1 else [],
                )
            )
            self.events.event("awaiting_human_phase", **summary)

            answer = self.operator.await_human_phase()
            if answer == "abort":
                self.events.log("the phase was abandoned; ending this attempt")
                self._show_human_phase(None)
                self._plan = None
                self._task_done = True
                self._end("aborted")
                return
            if answer == "teleop":
                self._human_leg(phase)

            ok, verdicts = self._verify(phase)
            self._human_phase.verified = ok
            self._human_phase.missing = _missing_from(verdicts)
            self.events.event("human_phase_verified", ok=ok, verdicts=verdicts)

            if ok or not cfg.verify_enforced:
                if not ok:
                    self.events.log("the step did not verify, but verify_enforced is off; carrying on")
                break
            if attempts_left <= 0:
                self.events.log("the step could not be verified; ending this attempt")
                self._show_human_phase(None)
                self._plan = None
                self._task_done = True
                self._end("failure", "verification")
                return
            attempts_left -= 1
            attempt += 1
            self.events.log(retry_message(self._human_phase.missing, attempts_left))

        self._show_human_phase(None)
        if self._plan is not None:
            self._plan.verdicts.extend(self._last_verdicts)
            self._plan.advance()
            self.operator.show_progress((self._plan.index, len(self._plan.phases)))

    def _human_leg(self, phase) -> None:
        """Give the arm to a person for one leg, through the human-leg seam."""
        self.human_leg(phase, self._leg_recorded)

    def _show_human_phase(self, view: HumanPhase | None) -> None:
        # The loop keeps its own reference as well, because a retry carries the previous attempt's
        # `missing` forward and the verdict is written back onto the view the operator is looking at.
        self._human_phase = view
        self.operator.show_human_phase(view)

    def _verify(self, phase) -> tuple[bool, list[dict]]:
        """Ask a vision model whether the workspace now looks like that step was done.

        A classifier that cannot be reached must never cost a demonstration, so an error here is
        an accepted step with no verdicts rather than a failed one.
        """
        import asyncio

        from tandem.planning.grounding import to_pil, verify_phase
        from tandem.planning.record import recording_to

        self._last_verdicts = []
        cfg = self.cfg
        try:
            from PIL import Image

            frame = self.backend.capture_frame(camera=cfg.verification_camera)
            image = to_pil(Image.open(frame).convert("RGB"))
            invented = self._plan.spec.invented if self._plan is not None else ()
            with recording_to(self._vlm_dir):
                ok, verdicts = asyncio.run(verify_phase(image, phase, invented, cfg, self.caps))
        except Exception as exc:
            self.events.log(f"could not check the step; accepting it unchecked ({exc})")
            return True, []
        self._last_verdicts = list(verdicts)
        return ok, [v.summary() for v in verdicts]


def _missing_from(verdicts: Iterable[dict]) -> list[str]:
    """What a failed verification says is still expected.

    Mirrors `grounding.missing_statements`, over the summary dicts the events carry: the verdicts
    that were not SATISFIED, already phrased for a person. Not the ones that do not hold -- a delete
    effect passes by not holding, so reading `holds` would list a delete effect that is done and drop
    one that is still true. The latter is said the other way round, or the operator is told to make
    true exactly what they were meant to undo. The `reason` is appended when the model gave one,
    because "the cloth is not folded" is much less useful than knowing it saw a corner sticking out.
    """
    missing = []
    for verdict in verdicts:
        if not isinstance(verdict, dict):
            continue
        # A summary written before verdicts carried an expectation has only `holds`, which meant
        # the same thing then: every atom was expected to hold.
        satisfied = verdict.get("satisfied", verdict.get("holds", True))
        if satisfied:
            continue
        statement = str(verdict.get("statement") or verdict.get("atom") or "").strip()
        if not statement:
            continue
        if verdict.get("expected", True) is False:
            statement = f"{statement} -- and it should no longer be"
        reason = str(verdict.get("reason") or "").strip()
        missing.append(f"{statement} — {reason}" if reason else statement)
    return missing
