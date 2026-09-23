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
  and nothing more. Two things the leg is for go with the subgoal, where the planner declares it
  honours them: which objects it may pick (``movables``, so the person's tool is an obstacle and
  never a thing to pick up) and whether to end at home (``return_home``, only on the task's last
  leg). A leg that was planned but did not EXECUTE ends the trial (failure stage
  ``tamp_execution``): the arm is somewhere no plan put it, and every later phase was planned
  against a scene that no longer exists. `_on_plan_failure` decides what happens when no plan is
  found (``on_robot_phase_failure``): the trial ends (``abort``), the phase is handed to a person
  (``teleop``), or the task is proposed again with the planner's failure fed back to the model
  (``replan``).
* **Human execution.** The operator is shown a natural-language version of the subgoal, takes
  control through teleoperation, and ends the phase. This is `_run_human_phase`, through the
  operator's prompt and the human-leg seam.
* **Re-perception and verification.** The scene is perceived afresh before every robot leg, and
  never before a person's: a human phase is judged on a fresh frame from the verification camera,
  and nothing about it is planned. After a human phase, the VLM classifiers g_psi check the phase's
  operator: its add effects must now hold and its delete effects must not
  (``grounding.verify_effects``). Its preconditions can be checked before the hand-off as well
  (``check_human_preconditions``, ``grounding.verify_preconditions``). A check that still fails
  once its retries are spent terminates the trial, and the trial is EXCLUDED from the dataset
  (``on_verification_failure``). In this module that is the perception pass that opens each robot
  leg, `_rebind` for object names that drift between passes, and the camera checks around
  `_run_human_phase`. The same classifiers can watch a robot leg too (``check_tamp_preconditions``,
  ``check_tamp_effects``), which the paper does not do; that is observational unless configured
  otherwise.
* **Demonstration generation**, tau = ((tau_1, phi_1), .., (tau_N, phi_N)). Every leg this loop
  runs is stamped with the one trajectory id the session minted (``LegSpec.trajectory_id``).
  `TrialOutcome.legs_recorded` tells the session whether there is anything to label or file, and
  `TrialOutcome.outcome` whether to ask for a label at all. ``tandem.core.episodes`` then merges the
  legs into one episode, excluded trials included, so their raw legs survive for inspection.

How a trial ended is the paper's Fig. 4 taxonomy: ``TrialOutcome.outcome`` and ``failure_stage``,
kept on the plan as well (``PhasePlan.set_outcome``) so that ``hitl.json`` says it, and announced
as a ``trial_outcome`` event whenever the loop ends a trial itself.

Where this departs from the paper, on purpose:

* Consecutive robot phases are handed to the planner as ONE goal where that is sound
  (``conjoin_robot_phases``, ``PhasePlan.robot_run``), so there is one perception pass for the run
  rather than one per phase. The paper re-perceives after every phase; turning the setting off does
  exactly that.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from tandem.core.episodes import LegDirs
    from tandem.planners.base import Capabilities, SceneView, TampBackend
    from tandem.planning.config import PlanningConfig
    from tandem.planning.grounding import Verdict
    from tandem.planning.plan import PhasePlan
    from tandem.planning.structs import Phase
    from tandem.planning.symbols import Atom


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
    # True or False once the camera has judged the step. None until then, and None for good when the
    # step was accepted without being judged (the check is off for it, or could not run).
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
    "verification" or "human_policy". The session reads `outcome` to decide whether to ask for a
    label at all: an "excluded" trial is filed without one.
    """

    trajectory_id: str
    outcome: str | None = None
    failure_stage: str | None = None
    # Why the loop ended the attempt, in words, for the log, the events and hitl.json.
    reason: str | None = None
    # The last plan proposed this attempt. Kept after the attempt drops it, so the audit record can
    # still be written for a task that was abandoned part-way -- which is exactly the episode whose
    # provenance you want.
    plan: PhasePlan | None = None
    legs_recorded: int = 0


@dataclass
class _Check:
    """What one camera check found, or why it could not be run.

    `error` set means there is no verdict at all: the frame could not be taken, or the classifier did
    not answer. That is a check that did not happen, not one that failed, and it is recorded as such
    (``PhasePlan.record_unchecked``) rather than folded into `ok`.
    """

    ok: bool
    verdicts: list[Verdict] = field(default_factory=list)
    error: str | None = None

    def summaries(self) -> list[dict]:
        return [verdict.summary() for verdict in self.verdicts]


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
        """The loop is busy: perceiving, decomposing, planning, executing, or checking a step."""

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
        # `on_robot_phase_failure: replan`: how many times this attempt has been proposed again, and
        # why each earlier plan could not be carried out -- all of them, not only the last, so a model
        # told about the second failure is not free to walk straight back into the first.
        self._replans = 0
        self._replan_feedback: list[str] = []
        # Whether a person had the arm last, through a human phase. The next perception pass opens
        # the gripper first (`_perceive`).
        self._after_human = False
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
        self._replans = 0
        self._replan_feedback = []
        self._after_human = False
        self._show_human_phase(None)
        self.operator.show_progress(None)
        self.operator.show_unrepresented([])
        leg = 0

        while not self._task_done:
            self.operator.check_preempt()
            # Before anything else, so a finished plan costs nothing more: the task is over, and a
            # perception pass here would only park the arm in the last frame of the demonstration.
            if self._plan is not None and self._plan.finished:
                break

            if self._plan is not None and self._plan.next_is_human():
                # A person's step has no perception pass. Nothing about it is planned, it is judged
                # on a fresh frame from the verification camera, and the scene is looked at again
                # before the next robot leg -- which is the pass that has to see what the person
                # left, not this one. A pass here also re-read every object label mid-task for
                # nothing, and each re-read is one more chance for the names to drift.
                self._take_turn(None, None)
            else:
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
                    # A plan just proposed may open with a person's step. It is taken here, from
                    # the pass the proposal needed anyway.
                    self._take_turn(scene, leg_dir)
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

    def _end(self, outcome: str, failure_stage: str | None = None, reason: str | None = None) -> None:
        """Record why the loop itself ended the attempt early, everywhere it is read from.

        Three places, because three different readers need it. `outcome` is what the session decides
        on (label, or file as excluded). The plan is what ``hitl.json`` is written from, and a record
        that did not say how the trial ended would file an excluded trial among the ordinary failures.
        The event is what is left when neither of those is: a trial that failed at invention has no
        plan and no legs, so the events file is the only trace of it.
        """
        self.outcome.outcome = outcome
        self.outcome.failure_stage = failure_stage
        self.outcome.reason = reason
        # The LAST plan, not the one being walked: every path that ends a trial drops `_plan`.
        plan = self.outcome.plan
        if plan is not None:
            plan.set_outcome(outcome, failure_stage)
        self.events.event(
            "trial_outcome",
            trajectory_id=self._trajectory_id,
            outcome=outcome,
            failure_stage=failure_stage,
            excluded=outcome == "excluded",
            reason=reason,
            phase_index=plan.index if plan is not None else None,
        )

    def _leg_recorded(self, n_frames: int) -> None:
        """Count a leg that reached disk. The session labels the attempt only if one did."""
        if n_frames:
            self.outcome.legs_recorded += 1

    # ---- one leg -----------------------------------------------------------

    def _take_turn(self, scene: SceneView | None, leg_dir: Path | None) -> None:
        """Carry out whatever is due now: a hand-off the operator asked for, a person's step, or a leg.

        ``scene`` and ``leg_dir`` are None exactly when the step due is a person's, which is never
        preceded by a perception pass.
        """
        phase = self._plan.current if self._plan is not None else None
        if self.operator.take_handoff_request():
            # An operator-asked hand-off, honoured at a phase boundary so the arm parks somewhere
            # sane rather than mid-motion. Nothing advances: the same phase (or, with no plan, the
            # same task) is re-perceived and planned afterwards. That pass leaves the gripper as the
            # operator left it (`_perceive`): whatever it holds when it comes back is their choice.
            self._human_leg(None)
        elif phase is not None and phase.is_human:
            self._run_human_phase(phase)
        else:
            self._run_robot_phase(scene, leg_dir)

    def _perceive(self, leg_dir: Path, *, first_leg: bool) -> SceneView:
        """Look at the workspace. The arm is only parked first when nothing is mid-task.

        Resetting between phases would undo the step before it — and after a hand-off it could
        drive an arm a person just handed us, holding something, back to home.

        The first pass after a human phase opens the gripper first, and moves nothing else. Nothing
        about a person driving the arm guarantees the fingers were left open, and a planner that
        plans every goal from an empty hand (cuTAMP's HandEmpty) would otherwise plan its first
        grasp through fingers that are closed. Only then: an open on any other pass could drop
        something a leg is still meant to be holding, and after a hand-off the operator asked for,
        what the arm holds is theirs to decide. It is passed only when wanted, so a backend that
        never follows a person is never handed it.
        """
        self.operator.rolling()
        options: dict[str, Any] = {}
        if self._after_human:
            options["open_gripper"] = True
            self.events.log("a person had the arm last, so the gripper is opened before looking")
        self._after_human = False
        scene = self.backend.perceive(
            task_hint=self._task,
            save_dir=leg_dir,
            reset_arm=first_leg,
            **options,
        )
        self.events.log(f"perceived: {', '.join(scene.object_labels) or 'nothing'}")
        return scene

    def _prepare_plan(self, scene) -> bool:
        """Decompose the task into phases, or fall through to the planner's own goal.

        With phase planning off there is nothing for a model to decompose, so the goal is the one
        the planner's own translator produced from the instruction during perception — exactly the
        behaviour a session had before any of this existed.

        After a ``replan``, the model is told why every earlier plan this attempt could not be
        carried out (`_replan_feedback`). Asked the same question of the same scene without it, it
        gives the same answer, and the re-plan budget is spent reproducing the failure.
        """
        if not self.cfg.enabled:
            self._detected_goal = scene.detected_goal
            return True

        from tandem.planning.plan import build_plan
        from tandem.planning.record import recording_to

        if not scene.rgb_path or not Path(scene.rgb_path).is_file():
            reason = "perception saved no image, so the task cannot be decomposed"
            self.events.log(reason)
            self._end("failure", "invention", reason)
            return False

        try:
            image = _open_image(scene.rgb_path)
        except Exception as exc:
            reason = f"could not read the perception image, so the task cannot be decomposed: {exc}"
            self.events.log(reason)
            self._end("failure", "invention", reason)
            return False

        import asyncio

        feedback = "\n".join(self._replan_feedback) or None
        self.operator.rolling()
        if feedback:
            self.events.log(
                f"decomposing the task again with {self.cfg.proposal_model}, told why the last plan "
                "could not be carried out"
            )
        else:
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
                        feedback=feedback,
                    )
                )
        except Exception as exc:
            # A proposal the repair loop never got to validate (ProposalError after max_attempts)
            # lands here, and so does a model that could not be reached at all. Both are the paper's
            # "invention" failure: there is no task plan to run.
            reason = f"could not decompose the task: {type(exc).__name__}: {exc}"
            self.events.log(reason)
            self._end("failure", "invention", reason)
            return False
        if plan is None:
            reason = failure or "the task could not be decomposed"
            self.events.log(reason)
            self._end("failure", "invention", reason)
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

        Only ever before a robot leg, since only a robot leg is preceded by a perception pass: what
        a leg needs is the names the planner will be handed, and a person needs none of them.
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
            # A leg the planner cannot be given: its goal names objects this scene does not have.
            reason = f"perception no longer detects {', '.join(missing)}, which the next robot leg needs"
            self.events.log(f"{reason}; abandoning the attempt")
            self._plan = None
            self._task_done = True
            self._end("failure", "tamp_planning", reason)
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
            goal, surfaces, run = list(self._detected_goal), frozenset(), ()
            description, index, total = self._task, None, None

        if not goal:
            self._nothing_to_plan(run, description)
            return

        # Before the leg starts, not after rollout_start: a leg its preconditions stop is not a leg,
        # and the operator must not be shown one beginning.
        checks_leg = self._plan is not None and self.cfg.check_tamp_preconditions
        if checks_leg and not self._leg_preconditions_hold(scene, run):
            return

        options = self._leg_options()
        self.events.event(
            "rollout_start",
            dir=str(save_dir),
            phase_index=index,
            n_phases=total,
            **{k: sorted(v) if isinstance(v, frozenset) else v for k, v in options.items()},
        )
        self.operator.rollout_started(save_dir)
        self.events.log(f"planning: {[a.to_dict() for a in goal]}")
        if "movables" in options:
            self.events.log(
                f"the planner may pick up only: {', '.join(sorted(options['movables'])) or 'nothing'}"
            )
        if options.get("return_home") is False:
            self.events.log(
                "more of the task follows this leg, so it ends where it stops rather than at home"
            )
        result = self.backend.plan(scene.scene_id, goal, surfaces=surfaces, save_dir=save_dir, **options)

        if not result.ok:
            self._on_plan_failure(result.failure_reason or "no plan found", run)
            return

        self.operator.check_preempt()
        if self._plan is not None:
            # On the record before the arm moves, not once the leg has run: a leg that then fails to
            # execute is audited by exactly this -- which plan it was carrying out.
            self._plan.record_plan(self._plan.index, result)
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
            self._execution_failed(execution.failure_reason, description)
            return
        if self._plan is not None and self.cfg.check_tamp_effects:
            # Before advance(), so the plan's index is still the leg that just ran. Only after a leg
            # that executed: after one that did not, the camera would be asked about motion nobody
            # finished, and its "no" is already known.
            self._check_leg_effects(run)

        if self._plan is not None:
            self._plan.advance()
            self.operator.show_progress((self._plan.index, len(self._plan.phases)))
        else:
            # No phase plan: that was the whole task, in one leg.
            self._detected_goal = ()
            self._task_done = True

    def _leg_options(self) -> dict[str, Any]:
        """What this leg is for, beyond its goal, as far as the planner declares it can be told.

        ``movables``: only the objects some robot phase moves may be picked (``robot_movables``).
        Every other detection -- the person's tool, say -- stays in the scene as an obstacle.

        ``return_home``: only the task's last leg ends at home (``is_last_leg``). Any other is
        continued from where it stops, by a person or by the next leg, and a trip home recorded in
        the middle of a demonstration is motion nobody asked for.

        Each is passed only where ``Capabilities`` declares support for it. A planner that does not
        may leave the keyword out of its signature altogether, and one handed it anyway would either
        fail or quietly plan without it. With phase planning off there is no plan to read either
        from, and the leg is an ordinary rollout: neither is passed.
        """
        if self._plan is None:
            return {}
        options: dict[str, Any] = {}
        if self.caps.supports_movable_restriction:
            options["movables"] = self._plan.robot_movables()
        if self.caps.supports_return_home:
            options["return_home"] = self._plan.is_last_leg()
        return options

    def _nothing_to_plan(self, run: Sequence[Phase], description: str) -> None:
        """End the trial over a leg whose goal says nothing the planner can be given.

        It used to drop the plan and go round again. That is a loop, not a retry: the next pass
        perceives, proposes (or, with phase planning off, reads the planner's own goal) and arrives
        at the same empty goal, until someone preempts it -- with the operator watching an arm that
        never moves, and nothing on the record to say why.
        """
        if self._plan is not None:
            # Every atom of the leg is one the planner supplies for itself (HandEmpty, for TipTop),
            # so no goal survives rendering into its language. The plan asked for the leg; the
            # planner was never going to be able to take it.
            atoms = ", ".join(sorted(str(a) for p in run for a in p.atoms)) or "no atoms"
            stage = "invention"
            reason = (
                f"the robot phase {description!r} asks the {self.caps.name} planner for nothing it can "
                f"plan: none of {atoms} is in its goal language"
            )
        else:
            stage = "tamp_planning"
            reason = f"the {self.caps.name} planner found no goal in the instruction {self._task!r}"
        self.events.log(f"nothing to plan for: {reason}; ending this attempt")
        self._plan = None
        self._task_done = True
        self._end("failure", stage, reason)

    def _execution_failed(self, failure: str | None, description: str) -> None:
        """End the trial over a leg that was planned and did not execute. The plan never advances.

        There is no policy for this, unlike a leg that cannot be planned. The arm is somewhere no
        plan put it, possibly holding something, and every later phase was planned against a scene
        that no longer exists. Advancing would ask the next phase of a world the robot did not
        produce, and record a demonstration of it. The legs already on disk still reach the
        operator's label, with the stage beside it.
        """
        reason = f"the robot could not carry out {description!r}: {failure or 'the planner gave no reason'}"
        self.events.log(f"execution failed: {reason}; ending this attempt")
        self._plan = None
        self._task_done = True
        self._end("failure", "tamp_execution", reason)

    def _on_plan_failure(self, reason: str, run: Sequence[Phase]) -> None:
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
            self._end("failure", "tamp_planning", f"the planner could not plan this phase: {reason}")
            return
        if policy == "replan":
            self._replan(reason, run)
            return
        self.events.log("offering this phase to you as teleop instead")
        phase = self._plan.hand_current_to_human()
        self._run_human_phase(phase)

    def _replan(self, reason: str, run: Sequence[Phase]) -> None:
        """Drop the plan and propose the task again, telling the model why this one failed.

        The next pass perceives afresh, and the task is decomposed against the scene as it now
        stands, with the planner's failure in the prompt (`_prepare_plan`). `_task_done` stays False,
        which is the whole difference from `abort`.

        Bounded by ``max_attempts``, the proposer's own repair budget: a re-plan is one more repair
        of the plan, with the planner rather than the validator saying what was wrong with it. A goal
        the planner genuinely cannot reach fails the same way every time, and an unbounded retry
        would perceive and re-propose forever with an operator watching an arm that never moves.
        """
        index = self._plan.index
        if self._replans >= self.cfg.max_attempts:
            self.events.log("out of re-planning attempts; giving up on this task")
            self._plan = None
            self._task_done = True
            self._end(
                "failure",
                "tamp_planning",
                f"phase {index} could not be planned after {self._replans} re-plan(s): {reason}",
            )
            return
        self._replans += 1
        # The model proposing again never sees the plan it is replacing, so "phase 2" alone would
        # mean nothing to it: the phase is said in its own words and atoms as well.
        asked = "; ".join(p.description for p in run) or "(no description)"
        atoms = ", ".join(sorted(str(a) for p in run for a in p.atoms))
        goal = f", with the goal {atoms}" if atoms else ""
        self._replan_feedback.append(
            f"phase {index} could not be planned: {reason} (the robot was asked to {asked}{goal})"
        )
        self.events.log(
            f"re-planning the task from the scene as it now stands "
            f"({self._replans} of at most {self.cfg.max_attempts})"
        )
        self._plan = None

    # ---- a phase for a person ----------------------------------------------

    def _run_human_phase(self, phase) -> None:
        """Hand the arm over, let a person do this step, and check they did.

        A failed check is not a lost demonstration straight away: the person is told what is still
        missing and given another go, because one bad classifier call should not cost an episode.
        A check that is still failing once the retries are spent is different. Every later phase is
        planned against the belief that this one happened, so the trial stops there, and what becomes
        of it is ``on_verification_failure`` (`_verification_failed`).
        """
        from tandem.planning.grounding import verify_effects
        from tandem.planning.plan import phase_summary, retry_message

        cfg = self.cfg
        index = self._plan.index if self._plan is not None else None
        # Whatever happens next, the next robot leg starts from a gripper a person may have closed.
        self._after_human = True

        # Is the workspace in a state this phase can be carried out from? Once, before the first
        # attempt: a retry runs the same phase in a world its own failed attempt may have changed, and
        # re-checking the entry conditions there would report the attempt rather than the set-up.
        if cfg.check_human_preconditions and not self._human_preconditions_hold(phase, index):
            return

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
                self._end("aborted", None, f"the operator abandoned the human phase {phase.description!r}")
                return
            if answer == "teleop":
                self._human_leg(phase)
            # Busy again: checking the step, then on to whatever is next. Said here because nothing
            # else says it any more -- the perception pass that used to follow every person's step
            # did, and without it a step answered "done" and followed by another person's step would
            # never leave the prompt state, so a UI that repaints on a change of state would go on
            # showing the first step.
            self.operator.rolling()

            skipped = self._effects_not_checked()
            if skipped is not None:
                # Recorded as NOT checked rather than as passed, so the trail cannot be read as a
                # verification that happened.
                self.events.log(f"not verifying this step: {skipped}")
                self.events.event(
                    "human_phase_verified",
                    phase_index=index,
                    attempt=attempt,
                    ok=None,
                    skipped=skipped,
                    verdicts=[],
                )
                break

            check = self._camera_check(
                self._verification_frame,
                lambda image: verify_effects(image, phase, self._invented(), cfg, self.caps),
            )
            if check.error is not None:
                # A classifier or camera that cannot be reached must never cost a demonstration, so
                # the step goes ahead. It is on the record as unchecked, which is not the same as
                # passed: a phase with no verdicts would otherwise read as one that was fine.
                self._record_unchecked(index, "effect check", check.error)
                self.events.log(f"could not check the step; accepting it unchecked ({check.error})")
                self.events.event(
                    "human_phase_verified",
                    phase_index=index,
                    attempt=attempt,
                    ok=None,
                    unchecked=check.error,
                    verdicts=[],
                )
                break

            verdicts = check.summaries()
            self._human_phase.verified = check.ok
            self._human_phase.missing = _missing_from(verdicts)
            self.events.event(
                "human_phase_verified", phase_index=index, attempt=attempt, ok=check.ok, verdicts=verdicts
            )

            if check.ok or not cfg.verify_enforced:
                # The attempt that settled the phase is the one on the record, failing verdicts
                # included when verify_enforced let it through: they are the evidence it was waved on.
                self._record_verdicts(index, check.verdicts)
                if not check.ok:
                    self.events.log("the step did not verify, but verify_enforced is off; carrying on")
                break
            if attempts_left <= 0:
                # Recorded BEFORE the trial ends: the failing verdicts are exactly what an excluded
                # trial is audited by, and a record without them reads as one that was never checked.
                self._record_verdicts(index, check.verdicts)
                still = "; ".join(self._human_phase.missing) or "nothing the camera could name"
                self._verification_failed(
                    f"the human phase {phase.description!r} did not verify after {attempt} attempt(s); "
                    f"still expected: {still}"
                )
                return
            attempts_left -= 1
            attempt += 1
            self.events.log(retry_message(self._human_phase.missing, attempts_left, by=cfg.human_executor))

        self._show_human_phase(None)
        if self._plan is not None:
            self._plan.advance()
            self.operator.show_progress((self._plan.index, len(self._plan.phases)))

    def _effects_not_checked(self) -> str | None:
        """Why this human phase's effects are not put to the camera, or None when they are."""
        if not self.cfg.check_human_effects:
            return "check_human_effects is off"
        if not self.cfg.verify_final_phase and self._plan is not None and self._plan.is_final_phase():
            # Nothing later is planned against it; the operator's label is the verdict on it.
            return "it is the last phase and verify_final_phase is off"
        return None

    def _human_preconditions_hold(self, phase, index: int | None) -> bool:
        """Put a human phase's preconditions to the camera. False means the trial was ended."""
        from tandem.planning.grounding import checkable, verify_preconditions

        invented = self._invented()
        # Nothing a camera can settle -- HandEmpty() alone, say -- is not a check that passed, and it
        # is no reason to take a frame either.
        if not checkable(phase.preconditions, invented, self.caps):
            return True
        check = self._camera_check(
            self._verification_frame,
            lambda image: verify_preconditions(image, phase, invented, self.cfg, self.caps),
        )
        return self._preconditions_verdict(check, index, what="human phase", description=phase.description)

    def _leg_preconditions_hold(self, scene, run: Sequence[Phase]) -> bool:
        """Is the world still as the plan believes, before the planner is asked to plan on it?

        ``check_tamp_preconditions``. A robot phase declares no preconditions of its own, so what is
        checked is what EARLIER phases are responsible for (``PhasePlan.expected_now``): the drift
        this catches is a human phase verified as done that left something undone. Checked on this
        pass's perception image rather than a fresh grab, so the check and the plan look at the same
        workspace. False means the trial was ended (``precondition_enforced``).
        """
        from tandem.planning.grounding import checkable, verify_atoms

        expected = self._plan.expected_now()
        invented = self._invented()
        if not checkable(expected, invented, self.caps):
            return True
        check = self._camera_check(
            lambda: _open_image(scene.rgb_path),
            lambda image: verify_atoms(
                image, invented, self.cfg, self.caps, expect_true=expected, role="precondition"
            ),
        )
        return self._preconditions_verdict(
            check, self._plan.index, what="robot leg", description="; ".join(p.description for p in run)
        )

    def _preconditions_verdict(
        self, check: _Check, index: int | None, *, what: str, description: str
    ) -> bool:
        """Record a precondition check and decide whether the work goes ahead.

        Serves both halves of the contract, a human phase's declared preconditions and a robot leg's
        expected state. A check that could not run is not a reason to stop, the same rule the effect
        check follows. Whether an unmet precondition stops anything is ``precondition_enforced``: off
        (the default), it is recorded and reported and the work goes ahead.
        """
        cfg = self.cfg
        if check.error is not None:
            self._record_unchecked(index, f"{what} precondition check", check.error)
            self.events.log(
                f"could not check the {what} preconditions; carrying on unchecked ({check.error})"
            )
            self.events.event(
                "phase_preconditions_checked",
                phase_index=index,
                description=description,
                what=what,
                ok=None,
                enforced=cfg.precondition_enforced,
                unchecked=check.error,
                verdicts=[],
            )
            return True

        self._record_verdicts(index, check.verdicts)
        verdicts = check.summaries()
        self.events.event(
            "phase_preconditions_checked",
            phase_index=index,
            description=description,
            what=what,
            ok=check.ok,
            enforced=cfg.precondition_enforced,
            verdicts=verdicts,
        )
        if check.ok:
            self.events.log(f"the {what} preconditions hold")
            return True
        detail = "; ".join(_missing_from(verdicts))
        if not cfg.precondition_enforced:
            self.events.log(
                f"the {what} preconditions do not hold ({detail}), but precondition_enforced is off; "
                "carrying on"
            )
            return True
        self._verification_failed(f"the preconditions of the {what} {description!r} do not hold: {detail}")
        return False

    def _check_leg_effects(self, run: Sequence[Phase]) -> None:
        """Did a robot leg leave the workspace as its phases said it would? Observational only.

        ``check_tamp_effects``. It never stops anything: the planner either executed a plan for
        ``On(toy, box)`` or reported that it could not, and the arm's own account of that is better
        evidence than a third-person camera. This makes the camera's disagreement visible, and puts it
        on the record: a placement the robot believes it made and the image does not show is worth
        knowing about when a later phase fails.
        """
        from tandem.planning.grounding import checkable, verify_atoms

        index = self._plan.index
        atoms: frozenset[Atom] = frozenset().union(*(p.add_effects for p in run))
        invented = self._invented()
        if not checkable(atoms, invented, self.caps):
            return
        check = self._camera_check(
            self._verification_frame,
            lambda image: verify_atoms(
                image, invented, self.cfg, self.caps, expect_true=atoms, role="effect"
            ),
        )
        description = "; ".join(p.description for p in run)
        if check.error is not None:
            self._record_unchecked(index, "robot leg effect check", check.error)
            self.events.log(f"could not check the robot leg's effects ({check.error})")
            self.events.event(
                "phase_effects_checked",
                phase_index=index,
                description=description,
                what="robot leg",
                ok=None,
                enforced=False,
                unchecked=check.error,
                verdicts=[],
            )
            return

        # One leg can carry several conjoined phases. Each verdict is filed against the first phase
        # in the run that asked for its atom, so the record reads per phase like every other verdict.
        filed: set[Atom] = set()
        for offset, phase in enumerate(run):
            mine = [v for v in check.verdicts if v.atom in phase.add_effects and v.atom not in filed]
            filed.update(v.atom for v in mine)
            self._record_verdicts(index + offset, mine)
        verdicts = check.summaries()
        self.events.event(
            "phase_effects_checked",
            phase_index=index,
            description=description,
            what="robot leg",
            ok=check.ok,
            enforced=False,
            verdicts=verdicts,
        )
        if not check.ok:
            self.events.log(
                f"the robot leg ran, but the camera does not show: {'; '.join(_missing_from(verdicts))}"
            )

    def _verification_failed(self, reason: str) -> None:
        """End the trial over a camera check that stopped it.

        What becomes of it is ``on_verification_failure``. With ``exclude`` (the paper's rule, and the
        default) it is EXCLUDED: the session files it under failure/ without asking for a label, with
        its failing verdicts and its raw legs kept for inspection. With ``label`` it is a failure the
        operator is asked about like any other, and their answer is the trial's outcome -- the setting
        for calibrating the classifier, where each disagreement with the check is the data point.
        """
        self._show_human_phase(None)
        self._plan = None
        self._task_done = True
        if self.cfg.on_verification_failure == "exclude":
            self.events.log(f"{reason}; ending this attempt and excluding it from the dataset")
            self._end("excluded", "verification", reason)
        else:
            self.events.log(f"{reason}; ending this attempt, and your label decides it")
            self._end("failure", "verification", reason)

    def _human_leg(self, phase) -> None:
        """Give the arm to a person for one leg, through the human-leg seam."""
        self.human_leg(phase, self._leg_recorded)

    def _show_human_phase(self, view: HumanPhase | None) -> None:
        # The loop keeps its own reference as well, because a retry carries the previous attempt's
        # `missing` forward and the verdict is written back onto the view the operator is looking at.
        self._human_phase = view
        self.operator.show_human_phase(view)

    # ---- the camera ---------------------------------------------------------

    def _invented(self):
        return self._plan.spec.invented if self._plan is not None else ()

    def _verification_frame(self):
        """A fresh frame from the verification camera: after a phase, not the perception pass's."""
        return _open_image(self.backend.capture_frame(camera=self.cfg.verification_camera))

    def _camera_check(
        self,
        frame: Callable[[], Any],
        verify: Callable[[Any], Awaitable[tuple[bool, list[Verdict]]]],
    ) -> _Check:
        """Take a frame and put it to the classifier, as one guarded step.

        `frame` is a thunk rather than an image because grabbing it is a camera read, and a camera
        read can fail exactly as a classifier call can. Either failure is a check that could not run
        (`_Check.error`), never a failed one: one unreachable service must not cost the operator a
        demonstration. That includes the ValueError ``verify_atoms`` raises for an operator that adds
        and deletes the same atom -- a bug upstream, reported as the error rather than as a phase the
        person could never pass.
        """
        import asyncio

        from tandem.planning.record import recording_to

        try:
            image = frame()
            with recording_to(self._vlm_dir):
                ok, verdicts = asyncio.run(verify(image))
        except Exception as exc:
            return _Check(ok=True, error=f"{type(exc).__name__}: {exc}")
        return _Check(ok=ok, verdicts=list(verdicts))

    def _record_verdicts(self, index: int | None, verdicts: Iterable[Verdict]) -> None:
        verdicts = list(verdicts)
        if self._plan is not None and index is not None and verdicts:
            self._plan.record_verdicts(index, verdicts)

    def _record_unchecked(self, index: int | None, what: str, error: str) -> None:
        """Put a check that could not run on the record, beside any other for the same phase."""
        if self._plan is None or index is None:
            return
        reason = f"{what}: {error}"
        earlier = self._plan.unchecked.get(index)
        self._plan.record_unchecked(index, f"{earlier}; {reason}" if earlier else reason)


def _open_image(path):
    """A saved camera frame as the image the classifier is given."""
    from PIL import Image

    from tandem.planning.grounding import to_pil

    return to_pil(Image.open(path).convert("RGB"))


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
