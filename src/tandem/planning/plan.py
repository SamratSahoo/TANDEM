"""One task's phases, and how far through them we are.

A phase-planned task is not one rollout. Each robot phase is an ordinary planner rollout aimed at
that phase's sub-goal, and each human phase is a teleop leg -- so the plan has to survive the
hand-off between them. It lives here, in tandem's own process, which is the difference this refactor
makes: the plan used to live inside the planner's process, where tandem could neither see it nor
change it, and could only watch a `rollout_start` go by and guess which phase it belonged to.

It is also the trial's record. Everything ``hitl.json`` says about a trial -- which phases ran, what
the planner found for each, which checks were on and what they saw, how the trial ended -- is kept on
the `PhasePlan` as it happens and written out by ``to_json``.
"""

from __future__ import annotations

import difflib
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from tandem.core.errors import TandemError
from tandem.planners.base import Capabilities, GoalAtom, to_goal_atoms
from tandem.planning import contracts, feasibility
from tandem.planning.config import PlanningConfig
from tandem.planning.grounding import Verdict, describe_expectations, descriptions_for
from tandem.planning.structs import Phase, TaskSpecification
from tandem.planning.symbols import Atom, describe

_log = logging.getLogger(__name__)

# How a trial ended, and at which stage when it ended early. The stages are the paper's (Fig. 4,
# Table III) plus `verification`, which is where a trial the method excludes is stopped.
OUTCOMES = ("success", "failure", "excluded", "aborted")
FAILURE_STAGES = ("invention", "tamp_planning", "tamp_execution", "verification", "human_policy")

# Appended to the proposal prompt when a plan is proposed again because the last one could not be
# carried out (`on_robot_phase_failure: replan`). Without it a replan is the same question asked of
# the same scene, and the model answers it the same way: the phase the planner could not plan comes
# back unchanged, and the retry budget is spent reproducing the failure.
_REPLAN = """\
An earlier plan for this task could not be carried out.

What went wrong:
{feedback}

Plan the task again from the workspace as it is in this image, in a way that does not run into that
problem."""

# The `?` a PDDL-style signature marks each parameter with: `Pick(?obj: movable)`.
_PARAMETER_MARK = re.compile(r"([(,]\s*)\?")


def operator_signature(signature: str) -> str:
    """``Pick(?obj: movable)`` -> ``Pick(obj: movable)``: the one spelling the record uses.

    The record names operators from two sources, and they disagree on one character. A human
    operator writes its own signature (``HumanOperator.signature``) as ``Open(x0: surface)``, which
    is the form ``HumanOperator.from_json`` reads back. A backend declares its operators
    (``Capabilities.robot_operators``) the way it declares its goal predicates, ``Pick(?obj:
    movable)``. Printed side by side in one record, the difference reads as though it meant
    something. The human side's spelling wins because it is the one already in every ``hitl.json``
    and the one the record is parsed back with; a declaration is only reworded, never checked, since
    it is provenance and nothing branches on it.
    """
    return _PARAMETER_MARK.sub(r"\1", signature.strip())


def _choice(value: str | None, choices: Sequence[str], what: str) -> None:
    """Refuse a value outside ``choices``, naming the nearest one."""
    if value in choices:
        return
    close = difflib.get_close_matches(str(value), list(choices), n=1)
    hint = f" Did you mean {close[0]!r}?" if close else ""
    raise ValueError(f"{what} must be one of {', '.join(choices)}, got {value!r}.{hint}")


@dataclass
class PhasePlan:
    """One task's plan, how far through it we are, and what has happened to it so far."""

    cfg: PlanningConfig
    caps: Capabilities
    instruction: str
    trajectory_id: str | None
    spec: TaskSpecification
    index: int = 0
    verdicts: list[Verdict] = field(default_factory=list)
    initially_true: frozenset[Atom] = frozenset()
    # Whether `initially_true` was MEASURED (classify_initial ran). An empty set means two different
    # things with and without it: "nothing invented holds in the workspace" versus "nobody looked".
    initial_state_known: bool = False
    # The plan-time contract check, run again once the starting state was measured
    # (`recheck_plan_effects`). `plan_effects_rechecked` says it ran; `inconsistency` is what it found.
    plan_effects_rechecked: bool = False
    inconsistency: str | None = None
    # Phase index -> what the planner reported about the plan it found, filled in as each one runs.
    plans: dict[int, dict] = field(default_factory=dict)
    # Phase index -> why the camera check of that phase could not be run, for a phase accepted
    # unchecked. Kept apart from `verdicts` because there is no verdict: the classifier never answered.
    unchecked: dict[int, str] = field(default_factory=dict)
    # How the trial ended, when the loop ended it itself (`set_outcome`). None for a trial that ran
    # to the end: the operator's label is its verdict, and the episode writer fills it in from that.
    outcome: str | None = None
    failure_stage: str | None = None
    # Position in `verdicts` -> the phase that verdict was about, for verdicts recorded through
    # `record_verdicts`. Keyed by position rather than kept as a parallel list, so a caller that
    # extends `verdicts` directly leaves those entries without a phase instead of misaligning them.
    _verdict_phases: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    @property
    def phases(self) -> tuple[Phase, ...]:
        return self.spec.phases

    @property
    def finished(self) -> bool:
        return self.index >= len(self.phases)

    @property
    def current(self) -> Phase | None:
        return None if self.finished else self.phases[self.index]

    def matches(self, instruction: str, trajectory_id: str | None) -> bool:
        """Whether this plan is still the right one for the rollout about to run.

        A different instruction is a different task. A different trajectory means the previous one was
        closed out (labeled, merged) and this is a fresh attempt, which must re-perceive and re-plan
        rather than resume halfway through a plan made for a scene that no longer exists.
        """
        return self.instruction == instruction and self.trajectory_id == trajectory_id

    def next_is_human(self) -> bool:
        return self.current is not None and self.current.is_human

    def is_final_phase(self) -> bool:
        """Whether the phase now current is the LAST one in the plan -- nothing follows it.

        Distinct from ``is_last_leg``, which asks about the whole ``robot_run()`` a leg covers: a leg
        of three conjoined robot phases is the last leg while only its third phase is the final one.
        This is the phase-level question, which is what decides whether a human phase is verified at
        all (``verify_final_phase``) and what the operator's one "next" control does once it is done.

        False for a finished plan: there is no current phase to be the final one.
        """
        return not self.finished and self.index + 1 >= len(self.phases)

    def is_last_leg(self) -> bool:
        """Whether the leg about to run is the last one of the task -- nothing follows it.

        Mirrors ``advance``: a human phase is one step, a robot leg is the whole ``robot_run()`` its
        single goal covers. This is what tells the planner whether to end the leg by driving the arm
        home (``TampBackend.plan(return_home=)``). Only the LAST leg should: a home in the middle of a
        task is motion nobody asked for, recorded into the middle of the demonstration, and the next
        leg then starts from home rather than from where this one left off.

        True for a finished plan too, so a caller that asks before checking ``finished`` gets the
        conservative answer (go home) rather than the surprising one.
        """
        return self.index + max(1, len(self.robot_run())) >= len(self.phases)

    def robot_run(self) -> tuple[Phase, ...]:
        """The consecutive robot phases this leg plans and executes as ONE goal.

        Empty when the plan is finished or the next phase is a human's. See
        ``feasibility.conjoinable_run`` for why consecutive robot phases may be conjoined, and what
        stops the run. With ``conjoin_robot_phases`` off, every robot phase is a leg of its own and is
        planned against its own perception pass: the paper's "re-perceive after every phase" read
        strictly, at the cost of the arm stopping between phases.
        """
        phase = self.current
        if phase is None or phase.is_human:
            return ()
        if not self.cfg.conjoin_robot_phases:
            return (phase,)
        remaining = self.phases[self.index :]
        return tuple(remaining[: feasibility.conjoinable_run(remaining, self.caps)])

    def goal(self) -> list[GoalAtom]:
        """This leg's goal, in the planner's own goal language.

        Every phase in ``robot_run()``, conjoined -- the backend satisfies the set, so two
        pick-and-places are one plan.
        """
        run = self.robot_run()
        if not run:
            raise ValueError("goal() is only for a robot phase")
        atoms: frozenset[Atom] = frozenset().union(*(p.atoms for p in run))
        return to_goal_atoms(sorted(atoms, key=str), self.caps)

    def surfaces(self) -> frozenset[str]:
        """Which objects are surfaces, pinned for the whole task.

        Passed on every leg, not just the ones that place onto something. A backend that UNIONS this
        with what it inferred still ends up with the right set, because the split was computed from
        every phase's atoms together; a backend that overrides gets the same answer. Either way an
        object cannot change type -- and with it, whether it is a static obstacle -- mid-task.
        """
        return self.spec.scene_types.surfaces

    def robot_movables(self) -> frozenset[str]:
        """The objects the planner may pick up: exactly the ones some ROBOT phase moves.

        A scene shared with a person contains the person's things. Perception detects them because
        the instruction mentions them -- "remove a block from the jenga tower USING THE SCREWDRIVER"
        is what makes the screwdriver a labelled object at all -- and every non-surface detection is
        a movable by default, so a planner told nothing treats the human's tool as a thing to pick up.
        On the run this was written for, three of the four skeletons cuTAMP enumerated for "put the
        block back on the tower" opened with Pick(screwdriver), and one of them placed the screwdriver
        on the tower. Nothing about that is wrong by the planner's lights; it was never told whose the
        tool is. The loop passes this as ``plan(movables=)`` where the backend declares
        ``supports_movable_restriction``, and everything left out stays in the scene as an obstacle.

        Computed over EVERY robot phase, not the leg about to run, for the reason ``SceneTypes`` gives
        for surfaces: an object that is pickable in one leg and a static obstacle in the next changes
        what the search may do partway through one task. What a phase moves is read through the
        backend's ``moved_arguments`` (``contracts.phase_moves``) -- the toy in ``On(toy, box)``, never
        the box -- and kept to the scene's movables, so a surface is never offered as something to pick.
        """
        if not self.caps.moved_arguments:
            # Not an empty answer: an empty set handed to plan(movables=) forbids every pick, and a
            # backend that honours it would fail every robot leg with no hint why.
            raise TandemError(
                f"The {self.caps.name!r} planner declares no moved_arguments, so tandem cannot tell "
                "which objects a robot phase moves, and cannot say which ones the planner may pick up.",
                hint=(
                    "Declare Capabilities.moved_arguments (for a pick-and-place goal language, the "
                    "position of the placed object, e.g. {'On': 0}), or leave "
                    "supports_movable_restriction False."
                ),
            )
        moved: set[str] = set()
        for phase in self.phases:
            if not phase.is_human:
                moved |= contracts.phase_moves(phase, caps=self.caps)
        return frozenset(moved) & self.spec.scene_types.movables

    def expected_now(self) -> frozenset[Atom]:
        """What EARLIER phases should have left true, entering the leg about to run.

        The robot-side precondition set (``check_tamp_preconditions``): every atom some earlier phase
        was responsible for establishing and no later one undid. See ``contracts.expected_before`` for
        why nothing else is put to a camera. Empty for a finished plan, which has no leg to enter.
        """
        if self.finished:
            return frozenset()
        return contracts.expected_before(self.spec, self.index, self.initially_true, caps=self.caps)

    def recheck_plan_effects(self) -> str | None:
        """Run the plan-time contract check again, now that the starting state is measured.

        The proposal already passed this check, against a starting workspace nobody had looked at. Once
        ``classify_initial`` has measured it, the same check can prove more: a precondition over an
        invented predicate that no phase establishes, and that the scene does not already satisfy, is
        now provably unmet. Recorded rather than refused. The proposer is out of the loop by now (the
        repair happened before the image was classified), and the phases are still worth running with
        the gap on the record.

        Runs only with ``check_plan_effects`` on and a MEASURED starting state -- keyed on the
        measurement having happened, not on it having found something true: a measurement that found
        nothing true is exactly when a never-established precondition is provable.
        """
        if not self.cfg.check_plan_effects or not self.initial_state_known:
            return None
        self.inconsistency = contracts.check_plan_effects(
            self.spec, self.initially_true, initial_state_known=True, caps=self.caps
        )
        self.plan_effects_rechecked = True
        return self.inconsistency

    def objects_named(self) -> set[str]:
        """Every object the remaining phases refer to, for the label-drift check."""
        names: set[str] = set()
        for phase in self.phases[self.index :]:
            names.update(phase.objects)
        return names & self.spec.scene_types.all_names

    def objects_needed_now(self) -> set[str]:
        """The subset of ``objects_named()`` this leg cannot proceed without.

        The difference is the whole point. ``objects_named()`` spans every phase still to come, human
        ones included, so a task whose LAST phase asks a person to cover the bowls with a cloth names
        that cloth on every leg before it -- and a robot phase that never mentions the cloth would be
        abandoned mid-task just because perception missed it once. What a leg actually needs is the
        phase it is about to carry out, plus the surfaces: those are pinned once for the whole task,
        and a surface that drifted away un-rebound would quietly become a movable, changing the world
        geometry between phases.
        """
        names = set(self.spec.scene_types.surfaces)
        for phase in self.robot_run() or ([self.current] if self.current is not None else []):
            names.update(phase.objects)
        return names & self.spec.scene_types.all_names

    def rebind(self, mapping: dict[str, str]) -> None:
        """Rename the plan's objects to this pass's labels, keeping the progress made so far."""
        if not mapping:
            return
        _log.info(f"re-binding plan objects to this pass's labels: {mapping}")
        self.spec = self.spec.rebind(mapping)
        self.initially_true = frozenset(a.rebind(mapping) for a in self.initially_true)

    def hand_current_to_human(self, instructions: str = "") -> Phase:
        """Turn the phase now due into a human one, and return it.

        This is what ``on_robot_phase_failure: teleop`` does. It is only possible because tandem owns
        the executor split: it used to be frozen at proposal time inside the planner's process, so a
        phase the planner turned out not to be able to plan could only end the attempt.
        """
        phase = self.current
        if phase is None:
            raise ValueError("the plan is finished")
        if phase.is_human:
            return phase
        handed = phase.as_human(instructions or self._describe_as_instructions(phase))
        self.spec = self.spec.replace_phase(self.index, handed)
        return handed

    def _describe_as_instructions(self, phase: Phase) -> str:
        """Wording for a robot phase a person is being asked to do instead.

        Built from the phase's own atoms rather than its description, so what the operator is asked
        for is exactly what will be checked afterwards.
        """
        wants = describe_expectations(phase, descriptions_for(self.spec.invented, self.caps))
        return (
            f"The planner could not work out how to {phase.description}. Do it by hand: "
            + "; ".join(wants)
            + "."
        )

    def advance(self) -> Phase | None:
        """Move past the work this leg carried out, and return the phase it started at.

        A human phase is one step. A robot leg is the whole ``robot_run()`` its single goal covered,
        so the plan does not stop between phases that were planned and executed together.
        """
        phase = self.current
        if phase is not None:
            self.index += max(1, len(self.robot_run()))
        return phase

    # ---- the record ---------------------------------------------------------

    def record_plan(self, index: int, result: Any) -> None:
        """Note what the planner reported, against every phase this leg carried out.

        One plan can cover several phases (``robot_run``), so each of them records it, and any phase
        sharing one also records which ones it was planned with -- otherwise the audit trail reads as
        though each phase had been solved on its own.

        ``task_plan`` is the operator sequence the planner ran (``PlanResult.task_plan``):
        ``Pick(bread)``, ``Place(bread, plate)`` -- the robot phase's half of what the paper's figures
        show next to each human operator. Left out when the planner did not say, which is not the same
        as a plan with nothing in it.

        A phase can be recorded twice: a replanned leg records again over the one it replaced. The
        later record wins, since it describes the plan that actually ran.
        """
        covered = [index]
        if index == self.index:
            covered = [index + offset for offset in range(max(1, len(self.robot_run())))]
        record: dict[str, Any] = {
            "planning_seconds": round(float(getattr(result, "planning_seconds", 0.0) or 0.0), 3),
            "plan_reused": bool(getattr(result, "skeleton_reused", False)),
        }
        # Read only when it is actually a sequence: a stand-in result that does not carry one is a
        # planner that did not say, and must not be recorded as an operator named after its repr.
        task_plan = getattr(result, "task_plan", ())
        if isinstance(task_plan, (list, tuple)) and task_plan:
            record["task_plan"] = [str(step) for step in task_plan]
        if len(covered) > 1:
            record["covers_phases"] = covered
        for i in covered:
            # A copy each, lists included: phases that shared a plan must not share one record.
            self.plans[i] = {k: list(v) if isinstance(v, list) else v for k, v in record.items()}

    def record_verdicts(self, index: int, verdicts: Iterable[Verdict]) -> None:
        """Keep what the camera said about phase ``index``, failing verdicts included.

        The failing ones matter most. A trial excluded for a phase that never verified is audited by
        exactly those verdicts, and a record that kept only the passing ones reads as a trial that was
        never checked. Each is written with the phase it was about.
        """
        for verdict in verdicts:
            self._verdict_phases[len(self.verdicts)] = index
            self.verdicts.append(verdict)

    def record_unchecked(self, index: int, reason: str) -> None:
        """Note that phase ``index`` was accepted without its camera check, and why.

        A classifier or camera error does not cost the operator a demonstration, so such a phase goes
        ahead. It has no verdicts, and without this the record could not tell it from a phase that was
        checked and passed.
        """
        self.unchecked[index] = str(reason)

    def set_outcome(self, outcome: str, failure_stage: str | None = None) -> None:
        """How the trial ended, when the loop ended it: one of ``OUTCOMES``, at one of ``FAILURE_STAGES``.

        Refused outside those, with the nearest name: a misspelt stage is a trial the analysis
        silently files under none of them.
        """
        _choice(outcome, OUTCOMES, "outcome")
        if failure_stage is not None:
            _choice(failure_stage, FAILURE_STAGES, "failure_stage")
            if outcome == "success":
                raise ValueError(f"A successful trial has no failure stage, got {failure_stage!r}.")
        self.outcome = outcome
        self.failure_stage = failure_stage

    def checks(self) -> dict:
        """Which halves of the operator contract this run put to a camera, and what else it checked.

        Without it a phase with no verdicts is ambiguous between "checked and fine" and "never
        checked". The switches are as configured; the rest says what actually ran.
        """
        cfg = self.cfg
        return {
            "human_preconditions": cfg.check_human_preconditions,
            "human_effects": cfg.check_human_effects,
            "tamp_preconditions": cfg.check_tamp_preconditions,
            "tamp_effects": cfg.check_tamp_effects,
            "plan_effects": cfg.check_plan_effects,
            "verify_enforced": cfg.verify_enforced,
            "precondition_enforced": cfg.precondition_enforced,
            "verify_final_phase": cfg.verify_final_phase,
            # Whether the starting workspace was measured, and whether the plan was then held to it.
            "initial_state_classified": self.initial_state_known,
            "plan_effects_rechecked": self.plan_effects_rechecked,
            # What that re-check found. The plan ran regardless; this is why a phase may fail anyway.
            "plan_effects_warning": self.inconsistency,
            # Phases accepted without a verdict because the check itself could not run.
            "unchecked_phases": sorted(self.unchecked),
        }

    def phase_record(self, index: int) -> dict:
        """One phase, with exactly what was handed to whoever carried it out."""
        phase = self.phases[index]
        # The phase's own summary carries its magic operator, for a human phase that has one.
        record = {"index": index, **phase.summary()}
        record["planned_by"] = (
            "vlm" if phase.is_human else f"vlm (order and sub-goal); {self.caps.name} (how)"
        )
        if not phase.is_human:
            # What the planner is actually given. Note it is ATOMS, not a sentence: the ordinary path
            # runs the instruction through a model to get these, and a phase substitutes them
            # directly. `goal` is the literal list the backend consumes; `goal_description` is the
            # same thing in words, for reading.
            record["goal"] = [a.to_dict() for a in to_goal_atoms(sorted(phase.atoms, key=str), self.caps)]
            record["goal_description"] = [
                describe(a, dict(self.caps.predicate_descriptions)) for a in sorted(phase.atoms, key=str)
            ]
            record.update(self.plans.get(index, {}))
        if index in self.unchecked:
            record["unchecked"] = self.unchecked[index]
        return record

    def _operators(self) -> tuple[list[str], list[str]]:
        """Omega_Delta and Omega_0: the human and robot operator signatures, in one spelling."""
        human: list[str] = []
        for operator in self.spec.operators:
            signature = operator_signature(operator.signature)
            if signature not in human:
                human.append(signature)
        robot = [operator_signature(s) for s in self.caps.robot_operators]
        return human, robot

    def to_json(self) -> dict:
        human_operators, robot_operators = self._operators()
        executor = self.cfg.human_executor
        verifications = []
        for position, verdict in enumerate(self.verdicts):
            entry = verdict.summary()
            if position in self._verdict_phases:
                entry["phase"] = self._verdict_phases[position]
            verifications.append(entry)
        return {
            "instruction": self.instruction,
            "trajectory_id": self.trajectory_id,
            "planner": self.caps.name,
            "human_executor": executor,
            # Set only when the loop ended the trial itself. The episode writer resolves the final
            # outcome against the operator's label (tandem.core.episodes).
            "outcome": self.outcome,
            "failure_stage": self.failure_stage,
            "excluded": self.outcome == "excluded",
            "specification": self.spec.to_json(),
            "initially_true": sorted(str(a) for a in self.initially_true),
            # Who produced which part of all this. The short version: the model decides WHAT each
            # phase must achieve and IN WHAT ORDER; the planner decides how the robot's phases are
            # carried out; tandem decides who does what and when.
            "provenance": {
                "phases_and_their_order": (
                    "vlm -- the instruction is broken into an ordered list of robot and human phases. "
                    "This replaced a symbolic search over the robot's operators plus invented ones, "
                    "which could only ever order a human step AFTER robot work, never before it"
                ),
                "phase_sub_goals": "vlm -- the atoms each robot phase must establish",
                "invented_predicates": "vlm -- name and the natural-language classifier behind it",
                "human_instructions": "vlm -- the text the operator is shown",
                # Both operator lists use one signature spelling, `Name(param: type)`; see
                # operator_signature for why it is the human side's.
                "human_operators": {
                    "by": (
                        "vlm -- each human phase is an explicit operator (name, args, preconditions, "
                        "add effects, delete effects). It is never searched over: the phase order is "
                        "already fixed. It is the contract the camera checks and the plan-time "
                        "consistency check are stated over"
                    ),
                    "signatures": human_operators,
                },
                "robot_operators": {
                    "by": (
                        f"{self.caps.name} -- the planner's own, fixed, with their own preconditions and "
                        "effects; nothing invents a robot operator"
                    ),
                    "signatures": robot_operators,
                },
                "robot_phases": (
                    f"{self.caps.name} -- each phase is planned from its goal against a fresh perception pass"
                ),
                "who_does_what": "tandem -- the executor split, the hand-offs, and the verification",
                "human_steps": (
                    "the teleoperator, following the phase's instructions"
                    if executor == "teleop"
                    else f"the {executor!r} human executor, given the phase's instructions and operator"
                ),
            },
            "checks": self.checks(),
            "phases": [self.phase_record(i) for i in range(len(self.phases))],
            "phase_index": self.index,
            "verifications": verifications,
        }


async def build_plan(
    image: Any,
    instruction: str,
    object_names: Sequence[str],
    table_name: str,
    cfg: PlanningConfig,
    caps: Capabilities,
    trajectory_id: str | None,
    *,
    feedback: str | None = None,
) -> tuple[PhasePlan | None, str | None]:
    """Propose the plan for this task. Returns ``(plan, failure_reason)``.

    A plan whose phases are all robot ones needs no human, which is how a task the planner already
    handles behaves exactly as it did before.

    Every check that can send a plan back for repair runs inside the proposal's repair loop now,
    ``feasibility.check_robot_phases`` included, so a plan that never validates raises
    ``ProposalError`` from ``propose_plan`` rather than coming back here as a failure reason. What is
    checked here can no longer be repaired, so it is recorded rather than refused.

    ``feedback`` is why the last plan for this task could not be carried out, in words -- the
    ``replan`` policy's whole point. It goes to the model with the prompt, and the proposal it gets
    back is never read from or written to the cache.
    """
    from tandem.planning.grounding import classify_initial_state
    from tandem.planning.proposal import propose_plan

    section = None
    if feedback and feedback.strip():
        _log.info(f"proposing again, told why the last plan failed: {feedback.strip()}")
        section = _REPLAN.format(feedback=feedback.strip())
    spec = await propose_plan(image, instruction, object_names, table_name, cfg, caps, feedback=section)

    plan = PhasePlan(cfg=cfg, caps=caps, instruction=instruction, trajectory_id=trajectory_id, spec=spec)
    if cfg.classify_initial and spec.invented:
        plan.initially_true = await classify_initial_state(image, spec, cfg, caps)
        plan.initial_state_known = True
        if plan.initially_true:
            _log.info(f"already true before starting: {sorted(str(a) for a in plan.initially_true)}")
        broken = plan.recheck_plan_effects()
        if broken:
            _log.warning(f"the plan does not hang together against the measured scene: {broken}")
    if not spec.needs_human:
        _log.info("the robot can do this whole task on its own; no human phases were proposed")
    return plan, None


def phase_summary(plan: PhasePlan, phase: Phase) -> dict:
    """The human phase as it goes into the session's event stream.

    Carries the phase's position in the plan, which the old integration did not: the emitter sent
    only description and expectations while the consumer read ``phase_index``/``n_phases`` off it, so
    "step N of M" in the UI was always the fallback and usually wrong. tandem holds the plan now, so
    there is one source for both.

    ``is_last_phase`` is the plan's decision about what the operator's one "next" control does once
    the step is done -- carry on with the robot, or close the trajectory out -- so the UI does not
    have to work it out. The explicit operator goes along when there is one, so the UI can show what
    the step needs and what it changes rather than only the sentence the person was given.
    """
    summary = {
        "description": phase.description,
        "instructions": phase.instructions,
        "expected": describe_expectations(phase, descriptions_for(plan.spec.invented, plan.caps)),
        # What the camera will be asked, as atoms, the two halves kept apart: `expected_atoms` must
        # hold afterwards and `expected_deleted_atoms` must no longer hold. One list would state a
        # delete effect as something to bring about -- the opposite of what the step is for.
        "expected_atoms": sorted(str(a) for a in phase.add_effects),
        "expected_deleted_atoms": sorted(str(a) for a in phase.delete_effects),
        "phase_index": plan.index,
        "n_phases": len(plan.phases),
        "is_last_phase": plan.is_final_phase(),
    }
    if phase.operator is not None:
        summary["operator"] = phase.operator.to_json()
    return summary


def handoff_message(plan: PhasePlan, phase: Phase) -> str:
    """What the operator is shown when the plan reaches a human phase.

    The instructions come from the phase itself; the expectations come from its atoms, so the human
    knows what will be checked afterwards -- the same list the model is about to be asked about.
    """
    descriptions = descriptions_for(plan.spec.invented, plan.caps)
    lines = [
        f"HUMAN STEP {plan.index + 1} of {len(plan.phases)}: {phase.description}",
        "",
        phase.instructions,
        "",
        "When you are done, the following should be true:",
    ]
    lines += [f"  - {text}" for text in describe_expectations(phase, descriptions)]
    remaining = len(plan.phases) - plan.index - 1
    if remaining:
        lines.append("")
        lines.append(f"({remaining} more phase(s) follow, so the robot carries on after this.)")
    return "\n".join(lines)


def retry_message(missing: Sequence[str], attempts_left: int, *, by: str = "teleop") -> str:
    """What the operator is shown when the check says the phase is not done.

    ``by`` is the human executor that carried the phase out (``hitl.human_executor``). It changes only
    the last line, which is an instruction to whoever gets another go: a person at the arm
    (``teleop``) is asked to take it again, and any other executor is simply said to be run again --
    "take the arm" printed into a log while a policy drives would be an instruction to nobody.
    """
    lines = ["The workspace does not look like that step was completed.", "Still expected:"]
    lines += [f"  - {text}" for text in missing]
    if attempts_left > 0:
        lines.append("")
        lines.append(
            "Take the arm again and finish it, or say it IS done."
            if by == "teleop"
            else f"Running the {by} executor on this phase again ({attempts_left} attempt(s) left)."
        )
    return "\n".join(lines)
