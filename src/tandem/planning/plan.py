"""One task's phases, and how far through them we are.

A phase-planned task is not one rollout. Each robot phase is an ordinary planner rollout aimed at
that phase's sub-goal, and each human phase is a teleop leg -- so the plan has to survive the
hand-off between them. It lives here, in tandem's own process, which is the difference this refactor
makes: the plan used to live inside the planner's process, where tandem could neither see it nor
change it, and could only watch a `rollout_start` go by and guess which phase it belonged to.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from tandem.planners.base import Capabilities, GoalAtom, to_goal_atoms
from tandem.planning import feasibility
from tandem.planning.config import PlanningConfig
from tandem.planning.grounding import Verdict, describe_expectations, descriptions_for
from tandem.planning.structs import Phase, TaskSpecification
from tandem.planning.symbols import Atom, describe

_log = logging.getLogger(__name__)


@dataclass
class PhasePlan:
    """One task's plan, and how far through it we are."""

    cfg: PlanningConfig
    caps: Capabilities
    instruction: str
    trajectory_id: str | None
    spec: TaskSpecification
    index: int = 0
    verdicts: list[Verdict] = field(default_factory=list)
    initially_true: frozenset[Atom] = frozenset()
    # Phase index -> what the planner reported about the plan it found, filled in as each one runs.
    plans: dict[int, dict] = field(default_factory=dict)

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

    def robot_run(self) -> tuple[Phase, ...]:
        """The consecutive robot phases this leg plans and executes as ONE goal.

        Empty when the plan is finished or the next phase is a human's. See
        ``feasibility.conjoinable_run`` for why consecutive robot phases may be conjoined, and what
        stops the run.
        """
        remaining = self.phases[self.index :]
        count = feasibility.conjoinable_run(remaining, self.spec.scene_types.movables, self.caps)
        return tuple(remaining[:count])

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

    def record_plan(self, index: int, result: Any) -> None:
        """Note what the planner reported, against every phase this leg carried out.

        One plan can cover several phases (``robot_run``), so each of them records it, and any phase
        sharing one also records which ones it was planned with -- otherwise the audit trail reads as
        though each phase had been solved on its own.
        """
        covered = [index]
        if index == self.index:
            covered = [index + offset for offset in range(max(1, len(self.robot_run())))]
        record = {
            "planning_seconds": round(float(getattr(result, "planning_seconds", 0.0) or 0.0), 3),
            "plan_reused": bool(getattr(result, "skeleton_reused", False)),
        }
        if len(covered) > 1:
            record["covers_phases"] = covered
        for i in covered:
            self.plans[i] = dict(record)

    def phase_record(self, index: int) -> dict:
        """One phase, with exactly what was handed to whoever carried it out."""
        phase = self.phases[index]
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
        return record

    def to_json(self) -> dict:
        return {
            "instruction": self.instruction,
            "trajectory_id": self.trajectory_id,
            "planner": self.caps.name,
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
                "robot_phases": (
                    f"{self.caps.name} -- each phase is planned from its goal against a fresh perception pass"
                ),
                "who_does_what": "tandem -- the executor split, the hand-offs, and the verification",
                "human_steps": "the teleoperator, following the phase's instructions",
            },
            "phases": [self.phase_record(i) for i in range(len(self.phases))],
            "phase_index": self.index,
            "verifications": [v.summary() for v in self.verdicts],
        }


async def build_plan(
    image: Any,
    instruction: str,
    object_names: Sequence[str],
    table_name: str,
    cfg: PlanningConfig,
    caps: Capabilities,
    trajectory_id: str | None,
) -> tuple[PhasePlan | None, str | None]:
    """Propose the plan for this task. Returns ``(plan, failure_reason)``.

    A plan whose phases are all robot ones needs no human, which is how a task the planner already
    handles behaves exactly as it did before.
    """
    from tandem.planning.grounding import classify_initial_state
    from tandem.planning.proposal import propose_plan

    spec = await propose_plan(image, instruction, object_names, table_name, cfg, caps)

    initially_true: frozenset[Atom] = frozenset()
    if cfg.classify_initial and spec.invented:
        initially_true = await classify_initial_state(image, spec, cfg, caps)
        if initially_true:
            _log.info(f"already true before starting: {sorted(str(a) for a in initially_true)}")

    reason = feasibility.check_robot_phases(spec, caps)
    if reason is not None:
        return None, f"phase planning failed: {reason}"
    if not spec.needs_human:
        _log.info("the robot can do this whole task on its own; no human phases were proposed")

    return (
        PhasePlan(
            cfg=cfg,
            caps=caps,
            instruction=instruction,
            trajectory_id=trajectory_id,
            spec=spec,
            initially_true=initially_true,
        ),
        None,
    )


def phase_summary(plan: PhasePlan, phase: Phase) -> dict:
    """The human phase as it goes into the session's event stream.

    Carries the phase's position in the plan, which the old integration did not: the emitter sent
    only description and expectations while the consumer read ``phase_index``/``n_phases`` off it, so
    "step N of M" in the UI was always the fallback and usually wrong. tandem holds the plan now, so
    there is one source for both.
    """
    return {
        "description": phase.description,
        "instructions": phase.instructions,
        "expected": describe_expectations(phase, descriptions_for(plan.spec.invented, plan.caps)),
        "expected_atoms": sorted(str(a) for a in phase.atoms),
        "phase_index": plan.index,
        "n_phases": len(plan.phases),
    }


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


def retry_message(missing: Sequence[str], attempts_left: int) -> str:
    """What the operator is shown when the check says the phase is not done."""
    lines = ["The workspace does not look like that step was completed.", "Still expected:"]
    lines += [f"  - {text}" for text in missing]
    if attempts_left > 0:
        lines.append("")
        lines.append("Take the arm again and finish it, or say it IS done.")
    return "\n".join(lines)
