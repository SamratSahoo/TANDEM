"""Checking a phase could ever be carried out, before anything expensive happens.

There is no outer symbolic search here. The proposer supplies the order, and each robot phase is
handed to the planner as an ordinary goal -- which is all an outer search ever contributed anyway,
since the planner replans every phase from its sub-goal regardless. What remains is the cheap, sound
check that a phase is achievable at all, and it is worth its own module because of WHEN it runs: a
phase rejected here costs one comparison, and the same phase rejected by the planner costs a
perception pass first. Worse, a planner given a goal it cannot reach may not fail at all -- cuTAMP's
search has no bound of any kind and mints fresh conf/traj symbols forever without ever yielding.

Sound but not complete, and instant: it rejects only what NO operator of the backend could ever make
true, and everything it passes may still turn out to be unreachable in this particular scene.
"""

from __future__ import annotations

from collections.abc import Sequence

from tandem.planners.base import Capabilities, to_goal_atoms
from tandem.planning.contracts import exclusive_conflicts, phase_moves
from tandem.planning.structs import Phase, TaskSpecification
from tandem.planning.symbols import Atom


def unachievable_atoms(atoms: frozenset[Atom], caps: Capabilities) -> list[Atom]:
    """Atoms in a robot phase that no operator of this backend can ever make true."""
    return sorted((a for a in atoms if a.predicate not in caps.achievable_predicates), key=str)


def check_robot_phases(spec: TaskSpecification, caps: Capabilities) -> str | None:
    """Why the plan cannot be carried out, or None if every robot phase is achievable."""
    for i, phase in enumerate(spec.phases):
        if phase.is_human:
            continue
        unachievable = unachievable_atoms(phase.atoms, caps)
        if unachievable:
            return (
                f"phase {i} ({phase.description!r}) asks the robot for "
                f"{', '.join(str(a) for a in unachievable)}, which no robot operator can achieve"
            )
    return None


def conjoinable_run(phases: Sequence[Phase], caps: Capabilities) -> int:
    """How many of ``phases``, from the front, can be planned as ONE goal.

    A phase only ever says what must be TRUE at its end, and where the backend declares
    ``initial_state_is_clean`` every robot phase is planned from the same clean state -- so nothing
    symbolic ever enforced the proposer's ordering BETWEEN two robot phases. Conjoining consecutive
    ones is the same problem stated once, and it is exactly what an ordinary planner run already does
    with a two-clause instruction: one plan, one continuous motion, and no re-perception in the
    middle for the object labels to drift across.

    The run STOPS at a phase that moves an object an earlier phase in the run already moved, when the
    backend declares ``one_pick_per_object``. cuTAMP's Pick requires and DELETES ``HasNotPickedUp``,
    so a single plan picks each object at most once: asking for ``On(toy, table)`` and
    ``On(toy, shelf)`` at once is unsatisfiable rather than merely slow. Phases like that are
    genuinely sequential -- "take the toy off the box … put the toy back in" -- and stay separate
    legs, which is what the robot-to-robot continuation carries.

    What a phase MOVES is read off the backend's ``moved_arguments`` (``contracts.phase_moves``), not
    off every object the phase names. The two agree for cuTAMP's On, whose surface is typed apart
    from its movables, and nowhere else: in a goal language where the thing placed onto is itself a
    movable -- ``Stacked(a, base)`` then ``Stacked(b, base)`` -- naming ``base`` twice would split a
    run that picks nothing twice, and a surface the plan moves would never count as picked at all.

    A backend that declares no ``moved_arguments`` has not said what its phases move, so every object
    a phase names is taken to be one it may pick. That splits more runs than it needs to, which costs
    a perception pass and never a plan; reading the missing declaration as "moves nothing" would
    conjoin exactly the goals ``one_pick_per_object`` makes unsatisfiable.

    The run also STOPS at a phase whose atoms claim a slot the run's goal already fills, by the
    backend's ``exclusive_arguments`` (``contracts.exclusive_conflicts``), whatever
    ``one_pick_per_object`` says. A goal is one final state, so ``On(toy, table)`` then
    ``On(toy, shelf)`` conjoined is ``{On(toy, table), On(toy, shelf)}``: unsatisfiable on ANY
    planner, not only one that picks each object once, even though each phase plans fine on its own.
    For TipTop the two rules split at the same place (its exclusive slot is the moved object); for a
    clean-state planner that can pick an object twice, this is the only thing that splits there.
    """
    if not caps.initial_state_is_clean:
        # Ordering between phases may be symbolically meaningful, so conjoining could silently drop
        # it. One phase per goal is always sound.
        return 1 if phases and not phases[0].is_human else 0

    count = 0
    claimed: set[str] = set()
    goal: set[Atom] = set()
    for phase in phases:
        if phase.is_human:
            break
        moved = phase_moves(phase, caps=caps) if caps.moved_arguments else frozenset(phase.objects)
        if count and caps.one_pick_per_object and moved & claimed:
            break
        if count and exclusive_conflicts(goal | phase.atoms, caps=caps):
            break
        count += 1
        claimed |= moved
        goal |= phase.atoms
    return count


def robot_leg_without_a_goal(spec: TaskSpecification, caps: Capabilities, *, conjoin: bool) -> str | None:
    """Why a robot leg of this plan would hand the planner an empty goal, or None if none would.

    A robot phase may state atoms the planner supplies for itself -- TipTop's ``HandEmpty``, which
    is achievable and in its goal language, and has no wire name (``to_goal_atoms`` drops it). A
    phase that is made of nothing else is accepted by ``check_robot_phases`` and then renders into
    an empty goal. The loop can only end the trial over that (``PhaseLoop._nothing_to_plan``, at
    ``invention``), after the robot's earlier legs have run and been recorded, and the proposer is
    never told. Refused here instead, inside the repair loop.

    Walks the legs exactly as ``PhasePlan.robot_run`` will cut them: with ``conjoin`` a HandEmpty
    phase next to a placement is part of a leg with a goal, and is fine; without it, it is a leg of
    its own, and is not.
    """
    phases = spec.phases
    i = 0
    while i < len(phases):
        if phases[i].is_human:
            i += 1
            continue
        n = max(1, conjoinable_run(phases[i:], caps)) if conjoin else 1
        run = phases[i : i + n]
        atoms = sorted(frozenset().union(*(p.atoms for p in run)), key=str)
        if not to_goal_atoms(atoms, caps):
            descriptions = "; ".join(repr(p.description) for p in run)
            which = (
                f"phase {i} ({descriptions}) asks"
                if n == 1
                else f"phases {i} to {i + n - 1} ({descriptions}), planned together, ask"
            )
            return (
                f"{which} the robot only for {', '.join(str(a) for a in atoms)}, which the "
                f"{caps.name} planner establishes for itself and cannot be given as a goal, so it "
                "would be handed nothing to plan"
            )
        i += n
    return None
