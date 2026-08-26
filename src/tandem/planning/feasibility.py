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

from tandem.planners.base import Capabilities
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


def conjoinable_run(phases: Sequence[Phase], movables: frozenset[str], caps: Capabilities) -> int:
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
    """
    if not caps.initial_state_is_clean:
        # Ordering between phases may be symbolically meaningful, so conjoining could silently drop
        # it. One phase per goal is always sound.
        return 1 if phases and not phases[0].is_human else 0

    count = 0
    claimed: set[str] = set()
    for phase in phases:
        if phase.is_human:
            break
        moved = phase.objects & movables
        if count and caps.one_pick_per_object and moved & claimed:
            break
        count += 1
        claimed |= moved
    return count
