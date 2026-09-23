"""The plan-time contract check: do the phases' operators hang together, before the arm moves?

A human phase carries an explicit operator (:class:`~tandem.planning.structs.HumanOperator`): what
must hold first, what it makes true, what it undoes. The proposer writes those contracts one phase at
a time, and nothing about writing one of them looks at the others -- which is how "the person closes
the box" came to be planned before "the toy goes into the box", every phase fine on its own and the
plan broken as a whole. This module walks the phase list symbolically and asks whether any phase needs
something an EARLIER phase made false.

It runs inside the proposal's repair loop, so a plan it rejects goes back to the model with the reason
instead of reaching the robot. That is why the messages here are written to the proposer, and why the
check has to be cheap: no image, no model call, no planner -- set arithmetic over atoms the plan
already holds. Everything here is a pure function.

Sound, not complete, on purpose. The only state tracked is what the PLAN establishes and undoes. What
the workspace looked like before phase 0 is unknown unless ``classify_initial`` measured it, and even
then only for the invented predicates. So ``check_plan_effects`` concludes a plan is broken only where
the plan itself is the reason, and passes everything it cannot prove wrong.

Nothing here names a predicate. What a robot phase undoes without declaring it (``On(toy, shelf)``
ending ``On(toy, table)``) and which object a robot phase moves are both facts about the planner's goal
language, so they are read from ``Capabilities.exclusive_arguments`` and
``Capabilities.moved_arguments``. A backend that declares neither gets a weaker check, never a wrong
one.

Ported from LJ tiptop ``cf75a68`` ``tiptop/hitl/planning.py``, with the two places it hard-coded
cuTAMP's ``On`` and ``Holding`` turned into those declarations.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tandem.core.errors import TandemError
from tandem.planning.structs import Phase, TaskSpecification
from tandem.planning.symbols import Atom

if TYPE_CHECKING:
    from tandem.planners.base import Capabilities


@dataclass(frozen=True)
class PhaseTrace:
    """What the plan believes about the world on either side of one phase.

    ``before`` is the state the phase is entered in and ``after`` the state it leaves; ``unmet`` are
    the phase's preconditions ``before`` does not contain. Only the atoms the plan can actually reason
    about are in these sets -- see ``simulate_phases`` for what that means and does not mean.
    """

    index: int
    phase: Phase
    before: frozenset[Atom]
    after: frozenset[Atom]
    unmet: frozenset[Atom]


def _positions(caps: Capabilities, field: str) -> Mapping[str, int]:
    """``caps.<field>``, checked against the goal language it is a statement about.

    Both ``exclusive_arguments`` and ``moved_arguments`` map a goal predicate to the argument that
    names the object. A key that is not a goal predicate (``"on"`` for ``On``), or a position past its
    arity, is a declaration that can never match an atom: the check would quietly get weaker and
    nothing would say so. So it is refused, the way an unknown prompt slot is.
    """
    positions: Mapping[str, int] = getattr(caps, field)
    for name, position in positions.items():
        predicate = caps.goal_predicates.get(name)
        if predicate is None:
            # Case first: the likeliest slip is the WIRE spelling (TipTop's "on"), which difflib's
            # case-sensitive ratio does not rate as close to "On" at all.
            by_case = {n.lower(): n for n in caps.goal_predicates}
            suggestion = (
                [by_case[name.lower()]]
                if name.lower() in by_case
                else difflib.get_close_matches(name, list(caps.goal_predicates), n=1, cutoff=0.6)
            )
            hint = (
                f"Did you mean {suggestion[0]!r}?"
                if suggestion
                else f"Its goal predicates are: {', '.join(caps.goal_predicates) or '(none)'}."
            )
            raise TandemError(
                f"The {caps.name!r} planner declares {field} for {name!r}, which is not one of its "
                "goal predicates, so it could never match an atom.",
                hint=hint,
            )
        if not 0 <= position < predicate.arity:
            raise TandemError(
                f"The {caps.name!r} planner declares {field}[{name!r}] = {position}, but {name} takes "
                f"{predicate.arity} argument(s).",
                hint=(
                    f"Use a position from 0 to {predicate.arity - 1}."
                    if predicate.arity
                    else f"{name} has no argument to name an object with; leave it out of {field}."
                ),
            )
    return positions


def displaced_by(
    atoms: Iterable[Atom], state: Iterable[Atom], exclusive_arguments: Mapping[str, int]
) -> frozenset[Atom]:
    """The atoms in ``state`` that asserting ``atoms`` makes false, because each is exclusive.

    ``exclusive_arguments`` maps a predicate to the argument that can hold in only one atom at a time.
    With ``{"On": 0}``, a thing rests on one surface at a time, so ``On(toy, shelf)`` replaces
    ``On(toy, table)`` outright. Only atoms of the SAME predicate are retracted, and an atom asserted
    again is never displaced by itself.

    This is the only delete effect a phase gets for free. A robot phase has no operator and declares no
    delete effects, and its placements are the one thing whose consequence is not in doubt. A predicate
    absent from the map displaces nothing.
    """
    asserted = frozenset(atoms)
    # Keyed by predicate as well as object: On(toy, shelf) says nothing about any other predicate the
    # toy appears in, only about where else it is On.
    claimed = {
        (atom.predicate, atom.values[exclusive_arguments[atom.predicate]])
        for atom in asserted
        if atom.predicate in exclusive_arguments
    }
    return frozenset(
        atom
        for atom in state
        if atom.predicate in exclusive_arguments
        and (atom.predicate, atom.values[exclusive_arguments[atom.predicate]]) in claimed
        and atom not in asserted
    )


def simulate_phases(
    spec: TaskSpecification, initially_true: Iterable[Atom] = frozenset(), *, caps: Capabilities
) -> list[PhaseTrace]:
    """Walk the phase list symbolically and report what holds on either side of each phase.

    Each phase leaves ``after = (before - displaced - delete effects) | add effects``, and its
    ``unmet`` preconditions are ``preconditions - before``.

    The state tracked here is the WORLD's, not the planner's. A planner that plans each robot leg from
    a fresh perception pass (``Capabilities.initial_state_is_clean``) carries nothing from one leg to
    the next. That is the right model for planning one leg and the wrong one for asking whether a plan
    hangs together across legs, which is what this is for.

    It starts from ``initially_true``, which is EMPTY unless ``classify_initial`` ran, and that is why
    this is a sound-not-complete check and not a simulator. An atom absent from a state here means
    "nothing in the plan established it", never "it is false": the workspace may well have started
    that way. ``check_plan_effects`` is what turns this into a verdict, and it only ever concludes
    something is wrong when the plan itself is the reason.
    """
    exclusive = _positions(caps, "exclusive_arguments")
    state = frozenset(initially_true)
    traces: list[PhaseTrace] = []
    for i, phase in enumerate(spec.phases):
        unmet = phase.preconditions - state
        displaced = displaced_by(phase.add_effects, state, exclusive)
        after = (state - displaced - phase.delete_effects) | phase.add_effects
        traces.append(PhaseTrace(index=i, phase=phase, before=state, after=after, unmet=unmet))
        state = after
    return traces


def check_plan_effects(
    spec: TaskSpecification,
    initially_true: Iterable[Atom] = frozenset(),
    *,
    initial_state_known: bool = False,
    caps: Capabilities,
) -> str | None:
    """Why the declared operators do not hang together, or None when they do.

    Two things are provable from the plan alone and nothing else is:

      * a precondition an earlier phase DELETED and no phase put back. The plan itself made it false,
        so no assumption about the starting workspace can rescue it. This is the check that catches
        "the person closes the box" placed before "the toy goes into the box". A displacement counts
        as a deletion: moving the toy into the box ends ``On(toy, table)`` as surely as a declared
        delete effect would.
      * a precondition over an INVENTED predicate that no phase establishes -- only when the starting
        state was actually measured (``classify_initial``, hence ``initial_state_known``). Without that
        measurement the workspace may simply have started that way, and rejecting the plan would be
        guessing. The planner's own predicates are never held to this, even when the state is known:
        only invented atoms are measured, so ``initially_true`` says nothing about them.

    Everything else is left alone, deliberately. A precondition over the planner's own predicates that
    no phase establishes is the commonest shape there is -- the toy was already on the box when the
    run started, the gripper was already empty -- and a check that refused it would reject almost
    every plan that is in fact fine.

    Only the first violation is reported, in phase order: the reason goes back to the proposer, and
    one concrete thing to fix is what gets a plan repaired.
    """
    initially_true = frozenset(initially_true)
    exclusive = _positions(caps, "exclusive_arguments")
    established: set[Atom] = set()
    deleted: set[Atom] = set()
    invented_names = {p.name for p in spec.invented}
    for trace in simulate_phases(spec, initially_true, caps=caps):
        for atom in sorted(trace.unmet, key=str):
            provably_false = atom in deleted and atom not in established
            never_established = (
                initial_state_known
                and atom.predicate in invented_names
                and atom not in established
                and atom not in initially_true
            )
            if not (provably_false or never_established):
                continue
            operator = trace.phase.operator
            where = f" ({operator.display})" if operator is not None else ""
            why = (
                "an earlier phase deletes it and no phase puts it back"
                if provably_false
                else "no phase makes it true and it is not true in the workspace to begin with"
            )
            return (
                f"phase {trace.index} ({trace.phase.description!r}){where} requires "
                f"{atom}, but {why}. Either reorder the phases so it still holds, or "
                f"drop it from that operator's preconditions."
            )
        # A displacement deletes whatever the object was resting on before, so it counts as something
        # the plan made false just as much as a declared delete effect does. Computed against what
        # was ESTABLISHED, not against the trace's state: the question is whether an earlier phase is
        # on the hook for the atom, and only an atom some phase established can be.
        displaced = displaced_by(trace.phase.add_effects, established, exclusive)
        deleted |= trace.phase.delete_effects | displaced
        established |= trace.phase.add_effects
        established -= trace.phase.delete_effects | displaced
    return None


def phase_moves(phase: Phase, *, caps: Capabilities) -> frozenset[str]:
    """The objects a phase MOVES: the thing placed or held, never the surface it lands on.

    Read off what the phase makes true, through ``Capabilities.moved_arguments``: with
    ``{"On": 0, "Holding": 0}`` the toy in ``On(toy, box)`` moves and the box does not. That is the
    distinction from ``Phase.objects``, which names the surface too, and it is the whole correctness of
    ``wasted_robot_move``: "put toy_a on the table" and "put toy_b on the table" name ``table`` in
    common while moving nothing in common.
    """
    moved = _positions(caps, "moved_arguments")
    return frozenset(
        atom.values[moved[atom.predicate]] for atom in phase.add_effects if atom.predicate in moved
    )


def wasted_robot_move(phases: Sequence[Phase], *, caps: Capabilities) -> str | None:
    """Why two consecutive ROBOT phases move the same object twice, or None if none do.

    Moving an object is not a step that composes: a later placement replaces the earlier one outright,
    so if nothing happens in between, the first is motion that achieves nothing. "Put the toy on the
    cloth, then put the toy on the board" leaves the toy on the board -- exactly where one phase would
    have put it.

    A human phase between them makes it legitimate, and is why this looks only at CONSECUTIVE robot
    phases. The canonical plan is "take the toy off the box / open the box / put the toy back in": the
    toy is placed twice, but the world changed in between, and the first placement is what made the
    opening possible.

    So the pattern is diagnostic: it is what a plan looks like when a step the robot CANNOT do was
    written as robot work anyway -- "solve the puzzle" as ``On(pink_toy, puzzle_board)``, which put
    the toy on the board, called the puzzle solved, and undid the phase before it to do so.

    Reported for the LOG, not as a rejection. The shape is supported: consecutive robot phases that
    move the same object become separate legs (``feasibility.conjoinable_run`` splits exactly here for
    a planner that picks each object once per plan), so refusing such a plan would make the
    robot-to-robot continuation unreachable. The prompt is where this is actually prevented; this is
    how a recurrence gets noticed without the operator having to watch the arm do the same pick twice.
    A backend that declares no ``moved_arguments`` gets no report.
    """
    last_robot_move: dict[str, int] = {}
    for i, phase in enumerate(phases):
        if phase.is_human:
            # The world changed; every earlier move is now something a later phase may redo.
            last_robot_move.clear()
            continue
        # phase_moves, NOT Phase.objects: two phases that put DIFFERENT toys on the same table share
        # that table and move nothing in common. Asking the wrong question here would flag the
        # commonest plan there is.
        for name in sorted(phase_moves(phase, caps=caps)):
            earlier = last_robot_move.get(name)
            if earlier is not None:
                return (
                    f"phases {earlier} ({phases[earlier].description!r}) and {i} "
                    f"({phase.description!r}) are both robot phases and both move {name}, with no "
                    f"human phase in between. The second move replaces the first, so phase "
                    f"{earlier} is wasted motion. Either a human phase belongs between them, or the "
                    f"step in phase {i} is not really robot work (the robot can only "
                    f"{caps.robot_description}) and should be a HUMAN phase."
                )
            last_robot_move[name] = i
    return None


def expected_before(
    spec: TaskSpecification,
    index: int,
    initially_true: Iterable[Atom] = frozenset(),
    *,
    caps: Capabilities,
) -> frozenset[Atom]:
    """What EARLIER phases should have left true, entering phase ``index``.

    Every atom some phase before ``index`` was responsible for establishing and no later one undid,
    whether by a declared delete effect or by a displacement. ``index == len(spec.phases)`` asks about
    the state the whole plan leaves behind.

    This is the robot-side precondition set (``check_tamp_preconditions``). A robot phase declares no
    preconditions of its own -- it has no operator, and the planner replans it from a fresh perception
    pass -- so what is worth checking before one is the plan's own beliefs.

    Restricted to exactly that. An atom nothing established may or may not hold, and the plan does not
    depend on which, so putting it to a camera would only manufacture failures. That includes anything
    in ``initially_true``: no earlier phase is on the hook for it. The atoms that ARE listed are the
    ones a human phase verified as done may nonetheless have left undone, which is the drift this check
    exists to catch.
    """
    if not 0 <= index <= len(spec.phases):
        raise IndexError(
            f"There is no phase {index} to enter in a plan of {len(spec.phases)} phase(s); "
            f"expected 0 to {len(spec.phases)}."
        )
    traces = simulate_phases(spec, initially_true, caps=caps)
    established: set[Atom] = set()
    for trace in traces[:index]:
        established |= trace.phase.add_effects
        established -= trace.phase.delete_effects
    if index < len(traces):
        before = traces[index].before
    else:
        before = traces[-1].after if traces else frozenset(initially_true)
    # Intersecting with the trace is what drops a displaced atom: `established` above only knows
    # declared delete effects, and the trace knows the toy is no longer on the table.
    return before & frozenset(established)
