"""The proposal stage: one instruction becomes an ordered list of robot and human phases.

Everything produced here is validated against what the rest of the system can actually consume before
it is returned, and a rejection is phrased for the model so the reprompt loop can fix it. The
alternative is a failure several seconds later with the arm warm and an operator watching -- or,
worse, a plan that validates and then does the wrong thing, which is how "put the toy in the box"
came to be planned before "open the box".

Every fact about the planner used here comes from ``Capabilities``: which predicates a robot phase
may use, which names are already taken, what the object types are called. Nothing is hardcoded, so
the same validator works for a backend with a different goal language.

A human phase must carry its magic operator (``Open(box)`` with preconditions, add effects and delete
effects), and the whole plan is checked as a unit before it is accepted: every robot phase must be
achievable, and when ``check_plan_effects`` is on, no phase may need something an earlier one undid.
Both checks run INSIDE the repair loop, so a plan that fails them is sent back to the model with the
reason rather than failing the trial -- a broken plan is cheapest to fix while it is still text.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tandem.planners.base import Capabilities
from tandem.planning import contracts, feasibility
from tandem.planning.cache import ProposalCache
from tandem.planning.config import PlanningConfig
from tandem.planning.prompts import PLAN_SCHEMA, plan_prompt
from tandem.planning.structs import HumanOperator, Phase, SceneTypes, TaskSpecification, VLMPredicate
from tandem.planning.symbols import Atom, Parameter, Predicate, ProposalError

_log = logging.getLogger(__name__)

# A predicate name has to survive being written into JSON, a filename and a prompt. Anything that is
# not an identifier is refused rather than sanitised, so the name the model chose is the name used
# everywhere and there is never a second spelling. An operator's name is held to the same rule: it is
# written into the same record, and read back from its signature (``HumanOperator.from_json``).
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

# One phase as the proposal wrote it: (executor, description, atom entries, instructions, the raw
# `operator` entry or None). Kept raw until the scene's types are known, because grounding an
# operator's atoms needs them and they are computed from every phase at once.
_PhaseEntry = tuple[str, str, list[tuple[str, list[str]]], str, Any]

# The three atom lists of an `operator` entry, in the order they are read.
_OPERATOR_ATOM_KEYS = ("preconditions", "add_effects", "delete_effects")


def proposal_cache(cfg: PlanningConfig) -> ProposalCache | None:
    """The proposal response cache, when the config asks for one."""
    return ProposalCache(Path(cfg.cache_path)) if cfg.cache_path else None


def _field(entry: Any, key: str, default: Any = None) -> Any:
    """Read a key from a decoded JSON object, with an error the model can act on."""
    if not isinstance(entry, dict):
        raise ProposalError(f"Expected a JSON object, got {type(entry).__name__}: {entry!r}")
    if key not in entry:
        if default is not None:
            return default
        raise ProposalError(f"The object {entry!r} is missing the required key '{key}'.")
    return entry[key]


def _atom_entries(data: Any, key: str) -> list[tuple[str, list[str]]]:
    """Parse a list of ``{"predicate": ..., "args": [...]}`` into (name, args) pairs."""
    entries = _field(data, key, [])
    if not isinstance(entries, list):
        raise ProposalError(f"'{key}' must be a list, got {type(entries).__name__}.")
    out = []
    for entry in entries:
        name = str(_field(entry, "predicate"))
        args = _field(entry, "args", [])
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise ProposalError(f"The args of {name} must be a list of strings, got {args!r}.")
        out.append((name, [str(a) for a in args]))
    return out


def _phase_entries(data: Any, caps: Capabilities) -> list[_PhaseEntry]:
    """Parse ``phases`` into (executor, description, atom entries, instructions, raw operator)."""
    entries = _field(data, "phases", [])
    if not isinstance(entries, list) or not entries:
        raise ProposalError("The plan must contain at least one phase.")
    out = []
    for entry in entries:
        executor = str(_field(entry, "executor")).strip().lower()
        if executor not in ("robot", "human"):
            raise ProposalError(f"A phase's executor must be 'robot' or 'human', got {executor!r}.")
        description = str(_field(entry, "description", "(no description)"))
        atoms = _atom_entries(entry, "atoms")
        if not atoms:
            raise ProposalError(
                f"The phase {description!r} has no atoms, so there is no way to tell when it is done."
            )
        instructions = str(entry.get("instructions") or "")
        if executor == "human" and not instructions.strip():
            raise ProposalError(
                f"The human phase {description!r} needs `instructions` telling the person what to do."
            )
        operator = entry.get("operator")
        if executor == "robot" and operator:
            # The robot's operators are the planner's own, and they are fixed. A proposal that
            # declares one is describing a step it has misjudged -- most often a human step written
            # as a robot phase, which is the failure the `operator` field exists to make visible.
            # Refused here, in words aimed at the proposer; Phase.__post_init__ is only a backstop.
            raise ProposalError(
                f"The ROBOT phase {description!r} declares an `operator`. Only a human phase does "
                f"that: the robot's operators are fixed (it can only {caps.robot_description}). If "
                "this step needs an operator of its own, it is a human phase."
            )
        out.append((executor, description, atoms, instructions, operator))
    return out


def _operator_predicate_uses(phases: Sequence[_PhaseEntry]) -> list[tuple[str, list[str]]]:
    """Every (predicate, args) pair named inside an operator, so one can type a predicate.

    An invented predicate may be named ONLY in an operator -- a precondition or a delete effect
    never has to appear in any phase's `atoms` -- and ``_build_invented`` types a predicate from its
    uses. Without counting these, such a predicate has no uses at all and is rejected as unused,
    which would make "the human closes what an earlier phase opened" unstatable.
    """
    uses: list[tuple[str, list[str]]] = []
    for _, _, _, _, raw in phases:
        if not isinstance(raw, dict):
            continue
        for key in _OPERATOR_ATOM_KEYS:
            uses.extend(_atom_entries(raw, key))
    return uses


def _scene_types(
    phases: Sequence[_PhaseEntry],
    objects: Sequence[str],
    table_name: str,
    caps: Capabilities,
) -> SceneTypes:
    """Split the perceived objects into surfaces and movables, from EVERY phase's atoms.

    Mirrors what a planner infers for itself -- an object used as the surface argument of a goal
    predicate is a surface -- but computed once across the whole plan rather than per phase, so an
    object does not change type, and with it whether the planner treats it as a static obstacle,
    between two phases of the same task.

    Only the planner's OWN goal predicates contribute: an invented predicate's parameter types are
    read off its uses, which is the step after this one, so it has nothing to say here. And only the
    phases' own ``atoms``: an operator's atoms are typed against this split, never the other way
    round, so a precondition cannot quietly turn an object the plan never places anything on into a
    surface -- and with it, into a static obstacle for the planner.
    """
    surfaces = {table_name}
    for _, _, atoms, _, _ in phases:
        for name, args in atoms:
            predicate = caps.goal_predicates.get(name)
            if predicate is None:
                continue
            # Not strict: arity is not checked until _ground_atom, and a proposal that applies a
            # predicate to the wrong number of objects must reach that check to be rejected with
            # a message the model can act on, not die here with a zip error.
            for arg, parameter in zip(args, predicate.parameters, strict=False):
                if parameter.type == caps.surface_type:
                    surfaces.add(arg)
    known = set(objects) | {table_name}
    return SceneTypes(
        surfaces=frozenset(surfaces & known),
        movables=frozenset(known - surfaces),
        surface_type=caps.surface_type,
        movable_type=caps.movable_type,
    )


def _build_invented(
    entries: Any, arg_types: dict[str, list[list[str]]], caps: Capabilities
) -> dict[str, VLMPredicate]:
    """Turn ``new_predicates`` entries into VLMPredicates, typed from how they are USED.

    The parameter types are not asked for and not guessed: they are read off the predicate's own uses,
    where each argument is a concrete object whose type this scene has already fixed. That matters
    because a planner validates goal literals per type -- an invented predicate over the cloth is a
    predicate over a SURFACE in a task that also puts something on the cloth, and over a MOVABLE in
    one that does not. Asking the model to declare the types instead invited placeholder answers
    ("container", "cover_object") that named no real object.
    """
    if entries is None:
        return {}
    if not isinstance(entries, list):
        raise ProposalError(f"'new_predicates' must be a list, got {type(entries).__name__}.")
    invented: dict[str, VLMPredicate] = {}
    for entry in entries:
        name = str(_field(entry, "name"))
        instructions = str(_field(entry, "instructions"))
        if name in caps.reserved_predicate_names:
            raise ProposalError(
                f"A predicate named '{name}' already exists. Use it directly instead of inventing one."
            )
        if not _NAME.match(name):
            raise ProposalError(
                f"'{name}' is not a valid predicate name. Use a letter followed by letters, digits "
                "or underscores."
            )
        if name in invented:
            raise ProposalError(f"You defined the predicate '{name}' more than once.")
        uses = arg_types.get(name, [])
        if not uses:
            raise ProposalError(
                f"You invented the predicate '{name}' but no phase uses it. Either use it or do "
                "not define it."
            )
        if len({tuple(u) for u in uses}) > 1:
            raise ProposalError(
                f"'{name}' is used with inconsistent arguments: {sorted({tuple(u) for u in uses})}. "
                "A predicate takes the same number of arguments, of the same kinds, everywhere."
            )
        parameters = tuple(Parameter(f"x{i}", type_name) for i, type_name in enumerate(uses[0]))
        invented[name] = VLMPredicate(Predicate(name, parameters), instructions)
    return invented


def _resolve_predicate(
    name: str, invented: dict[str, VLMPredicate], executor: str, caps: Capabilities
) -> Predicate:
    """Look up a predicate by the name a proposal calls it, for a phase with this executor."""
    if name in caps.goal_predicates:
        return caps.goal_predicates[name]
    if name in invented:
        if executor == "robot":
            # The one rule that makes a phase a HUMAN phase. The planner has no operator that can
            # make an invented predicate true, so a robot phase asking for one is unachievable --
            # and, before this check existed, produced a plan that simply never reached its goal.
            raise ProposalError(
                f"A robot phase cannot achieve '{name}': the robot only does "
                f"{caps.robot_description}, and no operator it has can make '{name}' true. Make this "
                f"a human phase, or state the phase with {', '.join(sorted(caps.goal_predicates))}."
            )
        return invented[name].predicate
    known = sorted(set(caps.goal_predicates) | (set(invented) if executor == "human" else set()))
    raise ProposalError(f"Unknown predicate '{name}'. Available predicates: {', '.join(known)}.")


def _ground_atom(
    name: str,
    args: Sequence[str],
    invented: dict[str, VLMPredicate],
    scene_types: SceneTypes,
    executor: str,
    caps: Capabilities,
) -> Atom:
    """Ground one atom, checking arity, that the objects exist, and that their types line up."""
    predicate = _resolve_predicate(name, invented, executor, caps)
    if len(args) != predicate.arity:
        raise ProposalError(
            f"{predicate.name} takes {predicate.arity} argument(s), but it is "
            f"applied to {len(args)}: {list(args)}."
        )
    # Strict: the arity check above has already passed, so a mismatch here would be a bug in it.
    for arg, parameter in zip(args, predicate.parameters, strict=True):
        actual = scene_types.type_of(arg)
        if actual != parameter.type:
            raise ProposalError(
                f"{name}({', '.join(args)}) applies the predicate to '{arg}', which is a {actual} in "
                f"this scene, but {name} expects a {parameter.type} there. A {caps.surface_type} is "
                f"something other things are put ON; a {caps.movable_type} is something the robot "
                "can pick up."
            )
    return predicate.ground(*args)


def _build_operator(
    raw: Any,
    *,
    description: str,
    phase_atoms: frozenset[Atom],
    invented: dict[str, VLMPredicate],
    scene_types: SceneTypes,
    caps: Capabilities,
) -> HumanOperator:
    """One human phase's ``operator`` entry -> a grounded :class:`HumanOperator`.

    Every atom is grounded through the same ``_ground_atom`` the phase's own atoms go through, with
    ``executor="human"``, so an invented predicate is legal here and an unknown one is rejected with
    the same message. What is checked beyond that is the operator's internal coherence, because the
    camera and the plan-time contract check both take it at its word: an add effect that is also a
    delete effect says nothing (and could not be verified -- one image cannot show an atom both true
    and false), and an operator whose effects contradict its own phase's ``atoms`` is contradicting
    the phase it belongs to.

    Every rejection here goes back to the proposer in the repair prompt, so each names the operator
    and says what to change. ``HumanOperator`` and ``Phase`` have backstops of their own; they are
    not what the model should ever see.
    """
    if not isinstance(raw, dict):
        raise ProposalError(
            f"The human phase {description!r} needs an `operator` object saying what the action is "
            "(`name`, `args`) and what it does (`preconditions`, `add_effects`, `delete_effects`)."
        )
    name = str(_field(raw, "name")).strip()
    if not name:
        raise ProposalError(f'The operator for {description!r} needs a `name`, e.g. "Push".')
    if not _NAME.match(name):
        raise ProposalError(
            f"'{name}' is not a valid operator name. Use one word: a letter followed by letters, "
            'digits or underscores, e.g. "Open" or "Unplug".'
        )
    args = _field(raw, "args", [])
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        raise ProposalError(f"The args of the operator {name} must be a list of strings, got {args!r}.")
    args = [str(a) for a in args]
    # type_of raises with the available object names, which is the message the model can act on.
    # Typed the way an invented predicate's parameters are: from the scene, never from the model.
    parameters = tuple(Parameter(f"x{i}", scene_types.type_of(a)) for i, a in enumerate(args))

    def atoms(key: str) -> frozenset[Atom]:
        return frozenset(
            _ground_atom(n, a, invented, scene_types, "human", caps) for n, a in _atom_entries(raw, key)
        )

    preconditions, add_effects, delete_effects = (atoms(k) for k in _OPERATOR_ATOM_KEYS)

    def listed(found: frozenset[Atom]) -> str:
        return ", ".join(sorted(str(a) for a in found))

    if not add_effects:
        raise ProposalError(
            f"The operator {name} has no `add_effects`, so nothing about the workspace changes when "
            "the person does it and there is no way to tell it happened."
        )
    both = add_effects & delete_effects
    if both:
        raise ProposalError(
            f"The operator {name} both adds and deletes {listed(both)}. An atom can be one or the other."
        )
    # Before the coverage check below, not after: once `atoms` is known to be covered by the add
    # effects, the rule just above has already ruled this out, and the model would never be told the
    # more useful thing -- that its operator undoes the very atom its phase promises.
    contradicted = phase_atoms & delete_effects
    if contradicted:
        raise ProposalError(
            f"The operator {name} deletes {listed(contradicted)}, which the phase {description!r} says "
            "must be TRUE afterwards. Take it out of `delete_effects`, or out of the phase's `atoms`."
        )
    uncovered = phase_atoms - add_effects
    if uncovered:
        raise ProposalError(
            f"The phase {description!r} says {listed(uncovered)} must be true afterwards, but its "
            f"operator {name} does not make that true. Every atom in a phase's `atoms` must appear "
            "in its operator's `add_effects`."
        )
    return HumanOperator(
        name=name,
        args=tuple(args),
        parameters=parameters,
        preconditions=preconditions,
        add_effects=add_effects,
        delete_effects=delete_effects,
    )


def check_coverage(data: Any, phase_count: int, unrepresented: Sequence[dict[str, str]]) -> tuple[dict, ...]:
    """Check the clause-by-clause mapping the proposer was asked for.

    Its real job is upstream of this check: enumerating the instruction's clauses and pinning each to
    a phase is the reasoning step whose absence produced a two-phase answer to a three-clause
    instruction, both phases covering clause one. Validating it here also catches a clause pointed at
    a phase that does not exist, and one marked as dropped without saying why.
    """
    entries = (data or {}).get("coverage") or []
    if not isinstance(entries, list):
        raise ProposalError(f"'coverage' must be a list, got {type(entries).__name__}.")
    dropped = {u["clause"] for u in unrepresented}
    coverage = []
    for entry in entries:
        clause = str(_field(entry, "clause"))
        try:
            index = int(_field(entry, "phase", -1))
        except (TypeError, ValueError) as exc:
            raise ProposalError(f"The phase for the clause {clause!r} must be a whole number.") from exc
        if index >= phase_count:
            raise ProposalError(
                f"The clause {clause!r} is assigned to phase {index}, but there are only {phase_count} "
                f"phase(s) (numbered 0 to {phase_count - 1}). Either add the phase that carries it "
                "out, or set its phase to -1 and list it in `unrepresented` with the reason."
            )
        if index < 0 and clause not in dropped:
            raise ProposalError(
                f"The clause {clause!r} has no phase, but it is not in `unrepresented` either. Either "
                "plan a phase for it, or say in `unrepresented` why it cannot be done."
            )
        coverage.append({"clause": clause, "phase": index})
    return tuple(coverage)


def parse_unrepresented(data: Any) -> tuple[dict[str, str], ...]:
    """Clauses of the instruction the proposer says it could not express.

    Almost always an object the instruction names that perception did not detect. Kept and reported
    rather than dropped: a run that silently plans two clauses of a three-clause instruction and then
    reports success is the exact failure this package exists to remove.
    """
    entries = (data or {}).get("unrepresented") or []
    if not isinstance(entries, list):
        raise ProposalError(f"'unrepresented' must be a list, got {type(entries).__name__}.")
    return tuple(
        {"clause": str(_field(e, "clause")), "reason": str(_field(e, "reason", "no reason given"))}
        for e in entries
    )


def parse_plan_response(
    data: Any,
    instruction: str,
    objects: Sequence[str],
    table_name: str,
    caps: Capabilities,
) -> TaskSpecification:
    """Validate a plan response into the phases the rest of the system executes.

    This is the per-phase half of the validation: every atom, every operator and the coverage are
    checked on their own. Whether the phases hang together as a PLAN is ``check_plan``, which
    ``propose_plan`` runs on what this returns.
    """
    phase_entries = _phase_entries(data, caps)
    scene_types = _scene_types(phase_entries, objects, table_name, caps)

    # An invented predicate's signature comes from its uses, where every argument is a concrete
    # object whose type this scene has already fixed.
    declared = {str(_field(e, "name")) for e in (data.get("new_predicates") or [])}
    arg_types: dict[str, list[list[str]]] = {}
    uses = [(n, a) for _, _, atoms, _, _ in phase_entries for n, a in atoms]
    # Uses inside an operator count too: a precondition or a delete effect may be the only place an
    # invented predicate is named, and an unused predicate is rejected.
    uses += _operator_predicate_uses(phase_entries)
    for name, args in uses:
        if name in declared:
            arg_types.setdefault(name, []).append([scene_types.type_of(a) for a in args])
    invented = _build_invented(data.get("new_predicates"), arg_types, caps)

    phases = []
    for executor, description, atoms, instructions, raw_operator in phase_entries:
        grounded = frozenset(_ground_atom(n, a, invented, scene_types, executor, caps) for n, a in atoms)
        operator = None
        if executor == "human":
            # Mandatory. A human phase with no operator has no stated preconditions and no stated
            # delete effects, so neither the camera nor the contract check could hold it to anything
            # beyond its atoms -- and the prompt asks for one, so a plan without it is incomplete.
            operator = _build_operator(
                raw_operator,
                description=description,
                phase_atoms=grounded,
                invented=invented,
                scene_types=scene_types,
                caps=caps,
            )
        phases.append(
            Phase(
                executor=executor,
                description=description,
                atoms=grounded,
                instructions=instructions,
                operator=operator,
            )
        )
    unrepresented = parse_unrepresented(data)
    return TaskSpecification(
        instruction=instruction,
        phases=tuple(phases),
        scene_types=scene_types,
        invented=tuple(invented.values()),
        unrepresented=unrepresented,
        coverage=check_coverage(data, len(phases), unrepresented),
    )


def check_plan(spec: TaskSpecification, cfg: PlanningConfig, caps: Capabilities) -> None:
    """Refuse a plan whose phases are each fine but which cannot work as a whole.

    Raises ``ProposalError`` phrased for the proposer, because it runs inside the repair loop: a plan
    refused here goes back to the model with the reason, which is the whole point of checking it
    before anything moves. The checks:

      * every robot phase is achievable by some robot operator (``feasibility.check_robot_phases``).
        It used to run after the proposal was accepted, where a failure could only end the trial;
        the model that wrote the phase never heard why.
      * every robot LEG, as the plan will cut them (``conjoin_robot_phases``), gives the planner a
        goal (``feasibility.robot_leg_without_a_goal``). A leg of nothing but atoms the planner
        supplies for itself (TipTop's ``HandEmpty``) is achievable and still unplannable, and the
        loop could only end the trial over it.
      * no phase asks for two atoms that claim one exclusive slot (``contracts.exclusive_conflicts``):
        ``On(toy, box)`` and ``On(toy, shelf)`` at once is a goal no planner can reach, and one that
        cuTAMP searches for until its timeout rather than refusing.
      * when ``cfg.check_plan_effects``, no phase needs something an earlier phase deleted
        (``contracts.check_plan_effects``). The starting workspace is unknown here -- nothing has
        been classified yet -- so only what the plan itself makes false is held against it.

    A malformed ``Capabilities`` declaration raises ``TandemError`` from the contract checks instead,
    on purpose: that is not the model's mistake, and reprompting it would only burn the attempts.
    """
    # What a robot phase can usefully be stated with: achievable, AND something the planner can be
    # handed. A predicate with no wire name (HandEmpty) is achievable too, but a phase stated only
    # with it is the empty-goal leg refused below, so suggesting it would steer the repair straight
    # into the next rejection.
    statable = sorted(
        p
        for p in set(caps.goal_predicates) & caps.achievable_predicates
        if caps.goal_predicate_wire_names.get(p)
    )
    unachievable = feasibility.check_robot_phases(spec, caps)
    if unachievable is not None:
        fix = f"Either state that phase with {', '.join(statable)}, or make it" if statable else "Make it"
        raise ProposalError(
            f"This plan cannot be carried out: {unachievable}. The robot can only "
            f"{caps.robot_description}. {fix} a human phase with an operator."
        )
    empty = feasibility.robot_leg_without_a_goal(spec, caps, conjoin=cfg.conjoin_robot_phases)
    if empty is not None:
        named = " or ".join([", ".join(statable[:-1]), statable[-1]] if len(statable) > 1 else statable)
        state = f"state what the robot must achieve with {named}, " if statable else ""
        raise ProposalError(
            f"This plan cannot be carried out: {empty}. Either {state}fold that phase into a "
            "neighbouring robot phase, or leave it out."
        )
    for i, phase in enumerate(spec.phases):
        # Every phase, a person's included: an operator that adds both is as impossible to verify
        # as a robot goal holding both is to plan -- one of the two verdicts fails whatever the
        # person does.
        clash = contracts.exclusive_conflicts(phase.add_effects, caps=caps)
        if clash:
            first, other = clash[0]
            thing = first.values[caps.exclusive_arguments[first.predicate]]
            raise ProposalError(
                f"This plan cannot be carried out: phase {i} ({phase.description!r}) asks for both "
                f"{first} and {other}, but {thing} can be {first.predicate} only one thing at a time, "
                "so the two can never hold together. Keep the one this phase is for, or split it into "
                "phases in the order they should happen."
            )
    if cfg.check_plan_effects:
        broken = contracts.check_plan_effects(spec, caps=caps)
        if broken:
            raise ProposalError(f"This plan does not hang together: {broken}")


async def propose_plan(
    image: Any,
    instruction: str,
    objects: Sequence[str],
    table_name: str,
    cfg: PlanningConfig,
    caps: Capabilities,
    *,
    feedback: str | None = None,
) -> TaskSpecification:
    """The instruction becomes an ordered plan of robot and human phases.

    ``feedback`` is a section appended to the prompt verbatim, the way a repair is: why an earlier
    plan for this task could not be carried out (``plan.build_plan`` writes it). A proposal with
    feedback never touches the cache. It is by definition a request for a DIFFERENT answer, and the
    same failure fed back the same way would otherwise replay the plan that just failed.
    """
    from tandem.planning.llm import query_json

    prompt = plan_prompt(instruction, list(objects), caps=caps)
    if feedback:
        prompt = f"{prompt}\n\n{feedback}"

    def parse(data: Any) -> TaskSpecification:
        spec = parse_plan_response(data, instruction, objects, table_name, caps)
        # Inside `parse`, so a plan that cannot be carried out, or whose operators contradict each
        # other, is REPROMPTED with the reason rather than failing the trial. It also means a cached
        # response the checks now refuse is asked for again rather than replayed.
        check_plan(spec, cfg, caps)
        return spec

    spec = await query_json(
        prompt,
        parse,
        model=cfg.proposal_model,
        schema=PLAN_SCHEMA,
        image=image,
        max_attempts=cfg.max_attempts,
        label="task plan",
        cache=None if feedback else proposal_cache(cfg),
    )
    _log.info(f"plan for {instruction!r}: {len(spec.phases)} phase(s)")
    for i, phase in enumerate(spec.phases):
        atoms = ", ".join(sorted(str(a) for a in phase.atoms))
        _log.info(f"phase {i} [{phase.executor}] {phase.description} -> {atoms}")
        if phase.is_human:
            _log.info(f"phase {i} instructions: {phase.instructions}")
    for predicate in spec.invented:
        _log.info(f"invented predicate {predicate.name}: {predicate.instructions}")
    for i, phase in enumerate(spec.phases):
        if phase.operator is None:
            continue
        op = phase.operator
        _log.info(
            f"phase {i} operator {op.display}: pre "
            f"{sorted(str(a) for a in op.preconditions) or 'none'} -> add "
            f"{sorted(str(a) for a in op.add_effects)} del "
            f"{sorted(str(a) for a in op.delete_effects) or 'none'}"
        )
    # A warning, deliberately NOT a rejection. Two consecutive robot phases moving the same object are
    # a SUPPORTED shape -- `feasibility.conjoinable_run` splits exactly there and the robot-to-robot
    # continuation is built on it -- so refusing the plan would make that path unreachable. But it is
    # also what a misclassified human step looks like, and the operator finding out by watching the
    # arm do the same pick twice is the outcome this line exists to prevent.
    wasted = contracts.wasted_robot_move(spec.phases, caps=caps)
    if wasted:
        _log.warning(f"this plan repeats work -- {wasted}")
    for dropped in spec.unrepresented:
        _log.warning(
            f"NOT part of the plan -- {dropped['clause']!r}: {dropped['reason']}. "
            f"Detected objects were: {', '.join(sorted(objects))}"
        )
    return spec
