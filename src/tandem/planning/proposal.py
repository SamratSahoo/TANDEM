"""The proposal stage: one instruction becomes an ordered list of robot and human phases.

Everything produced here is validated against what the rest of the system can actually consume before
it is returned, and a rejection is phrased for the model so the reprompt loop can fix it. The
alternative is a failure several seconds later with the arm warm and an operator watching -- or,
worse, a plan that validates and then does the wrong thing, which is how "put the toy in the box"
came to be planned before "open the box".

Every fact about the planner used here comes from ``Capabilities``: which predicates a robot phase
may use, which names are already taken, what the object types are called. Nothing is hardcoded, so
the same validator works for a backend with a different goal language.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tandem.planners.base import Capabilities
from tandem.planning.cache import ProposalCache
from tandem.planning.config import PlanningConfig
from tandem.planning.prompts import PLAN_SCHEMA, plan_prompt
from tandem.planning.structs import Phase, SceneTypes, TaskSpecification, VLMPredicate
from tandem.planning.symbols import Atom, Parameter, Predicate, ProposalError

_log = logging.getLogger(__name__)

# A predicate name has to survive being written into JSON, a filename and a prompt. Anything that is
# not an identifier is refused rather than sanitised, so the name the model chose is the name used
# everywhere and there is never a second spelling.
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


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


def _phase_entries(data: Any) -> list[tuple[str, str, list[tuple[str, list[str]]], str]]:
    """Parse the ``phases`` list into (executor, description, atom entries, instructions)."""
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
        out.append((executor, description, atoms, instructions))
    return out


def _scene_types(
    phases: Sequence[tuple[str, str, list[tuple[str, list[str]]], str]],
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
    read off its uses, which is the step after this one, so it has nothing to say here.
    """
    surfaces = {table_name}
    for _, _, atoms, _ in phases:
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
    """Validate a plan response into the phases the rest of the system executes."""
    phase_entries = _phase_entries(data)
    scene_types = _scene_types(phase_entries, objects, table_name, caps)

    # An invented predicate's signature comes from its uses, where every argument is a concrete
    # object whose type this scene has already fixed.
    declared = {str(_field(e, "name")) for e in (data.get("new_predicates") or [])}
    arg_types: dict[str, list[list[str]]] = {}
    for _, _, atoms, _ in phase_entries:
        for name, args in atoms:
            if name in declared:
                arg_types.setdefault(name, []).append([scene_types.type_of(a) for a in args])
    invented = _build_invented(data.get("new_predicates"), arg_types, caps)

    phases = tuple(
        Phase(
            executor=executor,
            description=description,
            atoms=frozenset(_ground_atom(n, a, invented, scene_types, executor, caps) for n, a in atoms),
            instructions=instructions,
        )
        for executor, description, atoms, instructions in phase_entries
    )
    unrepresented = parse_unrepresented(data)
    return TaskSpecification(
        instruction=instruction,
        phases=phases,
        scene_types=scene_types,
        invented=tuple(invented.values()),
        unrepresented=unrepresented,
        coverage=check_coverage(data, len(phases), unrepresented),
    )


async def propose_plan(
    image: Any,
    instruction: str,
    objects: Sequence[str],
    table_name: str,
    cfg: PlanningConfig,
    caps: Capabilities,
) -> TaskSpecification:
    """The instruction becomes an ordered plan of robot and human phases."""
    from tandem.planning.llm import query_json

    prompt = plan_prompt(instruction, list(objects), caps=caps)

    def parse(data: Any) -> TaskSpecification:
        return parse_plan_response(data, instruction, objects, table_name, caps)

    spec = await query_json(
        prompt,
        parse,
        model=cfg.proposal_model,
        schema=PLAN_SCHEMA,
        image=image,
        max_attempts=cfg.max_attempts,
        label="task plan",
        cache=proposal_cache(cfg),
    )
    _log.info(f"plan for {instruction!r}: {len(spec.phases)} phase(s)")
    for i, phase in enumerate(spec.phases):
        atoms = ", ".join(sorted(str(a) for a in phase.atoms))
        _log.info(f"phase {i} [{phase.executor}] {phase.description} -> {atoms}")
        if phase.is_human:
            _log.info(f"phase {i} instructions: {phase.instructions}")
    for predicate in spec.invented:
        _log.info(f"invented predicate {predicate.name}: {predicate.instructions}")
    for dropped in spec.unrepresented:
        _log.warning(
            f"NOT part of the plan -- {dropped['clause']!r}: {dropped['reason']}. "
            f"Detected objects were: {', '.join(sorted(objects))}"
        )
    return spec
