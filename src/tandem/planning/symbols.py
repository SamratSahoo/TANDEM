"""tandem's own symbolic vocabulary: predicates, atoms, and a state.

Deliberately NOT cuTAMP's. The phase planner used to import ``cutamp.task_planning`` for these,
which tied the orchestration layer to one planner and dragged three things along with it:

* **Interning.** cuTAMP grounds atoms through a process-global ``_ATOM_CACHE`` keyed on
  ``(predicate name, values)`` with the parameter TYPES left out of the key, and ``Atom.__eq__``
  compares only the rendered string. The first predicate object to ground a given ``(name, values)``
  pair therefore wins for the lifetime of the process — so an invented predicate proposed in two
  tasks of one long-lived run was served from the cache carrying the FIRST task's types. The
  workaround was to suffix every invented predicate name per session (``IsFolded#2``), and that
  suffix then leaked into the audit trail and, twice, into what the model was shown. Nothing here
  interns, so none of that exists: ``Atom`` is an ordinary frozen dataclass and two atoms are equal
  when their predicate and values are equal.
* **A domain tandem does not own.** ``Pick``/``Place`` and their preconditions belong to whichever
  planner is behind the backend. What the phase planner actually needs to know about them is small
  and declarative, and lives in ``tandem.planners.base.Capabilities``.
* **A heavy import.** ``tandem`` installs on a laptop with no torch and no CUDA. This module imports
  nothing outside the standard library.

The rendered form (``On(toy, table)``) is canonical: it is what a human is shown, what goes into the
audit trail, and what equality is defined by. There is no second internal spelling.
"""

from __future__ import annotations

import re
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


class ProposalError(Exception):
    """A model's proposal could not be parsed, or failed validation.

    The message is fed back to the model on the next attempt, so it has to read as an instruction to
    the proposer rather than as a stack trace.
    """


@dataclass(frozen=True)
class Parameter:
    """One argument slot of a predicate: a name and the object type it accepts."""

    name: str
    type: str

    def __post_init__(self) -> None:
        # The prompts spell parameters `?obj`, PDDL-style. Strip it once, here, so nothing
        # downstream has to know about the two spellings.
        object.__setattr__(self, "name", self.name.lstrip("?"))


@dataclass(frozen=True)
class Predicate:
    """Something that can be true of some objects: a name and its typed parameters."""

    name: str
    parameters: tuple[Parameter, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))

    @property
    def arity(self) -> int:
        return len(self.parameters)

    @property
    def types(self) -> tuple[str, ...]:
        return tuple(p.type for p in self.parameters)

    def ground(self, *values: str) -> Atom:
        if len(values) != self.arity:
            raise ProposalError(
                f"{self.name} takes {self.arity} argument(s), but it is applied to "
                f"{len(values)}: {list(values)}."
            )
        return Atom(self.name, tuple(str(v) for v in values))


@dataclass(frozen=True)
class Atom:
    """A predicate applied to specific objects. Identity is exactly the rendered form."""

    predicate: str
    values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))

    def __str__(self) -> str:
        return f"{self.predicate}({', '.join(self.values)})"

    def rebind(self, mapping: Mapping[str, str]) -> Atom:
        """The same atom under another pass's object labels."""
        return Atom(self.predicate, tuple(mapping.get(v, v) for v in self.values))


# A symbolic state is just the set of atoms that hold.
State = frozenset


def describe(atom: Atom, descriptions: Mapping[str, str] | None = None) -> str:
    """One atom in natural language, from a ``{0}``-style template when there is one.

    Falls back to the rendered form, which is always readable if not fluent.
    """
    template = (descriptions or {}).get(atom.predicate)
    if template is None:
        return str(atom)
    return template.format(*atom.values)


def validate_template(template: str, arity: int, context: str) -> None:
    """Reject a template ``describe`` could not render with this predicate's arguments.

    ``describe`` renders with ``str.format``, which reads a great deal more than ``{0}``: a named
    field (``{box}``), a spaced one (``{ 0 }``), attribute and index access (``{0.x}``, ``{0[1]}``), a
    conversion or a format spec, an auto-numbered ``{}``, and a stray brace. Only the ``{N}`` fields
    used to be looked at here, so every one of those was accepted -- and then raised KeyError,
    ValueError or AttributeError from the first render, in the middle of a trial (``phase_summary``,
    the classifier prompt), where the proposer can no longer be asked to fix it and the trial ends
    with no outcome on its record.

    So every field is read the way ``format`` reads it (``string.Formatter().parse``, which is also
    what raises on an unbalanced brace), and anything but a bare argument index is refused here, as a
    ``ProposalError`` the repair loop hands back to the model. Parsed rather than trial-rendered with
    stand-in values on purpose: ``{0[1]}`` renders fine against a stand-in and raises against a real
    one-letter object name.
    """
    try:
        fields = [
            (name, conversion, spec)
            for _, name, spec, conversion in string.Formatter().parse(template)
            if name is not None
        ]
    except ValueError as exc:
        raise ProposalError(
            f"{context} is not a valid template ({exc}). Write only {{0}}, {{1}}, ... for the arguments, "
            "and write a literal brace as {{ or }}."
        ) from exc
    for name, conversion, spec in fields:
        # ASCII digits only: "²".isdigit() is true, and int() refuses it.
        if not (name.isascii() and name.isdigit()) or conversion or spec:
            written = f"{{{name}{f'!{conversion}' if conversion else ''}{f':{spec}' if spec else ''}}}"
            raise ProposalError(
                f"{context} uses {written}, which is not a placeholder. Write only {{0}}, {{1}}, ... "
                "standing in for the arguments, in the order they are declared, with nothing else "
                "inside the braces, and write a literal brace as {{ or }}."
            )
    used = {int(name) for name, _, _ in fields}
    allowed = set(range(arity))
    if not used.issubset(allowed):
        raise ProposalError(
            f"{context} uses placeholders {sorted(used)}, but only {sorted(allowed)} are available."
        )


def atoms_of(values: Sequence[Atom]) -> frozenset[Atom]:
    return frozenset(values)


# `Pick(?obj: movable)` or `Pick(obj: movable)`: a typed operator signature. The `?` is optional
# because the human operators the proposer invents render without it (Parameter strips it) and a
# planner's declared robot operators usually carry it; whitespace around the parts is not meaning.
_SIGNATURE = re.compile(r"^([A-Za-z_]\w*)\((.*)\)$")
_TYPED_PARAMETER = re.compile(r"^\??([A-Za-z_]\w*)\s*:\s*([A-Za-z_][\w-]*)$")


def parse_operator_signature(signature: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    """``Pick(?obj: movable)`` -> ``("Pick", (("obj", "movable"),))``, or ValueError saying what is wrong.

    The one reading of an operator signature. There used to be two that disagreed: the planner SDK's
    conformance check accepted ``Pick(?obj:movable)`` and ``Place(?obj: movable,?surface: surface)``,
    and the record then printed them with only the ``?`` removed, so ``hitl.json`` listed
    ``Pick(obj:movable)`` beside ``Open(x0: surface)`` -- one record, two spellings. Both now read
    through this, and the record re-renders what it reads (``plan.operator_signature``).
    """
    match = _SIGNATURE.match(signature.strip()) if isinstance(signature, str) else None
    if match is None:
        raise ValueError(f"{signature!r} is not a signature such as 'Pick(?obj: movable)'")
    body = match.group(2).strip()
    parameters: list[tuple[str, str]] = []
    for part in (p.strip() for p in body.split(",")) if body else ():
        parameter = _TYPED_PARAMETER.match(part)
        if parameter is None:
            raise ValueError(f"{signature!r}: {part!r} is not a typed parameter such as '?obj: movable'")
        parameters.append((parameter.group(1), parameter.group(2)))
    return match.group(1), tuple(parameters)
