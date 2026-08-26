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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# ``{0}``, ``{1}`` ... in a natural-language template, standing in for a predicate's arguments.
_PLACEHOLDER = re.compile(r"\{(\d+)\}")


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
    """Reject a template that refers to arguments the predicate does not have."""
    used = {int(i) for i in _PLACEHOLDER.findall(template)}
    allowed = set(range(arity))
    if not used.issubset(allowed):
        raise ProposalError(
            f"{context} uses placeholders {sorted(used)}, but only {sorted(allowed)} are available."
        )


def atoms_of(values: Sequence[Atom]) -> frozenset[Atom]:
    return frozenset(values)
