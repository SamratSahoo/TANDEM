"""What a proposal turns an instruction into: an ordered list of phases, and the predicates behind it.

A task is an ORDERED LIST OF PHASES, each one carried out either by the planner or by a person.

That is deliberately not the design it grew out of, which planned over the robot's own operators
together with invented "magic" operators and let the ordering fall out of their preconditions. It had
to change because that ordering only ever flows one way. A magic operator can require robot work
BEFORE it — fold the cloth once the toy is on it — but nothing can make robot work wait on a human:
a planner's Pick and Place have fixed preconditions that never mention an invented predicate. "Open
the box, then put the toy in it" was therefore inexpressible, and the planner cheerfully put the toy
in the closed box instead.

Two limits go with it. A goal that is a set of atoms describes a FINAL state, so an intermediate one
— the toy resting on the table while the box is opened — cannot be said at all, and `On(toy, table)`
would contradict `On(toy, box)`. And where a planner can pick each object at most once per plan,
"toy off the box … toy into the box" could not appear in a single plan. Phases dissolve both: each
robot phase is planned from its own perception pass, so it starts from a clean state.

The magic operator did not go away; it changed jobs. A human phase still carries one
(:class:`HumanOperator`) -- ``Open(box)``, what must hold first, what it makes true and what it
undoes -- but nothing searches over it and nothing derives the ORDER from it. The order is the phase
list. The operator is the phase's contract: the camera checks it around the hand-off, and the
plan-time consistency check reads it to catch a phase that undoes what a later one needs.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tandem.planning.symbols import Atom, Parameter, Predicate, ProposalError, validate_template

# ``Name(a, b)``: how an atom, an operator instance and an operator signature are all written in the
# audit record. Used only to read that record back (``HumanOperator.from_json``).
_CALL = re.compile(r"^\s*([^\s(),]+)\((.*)\)\s*$")


@dataclass(frozen=True)
class SceneTypes:
    """Which object type each perceived object carries for the whole of one task.

    A planner normally infers this from the goal it is handed — anything appearing as the surface
    argument of an ``on(...)`` is a surface, everything else a movable. Under phase planning the goal
    is different for every phase, so inferring per phase would make a box a surface (and therefore a
    static obstacle) in the phase that puts a toy into it and a movable in the phase that does not,
    changing the world geometry mid-task. The split is computed ONCE, from every phase's atoms
    together, and held here.
    """

    surfaces: frozenset[str]
    movables: frozenset[str]
    surface_type: str = "surface"
    movable_type: str = "movable"

    def type_of(self, name: str) -> str:
        """The object type of ``name``, or raise if it was not perceived in this scene."""
        if name in self.surfaces:
            return self.surface_type
        if name in self.movables:
            return self.movable_type
        raise ProposalError(
            f"'{name}' is not an object in this scene. Available objects: "
            f"{', '.join(sorted(self.surfaces | self.movables))}."
        )

    @property
    def all_names(self) -> frozenset[str]:
        return self.surfaces | self.movables

    def rebind(self, mapping: Mapping[str, str]) -> SceneTypes:
        """The same split under another pass's object labels."""
        return SceneTypes(
            surfaces=frozenset(mapping.get(n, n) for n in self.surfaces),
            movables=frozenset(mapping.get(n, n) for n in self.movables),
            surface_type=self.surface_type,
            movable_type=self.movable_type,
        )


@dataclass(frozen=True)
class VLMPredicate:
    """A predicate whose truth value a vision model reads off an image.

    ``instructions`` is what must be VISIBLE for the atom to hold, with ``{0}``, ``{1}``, … standing
    in for the arguments — e.g. "the cloth {0} has been folded over on itself". That text is the
    predicate's entire definition: it is the classifier at verification time AND the description the
    person is shown when they are asked to bring it about.
    """

    predicate: Predicate
    instructions: str

    def __post_init__(self) -> None:
        validate_template(self.instructions, self.predicate.arity, f"Instructions for {self.name}")

    @property
    def name(self) -> str:
        return self.predicate.name

    def describe(self, values: Sequence[str]) -> str:
        return self.instructions.format(*values)


def _split_call(text: Any, what: str) -> tuple[str, list[str]]:
    """``Name(a, b)`` -> ``("Name", ["a", "b"])``, or raise naming what was being read."""
    match = _CALL.match(text) if isinstance(text, str) else None
    if match is None:
        raise ValueError(f"Cannot read {what} {text!r}: expected the form Name(arg, ...).")
    inner = match.group(2).strip()
    parts = [part.strip() for part in inner.split(",")] if inner else []
    if not all(parts):
        raise ValueError(f"Cannot read {what} {text!r}: it has an empty argument.")
    return match.group(1), parts


def _read_atom(text: Any) -> Atom:
    name, values = _split_call(text, "the atom")
    return Atom(name, tuple(values))


def _read_parameter(text: str, signature: str) -> Parameter:
    name, sep, type_name = text.partition(":")
    if not sep or not name.strip() or not type_name.strip():
        raise ValueError(f"Cannot read the parameter {text!r} of {signature!r}: expected 'name: type'.")
    return Parameter(name.strip(), type_name.strip())


@dataclass(frozen=True)
class HumanOperator:
    """The explicit operator behind ONE human phase -- ``Open(box)`` and what it does.

    A human phase used to carry only ``atoms``: what should be true when it is over. That is an
    effect list with no contract around it -- nothing said what the world had to look like for the
    step to be possible, and nothing said what the step UNDOES. This makes both explicit, in the
    shape a TAMP planner states its own operators in:

        Open(box)
            preconditions:   HandEmpty(), On(box, table)
            add effects:     IsOpen(box)
            delete effects:  IsClosed(box)

    Stored GROUNDED, not lifted. A planner's operators are lifted because its search instantiates
    them; this one is never searched over -- the proposal stage already decided that this phase
    happens here, to these objects -- so the useful form is the instance, which is what a camera can
    be asked about. ``parameters`` is kept only so the record can print a typed signature.

    Nothing here reaches the planner behind the backend. A human operator's effects are not
    achievable by any robot operator (that is what makes the phase a human's), so offering it to a
    skeleton search would only produce plans the robot cannot execute. It is used for exactly two
    things: the pre/post checks around the hand-off, and the plan-time consistency check.
    """

    name: str
    args: tuple[str, ...]
    parameters: tuple[Parameter, ...]
    preconditions: frozenset[Atom] = frozenset()
    add_effects: frozenset[Atom] = frozenset()
    delete_effects: frozenset[Atom] = frozenset()

    def __post_init__(self) -> None:
        # Normalised the way Predicate and Atom normalise theirs, so an operator built from lists
        # compares, hashes and rebinds like one built from the proposal parser.
        object.__setattr__(self, "args", tuple(str(a) for a in self.args))
        object.__setattr__(self, "parameters", tuple(self.parameters))
        for key in ("preconditions", "add_effects", "delete_effects"):
            object.__setattr__(self, key, frozenset(getattr(self, key)))
        # A ProposalError rather than an assert: an assert vanishes under `python -O`, and the
        # mismatch would then surface much later as a signature that silently drops an argument.
        # Phrased for the proposer like every other ProposalError, in case it ever reaches the
        # repair prompt.
        if len(self.args) != len(self.parameters):
            raise ProposalError(
                f"The operator {self.name} is applied to {len(self.args)} object(s) "
                f"({', '.join(self.args) or 'none'}), but its signature has {len(self.parameters)} "
                "parameter(s). An operator takes exactly one object per parameter."
            )

    @property
    def display(self) -> str:
        """``Open(white_box)`` -- the instance, as the proposer wrote it."""
        return f"{self.name}({', '.join(self.args)})"

    @property
    def signature(self) -> str:
        """``Open(x0: surface)`` -- the lifted shape, for the audit record."""
        return f"{self.name}({', '.join(f'{p.name}: {p.type}' for p in self.parameters)})"

    def apply(self, state: frozenset[Atom]) -> frozenset[Atom]:
        """``state`` after this operator ran: delete effects removed, add effects added.

        Add effects win a tie. An atom in both lists is rejected at parse time, so this only decides
        the order for a state that already held something the operator both deletes and adds.
        """
        return frozenset((set(state) - self.delete_effects) | self.add_effects)

    def unmet(self, state: frozenset[Atom]) -> frozenset[Atom]:
        """Preconditions this state does not satisfy."""
        return self.preconditions - frozenset(state)

    def rebind(self, mapping: Mapping[str, str]) -> HumanOperator:
        """The same operator under another pass's object labels.

        The parameters are left alone: a rebind renames objects, it never changes their type (see
        ``SceneTypes.rebind``), so the signature is the same before and after.
        """
        return HumanOperator(
            name=self.name,
            args=tuple(mapping.get(a, a) for a in self.args),
            parameters=self.parameters,
            preconditions=frozenset(a.rebind(mapping) for a in self.preconditions),
            add_effects=frozenset(a.rebind(mapping) for a in self.add_effects),
            delete_effects=frozenset(a.rebind(mapping) for a in self.delete_effects),
        )

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "args": list(self.args),
            "signature": self.signature,
            "instance": self.display,
            "preconditions": sorted(str(a) for a in self.preconditions),
            "add_effects": sorted(str(a) for a in self.add_effects),
            "delete_effects": sorted(str(a) for a in self.delete_effects),
        }

    @classmethod
    def from_json(cls, record: Mapping[str, Any]) -> HumanOperator:
        """The operator ``to_json`` wrote, read back from the audit record.

        hitl.json is the only lasting statement of what a person was asked to bring about, so it has
        to be readable back into the thing that wrote it -- to re-check a recorded episode, or to
        audit one after the fact -- rather than re-derived from prose. Loud on anything it cannot
        read: an operator half-recovered from a damaged record would be verified against the wrong
        contract without anyone noticing. Extra keys (the ``phase`` index ``human_operators``
        entries carry) are ignored.
        """
        signature = str(record["signature"])
        name, parameter_texts = _split_call(signature, "the operator signature")
        if name != record["name"]:
            raise ValueError(
                f"The operator record names {record['name']!r}, but its signature is {signature!r}."
            )
        args = record["args"]
        if not isinstance(args, (list, tuple)):
            # A bare string would otherwise become one argument per character.
            raise ValueError(f"The args of the operator record {signature!r} must be a list, got {args!r}.")
        operator = cls(
            name=name,
            args=tuple(args),
            parameters=tuple(_read_parameter(p, signature) for p in parameter_texts),
            preconditions=frozenset(_read_atom(a) for a in record.get("preconditions", ())),
            add_effects=frozenset(_read_atom(a) for a in record.get("add_effects", ())),
            delete_effects=frozenset(_read_atom(a) for a in record.get("delete_effects", ())),
        )
        if "instance" in record and record["instance"] != operator.display:
            raise ValueError(
                f"The operator record says it is {record['instance']!r}, but its name and args make "
                f"{operator.display!r}."
            )
        return operator


@dataclass(frozen=True)
class Phase:
    """One step of the task, carried out by one agent.

    A ``robot`` phase is a sub-goal handed to the planner: ``atoms`` are over the planner's own goal
    predicates and are rendered into its goal language unchanged. A ``human`` phase is one thing a
    person does; ``atoms`` are what should be true afterwards, and are what the vision model is asked
    about to check they did it.

    A human phase may also carry its ``operator``: the explicit ``Open(box)`` with preconditions and
    add/delete effects (:class:`HumanOperator`). ``atoms`` stays the phase's own statement of what
    must hold afterwards, and the parser holds it to a subset of the operator's add effects, so every
    reader of ``atoms`` keeps working unchanged. The operator is what the pre/post checks and the
    plan-time consistency check are stated over, through ``preconditions``, ``add_effects`` and
    ``delete_effects``. Without one -- every robot phase, a proposal from before operators, a robot
    phase handed to a person -- those fall back to exactly the old implicit contract: nothing
    required, ``atoms`` made true, nothing undone.
    """

    executor: str  # "robot" | "human"
    description: str
    atoms: frozenset[Atom]
    instructions: str = ""
    operator: HumanOperator | None = None

    def __post_init__(self) -> None:
        if self.executor not in ("robot", "human"):
            raise ProposalError(f"A phase's executor must be 'robot' or 'human', got {self.executor!r}.")
        if self.operator is not None and not self.is_human:
            # The robot's operators are the planner's own and fixed. A robot phase that declares one
            # is almost always a human step misjudged as robot work -- the misjudgement the operator
            # exists to make visible, so it is refused rather than quietly dropped.
            raise ProposalError(
                f"The robot phase {self.description!r} declares the operator {self.operator.display}. "
                "Only a human phase does that: the robot's operators are the planner's own and fixed. "
                "If this step needs an operator of its own, it is a human phase."
            )

    @property
    def is_human(self) -> bool:
        return self.executor == "human"

    @property
    def preconditions(self) -> frozenset[Atom]:
        """What must hold before this phase can be carried out. Empty without an operator.

        Deliberately empty for a robot phase too, rather than something inferred for it: what the
        planner needs first is the planner's business, and it re-perceives before every robot leg.
        """
        return self.operator.preconditions if self.operator is not None else frozenset()

    @property
    def add_effects(self) -> frozenset[Atom]:
        """What this phase makes true. The phase's own atoms when it has no operator."""
        return self.operator.add_effects if self.operator is not None else self.atoms

    @property
    def delete_effects(self) -> frozenset[Atom]:
        """What this phase makes false. Empty without an operator -- the old implicit behaviour."""
        return self.operator.delete_effects if self.operator is not None else frozenset()

    @property
    def objects(self) -> set[str]:
        """Every object this phase names, for the label-drift check."""
        return {value for atom in self.atoms for value in atom.values}

    def rebind(self, mapping: Mapping[str, str]) -> Phase:
        """The same phase with its objects renamed to another pass's labels.

        The operator is rebound with it. Skipping it would leave the pre/post checks stated over
        names this perception pass no longer produces -- the very drift a rebind exists to absorb.
        """
        return Phase(
            self.executor,
            self.description,
            frozenset(a.rebind(mapping) for a in self.atoms),
            self.instructions,
            self.operator.rebind(mapping) if self.operator is not None else None,
        )

    def as_human(self, instructions: str) -> Phase:
        """The same phase, handed to a person instead.

        Used when a robot phase turns out not to be plannable and the run is configured to offer it
        as teleop rather than give up on the task. The atoms are unchanged, so what is checked
        afterwards is exactly what the planner was going to be asked for.

        The result carries NO operator, on purpose. The model never wrote one for this step, and one
        made up here would be a contract nobody proposed: its preconditions unchecked by the
        plan-time consistency check that ran before the hand-off, its delete effects a guess. With
        none, the phase falls back to the atoms-only contract.
        """
        return Phase("human", self.description, self.atoms, instructions or self.instructions, None)

    def summary(self) -> dict:
        record = {
            "executor": self.executor,
            "description": self.description,
            "atoms": sorted(str(a) for a in self.atoms),
        }
        if self.is_human:
            record["instructions"] = self.instructions
            # The explicit operator behind this phase: what had to be true first, what it makes true
            # and what it undoes. Absent on a phase that has none (see as_human).
            if self.operator is not None:
                record["operator"] = self.operator.to_json()
        return record


@dataclass(frozen=True)
class TaskSpecification:
    """What the proposal stage made of the instruction: an ordered plan, and the predicates behind it.

    A specification whose phases are all ``robot`` needs no person at all, which is how a task the
    planner already handles degrades to exactly its previous behaviour.
    """

    instruction: str
    phases: tuple[Phase, ...]
    scene_types: SceneTypes
    invented: tuple[VLMPredicate, ...] = ()
    # Clauses of the instruction that made it into no phase, with the reason. Non-empty means the run
    # is deliberately doing LESS than it was asked to.
    unrepresented: tuple[dict[str, str], ...] = ()
    # The proposer's own clause-by-clause account of which phase carries out which part of the
    # instruction. Recorded so a plan that quietly covers less than it was asked can be read off the
    # audit trail rather than inferred from the phases.
    coverage: tuple[dict, ...] = ()

    @property
    def needs_human(self) -> bool:
        return any(p.is_human for p in self.phases)

    @property
    def operators(self) -> tuple[HumanOperator, ...]:
        """The explicit operator behind each human phase that has one, in phase order."""
        return tuple(p.operator for p in self.phases if p.operator is not None)

    @property
    def descriptions(self) -> dict[str, str]:
        """Predicate name -> natural-language template, for everything that has one."""
        return {p.name: p.instructions for p in self.invented}

    def replace_phase(self, index: int, phase: Phase) -> TaskSpecification:
        phases = list(self.phases)
        phases[index] = phase
        return TaskSpecification(
            instruction=self.instruction,
            phases=tuple(phases),
            scene_types=self.scene_types,
            invented=self.invented,
            unrepresented=self.unrepresented,
            coverage=self.coverage,
        )

    def rebind(self, mapping: Mapping[str, str]) -> TaskSpecification:
        """The same plan under another pass's object labels (see ``drift.match_drifted_names``)."""
        if not mapping:
            return self
        return TaskSpecification(
            instruction=self.instruction,
            phases=tuple(p.rebind(mapping) for p in self.phases),
            scene_types=self.scene_types.rebind(mapping),
            invented=self.invented,
            unrepresented=self.unrepresented,
            coverage=self.coverage,
        )

    def to_json(self) -> dict:
        """The audit record written beside each episode (see phases.json)."""
        return {
            "instruction": self.instruction,
            "invented_predicates": [
                {
                    "name": p.name,
                    "types": list(p.predicate.types),
                    "instructions": p.instructions,
                }
                for p in sorted(self.invented, key=lambda p: p.name)
            ],
            # One per human phase that has one, in phase order: the operator that phase IS, with its
            # preconditions and add/delete effects. `phase` is its index in `phases`. Empty on an
            # all-robot plan, and on a plan proposed without operators at all.
            "human_operators": [
                {"phase": i, **phase.operator.to_json()}
                for i, phase in enumerate(self.phases)
                if phase.operator is not None
            ],
            "surfaces": sorted(self.scene_types.surfaces),
            "movables": sorted(self.scene_types.movables),
            # Empty on a run that planned the whole instruction. Anything here is a clause the run
            # knowingly left out — read it before trusting a "success" label.
            "unrepresented": [dict(u) for u in self.unrepresented],
            "coverage": [dict(c) for c in self.coverage],
        }
