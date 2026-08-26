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
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tandem.planning.symbols import Atom, Predicate, ProposalError, validate_template


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


@dataclass(frozen=True)
class Phase:
    """One step of the task, carried out by one agent.

    A ``robot`` phase is a sub-goal handed to the planner: ``atoms`` are over the planner's own goal
    predicates and are rendered into its goal language unchanged. A ``human`` phase is one thing a
    person does; ``atoms`` are what should be true afterwards, and are what the vision model is asked
    about to check they did it.
    """

    executor: str  # "robot" | "human"
    description: str
    atoms: frozenset[Atom]
    instructions: str = ""

    def __post_init__(self) -> None:
        if self.executor not in ("robot", "human"):
            raise ProposalError(f"A phase's executor must be 'robot' or 'human', got {self.executor!r}.")

    @property
    def is_human(self) -> bool:
        return self.executor == "human"

    @property
    def objects(self) -> set[str]:
        """Every object this phase names, for the label-drift check."""
        return {value for atom in self.atoms for value in atom.values}

    def rebind(self, mapping: Mapping[str, str]) -> Phase:
        return Phase(
            self.executor,
            self.description,
            frozenset(a.rebind(mapping) for a in self.atoms),
            self.instructions,
        )

    def as_human(self, instructions: str) -> Phase:
        """The same phase, handed to a person instead.

        Used when a robot phase turns out not to be plannable and the run is configured to offer it
        as teleop rather than give up on the task. The atoms are unchanged, so what is checked
        afterwards is exactly what the planner was going to be asked for.
        """
        return Phase("human", self.description, self.atoms, instructions or self.instructions)

    def summary(self) -> dict:
        record = {
            "executor": self.executor,
            "description": self.description,
            "atoms": sorted(str(a) for a in self.atoms),
        }
        if self.is_human:
            record["instructions"] = self.instructions
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
            "surfaces": sorted(self.scene_types.surfaces),
            "movables": sorted(self.scene_types.movables),
            # Empty on a run that planned the whole instruction. Anything here is a clause the run
            # knowingly left out — read it before trusting a "success" label.
            "unrepresented": [dict(u) for u in self.unrepresented],
            "coverage": [dict(c) for c in self.coverage],
        }
