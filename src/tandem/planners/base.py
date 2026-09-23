"""What tandem needs from a task and motion planner, and nothing more.

tandem plans the task: it breaks an instruction into an ordered list of phases, decides which are
the robot's and which are a person's, runs the teleop legs itself, and verifies them. For a robot
phase it needs one thing from a planner — *achieve this goal in this scene, and record what you
did* — and that is the whole of this protocol.

Keeping the protocol this narrow is the point. The alternative, which is what this replaces, is to
teach every planner about phases, invented predicates and teleop hand-offs; that means forking each
one and keeping the fork alive. Here a new planner is a new implementation of ``TampBackend`` living
in *tandem*, and the planner's own sources stay untouched.

Nothing a planner knows is hardcoded on tandem's side. Which predicates a goal may be stated over,
which of them the planner supplies itself, whether one plan can pick the same object twice — all of
it is declared in ``Capabilities`` and read by ``tandem.planning``. That declaration is what makes
the phase planner planner-agnostic rather than merely planner-parameterised.

This module imports nothing outside the standard library: it is on the path of ``pip install
tandem-tamp`` on a laptop. A backend that needs torch imports it behind its own ``warm()``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

# `tandem.planning.symbols` is a leaf: it imports nothing but the standard library, and nothing in
# `tandem.planning` is imported from here. Sharing the vocabulary rather than restating it is what
# keeps a Capabilities declaration checkable against the atoms the phase planner builds.
from tandem.planning.symbols import Atom, Predicate

# --------------------------------------------------------------------------- what a planner can do


@dataclass(frozen=True)
class Capabilities:
    """Everything ``tandem.planning`` needs to know about a planner to plan for it.

    Every field here replaces something the phase planner used to import from cuTAMP directly. Read
    it as the planner's answer to "what can I be asked for, and what do I assume".
    """

    name: str

    # Predicates a GOAL may be stated over, by name, with their typed parameters. cuTAMP via tiptop
    # declares three: On(?obj: movable, ?surface: surface), Holding(?obj: movable), HandEmpty().
    # This IS the goal language shown to the proposer, so the parameter names are read by a human
    # and are worth choosing.
    goal_predicates: Mapping[str, Predicate] = field(default_factory=dict)

    # One sentence describing what the planner can physically do, in ABSTRACT terms — "pick an
    # object up and place it on a surface", not the planner's real operator signatures. Those carry
    # motion-level parameters (conf, traj, grasp) and bookkeeping predicates that a proposer has no
    # business reasoning about, and shown them it writes goals over the alternation lock.
    robot_description: str = "pick an object up and place it on a surface"

    # How each goal predicate is spelled on the wire. tiptop's create_tamp_environment reads
    # lowercase {"predicate": "on", "args": [...]}, so this is {"On": "on", "Holding": "holding"}.
    # A goal predicate ABSENT from this map is one the planner supplies for itself and that must be
    # dropped from a goal rather than sent — HandEmpty is added by tiptop whenever nothing is held.
    goal_predicate_wire_names: Mapping[str, str] = field(default_factory=dict)

    # Every predicate the planner could make true. The phase planner uses it for one cheap, sound,
    # instant check: reject a phase asking for something no operator can ever achieve, BEFORE
    # perception is paid for. Without it cuTAMP's search has no bound of any kind and, given an
    # unreachable goal, mints fresh conf/traj symbols forever without yielding.
    achievable_predicates: frozenset[str] = frozenset()

    # Predicate names a proposal may not reuse, because the planner already has them. Broader than
    # goal_predicates: motion bookkeeping (At, CanMove, JustMoved) and type declarations
    # (IsMovable, IsSurface) are not goal-statable but are still taken.
    reserved_predicate_names: frozenset[str] = frozenset()

    # Object types, and which is which. A `surface` is something other things are put ON; a
    # `movable` is something the robot can pick up.
    movable_type: str = "movable"
    surface_type: str = "surface"

    # Natural-language templates for the planner's own goal predicates, `{0}`-style. Used to tell an
    # operator what a phase will leave true. An invented predicate brings its own.
    predicate_descriptions: Mapping[str, str] = field(default_factory=dict)

    # Goal predicates a camera can settle, so a phase can be checked from a photo. cuTAMP's
    # Holding/HandEmpty are deliberately absent: verification runs on a third-person frame in which
    # the gripper is usually out of shot, and the classifier is told to answer false when it cannot
    # see the statement to be true.
    checkable_predicates: frozenset[str] = frozenset()

    # True when one plan may pick each object at most once (cuTAMP's Pick requires and DELETES
    # HasNotPickedUp). It makes `On(toy, table) and On(toy, shelf)` unsatisfiable rather than merely
    # slow, so two phases that move the same object must stay separate legs.
    one_pick_per_object: bool = True

    # True when every goal is planned from the same clean state — no `On` atom in the initial state,
    # so nothing symbolic enforces an ordering BETWEEN two robot phases and consecutive ones may be
    # conjoined into a single goal.
    initial_state_is_clean: bool = True

    # Whether execute() honours should_stop at step boundaries. False means preempt is abort.
    supports_cooperative_stop: bool = False

    # Whether plan() can be handed a previous PlanResult.skeleton to skip the symbolic search.
    supports_skeleton_reuse: bool = False

    # Paragraphs of the phase-segmentation prompt that only make sense for THIS planner, keyed by the
    # slot ``tandem.planning.prompts`` renders them into. The prompt the method was evaluated with
    # explains what On means, which predicates a precondition may be written in, and what a robot
    # phase may ask for -- every one of them a statement about cuTAMP's goal language, not about
    # phase planning. Held here so the prompt template itself names no predicate; a slot a backend
    # leaves out gets a generic paragraph rendered from goal_predicates, so an empty map is a
    # complete declaration, just a less specific prompt.
    prompt_fragments: Mapping[str, str] = field(default_factory=dict)

    # Predicates that can hold of an object in only ONE atom at a time, by predicate name -> the
    # position of that object's argument. {"On": 0} says a thing rests on one surface: asserting
    # On(toy, shelf) retracts On(toy, table) outright. The symbolic contract check
    # (``tandem.planning.contracts``) reads it as the one delete effect a phase gets for free -- a
    # robot phase has no operator and declares no delete effects, yet its placements unmistakably end
    # the old ones -- and without it a plan that moves the toy away and then needs it where it was
    # passes as consistent. A predicate absent here displaces nothing, which is right for a goal
    # language with no such exclusivity.
    exclusive_arguments: Mapping[str, int] = field(default_factory=dict)

    # Which argument of a goal atom names the object a robot phase physically MOVES, by predicate
    # name -> position. {"On": 0, "Holding": 0}: the toy in On(toy, box) moves, the box does not.
    # The distinction is the whole point -- "put toy_a on the table" and "put toy_b on the table"
    # share the table and move nothing in common. Read to tell the planner which objects the plan
    # actually asks the robot to pick (see supports_movable_restriction), and to flag two
    # consecutive robot phases that move the same object, the shape of a plan that wrote a step the
    # robot cannot do as a pick-and-place anyway. A predicate absent here moves nothing.
    moved_arguments: Mapping[str, int] = field(default_factory=dict)

    # The planner's own operators, as one signature each: Omega_0 in the paper's terms. Never shown
    # to the proposer (robot_description is what it reasons over, for the reason given there) and
    # never searched over. It is written into each rollout's provenance so the record says what the
    # robot side could do alongside the human operators the model invented -- a record that
    # hard-coded cuTAMP's would be a false statement the moment another planner ran the phases.
    robot_operators: tuple[str, ...] = ()

    # Whether plan() honours `movables=`: only the named objects may be picked, every other one is
    # an obstacle. A scene shared with a person contains the person's things -- "pull the block out
    # USING THE SCREWDRIVER" is what makes the screwdriver a detected object at all -- and a planner
    # told nothing treats the human's tool as one more thing to pick up. False means the phase
    # planner does not pass it, and the planner may move anything it detected.
    supports_movable_restriction: bool = False

    # Whether plan() honours `return_home=False`: end the recorded leg where the last operation left
    # the arm, instead of driving it home. Only the task's last leg should go home. One in the middle
    # is motion nobody asked for, recorded into the middle of the demonstration, and the next leg
    # then plans from home rather than from where this one stopped. False means every leg goes home.
    supports_return_home: bool = False

    def goal_predicate_names(self) -> frozenset[str]:
        return frozenset(self.goal_predicates)

    def describes(self, predicate: str) -> str | None:
        return self.predicate_descriptions.get(predicate)

    def predicate_menu(self) -> str:
        """The goal language as the proposer is shown it, one predicate per line.

        Rendered from the declaration rather than written out in the prompt, so a backend whose goal
        language differs cannot silently be asked for predicates it does not have.
        """
        # Declaration order, not sorted: the first line is the one the proposer leans on, and a
        # backend declaring its predicates puts the load-bearing one first for a reason.
        lines = []
        for name, predicate in self.goal_predicates.items():
            args = ", ".join(f"?{p.name}: {p.type}" for p in predicate.parameters)
            description = self.predicate_descriptions.get(name, "")
            suffix = f": {description}" if description else ""
            lines.append(f"- {name}({args}){suffix}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- wire types


@dataclass(frozen=True)
class GoalAtom:
    """One goal literal, in the form a planner consumes. JSON-safe by construction."""

    predicate: str
    args: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"predicate": self.predicate, "args": list(self.args)}


def to_goal_atoms(atoms: Sequence[Atom], caps: Capabilities) -> list[GoalAtom]:
    """Render tandem's atoms into the planner's goal language.

    Anything the planner supplies for itself is dropped rather than sent — see
    ``goal_predicate_wire_names``. Anything it cannot express at all is dropped too, which is safe
    only because ``tandem.planning.feasibility`` has already refused a robot phase that asks for one.
    """
    out: list[GoalAtom] = []
    for atom in sorted(atoms, key=str):
        wire = caps.goal_predicate_wire_names.get(atom.predicate)
        if wire is None:
            continue
        out.append(GoalAtom(wire, tuple(atom.values)))
    return out


@dataclass(frozen=True)
class SceneView:
    """What the planner perceived this pass.

    tandem needs the labels for two things it cannot do without them: stating a goal in the names
    this pass actually produced, and re-binding a plan whose object names have drifted since it was
    made (perception names objects afresh every pass — ``toy`` one time, ``blue_toy`` the next).
    """

    object_labels: tuple[str, ...] = ()
    table_label: str = "table"
    surface_labels: frozenset[str] = frozenset()
    # A frame tandem can hand to the verifier. Written by the backend; owned by the caller after.
    rgb_path: str | None = None
    # Opaque, and valid only until the next perceive() on the same backend.
    scene_id: str = ""
    # The goal the PLANNER's own translator made of the instruction during this pass. tandem ignores
    # it whenever it has a plan of its own -- but with phase planning off there is nothing to
    # decompose, and this is what keeps that path exactly the behaviour it always had: the same
    # translator, the same atoms, one code path, and no extra model call to get them.
    detected_goal: tuple[GoalAtom, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SceneView:
        return cls(
            object_labels=tuple(data.get("object_labels") or ()),
            table_label=str(data.get("table_label") or "table"),
            surface_labels=frozenset(data.get("surface_labels") or ()),
            rgb_path=data.get("rgb_path") or None,
            scene_id=str(data.get("scene_id") or ""),
            detected_goal=tuple(
                GoalAtom(str(a.get("predicate")), tuple(a.get("args") or ()))
                for a in (data.get("detected_goal") or ())
                if isinstance(a, Mapping)
            ),
        )


@dataclass
class PlanResult:
    ok: bool = False
    failure_reason: str | None = None
    planning_seconds: float = 0.0
    # Opaque handle passed straight back to execute(). Never inspected by tandem.
    plan_handle: Any = None
    # Opaque handle for skeleton reuse, when capabilities().supports_skeleton_reuse.
    skeleton: Any = None
    skeleton_reused: bool = False
    # What the planner wrote for this sub-goal, by role ("plan", "scene", ...).
    artifacts: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlanResult:
        return cls(
            ok=bool(data.get("ok")),
            failure_reason=data.get("failure_reason"),
            planning_seconds=float(data.get("planning_seconds") or 0.0),
            plan_handle=data.get("plan_handle"),
            skeleton=data.get("skeleton"),
            skeleton_reused=bool(data.get("skeleton_reused")),
            artifacts=dict(data.get("artifacts") or {}),
        )


@dataclass
class ExecuteResult:
    ok: bool = False
    stopped_early: bool = False
    n_frames: int = 0
    rollout_dir: str | None = None
    failure_reason: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExecuteResult:
        return cls(
            ok=bool(data.get("ok")),
            stopped_early=bool(data.get("stopped_early")),
            n_frames=int(data.get("n_frames") or 0),
            rollout_dir=data.get("rollout_dir"),
            failure_reason=data.get("failure_reason"),
        )


@dataclass(frozen=True)
class LegSpec:
    """Everything tandem knows about the leg it wants recorded.

    ``trajectory_id`` is minted by tandem, not the planner: it is what joins a task's legs — planner
    legs and teleop legs alike — into one episode, and only tandem sees both kinds. The backend must
    stamp it, and ``segment_source``, into the leg's ``_meta.json``, because that is what
    ``tandem.core.merge`` keys on.

    ``instruction`` is the WHOLE task, not the phase's own words: it is the dataset's language label,
    and an episode labelled with one phase of itself would be mislabelled training data. The phase
    text travels in ``phase_description`` as provenance.
    """

    trajectory_id: str
    instruction: str = ""
    segment_source: str = "tamp"
    phase_index: int | None = None
    n_phases: int | None = None
    phase_description: str = ""
    record: bool = True

    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "instruction": self.instruction,
            "segment_source": self.segment_source,
            "phase_index": self.phase_index,
            "n_phases": self.n_phases,
            "phase_description": self.phase_description,
            "record": self.record,
        }


class BackendError(Exception):
    """The backend could not do what was asked. Carries a message meant for an operator."""


# --------------------------------------------------------------------------- the protocol


@runtime_checkable
class TampBackend(Protocol):
    """A task and motion planner tandem can drive.

    The lifecycle is: ``require_ready`` → ``warm`` → (``perceive`` → ``plan`` → ``execute``)* →
    ``close``, with ``release_hardware``/``reacquire_hardware`` bracketing every teleop leg because
    the robot and the cameras admit exactly one owner.
    """

    name: str

    # -- what this planner is ------------------------------------------------
    def capabilities(self) -> Capabilities:
        """Declared, not discovered. Cheap, and must not need the planner to be warm."""

    def require_ready(self) -> None:
        """Raise ``RuntimeNotReady`` naming what is missing, or return."""

    # -- lifecycle -----------------------------------------------------------
    def warm(self) -> None:
        """Open cameras, connect the robot, build and warm the solvers. Tens of seconds."""

    def close(self) -> None:
        """Release everything. Safe to call twice, and safe on a backend that never warmed."""

    # -- hardware custody, so tandem can hand the arm to a person ------------
    def release_hardware(self) -> None:
        """Release the robot and every camera, and block until they are genuinely free.

        Not a formality: the cameras are held exclusively per process by serial, and a save pool
        forked after them inherits their handles, so a teleop process started too early sees a
        camera with serial 0. The planner must be fully out of the way before this returns.
        """

    def reacquire_hardware(self) -> None:
        """Take the robot and cameras back, from wherever the operator left the arm."""

    def capture_frame(self, *, camera: str = "external") -> str:
        """One RGB frame, written to a file, whose path is returned.

        This is what phase verification looks at. A third-person view by default: after a hand-off
        the arm is wherever the operator left it, so a wrist camera points nowhere useful.
        """

    def home(self) -> None:
        """Park the arm. Deliberately does not open the gripper — it may be holding something."""

    # -- the sub-goal cycle --------------------------------------------------
    def perceive(self, *, task_hint: str, save_dir: Path, reset_arm: bool = True) -> SceneView:
        """Look at the workspace and report what is in it.

        ``task_hint`` steers DETECTION only, never the goal: the full instruction is what makes a
        detector name things in task-relevant terms. The goal arrives separately, in ``plan``.

        ``reset_arm`` parks the arm first, which is what an ordinary planner rollout does. Turn it
        OFF for a phase resumed after a hand-off: the arm is where a person left it, quite possibly
        holding something, and driving it home would undo the step they just did.
        """

    def plan(
        self,
        scene_id: str,
        goal: Sequence[GoalAtom],
        *,
        surfaces: frozenset[str] = frozenset(),
        save_dir: Path,
        reuse_skeleton: Any = None,
    ) -> PlanResult:
        """Find a motion plan achieving ``goal`` in the scene ``scene_id`` named.

        ``surfaces`` pins which objects are surfaces for the whole task. It matters because an
        object's type decides its geometry: a box that is a surface in the phase that puts a toy
        into it and a movable in the phase that does not would change the world between two phases
        of one task.
        """

    def execute(
        self,
        plan_handle: Any,
        leg: LegSpec,
        *,
        save_dir: Path,
        should_stop: Callable[[], bool] | None = None,
    ) -> ExecuteResult:
        """Run the plan on the robot and record it as one leg of ``leg.trajectory_id``.

        ``should_stop`` is polled at step boundaries only when
        ``capabilities().supports_cooperative_stop``; otherwise it is ignored and a preempt is an
        abort.
        """


# Verbs a hosted backend answers, and the only strings that cross the wire. Kept here so the one
# canonical list lives beside the protocol it mirrors; `tandem/planners/tiptop/sidecar.py` repeats
# them because it must not import tandem, and a test pins the two together.
VERBS: tuple[str, ...] = (
    "capabilities",
    "warm",
    "close",
    "release_hardware",
    "reacquire_hardware",
    "capture_frame",
    "home",
    "perceive",
    "plan",
    "execute",
)

Camera = Literal["external", "perception", "hand"]
