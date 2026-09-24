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

import json
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
    # The operator sequence the plan runs, one label per operator in execution order, with its
    # motion-level arguments dropped: ("Pick(bread)", "Place(bread, plate)"), which is the task plan
    # the paper's figures show for a robot phase. Plain strings, so it is JSON-safe as it stands and
    # goes into a rollout's record next to the human operators the proposer invented. Provenance
    # only: nothing in tandem parses it or decides anything by it. Empty when the planner does not
    # say, which is not the same as a plan with nothing in it.
    task_plan: tuple[str, ...] = ()

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
            task_plan=tuple(str(label) for label in (data.get("task_plan") or ())),
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

    A new planner does not implement this by hand: it subclasses ``tandem.planners.Planner`` (or
    ``SidecarPlanner``, for one that runs in an environment of its own), which supplies every verb
    but ``perceive``, ``plan`` and ``execute`` and is its own factory. ``tandem.planners.testing``
    checks an implementation of this protocol, however it was written.
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
    def perceive(
        self,
        *,
        task_hint: str,
        save_dir: Path,
        reset_arm: bool = True,
        open_gripper: bool = False,
    ) -> SceneView:
        """Look at the workspace and report what is in it.

        ``task_hint`` steers DETECTION only, never the goal: the full instruction is what makes a
        detector name things in task-relevant terms. The goal arrives separately, in ``plan``.

        ``reset_arm`` parks the arm first, which is what an ordinary planner rollout does. Turn it
        OFF for a phase resumed after a hand-off: the arm is where a person left it, quite possibly
        holding something, and driving it home would undo the step they just did.

        ``open_gripper`` opens the gripper before looking, and moves nothing else. It is for the first
        robot leg after a human phase: nothing about a person driving the arm guarantees the fingers
        were left open, and a planner that starts every goal from an empty hand (cuTAMP's HandEmpty)
        would plan its first grasp as though they were. Off by default, and never implied by
        ``reset_arm``, because opening the hand of an arm that is holding something drops it -- the
        caller, which knows what the phase before was for, is the one to decide.
        """

    def plan(
        self,
        scene_id: str,
        goal: Sequence[GoalAtom],
        *,
        surfaces: frozenset[str] = frozenset(),
        movables: frozenset[str] | None = None,
        return_home: bool = True,
        save_dir: Path,
        reuse_skeleton: Any = None,
    ) -> PlanResult:
        """Find a motion plan achieving ``goal`` in the scene ``scene_id`` named.

        ``surfaces`` pins which objects are surfaces for the whole task. It matters because an
        object's type decides its geometry: a box that is a surface in the phase that puts a toy
        into it and a movable in the phase that does not would change the world between two phases
        of one task.

        ``movables``, when given, are the only objects the plan may pick up. Every other detected
        object stays in the world as an obstacle: perceived, avoided, never grasped. None means no
        restriction, which is an ordinary rollout. A goal that itself moves an object outside the set
        is not planned with the restriction quietly widened -- the result is ``ok=False``, with the
        object named, because the phase planner and the backend disagree about what this leg is for.

        ``return_home=False`` ends the plan where its last operation leaves the arm instead of
        driving it home. Only the leg that ends the task should go home; any other is continued from
        where it stops, by a person or by the next leg.

        tandem passes ``movables`` only when ``capabilities().supports_movable_restriction`` and
        ``return_home`` only when ``capabilities().supports_return_home``. A backend declaring
        neither is never handed either, and may leave both out of its signature.
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


# --------------------------------------------------------------------------- building one, and the catalog
#
# A backend is never constructed by name-specific code in tandem. The registry maps a planner's name
# to a FACTORY, and the factory is the only thing that knows how its backend is put together -- which
# runtime it runs in, which environment variables it reads, which files it wants written first. The
# session hands every factory the same BackendContext and gets a TampBackend back. That one seam is
# what lets a planner tandem has never heard of be installed as a package and named in a profile.
#
# The same factory is also what a catalog of planners reads: what each one is (PlannerInfo), what it
# can be asked for (Capabilities), and what it needs installed before it can run (BackendRuntime).
# All three are answerable without building the backend, and without the heavy environment.


@dataclass(frozen=True)
class SourcePin:
    """One source tree a planner's runtime is built from, pinned to an exact commit.

    A commit rather than a branch or a tag: a dataset has to be traceable to the planner that
    produced it, and "main" names a different planner every week.
    """

    name: str
    url: str
    commit: str

    def short(self) -> str:
        return self.commit[:7]

    def to_dict(self) -> dict:
        return {"name": self.name, "url": self.url, "commit": self.commit}


@dataclass(frozen=True)
class PlannerInfo:
    """What a catalog says about a planner before any of it is installed.

    Static and cheap by contract: a listing of every planner reads this for each of them, on a
    laptop, and must not import a solver or touch the network to do it.
    """

    # The name a profile's ``planner.backend`` uses. The registry refuses a factory whose info names
    # a different planner from the one it was registered as, so the two can never disagree.
    name: str
    display_name: str = ""
    # One sentence: what this planner does and what it drives.
    summary: str = ""
    homepage: str = ""
    # What the machine needs before this planner can run, one human-readable line each ("an NVIDIA
    # GPU with CUDA 12 or newer", "a Franka FR3"). Shown, never checked -- checking is the runtime's
    # status() and ``tandem doctor``.
    requires: tuple[str, ...] = ()
    # The sources an install builds the runtime from. Empty for a planner that is pure Python and
    # installs with pip like any other package.
    sources: tuple[SourcePin, ...] = ()

    @property
    def title(self) -> str:
        return self.display_name or self.name

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.title,
            "summary": self.summary,
            "homepage": self.homepage,
            "requires": list(self.requires),
            "sources": [pin.to_dict() for pin in self.sources],
        }


@dataclass(frozen=True)
class BackendContext:
    """Everything a session hands a factory to build its backend.

    Deliberately planner-neutral. Nothing here is TipTop's, and a field a backend has no use for is
    simply ignored by it; what one backend needs that no other does belongs in ``options``, which is
    the profile's ``planner.options`` block and is the backend's own to define and validate.
    """

    # The profile the session runs under (``tandem.core.profiles.Profile``; typed loosely so this
    # module keeps importing nothing but the standard library).
    profile: Any
    # This session's scratch directory. The backend may write whatever it needs to start here.
    session_dir: Path
    # Where finished legs are filed: the profile's trajectories directory.
    output_dir: Path
    execute: bool = True
    record: bool = True
    # (stream, text) -> None. The session's log, which is what an operator sees.
    on_log: Callable[[str, str], None] | None = None
    # The profile's ``planner.options``: per-backend settings, verbatim. A backend must refuse a key
    # it does not read rather than ignore it -- an option that silently does nothing is a setting
    # the operator believes is in force and is not.
    options: Mapping[str, Any] = field(default_factory=dict)
    # tandem's machine settings (``tandem.core.settings.Settings``), or None for the saved ones.
    settings: Any = None
    session_id: str = ""
    # The task the session starts on. Later tasks reach the backend through ``perceive(task_hint=)``.
    task: str = ""
    # The session's append-only events file, for a backend that writes its own events there too.
    events_file: Path | None = None
    # Where the caller has already located this planner's runtime, if it has. None means the factory
    # resolves its own from ``settings``. A backend with no runtime ignores it.
    runtime_dir: Path | None = None

    def log(self, text: str, *, stream: str = "tandem") -> None:
        if self.on_log is not None:
            self.on_log(stream, text)


@dataclass(frozen=True)
class RuntimeStatus:
    """Whether a planner's runtime is installed, and what it was built from.

    ``installed`` means ready to run, not merely present: a runtime whose sources are on disk but
    whose kernels never compiled is not installed, and ``problems`` says why.
    """

    installed: bool = False
    # Where the runtime lives, as a string so the status is JSON-safe as it stands.
    path: str | None = None
    # The sources the INSTALLED runtime was built from, which is not necessarily what the planner
    # currently pins (see ``mismatched``). Empty when it does not say.
    pins: tuple[SourcePin, ...] = ()
    # For a runtime identified by a version rather than by commits.
    version: str | None = None
    # One line of what is and is not there, for a listing.
    detail: str = ""
    problems: tuple[str, ...] = ()

    def mismatched(self, wanted: Sequence[SourcePin]) -> tuple[str, ...]:
        """Names of the pinned sources this runtime was NOT built at.

        A source the installed runtime does not record at all counts as mismatched: a runtime that
        cannot say what it was built from cannot be said to match anything.
        """
        have = {pin.name: pin.commit for pin in self.pins}
        return tuple(pin.name for pin in wanted if have.get(pin.name) != pin.commit)

    def to_dict(self) -> dict:
        return {
            "installed": self.installed,
            "path": self.path,
            "pins": [pin.to_dict() for pin in self.pins],
            "version": self.version,
            "detail": self.detail,
            "problems": list(self.problems),
        }


@runtime_checkable
class BackendRuntime(Protocol):
    """The heavy environment a planner runs in, as something that can be inspected and installed.

    A planner that is pure Python has none: its factory's ``runtime()`` returns None, and installing
    it is ``pip install``.
    """

    def status(self) -> RuntimeStatus:
        """What is installed. Cheap -- a few stat calls, never a build, never the network."""

    def install(
        self,
        *,
        on_progress: Callable[[str], None] | None = None,
        sources_dir: Path | None = None,
        force: bool = False,
    ) -> None:
        """Build the runtime, or repair it. Idempotent: a step already done is skipped.

        ``on_progress`` receives one line at a time, as the build prints them. ``sources_dir``
        overrides where the sources come from -- a directory of checkouts, for a machine with no
        network or for a planner under development. ``force`` redoes every step.
        """

    def uninstall(self) -> None:
        """Delete the runtime. Refuses, loudly, anything that does not look like one."""


@dataclass(frozen=True)
class OptionsSection:
    """One titled group of a planner's settings, as `tandem profile show` and the web editor list them."""

    title: str
    rows: tuple[tuple[str, str], ...] = ()
    subtitle: str = ""

    def to_dict(self) -> dict:
        return {"title": self.title, "subtitle": self.subtitle, "rows": [list(row) for row in self.rows]}


@dataclass(frozen=True)
class OptionsView:
    """How a planner describes a profile's ``planner.options`` to a person. JSON-safe by construction.

    A profile page, `tandem profile show` and a session header show every planner's settings through
    this, so none of them knows any planner's schema -- a robot's address, a TAMP override -- and a
    planner written tomorrow is shown the day it is registered.
    """

    # One line: the setup at a glance, for a profile card or a session header.
    summary: str = ""
    sections: tuple[OptionsSection, ...] = ()
    # What the planner will actually be handed from these options, resolved (a relative path made
    # absolute): the answer to "did my setting apply?". JSON-safe.
    receives: Mapping[str, Any] = field(default_factory=dict)
    # How it is handed over, for the heading above ``receives``.
    receives_note: str = ""
    # Problems these settings have that would otherwise surface minutes into a session.
    warnings: tuple[str, ...] = ()

    @classmethod
    def generic(cls, options: Mapping[str, Any], descriptions: Mapping[str, str] | None = None) -> OptionsView:
        """The view of a planner that does not describe its own: its options, one row each, as they are."""
        rows = tuple((str(key), _shown(value)) for key, value in options.items())
        unset = [key for key in (descriptions or {}) if key not in options]
        subtitle = "none set" if not rows else ""
        if unset:
            subtitle = (subtitle + "; " if subtitle else "") + "also reads " + ", ".join(unset)
        return cls(
            summary=", ".join(f"{key}={_shown(value)}" for key, value in options.items())[:120],
            sections=(OptionsSection("options", rows, subtitle),),
            receives=dict(options),
            receives_note="planner.options, as the profile sets them",
        )

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "sections": [section.to_dict() for section in self.sections],
            "receives": dict(self.receives),
            "receives_note": self.receives_note,
            "warnings": list(self.warnings),
        }


def _shown(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, Mapping):
        return json.dumps(value, default=str)
    return str(value)


@runtime_checkable
class BackendFactory(Protocol):
    """How the registry builds a planner, and what a catalog of planners reads about it.

    Register one under a name with ``tandem.planners.registry.register_backend``, or from a package
    through the ``tandem.planners`` entry-point group, and a profile can name it.

    Beyond the four members below, a factory MAY offer any of these. Each has a default in
    ``tandem.planners.registry`` for a factory that does not, and ``tandem.planners.Planner`` supplies
    every one, so a planner written with the SDK overrides only what it has something to say about:

    - ``validate_options(options) -> dict``: ``planner.options`` checked and normalised, called when
      a profile naming the planner loads. Default: taken as written.
    - ``describe_options(profile, *, settings=None) -> OptionsView``: those options for a person.
      Default: ``OptionsView.generic``.
    - ``doctor_checks(profile, *, settings=None, probe_hardware=True) -> list[tandem.core.probe.Check]``:
      what `tandem doctor` (and, with ``profile=None``, `tandem init`'s preflight) should check for this
      planner -- a GPU, a server it calls, its hardware. Default: nothing beyond its runtime, which
      doctor checks for every planner.
    - ``replay(rollout_dir, *, settings=None) -> None``: open a recorded leg in the planner's own
      viewer (`tandem traj open`). Default: refused, saying the planner has none.
    - ``importer``: a ``ProfileImporter`` building a profile from the planner's own older
      configuration (`tandem profile create --import-from`). Default: None.
    - ``presets_dir``: a directory of ``<name>.yml`` presets for this planner's ``planner.options``
      (`tandem profile create --preset NAME`; the layout is in ``tandem.core.presets``). Default: None.
    """

    info: PlannerInfo

    def capabilities(self) -> Capabilities:
        """The declaration, with nothing built and nothing heavy imported."""

    def create(self, ctx: BackendContext) -> TampBackend:
        """Build the backend a session will drive. Not warmed: the session calls ``warm()`` itself.

        Everything this planner needs set up before it can be warmed happens here -- its runtime
        located, its config rendered, its own preflight checks run. A problem it can already see is
        raised here, loudly, before the session owns anything.
        """

    def runtime(self, settings: Any = None) -> BackendRuntime | None:
        """This planner's runtime on this machine, or None when it is pure Python and has none."""


class ProfileImporter(Protocol):
    """Builds a profile from a planner's own configuration elsewhere -- a checkout of the system it came from.

    What `tandem profile create --import-from` and `tandem init` run, through the named planner's
    factory, so neither has to know what that other system's files look like.
    """

    #: What it imports from, for a help line: "a hitl-tamp-vla checkout".
    source: str

    def find(self, near: Path) -> Path | None:
        """A source at or above ``near`` worth suggesting, or None. Only a suggestion: nothing is read."""

    def configs(self, source: Path) -> list[Path]:
        """Task configurations inside ``source`` a person may pick one of, for the task and its settings."""

    def build(self, name: str, *, source: Path | None = None, config: Path | None = None) -> tuple[Any, dict, list[str]]:
        """``(profile, calibration, notes)``: a ``Profile`` named ``name``, extrinsics keyed by camera
        serial, and one line for each thing a person should know about what was and was not imported.

        A note starting with ``WARNING_NOTE`` is shown as a warning. Use it for something the source set
        that the profile does not carry, and that the person has to decide about before collecting.
        """


#: The prefix of a ``ProfileImporter`` note that is a warning rather than information. A prefix, so the
#: notes stay plain strings every importer and every caller already handles.
WARNING_NOTE = "warning: "


# Verbs a hosted backend answers, and the only strings that cross the wire. Kept here so the one
# canonical list lives beside the protocol it mirrors; `tandem/planners/sidecar_kit/tandem_sidecar.py`
# and `tandem/planners/tiptop/sidecar.py` repeat them because neither may import tandem, and tests pin
# every copy to this one.
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
