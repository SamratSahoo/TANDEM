"""The base class a planner is written against: declare what it is, implement three verbs, register it.

``tandem.planners.base.TampBackend`` is the protocol tandem drives. Implementing all of it by hand
means twelve members, a factory beside them, and a dozen conventions that exist only in docstrings.
``Planner`` collapses that into what is actually specific to a planner:

    from tandem.planners import (
        Capabilities, ExecuteResult, Parameter, Planner, PlannerInfo, PlanResult, Predicate,
        SceneView, register_backend,
    )

    IN_BIN = Predicate("InBin", (Parameter("obj", "item"), Parameter("bin", "container")))

    class BinPlanner(Planner):
        info = PlannerInfo(name="bins", display_name="Bin sorter", summary="Drops items into bins.")
        CAPABILITIES = Capabilities(
            name="bins",
            goal_predicates={"InBin": IN_BIN},
            robot_description="drop an item into a bin",
            goal_predicate_wire_names={"InBin": "in_bin"},
            achievable_predicates=frozenset({"InBin"}),
            reserved_predicate_names=frozenset({"InBin"}),
            movable_type="item",
            surface_type="container",
            predicate_descriptions={"InBin": "{0} is inside {1}"},
            checkable_predicates=frozenset({"InBin"}),
            moved_arguments={"InBin": 0},
        )

        def perceive(self, *, task_hint, save_dir, reset_arm=True, open_gripper=False) -> SceneView: ...
        def plan(self, scene_id, goal, *, surfaces=frozenset(), save_dir, reuse_skeleton=None) -> PlanResult: ...
        def execute(self, plan_handle, leg, *, save_dir, should_stop=None) -> ExecuteResult: ...

    register_backend("bins", BinPlanner)      # or the "tandem.planners" entry point, pointing at the class

and ``planner: {backend: bins}`` in a profile is all it takes after that.

What the base class supplies, and why each default is the one it is:

- **It is its own factory.** ``info``, ``capabilities()``, ``create(ctx)`` and ``runtime(settings)``
  are class-level, so the class itself satisfies ``BackendFactory`` and the registry uses it as one
  (it does NOT instantiate it to get a factory: an instance is a backend, built per session).
  ``create(ctx)`` checks the context's ``planner.options`` (``validate_options``) and then calls
  ``cls(ctx)`` with them as checked; override it when construction needs more than the context.
- **Its options are its own.** ``OPTIONS`` names the ``planner.options`` keys it reads, one line each,
  and the default ``validate_options`` refuses any other -- when a profile naming the planner loads,
  not when a session starts. A planner whose options have structure (types, ranges, nested blocks)
  overrides ``validate_options`` to check and normalise them; TiPToP's is a pydantic model.
- **What else tandem asks it has a default too.** ``describe_options`` lists its options as they are
  (`tandem profile show`, the web editor); ``doctor_checks`` adds nothing to `tandem doctor` beyond
  its runtime, which doctor checks for every planner; ``replay`` says it has no viewer; ``importer``
  is None. Override one when the planner has something to say: a server it calls, a GPU it needs.
- **The declarations are checked when the class is defined**, not when a session first reads them.
  A moved argument that points at a surface, a wire name for a predicate that does not exist, a
  prompt slot misspelt -- each is a planner that would load, list and start, and then plan the wrong
  thing or silently ignore a setting. They raise ``TandemError`` at import, naming the field.
- **Every optional verb has a default.** ``warm``, ``close``, ``home``, ``release_hardware`` and
  ``reacquire_hardware`` do nothing: right for a planner that holds no hardware, and a planner that
  does overrides them. ``require_ready`` checks the declared runtime (``recipe``) and is otherwise a
  no-op. ``capture_frame`` and ``move_to_joints`` raise ``UnsupportedVerb``, loudly: there is no
  honest default for a camera frame or an arm motion, and a verifier handed a made-up frame would
  pass or fail a human phase on nothing.
- ``perceive``, ``plan`` and ``execute`` are abstract, because they ARE the planner. Their docstrings
  here are the contract, including the recording contract ``execute`` must meet.

A planner that has to run in an environment of its own (torch, CUDA, a robot SDK) subclasses
``tandem.planners.sidecar.SidecarPlanner`` instead, which implements all three verbs by talking to
a script running there. ``tandem.planners.testing`` checks either kind against the whole protocol.

Standard library and tandem's light modules only: a planner class is imported to list planners.
"""

from __future__ import annotations

import abc
import difflib
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

from tandem.core.errors import RuntimeNotReady, TandemError
from tandem.planners.base import (
    BackendContext,
    BackendRuntime,
    Capabilities,
    ExecuteResult,
    GoalAtom,
    LegSpec,
    OptionsView,
    PlannerInfo,
    PlanResult,
    SceneView,
    SourcePin,
)
from tandem.planners.registry import _NAME as _PLANNER_NAME
from tandem.planners.runtime import RuntimeRecipe
from tandem.planning.prompts import PROMPT_SLOTS
from tandem.planning.symbols import Predicate

_log = logging.getLogger(__name__)

# `Pick(?obj: movable)` or `Pick(obj: movable)`: the operator signatures Capabilities.robot_operators
# holds. Both spellings are accepted because the human operators the proposer invents render without
# the `?` (Parameter strips it) and the BRIEF's robot ones with it; what matters is that each names
# typed parameters, since that is what a reader of the provenance record needs.
_SIGNATURE = re.compile(r"^([A-Za-z_]\w*)\((.*)\)$")
_TYPED_PARAMETER = re.compile(r"^\??([A-Za-z_]\w*)\s*:\s*([A-Za-z_][\w-]*)$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")

# The boolean switches of a Capabilities. A truthy string ("false") in one of these reads as True,
# which is the worst possible way for a declaration to be wrong.
_FLAGS = (
    "one_pick_per_object",
    "initial_state_is_clean",
    "supports_cooperative_stop",
    "supports_skeleton_reuse",
    "supports_movable_restriction",
    "supports_return_home",
)


class UnsupportedVerb(TandemError):
    """A planner was asked for something it does not do, and has no honest default for.

    Raised by ``Planner``'s defaults for ``capture_frame`` and ``move_to_joints``. It is a
    ``TandemError``, so it reaches an operator as a message and a hint rather than a traceback.
    """


# --------------------------------------------------------------------------- checking a declaration


def capability_problems(caps: Any) -> list[str]:
    """Everything wrong with a ``Capabilities`` declaration, one sentence each. Empty means sound.

    Each check is a declaration that would load and then do the wrong thing without saying so: the
    phase planner would reject every phase asking for an unachievable goal predicate, tell a planner
    to pick nothing because no moved argument was declared, or render the generic prompt paragraph
    in place of a misspelt fragment. ``Planner`` runs this when a subclass is defined, and
    ``tandem.planners.testing`` runs it on any factory.
    """
    if not isinstance(caps, Capabilities):
        return [f"CAPABILITIES is a {type(caps).__name__}, not a tandem.planners.Capabilities"]
    problems: list[str] = []

    if not isinstance(caps.name, str) or not caps.name.strip():
        problems.append("Capabilities.name is empty")
    if not isinstance(caps.robot_description, str) or not caps.robot_description.strip():
        problems.append(
            "robot_description is empty; it is the one sentence the proposer reads about what the robot can do"
        )
    for field in ("movable_type", "surface_type"):
        value = getattr(caps, field)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{field} is empty")
    if caps.movable_type == caps.surface_type:
        problems.append(
            f"movable_type and surface_type are both {caps.movable_type!r}, so nothing can tell a thing the "
            "robot moves from a thing it puts things on"
        )

    goal = caps.goal_predicates if isinstance(caps.goal_predicates, Mapping) else {}
    if not isinstance(caps.goal_predicates, Mapping):
        problems.append("goal_predicates is not a mapping of name -> Predicate")
    if not goal:
        problems.append("it declares no goal_predicates, so no robot phase could ever be stated for it")
    predicates: dict[str, Predicate] = {}
    for key, predicate in goal.items():
        if not isinstance(predicate, Predicate):
            problems.append(f"goal_predicates[{key!r}] is a {type(predicate).__name__}, not a Predicate")
            continue
        if predicate.name != key:
            problems.append(f"goal_predicates[{key!r}] is a predicate named {predicate.name!r}")
            continue
        for parameter in predicate.parameters:
            if not parameter.type:
                problems.append(f"{key}'s parameter {parameter.name!r} has no type")
        predicates[key] = predicate
    names = set(predicates)

    wire = caps.goal_predicate_wire_names
    if not isinstance(wire, Mapping):
        problems.append("goal_predicate_wire_names is not a mapping of name -> wire name")
        wire = {}
    for key, spelled in wire.items():
        if key not in names:
            problems.append(
                f"goal_predicate_wire_names names {key!r}, which is not a goal predicate{_hint(key, names)}"
            )
        if not isinstance(spelled, str) or not spelled:
            problems.append(f"goal_predicate_wire_names[{key!r}] is empty")
    spellings = [s for s in wire.values() if isinstance(s, str)]
    shared = sorted({s for s in spellings if spellings.count(s) > 1})
    if shared:
        problems.append(
            f"two goal predicates share the wire name(s) {', '.join(shared)}, so the planner cannot tell them apart"
        )
    if names and not names & set(wire):
        problems.append(
            "none of its goal predicates has a wire name (goal_predicate_wire_names), so every goal would "
            "reach the planner empty"
        )

    sets = {}
    for field in ("achievable_predicates", "reserved_predicate_names", "checkable_predicates"):
        value = getattr(caps, field)
        if isinstance(value, (str, bytes)) or not all(isinstance(v, str) for v in _iterable(value)):
            problems.append(f"{field} must be a set of predicate names")
            sets[field] = set()
        else:
            sets[field] = set(value)
    unachievable = sorted(names - sets["achievable_predicates"])
    if unachievable:
        problems.append(
            f"goal predicate(s) {', '.join(unachievable)} are not in achievable_predicates, so the phase "
            "planner rejects every robot phase that asks for them"
        )
    unreserved = sorted(names - sets["reserved_predicate_names"])
    if unreserved:
        problems.append(
            f"goal predicate(s) {', '.join(unreserved)} are not in reserved_predicate_names, so a proposal "
            "could invent a predicate of the same name"
        )
    uncheckable = sorted(sets["checkable_predicates"] - names)
    if uncheckable:
        problems.append(
            f"checkable_predicates names {', '.join(uncheckable)}, which are not goal predicates -- an "
            "invented predicate is always checkable and needs no entry here"
        )

    descriptions = caps.predicate_descriptions if isinstance(caps.predicate_descriptions, Mapping) else {}
    for key, template in descriptions.items():
        if key not in names:
            problems.append(
                f"predicate_descriptions describes {key!r}, which is not a goal predicate{_hint(key, names)}"
            )
            continue
        if not isinstance(template, str):
            problems.append(f"predicate_descriptions[{key!r}] is not a string")
            continue
        try:
            template.format(*(f"x{i}" for i in range(predicates[key].arity)))
        except (IndexError, KeyError, ValueError) as exc:
            problems.append(
                f"predicate_descriptions[{key!r}] does not format with {predicates[key].arity} argument(s) "
                f"({{0}}, {{1}}, ...): {type(exc).__name__}: {exc}"
            )

    for field in ("exclusive_arguments", "moved_arguments"):
        positions = getattr(caps, field)
        if not isinstance(positions, Mapping):
            problems.append(f"{field} is not a mapping of predicate -> argument position")
            continue
        for key, position in positions.items():
            predicate = predicates.get(key)
            if predicate is None:
                problems.append(f"{field} names {key!r}, which is not a goal predicate{_hint(key, names)}")
                continue
            if (
                isinstance(position, bool)
                or not isinstance(position, int)
                or not 0 <= position < predicate.arity
            ):
                problems.append(
                    f"{field}[{key!r}] = {position!r}, but {key} takes {predicate.arity} argument(s)"
                )
                continue
            # The moved argument is the object a robot phase PICKS -- what a movable restriction is
            # computed from. Pointing it at a surface would tell the planner to pick up the table.
            parameter = predicate.parameters[position]
            if field == "moved_arguments" and parameter.type != caps.movable_type:
                problems.append(
                    f"moved_arguments[{key!r}] points at ?{parameter.name}, a {parameter.type}, but the "
                    f"object a robot phase moves is a {caps.movable_type} (movable_type)"
                )

    fragments = caps.prompt_fragments if isinstance(caps.prompt_fragments, Mapping) else {}
    for key, text in fragments.items():
        if key not in PROMPT_SLOTS:
            problems.append(
                f"prompt_fragments has {key!r}, which is not a slot of the phase-planning prompt{_hint(key, PROMPT_SLOTS)}"
            )
        elif not isinstance(text, str):
            problems.append(f"prompt_fragments[{key!r}] is not a string")

    operators = caps.robot_operators
    if isinstance(operators, str) or not isinstance(operators, Sequence):
        problems.append("robot_operators must be a tuple of signatures such as 'Pick(?obj: movable)'")
        operators = ()
    known_types = {caps.movable_type, caps.surface_type} | {
        parameter.type for predicate in predicates.values() for parameter in predicate.parameters
    }
    for signature in operators:
        problem = _operator_problem(signature, known_types)
        if problem:
            problems.append(problem)

    for flag in _FLAGS:
        if not isinstance(getattr(caps, flag), bool):
            problems.append(f"{flag} is {getattr(caps, flag)!r}, not True or False")
    if caps.supports_movable_restriction is True and not (
        isinstance(caps.moved_arguments, Mapping) and caps.moved_arguments
    ):
        problems.append(
            "supports_movable_restriction is set but no moved_arguments are declared: the objects a leg may "
            "pick are read from them, so every leg would be told it may pick nothing"
        )
    return problems


def info_problems(info: Any) -> list[str]:
    """Everything wrong with a ``PlannerInfo``, one sentence each."""
    if not isinstance(info, PlannerInfo):
        return [f"info is a {type(info).__name__}, not a tandem.planners.PlannerInfo"]
    problems: list[str] = []
    if not isinstance(info.name, str) or not _PLANNER_NAME.match(info.name):
        problems.append(
            f"info.name {info.name!r} is not a usable planner name (lowercase letters, digits, _ and -, "
            "starting with a letter): it is typed into profiles and onto command lines"
        )
    if isinstance(info.requires, str) or not all(isinstance(line, str) for line in _iterable(info.requires)):
        problems.append("info.requires must be a tuple of lines, one requirement each")
    for pin in _iterable(info.sources):
        if not isinstance(pin, SourcePin):
            problems.append(f"info.sources holds a {type(pin).__name__}, not a SourcePin")
        elif not _COMMIT.match(pin.commit or ""):
            problems.append(
                f"info.sources pins {pin.name!r} to {pin.commit!r}, which is not a full 40-character commit: "
                "a dataset has to be traceable to the exact planner that produced it"
            )
    return problems


def factory_problems(factory: Any) -> list[str]:
    """What is wrong with a planner factory's declarations: its info, its capabilities, and the two agreeing.

    Works on a ``Planner`` subclass, on any ``BackendFactory`` instance, and on a class that is one.
    Builds nothing and touches no runtime.
    """
    missing = [m for m in ("info", "capabilities", "create", "runtime") if not hasattr(factory, m)]
    if missing:
        return [f"it is not a planner factory: it has no {', '.join(missing)}"]
    problems = info_problems(factory.info)
    try:
        caps = factory.capabilities()
    except Exception as exc:
        return [*problems, f"capabilities() raised {type(exc).__name__}: {exc}"]
    problems += capability_problems(caps)
    if (
        isinstance(caps, Capabilities)
        and isinstance(factory.info, PlannerInfo)
        and caps.name != factory.info.name
    ):
        problems.append(
            f"its capabilities are named {caps.name!r} but its info {factory.info.name!r}: a rollout's record "
            "and the catalog would call one planner by two names"
        )
    return problems


def _operator_problem(signature: Any, known_types: set[str]) -> str | None:
    if not isinstance(signature, str):
        return f"robot_operators holds a {type(signature).__name__}, not a signature string"
    match = _SIGNATURE.match(signature.strip())
    if match is None:
        return f"robot_operators entry {signature!r} is not a signature such as 'Pick(?obj: movable)'"
    body = match.group(2).strip()
    for part in (p.strip() for p in body.split(",")) if body else ():
        parameter = _TYPED_PARAMETER.match(part)
        if parameter is None:
            return f"robot_operators entry {signature!r}: {part!r} is not a typed parameter such as '?obj: movable'"
        if parameter.group(2) not in known_types:
            return (
                f"robot_operators entry {signature!r} uses the type {parameter.group(2)!r}, which is none of "
                f"the declared types ({', '.join(sorted(known_types))})"
            )
    return None


def _hint(key: str, names: Sequence[str] | set[str]) -> str:
    """ " (did you mean 'On'?)" -- case first, because the likeliest slip is the wire spelling."""
    by_case = {n.lower(): n for n in names}
    if isinstance(key, str) and key.lower() in by_case:
        return f" (did you mean {by_case[key.lower()]!r}?)"
    close = difflib.get_close_matches(str(key), list(names), n=1, cutoff=0.6)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _iterable(value: Any) -> tuple:
    try:
        return tuple(value)
    except TypeError:
        return ()


def _abstract_methods(cls: type) -> set[str]:
    """The abstract methods ``cls`` leaves unimplemented, worked out the way ABCMeta will.

    ``__init_subclass__`` runs inside ``type.__new__``, before ABCMeta has computed
    ``__abstractmethods__`` for the new class, so ``inspect.isabstract`` cannot answer yet.
    """
    names = {name for name, value in vars(cls).items() if getattr(value, "__isabstractmethod__", False)}
    for base in cls.__bases__:
        for name in getattr(base, "__abstractmethods__", ()):
            if getattr(getattr(cls, name, None), "__isabstractmethod__", False):
                names.add(name)
    return names


# --------------------------------------------------------------------------- the base class


class Planner(abc.ABC):
    """A task and motion planner tandem can drive, written as one class.

    Subclass it, declare ``info`` and ``CAPABILITIES`` (and ``recipe`` if it needs a runtime built),
    implement ``perceive``, ``plan`` and ``execute``, and register the class. See the module
    docstring for an example and for why each default is what it is.

    Passing ``abstract=True`` in the class statement (``class MyBase(Planner, abstract=True)``) marks
    a base for other planners -- one that declares nothing itself and is never registered. A class
    with abstract methods left is treated the same way. Every other subclass is checked, when it is
    defined, for a complete and consistent declaration.
    """

    # ---- declarations: what the planner IS, read without building it ---------------------------

    #: What a catalog says about the planner. ``info.name`` is the name a profile's
    #: ``planner.backend`` uses and the one it must be registered under.
    info: ClassVar[PlannerInfo]
    #: What the phase planner may ask it for. Its ``name`` must equal ``info.name``.
    CAPABILITIES: ClassVar[Capabilities]
    #: The runtime it runs in, when it needs more than pip: pinned sources, an environment, build
    #: steps. None for a pure-Python planner. ``info.sources`` is filled in from it when left empty.
    recipe: ClassVar[RuntimeRecipe | None] = None
    #: The ``planner.options`` keys it reads, each with one line saying what it does. Anything else
    #: in a profile's options is refused (``validate_options``) rather than ignored: an option that
    #: silently does nothing is a setting the operator believes is in force and is not.
    OPTIONS: ClassVar[Mapping[str, str]] = {}
    #: The backend's name, as ``TampBackend`` has it. Defaults to ``info.name``.
    name: ClassVar[str] = ""
    #: Builds a profile from this planner's own older configuration (``base.ProfileImporter``), for
    #: `tandem profile create --import-from`. None: there is nothing to import from.
    importer: ClassVar[Any] = None

    # Set per class by __init_subclass__: whether this class is a base rather than a planner.
    _planner_base: ClassVar[bool] = True

    def __init_subclass__(cls, *, abstract: bool = False, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        leftover = _abstract_methods(cls)
        cls._planner_base = bool(abstract or leftover)
        problems = _declaration_problems(cls, complete=not cls._planner_base)
        if problems:
            raise TandemError(
                f"{cls.__module__}.{cls.__qualname__} is not a usable planner:\n"
                + "\n".join(f"  - {problem}" for problem in problems),
                hint="Fix the class's declarations. They are checked when the class is defined so that a "
                "planner declared wrongly never reaches a catalog or a session.",
            )
        if cls._planner_base:
            return
        # Filled in rather than asked for twice: a catalog listing commits the install does not
        # deliver would be a false statement about every dataset collected with it.
        if cls.recipe is not None and not cls.info.sources:
            cls.info = replace(cls.info, sources=cls.recipe.pins)
        if "name" not in vars(cls):
            cls.name = cls.info.name

    def __init__(self, ctx: BackendContext | None = None) -> None:
        """``ctx`` is what the session hands the factory; None for a planner built by hand (a test)."""
        self.ctx = ctx
        self._on_log: Callable[[str, str], None] | None = ctx.on_log if ctx is not None else None

    # ---- the factory, which is the class itself -------------------------------------------------

    @classmethod
    def capabilities(cls) -> Capabilities:
        """The declaration. Cheap, static, and available before anything is warmed or even built."""
        return cls.CAPABILITIES

    @classmethod
    def create(cls, ctx: BackendContext) -> Planner:
        """The backend a session drives, not yet warmed, built with its options as ``validate_options`` left them."""
        return cls(replace(ctx, options=cls.validate_options(ctx.options)))

    @classmethod
    def validate_options(cls, options: Mapping[str, Any] | None) -> dict[str, Any]:
        """``planner.options`` as this planner reads them: every key checked, defaults filled in.

        Called when a profile naming the planner loads -- so a mistake is found when the profile is
        edited, not when a session starts -- and again by ``create``. What it returns is what the
        profile stores and what ``self.options`` is. The default refuses any key ``OPTIONS`` does not
        name and returns the rest unchanged. Override it for options with structure or types (a
        pydantic model is the natural tool) and raise ``TandemError``, or ``ValueError`` -- a pydantic
        ``ValidationError`` is one -- naming what is wrong. It must accept its own output unchanged:
        a saved profile is validated again when it is read back.
        """
        cls.check_options(options)
        return dict(options or {})

    @classmethod
    def runtime(cls, settings: Any = None) -> BackendRuntime | None:
        """This planner's runtime on this machine: built from ``recipe``, or None when there is none."""
        if cls.recipe is None:
            return None
        from tandem.planners.runtime import RecipeRuntime

        return RecipeRuntime(cls.recipe, cls.runtime_root(settings))

    @classmethod
    def runtime_root(cls, settings: Any = None) -> Path:
        """Where the runtime lives: ``<runtimes dir>/<planner name>`` unless a subclass says otherwise."""
        from tandem.planners.runtime import default_root

        return default_root(cls.info.name)

    @classmethod
    def describe_options(cls, profile: Any, *, settings: Any = None) -> OptionsView:
        """A profile's options as a person reads them. Default: each one, as it is set."""
        options = dict(getattr(getattr(profile, "planner", None), "options", None) or {})
        return OptionsView.generic(options, cls.OPTIONS)

    @classmethod
    def doctor_checks(cls, profile: Any, *, settings: Any = None, probe_hardware: bool = True) -> list:
        """Rows for `tandem doctor`, as ``tandem.core.probe.Check``: what this planner needs of the machine.

        ``profile`` is None for `tandem init`'s preflight, before a profile exists: check the machine
        only. ``probe_hardware`` False means touch nothing on the network or the bus (`--no-hardware`).
        A FAIL is something that stops a session; a WARN something that will cost one later. Default:
        nothing -- doctor already reports the planner's runtime, as it does for every planner.
        """
        return []

    @classmethod
    def replay(cls, rollout_dir: Path, *, settings: Any = None) -> None:
        """Open a leg this planner recorded in its own viewer. Unsupported unless implemented."""
        raise UnsupportedVerb(
            f"The {cls.info.title} planner has no viewer to replay a trajectory in.",
            hint="`tandem ui` shows every trajectory's cameras and robot state, whichever planner recorded it.",
        )

    @classmethod
    def check_options(cls, options: Mapping[str, Any] | None) -> None:
        """Refuse any ``planner.options`` key this planner does not declare in ``OPTIONS``."""
        unknown = sorted(str(key) for key in (options or {}) if key not in cls.OPTIONS)
        if not unknown:
            return
        title = cls.info.title if hasattr(cls, "info") else cls.__name__
        if not cls.OPTIONS:
            hint = f"The {title} planner reads no options. Remove planner.options from the profile."
        else:
            close = difflib.get_close_matches(unknown[0], list(cls.OPTIONS), n=1, cutoff=0.6)
            hint = (
                f"Did you mean {close[0]!r}?"
                if close
                else "It reads: " + "; ".join(f"{key} ({text})" for key, text in cls.OPTIONS.items()) + "."
            )
        raise TandemError(
            f"The {title} planner does not read planner.options {', '.join(unknown)}.",
            hint=hint,
        )

    # ---- conveniences for the implementation ----------------------------------------------------

    @property
    def options(self) -> dict[str, Any]:
        """The profile's ``planner.options``, as validated by ``create``."""
        return dict(self.ctx.options) if self.ctx is not None else {}

    @property
    def settings(self) -> Any:
        return self.ctx.settings if self.ctx is not None else None

    def log(self, text: str, *, stream: str = "tandem") -> None:
        """A line in the session's log -- what the operator sees -- or Python's logging outside a session."""
        if self._on_log is not None:
            self._on_log(stream, text)
        else:
            _log.info("%s: %s", stream, text)

    # ---- lifecycle: defaults for a planner that holds nothing ------------------------------------

    def require_ready(self) -> None:
        """Raise ``RuntimeNotReady`` if the declared runtime is not installed. No runtime: nothing to check."""
        self.check_runtime(type(self).runtime(self.settings))

    def check_runtime(self, runtime: Any) -> None:
        """Raise ``RuntimeNotReady`` naming what is missing from ``runtime``; None is always ready."""
        if runtime is None:
            return
        check = getattr(runtime, "require_ready", None)
        if callable(check):
            check()
            return
        status = runtime.status()
        if not status.installed:
            detail = "\n".join(f"  - {p}" for p in status.problems)
            raise RuntimeNotReady(
                f"The {self.info.title} runtime is not installed.\n{detail}".rstrip(),
                hint=f"Run `tandem planners install {self.info.name}`.",
            )

    def warm(self) -> None:
        """Open whatever the planner holds and build its solvers. Nothing, by default."""
        return None

    def close(self) -> None:
        """Release everything. Nothing, by default. Must be safe twice, and on a planner never warmed."""
        return None

    def release_hardware(self) -> None:
        """Hand the robot and cameras over, blocking until they are free. Nothing held, by default."""
        return None

    def reacquire_hardware(self) -> None:
        """Take the robot and cameras back. Nothing held, by default."""
        return None

    def home(self) -> None:
        """Park the arm, without opening the gripper. Nothing to park, by default."""
        return None

    def capture_frame(self, *, camera: str = "external") -> str:
        """One RGB frame written to a file; its path. Unsupported unless implemented.

        This is the image a human phase is verified from, and a precondition checked against. There
        is no default that would not be a lie: a verifier shown a stale or synthetic frame passes or
        fails a person's work on nothing.
        """
        raise UnsupportedVerb(
            f"The {self.info.title} planner cannot capture a camera frame: "
            f"{type(self).__qualname__}.capture_frame is not implemented.",
            hint="tandem needs one to verify a human phase (hitl.check_human_effects) and to check "
            "preconditions. Implement capture_frame(camera=...) to return an image path, or turn those "
            "checks off in the profile's hitl block.",
        )

    def move_to_joints(self, q: Sequence[float]) -> None:
        """Drive the arm to a joint configuration. Unsupported unless implemented.

        Not called by tandem today: it is reserved for executors that start a learned policy from a
        fixed pose. Declared here so such an executor fails with a clear message, not an AttributeError.
        """
        raise UnsupportedVerb(
            f"The {self.info.title} planner cannot move the arm to a joint configuration: "
            f"{type(self).__qualname__}.move_to_joints is not implemented.",
            hint="Implement move_to_joints(q), or use a human executor that does not need it.",
        )

    # ---- the planner itself ----------------------------------------------------------------------

    @abc.abstractmethod
    def perceive(
        self,
        *,
        task_hint: str,
        save_dir: Path,
        reset_arm: bool = True,
        open_gripper: bool = False,
    ) -> SceneView:
        """Look at the workspace and report what is in it.

        ``task_hint`` is the whole instruction, and steers DETECTION only: the goal arrives in
        ``plan``. ``reset_arm`` parks the arm first (an ordinary rollout); it is False for a phase
        resumed after a hand-off, when the arm is where a person left it. ``open_gripper`` opens the
        hand before looking, and nothing else -- the first robot leg after a human phase.

        Return every object's label (``object_labels``), the support surface's (``table_label``, not
        among the objects), the objects that are surfaces (``surface_labels``), a ``scene_id`` that
        ``plan`` will be handed back, and ``rgb_path``, an image of what was seen, when there is one.
        Labels may differ from one pass to the next; tandem re-binds its plan to them.
        """

    @abc.abstractmethod
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
        """Find a plan achieving ``goal`` (atoms in the wire spelling) in the scene ``scene_id`` named.

        A goal that cannot be planned is ``PlanResult(ok=False, failure_reason=...)``, not an
        exception. ``task_plan`` should list the operators the plan runs, object arguments only
        (``"Drop(apple, red_bin)"``): it is written into the rollout's record. ``plan_handle`` is
        handed back to ``execute`` unexamined.

        ``movables`` and ``return_home`` are passed only when ``CAPABILITIES`` declares
        ``supports_movable_restriction`` / ``supports_return_home``; a planner declaring neither
        may leave both out of its signature. When declared: only ``movables`` may be picked (None:
        no restriction), and a goal moving anything else is refused with ``ok=False`` naming it;
        ``return_home=False`` ends the plan where its last operation leaves the arm.
        """

    @abc.abstractmethod
    def execute(
        self,
        plan_handle: Any,
        leg: LegSpec,
        *,
        save_dir: Path,
        should_stop: Callable[[], bool] | None = None,
    ) -> ExecuteResult:
        """Run the plan, and record it as one leg of ``leg.trajectory_id``.

        THE RECORDING CONTRACT (``tandem.core.trajectories.is_complete``, and what ``tandem.core.merge``
        reads). When ``leg.record`` is set, ``save_dir`` must end up holding:

        - ``_meta.json`` with ``trajectory_id`` and ``segment_source`` copied from ``leg`` (this is how
          merging finds a task's legs), ``phase_index``, ``n_phases`` and ``phase_description`` when
          ``leg.phase_index`` is not None, ``record_start``/``record_stop`` (seconds; legs are ordered
          by them), ``fps``, and ``cameras``: a dataset key -> clip file name map;
        - ``robot_state.npz`` with every array in ``tandem.core.merge.STATE_KEYS``, one row per frame,
          plus optionally ``OPTIONAL_STATE_KEYS``, and nothing else;
        - the camera clips ``cameras`` names (or at least one of ``trajectories.CAMERA_FILES``).

        Stamp ``_meta.json`` even when execution fails part-way: a leg on disk without its
        trajectory id files as an episode of its own. Return ``rollout_dir`` (usually ``save_dir``)
        and ``n_frames``.

        ``should_stop`` is honoured only when ``CAPABILITIES.supports_cooperative_stop``: poll it at
        step boundaries, stop when it says so, and return ``stopped_early=True``. Otherwise ignore it.
        """


def _declaration_problems(cls: type[Planner], *, complete: bool) -> list[str]:
    """What is wrong with the declarations ``cls`` makes (or, when ``complete``, inherits)."""
    own = vars(cls)
    problems: list[str] = []

    def declared(attribute: str) -> bool:
        return attribute in own or (complete and hasattr(cls, attribute))

    if complete:
        missing = [a for a in ("info", "CAPABILITIES") if not hasattr(cls, a)]
        if missing:
            return [
                f"it declares no {' and no '.join(missing)} (a PlannerInfo and a Capabilities, as class "
                "attributes); pass abstract=True in the class statement if it is a base for other planners"
            ]
    info = getattr(cls, "info", None) if declared("info") else None
    caps = getattr(cls, "CAPABILITIES", None) if declared("CAPABILITIES") else None
    if declared("info"):
        problems += info_problems(info)
    if declared("CAPABILITIES"):
        problems += capability_problems(caps)
    if isinstance(info, PlannerInfo) and isinstance(caps, Capabilities) and caps.name != info.name:
        problems.append(
            f"CAPABILITIES.name is {caps.name!r} but info.name is {info.name!r}; a rollout's record and the "
            "catalog would call one planner by two names"
        )

    recipe = cls.recipe
    if "recipe" in own or (complete and recipe is not None):
        if recipe is not None and not isinstance(recipe, RuntimeRecipe):
            problems.append(f"recipe is a {type(recipe).__name__}, not a tandem.planners.RuntimeRecipe")
        elif recipe is not None and isinstance(info, PlannerInfo):
            if recipe.planner != info.name:
                problems.append(f"its recipe is for the planner {recipe.planner!r}, not {info.name!r}")
            if info.sources and tuple(info.sources) != recipe.pins:
                problems.append(
                    "info.sources differs from the commits its recipe fetches; leave info.sources empty and "
                    "it is filled in from the recipe"
                )

    options = cls.OPTIONS
    if not isinstance(options, Mapping) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in options.items()
    ):
        problems.append("OPTIONS must map each option name to one line describing it")

    name = own.get("name", "")
    if complete and name and isinstance(info, PlannerInfo) and name != info.name:
        problems.append(
            f"name is {name!r} but info.name is {info.name!r}; leave name out and it follows info"
        )
    return problems
