"""A conformance kit for planners: run the whole protocol against yours, from your own test suite.

tandem drives a planner through a protocol with a lot of fine print -- which keywords every verb
takes, what a scene must contain, that a plan result is JSON-safe, what a recorded leg must leave on
disk for merging to find it, that a stop is honoured when it is promised. Most of it fails quietly
when it is wrong: a leg with no trajectory id files as an episode of its own, a missing keyword is a
TypeError at the first robot phase after a human one. This kit checks all of it with no GPU, no robot
and no tandem session, so a planner's author finds out at ``pytest`` time.

The easy way in is to subclass the test class in your own suite::

    from tandem.planners.testing import PlannerConformance
    from my_planner import MyPlanner

    class TestMyPlanner(PlannerConformance):
        planner = MyPlanner              # a Planner subclass, any BackendFactory, or a registered name

pytest collects every ``test_*`` it inherits. Tune it with class attributes (``records_legs``,
``options``, ``task_hint``) and hooks (``goal`` to choose what is planned, ``make_backend`` to build
the backend some other way). Each check is also a plain function (``check_declarations``,
``check_protocol``, ``check_scene``, ``check_plan_result``, ``check_leg``, ``check_sidecar_script``)
raising ``ConformanceError`` -- an ``AssertionError`` listing every problem found, not just the first.

What the dynamic tests need from the planner is only that it runs where the tests run: an in-process
planner as it stands, a ``SidecarPlanner`` with its sidecar launchable (a fake world, or a real runtime
on a machine that has one). They build it through its factory with a ``BackendContext`` pointing into
pytest's ``tmp_path``, warm it, and always close it.

pytest is imported only by the test class's skips, so the check functions work without it.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import math
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from tandem.planners.base import (
    BackendContext,
    BackendFactory,
    Capabilities,
    ExecuteResult,
    GoalAtom,
    LegSpec,
    PlanResult,
    SceneView,
    TampBackend,
)
from tandem.planners.registry import _BACKEND_MEMBERS
from tandem.planners.sdk import Planner, UnsupportedVerb, factory_problems

# How a task_plan label reads: an operator applied to its object arguments, "Drop(apple, red_bin)".
_TASK_LABEL = re.compile(r"^[A-Za-z_][\w.-]*\((.*)\)$")
# The protocol keywords a backend may leave out, and the capability that says it will never be sent it.
_GATED_KEYWORDS = {
    "movables": "supports_movable_restriction",
    "return_home": "supports_return_home",
}
# Every member the session reaches for on a backend: the list the registry refuses a built backend
# for lacking. The kit says so before anyone registers it.
_MEMBERS = _BACKEND_MEMBERS


class ConformanceError(AssertionError):
    """A planner does not do what tandem's protocol says it must. Lists every problem found."""

    def __init__(self, what: str, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        super().__init__(f"{what}:\n" + "\n".join(f"  - {p}" for p in self.problems))


def _raise(what: str, problems: Sequence[str]) -> None:
    if problems:
        raise ConformanceError(what, problems)


# --------------------------------------------------------------------------- resolving what is tested


def as_factory(planner: Any) -> BackendFactory:
    """A ``Planner`` subclass, a factory (or factory class), or a registered name, as a factory."""
    if isinstance(planner, str):
        from tandem.planners import registry

        return registry.factory(planner)
    if isinstance(planner, type) and not issubclass(planner, Planner):
        return planner()
    return planner


def planner_title(factory: Any) -> str:
    info = getattr(factory, "info", None)
    return getattr(info, "title", None) or getattr(info, "name", None) or type(factory).__name__


# --------------------------------------------------------------------------- static checks


def check_declarations(planner: Any) -> None:
    """``info`` and ``capabilities()`` are sound and name the same planner. Builds nothing."""
    factory = as_factory(planner)
    _raise(f"{planner_title(factory)} declares itself wrongly", factory_problems(factory))


def check_protocol(backend: Any, caps: Capabilities | None = None) -> None:
    """Every member the session calls is there, taking every keyword the protocol names, with its default.

    ``backend`` is a built backend or its class. Method presence is all ``runtime_checkable`` checks;
    a missing keyword is a TypeError at the first call that passes it, which for ``open_gripper`` is
    the first robot phase after a human one. ``movables`` and ``return_home`` may be left out by a
    backend whose capabilities do not declare them, since tandem never passes them to it then.
    """
    problems: list[str] = []
    title = getattr(backend, "__name__", None) or type(backend).__name__
    missing = [member for member in _MEMBERS if not hasattr(backend, member)]
    if missing:
        problems.append(f"it has no {', '.join(missing)}")
    if caps is None:
        try:
            caps = backend.capabilities()
        except Exception as exc:
            problems.append(f"capabilities() raised {type(exc).__name__}: {exc}")
    if caps is not None and not isinstance(caps, Capabilities):
        problems.append(f"capabilities() returned a {type(caps).__name__}, not a Capabilities")
        caps = None

    for verb in ("perceive", "plan", "execute", "capture_frame"):
        if not hasattr(backend, verb):
            continue
        wanted = inspect.signature(getattr(TampBackend, verb)).parameters
        try:
            got = inspect.signature(getattr(backend, verb)).parameters
        except (TypeError, ValueError):
            problems.append(f"{verb} has no signature tandem can read")
            continue
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in got.values()):
            continue  # **kwargs takes every keyword; nothing more can be read off it
        positional = [
            p
            for p in got.values()
            if p.name != "self" and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        for index, (name, parameter) in enumerate((n, p) for n, p in wanted.items() if n != "self"):
            gate = _GATED_KEYWORDS.get(name)
            if gate is not None and caps is not None and not getattr(caps, gate):
                continue
            if name not in got:
                problems.append(f"{verb} does not take {name}")
                continue
            if parameter.default is not inspect.Parameter.empty and got[name].default != parameter.default:
                problems.append(
                    f"{verb}({name}=...) defaults to {got[name].default!r}, not {parameter.default!r}"
                )
            if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                # tandem passes these positionally: plan(scene_id, goal, ...), execute(plan_handle, leg, ...)
                if index >= len(positional) or positional[index].name != name:
                    problems.append(f"{verb} must take {name} as positional argument {index + 1}")
    _raise(f"{title} does not implement the planner protocol", problems)


def check_scene(scene: Any) -> None:
    """A perception pass reported what the phase planner can use."""
    problems: list[str] = []
    if not isinstance(scene, SceneView):
        _raise(
            "perceive() returned something that is not a scene",
            [f"it is a {type(scene).__name__}, not a SceneView"],
        )
    labels = scene.object_labels
    if not isinstance(labels, tuple) or not all(isinstance(label, str) and label for label in labels):
        problems.append(f"object_labels must be a tuple of non-empty strings, not {labels!r}")
        labels = ()
    duplicated = sorted({label for label in labels if labels.count(label) > 1})
    if duplicated:
        problems.append(
            f"object_labels names {', '.join(duplicated)} more than once; a goal could not say which"
        )
    if not isinstance(scene.table_label, str) or not scene.table_label:
        problems.append("table_label is empty")
    elif scene.table_label in labels:
        problems.append(
            f"table_label {scene.table_label!r} is also an object label; report it once, as the table"
        )
    stray = sorted(set(scene.surface_labels) - set(labels) - {scene.table_label})
    if stray:
        problems.append(f"surface_labels names {', '.join(stray)}, which this pass did not report as objects")
    if not isinstance(scene.scene_id, str) or not scene.scene_id:
        problems.append("scene_id is empty, so plan() cannot be told which pass to plan in")
    if scene.rgb_path is not None and not Path(scene.rgb_path).is_file():
        problems.append(f"rgb_path {scene.rgb_path!r} is not a file; the verifier would be shown nothing")
    for atom in scene.detected_goal:
        if not (
            isinstance(atom, GoalAtom)
            and isinstance(atom.predicate, str)
            and all(isinstance(a, str) for a in atom.args)
        ):
            problems.append(f"detected_goal holds {atom!r}, which is not a GoalAtom of strings")
    _raise("perceive() reported a malformed scene", problems)


def check_plan_result(result: Any) -> None:
    """A plan result tandem can record: plain types throughout, a reason when it failed, labelled operators."""
    if not isinstance(result, PlanResult):
        _raise(
            "plan() returned something that is not a plan result",
            [f"it is a {type(result).__name__}, not a PlanResult"],
        )
    problems: list[str] = []
    if not isinstance(result.ok, bool):
        problems.append(f"ok is {result.ok!r}, not True or False")
    if not result.ok and not (isinstance(result.failure_reason, str) and result.failure_reason.strip()):
        problems.append("a failed plan must say why (failure_reason); the operator is told it")
    if result.failure_reason is not None and not isinstance(result.failure_reason, str):
        problems.append("failure_reason is not a string")
    seconds = result.planning_seconds
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(seconds)
        or seconds < 0
    ):
        problems.append(f"planning_seconds is {seconds!r}, not a non-negative number")
    if result.ok and result.plan_handle is None:
        problems.append("an ok plan has no plan_handle, so there is nothing to hand execute()")
    if not isinstance(result.artifacts, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in result.artifacts.items()
    ):
        problems.append("artifacts must map a role to a path, both strings")
    if not isinstance(result.task_plan, tuple):
        problems.append(f"task_plan is a {type(result.task_plan).__name__}, not a tuple of labels")
    else:
        for label in result.task_plan:
            if not isinstance(label, str) or not _TASK_LABEL.match(label):
                problems.append(
                    f"task_plan label {label!r} is not an operator over its objects, such as 'Pick(bread)'"
                )
    try:
        json.dumps(
            {
                "ok": result.ok,
                "failure_reason": result.failure_reason,
                "planning_seconds": result.planning_seconds,
                "artifacts": result.artifacts,
                "task_plan": list(result.task_plan),
            }
        )
    except (TypeError, ValueError) as exc:
        problems.append(f"it does not serialise to JSON, and it is written into the rollout's record: {exc}")
    _raise("plan() returned a result tandem cannot record", problems)


def check_leg(result: Any, leg: LegSpec, save_dir: Path, *, records: bool = True) -> None:
    """``execute`` stamped its leg so merging can find it -- and, when ``records``, recorded it completely.

    The stamp: ``_meta.json`` carrying ``leg``'s trajectory id, segment source and phase. The
    recording contract (``tandem.core.trajectories.is_complete``): ``_meta.json`` with
    ``record_start``/``record_stop``/``fps``, ``robot_state.npz`` with exactly the arrays
    ``tandem.core.merge`` joins, all one row per frame, and the camera clips ``_meta.json`` names.
    """
    if not isinstance(result, ExecuteResult):
        _raise(
            "execute() returned something that is not a result",
            [f"it is a {type(result).__name__}, not an ExecuteResult"],
        )
    problems: list[str] = []
    if not isinstance(result.ok, bool) or not isinstance(result.stopped_early, bool):
        problems.append("ok and stopped_early must be True or False")
    if isinstance(result.n_frames, bool) or not isinstance(result.n_frames, int) or result.n_frames < 0:
        problems.append(f"n_frames is {result.n_frames!r}, not a count")
    if not result.ok and not result.failure_reason:
        problems.append("a failed execution must say why (failure_reason)")
    directory = Path(result.rollout_dir) if result.rollout_dir else Path(save_dir)
    meta_path = directory / "_meta.json"
    meta: dict = {}
    if not meta_path.is_file():
        problems.append(
            f"it left no _meta.json in {directory}, so merging cannot tell which task the leg belongs to"
        )
    else:
        try:
            meta = json.loads(meta_path.read_text())
        except ValueError as exc:
            problems.append(f"_meta.json is not JSON: {exc}")
    if meta:
        expected = {"trajectory_id": leg.trajectory_id, "segment_source": leg.segment_source}
        if leg.phase_index is not None:
            expected.update(
                phase_index=leg.phase_index, n_phases=leg.n_phases, phase_description=leg.phase_description
            )
        for key, value in expected.items():
            if meta.get(key) != value:
                problems.append(f"_meta.json has {key}={meta.get(key)!r}; the leg asked for {value!r}")

    if records and meta:
        problems += _recording_problems(result, directory, meta)
    _raise("execute() did not leave the leg tandem asked for", problems)


def _recording_problems(result: ExecuteResult, directory: Path, meta: dict) -> list[str]:
    import numpy as np

    from tandem.core import merge, trajectories

    problems: list[str] = []
    if result.n_frames <= 0:
        problems.append("a recorded leg reports n_frames=0")
    for key in ("record_start", "record_stop", "fps"):
        value = meta.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"_meta.json has no numeric {key}; merging orders and times legs by it")
    start, stop = meta.get("record_start"), meta.get("record_stop")
    if isinstance(start, (int, float)) and isinstance(stop, (int, float)) and stop < start:
        problems.append("record_stop is before record_start")
    state_path = directory / trajectories.STATE_FILE
    if state_path.is_file():
        try:
            with np.load(state_path) as store:
                arrays = {key: store[key] for key in store.files}
        except Exception as exc:
            problems.append(f"{trajectories.STATE_FILE} cannot be read: {type(exc).__name__}: {exc}")
            arrays = {}
        absent = [key for key in merge.STATE_KEYS if key not in arrays]
        if absent:
            problems.append(f"{trajectories.STATE_FILE} lacks {', '.join(absent)}")
        extra = sorted(set(arrays) - set(merge.STATE_KEYS) - set(merge.OPTIONAL_STATE_KEYS))
        if extra:
            problems.append(f"{trajectories.STATE_FILE} carries {', '.join(extra)}, which merging refuses")
        lengths = {key: len(value) for key, value in arrays.items() if getattr(value, "ndim", 0) >= 1}
        if len(set(lengths.values())) > 1:
            problems.append(f"the arrays in {trajectories.STATE_FILE} are not one row per frame: {lengths}")
        if arrays and not any(lengths.values()):
            problems.append(f"{trajectories.STATE_FILE} holds no frames")
    if not trajectories.is_complete(directory, meta):
        reasons = []
        if not state_path.is_file():
            reasons.append(f"there is no {trajectories.STATE_FILE}")
        named = meta.get("cameras") if isinstance(meta.get("cameras"), dict) else {}
        if named:
            missing = sorted(str(name) for name in named.values() if not (directory / str(name)).is_file())
            if missing:
                reasons.append(f"the clips _meta.json names are missing: {', '.join(missing)}")
        elif not any((directory / name).is_file() for name in trajectories.CAMERA_FILES):
            reasons.append(
                f"there is no camera clip (_meta.json names none, and none of {', '.join(trajectories.CAMERA_FILES)})"
            )
        problems.append(
            "the leg does not meet the recording contract (tandem.core.trajectories.is_complete): "
            + "; ".join(reasons or ["see is_complete"])
        )
    return problems


def check_sidecar_script(planner: Any) -> None:
    """A ``SidecarPlanner``'s script can run where tandem is not, and keeps its stdout clean.

    It must not import ``tandem`` (the planner's environment has none), must import
    ``tandem_sidecar``, and must import it before any module-level import that is not the standard
    library -- the kit takes stdout for the protocol when it is imported, and a library imported first
    may already have printed into it.
    """
    import ast

    from tandem.planners.sidecar import SidecarPlanner

    if not (isinstance(planner, type) and issubclass(planner, SidecarPlanner)):
        _raise("not a sidecar planner", [f"{planner!r} is not a SidecarPlanner subclass"])
    script = planner.sidecar_script()
    problems: list[str] = []
    try:
        tree = ast.parse(script.read_text(), filename=str(script))
    except (OSError, SyntaxError) as exc:
        _raise(f"the sidecar script {script} cannot be read", [f"{type(exc).__name__}: {exc}"])

    imported: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [(node.lineno, alias.name.split(".")[0]) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.append((node.lineno, node.module.split(".")[0]))
    if any(root == "tandem" for _, root in imported):
        problems.append("it imports tandem, which the planner's environment does not have")
    kit_lines = [line for line, root in imported if root == "tandem_sidecar"]
    if not kit_lines:
        problems.append(
            "it does not import tandem_sidecar, so nothing speaks tandem's protocol or guards its stdout"
        )
    else:
        stdlib = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}
        top_level = [
            (node.lineno, root)
            for node in tree.body
            for root in (
                [alias.name.split(".")[0] for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module.split(".")[0]]
                if isinstance(node, ast.ImportFrom) and node.module and not node.level
                else []
            )
        ]
        early = sorted({root for line, root in top_level if line < min(kit_lines) and root not in stdlib})
        if early:
            problems.append(
                f"it imports {', '.join(early)} before tandem_sidecar; anything such a module prints on import "
                "goes into the protocol stream"
            )
    _raise(f"the sidecar script {script.name} cannot be run by tandem", problems)


# --------------------------------------------------------------------------- building a goal


def default_goal(scene: SceneView, caps: Capabilities) -> list[GoalAtom] | None:
    """One goal atom the planner should be able to plan in ``scene``, or None when none can be formed.

    The first goal predicate on the wire -- one with a moved argument first, so the movable
    restriction has something to restrict -- with its surface-typed arguments filled from the
    scene's surfaces (or the table) and the rest from its other objects.
    """
    objects = [label for label in scene.object_labels if label not in scene.surface_labels]
    surfaces = sorted(scene.surface_labels) or [scene.table_label]
    ordered = sorted(caps.goal_predicates.items(), key=lambda item: item[0] not in caps.moved_arguments)
    for name, predicate in ordered:
        wire = caps.goal_predicate_wire_names.get(name)
        if wire is None or predicate.arity == 0:
            continue
        args: list[str] = []
        for parameter in predicate.parameters:
            pool = surfaces if parameter.type == caps.surface_type else objects
            choice = next((label for label in pool if label not in args), None)
            if choice is None:
                break
            args.append(choice)
        else:
            return [GoalAtom(wire, tuple(args))]
    return None


def moved_objects(goal: Sequence[GoalAtom], caps: Capabilities) -> set[str]:
    """The objects ``goal`` asks the robot to move, read through ``moved_arguments``."""
    by_wire = {wire: name for name, wire in caps.goal_predicate_wire_names.items()}
    moved: set[str] = set()
    for atom in goal:
        position = caps.moved_arguments.get(by_wire.get(atom.predicate, ""))
        if position is not None and position < len(atom.args):
            moved.add(atom.args[position])
    return moved


# --------------------------------------------------------------------------- the test class


def _skip(reason: str) -> None:
    import pytest

    pytest.skip(reason)


class PlannerConformance:
    """The whole protocol as pytest tests. Subclass it as ``Test<Something>`` and set ``planner``.

    Every test builds a fresh backend through the factory, warms it, and closes it however the test
    ends. Nothing here needs a GPU, a robot or a network -- only whatever the planner itself needs
    to run where the tests do.
    """

    #: What is tested: a Planner subclass, a BackendFactory (or its class), or a registered name.
    planner: Any = None
    #: Whether ``execute`` records legs completely (state, video, timing). False for a planner that
    #: only stamps ``_meta.json`` -- a stand-in with nothing to record; then only the stamp is checked.
    records_legs: bool = True
    #: The ``planner.options`` the backend is built with.
    options: Mapping[str, Any] = {}
    #: The instruction handed to ``perceive`` as its detection hint.
    task_hint: str = "put one thing where it belongs"

    # ---- hooks ---------------------------------------------------------------------------------

    def factory(self) -> BackendFactory:
        if self.planner is None:
            raise AssertionError(f"{type(self).__name__} sets no `planner` to test")
        return as_factory(self.planner)

    def context(self, tmp_path: Path) -> BackendContext:
        """The context the backend is built from: every directory under ``tmp_path``."""
        self.logs: list[tuple[str, str]] = []
        session_dir = tmp_path / "session"
        session_dir.mkdir(parents=True, exist_ok=True)
        return BackendContext(
            profile=None,
            session_dir=session_dir,
            output_dir=tmp_path / "legs",
            execute=True,
            record=True,
            on_log=lambda stream, text: self.logs.append((stream, text)),
            options=dict(self.options),
            session_id="conformance",
            task=self.task_hint,
            events_file=session_dir / "events.jsonl",
        )

    def make_backend(self, tmp_path: Path) -> Any:
        return self.factory().create(self.context(tmp_path))

    def goal(self, scene: SceneView, caps: Capabilities) -> list[GoalAtom]:
        """What to plan in ``scene``. Override when ``default_goal`` cannot guess it."""
        goal = default_goal(scene, caps)
        if goal is None:
            raise AssertionError(
                f"the kit could not form a goal from this scene ({scene.object_labels}, surfaces "
                f"{sorted(scene.surface_labels)}); override {type(self).__name__}.goal()"
            )
        return goal

    def leg(self, name: str = "leg") -> LegSpec:
        return LegSpec(
            trajectory_id=f"conformance-{name}",
            instruction=self.task_hint,
            segment_source="tamp",
            phase_index=1,
            n_phases=3,
            phase_description="the robot's phase, as the phase planner wrote it",
            record=True,
        )

    @contextlib.contextmanager
    def warmed(self, tmp_path: Path) -> Iterator[Any]:
        backend = self.make_backend(tmp_path)
        try:
            backend.require_ready()
            backend.warm()
            yield backend
        finally:
            backend.close()

    def perceived(self, backend: Any, tmp_path: Path) -> tuple[SceneView, list[GoalAtom]]:
        scene = backend.perceive(task_hint=self.task_hint, save_dir=tmp_path / "perceive")
        check_scene(scene)
        return scene, self.goal(scene, backend.capabilities())

    # ---- declarations ----------------------------------------------------------------------------

    def test_it_declares_what_it_is(self) -> None:
        check_declarations(self.factory())

    def test_it_takes_every_keyword_the_protocol_names(self, tmp_path: Path) -> None:
        backend = self.make_backend(tmp_path)
        try:
            check_protocol(backend)
        finally:
            backend.close()

    def test_the_backend_declares_what_its_factory_does(self, tmp_path: Path) -> None:
        # Declared, not discovered: the phase planner reads the factory's copy on a laptop and the
        # backend's in a session, and the two must be the same statement.
        backend = self.make_backend(tmp_path)
        try:
            assert backend.capabilities() == self.factory().capabilities()
            assert isinstance(backend.name, str) and backend.name
        finally:
            backend.close()

    def test_its_options_check_accepts_what_it_returns(self) -> None:
        # A profile stores what validate_options returned and validates it again every time it is
        # read, so a check that refuses (or changes) its own output makes a profile unloadable the
        # first time it is saved.
        from tandem.planners.registry import options_for

        factory = self.factory()
        checked = options_for(factory, self.options)
        again = options_for(factory, checked)
        if again != checked:
            _raise(
                "its validate_options",
                [f"given its own output {checked!r} it returned {again!r}; it must accept it unchanged"],
            )
        json.dumps(checked, default=str)

    def test_its_doctor_checks_are_doctor_rows(self) -> None:
        # With no profile, as `tandem init` asks before one exists, and touching no hardware.
        from tandem.core import probe

        hook = getattr(self.factory(), "doctor_checks", None)
        if not callable(hook):
            _skip(f"{planner_title(self.factory())} adds nothing to `tandem doctor`")
        rows = hook(None, settings=None, probe_hardware=False)
        bad = [type(row).__name__ for row in rows if not isinstance(row, probe.Check)]
        if bad:
            _raise("its doctor_checks", [f"returned {', '.join(sorted(set(bad)))}, not tandem.core.probe.Check"])

    def test_its_sidecar_can_run_without_tandem(self) -> None:
        from tandem.planners.sidecar import SidecarPlanner

        factory = self.factory()
        if not (isinstance(factory, type) and issubclass(factory, SidecarPlanner)):
            _skip(f"{planner_title(factory)} runs in tandem's process; it has no sidecar script")
        check_sidecar_script(factory)

    # ---- lifecycle -------------------------------------------------------------------------------

    def test_it_closes_twice_and_before_it_ever_warmed(self, tmp_path: Path) -> None:
        backend = self.make_backend(tmp_path)
        backend.close()
        backend.close()
        with self.warmed(tmp_path / "again") as warm:
            warm.close()  # and once more by warmed(), on the way out

    # ---- the sub-goal cycle ------------------------------------------------------------------------

    def test_perception_reports_a_scene(self, tmp_path: Path) -> None:
        with self.warmed(tmp_path) as backend:
            check_scene(backend.perceive(task_hint=self.task_hint, save_dir=tmp_path / "first"))
            # The pass after a human phase: the arm where a person left it, the hand opened first.
            check_scene(
                backend.perceive(
                    task_hint=self.task_hint, save_dir=tmp_path / "after", reset_arm=False, open_gripper=True
                )
            )

    def test_a_plan_is_json_safe_and_says_what_it_runs(self, tmp_path: Path) -> None:
        with self.warmed(tmp_path) as backend:
            scene, goal = self.perceived(backend, tmp_path)
            result = backend.plan(
                scene.scene_id, goal, surfaces=frozenset(scene.surface_labels), save_dir=tmp_path / "plan"
            )
            check_plan_result(result)
            assert result.ok, (
                f"the planner could not plan {[a.to_dict() for a in goal]}: {result.failure_reason}"
            )

    def test_an_executed_plan_is_recorded_as_the_leg_it_was_asked_for(self, tmp_path: Path) -> None:
        with self.warmed(tmp_path) as backend:
            scene, goal = self.perceived(backend, tmp_path)
            result = backend.plan(
                scene.scene_id, goal, surfaces=frozenset(scene.surface_labels), save_dir=tmp_path / "plan"
            )
            assert result.ok, result.failure_reason
            leg, save_dir = self.leg(), tmp_path / "leg"
            # should_stop is always passed here and never says stop: every backend must take it.
            executed = backend.execute(result.plan_handle, leg, save_dir=save_dir, should_stop=lambda: False)
            assert executed.ok, executed.failure_reason
            assert not executed.stopped_early
            check_leg(executed, leg, save_dir, records=self.records_legs)

    def test_a_stop_is_honoured_when_promised(self, tmp_path: Path) -> None:
        with self.warmed(tmp_path) as backend:
            caps = backend.capabilities()
            scene, goal = self.perceived(backend, tmp_path)
            result = backend.plan(
                scene.scene_id, goal, surfaces=frozenset(scene.surface_labels), save_dir=tmp_path / "plan"
            )
            assert result.ok, result.failure_reason
            asked: list[bool] = []

            def should_stop() -> bool:
                asked.append(True)
                return True

            leg, save_dir = self.leg("stopped"), tmp_path / "stopped"
            executed = backend.execute(result.plan_handle, leg, save_dir=save_dir, should_stop=should_stop)
            if caps.supports_cooperative_stop:
                assert asked, "supports_cooperative_stop is declared but should_stop was never asked"
                assert executed.stopped_early, (
                    "should_stop said stop from the start, and the leg ran to the end"
                )
                if (Path(executed.rollout_dir or save_dir) / "_meta.json").is_file():
                    check_leg(executed, leg, save_dir, records=False)
            else:
                # Declared as not honoured: a preempt is an abort, and the leg runs as though unasked.
                assert executed.ok, executed.failure_reason

    def test_a_leg_picks_only_what_it_is_allowed_to(self, tmp_path: Path) -> None:
        with self.warmed(tmp_path) as backend:
            caps = backend.capabilities()
            if not caps.supports_movable_restriction:
                _skip(f"{backend.name} does not declare supports_movable_restriction; it is never restricted")
            scene, goal = self.perceived(backend, tmp_path)
            moved = moved_objects(goal, caps)
            if not moved:
                _skip("the goal moves nothing the kit can name through moved_arguments")
            surfaces = frozenset(scene.surface_labels)
            allowed = backend.plan(
                scene.scene_id, goal, surfaces=surfaces, movables=frozenset(moved), save_dir=tmp_path / "a"
            )
            check_plan_result(allowed)
            assert allowed.ok, f"a leg allowed to pick {sorted(moved)} could not: {allowed.failure_reason}"
            refused = backend.plan(
                scene.scene_id, goal, surfaces=surfaces, movables=frozenset(), save_dir=tmp_path / "r"
            )
            check_plan_result(refused)
            assert not refused.ok, "a leg allowed to pick nothing planned a goal that moves something"
            for obj in moved:
                assert obj in (refused.failure_reason or ""), (
                    f"the refusal does not name {obj}: {refused.failure_reason}"
                )

    def test_a_leg_that_does_not_go_home_still_plans_and_runs(self, tmp_path: Path) -> None:
        with self.warmed(tmp_path) as backend:
            if not backend.capabilities().supports_return_home:
                _skip(f"{backend.name} does not declare supports_return_home; every leg goes home")
            scene, goal = self.perceived(backend, tmp_path)
            result = backend.plan(
                scene.scene_id,
                goal,
                surfaces=frozenset(scene.surface_labels),
                return_home=False,
                save_dir=tmp_path / "plan",
            )
            check_plan_result(result)
            assert result.ok, result.failure_reason
            leg, save_dir = self.leg("stays"), tmp_path / "stays"
            executed = backend.execute(result.plan_handle, leg, save_dir=save_dir)
            assert executed.ok, executed.failure_reason
            check_leg(executed, leg, save_dir, records=self.records_legs)

    # ---- hardware custody ------------------------------------------------------------------------

    def test_hardware_is_handed_over_and_back_idempotently(self, tmp_path: Path) -> None:
        with self.warmed(tmp_path) as backend:
            # The session releases around every human phase and may retry a release that timed out,
            # so twice has to be the same as once, in both directions.
            backend.release_hardware()
            backend.release_hardware()
            backend.reacquire_hardware()
            backend.reacquire_hardware()
            check_scene(
                backend.perceive(task_hint=self.task_hint, save_dir=tmp_path / "after", reset_arm=False)
            )
            backend.home()

    def test_a_frame_for_the_verifier_is_an_image_on_disk(self, tmp_path: Path) -> None:
        with self.warmed(tmp_path) as backend:
            try:
                path = backend.capture_frame(camera="external")
            except UnsupportedVerb:
                _skip(f"{backend.name} cannot capture a frame, so human phases cannot be verified with it")
            assert isinstance(path, str) and Path(path).is_file(), (
                f"capture_frame returned {path!r}, not an image file"
            )
