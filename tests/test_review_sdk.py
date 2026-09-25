"""What the planner SDK's review found it let through, pinned so it cannot let it through again.

Each test here is a planner, a leg or a plugin that tandem used to accept and then trip over later --
at a proposal, a merge, an export, a listing -- far from the declaration that caused it:

- a goal predicate over a third object type, which no atom could ever be grounded against;
- Capabilities defaults that were TiPToP's promises, so a planner that said nothing made them;
- a conformance kit that certified a planner with no perception image and no camera frame, a leg
  with no language label, and a 6-joint arm the export then crashed on;
- a plugin that exits at import, which took the whole planner listing down;
- an editable install made while `tandem ui` runs, listed as broken with the wrong advice;
- a sidecar planner defined in a notebook, whose script was looked for one directory up.

The process-level findings (a sidecar's pipes, its process group, its crashes) are in
``test_review_sdk_process.py``; what a planner that raises does to a trial, in
``test_review_sdk_loop.py``.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import site
import sys
import textwrap
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from helpers import isolate_registry
from toy_planner import TOY_CAPABILITIES, ToyPlanner

from tandem.cli import planners as planners_cli
from tandem.core import merge
from tandem.core.errors import TandemError
from tandem.executors import base as executors
from tandem.planners import (
    Capabilities,
    ExecuteResult,
    LegSpec,
    Parameter,
    Planner,
    PlannerInfo,
    Predicate,
    SceneView,
    registry,
)
from tandem.planners.sdk import UnsupportedVerb, capability_problems
from tandem.planners.testing import ConformanceError, PlannerConformance, check_leg, check_scene
from tandem.planners.tiptop.capabilities import CAPABILITIES as TIPTOP_CAPABILITIES
from tandem.planning import feasibility
from tandem.planning.structs import Phase
from tandem.planning.symbols import Atom


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    isolate_registry(monkeypatch)


def _problems(excinfo) -> str:
    return "\n".join(excinfo.value.problems)


# --- a goal predicate over a type no object has -----------------------------------------------------------

IN = Predicate("In", (Parameter("obj", "movable"), Parameter("box", "container")))
THIRD_TYPE = Capabilities(
    name="boxes",
    goal_predicates={"In": IN},
    robot_description="put a thing in a box",
    goal_predicate_wire_names={"In": "in"},
    achievable_predicates=frozenset({"In"}),
    reserved_predicate_names=frozenset({"In"}),
    movable_type="movable",
    surface_type="surface",
    moved_arguments={"In": 0},
)


def test_a_goal_predicate_over_a_third_type_is_refused_when_the_class_is_defined():
    # tandem types every perceived object as movable or surface, so In(?obj, ?box: container) could
    # never be grounded: every proposal using it refused, every trial ending at invention.
    problems = capability_problems(THIRD_TYPE)
    assert any("?box is a 'container'" in p and "could ever be grounded" in p for p in problems), problems

    with pytest.raises(TandemError, match=r"In's parameter \?box is a 'container'"):

        class Boxes(ToyPlanner):
            info = PlannerInfo(name="boxes")
            CAPABILITIES = THIRD_TYPE


def test_a_robot_operator_over_a_third_type_is_refused_too():
    caps = replace(TOY_CAPABILITIES, robot_operators=("Drop(?obj: item, ?bin: box)",))
    assert any("uses the type 'box'" in p for p in capability_problems(caps))
    # A goal predicate's own type used to be accepted here even when it was neither of the two.
    caps = replace(THIRD_TYPE, robot_operators=("Put(?obj: movable, ?box: container)",))
    assert any("uses the type 'container'" in p for p in capability_problems(caps))


def test_the_shipped_declarations_use_only_the_two_types():
    assert capability_problems(TIPTOP_CAPABILITIES) == []
    assert capability_problems(TOY_CAPABILITIES) == []


# --- Capabilities that said nothing ------------------------------------------------------------------------

AT = Predicate("At", (Parameter("obj", "movable"), Parameter("place", "surface")))


def _robot(obj: str, place: str) -> Phase:
    return Phase("robot", f"put {obj} at {place}", frozenset({Atom("At", (obj, place))}))


def test_a_declaration_that_states_no_flag_plans_every_robot_phase_as_its_own_leg():
    # initial_state_is_clean used to default to True -- cuTAMP's promise, made for every planner that
    # left it out -- and consecutive robot phases were conjoined into one goal with their order sorted
    # away. Unstated, it is now the safe reading.
    bare = Capabilities(name="bare", goal_predicates={"At": AT}, moved_arguments={"At": 0})
    assert bare.initial_state_is_clean is False
    assert bare.one_pick_per_object is True, "kept: True only ever splits a run"
    assert feasibility.conjoinable_run([_robot("mug", "tray"), _robot("block", "slot")], bare) == 1
    assert feasibility.conjoinable_run(
        [_robot("mug", "tray"), _robot("block", "slot")], replace(bare, initial_state_is_clean=True)
    ) == 2, "a planner that makes the promise still gets the conjoining"


def test_a_planner_that_does_not_say_what_its_robot_does_is_refused():
    # It used to default to TiPToP's sentence: the proposer was told any robot picks and places.
    assert Capabilities(name="bare").robot_description == ""
    assert any("robot_description is empty" in p for p in capability_problems(Capabilities(name="bare")))
    with pytest.raises(TandemError, match="robot_description is empty"):

        class Silent(ToyPlanner):
            info = PlannerInfo(name="toy")
            CAPABILITIES = replace(TOY_CAPABILITIES, robot_description="")


@pytest.mark.parametrize("sidecar", [False, True], ids=["in-process", "sidecar"])
def test_a_scaffolded_planner_states_what_its_solver_assumes(tmp_path, sidecar):
    package = tmp_path / "pkg"
    planners_cli.scaffold("assumer", package, sidecar=sidecar)
    source = package / "src" / "tandem_assumer" / "planner.py"
    text = source.read_text()
    assert "initial_state_is_clean=False" in text and "one_pick_per_object=True" in text

    name = f"_scaffolded_assumer_{int(sidecar)}"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    caps = module.CAPABILITIES
    assert caps.initial_state_is_clean is False
    # Its stand-in world keeps where things are between legs: two robot phases are two legs.
    assert feasibility.conjoinable_run([_robot("block", "tray"), _robot("cup", "table")], caps) == 1


# --- the conformance kit: what phase planning needs ---------------------------------------------------------


class _Blind(ToyPlanner):
    """Perceives, plans and executes, and reports no image of what it saw."""

    def perceive(self, *, task_hint, save_dir, reset_arm=True, open_gripper=False) -> SceneView:
        return replace(
            super().perceive(
                task_hint=task_hint, save_dir=save_dir, reset_arm=reset_arm, open_gripper=open_gripper
            ),
            rgb_path=None,
        )


class _NoCamera(ToyPlanner):
    """Everything but a frame for the verifier: the SDK's default, which refuses."""

    def capture_frame(self, *, camera: str = "external") -> str:
        return Planner.capture_frame(self, camera=camera)


class _NotAnImage(ToyPlanner):
    """A 'frame' that is a file, and not an image."""

    def capture_frame(self, *, camera: str = "external") -> str:
        path = Path(self.world._frames) / "frame.png"
        path.write_text("not an image")
        return str(path)


def _suite(planner, **attributes):
    return type("Suite", (PlannerConformance,), {"planner": planner, **attributes})()


def test_a_planner_with_no_perception_image_fails_the_kit_unless_it_opts_out(tmp_path):
    with pytest.raises(ConformanceError, match="rgb_path is None: phase planning decomposes"):
        _suite(_Blind).test_perception_reports_a_scene(tmp_path / "a")
    with pytest.raises(ConformanceError, match="rgb_path is None"):
        _suite(_Blind).test_a_plan_is_json_safe_and_says_what_it_runs(tmp_path / "b")
    # Opted out, visibly: a planner never run with phase planning.
    _suite(_Blind, phase_planning=False).test_perception_reports_a_scene(tmp_path / "c")


def test_a_perception_image_that_is_not_an_image_is_caught(tmp_path):
    text = tmp_path / "rgb.png"
    text.write_text("a log line, not a photo")
    scene = SceneView(object_labels=("apple",), table_label="floor", rgb_path=str(text), scene_id="s")
    check_scene(scene)  # without require_image, only that the file exists
    with pytest.raises(ConformanceError, match="is not an image"):
        check_scene(scene, require_image=True)


def test_a_planner_with_no_camera_frame_fails_the_kit_unless_it_opts_out(tmp_path):
    # A skip is not an answer here: it is what the kit used to give, and it certified the planner.
    try:
        with pytest.raises(ConformanceError) as excinfo:
            _suite(_NoCamera).test_a_frame_for_the_verifier_is_an_image_on_disk(tmp_path / "a")
    except pytest.skip.Exception as skipped:
        pytest.fail(f"the kit skipped a planner that cannot capture a frame: {skipped}")
    problems = _problems(excinfo)
    assert "hitl.check_human_effects" in problems and "hitl.verify_final_phase" in problems
    assert "verifies_human_phases = False" in problems

    with pytest.raises(pytest.skip.Exception, match="cannot capture a frame"):
        _suite(_NoCamera, verifies_human_phases=False).test_a_frame_for_the_verifier_is_an_image_on_disk(
            tmp_path / "b"
        )
    with pytest.raises(ConformanceError, match="is not an image"):
        _suite(_NotAnImage).test_a_frame_for_the_verifier_is_an_image_on_disk(tmp_path / "c")


def test_the_sdk_default_capture_frame_is_what_the_kit_refuses():
    with pytest.raises(UnsupportedVerb):
        _NoCamera().capture_frame()


# --- the conformance kit: what the dataset needs ------------------------------------------------------------


def _leg(**overrides) -> LegSpec:
    return LegSpec(
        **{
            "trajectory_id": "t-1",
            "instruction": "put the apple in the red bin",
            "segment_source": "tamp",
            "phase_index": 0,
            "n_phases": 2,
            "phase_description": "drop it",
            **overrides,
        }
    )


def _recorded(directory: Path, *, widths: int = 7, frame_time=np.float64, **meta_overrides) -> ExecuteResult:
    directory.mkdir(parents=True, exist_ok=True)
    frames = 10
    arrays = {key: np.zeros(frames) for key in merge.STATE_KEYS}
    for key in ("joint_position", "cmd_joint_position", "cmd_joint_velocity"):
        arrays[key] = np.zeros((frames, widths))
    arrays["frame_time"] = (1.8e9 + np.arange(frames) / 15).astype(frame_time)
    np.savez(directory / "robot_state.npz", **arrays)
    (directory / "external_cam.mp4").write_bytes(b"clip")
    meta = {
        "trajectory_id": "t-1",
        "segment_source": "tamp",
        "instruction": "put the apple in the red bin",
        "phase_index": 0,
        "n_phases": 2,
        "phase_description": "drop it",
        "record_start": 1.0,
        "record_stop": 2.0,
        "fps": 15,
        "cameras": {"exterior_image_1_left": "external_cam.mp4"},
        **meta_overrides,
    }
    (directory / "_meta.json").write_text(json.dumps({k: v for k, v in meta.items() if v is not None}))
    return ExecuteResult(ok=True, n_frames=frames, rollout_dir=str(directory))


def test_a_leg_without_its_language_label_is_caught(tmp_path):
    # The export labels an episode with _meta.json's instruction and otherwise with the profile's
    # prompt -- not the task typed at the prompt that actually ran.
    result = _recorded(tmp_path / "leg", instruction=None)
    with pytest.raises(ConformanceError, match="instruction=None"):
        check_leg(result, _leg(), tmp_path / "leg")
    with pytest.raises(ConformanceError, match="instruction=None"):
        check_leg(result, _leg(), tmp_path / "leg", records=False)
    check_leg(_recorded(tmp_path / "ok"), _leg(), tmp_path / "ok")
    # A leg asked for with no label has none to carry.
    check_leg(result, _leg(instruction=""), tmp_path / "leg")


def test_a_recording_of_the_wrong_arm_width_is_caught(tmp_path):
    result = _recorded(tmp_path / "six", widths=6)
    with pytest.raises(ConformanceError) as excinfo:
        check_leg(result, _leg(), tmp_path / "six")
    problems = _problems(excinfo)
    assert "joint_position is [10, 6]; merging and the export expect [F,7]" in problems
    assert "cmd_joint_velocity is [10, 6]" in problems


def test_a_recording_on_a_coarse_clock_is_caught(tmp_path):
    result = _recorded(tmp_path / "f32", frame_time=np.float32)
    with pytest.raises(ConformanceError, match="frame_time is float32, not float64"):
        check_leg(result, _leg(), tmp_path / "f32")


@pytest.mark.parametrize("frames", [10, 7], ids=["does-not-divide", "divides-by-seven"])
def test_the_export_skips_a_leg_of_another_arm_width_with_the_reason(tmp_path, frames):
    """[10,6] used to raise ValueError out of reshape(-1, 7) and end the whole export; [7,6] reshaped
    into [6,7] and was skipped as 'state arrays disagree on length', hiding the reason."""
    from tandem.export import build

    directory = tmp_path / "six"
    directory.mkdir()
    arrays = {key: np.zeros(frames) for key in merge.STATE_KEYS}
    for key in ("joint_position", "cmd_joint_position", "cmd_joint_velocity"):
        arrays[key] = np.zeros((frames, 6))
    np.savez(directory / "robot_state.npz", **arrays)
    assert build._add_episode(None, directory, "x") is None
    assert build._last_skip_reason == (
        f"joint_position has shape [{frames}, 6], expected [F,7] (the export writes DROID's 7-joint schema)"
    )


# --- a plugin that exits at import --------------------------------------------------------------------------


@pytest.fixture
def plugin_module(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    written: list[str] = []

    def write(module: str, body: str) -> str:
        (tmp_path / f"{module}.py").write_text(textwrap.dedent(body))
        written.append(module)
        return module

    yield write
    for module in written:
        sys.modules.pop(module, None)


def _declare_planners(monkeypatch, *targets: tuple[str, str]) -> None:
    declared = [importlib.metadata.EntryPoint(name, value, registry.GROUP) for name, value in targets]
    real = importlib.metadata.entry_points
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda **params: list(declared) if params.get("group") == registry.GROUP else real(**params),
    )


def test_a_planner_plugin_that_exits_at_import_is_listed_as_broken_and_breaks_nothing_else(
    plugin_module, monkeypatch
):
    module = plugin_module("tandem_exits_at_import", 'import sys\nsys.exit("argv parsing at import")\n')
    _declare_planners(monkeypatch, ("exits", f"{module}:FACTORY"))

    by_name = {entry.name: entry for entry in registry.catalog()}  # must not exit
    assert not by_name["exits"].ok
    assert "SystemExit: argv parsing at import" in by_name["exits"].error
    assert by_name["tiptop"].ok
    with pytest.raises(TandemError, match="could not be loaded: SystemExit"):
        registry.factory("exits")


def test_a_planner_whose_info_exits_is_its_own_problem(plugin_module, monkeypatch):
    module = plugin_module(
        "tandem_info_exits",
        """
        import sys

        class Factory:
            @property
            def info(self):
                sys.exit(2)

            def capabilities(self): ...
            def create(self, ctx): ...
            def runtime(self, settings=None): ...

        FACTORY = Factory()
        """,
    )
    _declare_planners(monkeypatch, ("infoexits", f"{module}:FACTORY"))
    by_name = {entry.name: entry for entry in registry.catalog()}
    assert not by_name["infoexits"].ok and "SystemExit" in by_name["infoexits"].error
    assert by_name["tiptop"].ok


def test_an_executor_plugin_that_exits_at_import_is_listed_as_broken(plugin_module, monkeypatch):
    module = plugin_module("tandem_executor_exits", "raise SystemExit(3)\n")
    points = [types.SimpleNamespace(name="exits", value=f"{module}:FACTORY", group="g", dist=None)]
    monkeypatch.setattr(executors, "_registered", {})
    monkeypatch.setattr(executors, "metadata", types.SimpleNamespace(entry_points=lambda group: points))
    monkeypatch.setattr(executors, "_discovered", None)

    listed = {entry.name: entry for entry in executors.catalog()}  # must not exit
    assert set(listed) == {"exits", "teleop"}
    assert "SystemExit" in listed["exits"].error
    assert listed["teleop"].error is None


# --- an editable install made while tandem runs -------------------------------------------------------------


def _editable_install(root: Path) -> Path:
    """What `pip install -e .` leaves: metadata in site-packages, and a .pth pointing at the sources."""
    site_dir = root / "site-packages"
    dist = site_dir / "tandem_hot-0.1.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: tandem-hot\nVersion: 0.1\n")
    (dist / "entry_points.txt").write_text("[tandem.planners]\nhot = tandem_hot:HotPlanner\n")
    package = root / "src" / "tandem_hot"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        textwrap.dedent(
            """
            from dataclasses import replace

            from toy_planner import TOY_CAPABILITIES, ToyPlanner
            from tandem.planners import PlannerInfo


            class HotPlanner(ToyPlanner):
                info = PlannerInfo(name="hot")
                CAPABILITIES = replace(TOY_CAPABILITIES, name="hot")
            """
        )
    )
    (site_dir / "__editable__.tandem_hot-0.1.pth").write_text(str(root / "src") + "\n")
    return site_dir


def test_a_planner_installed_editable_while_tandem_runs_is_found_on_the_next_listing(tmp_path, monkeypatch):
    # The metadata directory is on the path, as site-packages always is; the .pth file that puts the
    # sources there is only read by site.py at startup, which for this process has been and gone.
    site_dir = _editable_install(tmp_path)
    monkeypatch.syspath_prepend(str(site_dir))
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(site_dir)])
    monkeypatch.setattr(site, "ENABLE_USER_SITE", False)
    # Nothing in this site directory was there when tandem started.
    monkeypatch.setattr(registry, "_STARTUP_PTH", {}, raising=False)
    monkeypatch.delitem(sys.modules, "tandem_hot", raising=False)
    try:
        (entry,) = [e for e in registry.catalog() if e.name == "hot"]
        assert entry.ok, entry.error
        assert entry.origin.startswith("entry point (tandem-hot 0.1)")
        assert registry.factory("hot").info.name == "hot"
    finally:
        sys.modules.pop("tandem_hot", None)


def test_a_plugin_whose_module_is_missing_is_told_a_restart_may_find_it(plugin_module, monkeypatch):
    monkeypatch.setattr(registry, "_pick_up_new_site_paths", lambda: False, raising=False)
    _declare_planners(monkeypatch, ("gone", "tandem_not_on_the_path:FACTORY"))
    with pytest.raises(TandemError) as excinfo:
        registry.factory("gone")
    assert "restart it (`tandem ui`)" in excinfo.value.hint
    assert "Otherwise reinstall that package" in excinfo.value.hint


# --- a sidecar planner defined where there is no module file -----------------------------------------------


def test_a_sidecar_planner_defined_in_a_notebook_finds_its_script_in_the_working_directory(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "nb_sidecar.py").write_text("from tandem_sidecar import serve\n")
    module = types.ModuleType("nb_mod")  # a notebook's __main__: no __file__
    monkeypatch.setitem(sys.modules, "nb_mod", module)
    namespace = {"__name__": "nb_mod"}
    exec(
        textwrap.dedent(
            """
            from dataclasses import replace

            from toy_planner import TOY_CAPABILITIES
            from tandem.planners import PlannerInfo, SidecarPlanner


            class Notebook(SidecarPlanner):
                info = PlannerInfo(name="notebook")
                CAPABILITIES = replace(TOY_CAPABILITIES, name="notebook")
                SIDECAR = "nb_sidecar.py"
            """
        ),
        namespace,
    )
    assert namespace["Notebook"].sidecar_script() == tmp_path / "nb_sidecar.py"


def test_a_sidecar_script_that_is_not_there_says_where_it_was_looked_for(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "nb_mod2", types.ModuleType("nb_mod2"))
    with pytest.raises(TandemError, match=rf"resolved against {tmp_path}\)") as excinfo:
        exec(
            textwrap.dedent(
                """
                from dataclasses import replace

                from toy_planner import TOY_CAPABILITIES
                from tandem.planners import PlannerInfo, SidecarPlanner


                class Missing(SidecarPlanner):
                    info = PlannerInfo(name="missing")
                    CAPABILITIES = replace(TOY_CAPABILITIES, name="missing")
                    SIDECAR = "not_here.py"
                """
            ),
            {"__name__": "nb_mod2"},
        )
    assert "working directory" in excinfo.value.hint
