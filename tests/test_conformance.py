"""The planner conformance kit, run against every planner tandem's own suite has -- and caught out.

``tandem.planners.testing`` is what a plugin author runs from their own tests. Two things have to be
true of it, and both are tested here:

- a planner that does what the protocol says passes it. Four do: the ``FakeBackend`` every session
  test drives (only its stamp is checked -- it records nothing), a ``Planner`` with a non-cuTAMP goal
  language in tandem's process, the same world behind a ``SidecarPlanner`` in a child process, and
  TiPToP's declarations, signatures and sidecar script (its dynamic half needs a GPU and a robot);
- a planner that does NOT is caught, with every problem named. A kit that passes everything is
  worse than none, because it is a certificate.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from helpers import FakeFactory
from toy_planner import TOY_CAPABILITIES, ToyPlanner, ToySidecarPlanner
from toy_world import STATE_KEYS as TOY_STATE_KEYS

from tandem.core import merge
from tandem.planners import (
    ExecuteResult,
    GoalAtom,
    LegSpec,
    PlannerInfo,
    PlanResult,
    SceneView,
    SidecarPlanner,
)
from tandem.planners.testing import (
    ConformanceError,
    PlannerConformance,
    check_declarations,
    check_leg,
    check_plan_result,
    check_protocol,
    check_scene,
    check_sidecar_script,
    default_goal,
    moved_objects,
)
from tandem.planners.tiptop import FACTORY as TIPTOP
from tandem.planners.tiptop.backend import TiptopBackend
from tandem.planners.tiptop.capabilities import CAPABILITIES as TIPTOP_CAPABILITIES

# --- planners that conform ------------------------------------------------------------------------------


class TestTheFakeBackendConforms(PlannerConformance):
    """The stand-in every session test drives. It stamps legs but records nothing, so only the stamp is
    checked -- which is exactly what a session test relies on it for."""

    planner = FakeFactory("tiptop")
    records_legs = False


class TestAnInProcessPlannerConforms(PlannerConformance):
    planner = ToyPlanner
    options = {"items": ["apple", "pear"]}


class TestASidecarPlannerConforms(PlannerConformance):
    planner = ToySidecarPlanner


def test_tiptop_passes_everything_that_can_be_checked_without_a_robot():
    check_declarations(TIPTOP)
    check_declarations(TiptopBackend)
    check_protocol(TiptopBackend, TIPTOP_CAPABILITIES)
    check_sidecar_script(TiptopBackend)


def test_the_toy_world_records_what_merging_joins():
    # The sidecar half cannot import tandem, so it restates the state keys. Pinned here.
    assert TOY_STATE_KEYS == merge.STATE_KEYS


# --- the goal the kit plans for ---------------------------------------------------------------------------


def test_the_default_goal_moves_something_into_a_surface_the_scene_reported():
    scene = SceneView(
        object_labels=("apple", "red_bin"), table_label="floor", surface_labels=frozenset({"red_bin"})
    )
    goal = default_goal(scene, TOY_CAPABILITIES)
    assert goal == [GoalAtom("in_bin", ("apple", "red_bin"))]
    assert moved_objects(goal, TOY_CAPABILITIES) == {"apple"}

    # TiPToP's language, with no surface detected: the table is the surface.
    scene = SceneView(object_labels=("blue_toy",), table_label="table")
    assert default_goal(scene, TIPTOP_CAPABILITIES) == [GoalAtom("on", ("blue_toy", "table"))]
    # Nothing to move: no goal, and the kit says to override goal() rather than guessing.
    assert default_goal(SceneView(table_label="table"), TOY_CAPABILITIES) is None


# --- what it catches: scenes, plans, legs -----------------------------------------------------------------


def _problems(excinfo) -> str:
    return "\n".join(excinfo.value.problems)


def test_a_malformed_scene_is_caught_with_every_problem_named(tmp_path):
    scene = SceneView(
        object_labels=("apple", "apple", "floor"),
        table_label="floor",
        surface_labels=frozenset({"basket"}),
        rgb_path=str(tmp_path / "missing.png"),
        scene_id="",
    )
    with pytest.raises(ConformanceError) as excinfo:
        check_scene(scene)
    problems = _problems(excinfo)
    for fragment in (
        "more than once",
        "also an object label",
        "basket",
        "scene_id is empty",
        "is not a file",
    ):
        assert fragment in problems
    assert len(excinfo.value.problems) == 5

    with pytest.raises(ConformanceError, match="not a SceneView"):
        check_scene({"scene_id": "s1"})


def test_a_plan_result_tandem_could_not_record_is_caught():
    with pytest.raises(ConformanceError) as excinfo:
        check_plan_result(
            PlanResult(
                ok=True, planning_seconds=-1.0, artifacts={"plan": Path("/x")}, task_plan=("drop the apple",)
            )
        )
    problems = _problems(excinfo)
    assert "no plan_handle" in problems
    assert "planning_seconds" in problems
    assert "artifacts must map" in problems
    assert "'drop the apple' is not an operator" in problems

    with pytest.raises(ConformanceError, match="must say why"):
        check_plan_result(PlanResult(ok=False))
    check_plan_result(PlanResult(ok=False, failure_reason="unreachable"))
    check_plan_result(PlanResult(ok=True, plan_handle="p", task_plan=("Drop(apple, red_bin)", "GoHome()")))


def _leg() -> LegSpec:
    return LegSpec(
        trajectory_id="t-1", segment_source="tamp", phase_index=0, n_phases=2, phase_description="drop it"
    )


def _recorded(directory: Path, **meta_overrides) -> ExecuteResult:
    """A complete toy leg, with _meta.json overridden."""
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "robot_state.npz", **{key: np.zeros(4) for key in merge.STATE_KEYS})
    (directory / "external_cam.mp4").write_bytes(b"clip")
    meta = {
        "trajectory_id": "t-1",
        "segment_source": "tamp",
        "phase_index": 0,
        "n_phases": 2,
        "phase_description": "drop it",
        "record_start": 1.0,
        "record_stop": 2.0,
        "fps": 15,
        "cameras": {"exterior_image_1_left": "external_cam.mp4"},
        **meta_overrides,
    }
    (directory / "_meta.json").write_text(json.dumps(meta))
    return ExecuteResult(ok=True, n_frames=4, rollout_dir=str(directory))


def test_a_complete_leg_passes(tmp_path):
    check_leg(_recorded(tmp_path / "leg"), _leg(), tmp_path / "leg")


def test_a_leg_merging_could_not_place_is_caught(tmp_path):
    result = _recorded(tmp_path / "leg", trajectory_id="someone-else", phase_index=None)
    with pytest.raises(ConformanceError) as excinfo:
        check_leg(result, _leg(), tmp_path / "leg")
    assert "trajectory_id='someone-else'" in _problems(excinfo)
    assert "phase_index=None" in _problems(excinfo)

    (tmp_path / "bare").mkdir()
    with pytest.raises(ConformanceError, match="no _meta.json"):
        check_leg(ExecuteResult(ok=True, rollout_dir=str(tmp_path / "bare")), _leg(), tmp_path / "bare")


def test_a_recording_that_breaks_the_contract_is_caught(tmp_path):
    directory = tmp_path / "leg"
    result = _recorded(directory, record_start=None, fps="fast")
    np.savez(
        directory / "robot_state.npz",
        joint_position=np.zeros(4),
        gripper_position=np.zeros(3),
        depth=np.zeros(4),
    )
    (directory / "external_cam.mp4").unlink()
    with pytest.raises(ConformanceError) as excinfo:
        check_leg(result, _leg(), directory)
    problems = _problems(excinfo)
    assert "no numeric record_start" in problems and "no numeric fps" in problems
    assert "lacks cmd_joint_position" in problems
    assert "carries depth, which merging refuses" in problems
    assert "not one row per frame" in problems
    assert "missing: external_cam.mp4" in problems

    # A planner that only stamps (a stand-in with nothing to record) is held to the stamp alone.
    check_leg(result, _leg(), directory, records=False)


# --- what it catches: the protocol ------------------------------------------------------------------------


class _Lacking:
    """A backend written against an older protocol: no open_gripper, the wrong defaults, a verb short."""

    name = "lacking"

    def capabilities(self):
        return TOY_CAPABILITIES

    def require_ready(self): ...
    def warm(self): ...
    def close(self): ...
    def release_hardware(self): ...
    def reacquire_hardware(self): ...
    def capture_frame(self, *, camera="hand"): ...

    def perceive(self, *, task_hint, save_dir, reset_arm=True): ...

    def plan(self, goal, scene_id, *, surfaces=frozenset(), save_dir, reuse_skeleton=None): ...

    def execute(self, plan_handle, leg, *, save_dir): ...


def test_a_backend_short_of_the_protocol_is_caught_before_a_session_trips_on_it():
    with pytest.raises(ConformanceError) as excinfo:
        check_protocol(_Lacking())
    problems = _problems(excinfo)
    assert "it has no home" in problems
    assert "perceive does not take open_gripper" in problems
    assert "capture_frame(camera=...) defaults to 'hand', not 'external'" in problems
    assert "plan must take scene_id as positional argument 1" in problems
    assert "execute does not take should_stop" in problems
    # The toy declares both restrictions, so a plan() without them is short of what it promised.
    assert "plan does not take movables" in problems and "plan does not take return_home" in problems

    # A planner that declares neither may leave both out: tandem never passes them to it.
    class Unrestricted(_Lacking):
        def capabilities(self):
            return replace(TOY_CAPABILITIES, supports_movable_restriction=False, supports_return_home=False)

    with pytest.raises(ConformanceError) as excinfo:
        check_protocol(Unrestricted())
    assert "movables" not in _problems(excinfo) and "return_home" not in _problems(excinfo)


# --- what it catches: a sidecar script --------------------------------------------------------------------


def _sidecar_planner(script: Path) -> type[SidecarPlanner]:
    class Scripted(SidecarPlanner):
        info = PlannerInfo(name="toy-sidecar")
        CAPABILITIES = ToySidecarPlanner.CAPABILITIES
        SIDECAR = str(script)

    return Scripted


@pytest.mark.parametrize(
    ("body", "complaint"),
    [
        ("import numpy\nfrom tandem_sidecar import serve\n", "imports numpy before tandem_sidecar"),
        ("from tandem.planners import base\nfrom tandem_sidecar import serve\n", "imports tandem"),
        ("import json\nprint(json.dumps({}))\n", "does not import tandem_sidecar"),
    ],
    ids=["library-first", "imports-tandem", "no-kit"],
)
def test_a_sidecar_script_that_could_not_run_cleanly_is_caught(tmp_path, body, complaint):
    script = tmp_path / "sidecar.py"
    script.write_text(textwrap.dedent(body))
    with pytest.raises(ConformanceError, match=complaint):
        check_sidecar_script(_sidecar_planner(script))


def test_a_sidecar_script_may_import_the_standard_library_first(tmp_path):
    script = tmp_path / "sidecar.py"
    script.write_text(
        "from __future__ import annotations\nimport json, os\nfrom tandem_sidecar import serve\nimport numpy\n"
    )
    check_sidecar_script(_sidecar_planner(script))


# --- what it catches: whole planners that break a promise -------------------------------------------------


class _Unstamped(ToyPlanner):
    """Executes, records, and forgets which leg it was: a leg that files as an episode of its own."""

    def execute(self, plan_handle, leg, *, save_dir, should_stop=None):
        return super().execute(
            plan_handle, replace(leg, trajectory_id="its-own"), save_dir=save_dir, should_stop=should_stop
        )


class _Unstoppable(ToyPlanner):
    """Promises cooperative stop and never asks."""

    def execute(self, plan_handle, leg, *, save_dir, should_stop=None):
        return super().execute(plan_handle, leg, save_dir=save_dir)


class _Unrestricted(ToyPlanner):
    """Declares the movable restriction and plans as though it had not been given one."""

    def plan(
        self,
        scene_id,
        goal,
        *,
        surfaces=frozenset(),
        movables=None,
        return_home=True,
        save_dir,
        reuse_skeleton=None,
    ):
        return super().plan(scene_id, goal, surfaces=surfaces, return_home=return_home, save_dir=save_dir)


@pytest.mark.parametrize(
    ("planner", "test", "complaint"),
    [
        (
            _Unstamped,
            "test_an_executed_plan_is_recorded_as_the_leg_it_was_asked_for",
            "trajectory_id='its-own'",
        ),
        (_Unstoppable, "test_a_stop_is_honoured_when_promised", "should_stop was never asked"),
        (_Unrestricted, "test_a_leg_picks_only_what_it_is_allowed_to", "allowed to pick nothing planned"),
    ],
    ids=["unstamped", "unstoppable", "unrestricted"],
)
def test_a_planner_that_breaks_a_promise_fails_the_kit(tmp_path, planner, test, complaint):
    suite = type("Suite", (PlannerConformance,), {"planner": planner})()
    with pytest.raises(AssertionError, match=complaint):
        getattr(suite, test)(tmp_path)
