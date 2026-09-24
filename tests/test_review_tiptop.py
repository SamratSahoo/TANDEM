"""TiPToP's side of the runtime, held to what the pinned planner does where the review found it was not.

- The options refuse what the pinned tiptop and cuTAMP refuse later, with the arm already moving: a
  speed override outside (0, 1], a trajectory norm below 1, an arm neither knows, a remote SAM-2 with
  no address.
- The factory and doctor refuse a profile without both of the cameras tiptop opens at every warm-up.
- A leg tandem recorded with TiPToP can be replayed in tiptop's own viewer (`tandem traj open`): the
  sidecar leaves what the viewer reads, and tandem's wrapper gets it past the viewer's stale version gate.
- The SAM-2 checkpoint tiptop downloads into its own tree is kept out of it.

The checks against the pinned sources read them where tests/planner_sources.py finds them, as
tests/test_tiptop_bump.py does. (Kept apart from tests/test_review_runtime.py, whose fixtures unset
$TANDEM_PLANNER_SOURCES for every test in the module.)
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest
from planner_sources import planner_sources
from pydantic import ValidationError

from tandem.core.errors import TandemError
from tandem.planners import runtime as rt_mod


def _function_in(path: Path, name: str) -> ast.FunctionDef:
    found = [
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    ]
    assert found, f"{path.name} no longer defines {name}"
    return found[0]


# --- the options: refused at load, not at the first plan ---------------------------------------------------


@pytest.mark.parametrize(
    "tamp, message",
    [
        ({"time_dilation_factor": 3.0}, r"time_dilation_factor must be in \(0, 1\]"),
        ({"time_dilation_factor": 0.0}, r"time_dilation_factor must be in \(0, 1\]"),
        ({"time_dilation_factor": -0.5}, r"time_dilation_factor must be in \(0, 1\]"),
        ({"time_dilation_factor": float("nan")}, r"time_dilation_factor must be in \(0, 1\]"),
        ({"time_dilation_factor_literal": 1.5}, r"time_dilation_factor_literal must be in \(0, 1\]"),
        ({"traj_length_norm": 0.5}, r"traj_length_norm must be >= 1 \(or inf\)"),
        ({"traj_length_norm": "0"}, r"traj_length_norm must be >= 1"),
        ({"traj_length_norm": float("nan")}, r"traj_length_norm must be >= 1"),
        ({"traj_length_norm": float("-inf")}, r"traj_length_norm must be >= 1"),
        ({"traj_length_norm": True}, r"traj_length_norm must be a number"),
    ],
)
def test_a_tamp_value_the_planner_refuses_later_is_refused_now(tamp, message):
    from tandem.planners.tiptop.options import validate_tamp

    with pytest.raises(ValueError, match=message):
        validate_tamp(tamp)


def test_the_edges_of_those_ranges_are_accepted():
    from tandem.planners.tiptop.options import validate_tamp

    # 1.0 is tiptop's "no extra scaling": it falls back to robot.time_dilation_factor.
    assert validate_tamp({"time_dilation_factor": 1.0}) == {"time_dilation_factor": 1.0}
    assert validate_tamp({"time_dilation_factor": 0.5}) == {"time_dilation_factor": 0.5}
    assert validate_tamp({"traj_length_norm": 1}) == {"traj_length_norm": 1.0}
    assert validate_tamp({"traj_length_norm": float("inf")}) == {"traj_length_norm": "inf"}


def test_the_pinned_cutamp_refuses_a_norm_below_one_in_the_words_tandem_uses():
    root = planner_sources("cuTAMP/cutamp/config.py")
    text = (root / "cuTAMP" / "cutamp" / "config.py").read_text()
    assert "traj_length_norm must be >= 1 (or inf)" in text


def test_the_arms_tandem_accepts_are_the_ones_the_pinned_planner_knows():
    from tandem.planners.tiptop.options import ROBOT_TYPES, TiptopOptions

    for known in ROBOT_TYPES:
        assert TiptopOptions.model_validate({"robot": {"type": known}}).robot.type == known
    with pytest.raises(ValidationError, match=r"unsupported robot type 'franka' .*did you mean panda"):
        TiptopOptions.model_validate({"robot": {"type": "franka"}})
    with pytest.raises(ValidationError, match="Franka Hand is not supported"):
        TiptopOptions.model_validate({"robot": {"type": "fr3"}})
    with pytest.raises(ValidationError, match="did you mean ur5"):
        TiptopOptions.model_validate({"robot": {"type": "ur6"}})


def _compared_constants(fn: ast.AST, attr: str) -> set[str]:
    """The string constants ``<x>.<attr>`` is compared against in ``fn``, by == or in {...}."""
    out: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Attribute) and node.left.attr == attr:
            for right in node.comparators:
                values = right.elts if isinstance(right, (ast.Set, ast.Tuple, ast.List)) else [right]
                out |= {v.value for v in values if isinstance(v, ast.Constant) and isinstance(v.value, str)}
    return out


def test_the_robot_types_are_read_out_of_the_pinned_tiptop_and_cutamp():
    from tandem.planners.tiptop.options import ROBOT_TYPES

    root = planner_sources("tiptop/tiptop/utils.py", "tiptop/tiptop/motion_planning.py", "cuTAMP/cutamp/config.py")
    tiptop = root / "tiptop" / "tiptop"
    tiptop_types = None
    for path, name in (
        (tiptop / "utils.py", "get_robot_client"),
        (tiptop / "motion_planning.py", "get_ik_solver"),
        (tiptop / "motion_planning.py", "get_motion_gen"),
    ):
        found = _compared_constants(_function_in(path, name), "type")
        assert found, f"{name} no longer compares cfg.robot.type"
        tiptop_types = found if tiptop_types is None else tiptop_types & found
    cutamp_types = _compared_constants(
        _function_in(root / "cuTAMP" / "cutamp" / "config.py", "validate_tamp_config"), "robot"
    )
    assert cutamp_types, "validate_tamp_config no longer lists the embodiments"
    # Equal, not just contained: an arm both come to know is one to add, deliberately.
    assert tiptop_types & cutamp_types == ROBOT_TYPES


def test_a_remote_sam_needs_an_address_and_gets_it_rendered(profile):
    from tandem.planners.tiptop import render
    from tandem.planners.tiptop.options import TiptopOptions

    with pytest.raises(ValidationError, match="sam_url is not set"):
        TiptopOptions.model_validate({"perception": {"sam_mode": "remote"}})
    with pytest.raises(ValidationError, match="sam_mode"):
        TiptopOptions.model_validate({"perception": {"sam_mode": "Local"}})
    with pytest.raises(ValidationError, match="needs a scheme and a host"):
        TiptopOptions.model_validate({"perception": {"sam_mode": "remote", "sam_url": "localhost"}})

    options = {**profile.planner.options, "perception": {"sam_mode": "remote", "sam_url": "http://h:8000"}}
    assert render.render_tiptop_config(profile, options)["perception"]["sam"] == {
        "mode": "remote",
        "url": "http://h:8000",
    }
    assert render.render_tiptop_config(profile)["perception"]["sam"] == {"mode": "local"}


def test_the_pinned_tiptop_reads_the_sam_keys_tandem_renders():
    root = planner_sources("tiptop/tiptop/perception/sam2.py")
    text = (root / "tiptop" / "tiptop" / "perception" / "sam2.py").read_text()
    assert "cfg.perception.sam.mode" in text and "cfg.perception.sam.url" in text
    assert 'mode == "local"' in text and 'mode == "remote"' in text


# --- the cameras tiptop opens -----------------------------------------------------------------------------


@pytest.fixture
def gemini_key(monkeypatch):
    from tandem.core import secrets

    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")


def _tiptop_context(profile, tmp_path):
    from tandem.core import paths
    from tandem.planners.base import BackendContext

    session_dir = paths.session_scratch_dir() / profile.name / "s1"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "events.jsonl").touch()
    return BackendContext(
        profile=profile,
        session_dir=session_dir,
        output_dir=profile.trajectories_dir(),
        execute=False,
        record=True,
        session_id="s1",
        task="stack the cups",
        events_file=session_dir / "events.jsonl",
        runtime_dir=tmp_path / "runtime",
        options=dict(profile.planner.options),
    )


@pytest.mark.parametrize("keep, missing", [("external", "hand"), ("hand", "external")])
def test_tiptop_refuses_a_profile_without_both_cameras_before_writing_anything(
    profile, tmp_path, gemini_key, keep, missing
):
    from tandem.core import paths
    from tandem.core.profiles import CamerasSpec
    from tandem.planners.tiptop import FACTORY

    profile.cameras = CamerasSpec(perception=keep, **{keep: getattr(profile.cameras, keep)})
    with pytest.raises(TandemError, match=rf"no cameras\.{missing}: the pinned tiptop opens") as caught:
        FACTORY.create(_tiptop_context(profile, tmp_path))
    assert "tandem profile edit" in caught.value.hint
    assert not (paths.session_scratch_dir() / profile.name / "tiptop.yml").exists()


def test_doctor_fails_a_tiptop_profile_without_both_cameras(profile):
    from tandem.core import probe
    from tandem.core.profiles import CamerasSpec
    from tandem.planners.tiptop import doctor

    complete = doctor.doctor_checks(profile, settings=None, runtime_ready=True, probe_hardware=False)
    assert [c.state for c in complete if c.name == "tiptop cameras"] == [probe.OK]

    profile.cameras = CamerasSpec(perception="external", external=profile.cameras.external)
    checks = doctor.doctor_checks(profile, settings=None, runtime_ready=True, probe_hardware=False)
    (row,) = [c for c in checks if c.name == "tiptop cameras"]
    assert row.state == probe.FAIL and "cameras.hand" in row.detail
    assert not any(c.name == "tamp settings" and "cameras" in c.detail for c in checks), "not a TAMP warning"
    # A machine that cannot collect with TiPToP anyway is told, not failed.
    laptop = doctor.doctor_checks(profile, settings=None, runtime_ready=False, probe_hardware=False)
    assert [c.state for c in laptop if c.name == "tiptop cameras"] == [probe.WARN]


def test_the_pinned_tiptop_opens_both_cameras_at_warm_up():
    root = planner_sources("tiptop/tiptop/tiptop_run.py")
    warm = _function_in(root / "tiptop" / "tiptop" / "tiptop_run.py", "get_demo_container")
    called = {n.func.id for n in ast.walk(warm) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert {"get_hand_camera", "get_external_camera"} <= called


# --- the SAM-2 checkpoint ---------------------------------------------------------------------------------


def test_tiptop_keeps_its_sam2_cache_out_of_its_tree():
    from tandem.planners.tiptop.recipe import RECIPE

    assert "tiptop/.cache" in RECIPE.source("tiptop").persistent


def test_tiptops_cache_is_where_the_pinned_tiptop_downloads_sam2():
    root = planner_sources("tiptop/tiptop/utils.py", "tiptop/tiptop/perception/sam2.py")
    cache_dir = _function_in(root / "tiptop" / "tiptop" / "utils.py", "get_tiptop_cache_dir")
    assert ".cache" in {n.value for n in ast.walk(cache_dir) if isinstance(n, ast.Constant)}
    assert "get_tiptop_cache_dir()" in (root / "tiptop" / "tiptop" / "perception" / "sam2.py").read_text()


# --- replaying a leg --------------------------------------------------------------------------------------


def test_the_replay_runs_tandems_viewer_in_the_runtime_with_the_directory_as_a_flag(tmp_path, monkeypatch):
    from tandem.planners.tiptop.runtime import VIEWER, TiptopRuntime

    monkeypatch.setattr(rt_mod, "_find_pixi", lambda: Path("/opt/pixi"))
    rt = TiptopRuntime(tmp_path / "runtime")
    leg = tmp_path / "leg"
    assert rt.replay_command(leg) == [
        "/opt/pixi",
        "run",
        "--manifest-path",
        str(rt.root / "tiptop" / "pixi.toml"),
        "python",
        str(VIEWER),
        "--save-dir",
        str(leg),
    ]
    assert VIEWER.is_file()


def test_the_viewer_opens_the_gate_for_1x_plans_only():
    from tandem.planners.tiptop.viewer import accept_any_1x

    load = accept_any_1x(lambda path: {"version": path, "steps": []})
    assert load("1.4.0")["version"] == "1.0.0"
    assert load("1.0.0")["version"] == "1.0.0"
    assert load("2.0.0")["version"] == "2.0.0", "a breaking version is still the viewer's to refuse"


# A stand-in for the pinned viewer, down to what viewer.py relies on: load_tiptop_plan called through the
# module's own globals, the "1.0.0" gate after it, and a tyro entry point reading sys.argv.
_FAKE_VIEWER = """
import json, sys
from tiptop.planning import load_tiptop_plan

def viz_tiptop_run(save_dir):
    plan = load_tiptop_plan(save_dir + "/tiptop_plan.json")
    if plan["version"] != "1.0.0":
        raise NotImplementedError(f"TiPToP plan version {plan['version']} not supported")
    print("replayed", len(plan["steps"]), "steps")

def viz_tiptop_run_entrypoint():
    assert sys.argv[0] == "viz-tiptop-run" and sys.argv[1] == "--save-dir", sys.argv
    viz_tiptop_run(sys.argv[2])
"""


@pytest.mark.parametrize("version, replayed", [("1.4.0", True), ("2.0.0", False)])
def test_the_viewer_replays_the_plans_the_pinned_tiptop_writes(tmp_path, version, replayed):
    import json
    import os
    import sys

    from tandem.planners.tiptop.runtime import VIEWER

    fake = tmp_path / "site"
    (fake / "tiptop" / "scripts").mkdir(parents=True)
    (fake / "tiptop" / "__init__.py").write_text("")
    (fake / "tiptop" / "planning.py").write_text(
        "import json\ndef load_tiptop_plan(path):\n    return json.load(open(path))\n"
    )
    (fake / "tiptop" / "scripts" / "viz_tiptop_run.py").write_text(_FAKE_VIEWER)
    leg = tmp_path / "leg"
    leg.mkdir()
    (leg / "tiptop_plan.json").write_text(json.dumps({"version": version, "q_init": [], "steps": [{}]}))

    result = subprocess.run(
        [sys.executable, str(VIEWER), "--save-dir", str(leg)],
        env={**os.environ, "PYTHONPATH": str(fake)},
        capture_output=True,
        text=True,
    )
    if replayed:
        assert result.returncode == 0 and "replayed 1 steps" in result.stdout, result.stderr
    else:
        assert result.returncode != 0 and "version 2.0.0 not supported" in result.stderr


def test_the_viewer_imports_nothing_from_tandem():
    from tandem.planners.tiptop.runtime import VIEWER

    tree = ast.parse(VIEWER.read_text())
    imported = {
        (n.module or "") if isinstance(n, ast.ImportFrom) else a.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
        for a in n.names
    }
    assert not any(name.split(".")[0] == "tandem" for name in imported), imported


def _leg(tmp_path: Path, files) -> Path:
    leg = tmp_path / "20260923_101010"
    for name in files:
        (leg / name).parent.mkdir(parents=True, exist_ok=True)
        (leg / name).write_text("")
    return leg


def test_a_leg_missing_what_the_viewer_reads_is_named_before_anything_starts(tmp_path, monkeypatch):
    from tandem.planners.tiptop.factory import FACTORY, PLAN_FILE, REPLAY_FILES

    launched: list = []
    monkeypatch.setattr(subprocess, "call", lambda *a, **k: launched.append(a) or 0)
    leg = _leg(tmp_path, [f for f in REPLAY_FILES if f != "metadata.json"])
    with pytest.raises(TandemError, match="has no metadata.json") as caught:
        FACTORY.replay(leg)
    assert "before tandem wrote" in caught.value.hint
    with pytest.raises(TandemError, match=f"has no {PLAN_FILE}"):
        FACTORY.replay(_leg(tmp_path / "other", ["metadata.json"]))
    assert launched == []


def test_a_viewer_that_fails_is_an_error(tmp_path, monkeypatch):
    from tandem.planners.tiptop.factory import FACTORY, REPLAY_FILES
    from tandem.planners.tiptop.runtime import VIEWER, TiptopRuntime

    monkeypatch.setattr(rt_mod, "_find_pixi", lambda: Path("/opt/pixi"))
    monkeypatch.setattr(TiptopRuntime, "require_ready", lambda self: None)
    calls: list = []
    code = {"value": 1}

    def call(argv, cwd=None):
        calls.append(argv)
        return code["value"]

    monkeypatch.setattr(subprocess, "call", call)
    leg = _leg(tmp_path, REPLAY_FILES)
    with pytest.raises(TandemError, match="viewer exited with status 1"):
        FACTORY.replay(leg)
    code["value"] = 0
    FACTORY.replay(leg)
    assert str(VIEWER) in calls[-1] and calls[-1][-2:] == ["--save-dir", str(leg)]


def _written_paths(fn: ast.AST) -> set[str]:
    """``save_dir / "x"`` and ``perception_dir / "x"`` in ``fn``, as leg-relative paths."""
    out = set()
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            and isinstance(node.left, ast.Name)
            and isinstance(node.right, ast.Constant)
            and isinstance(node.right.value, str)
            and re.fullmatch(r"[\w.-]+\.\w+", node.right.value)
        ):
            if node.left.id == "save_dir":
                out.add(node.right.value)
            elif node.left.id == "perception_dir":
                out.add(f"perception/{node.right.value}")
    return out


def test_what_the_viewer_reads_is_what_a_sidecar_leg_holds():
    """Read out of the pinned sources, so a bump that makes the viewer read one more file, or the
    recorder stop writing one, fails here rather than on the first `tandem traj open` after it."""
    from tandem.planners.tiptop.backend import sidecar_path
    from tandem.planners.tiptop.factory import PLAN_FILE, REPLAY_FILES

    root = planner_sources(
        "tiptop/tiptop/scripts/viz_tiptop_run.py", "tiptop/tiptop/recording.py", "tiptop/tiptop/tiptop_run.py"
    )
    tiptop = root / "tiptop" / "tiptop"
    viz = _function_in(tiptop / "scripts" / "viz_tiptop_run.py", "viz_tiptop_run")
    # Every file the viewer reads, except the one it reads only if it is there.
    assert _written_paths(viz) - {"perception/gripper_mask.png"} == set(REPLAY_FILES)
    assert viz.args.args[0].arg == "save_dir", "the flag replay_command passes is --save-dir"
    module = ast.parse((tiptop / "scripts" / "viz_tiptop_run.py").read_text())
    assert any(isinstance(n, ast.FunctionDef) and n.name == "viz_tiptop_run_entrypoint" for n in module.body)
    assert any(
        isinstance(n, ast.ImportFrom) and any(a.name == "load_tiptop_plan" for a in n.names) for n in module.body
    ), "the viewer calls load_tiptop_plan through its own module, which is where viewer.py replaces it"

    recording = tiptop / "recording.py"
    written = {PLAN_FILE}  # the sidecar's own save_tiptop_plan(..., directory / PLAN_FILE)
    for name in ("save_perception_outputs", "save_run_outputs", "save_run_metadata"):
        written |= _written_paths(_function_in(recording, name))
    assert set(REPLAY_FILES) <= written, sorted(set(REPLAY_FILES) - written)

    # ...and the sidecar reaches every one of those writers for each leg.
    sidecar = ast.parse(sidecar_path().read_text())
    called = {
        n.func.id if isinstance(n.func, ast.Name) else n.func.attr
        for n in ast.walk(sidecar)
        if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))
    }
    assert {"run_perception", "save_run_outputs", "save_run_metadata", "save_tiptop_plan"} <= called
    run_perception = _function_in(tiptop / "tiptop_run.py", "run_perception")
    assert "save_perception_outputs" in {n.id for n in ast.walk(run_perception) if isinstance(n, ast.Name)}

    # The two version gates: metadata.json's, which save_run_metadata satisfies, and the plan's, which
    # viewer.py opens for any 1.x -- and serialize_plan writes a 1.x.
    metadata = _function_in(recording, "save_run_metadata")
    assert "1.0.0" in {n.value for n in ast.walk(metadata) if isinstance(n, ast.Constant)}
    serialize = _function_in(tiptop / "planning.py", "serialize_plan")
    (version,) = [
        v.value
        for d in ast.walk(serialize)
        if isinstance(d, ast.Dict)
        for k, v in zip(d.keys, d.values, strict=True)
        if isinstance(k, ast.Constant) and k.value == "version" and isinstance(v, ast.Constant)
    ]
    assert version.split(".")[0] == "1"


def test_the_sidecar_writes_metadata_with_everything_the_recorder_requires_once_planning_answers():
    from tandem.planners.tiptop.backend import sidecar_path

    root = planner_sources("tiptop/tiptop/recording.py")
    recorder = _function_in(root / "tiptop" / "tiptop" / "recording.py", "save_run_metadata")
    positional = [a.arg for a in recorder.args.args]
    required = set(positional[: len(positional) - len(recorder.args.defaults)])

    tree = ast.parse(sidecar_path().read_text())
    (sidecar,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Sidecar"]
    methods = {n.name: n for n in sidecar.body if isinstance(n, ast.FunctionDef)}
    (call,) = [
        n
        for n in ast.walk(methods["_save_metadata"])
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "save_run_metadata"
    ]
    passed = {k.arg for k in call.keywords}
    assert required <= passed and passed <= set(positional), (required - passed, passed - set(positional))

    # In plan(), after the planner answered and before the first return that follows -- so a leg whose
    # planning failed has one too, as a tiptop-run rollout does.
    plan = methods["plan"]
    calls = [n for n in ast.walk(plan) if isinstance(n, ast.Call)]
    (planned,) = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "run_planning"]
    (saved,) = [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "_save_metadata"]
    returns_after = [n.lineno for n in ast.walk(plan) if isinstance(n, ast.Return) and n.lineno > planned.lineno]
    assert planned.lineno < saved.lineno < min(returns_after)
