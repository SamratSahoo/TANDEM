"""Surface-fitted placement: the placement_* keys, the three switches beside them, and the sidecar's half.

LJ1356's placement support is ported to the TANDEM branches of SamratSahoo's tiptop and cuTAMP, which
tandem pins (planners/tiptop/recipe.py). With it, "Solve Constrained Puzzle" and "Store Bread in Closed
Box" plan with the placement_* settings their configs were tuned with: an object is put down on a level
patch of the surface's OBSERVED points, at that patch's height, instead of anywhere in the surface's
bounding box at the height of its highest vertex -- the top of a box's folded-back lid, where the bread
was released 19 cm up and fell.

Everything here is opt-in, in tiptop as in tandem: no placement_* key, no change. What is checked:

- the keys a profile may set, their types and ranges, against what the pinned tiptop reads
  (``resolve_placement_support``) and what the pinned cuTAMP refuses (``validate_tamp_config``);
- where the two perception switches land in the rendered tiptop.yml, and the "does nothing without"
  warnings;
- that tandem's sidecar -- which bypasses tiptop's own rollout loop -- threads the placement config and
  every surface's observed points through exactly as that loop does, and turns a goal no surface can
  hold into an ordinary plan failure, which ``hitl.on_robot_phase_failure`` then decides about.

The source checks read the pinned trees where tests/planner_sources.py finds them.
"""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path

import pytest
from planner_sources import planner_sources
from ruamel.yaml import YAML

from tandem.core import rig as rig_mod
from tandem.planners.tiptop import render, tamp_keys
from tandem.planners.tiptop.backend import sidecar_path
from tandem.planners.tiptop.options import resolve_profile, validate_tamp


def _opts(profile):
    """TiPToP's whole configuration for ``profile``, on this machine's rig (the ``machine_rig`` fixture's)."""
    return resolve_profile(profile, rig_mod.load())

FIXTURES = Path(__file__).parent / "fixtures" / "cfg_tamp"
_yaml = YAML(typ="safe")

PLACEMENT = ("placement_support", *tamp_keys.PLACEMENT_GATED)
SWITCHES = ("table_plane_support_vote", "disjoint_object_masks", "blend_stretch_to_caps")

# What resolve_placement_support uses for each gated key the profile leaves out, once placement_support
# is on -- as docs/CONFIGURATION.md documents them. Checked against the pinned source below.
DEFAULTS = {
    "placement_support_margin": 0.01,
    "placement_flatness_tol": 0.008,
    "placement_support_required": True,
    "placement_into_surface": True,
    "placement_fill_occluded": False,
    "placement_min_seen_frac": 0.25,
}


def _overrides(name: str) -> dict:
    return dict(_yaml.load((FIXTURES / name).read_text())["tamp_overrides"])


# --- the keys, as a profile states them ----------------------------------------------------------------


def test_the_placement_keys_and_switches_are_settings_of_their_types():
    types = {key: tamp_keys.SCALAR_KEYS[key] for key in (*PLACEMENT, *SWITCHES)}
    assert types == {
        "placement_support": bool,
        "placement_support_margin": float,
        "placement_flatness_tol": float,
        "placement_support_required": bool,
        "placement_into_surface": bool,
        "placement_fill_occluded": bool,
        "placement_min_seen_frac": float,
        "table_plane_support_vote": bool,
        "disjoint_object_masks": bool,
        "blend_stretch_to_caps": bool,
    }
    assert not set(types) & set(tamp_keys.REFUSED)
    assert set(DEFAULTS) == set(tamp_keys.PLACEMENT_GATED)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "4_bread_box_v3.yml",
            {
                "placement_support": True,
                "placement_support_margin": 0.005,
                "placement_support_required": True,
                "placement_into_surface": True,
                "placement_fill_occluded": True,
                "placement_min_seen_frac": 0.25,
                "placement_flatness_tol": 0.012,
            },
        ),
        ("4_bread_box.yml", None),  # the config 7412655 introduced; the same placement block
        (
            "1_toy_puzzle_v3.yml",
            {"placement_support": True, "placement_support_required": True, "placement_into_surface": True},
        ),
    ],
)
def test_the_two_tasks_configs_validate_with_their_placement_settings(name, expected):
    out = validate_tamp(_overrides(name))
    placement = {k: v for k, v in out.items() if k.startswith("placement_")}
    expected = expected or validate_tamp(_overrides("4_bread_box_v3.yml"))
    assert placement == {k: v for k, v in expected.items() if k.startswith("placement_")}


def test_the_switches_are_accepted_either_way():
    on = dict.fromkeys(SWITCHES, True)
    assert validate_tamp(on) == on
    off = dict.fromkeys(SWITCHES, False)
    assert validate_tamp(off) == off


@pytest.mark.parametrize("key", [*[k for k in PLACEMENT if tamp_keys.SCALAR_KEYS[k] is bool], *SWITCHES])
@pytest.mark.parametrize("value", ["false", "true", 1, 0, "yes"])
def test_a_switch_is_a_real_boolean(key, value):
    # tiptop casts the placement booleans with bool(), which reads a quoted "false" as TRUE -- on a
    # setting that decides whether the arm may place into a container. tandem refuses anything else.
    with pytest.raises(ValueError, match=f"{key} must be a bool"):
        validate_tamp({key: value})


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("placement_support_margin", -0.001, ">= 0"),
        ("placement_flatness_tol", 0.0, "> 0"),
        ("placement_flatness_tol", -0.01, "> 0"),
        ("placement_min_seen_frac", 1.01, "[0, 1]"),
        ("placement_min_seen_frac", -0.1, "[0, 1]"),
        ("placement_min_seen_frac", math.nan, "[0, 1]"),
        ("placement_support_margin", "5mm", "must be a float"),
    ],
)
def test_the_placement_ranges_are_cutamps(key, value, message):
    """cuTAMP refuses each of these too -- at the first plan, when run_cutamp validates its config, with
    the arm already at the capture pose and nothing in run_planning to catch it."""
    with pytest.raises(ValueError, match=re.escape(message)):
        validate_tamp({"placement_support": True, key: value})


def test_the_edges_of_the_placement_ranges_are_accepted():
    out = validate_tamp(
        {
            "placement_support": True,
            "placement_support_margin": 0,  # no clearance beyond the footprint itself
            "placement_min_seen_frac": 1,
            "placement_flatness_tol": 1e-4,
        }
    )
    assert out["placement_support_margin"] == 0.0 and isinstance(out["placement_support_margin"], float)
    assert out["placement_min_seen_frac"] == 1.0
    assert validate_tamp({"placement_min_seen_frac": 0.0}) == {"placement_min_seen_frac": 0.0}


def test_a_null_placement_key_means_unset():
    assert validate_tamp({"placement_support": None, "placement_flatness_tol": None}) == {}


# --- rendering and warnings ----------------------------------------------------------------------------


def test_the_perception_switches_land_in_the_rendered_tiptop_yml(profile):
    profile.planner.options["tamp"] = validate_tamp(
        {"table_plane_support_vote": True, "disjoint_object_masks": False}
    )
    perception = render.render_tiptop_config(rig_mod.load(), _opts(profile))["perception"]
    assert perception["table_plane_support_vote"] is True and perception["disjoint_object_masks"] is False
    # And they are still passed on with the rest, where the sidecar's apply_perception_overrides reads them.
    assert render.render_tamp_overrides(profile, _opts(profile))["table_plane_support_vote"] is True


def test_unset_switches_leave_tiptops_defaults_in_force(profile):
    profile.planner.options["tamp"] = {}
    perception = render.render_tiptop_config(rig_mod.load(), _opts(profile))["perception"]
    assert "table_plane_support_vote" not in perception and "disjoint_object_masks" not in perception


def test_the_placement_keys_reach_the_planner_as_they_are(profile):
    profile.planner.options["tamp"] = validate_tamp(_overrides("4_bread_box_v3.yml"))
    rendered = render.render_tamp_overrides(profile, _opts(profile))
    assert {k: rendered[k] for k in PLACEMENT} == {k: _overrides("4_bread_box_v3.yml")[k] for k in PLACEMENT}


@pytest.mark.parametrize("key", tamp_keys.PLACEMENT_GATED)
@pytest.mark.parametrize("gate", [None, False])
def test_a_placement_key_without_placement_support_is_said_to_do_nothing(profile, key, gate):
    value = {bool: True, float: 0.5}[tamp_keys.SCALAR_KEYS[key]]
    tamp = {key: value} if gate is None else {"placement_support": gate, key: value}
    profile.planner.options["tamp"] = validate_tamp(tamp)
    problems = render.check_assets(profile, rig_mod.load(), _opts(profile))
    assert f"{key} only applies when placement_support is true; it is ignored here" in problems


def test_blend_stretch_to_caps_without_blending_is_said_to_do_nothing(profile):
    profile.planner.options["tamp"] = validate_tamp({"blend_stretch_to_caps": True})
    assert any(
        "blend_stretch_to_caps only applies when blend_trajectory" in p for p in render.check_assets(profile, rig_mod.load(), _opts(profile))
    )
    profile.planner.options["tamp"] = validate_tamp({"blend_stretch_to_caps": True, "blend_trajectory": True})
    assert not any("blend_stretch_to_caps" in p for p in render.check_assets(profile, rig_mod.load(), _opts(profile)))


@pytest.mark.parametrize("name", ["1_toy_puzzle_v3.yml", "4_bread_box.yml", "4_bread_box_v3.yml"])
def test_the_two_tasks_with_all_three_switches_raise_no_warning(profile, name):
    profile.planner.options["tamp"] = validate_tamp({**_overrides(name), **dict.fromkeys(SWITCHES, True)})
    problems = [p for p in render.check_assets(profile, rig_mod.load(), _opts(profile)) if "vae_path does not exist" not in p]
    assert not [p for p in problems if "only applies" in p], problems


# --- against the pinned tiptop and cuTAMP ------------------------------------------------------------


def _sources() -> Path:
    return planner_sources(
        "tiptop/tiptop/motion_planning.py",
        "tiptop/tiptop/planning.py",
        "tiptop/tiptop/tiptop_run.py",
        "cuTAMP/cutamp/config.py",
        "cuTAMP/cutamp/utils/support.py",
    )


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _function(module: ast.Module, name: str, *, cls: str | None = None) -> ast.FunctionDef:
    body = module.body
    if cls is not None:
        (klass,) = [n for n in body if isinstance(n, ast.ClassDef) and n.name == cls]
        body = klass.body
    found = [n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    assert found, f"no {cls + '.' if cls else ''}{name}"
    return found[0]


def _calls(node: ast.AST, callee: str) -> list[ast.Call]:
    def name(func: ast.AST) -> str | None:
        return (
            func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        )

    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and name(n.func) == callee]


def _resolver() -> ast.FunctionDef:
    return _function(
        _parse(_sources() / "tiptop" / "tiptop" / "motion_planning.py"), "resolve_placement_support"
    )


def _resolved_fields() -> dict[str, tuple[str, ast.AST | None, str]]:
    """cuTAMP field -> (the override key it is read from, its default, the cast), from the returned dict."""
    (returned,) = [
        n.value
        for n in ast.walk(_resolver())
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict) and n.value.keys
    ]
    out = {}
    for key, value in zip(returned.keys, returned.values, strict=True):
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.args:
            (get,) = [c for c in _calls(value, "get")]
            default = get.args[1] if len(get.args) > 1 else None
            out[key.value] = (get.args[0].value, default, value.func.id)
        else:
            out[key.value] = (None, value, "")
    return out


def test_tiptop_reads_placement_only_behind_placement_support():
    resolver = _resolver()
    # The first statement that does anything is the gate: nothing is read, and {} returned, when it is off.
    gate = next(n for n in resolver.body if isinstance(n, ast.If))
    assert "placement_support" in ast.unparse(gate.test)
    assert [ast.unparse(n) for n in gate.body] == ["return {}"]

    read = {
        c.args[0].value for c in _calls(resolver, "get") if c.args and isinstance(c.args[0], ast.Constant)
    }
    assert read == set(PLACEMENT), "tandem's placement keys are exactly the ones tiptop reads"
    assert {key for key, _, _ in _resolved_fields().values() if key} == set(tamp_keys.PLACEMENT_GATED)


def test_tandem_types_each_placement_key_the_way_tiptop_casts_it():
    for field, (key, _, cast) in _resolved_fields().items():
        if key:
            assert tamp_keys.SCALAR_KEYS[key].__name__ == cast, f"{key} -> {field}"


def test_the_defaults_tandem_documents_are_tiptops():
    documented = {}
    for key, default, _ in _resolved_fields().values():
        if key:
            documented[key] = ast.literal_eval(default)
    assert documented == DEFAULTS
    fields = _resolved_fields()
    assert ast.unparse(fields["placement_check"][1]) == "'support'"
    assert ast.unparse(fields["placement_shrink_dist"][1]) == "None", "support_margin replaces the shrink"


def test_every_field_tiptop_sets_is_one_the_pinned_cutamp_has():
    (klass,) = [
        n
        for n in _parse(_sources() / "cuTAMP" / "cutamp" / "config.py").body
        if isinstance(n, ast.ClassDef) and n.name == "TAMPConfiguration"
    ]
    fields = {n.target.id for n in klass.body if isinstance(n, ast.AnnAssign)}
    assert set(_resolved_fields()) <= fields
    placement_check = next(
        n for n in klass.body if isinstance(n, ast.AnnAssign) and n.target.id == "placement_check"
    )
    assert "'support'" in ast.unparse(placement_check.annotation)


def test_tandems_ranges_are_the_ones_the_pinned_cutamp_refuses():
    """Refused by tandem when the profile loads, rather than by cuTAMP at the first plan."""
    validate = _function(_parse(_sources() / "cuTAMP" / "cutamp" / "config.py"), "validate_tamp_config")
    text = ast.unparse(validate)
    field_of = {key: field for field, (key, _, _) in _resolved_fields().items() if key}
    expectations = {
        "placement_support_margin": (tamp_keys.NON_NEGATIVE_KEYS, "config.support_margin < 0.0"),
        "placement_flatness_tol": (tamp_keys.POSITIVE_KEYS, "config.support_flatness_tol <= 0.0"),
        "placement_min_seen_frac": (
            tamp_keys.UNIT_INTERVAL_KEYS,
            "not 0.0 <= config.support_min_seen_frac <= 1.0",
        ),
    }
    for key, (ours, theirs) in expectations.items():
        assert key in ours, key
        assert theirs in text, f"cuTAMP no longer checks {field_of[key]} as {theirs!r}"
        assert field_of[key] in theirs


def test_build_tamp_config_keeps_the_bounding_box_unless_told_otherwise():
    build = _function(_parse(_sources() / "tiptop" / "tiptop" / "planning.py"), "build_tamp_config")
    assert "placement" in {a.arg for a in build.args.args + build.args.kwonlyargs}
    text = ast.unparse(build)
    assert "{'placement_check': 'obb', 'placement_shrink_dist': 0.01} | (placement or {})" in text


def test_the_perception_switches_default_off_in_tiptop():
    """Unset in a profile, each switch is left out of the rendered tiptop.yml, and tiptop reads it with a
    default of False -- the behaviour of its main."""
    root = _sources() / "tiptop" / "tiptop"
    run = (root / "tiptop_run.py").read_text()
    for key in ("table_plane_support_vote", "disjoint_object_masks"):
        assert f'tiptop_cfg().perception.get("{key}", False)' in run, key
    stock = _yaml.load((root / "config" / "tiptop.yml").read_text())
    assert stock["perception"]["table_plane_support_vote"] is False
    assert stock["perception"]["disjoint_object_masks"] is False

    blending = _function(_parse(root / "trajectory_blending.py"), "resolve_blend_config")
    assert "_as_bool('blend_stretch_to_caps', raw_stretch)" in ast.unparse(blending)
    assert "stretch_to_caps = False if raw_stretch is None" in ast.unparse(blending)


# --- the sidecar's half ----------------------------------------------------------------------------------


def _sidecar() -> ast.Module:
    return _parse(sidecar_path())


def test_the_sidecar_builds_every_config_with_tiptops_placement_resolution():
    warm = _function(_sidecar(), "warm", cls="Sidecar")
    (build,) = _calls(warm, "build_tamp_config")
    (placement,) = [k for k in build.keywords if k.arg == "placement"]
    assert ast.unparse(placement.value) == "placement"
    (resolved,) = [
        n
        for n in ast.walk(warm)
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "placement" for t in n.targets)
    ]
    assert ast.unparse(resolved.value) == "resolve_placement_support(self.cost_overrides)"


def test_the_sidecar_hands_the_environment_the_surfaces_observed_points():
    """A sidecar leg that did not would place every object on bounding boxes whatever the profile said,
    with nothing to say so."""
    (ours,) = _calls(_function(_sidecar(), "plan", cls="Sidecar"), "create_tamp_environment")
    (points,) = [k for k in ours.keywords if k.arg == "support_points"]
    assert ast.unparse(points.value) == "processed_scene.object_support_points"


def test_tiptop_hands_every_environment_the_same_points():
    """What the sidecar mirrors: tiptop's own loop builds every environment with the scene's support
    points -- perception, the clearing plan, the task plan, the reset's retry -- and the scene the
    sidecar's perceive() gets from run_perception carries them."""
    root = _sources() / "tiptop" / "tiptop"
    run = _parse(root / "tiptop_run.py")
    theirs = [c for c in _calls(run, "create_tamp_environment")]
    assert theirs and all(
        any(
            k.arg == "support_points" and ast.unparse(k.value) == "processed_scene.object_support_points"
            for k in c.keywords
        )
        for c in theirs
    ), "tiptop no longer passes the support points on every call; re-read what it does instead"

    # ProcessedScene carries them, filled by process_scene_geometry, which run_perception -- the
    # sidecar's perceive() -- calls.
    (scene,) = [n for n in run.body if isinstance(n, ast.ClassDef) and n.name == "ProcessedScene"]
    assert "object_support_points" in {n.target.id for n in scene.body if isinstance(n, ast.AnnAssign)}
    (built,) = _calls(_function(run, "process_scene_geometry"), "ProcessedScene")
    assert "object_support_points" in {k.arg for k in built.keywords}
    perception = _function(run, "run_perception")
    assert any(isinstance(n, ast.Name) and n.id == "process_scene_geometry" for n in ast.walk(perception))


def _catches(handler: ast.ExceptHandler) -> set[str]:
    kinds = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return {ast.unparse(k) for k in kinds}


def test_the_sidecar_reports_a_goal_no_surface_can_hold_as_a_plan_failure():
    """NoSupportRegion (no level patch big enough, with placement_support_required) and NoGraspsError
    (require_m2t2_grasps) are what the scene says, not a broken planner. A verb that raised would end the
    trial outright; ok=False goes to hitl.on_robot_phase_failure -- abort, replan, or the operator. The
    pinned run_planning already reports both that way (below); the sidecar keeps it so should one ever
    escape: its run_planning call is inside a try that catches both and makes them the failure reason."""
    plan = _function(_sidecar(), "plan", cls="Sidecar")
    (guard,) = [
        n for n in ast.walk(plan) if isinstance(n, ast.Try) and any(_calls(s, "run_planning") for s in n.body)
    ]
    (handler,) = [h for h in guard.handlers if {"NoGraspsError", "NoSupportRegion"} <= _catches(h)]
    assigned = {ast.unparse(t) for n in handler.body if isinstance(n, ast.Assign) for t in n.targets}
    assert {"failure_reason", "(cutamp_plan, planning_seconds)"} <= assigned
    cleared = [
        n for n in handler.body if isinstance(n, ast.Assign) and "cutamp_plan" in ast.unparse(n.targets[0])
    ]
    assert ast.unparse(cleared[0].value).startswith("(None,"), "no plan, so plan() answers ok=False"
    imported = {
        (node.module, alias.name)
        for node in ast.walk(plan)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert ("cutamp.utils.support", "NoSupportRegion") in imported
    assert ("cutamp.particle_initialization", "NoGraspsError") in imported


def test_the_pinned_planner_reports_a_goal_no_surface_can_hold_as_a_plan_failure():
    root = _sources()
    support = _parse(root / "cuTAMP" / "cutamp" / "utils" / "support.py")
    (klass,) = [n for n in support.body if isinstance(n, ast.ClassDef) and n.name == "NoSupportRegion"]
    assert [ast.unparse(b) for b in klass.bases] == ["RuntimeError"]
    particles = _parse(root / "cuTAMP" / "cutamp" / "particle_initialization.py")
    assert any(isinstance(n, ast.ClassDef) and n.name == "NoGraspsError" for n in particles.body)

    # run_planning reports both as a plan not found: (None, seconds, reason).
    run_planning = _function(_parse(root / "tiptop" / "tiptop" / "planning.py"), "run_planning")
    handlers = [h for n in ast.walk(run_planning) if isinstance(n, ast.Try) for h in n.handlers]
    assert any({"NoGraspsError", "NoSupportRegion"} <= _catches(h) for h in handlers)
