"""TiPToP at the TANDEM branches (tiptop 6820474 / cuTAMP fc8f233): what tandem accepts, renders and builds,
checked against them.

A bump of the planner changes three things on tandem's side, and each can go wrong without an error:

- the ``tamp:`` keys a profile may set (planners/tiptop/tamp_keys.py). A key tiptop reads that tandem rejects
  makes a real config fail to import; a key tandem accepts that tiptop no longer reads -- or reads
  only in its own interactive loop, which tandem does not run -- is a setting that does nothing.
- where the perception knobs land in the tiptop.yml tandem renders (planners/tiptop/render.py).
- the TAMP config the sidecar builds at warm-up (planners/tiptop/sidecar.py). A knob tiptop's own
  entrypoint threads into ``build_tamp_config`` and the sidecar does not is validated, passed on,
  and then quietly ignored by every plan.

The checks that need the planner's own source read it where tests/planner_sources.py finds it: CI's
planner-sources job fetches the pinned trees, and a workstation with a built runtime uses that.
The rest run anywhere, including against the monorepo's cfg/tamp files kept as fixtures
(tests/fixtures/cfg_tamp, copied verbatim from hitl-tamp-vla 90671e1; 4_bread_box.yml is the config
7412655 added with the placement settings, unchanged since). The placement support itself, and the
sidecar's half of it, are tests/test_tiptop_placement.py.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from planner_sources import planner_sources
from ruamel.yaml import YAML

from tandem.planners.tiptop import render, tamp_keys
from tandem.planners.tiptop.backend import sidecar_path
from tandem.planners.tiptop.options import resolve_profile, validate_tamp

FIXTURES = Path(__file__).parent / "fixtures" / "cfg_tamp"
_yaml = YAML(typ="safe")

# The keys tiptop reads only on paths tandem never runs, and the function each one is read through.
LOOP_ONLY = {
    "auto_mode": "resolve_auto_mode",
    "reset_placement_region": "reset_placement_region",
    "clear_goal_surfaces": "resolve_clear_goal_surfaces",
}


def _sources() -> Path:
    return planner_sources("tiptop/tiptop/motion_planning.py", "tiptop/tiptop/tiptop_run.py", "cuTAMP/cutamp")


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _function(module: ast.Module, name: str, *, cls: str | None = None):
    body = module.body
    if cls is not None:
        (klass,) = [n for n in body if isinstance(n, ast.ClassDef) and n.name == cls]
        body = klass.body
    found = [n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    assert found, f"no {cls + '.' if cls else ''}{name}"
    return found[0]


def _calls(node: ast.AST, callee: str) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == callee
    ]


def _called_names(node: ast.AST) -> set[str]:
    names = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                names.add(n.func.id)
            elif isinstance(n.func, ast.Attribute):
                names.add(n.func.attr)
    return names


def _looks_up(loop: ast.For, variable: str, overrides) -> bool:
    """Whether ``loop``'s body reads the overrides dict at ``variable`` (``ov.get(key)``, ``ov[key]``)."""
    for n in ast.walk(loop):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "get" and n.args:
            if isinstance(n.args[0], ast.Name) and n.args[0].id == variable and overrides(n.func.value):
                return True
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Name) and n.slice.id == variable:
            if overrides(n.value):
                return True
    return False


def _keys_tiptop_reads(tiptop: Path) -> dict[str, set[str]]:
    """Every tamp_overrides key the tiptop tree reads, with where. ``tiptop`` is the repository root.

    tiptop has no schema for these -- each knob is read where it is used, as ``ov.get("key")``,
    ``overrides["key"]``, ``"key" in overrides``, or a ``for key in (...)`` loop over one -- so this
    reads the code the way a person would: a string constant used as a key on anything whose source
    mentions the overrides dict. Plus tiptop's one table of them, _PERCEPTION_OVERRIDE_KEYS.
    """
    names = {"ov", "o", "overrides", "cost_overrides", "curobo_overrides", "tamp_overrides"}

    def overrides(node: ast.AST) -> bool:
        return "overrid" in ast.unparse(node) or (isinstance(node, ast.Name) and node.id in names)

    def text(node) -> str | None:
        return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None

    found: dict[str, set[str]] = {}

    def add(key: str | None, where: str) -> None:
        if key:
            found.setdefault(key, set()).add(where)

    for path in sorted((tiptop / "tiptop").rglob("*.py")):
        rel = path.relative_to(tiptop)
        for n in ast.walk(_parse(path)):
            where = f"{rel}:{getattr(n, 'lineno', '?')}"
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "get"
                and n.args
            ):
                if overrides(n.func.value):
                    add(text(n.args[0]), where)
            elif isinstance(n, ast.Subscript) and overrides(n.value):
                add(text(n.slice), where)
            elif isinstance(n, ast.Compare) and any(overrides(c) for c in n.comparators):
                add(text(n.left), where)
            elif (
                isinstance(n, ast.For)
                and isinstance(n.iter, (ast.Tuple, ast.List))
                and isinstance(n.target, ast.Name)
                and _looks_up(n, n.target.id, overrides)
            ):
                for element in n.iter.elts:
                    add(text(element), where)
            elif isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_PERCEPTION_OVERRIDE_KEYS" for t in n.targets
            ):
                for key in n.value.keys:
                    add(text(key), where)
    return found


# --- the key set, against the pinned tiptop --------------------------------------------------------


def test_tandem_accepts_exactly_the_tamp_keys_the_pinned_tiptop_reads():
    read = _keys_tiptop_reads(_sources() / "tiptop")
    # The extraction itself is sound: it finds knobs of every shape tiptop reads them in.
    shapes = {"vae_retiming", "retime_scale", "m2t2_num_runs", "posture_grasp_roll", "smooth_weight"}
    assert shapes <= set(read)

    unhandled = sorted(set(read) - tamp_keys.ALL_KEYS - set(tamp_keys.REFUSED))
    assert not unhandled, (
        "the pinned tiptop reads keys a profile can neither set nor is told it cannot: "
        + ", ".join(f"{k} ({sorted(read[k])[0]})" for k in unhandled)
    )
    dead = sorted(tamp_keys.ALL_KEYS - set(read))
    assert not dead, f"a profile may set keys the pinned tiptop no longer reads: {dead}"
    assert not (tamp_keys.ALL_KEYS & set(tamp_keys.REFUSED))


def test_the_keys_refused_as_tiptops_own_are_read_only_where_tandem_never_goes():
    """The reason given for refusing auto_mode, reset_placement_region and clear_goal_surfaces is that
    only tiptop's own rollout loop and websocket server read them. That is checked here rather than
    trusted: a bump that starts reading one on a path the sidecar runs fails, and the key moves to
    ALL_KEYS -- as the placement_* keys did when the pins moved to the TANDEM branches."""
    root = _sources() / "tiptop"
    read = _keys_tiptop_reads(root)
    assert {k for k in tamp_keys.REFUSED if k in read} == set(LOOP_ONLY)

    # Every function, anywhere in tiptop, from which a loop-only reader can be reached, by name.
    defs: dict[str, ast.AST] = {}
    for path in sorted((root / "tiptop").rglob("*.py")):
        for node in _parse(path).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defs.setdefault(node.name, node)
    calls = {name: _called_names(node) for name, node in defs.items()}
    reaches = set(LOOP_ONLY.values())
    while True:
        more = {name for name, called in calls.items() if called & reaches} - reaches
        if not more:
            break
        reaches |= more
    assert {"async_entrypoint", "TiptopPlanningServer"} <= reaches, "the readers moved; re-check the reasons"

    # What the sidecar takes from tiptop: the names it imports, and what it reaches for on tiptop_run.
    sidecar = _parse(sidecar_path())
    used = {
        alias.asname or alias.name
        for node in ast.walk(sidecar)
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "tiptop"
        for alias in node.names
    } | {
        node.attr
        for node in ast.walk(sidecar)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "tiptop_run"
    }
    assert {"get_demo_container", "run_perception", "_planning_robot_types"} <= used
    assert not (used & reaches), f"the sidecar reaches a loop-only reader through {sorted(used & reaches)}"


def test_the_perception_keys_are_tiptops_own_table():
    module = _parse(_sources() / "tiptop" / "tiptop" / "motion_planning.py")
    (table,) = [
        n.value
        for n in module.body
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "_PERCEPTION_OVERRIDE_KEYS" for t in n.targets)
    ]
    theirs = {
        key.value: (tuple(e.value for e in value.elts[0].elts), value.elts[1].id)
        for key, value in zip(table.keys, table.values, strict=True)
    }
    ours = {
        key: (path, tamp_keys.SCALAR_KEYS[key].__name__) for key, path in tamp_keys.PERCEPTION_KEYS.items()
    }
    assert ours == theirs


def test_the_perception_keys_are_rendered_where_the_pinned_tiptop_reads_them():
    root = _sources() / "tiptop" / "tiptop"
    stock = _yaml.load((root / "config" / "tiptop.yml").read_text())
    for key, path in tamp_keys.PERCEPTION_KEYS.items():
        node = stock
        for part in path:
            assert isinstance(node, dict) and part in node, f"tiptop.yml has no {'.'.join(path)} for {key}"
            node = node[part]
    wrapper = (root / "perception_wrapper.py").read_text()
    assert 'cfg.perception.m2t2.get("grasp_threshold"' in wrapper
    assert 'cfg.perception.m2t2.get("num_runs"' in wrapper


def test_the_enumerated_blend_settings_are_the_ones_tiptop_accepts():
    module = _parse(_sources() / "tiptop" / "tiptop" / "trajectory_blending.py")
    fn = _function(module, "resolve_blend_config")
    variable_to_key = {
        "mode": "blend_mode",
        "pace_mode": "blend_pace",
        "boundary_mode": "blend_boundary_mode",
    }
    theirs = {
        variable_to_key[n.left.id]: frozenset(e.value for e in n.comparators[0].elts)
        for n in ast.walk(fn)
        if isinstance(n, ast.Compare)
        and isinstance(n.ops[0], ast.NotIn)
        and isinstance(n.left, ast.Name)
        and n.left.id in variable_to_key
    }
    assert theirs == tamp_keys.ENUMS
    assert "vae" in tamp_keys.ENUMS["blend_mode"]


# --- the sidecar, against the pinned tiptop --------------------------------------------------------


def test_the_sidecar_builds_the_tamp_config_the_way_tiptop_run_does():
    """Knob for knob: the same build_tamp_config keywords, and every override resolver tiptop-run
    calls to build its solvers and TAMP configs, the sidecar calls too."""
    root = _sources() / "tiptop" / "tiptop"
    run_module = _parse(root / "tiptop_run.py")
    theirs = _function(run_module, "_sync_entrypoint")
    ours = _function(_parse(sidecar_path()), "warm", cls="Sidecar")

    (their_build,) = _calls(theirs, "build_tamp_config")
    (our_build,) = _calls(ours, "build_tamp_config")
    assert not our_build.args, "pass build_tamp_config's arguments by name, so this check can read them"
    missing = {k.arg for k in their_build.keywords} - {k.arg for k in our_build.keywords}
    assert not missing, f"tiptop-run configures cuTAMP with {sorted(missing)}, and the sidecar does not"
    assert {k.arg for k in our_build.keywords} == {k.arg for k in their_build.keywords}

    from_motion_planning = {
        alias.asname or alias.name
        for node in run_module.body
        if isinstance(node, ast.ImportFrom) and node.module == "tiptop.motion_planning"
        for alias in node.names
    }
    needed = (_called_names(theirs) & from_motion_planning) - _called_names(ours)
    assert not needed, f"tiptop-run resolves overrides through {sorted(needed)}; the sidecar does not"


def test_the_sidecar_hands_the_planner_its_resolved_solver_config_and_records_it_per_leg():
    sidecar = _parse(sidecar_path())
    warm = _function(sidecar, "warm", cls="Sidecar")
    (container,) = _calls(warm, "get_demo_container")
    # get_demo_container(num_particles, num_spheres, activation, record, overrides, SUMMARY, ...)
    assert not isinstance(container.args[5], ast.Dict), "the solver-config summary is not an empty dict"
    assert _calls(warm, "summarize_curobo_config")

    plan = _function(sidecar, "plan", cls="Sidecar")
    written = [n for n in ast.walk(plan) if isinstance(n, ast.Constant) and n.value == "curobo_config.json"]
    (planning,) = _calls(plan, "run_planning")
    assert written and written[0].lineno < planning.lineno, "the record is written before the leg is planned"


def test_the_pinned_tiptop_and_cutamp_are_a_pair():
    """Why they move together: tiptop imports from cuTAMP, and constructs cuTAMP's config with fields
    only a matching cuTAMP has. A tiptop pinned forward alone is an ImportError or a TypeError at
    warm-up, and cuTAMP's version (0.0.5 in both) cannot tell the two apart."""
    root = _sources()
    cutamp = root / "cuTAMP"

    def module_path(dotted: str) -> Path | None:
        rel = Path(*dotted.split("."))
        for candidate in (cutamp / rel.with_suffix(".py"), cutamp / rel / "__init__.py"):
            if candidate.is_file():
                return candidate
        return None

    def defined(path: Path) -> set[str]:
        names: set[str] = set()
        for node in _parse(path).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, (ast.ImportFrom, ast.Import)):
                names.update((a.asname or a.name).split(".")[0] for a in node.names)
        return names

    missing = []
    for path in sorted((root / "tiptop" / "tiptop").rglob("*.py")):
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "cutamp":
                target = module_path(node.module)
                if target is None:
                    missing.append(f"{node.module} (from {path.name})")
                    continue
                for alias in node.names:
                    if module_path(f"{node.module}.{alias.name}") is None and alias.name not in defined(
                        target
                    ):
                        missing.append(f"{node.module}.{alias.name} (from {path.name})")
    assert not missing, "the pinned tiptop imports what the pinned cuTAMP does not have: " + ", ".join(
        missing
    )

    (klass,) = [
        n
        for n in _parse(cutamp / "cutamp" / "config.py").body
        if isinstance(n, ast.ClassDef) and n.name == "TAMPConfiguration"
    ]
    fields = {n.target.id for n in klass.body if isinstance(n, ast.AnnAssign)}
    planning = _parse(root / "tiptop" / "tiptop" / "planning.py")
    (built,) = _calls(_function(planning, "build_tamp_config"), "TAMPConfiguration")
    passed = {k.arg for k in built.keywords if k.arg}
    assert passed <= fields, f"cuTAMP's TAMPConfiguration has no {sorted(passed - fields)}"
    # The posture keys arrive through **posture_selection, so read them where they are made.
    motion = _parse(root / "tiptop" / "tiptop" / "motion_planning.py")
    posture = {
        n.value
        for n in ast.walk(_function(motion, "resolve_posture_selection"))
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith("posture_")
    }
    assert posture and posture <= fields, f"cuTAMP's TAMPConfiguration has no {sorted(posture - fields)}"


# --- the monorepo's own configs --------------------------------------------------------------------


def _fixture_overrides(name: str) -> dict:
    return dict(_yaml.load((FIXTURES / name).read_text())["tamp_overrides"])


FIXTURE_FILES = sorted(p.name for p in FIXTURES.glob("*.yml"))


def test_the_fixtures_cover_what_the_monorepo_configs_use():
    used = set().union(*(_fixture_overrides(name) for name in FIXTURE_FILES))
    assert {
        "placement_support",
        "grasp_center_weight",
        "grasp_threshold",
        "voxel_downsample_size",
        "m2t2_num_runs",
    } <= used
    assert {
        "transit_apex_height",
        "posture_selection_seeds",
        "blend_vae_sample_target",
        "vae_retiming",
    } <= used


@pytest.mark.parametrize("name", FIXTURE_FILES)
def test_a_monorepo_config_validates_whole(name):
    raw = _fixture_overrides(name)
    out = validate_tamp(raw)
    assert set(out) == set(raw), "nothing the config sets is dropped"
    assert out["blend_mode"] == "vae" and out["traj_length_norm"] == "inf"
    assert out["m2t2_num_runs"] == 60 and isinstance(out["m2t2_num_runs"], int)
    assert out["posture_selection_seeds"] == 12 and out["posture_grasp_roll"] is True
    assert out["vae_retiming"] is False and out["require_m2t2_grasps"] is False
    for key in (k for k in raw if k.startswith("placement_")):
        assert out[key] == raw[key] and type(out[key]) is tamp_keys.SCALAR_KEYS[key], key


def test_the_monorepo_placement_keys_are_the_ones_tandem_accepts():
    """Every placement_* key a monorepo config sets is a setting (none refused, none a typo), and the
    bread/box configs between them set all seven."""
    used = {k for name in FIXTURE_FILES for k in _fixture_overrides(name) if k.startswith("placement_")}
    assert used == {"placement_support", *tamp_keys.PLACEMENT_GATED}
    assert used <= tamp_keys.ALL_KEYS and not used & set(tamp_keys.REFUSED)
    for name in ("1_toy_puzzle_v3.yml", "4_bread_box.yml", "4_bread_box_v3.yml"):
        assert _fixture_overrides(name)["placement_support"] is True, name


# --- validation ------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(LOOP_ONLY))
def test_a_key_only_tiptops_own_loop_reads_is_refused_with_the_reason(key):
    with pytest.raises(ValueError, match="which tandem does not run") as caught:
        validate_tamp({key: True})
    assert "Did you mean" not in str(caught.value), "it is tiptop's key, not a typo"


def test_a_typo_of_a_new_key_is_still_a_typo():
    with pytest.raises(ValueError, match="Did you mean: transit_apex_height"):
        validate_tamp({"transit_apex_hieght": 0.075})


def test_a_null_that_means_something_to_tiptop_is_refused_not_read_as_unset():
    # tiptop: null = try every satisfying particle. A profile: null = unset = 32. Neither is silent.
    with pytest.raises(ValueError, match="try every satisfying particle"):
        validate_tamp({"max_motion_refine_attempts": None})
    assert validate_tamp({"max_motion_refine_attempts": 64}) == {"max_motion_refine_attempts": 64}
    assert validate_tamp({"num_particles": None}) == {}, "any other null still means unset"


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("m2t2_num_runs", 0, "> 0"),
        ("grasp_threshold", 0.0, "> 0"),
        ("voxel_downsample_size", -0.005, "> 0"),
        ("contact_threshold_m", 0.0, "> 0"),
        ("ik_num_seeds", 0, "> 0"),
        ("max_motion_refine_attempts", 0, "> 0"),
        ("posture_pos_tol", 0.0, "> 0"),
        ("blend_boundary_window_sec", 0.0, "> 0"),
        ("transit_apex_height", -0.1, ">= 0"),
        ("transit_apex_min_dist", -0.1, ">= 0"),
        ("posture_selection_seeds", -1, ">= 0"),
        ("grasp_rank_conf_weight", -1.0, ">= 0"),
        ("posture_grasp_roll", "yes", "must be a bool"),
        ("require_m2t2_grasps", 1, "must be a bool"),
    ],
)
def test_the_new_knobs_are_checked_the_way_the_planner_checks_them(key, value, message):
    """Each of these the planner refuses too -- at warm-up or at the first plan, with the arm moving."""
    with pytest.raises(ValueError, match=re.escape(message)):
        validate_tamp({key: value})


def test_the_new_knobs_accept_their_meaningful_edge_values():
    out = validate_tamp(
        {
            "grasp_rank_conf_weight": 0.0,  # rank on soft cost alone
            "transit_apex_height": 0.0,  # apex off
            "posture_selection_seeds": 0,  # selection off
            "posture_ref": "priors/fr3.npz",
            "retime_scale": 1.1,
            "blend_mode": "VAE",
        }
    )
    assert out["grasp_rank_conf_weight"] == 0.0 and out["posture_ref"] == "priors/fr3.npz"
    assert out["blend_mode"] == "VAE", "case is tiptop's to fold, and it does"


# --- rendering and warnings ------------------------------------------------------------------------


def test_perception_knobs_set_in_tamp_land_where_tiptop_reads_them(profile, machine_rig, tmp_path):
    profile.planner.options["tamp"] = validate_tamp(
        {"m2t2_num_runs": 60, "grasp_threshold": 0.02, "voxel_downsample_size": 0.005}
    )
    options = resolve_profile(profile, machine_rig)
    rendered = render.render_tiptop_config(machine_rig, options)
    m2t2 = rendered["perception"]["m2t2"]
    assert m2t2["num_runs"] == 60 and m2t2["grasp_threshold"] == 0.02
    assert m2t2["url"] == options.perception.m2t2.url, "the rest of the block is untouched"
    # The task's tamp value wins over the rig's perception block, as tiptop's own override does.
    assert options.perception.voxel_downsample_size == 0.0075
    assert rendered["perception"]["voxel_downsample_size"] == 0.005
    assert rendered["perception"]["contact_threshold_m"] == options.perception.contact_threshold_m

    # Through the file tiptop actually loads, with the int still an int.
    written = _yaml.load(render.write_tiptop_config(machine_rig, tmp_path / "tiptop.yml", options).read_text())
    assert written["perception"]["m2t2"]["num_runs"] == 60

    # And still passed on with the rest: the sidecar's apply_perception_overrides and the per-leg
    # solver record both read them from there.
    assert render.render_tamp_overrides(profile, options)["m2t2_num_runs"] == 60


def test_unset_perception_knobs_leave_tiptops_own_defaults_in_force(profile, machine_rig):
    profile.planner.options["tamp"] = {}
    rendered = render.render_tiptop_config(machine_rig, resolve_profile(profile, machine_rig))
    m2t2 = rendered["perception"]["m2t2"]
    assert "num_runs" not in m2t2 and "grasp_threshold" not in m2t2


@pytest.mark.parametrize(
    ("tamp", "warning"),
    [
        (
            {"posture_grasp_roll": True},
            "posture_grasp_roll only applies when posture_selection_seeds is above 1",
        ),
        ({"posture_selection_seeds": 1, "posture_rot_tol": 0.1}, "posture_rot_tol only applies"),
        ({"posture_selection_seeds": 12, "posture_ref": "missing.npz"}, "posture_ref does not exist"),
        ({"retime_scale": 1.1}, "retime_scale only applies when vae_retiming is on"),
        ({"vae_retiming": True}, "vae_manifold_weight is 0 or unset"),
        (
            {"vae_retiming": True, "vae_manifold_weight": 25000.0, "blend_trajectory": True},
            "trajectory blending (blend_trajectory and every blend_* key) is switched off",
        ),
        ({"blend_vae_sample_target": True, "blend_mode": "spline"}, "only applies when blend_mode is 'vae'"),
        ({"transit_apex_min_dist": 0.1}, "transit_apex_min_dist only applies when transit_apex_height"),
    ],
)
def test_a_knob_read_only_behind_another_says_when_it_does_nothing(profile, machine_rig, tamp, warning):
    profile.planner.options["tamp"] = validate_tamp(tamp)
    problems = render.check_assets(profile, machine_rig, resolve_profile(profile, machine_rig))
    assert any(warning in p for p in problems), problems
    assert not any(p.startswith("no camera extrinsics") for p in problems)


def test_the_monorepo_v3_settings_raise_none_of_those_warnings(profile, machine_rig):
    raw = _fixture_overrides("4_bread_box_v3.yml")
    profile.planner.options["tamp"] = validate_tamp(raw)
    problems = [
        p
        for p in render.check_assets(profile, machine_rig, resolve_profile(profile, machine_rig))
        if "vae_path does not exist" not in p
    ]
    assert problems == []


# --- the runtime recipe ----------------------------------------------------------------------------


def test_no_patch_touches_what_tiptops_pixi_lock_was_solved_from():
    """The pixi.lock records tiptop's own requires-dist. A patch to tiptop's dependencies (the dropped
    0002 made pyrealsense2 an extra) makes that lock stale, and `pixi install` then re-solves the PyPI
    side instead of installing what tiptop locked. Such a change belongs upstream, lock and all."""
    from tandem.planners.tiptop.recipe import PATCHES, RECIPE

    patches = [p for source in RECIPE.sources for p in source.patches]
    touched = {
        line[len("+++ b/") :].strip()
        for patch in patches
        for line in patch.read_text().splitlines()
        if line.startswith("+++ b/")
    }
    assert touched and not (touched & {"pyproject.toml", "pixi.toml", "pixi.lock"}), touched
    # And nothing ships that the recipe does not apply: a stray patch is package data doing nothing.
    assert sorted(PATCHES.glob("*.patch")) == sorted(patches)
