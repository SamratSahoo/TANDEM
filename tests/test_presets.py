"""WI-17: presets, and the paper's collection settings as one (`tandem profile create NAME --preset paper`).

A preset is split the way a profile is: tandem's half (phase planning) in tandem/resources/presets, a
planner's half (its planner.options) in the directory its factory names. The paper preset's values are
pinned to where they came from -- the monorepo's v3 task configs, kept verbatim as fixtures -- so a
value nobody can trace fails here rather than shipping.
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

import pytest
import tomlkit
from helpers import isolate_registry
from ruamel.yaml import YAML
from toy_planner import ToyPlanner

from tandem import resources
from tandem.core import presets, profiles
from tandem.core.errors import TandemError
from tandem.core.profiles import HitlSpec, Profile
from tandem.planners import registry
from tandem.planners.testing import ConformanceError, check_presets
from tandem.planners.tiptop import tamp_keys
from tandem.planners.tiptop.options import validate_tamp
from tandem.planners.tiptop.recipe import RECIPE

FIXTURES = Path(__file__).parent / "fixtures" / "cfg_tamp"
V3 = sorted(FIXTURES.glob("*_v3.yml"))
SRC = Path(__file__).resolve().parents[1] / "src" / "tandem"
_yaml = YAML(typ="safe")


@pytest.fixture(autouse=True)
def _registry(monkeypatch):
    isolate_registry(monkeypatch)


def _template(name: str = "t") -> Profile:
    return profiles.load_file(resources.path("profile_template.yml"), name=name)


def _paper() -> Profile:
    return presets.apply(_template(), "paper")


def _tamp(profile: Profile) -> dict:
    return dict(profile.planner.options.get("tamp") or {})


def _raw(path: Path) -> dict:
    return _yaml.load(path.read_text())


def _flat(settings: dict) -> dict:
    return {path: new for path, (_, new) in presets.differences({}, settings).items()}


def _toy(presets_dir: Path | None = None, **attributes):
    """A ToyPlanner registered as "toy" whose presets are the files in ``presets_dir``."""
    cls = type("Toy", (ToyPlanner,), {"__module__": __name__, "presets_dir": presets_dir, **attributes})
    registry.register_backend("toy", cls)
    return cls


def _preset(directory: Path, name: str, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yml"
    path.write_text(text)
    return path


# --- the paper preset: it loads, it validates, it changes only what it says -------------------------


def test_the_paper_preset_is_found_for_tiptop_in_two_layers():
    assert "paper" in presets.available("tiptop")
    stack = presets.layers("paper", "tiptop")
    assert [layer.origin for layer in stack] == ["tandem", "tiptop"]
    assert stack[0].path == presets.tandem_dir() / "paper.yml"
    assert stack[1].path == registry.presets_dir("tiptop") / "paper.yml"
    assert stack[1].replace == ("planner.options.tamp",)


def test_the_paper_preset_makes_a_valid_profile_that_survives_a_save():
    profile = _paper()
    assert profile.hitl.enabled and profile.planner.backend == "tiptop"
    profiles.save(profile)
    assert profiles.load(profile.name) == profile


def test_the_paper_preset_differs_from_the_template_only_where_it_says():
    template = _template()
    changes = presets.differences(template.model_dump(mode="python"), _paper().model_dump(mode="python"))

    stated: dict = {}
    for layer in presets.layers("paper", "tiptop"):
        stated = presets.overlay(stated, layer.settings, replace=layer.replace)
    before = _flat(template.model_dump(mode="python"))
    intended = {path: value for path, value in _flat(stated).items() if before.get(path) != value}
    # What `replace` removes: a template override the paper's TAMP block does not state.
    removed = {
        path for path in before if path.startswith("planner.options.tamp.") and path not in _flat(stated)
    }
    assert {path: new for path, (_, new) in changes.items()} == {**intended, **dict.fromkeys(removed)}
    assert all(path.startswith(("hitl.", "planner.options.tamp.")) for path in changes)
    for section in ("task", "recording", "export", "name", "description"):
        assert getattr(_paper(), section) == getattr(template, section), section


def test_the_papers_phase_planning_is_tandems_defaults_switched_on():
    # tandem's defaults are the paper's settings. If one ever changes, this says which preset to revisit.
    changes = presets.differences(HitlSpec().model_dump(), _paper().hitl.model_dump())
    assert changes == {"enabled": (False, True)}
    # And the template new profiles start from is the paper's already: the preset changes nothing in it.
    assert _template().hitl == _paper().hitl and _tamp(_template()) == _tamp(_paper())


def test_tandems_half_states_every_phase_planning_setting():
    # So a profile cloned with --from keeps none of its own under the paper's name.
    (tandem_half, _) = presets.layers("paper", "tiptop")
    assert set(tandem_half.settings) == {"hitl"}
    assert set(tandem_half.settings["hitl"]) == set(HitlSpec.model_fields)


# What the paper's runs had that no v3 config names: LJ1356's tiptop did each unconditionally, and the
# pinned TANDEM branch does each only when asked. The preset asks.
LJ_ON = {"table_plane_support_vote": True, "disjoint_object_masks": True, "blend_stretch_to_caps": True}


def test_the_tamp_half_is_what_all_five_v3_configs_share_plus_what_their_tiptop_always_did():
    # Derived, not restated: the keys every v3 config sets to the same value, less what tandem refuses,
    # plus the switches for what the tiptop those configs ran on did without being asked.
    assert [p.name for p in V3] == [
        "1_toy_puzzle_v3.yml",
        "2_bread_fruit_bowl_cloth_v3.yml",
        "3_pen_open_book_v3.yml",
        "4_bread_box_v3.yml",
        "8c_pp_3bread_cloth_08272026_v3.yml",
    ]
    blocks = [_raw(path)["tamp_overrides"] for path in V3]
    shared = {k: v for k, v in blocks[0].items() if all(k in b and b[k] == v for b in blocks[1:])}
    shared = {k: v for k, v in shared.items() if k not in tamp_keys.REFUSED}
    assert _tamp(_paper()) == validate_tamp({**shared, **LJ_ON})
    # The DATAFARM alignment the paper describes, spelled out.
    tamp = _tamp(_paper())
    assert tamp["vae_manifold_weight"] == 25000 and tamp["blend_mode"] == "vae" and tamp["blend_trajectory"]
    assert tamp["vae_path"] == "vae/checkpoints/vae_full_v2.pt"


def test_the_hitl_half_is_the_v3_hitl_block():
    blocks = [_raw(path)["hitl"] for path in V3]
    assert all(block == blocks[0] for block in blocks), "the five v3 tasks share one hitl block"
    hitl = _paper().hitl
    for key, value in blocks[0].items():
        assert getattr(hitl, key) == value, key


@pytest.mark.parametrize("path", V3, ids=lambda p: p.stem)
def test_each_v3_config_is_the_paper_preset_plus_that_tasks_own_settings(path):
    tamp = validate_tamp({**_raw(path)["tamp_overrides"], **LJ_ON})
    paper = _tamp(_paper())
    extra = set(tamp) - set(paper)
    # The task's own: its grasp and perception tuning, and -- for the puzzle and the bread/box task --
    # the surface-fitted placement its runs were tuned with (tamp_keys.PLACEMENT_GATED and the gate).
    own = {"grasp_center_weight", "grasp_threshold", "voxel_downsample_size", "placement_support"}
    assert extra <= own | set(tamp_keys.PLACEMENT_GATED)
    assert {k: v for k, v in tamp.items() if k not in extra} == paper


def test_the_vae_checkpoint_is_the_one_tandem_ships_where_the_runtime_puts_it(tmp_path):
    from tandem.core.rig import Rig
    from tandem.planners.tiptop import render
    from tandem.planners.tiptop.options import resolve

    vae = next(asset for asset in RECIPE.assets if asset.source.name == "vae_full_v2.pt")
    assert _tamp(_paper())["vae_path"] == vae.dest
    runtime = tmp_path / "runtime"
    (runtime / vae.dest).parent.mkdir(parents=True)
    (runtime / vae.dest).write_bytes(b"")
    profile = _paper()
    options = resolve(Rig(), {}, profile.planner.options)
    assert render.render_tamp_overrides(profile, options, runtime_dir=runtime)["vae_path"] == str(
        (runtime / vae.dest).resolve()
    )


def test_the_paper_preset_raises_none_of_tiptops_warnings(profile, machine_rig):
    # None of check_assets' "this knob does nothing without that one" warnings: the preset is
    # internally consistent. (The checkpoint is only in a built runtime.)
    from tandem.planners.tiptop import render
    from tandem.planners.tiptop.options import resolve_profile

    laid = presets.apply(profile, "paper")
    problems = render.check_assets(laid, machine_rig, resolve_profile(laid, machine_rig))
    assert [p for p in problems if "vae_path does not exist" not in p] == []


# --- laid over a profile that already has settings --------------------------------------------------


def test_the_preset_keeps_the_task_and_replaces_the_experiment():
    base = _template("rig")
    base.planner.options["tamp"]["grasp_center_weight"] = 5.0
    base.hitl.on_robot_phase_failure = "teleop"
    base.hitl.cache_path = "proposals.sqlite"
    base.task.prompt = "stack the cups"
    base = Profile.model_validate(base.model_dump())

    laid = presets.apply(base, "paper")
    assert laid.task.prompt == "stack the cups"
    # The experiment is the paper's, whatever the clone had.
    assert "grasp_center_weight" not in _tamp(laid)
    assert laid.hitl == _paper().hitl
    assert base.hitl.on_robot_phase_failure == "teleop", "the profile passed in is not changed"


# --- a planner's own presets ------------------------------------------------------------------------


def test_a_planner_that_ships_no_preset_of_that_name_gets_tandems_half_only(tmp_path):
    _toy()
    toy = Profile.model_validate(
        {"name": "bins", "planner": {"backend": "toy", "options": {"items": ["duck"]}}}
    )
    assert [layer.origin for layer in presets.layers("paper", "toy")] == ["tandem"]
    laid = presets.apply(toy, "paper")
    assert laid.hitl == _paper().hitl and laid.planner.options == {"items": ["duck"]}


def test_a_planner_ships_its_own_presets_through_its_factory(tmp_path):
    _preset(
        tmp_path / "presets",
        "busy",
        "title: A busy floor\nsummary: five things to drop\nprofile:\n  planner:\n    options:\n"
        "      items: [duck, ball, cup, key, pen]\n",
    )
    _toy(tmp_path / "presets")
    assert registry.presets_dir("toy") == tmp_path / "presets"
    assert sorted(presets.available("toy")) == ["busy", "paper"]
    toy = Profile.model_validate({"name": "bins", "planner": {"backend": "toy"}})
    assert presets.apply(toy, "busy").planner.options == {"items": ["duck", "ball", "cup", "key", "pen"]}
    assert check_presets("toy") == 1


def test_a_planners_preset_is_checked_by_that_planner(tmp_path):
    _preset(
        tmp_path / "presets",
        "bad",
        "title: Bad\nprofile:\n  planner:\n    options:\n      num_particles: 5\n",
    )
    _toy(tmp_path / "presets")
    toy = Profile.model_validate({"name": "bins", "planner": {"backend": "toy"}})
    with pytest.raises(TandemError, match="The 'bad' preset does not make a valid toy profile") as caught:
        presets.apply(toy, "bad")
    assert "num_particles" in caught.value.message and "bad.yml" in caught.value.hint
    with pytest.raises(ConformanceError, match="num_particles"):
        check_presets("toy")


def test_a_planners_preset_named_like_one_of_tandems_must_extend_it(tmp_path):
    _preset(
        tmp_path / "presets",
        "paper",
        "title: Mine\nprofile:\n  planner:\n    options:\n      items: [duck]\n",
    )
    _toy(tmp_path / "presets")
    with pytest.raises(TandemError, match="without extending it") as caught:
        presets.available("toy")
    assert "extends: paper" in caught.value.hint
    with pytest.raises(ConformanceError, match="must extend it"):
        check_presets("toy")


def test_a_planner_preset_extending_nothing_tandem_has_is_refused(tmp_path):
    _preset(tmp_path / "presets", "mine", "title: Mine\nextends: papr\nprofile:\n  planner: {options: {items: [duck]}}\n")
    _toy(tmp_path / "presets")
    with pytest.raises(TandemError, match="did you mean 'paper'"):
        presets.layers("mine", "toy")


def test_tiptops_presets_pass_the_conformance_check():
    assert check_presets("tiptop") == 1


def test_a_planner_without_presets_passes_the_conformance_check():
    _toy()
    assert check_presets("toy") == 0


# --- a preset file ----------------------------------------------------------------------------------


def test_a_malformed_preset_is_refused_with_every_problem_at_once(tmp_path):
    path = _preset(
        tmp_path,
        "odd",
        "titel: x\ncaution: careful\nreplace: [planner.options.tamp]\n"
        "profile:\n  name: other\n  hitll: {}\n  planner: {backend: toy}\n",
    )
    with pytest.raises(TandemError) as caught:
        presets.load(path, origin="toy")
    message = caught.value.message
    for problem in (
        "unknown key 'titel' (did you mean 'title'?)",
        "`title:` must be one line",
        "`caution:` must be a list of lines",
        "it states 'name'",
        "profile.hitll is not a profile setting (did you mean 'hitl'?)",
        "it states planner.backend",
        "`replace:` names 'planner.options.tamp'",
    ):
        assert problem in message, problem


def test_tandems_own_presets_may_not_state_a_planners_options(tmp_path):
    path = _preset(tmp_path, "mine", "title: Mine\nprofile:\n  planner: {options: {tamp: {}}}\n")
    with pytest.raises(TandemError, match="only a planner's own preset may"):
        presets.load(path, origin=presets.TANDEM_ORIGIN)
    path = _preset(tmp_path, "other", "title: Other\nextends: paper\nprofile:\n  hitl: {enabled: true}\n")
    with pytest.raises(TandemError, match="`extends:` is for a planner's preset"):
        presets.load(path, origin=presets.TANDEM_ORIGIN)


def test_a_preset_named_against_the_name_rule_is_refused(tmp_path):
    with pytest.raises(TandemError, match="lowercase letters"):
        presets.load(_preset(tmp_path, "Paper", "title: x\nprofile: {hitl: {}}\n"), origin="tiptop")


def test_an_unknown_preset_is_refused_with_the_nearest_and_the_list():
    with pytest.raises(TandemError, match="no preset named 'papr'.*did you mean 'paper'") as caught:
        presets.get("papr", "tiptop")
    assert "paper" in caught.value.hint and "tandem profile presets --planner tiptop" in caught.value.hint


def test_overlay_merges_mappings_substitutes_lists_and_replaces_what_it_is_told():
    base = {"a": {"x": 1, "y": [1, 2]}, "b": {"keep": 1, "drop": 2}}
    over = {"a": {"y": [3]}, "b": {"keep": 9}}
    assert presets.overlay(base, over) == {"a": {"x": 1, "y": [3]}, "b": {"keep": 9, "drop": 2}}
    assert presets.overlay(base, over, replace=("b",)) == {"a": {"x": 1, "y": [3]}, "b": {"keep": 9}}
    assert base == {"a": {"x": 1, "y": [1, 2]}, "b": {"keep": 1, "drop": 2}}, "nothing passed in changes"


def test_the_preset_files_ship_in_the_wheel():
    pyproject = tomlkit.parse((SRC.parents[1] / "pyproject.toml").read_text())
    globs = [str(g) for g in pyproject["tool"]["setuptools"]["package-data"]["tandem"]]
    shipped = [presets.tandem_dir() / "paper.yml", registry.presets_dir("tiptop") / "paper.yml"]
    for path in shipped:
        relative = path.resolve().relative_to(SRC).as_posix()
        assert any(fnmatch(relative, g) or fnmatch(relative, g.replace("**/", "")) for g in globs), relative


# --- the commands -----------------------------------------------------------------------------------


def test_planners_info_lists_a_planners_own_presets():
    from tandem.cli.planners import info_payload

    payload = info_payload("tiptop")
    assert [p["name"] for p in payload["presets"]] == ["paper"] and payload["presets_problem"] is None
    assert payload["presets"][0]["extends"] == "paper"


def test_planners_info_says_a_broken_preset_rather_than_failing(tmp_path):
    from tandem.cli.planners import info_payload

    _preset(tmp_path / "presets", "odd", "title: x\n")
    _toy(tmp_path / "presets")
    payload = info_payload("toy")
    assert (
        payload["presets"] == []
        and "`profile:` must state at least one setting" in payload["presets_problem"]
    )
