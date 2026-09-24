"""WI-17: a hitl-tamp-vla task config's `hitl:` block, imported as the profile's own.

The block was read by LJ1356/tiptop cf75a68's HITLConfig. tandem re-implements phase planning, so most
keys carry over as they are; what tandem cannot do -- a learned policy for the human phases, a robot
planner other than cuTAMP -- is refused rather than swapped for something it can. The fixtures are the
monorepo's own configs, copied verbatim (tests/fixtures/cfg_tamp).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from ruamel.yaml import YAML
from typer.testing import CliRunner

from tandem import resources
from tandem.cli import theme
from tandem.core import profiles
from tandem.core.errors import TandemError
from tandem.core.profiles import HitlSpec
from tandem.planners.base import WARNING_NOTE
from tandem.planners.tiptop import importers, tamp_keys
from tandem.planners.tiptop.importers import HITL_KEYS, IMPORTER
from tandem.planners.tiptop.options import options_of

FIXTURES = Path(__file__).parent / "fixtures" / "cfg_tamp"
V3 = sorted(p.name for p in FIXTURES.glob("*_v3.yml"))
_yaml = YAML(typ="safe")

# Every field of HITLConfig at LJ1356/tiptop cf75a68 (tiptop/hitl/config.py), the tiptop hitl-tamp-vla
# pins, with its default. resolve_hitl_config refuses any other key.
CF75A68_HITL_DEFAULTS = {
    "enabled": False,
    "proposal_model": "gemini-2.5-pro",
    "vlm_model": "gemini-2.5-flash",
    "max_attempts": 3,
    "classify_initial": False,
    "verify_retries": 1,
    "verify_enforced": True,
    "check_human_preconditions": False,
    "check_human_effects": True,
    "check_tamp_preconditions": False,
    "check_tamp_effects": False,
    "precondition_enforced": False,
    "check_plan_effects": True,
    "save_vlm_io": True,
    "robot_planner": "cutamp",
    "policy_type": "human",
    "policy_checkpoint": None,
    "open_loop_horizon": 8,
    "policy_num_inference_steps": 10,
    "policy_max_steps": 450,
    "policy_velocity_scale": 1.0,
    "policy_start_joint_angle": None,
    "policy_python": None,
    "cache_path": None,
}


def _raw(name: str) -> dict:
    return _yaml.load((FIXTURES / name).read_text())


def _import(config: Path, name: str = "imported"):
    return IMPORTER.build(name, config=config)


def _write(tmp_path: Path, name: str, hitl, **top) -> Path:
    """A task config like the monorepo's, with its own hitl block."""
    path = tmp_path / name
    data = {"prompt": top.pop("prompt", "put the toy in the box"), "tamp_overrides": {"num_particles": 128}}
    data.update(top)
    if hitl is not ...:
        data["hitl"] = hitl
    with path.open("w") as fh:
        YAML().dump(data, fh)
    return path


def _warnings(notes: list[str]) -> list[str]:
    return [note[len(WARNING_NOTE) :] for note in notes if note.startswith(WARNING_NOTE)]


# --- the mapping -------------------------------------------------------------------------------------


def test_the_mapping_covers_every_key_hitl_tamp_vla_reads_and_nothing_else():
    assert set(HITL_KEYS) == set(CF75A68_HITL_DEFAULTS)


def test_every_key_it_maps_onto_is_a_real_tandem_setting():
    targets = {target for target, _ in HITL_KEYS.values() if target is not None}
    assert targets <= set(HitlSpec.model_fields)
    # One to one where it maps at all: two monorepo keys landing on one tandem key would let the
    # file's order decide which of them wins.
    mapped = [target for target, _ in HITL_KEYS.values() if target is not None]
    assert len(mapped) == len(set(mapped))
    for key, (target, why) in HITL_KEYS.items():
        assert why.strip(), f"{key} says nothing about why it maps as it does"
        if target is not None and key != "policy_type":
            assert target == key, f"{key} is renamed to {target} without a translation"


def test_what_it_carries_over_has_the_same_default_in_both_systems():
    # So a key the block leaves out means the same here as it did there: the import lays the block
    # over the template, whose hitl settings are tandem's defaults.
    template = profiles.load_file(resources.path("profile_template.yml"), name="t")
    assert template.hitl == HitlSpec()
    for key, (target, _) in HITL_KEYS.items():
        if target == key:
            assert getattr(HitlSpec(), key) == CF75A68_HITL_DEFAULTS[key], key
    assert HitlSpec().human_executor == importers.HUMAN_EXECUTORS[CF75A68_HITL_DEFAULTS["policy_type"]]


# --- importing the paper's configs ---------------------------------------------------------------


def test_importing_4_bread_box_v3_preserves_its_hitl_block():
    raw = _raw("4_bread_box_v3.yml")
    profile, _, notes = _import(FIXTURES / "4_bread_box_v3.yml")

    for key, value in raw["hitl"].items():
        assert getattr(profile.hitl, HITL_KEYS[key][0]) == value, key
    # What the block does not say keeps tandem's default, and a person does the human phases.
    stated = {HITL_KEYS[key][0] for key in raw["hitl"]}
    for field in set(HitlSpec.model_fields) - stated:
        assert getattr(profile.hitl, field) == getattr(HitlSpec(), field), field
    assert profile.hitl.enabled and profile.hitl.human_executor == "teleop"

    # The rest of the config came too.
    assert profile.task.prompt == raw["prompt"] and profile.task.target_episodes == 20
    assert profile.export.hf_repo == "4_bread_box_v2"
    tamp = options_of(profile).tamp
    assert tamp["vae_manifold_weight"] == 25000 and tamp["blend_mode"] == "vae"
    assert any("phase planning settings from its hitl block" in note for note in notes)


@pytest.mark.parametrize("name", V3)
def test_every_v3_config_imports_with_phase_planning_on(name):
    raw = _raw(name)
    profile, _, _ = _import(FIXTURES / name)
    assert profile.hitl.model_dump(include=set(raw["hitl"])) == raw["hitl"]
    assert profile.planner.backend == "tiptop"


def test_the_no_hitl_control_imports_with_phase_planning_off():
    # v4 is v3 with only `enabled: false`: the Raw TAMP control.
    profile, _, notes = _import(FIXTURES / "8c_pp_3bread_cloth_08272026_v4.yml")
    assert profile.hitl.enabled is False
    assert profile.hitl.check_plan_effects is True and profile.hitl.verify_retries == 1
    assert not _warnings(notes)
    assert not any("phase planning settings" in note for note in notes)


def test_the_placement_keys_import_as_they_are():
    """The pinned TiPToP reads LJ1356's placement_* keys under the same names, so they carry over 1:1."""
    raw = _raw("4_bread_box_v3.yml")["tamp_overrides"]
    placement = sorted(k for k in raw if k.startswith("placement_"))
    assert len(placement) == 7

    profile, _, notes = _import(FIXTURES / "4_bread_box_v3.yml")
    tamp = options_of(profile).tamp
    assert {k: tamp[k] for k in placement} == {k: raw[k] for k in placement}
    added = set(importers.LJ_BEHAVIOURS)  # raw blends, so all three (test below)
    assert set(tamp) == set(raw) | added, "nothing in the block is dropped; only LJ's three are added"
    assert not any("not imported" in warning for warning in _warnings(notes))


@pytest.mark.parametrize("name", ["1_toy_puzzle_v3.yml", "2_bread_fruit_bowl_cloth_v3.yml"])
def test_an_imported_config_plans_as_its_runs_did(name):
    """LJ1356's tiptop voted for the table by support, built meshes from disjoint masks and slowed
    strokes into the caps unconditionally, for every config and not only the placement ones; the pinned
    TiPToP has a switch for each, off by default. The import turns each on and says so in a note, not a
    warning: nothing is lost, one key per behaviour is added."""
    profile, _, notes = _import(FIXTURES / name)
    tamp = options_of(profile).tamp
    assert {key: tamp[key] for key in importers.LJ_BEHAVIOURS} == dict.fromkeys(importers.LJ_BEHAVIOURS, True)
    (note,) = [note for note in notes if "LJ1356's tiptop, which always" in note]
    assert not note.startswith(WARNING_NOTE)
    assert all(key in note for key in importers.LJ_BEHAVIOURS)
    assert set(importers.LJ_BEHAVIOURS) <= set(tamp_keys.SCALAR_KEYS), "each is a setting a profile can make"


def test_a_switch_a_config_sets_itself_keeps_its_value(tmp_path):
    raw = _raw("4_bread_box_v3.yml")
    raw["tamp_overrides"].update(table_plane_support_vote=True, disjoint_object_masks=False)
    path = tmp_path / "box.yml"
    with path.open("w") as fh:
        YAML().dump(raw, fh)
    profile, _, notes = _import(path)
    tamp = options_of(profile).tamp
    assert tamp["disjoint_object_masks"] is False, "the config's own value stands"
    assert tamp["table_plane_support_vote"] is True and tamp["blend_stretch_to_caps"] is True
    (note,) = [note for note in notes if "LJ1356's tiptop, which always" in note]
    assert "blend_stretch_to_caps" in note
    assert "table_plane_support_vote" not in note and "disjoint_object_masks" not in note

    # Without blending there is nothing for blend_stretch_to_caps to stretch, so it is not added.
    raw["tamp_overrides"].update(blend_trajectory=False, disjoint_object_masks=True)
    with path.open("w") as fh:
        YAML().dump(raw, fh)
    profile, _, notes = _import(path)
    assert "blend_stretch_to_caps" not in options_of(profile).tamp
    assert not any("LJ1356's tiptop, which always" in note for note in notes), "nothing left to add"


def test_a_key_only_tiptops_own_loop_reads_is_left_out_with_a_warning(tmp_path):
    raw = _raw("2_bread_fruit_bowl_cloth_v3.yml")
    raw["tamp_overrides"]["auto_mode"] = True
    path = tmp_path / "bowl.yml"
    with path.open("w") as fh:
        YAML().dump(raw, fh)
    profile, _, notes = _import(path)
    assert "auto_mode" not in options_of(profile).tamp
    (warning,) = _warnings(notes)
    assert "TAMP setting auto_mode not imported" in warning and "which tandem does not run" in warning


def test_a_config_without_placement_keys_imports_with_no_warning():
    _, _, notes = _import(FIXTURES / "2_bread_fruit_bowl_cloth_v3.yml")
    assert _warnings(notes) == []


# --- policy_type ---------------------------------------------------------------------------------


def test_policy_type_human_is_the_teleop_executor(tmp_path):
    config = _write(tmp_path, "task.yml", {"enabled": True, "policy_type": "human"})
    profile, _, notes = _import(config)
    assert profile.hitl.human_executor == "teleop"
    assert not _warnings(notes)


def test_policy_settings_beside_a_human_policy_are_left_out_and_said_to_be(tmp_path):
    config = _write(
        tmp_path,
        "task.yml",
        {"enabled": True, "policy_type": "human", "policy_max_steps": 300, "open_loop_horizon": 8},
    )
    profile, _, notes = _import(config)
    assert profile.hitl.human_executor == "teleop"
    (note,) = [n for n in notes if "policy_max_steps" in n]
    assert "open_loop_horizon" in note and "read only when policy_type names a learned policy" in note
    assert not note.startswith(WARNING_NOTE), "cf75a68 never read them either, so nothing is lost"


def test_a_config_whose_human_phases_a_policy_does_is_refused_and_pointed_at_the_same_task_for_a_person():
    # 3_pen_open_book_diffusion is the HITL-TAMP baseline's config: its dataset is the policy's legs.
    with pytest.raises(TandemError) as caught:
        _import(FIXTURES / "3_pen_open_book_diffusion.yml")
    error = caught.value
    assert "learned diffusion policy" in error.message and "no executor for that" in error.message
    assert "3_pen_open_book_diffusion" in error.message, "it names the dataset it would have mislabelled"
    assert "3_pen_open_book_v3.yml" in error.hint and "--tamp-config" in error.hint
    assert "8c_pp" not in error.hint, "a twin is the same task"


@pytest.mark.parametrize("policy", ["diffusion", "act"])
def test_both_learned_policies_are_refused(tmp_path, policy):
    config = _write(
        tmp_path, f"task_{policy}.yml", {"enabled": True, "policy_type": policy, "policy_checkpoint": "/x"}
    )
    # A same-named config for the same task with phase planning off is not a twin: it runs no human phase.
    _write(tmp_path, "task_v4.yml", {"enabled": False})
    with pytest.raises(TandemError, match=f"learned {policy} policy") as caught:
        _import(config)
    assert "task_v4" not in caught.value.hint
    assert "remove policy_type" in caught.value.hint

    _write(tmp_path, "task_v3.yml", {"enabled": True})
    _write(tmp_path, "task_other.yml", {"enabled": True}, prompt="another task")
    with pytest.raises(TandemError) as caught:
        _import(config)
    assert "task_v3.yml" in caught.value.hint and "task_other" not in caught.value.hint


def test_a_policy_config_with_phase_planning_off_imports_with_a_warning(tmp_path):
    config = _write(
        tmp_path,
        "task_diffusion.yml",
        {"enabled": False, "policy_type": "diffusion", "policy_max_steps": 450},
    )
    profile, _, notes = _import(config)
    assert profile.hitl.enabled is False and profile.hitl.human_executor == "teleop"
    (warning,) = _warnings(notes)
    assert "policy_type 'diffusion'" in warning and "policy_max_steps" in warning
    assert "phase planning is off" in warning


def test_an_unknown_policy_type_is_refused_with_the_nearest_one(tmp_path):
    config = _write(tmp_path, "task.yml", {"enabled": True, "policy_type": "difusion"})
    with pytest.raises(TandemError, match="did you mean 'diffusion'"):
        _import(config)


# --- robot_planner -------------------------------------------------------------------------------


def test_robot_planner_cutamp_is_what_tiptop_runs(tmp_path):
    config = _write(tmp_path, "task.yml", {"enabled": True, "robot_planner": "cutamp"})
    profile, _, notes = _import(config)
    assert profile.planner.backend == "tiptop" and not _warnings(notes)


def test_another_robot_planner_is_refused_when_it_planned_phases(tmp_path):
    config = _write(tmp_path, "task.yml", {"enabled": True, "robot_planner": "pddlstream"})
    with pytest.raises(TandemError, match="'pddlstream'") as caught:
        _import(config)
    assert "cuTAMP" in caught.value.message and "robot_planner: cutamp" in caught.value.hint


def test_another_robot_planner_is_left_out_loudly_when_it_never_ran(tmp_path):
    config = _write(tmp_path, "task.yml", {"enabled": False, "robot_planner": "pddlstream"})
    profile, _, notes = _import(config)
    (warning,) = _warnings(notes)
    assert "robot_planner 'pddlstream' not imported" in warning
    assert profile.planner.backend == "tiptop"


# --- what is refused outright --------------------------------------------------------------------


def test_an_unknown_hitl_key_is_refused_with_the_nearest_real_one(tmp_path):
    config = _write(tmp_path, "task.yml", {"enabled": True, "verify_retry": 2, "zzz": 1})
    with pytest.raises(TandemError) as caught:
        _import(config)
    assert "hitl.verify_retry (did you mean 'verify_retries'?)" in caught.value.message
    assert "hitl.zzz" in caught.value.message, "every unknown key is named, not just the first"
    assert "HITL_KEYS" in caught.value.hint


def test_a_tandem_only_setting_in_the_source_is_unknown_there_too(tmp_path):
    # hitl-tamp-vla never had human_executor, so a config stating it never ran as written.
    config = _write(tmp_path, "task.yml", {"enabled": True, "human_executor": "teleop"})
    with pytest.raises(TandemError, match="hitl.human_executor"):
        _import(config)


def test_a_hitl_block_that_is_not_a_mapping_is_refused(tmp_path):
    config = _write(tmp_path, "task.yml", ["enabled"])
    with pytest.raises(TandemError, match="is a list, not a mapping"):
        _import(config)


def test_a_value_tandem_refuses_is_named_where_it_is(tmp_path):
    config = _write(tmp_path, "task.yml", {"enabled": True, "max_attempts": 0})
    with pytest.raises(TandemError) as caught:
        _import(config)
    assert "hitl.max_attempts" in caught.value.message


def test_a_config_without_a_hitl_block_keeps_the_templates(tmp_path):
    profile, _, notes = _import(_write(tmp_path, "task.yml", ...))
    assert profile.hitl == HitlSpec()


def test_a_relative_cache_path_is_said_to_mean_beside_the_profile(tmp_path):
    config = _write(tmp_path, "task.yml", {"enabled": True, "cache_path": "cache/proposals.sqlite"})
    profile, _, notes = _import(config)
    assert profile.hitl.cache_path == "cache/proposals.sqlite"
    assert any("beside the profile" in note for note in notes)


# --- through the commands ------------------------------------------------------------------------


def test_profile_create_shows_the_importers_warnings_as_warnings(tmp_path):
    from tandem.cli.app import app

    # A key only tiptop's own loop reads is left out with a warning; LJ's three are added with a note.
    raw = _raw("4_bread_box_v3.yml")
    raw["tamp_overrides"]["auto_mode"] = True
    config = tmp_path / "4_bread_box_v3.yml"
    with config.open("w") as fh:
        YAML().dump(raw, fh)
    result = CliRunner().invoke(app, ["profile", "create", "box", "--tamp-config", str(config)])
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert f"{theme.WARN} TAMP setting auto_mode not imported" in output
    assert "LJ1356's tiptop, which always" in output
    assert f"{theme.WARN} set table_plane_support_vote" not in output, "a note, not a warning"
    assert WARNING_NOTE not in output, "the marker is the importer's, not something to print"
    saved = profiles.load("box")
    assert saved.hitl.enabled and saved.hitl.verify_retries == 1
    assert options_of(saved).tamp["placement_flatness_tol"] == 0.012, "stored as imported"


def test_profile_create_refuses_a_policy_config_before_writing_anything():
    from tandem.cli.app import app

    result = CliRunner().invoke(
        app, ["profile", "create", "pen", "--tamp-config", str(FIXTURES / "3_pen_open_book_diffusion.yml")]
    )
    assert result.exit_code != 0
    assert "learned diffusion policy" in result.exception.message
    assert not profiles.exists("pen")


def test_tandem_init_imports_the_hitl_block_of_the_config_it_is_given(tmp_path, monkeypatch):
    import typer

    from tandem.cli import init as init_cli

    rig = tmp_path / "hitl-tamp-vla"
    config_dir = rig / "tiptop" / "tiptop" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "tiptop.yml").write_text("cameras: {perception: external, external: {serial: '111'}}\n")
    tamp_dir = rig / "data-collection" / "cfg" / "tamp"
    tamp_dir.mkdir(parents=True)
    shutil.copy(FIXTURES / "4_bread_box_v3.yml", tamp_dir)

    # The one config there is picked; the task prompt is left as the config has it.
    def prompt(text, default="", **_):
        return "1" if text.strip().startswith("Import one?") else default

    monkeypatch.setattr(typer, "prompt", prompt)
    warned: list[str] = []
    told: list[str] = []
    monkeypatch.setattr(theme, "warn", lambda message, detail="": warned.append(message))
    monkeypatch.setattr(theme, "info", lambda message, detail="": told.append(message))

    init_cli._create_profile("rig", import_from=rig, interactive=True, planner="tiptop")
    profile = profiles.load("rig")
    assert profile.hitl.enabled and profile.hitl.save_vlm_io and profile.hitl.human_executor == "teleop"
    assert profile.cameras.external.serial == "111"
    assert options_of(profile).tamp["placement_support"] is True, "the placement settings came across"
    assert options_of(profile).tamp["table_plane_support_vote"] is True, "LJ's behaviours switched on"
    assert any("LJ1356's tiptop, which always" in message for message in told)
    assert not any("LJ1356's tiptop" in message for message in warned)
    assert not any("not imported" in message for message in warned)
