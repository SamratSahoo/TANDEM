"""The paper's five tasks as profiles, and the template every new profile starts from.

Each built-in profile is the settings the paper collected that task with, and the source of truth is the
task's config in hitl-tamp-vla (tests/fixtures/cfg_tamp/*_v3.yml, copied verbatim): its prompt, its
`hitl:` block and its `tamp_overrides:`, plus the three switches LJ1356's tiptop always had on (so every v3
run had them). Derived from the fixtures here, not restated, so a hand edit of a profile that drifts from
the paper fails.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers import builtin_path
from ruamel.yaml import YAML

from tandem import resources
from tandem.core import profiles
from tandem.core.errors import ProfileError
from tandem.core.profiles import HitlSpec
from tandem.planners.tiptop import tamp_keys
from tandem.planners.tiptop.options import validate_tamp

FIXTURES = Path(__file__).parent / "fixtures" / "cfg_tamp"
PACKAGED = Path(resources.__file__).parent / "profiles"
_yaml = YAML(typ="safe")

#: Each built-in profile, its task in the paper (Fig. 3), and the v3 config it was collected with.
PAPER = {
    "cover-bread-rolls": ("Cover Bread Rolls", 1, "8c_pp_3bread_cloth_08272026_v3.yml"),
    "solve-constrained-puzzle": ("Solve Constrained Puzzle", 2, "1_toy_puzzle_v3.yml"),
    "sort-and-cover-snacks": ("Sort & Cover Snacks", 3, "2_bread_fruit_bowl_cloth_v3.yml"),
    "open-obstructed-book": ("Open Obstructed Book", 4, "3_pen_open_book_v3.yml"),
    "store-bread-in-closed-box": ("Store Bread in Closed Box", 5, "4_bread_box_v3.yml"),
}

#: What LJ1356's tiptop did unconditionally, and the pinned TANDEM branch does only when asked.
LJ3 = {"table_plane_support_vote": True, "disjoint_object_masks": True, "retime_stretch_to_caps": True}

#: What a v3 config says about the machine it ran on, none of which is a task's.
RIG_KEYS = {"cameras", "robot", "perception", "calibration", "extrinsics"}


def _config(name: str) -> dict:
    return _yaml.load((FIXTURES / PAPER[name][2]).read_text())


def _raw(path: Path) -> dict:
    return _yaml.load(path.read_text())


def _builtin(name: str) -> profiles.Profile:
    return profiles.load_file(builtin_path(name), name=name)


def _template() -> profiles.Profile:
    return profiles.load_file(resources.path(profiles.TEMPLATE), name="template")


# --------------------------------------------------------------------------- the five


def test_the_five_are_the_papers_tasks_in_the_papers_order():
    assert profiles.BUILTIN == tuple(PAPER)
    assert sorted(p.stem for p in PACKAGED.glob("*.yml")) == sorted(profiles.BUILTIN)
    for name, (title, number, _source) in PAPER.items():
        assert _builtin(name).description == f"{title}: task {number} of the TANDEM paper"


@pytest.mark.parametrize("name", list(PAPER))
def test_a_built_in_profile_has_its_configs_prompt_and_episode_count(name):
    config, profile = _config(name), _builtin(name)
    assert profile.task.prompt == config["prompt"]
    assert profile.task.target_episodes == config["num_episodes"]
    assert profile.task.goal is None, "no v3 config has a tamp_prompt"
    # The configs' hugginface_slug names the lab's own datasets; a user's export goes to a repo of theirs.
    assert profile.export.hf_repo == ""


@pytest.mark.parametrize("name", list(PAPER))
def test_a_built_in_profile_plans_with_its_configs_hitl_block_and_the_papers_defaults(name):
    config, profile = _config(name), _builtin(name)
    hitl = profile.hitl.model_dump(mode="python")
    stated = config["hitl"]
    assert {key: hitl[key] for key in stated} == stated
    defaults = HitlSpec().model_dump(mode="python")
    assert {key: hitl[key] for key in defaults if key not in stated} == {
        key: value for key, value in defaults.items() if key not in stated
    }
    # Pinned to what hitl-tamp-vla's code (cf75a68) ran with, not only to tandem's defaults: a later change
    # of a default must not drift the paper's tasks without a test saying so.
    assert hitl["max_attempts"] == 3 and hitl["cache_path"] is None
    # The two keys that follow the paper rather than that code, and say so in the file.
    assert hitl["on_verification_failure"] == "exclude" and hitl["verify_final_phase"] is True
    # Every key is written, so a reader sees the whole of what the paper ran with.
    assert set(_raw(builtin_path(name))["hitl"]) == set(HitlSpec.model_fields)


@pytest.mark.parametrize("name", list(PAPER))
def test_a_built_in_profiles_tamp_settings_are_its_configs_plus_the_three_switches(name):
    config, profile = _config(name), _builtin(name)
    assert profile.planner.backend == "tiptop"
    assert profile.planner.options["tamp"] == validate_tamp({**config["tamp_overrides"], **LJ3})
    assert set(profile.planner.options) == {"tamp"}, "a task's options only: the robot and perception are the rig's"
    assert set(profile.planner.options["tamp"]) <= tamp_keys.ALL_KEYS


@pytest.mark.parametrize("name", list(PAPER))
def test_a_built_in_profile_holds_nothing_of_the_machine(name):
    raw = _raw(builtin_path(name))
    assert not RIG_KEYS & set(raw)
    assert not RIG_KEYS & set(raw["planner"]["options"])
    assert raw["version"] == profiles.LAYOUT_VERSION and "name" not in raw


@pytest.mark.parametrize("name", [*PAPER, "template"])
def test_the_encoder_checkpoint_is_the_one_tandem_ships_where_the_runtime_puts_it(name, tmp_path):
    # encoder_path is relative: render finds it in the runtime, where the recipe puts the checkpoint the wheel
    # ships. A path that drifted from the recipe's would be "encoder_path does not exist" on every machine.
    from tandem.core.rig import Rig
    from tandem.planners.tiptop import render
    from tandem.planners.tiptop.options import resolve
    from tandem.planners.tiptop.recipe import RECIPE

    profile = _template() if name == "template" else _builtin(name)
    vae = next(asset for asset in RECIPE.assets if asset.source.name == "vae_full_v2.pt")
    assert profile.planner.options["tamp"]["encoder_path"] == vae.dest
    runtime = tmp_path / "runtime"
    (runtime / vae.dest).parent.mkdir(parents=True)
    (runtime / vae.dest).write_bytes(b"")
    options = resolve(Rig(), {}, profile.planner.options)
    rendered = render.render_tamp_overrides(profile, options, runtime_dir=runtime)
    assert rendered["encoder_path"] == str((runtime / vae.dest).resolve())


@pytest.mark.parametrize("name", [*PAPER, "template"])
def test_a_built_in_profile_raises_none_of_tiptops_warnings(name, machine_rig):
    # None of check_assets' "this knob does nothing without that one" warnings: each is internally
    # consistent. (The checkpoint is only there once a runtime is built.)
    from tandem.planners.tiptop import render
    from tandem.planners.tiptop.options import resolve_profile

    profile = _template() if name == "template" else _builtin(name)
    problems = render.check_assets(profile, machine_rig, resolve_profile(profile, machine_rig))
    assert [p for p in problems if "encoder_path does not exist" not in p] == []


@pytest.mark.parametrize("name", list(PAPER))
def test_a_built_in_profile_says_which_task_it_is_and_where_its_values_came_from(name):
    title, number, source = PAPER[name]
    text = builtin_path(name).read_text()
    header = text[: text.index("version:")]
    assert f"{title}  —  task {number} of the TANDEM paper" in header
    assert source in header and "hitl-tamp-vla" in header
    assert "rig" in header, "and that the robot and cameras are not here"
    flat = " ".join(line.lstrip("# ") for line in header.splitlines())
    assert "on_verification_failure: exclude (its code asked for a label instead)" in flat
    assert "verify_final_phase: true (its code left the last step to that label)" in flat
    assert "which are the paper's" not in flat, "the unset keys are not all what the paper's code did"


# --------------------------------------------------------------------------- the template


def _common_tamp() -> dict:
    """The tamp_overrides every one of the five configs sets, to the same value."""
    blocks = [_config(name)["tamp_overrides"] for name in PAPER]
    return {key: value for key, value in blocks[0].items() if all(key in b and b[key] == value for b in blocks)}


def test_the_template_is_what_the_five_share():
    template = _template()
    assert len(_common_tamp()) == 25
    assert template.planner.options["tamp"] == validate_tamp({**_common_tamp(), **LJ3})
    for name in PAPER:
        assert _builtin(name).hitl == template.hitl
    # Each built-in is the template plus its own task: the keys it adds are exactly its config's extras.
    for name in PAPER:
        extra = set(_builtin(name).planner.options["tamp"]) - set(template.planner.options["tamp"])
        assert extra == set(_config(name)["tamp_overrides"]) - set(_common_tamp())


def test_the_template_has_no_task_until_one_is_given():
    template = _template()
    assert template.task.prompt == "describe the task here"
    assert template.description == ""


def test_a_built_in_profile_is_the_template_with_its_own_task():
    # The same comments and order, so a copy of any of them reads like a new profile does.
    body = profiles._without_header(resources.read(profiles.TEMPLATE))
    for name in PAPER:
        own = profiles._without_header(builtin_path(name).read_text())
        kept = [line for line in body.splitlines() if not line.startswith(("description:", "  prompt:"))]
        lines = own.splitlines()
        assert all(line in lines for line in kept), name


# --------------------------------------------------------------------------- seeding


def test_seeding_copies_the_five_word_for_word():
    assert profiles.list_names() == []
    assert profiles.seed_builtins() == list(profiles.BUILTIN)
    assert profiles.list_names() == sorted(profiles.BUILTIN)
    for name in profiles.BUILTIN:
        assert profiles.path_of(name).read_text() == builtin_path(name).read_text()
        assert (profiles.trajectories_root() / name / "success").is_dir()
        profiles.load(name)


def test_seeding_never_overwrites_and_brings_back_a_deleted_one():
    profiles.seed_builtins()
    edited = profiles.path_of("cover-bread-rolls")
    edited.write_text(edited.read_text().replace("target_episodes: 20", "target_episodes: 50"))
    profiles.delete("open-obstructed-book")

    assert profiles.seed_builtins() == ["open-obstructed-book"]
    assert profiles.load("cover-bread-rolls").task.target_episodes == 50
    assert profiles.exists("open-obstructed-book")
    assert profiles.seed_builtins() == []


def test_a_profile_to_copy_is_yours_first_then_the_papers():
    assert "Cover Bread Rolls" in profiles.source_text("cover-bread-rolls"), "before init: the packaged copy"
    profiles.seed_builtins()
    path = profiles.path_of("cover-bread-rolls")
    path.write_text(path.read_text().replace("target_episodes: 20", "target_episodes: 7"))
    assert "target_episodes: 7" in profiles.source_text("cover-bread-rolls")

    with pytest.raises(ProfileError) as info:
        profiles.source_text("cover-bread-roll")
    assert "cover-bread-rolls" in info.value.hint and "Did you mean" in info.value.hint
