"""Making a profile: the paper's settings with a task of your own, or a copy of another profile.

    tandem profile create NAME --prompt "..."     the template (what the paper's five tasks share), with that task
    tandem profile create NAME --from PROFILE     a copy of one of yours, or of the paper's five
"""

from __future__ import annotations

import pytest
from helpers import isolate_registry
from ruamel.yaml import YAML
from toy_planner import ToyPlanner

from tandem import resources
from tandem.core import profiles
from tandem.core import settings as settings_mod
from tandem.core.errors import ProfileError
from tandem.planners import registry

_yaml = YAML(typ="safe")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    isolate_registry(monkeypatch)


def _template_tamp() -> dict:
    return profiles.load_file(resources.path(profiles.TEMPLATE), name="t").planner.options["tamp"]


# --------------------------------------------------------------------------- profiles.create


def test_a_new_profile_is_the_papers_settings_with_its_own_task():
    made = profiles.create("my-task", prompt="put the cup on the plate")
    assert made.task.prompt == "put the cup on the plate"
    assert made.hitl.enabled and made.planner.options["tamp"] == _template_tamp()

    text = profiles.path_of("my-task").read_text()
    assert text.startswith("# my-task: made by `tandem profile create` with the TANDEM paper's settings on ")
    assert "rig" in text.splitlines()[1]
    raw = _yaml.load(text)
    assert raw["version"] == 3 and raw["description"] == "" and "name" not in raw
    assert "goal: null" in text, "written as the template writes it, not as a bare `goal:`"
    # The template's own comments come along; its header, which says what the TEMPLATE is, does not.
    assert "# DATAFARM: plan motions that look like the DROID teleoperation data" in text
    assert "starts every new profile from this file" not in text
    for status in profiles.STATUSES:
        assert (profiles.trajectories_root() / "my-task" / status).is_dir()
    assert profiles.load("my-task") == made


def test_a_new_profile_needs_its_task():
    with pytest.raises(ProfileError) as info:
        profiles.create("my-task")
    assert "needs its task" in info.value.message and "--prompt" in info.value.hint
    with pytest.raises(ProfileError):
        profiles.create("my-task", prompt="   ")
    assert not profiles.exists("my-task")


def test_one_of_the_papers_five_is_copied_before_init_has_added_them():
    made = profiles.create("bread", source="store-bread-in-closed-box")
    assert made.description == "copied from store-bread-in-closed-box"
    assert made.task.prompt == "place the bread on the plate, and then open the box and place the bread in the box"
    assert made.planner.options["tamp"]["placement_fill_occluded"] is True
    text = profiles.path_of("bread").read_text()
    assert "as a copy of store-bread-in-closed-box" in text.splitlines()[0]
    assert "# this task's own: place into the box's tray, fitted to its observed floor" in text
    assert not profiles.exists("store-bread-in-closed-box"), "copying one does not add it"


def test_a_copy_takes_a_new_prompt_and_keeps_everything_else():
    profiles.seed_builtins()
    source = profiles.path_of("cover-bread-rolls")
    source.write_text(source.read_text().replace("target_episodes: 20", "target_episodes: 40"))
    made = profiles.create("rolls-2", source="cover-bread-rolls", prompt="cover the 2 rolls with the cloth")
    assert made.task.prompt == "cover the 2 rolls with the cloth"
    assert made.task.target_episodes == 40, "the profile here, as edited, not the packaged one"
    assert made.planner.options == profiles.load("cover-bread-rolls").planner.options


def test_a_prompt_yaml_would_misread_is_written_so_it_reads_back():
    made = profiles.create("odd", prompt="step 1: open the box # then close it")
    assert profiles.load("odd").task.prompt == made.task.prompt == "step 1: open the box # then close it"
    long = "first, " + "place the bread in the blue bowl, " * 6 + "and stop"
    profiles.create("long", prompt=long)
    assert profiles.load("long").task.prompt == long
    assert f"prompt: {long}" in profiles.path_of("long").read_text(), "one line, not folded at 100 columns"


def test_an_existing_profile_is_replaced_only_with_force():
    profiles.create("x", prompt="one")
    with pytest.raises(ProfileError) as info:
        profiles.create("x", prompt="two")
    assert "--force" in info.value.hint
    assert profiles.load("x").task.prompt == "one"
    profiles.create("x", prompt="two", force=True)
    assert profiles.load("x").task.prompt == "two"


def test_a_copy_of_nothing_says_what_there_is():
    with pytest.raises(ProfileError) as info:
        profiles.create("x", source="store-bread-in-closed-bx")
    assert "Did you mean 'store-bread-in-closed-box'?" in info.value.hint
    with pytest.raises(ProfileError):
        profiles.create("../x", prompt="p")


def test_a_new_task_on_another_planner_starts_from_that_planners_defaults():
    registry.register_backend("toy", ToyPlanner)
    made = profiles.create("shelf", prompt="stack the blocks", planner="toy")
    assert made.planner.backend == "toy" and "tamp" not in made.planner.options
    assert made.hitl.enabled, "phase planning is tandem's, whichever planner runs"

    cfg = settings_mod.load()
    cfg.default_planner = "toy"
    settings_mod.save(cfg)
    assert profiles.create("shelf-2", prompt="stack them").planner.backend == "toy"
    # A copy is its source as written, planner and all.
    assert profiles.create("copy", source="cover-bread-rolls").planner.backend == "tiptop"
