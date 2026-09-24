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


# --------------------------------------------------------------------------- `tandem profile create`


def _run(*args: str):
    from typer.testing import CliRunner

    from tandem.cli.app import app

    return CliRunner().invoke(app, ["profile", *args])


def _said(result) -> str:
    return " ".join(result.output.split())


def test_create_with_a_prompt_is_one_flag():
    result = _run("create", "my-task", "--prompt", "put the cup on the plate")
    assert result.exit_code == 0, result.output
    assert "Created profile 'my-task' the paper's settings" in _said(result)
    assert "tandem profile edit my-task" in _said(result) and "tandem collect my-task" in _said(result)
    assert profiles.load("my-task").task.prompt == "put the cup on the plate"
    assert settings_mod.load(force=True).active_profile != "my-task", "only --use makes it active"


def test_create_without_a_task_says_to_give_one():
    result = _run("create", "my-task")
    assert result.exit_code == 1
    assert "needs its task" in result.exception.message and "--prompt" in result.exception.hint
    assert not profiles.exists("my-task")


def test_create_at_a_terminal_asks_for_the_task(monkeypatch):
    from tandem.cli import profile as profile_cli

    asked = []
    monkeypatch.setattr(profile_cli.theme, "is_tty", lambda: True)
    monkeypatch.setattr(profile_cli.typer, "prompt", lambda text, **kw: asked.append(text) or "open the box")
    result = _run("create", "box")
    assert result.exit_code == 0, result.output
    assert asked == ["  Task prompt"] and profiles.load("box").task.prompt == "open the box"
    # A copy has its task already: nothing is asked.
    assert _run("create", "box-2", "--from", "box").exit_code == 0 and len(asked) == 1


def test_create_from_one_of_the_papers_five_and_use_it():
    result = _run("create", "rolls", "--from", "cover-bread-rolls", "--use")
    assert result.exit_code == 0, result.output
    assert "a copy of cover-bread-rolls" in _said(result)
    assert settings_mod.load(force=True).active_profile == "rolls"
    assert profiles.load("rolls").task.prompt == profiles.load_file(
        profiles.builtin_path("cover-bread-rolls"), name="x"
    ).task.prompt


def test_create_replaces_only_with_force():
    assert _run("create", "x", "--prompt", "one").exit_code == 0
    again = _run("create", "x", "--prompt", "two")
    assert again.exit_code == 1 and "--force" in again.exception.hint
    assert _run("create", "x", "--prompt", "two", "--force").exit_code == 0
    assert profiles.load("x").task.prompt == "two"


@pytest.mark.parametrize(
    "flags",
    [["--preset", "paper"], ["--import-from", "/tmp"], ["--tamp-config", "x.yml"], ["--planner", "tiptop"]],
)
def test_the_flags_that_are_gone_are_refused(flags):
    result = _run("create", "x", "--prompt", "p", *flags)
    assert result.exit_code == 2 and "No such option" in result.output
    assert not profiles.exists("x")


def test_there_is_no_presets_command_and_an_empty_list_says_where_profiles_come_from():
    gone = _run("presets")
    assert gone.exit_code == 2 and "No such command" in gone.output
    empty = _run("list")
    assert empty.exit_code == 0
    assert "tandem init" in empty.output and "add the paper's five tasks" in _said(empty)
    assert "tandem profile create NAME --prompt" in _said(empty)


def test_the_web_writes_no_profile_without_its_task():
    from fastapi.testclient import TestClient

    from tandem.server.app import create_app

    client = TestClient(create_app())
    refused = client.post("/api/profiles", json={"name": "fresh"})
    assert refused.status_code == 400 and "needs its task" in refused.json()["error"]
    assert not profiles.exists("fresh")
    made = client.post("/api/profiles", json={"name": "fresh", "prompt": "sort the bins"})
    assert made.status_code == 200, made.text
    assert profiles.load("fresh").task.prompt == "sort the bins"
