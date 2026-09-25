"""The web's profiles page: the paper's five and your own, one YAML file each, made as `tandem profile create`
makes them.

POST /api/profiles takes a name and a task (a new task on the paper's settings) or a profile to copy -- one
here, or one of the paper's five before `tandem init` has copied them in -- and nothing else. What it writes
is what the CLI writes, comments included, because it is the same function (``profiles.create``).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tandem import resources
from tandem.core import profiles
from tandem.core import settings as settings_mod
from tandem.server.app import create_app

STATIC = Path(__file__).resolve().parents[1] / "src/tandem/server/static"


@pytest.fixture
def client(machine_rig):
    return TestClient(create_app())


def _template() -> profiles.Profile:
    return profiles.load_file(resources.path(profiles.TEMPLATE), name="template")


def test_a_new_task_is_the_papers_settings_with_its_prompt(client):
    response = client.post("/api/profiles", json={"name": "cups", "prompt": "put the cup on the plate"})
    assert response.status_code == 200, response.text
    card = response.json()
    assert card["name"] == "cups" and card["prompt"] == "put the cup on the plate" and card["builtin"] is False
    made, template = profiles.load("cups"), _template()
    assert made.hitl == template.hitl and made.hitl.enabled
    assert made.planner == template.planner and made.description == ""
    text = profiles.path_of("cups").read_text()
    assert text.startswith("# cups: made by `tandem profile create` with the TANDEM paper's settings")
    assert "# What to do" in text, "the template's comments come with it"
    assert (profiles.trajectories_root() / "cups" / "success").is_dir()


def test_a_paper_task_is_copied_before_init_has_copied_it_in(client):
    assert not profiles.exists("store-bread-in-closed-box")
    response = client.post("/api/profiles", json={"name": "my-box", "from": "store-bread-in-closed-box"})
    assert response.status_code == 200, response.text
    copy = profiles.load("my-box")
    paper = profiles.load_file(profiles.builtin_path("store-bread-in-closed-box"), name="x")
    assert copy.task == paper.task and copy.planner == paper.planner, "its own task and placement settings"
    assert copy.description == "copied from store-bread-in-closed-box"
    assert "# this task's own" in profiles.path_of("my-box").read_text()


def test_a_copy_takes_a_new_task_when_given_one(client, profile):
    response = client.post("/api/profiles", json={"name": "other", "from": profile.name, "prompt": "stack the cups"})
    assert response.status_code == 200, response.text
    other = profiles.load("other")
    assert other.task.prompt == "stack the cups" and other.planner == profile.planner


@pytest.mark.parametrize(
    ("body", "status", "said"),
    [
        ({"prompt": "x"}, 400, "needs a name"),
        ({"name": "Fold Cloth", "prompt": "x"}, 400, "is not a profile name"),
        ({"name": "../up", "prompt": "x"}, 400, "is not a profile name"),
        ({"name": "fresh"}, 400, "needs its task"),
        ({"name": "fresh", "prompt": "   "}, 400, "needs its task"),
        ({"name": "fresh", "from": "cover-bread-roll"}, 404, "no profile 'cover-bread-roll' to copy"),
        ({"name": "fresh", "prompt": "x", "planner": "tiptop"}, 400, "does not take planner"),
    ],
)
def test_what_the_form_can_fix_is_said_and_nothing_is_written(client, body, status, said):
    response = client.post("/api/profiles", json=body)
    assert response.status_code == status, response.text
    assert said in response.json()["error"]
    assert profiles.list_names() == []


def test_a_name_already_taken_is_refused_rather_than_replaced(client, profile):
    before = profile.file().read_text()
    response = client.post("/api/profiles", json={"name": profile.name, "prompt": "x"})
    assert response.status_code == 400 and "already exists" in response.json()["error"]
    assert profile.file().read_text() == before


def test_the_listing_names_the_papers_five_and_the_profiles_not_yet_moved(client):
    old = profiles.profiles_root() / "bread-box"
    old.mkdir(parents=True)
    shutil.copy(Path(__file__).parent / "fixtures" / "profiles" / "v2_2797491.yml", old / "profile.yml")
    payload = client.get("/api/profiles").json()
    assert payload["builtin"] == list(profiles.BUILTIN)
    assert payload["old_layout"] == ["bread-box"] and payload["profiles"] == []


def test_the_papers_five_are_added_as_init_adds_them(client):
    settings = settings_mod.load()
    settings.active_profile = "gone"
    settings_mod.save(settings)

    added = client.post("/api/profiles/builtin").json()
    assert added == {"added": list(profiles.BUILTIN), "active": "cover-bread-rolls", "held_back": []}
    cards = {card["name"]: card for card in client.get("/api/profiles").json()["profiles"]}
    assert set(cards) == set(profiles.BUILTIN) and all(card["builtin"] for card in cards.values())
    assert cards["cover-bread-rolls"]["active"]

    path = profiles.path_of("open-obstructed-book")
    path.write_text(path.read_text().replace("target_episodes: 20", "target_episodes: 9"))
    assert client.post("/api/profiles/builtin").json() == {"added": [], "active": "cover-bread-rolls", "held_back": []}
    assert profiles.load("open-obstructed-book").task.target_episodes == 9, "never over one that is here"


def test_the_page_creates_from_a_task_or_a_copy_and_says_what_is_not_shown():
    page = (STATIC / "pages/profiles.js").read_text()
    for needle in (
        "the paper's settings (a new task)",
        "state.builtin",
        "state.oldLayout",
        "tandem profile migrate",
        "api.addPaperProfiles",
        "needs its task",
        'span.chip.violet", { title: profile.description',
    ):
        assert needle in page, needle
    assert "preset" not in page.lower() and "cams" not in page
    shell = (STATIC / "app.js").read_text()
    assert "state.builtin = payload.builtin" in shell and "state.oldLayout = payload.old_layout" in shell


def test_a_new_profile_is_made_active_when_none_is_and_a_delete_falls_back_to_none(client):
    assert settings_mod.load().active_profile == ""
    assert client.post("/api/profiles", json={"name": "cups", "prompt": "stack the cups"}).json()["active"]
    assert settings_mod.load(force=True).active_profile == "cups"
    assert not client.post("/api/profiles", json={"name": "bowls", "prompt": "stack the bowls"}).json()["active"]
    assert client.delete("/api/profiles/cups").json()["active"] == "bowls"
    assert client.delete("/api/profiles/bowls").json()["active"] == ""
