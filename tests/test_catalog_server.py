"""The planner and executor catalogs over HTTP, for the settings page.

The page must never tell a different story from the terminal, so the API returns the very payloads
`tandem planners list --json` and `tandem executors list --json` print -- checked here by comparing the
two -- and choosing a planner or an executor from the page is the same call as `use`. Installing is not
an endpoint; the page shows the command instead, and that command is checked to be there.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from helpers import FakeFactory, isolate_registry
from typer.testing import CliRunner

from tandem.cli.app import app
from tandem.core import profiles
from tandem.core import settings as settings_mod
from tandem.executors import base as executors
from tandem.planners import registry
from tandem.server.app import create_app

SETTINGS_PAGE = Path(__file__).resolve().parents[1] / "src/tandem/server/static/pages/settings.js"


class ReadyExecutor:
    segment_source = "policy"
    display_name = "Ready policy"
    summary = "Always ready."

    def __init__(self, ctx) -> None:
        self.ctx = ctx


@pytest.fixture(autouse=True)
def clean_registries(monkeypatch):
    isolate_registry(monkeypatch)
    monkeypatch.setattr(executors, "_registered", {})
    monkeypatch.setattr(executors, "metadata", SimpleNamespace(entry_points=lambda group: []))
    monkeypatch.setattr(executors, "_discovered", None)
    registry.register_backend("pure", FakeFactory("pure"))
    executors.register_human_executor("ready", ReadyExecutor)


@pytest.fixture
def client(profile):
    cfg = settings_mod.load()
    cfg.active_profile = profile.name
    settings_mod.save(cfg)
    return TestClient(create_app())


def _cli_json(*args: str) -> dict:
    result = CliRunner().invoke(app, list(args))
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def test_the_page_reads_the_catalogs_the_terminal_prints(client, profile):
    planners = client.get("/api/planners").json()
    assert planners == _cli_json("planners", "list", "--json")
    rows = {row["name"]: row for row in planners["planners"]}
    assert rows["tiptop"]["active"] and rows["tiptop"]["status"] == "not installed"
    # Installing is a terminal's job; the page is handed the exact command.
    assert rows["tiptop"]["install_command"] == "tandem planners install tiptop"
    assert rows["pure"]["status"] == "no runtime needed" and rows["pure"]["install_command"] is None

    executors_payload = client.get("/api/executors").json()
    assert executors_payload == _cli_json("executors", "list", "--json")
    rows = {row["name"]: row for row in executors_payload["executors"]}
    assert rows["teleop"]["active"] and rows["teleop"]["status"] == "needs setup"
    assert rows["ready"]["status"] == "ready"

    info = client.get("/api/planners/tiptop").json()
    assert info == _cli_json("planners", "info", "tiptop", "--json")
    assert info["capabilities"]["supports"]["movable_restriction"] is True


def test_a_profile_is_switched_from_the_page_the_way_use_switches_it(client, profile):
    other = profile.model_copy(deep=True)
    other.name = "other"
    profiles.save(other)

    response = client.post("/api/planners/pure/use", json={"profile": "other"})
    assert response.status_code == 200, response.text
    assert response.json()["changed"] is True and response.json()["profile"] == "other"
    assert profiles.load("other").planner.backend == "pure"
    assert profiles.load(profile.name).planner.backend == "tiptop"

    # No body: the active profile.
    assert client.post("/api/planners/pure/use").status_code == 200
    assert profiles.load(profile.name).planner.backend == "pure"
    assert client.get("/api/planners").json()["profile_planner"] == "pure"

    response = client.post("/api/executors/ready/use", json={"profile": profile.name})
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "ready"
    assert profiles.load(profile.name).hitl.human_executor == "ready"


def test_the_default_for_new_profiles_is_set_without_touching_any_profile(client, profile):
    response = client.post("/api/planners/pure/default")
    assert response.status_code == 200 and response.json() == {"default_planner": "pure"}
    assert settings_mod.load(force=True).default_planner == "pure"
    assert profiles.load(profile.name).planner.backend == "tiptop"

    created = client.post("/api/profiles", json={"name": "fresh", "prompt": "sort the bins"})
    assert created.status_code == 200, created.text
    assert profiles.load("fresh").planner.backend == "pure"


def test_a_name_nothing_provides_is_a_400_with_the_nearest_and_the_listing(client):
    for method, path in (
        ("get", "/api/planners/tiptopp"),
        ("post", "/api/planners/tiptopp/use"),
        ("post", "/api/planners/tiptopp/default"),
    ):
        response = getattr(client, method)(path)
        assert response.status_code == 400, path
        assert response.json()["error"] == "Unknown planner backend 'tiptopp'."
        assert "tandem planners list" in response.json()["hint"]

    response = client.post("/api/executors/teleopp/use")
    assert response.status_code == 400
    assert "tandem executors list" in response.json()["hint"]

    # A profile that is not there is the profile's 404, as everywhere else in the API.
    assert client.post("/api/planners/pure/use", json={"profile": "nope"}).status_code == 404


def test_the_settings_page_shows_both_catalogs_and_the_install_command():
    page = SETTINGS_PAGE.read_text()
    for needle in ("/planners", "/executors", "install_command", "/default", "Human executors", "Planners"):
        assert needle in page, needle
