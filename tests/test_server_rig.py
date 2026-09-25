"""The web's rig card: GET /api/rig is `tandem rig show --json`, PATCH /api/rig is `tandem rig set` several keys at a time.

What matters is what a save does to rig.yml, a file people annotate by hand: only the settings sent change,
the comments stay, a default is not written into the file by a save that did not touch it, and a bad value
-- anywhere in the whole rig -- leaves the file exactly as it was, with a 400 that says which line.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tandem.core import paths
from tandem.core import rig as rig_mod
from tandem.server.app import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def test_before_init_it_is_the_defaults_and_says_so(client):
    payload = client.get("/api/rig").json()
    assert payload["exists"] is False and payload["file"] == str(paths.rig_file())
    assert payload["rig"]["robot"] == {"type": "fr3_robotiq", "host": "172.16.0.2"}
    assert all(payload["rig"]["cameras"][role] is None for role in rig_mod.ROLES)
    # TiPToP's machine settings, as it validated them, with the keys it declares and what each is.
    tiptop = payload["planners"]["tiptop"]
    assert set(tiptop["declared"]) == {"robot", "perception"} and tiptop["installed"]
    assert tiptop["options"]["perception"]["foundation_stereo"]["url"] == "http://localhost:1234"
    assert tiptop["set"] == []


def test_a_save_changes_only_what_it_sends_and_keeps_the_comments(client, machine_rig):
    before = paths.rig_file().read_text()
    assert "# the robot computer: the NUC" in before

    response = client.patch("/api/rig", json={"robot.host": "172.16.0.5", "cameras.external_2.serial": "31425515"})
    assert response.status_code == 200, response.text
    assert response.json()["rig"]["robot"]["host"] == "172.16.0.5"

    rig = rig_mod.load(force=True)
    assert rig.robot.host == "172.16.0.5" and rig.cameras.external_2.serial == "31425515"
    assert rig.cameras.hand.serial == "14846828", "what was not sent is as it was"
    text = paths.rig_file().read_text()
    assert "# the robot computer: the NUC" in text and "serial: '31425515'" in text, "a serial stays text"
    assert "planners:" in text and "dof:" not in text, "no default was written into the file"


def test_a_camera_left_blank_is_removed_whole(client, machine_rig):
    response = client.patch("/api/rig", json={"cameras.hand": None})
    assert response.status_code == 200, response.text
    assert rig_mod.load(force=True).cameras.hand is None
    assert response.json()["rig"]["cameras"]["hand"] is None


def test_a_planners_machine_setting_is_changed_by_its_own_key(client, machine_rig):
    response = client.patch("/api/rig", json={"planners.tiptop.perception.m2t2.url": "http://gpu:8123"})
    assert response.status_code == 200, response.text
    tiptop = response.json()["planners"]["tiptop"]
    assert tiptop["options"]["perception"]["m2t2"]["url"] == "http://gpu:8123"
    assert tiptop["set"] == ["perception.m2t2.url"], "only that one is written; the rest stay defaults"


@pytest.mark.parametrize(
    ("changes", "said"),
    [
        ({"robot.host": "http://172.16.0.5:5555"}, "robot.host"),
        ({"cameras.hand.serial": 14846828}, "quote it"),
        ({"robot.hots": "172.16.0.5"}, "did you mean robot.host"),
        ({"planners.tiptop.perception.m2t2.url": "gpu-box"}, "planners.tiptop"),
        ({"planners.tiptopp.robot.port": 1}, "tiptop"),
        ({"version": 2}, "tandem's to write"),
    ],
)
def test_a_bad_change_is_a_400_that_says_where_and_writes_nothing(client, machine_rig, changes, said):
    before = paths.rig_file().read_bytes()
    response = client.patch("/api/rig", json=changes)
    assert response.status_code == 400, response.text
    assert said in response.json()["error"]
    assert paths.rig_file().read_bytes() == before


def test_an_empty_save_is_refused(client, machine_rig):
    response = client.patch("/api/rig", json={})
    assert response.status_code == 400 and "Nothing to change" in response.json()["error"]


def test_a_rig_file_that_does_not_validate_is_a_400_pointing_at_the_editor(client, machine_rig):
    paths.rig_file().write_text(paths.rig_file().read_text().replace("host: 172.16.0.2", "host: http://nuc"))
    response = client.get("/api/rig")
    assert response.status_code == 400
    assert "robot.host" in response.json()["error"] and "tandem rig edit" in response.json()["hint"]


def test_what_stops_collection_is_said_beside_the_rig(client, machine_rig):
    assert client.get("/api/rig").json()["problems"] == [], "the fixture's rig collects"
    payload = client.patch("/api/rig", json={"cameras.external": None}).json()
    problems = {problem["name"]: problem for problem in payload["problems"]}
    assert "cameras.external is not configured" in problems["cameras"]["detail"]
    assert payload["perception_missing"]


def test_the_settings_page_has_the_rig_card_and_saves_only_what_changed():
    from pathlib import Path

    static = Path(__file__).resolve().parents[1] / "src/tandem/server/static"
    page = (static / "pages/settings.js").read_text()
    for needle in ("rigCard()", "api.saveRig", "api.rig()", "Shared by every profile", "flatten("):
        assert needle in page, needle
    assert page.index("rigCard(),") < page.index("credentialsCard(payload, load)"), "the rig card is first"
    api = (static / "api.js").read_text()
    assert 'request("PATCH", "/rig", changes)' in api and 'request("GET", "/rig")' in api
