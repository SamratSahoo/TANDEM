"""The HTTP API, including the parts a browser is fussy about."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tandem.server.app import create_app


@pytest.fixture
def client(profile):
    from tandem.core import settings as settings_mod

    cfg = settings_mod.load()
    cfg.active_profile = profile.name
    settings_mod.save(cfg)
    return TestClient(create_app())


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_list_profiles(client, profile):
    payload = client.get("/api/profiles").json()
    assert payload["active"] == profile.name
    assert [p["name"] for p in payload["profiles"]] == [profile.name]
    assert payload["profiles"][0]["valid"] is True


def test_get_profile_includes_the_resolved_overrides(client, profile):
    payload = client.get(f"/api/profiles/{profile.name}").json()
    assert payload["profile"]["name"] == profile.name
    # Exactly what the planner will receive — the answer to "did my override apply?".
    assert payload["tamp_overrides"]["traj_length_norm"] == "inf"


def test_update_profile_rejects_an_unknown_tamp_key(client, profile):
    body = profile.model_dump(mode="json")
    body["tamp"] = {"nonsense_weight": 1}
    response = client.put(f"/api/profiles/{profile.name}", json=body)
    assert response.status_code == 400
    assert "nonsense_weight" in response.json()["error"]


def test_update_profile_persists(client, profile):
    body = profile.model_dump(mode="json")
    body["task"]["prompt"] = "a brand new task"
    assert client.put(f"/api/profiles/{profile.name}", json=body).status_code == 200
    assert client.get(f"/api/profiles/{profile.name}").json()["prompt"] == "a brand new task"


def test_unknown_profile_is_404(client):
    response = client.get("/api/profiles/nope")
    assert response.status_code == 404
    assert "hint" in response.json()


def test_trajectory_index_and_series(client, profile, make_trajectory):
    make_trajectory(profile, "2026-01-01_00-00-00", n_frames=120)
    listing = client.get(f"/api/trajectories?profile={profile.name}").json()
    assert listing["counts"]["success"] == 1
    assert listing["trajectories"][0]["n_frames"] == 120

    series = client.get(f"/api/trajectories/{profile.name}/2026-01-01_00-00-00/series").json()
    assert len(series["t"]) == 120
    assert series["filter"]["n_frames"] == 120


def test_relabel_and_delete(client, profile, make_trajectory):
    make_trajectory(profile, "2026-01-01_00-00-00", status="eval")
    response = client.post(
        f"/api/trajectories/{profile.name}/2026-01-01_00-00-00/relabel", json={"status": "success"}
    )
    assert response.json()["counts"]["success"] == 1

    response = client.delete(f"/api/trajectories/{profile.name}/2026-01-01_00-00-00")
    assert response.json()["counts"]["success"] == 0


def test_media_supports_range_requests(client, profile, make_trajectory):
    """Chrome will not scrub an mp4 without Accept-Ranges and a 206 — a video you cannot
    seek makes the whole review view useless."""
    directory = make_trajectory(profile, "2026-01-01_00-00-00")
    (directory / "external_cam.mp4").write_bytes(b"x" * 5000)

    url = f"/api/media/{profile.name}/2026-01-01_00-00-00/external_cam.mp4"

    full = client.get(url)
    assert full.status_code == 200
    assert full.headers["accept-ranges"] == "bytes"

    partial = client.get(url, headers={"range": "bytes=100-199"})
    assert partial.status_code == 206
    assert partial.headers["content-range"] == "bytes 100-199/5000"
    assert len(partial.content) == 100

    suffix = client.get(url, headers={"range": "bytes=-50"})
    assert suffix.status_code == 206
    assert len(suffix.content) == 50

    unsatisfiable = client.get(url, headers={"range": "bytes=99999-"})
    assert unsatisfiable.status_code == 416


def test_media_rejects_traversal(client, profile, make_trajectory):
    """An escaping path must never return file content. It cannot even match the route (the
    decoded slashes make it too many segments), so it lands on the api-404."""
    directory = make_trajectory(profile, "2026-01-01_00-00-00")
    secret = directory.parent.parent.parent / "profile.yml"
    assert secret.is_file()

    response = client.get(f"/api/media/{profile.name}/2026-01-01_00-00-00/..%2F..%2F..%2Fprofile.yml")
    assert response.status_code == 404
    assert "profile.yml" not in response.text or "No such endpoint" in response.text


def test_unknown_api_path_is_a_json_404(client):
    """Not the HTML app: a typo'd endpoint must not look like a request whose JSON failed
    to parse."""
    response = client.get("/api/nope")
    assert response.status_code == 404
    assert response.json()["error"].startswith("No such endpoint")


def test_settings_never_returns_a_secret(client, monkeypatch):
    from tandem.core import secrets

    secrets.set_gemini_api_key("AIzaSyTOPSECRETVALUE1234567890")
    payload = client.get("/api/settings").json()
    body = repr(payload)
    assert "TOPSECRET" not in body
    assert payload["credentials"]["gemini"]["source"] == "file"
    assert "…" in payload["credentials"]["gemini"]["masked"]


def test_secrets_can_be_set_but_not_read_back(client):
    response = client.put("/api/settings/secrets", json={"gemini_api_key": "AIzaSyNEWKEY0987654321XYZ"})
    assert response.status_code == 200
    assert "NEWKEY" not in repr(response.json())

    from tandem.core import secrets

    assert secrets.gemini_api_key() == "AIzaSyNEWKEY0987654321XYZ"


def test_runtime_status(client):
    payload = client.get("/api/runtime").json()
    assert payload["ready"] is False
    assert payload["problems"]


def test_doctor_endpoint(client):
    payload = client.get("/api/doctor").json()
    assert payload["summary"]["ok"] >= 1
    assert any(check["name"] == "profile" for check in payload["checks"])


def test_spa_is_served(client):
    index = client.get("/")
    assert index.status_code == 200
    assert "<div id=\"root\">" in index.text
    assert client.get("/app.js").status_code == 200
    assert client.get("/pages/collect.js").status_code == 200
    assert client.get("/theme.css").status_code == 200


def test_unknown_path_falls_through_to_the_spa(client):
    """The client-side router owns the URL space, so a deep link must not 404."""
    response = client.get("/some/client/route")
    assert response.status_code == 200
    assert "<div id=\"root\">" in response.text
