"""The profile store under the ways it was found to break: from a URL, from a planner, from an uninstall.

Each test here failed on the branch as it was reviewed:

- a profile name from a URL became a path, so ``DELETE /api/profiles/%2E%2E?purge=true`` removed the data
  root;
- ``save`` truncated profile.yml before serialising it, and an empty file then loaded as a profile of
  defaults -- another robot address, another planner, the template's task -- without a word;
- a profile.yml that is not a mapping took down the listing meant to show which profile is broken;
- a version-1 profile carried its old ``on_robot_phase_failure: teleop`` default over silently;
- an executor had no settings a profile could hold;
- a profile naming a planner or executor this machine has not installed could not even be browsed.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from helpers import FakeFactory, isolate_registry
from typer.testing import CliRunner

from tandem.cli.app import app
from tandem.core import profiles
from tandem.core import settings as settings_mod
from tandem.core.errors import ProfileError, TandemError
from tandem.executors import base as executors
from tandem.planners import registry
from tandem.server.app import create_app

FIXTURES = Path(__file__).parent / "fixtures" / "profiles"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    isolate_registry(monkeypatch)
    monkeypatch.setattr(executors, "_registered", dict(executors._registered))
    monkeypatch.setattr(executors, "_discovered", None)


def _client(profile) -> TestClient:
    cfg = settings_mod.load()
    cfg.active_profile = profile.name
    settings_mod.save(cfg)
    return TestClient(create_app())


def _write(name: str, text: str) -> Path:
    path = profiles.profiles_root() / name / "profile.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# --- a name from a URL never becomes a path outside profiles/ -----------------------------------------


@pytest.mark.parametrize("segment", ["%2E", "%2E%2E"])
@pytest.mark.parametrize("purge", [True, False])
def test_a_dot_segment_cannot_delete_the_profiles_or_the_data_root(profile, segment, purge):
    client = _client(profile)
    precious = settings_mod.load().resolved_data_root() / "precious.txt"
    precious.write_text("keep me")
    before = profiles.list_names()

    response = client.delete(f"/api/profiles/{segment}" + ("?purge=true" if purge else ""))
    assert response.status_code in (400, 404), response.text
    assert precious.read_text() == "keep me"
    assert profile.profile_file().is_file()
    assert profiles.list_names() == before


@pytest.mark.parametrize("name", [".", "..", "../x", "a/b", "test\n"])
def test_delete_exists_and_load_refuse_what_is_not_a_profile_name(profile, name):
    with pytest.raises(ProfileError):
        profiles.delete(name, keep_data=False)
    with pytest.raises(ProfileError):
        profiles.load(name)
    assert not profiles.exists(name)
    assert profile.profile_file().is_file()


def test_a_real_profile_still_deletes_and_a_soft_deleted_one_can_still_be_purged(profile):
    client = _client(profile)
    other = profile.model_copy(deep=True)
    other.name = "other"
    profiles.save(other)
    assert client.delete("/api/profiles/other").status_code == 200
    assert not profiles.exists("other") and other.dir().is_dir(), "data kept"
    assert client.delete("/api/profiles/other?purge=true").status_code == 200
    assert not other.dir().exists()
    assert profile.profile_file().is_file()


# --- a save that cannot finish never leaves an empty profile -------------------------------------------


class _Colour(str, Enum):
    RED = "red"


class _PathPlanner(FakeFactory):
    """A planner whose validate_options normalises into values YAML cannot write."""

    def __init__(self, value) -> None:
        super().__init__("pathy")
        self._value = value

    def validate_options(self, options):
        return {**dict(options or {}), "scene": self._value}


@pytest.mark.parametrize("value", [Path("/x/scene"), _Colour.RED], ids=["path", "enum"])
def test_options_a_profile_cannot_store_are_refused_and_the_file_is_untouched(profile, value):
    registry.register_backend("pathy", _PathPlanner(value))
    before = profile.profile_file().read_bytes()

    changed = profile.model_copy(deep=True)
    changed.planner = profiles.PlannerSpec.model_construct(backend="pathy", options={})
    with pytest.raises(TandemError, match="options.scene"):
        profiles.save(changed)
    assert profile.profile_file().read_bytes() == before

    # And a save after the refusal writes a whole file: nothing was left half-written to reuse.
    again = profiles.load(profile.name)
    again.task.prompt = "still here"
    profiles.save(again)
    assert profiles.load(profile.name).task.prompt == "still here"


def test_an_interrupted_save_leaves_the_old_file_and_no_partial(profile, monkeypatch):
    before = profile.profile_file().read_bytes()

    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt

    changed = profile.model_copy(deep=True)
    changed.task.prompt = "never written"
    with monkeypatch.context() as patch:
        patch.setattr(profiles, "_new_yaml", lambda: SimpleNamespace(dump=interrupted))
        with pytest.raises(KeyboardInterrupt):
            profiles.save(changed)
    assert profile.profile_file().read_bytes() == before
    assert not list(profile.dir().glob(".*.partial"))


@pytest.mark.parametrize("text", ["", "# only a comment\n"], ids=["empty", "comment"])
def test_an_empty_profile_file_is_an_error_not_a_profile_of_defaults(profile, text):
    profile.profile_file().write_text(text)
    with pytest.raises(ProfileError, match="is empty"):
        profiles.load(profile.name)


@pytest.mark.parametrize("text", ["- oops\n", "just a string\n"])
def test_a_profile_file_that_is_not_a_mapping_is_refused_and_listed_as_invalid(profile, text):
    _write("zeta", text)
    with pytest.raises(ProfileError, match="mapping"):
        profiles.load("zeta")

    listed = CliRunner().invoke(app, ["profile", "list", "--json"])
    assert listed.exit_code == 0, listed.output
    rows = {row["name"]: row for row in json.loads(listed.output)}
    assert rows["zeta"]["valid"] is False and rows[profile.name]["valid"] is True

    cards = {c["name"]: c for c in _client(profile).get("/api/profiles").json()["profiles"]}
    assert cards["zeta"]["valid"] is False and cards[profile.name]["valid"] is True


def test_the_conformance_kit_refuses_options_that_are_not_plain_data():
    from tandem.planners.testing import PlannerConformance

    registry.register_backend("pathy", _PathPlanner(_Colour.RED))
    kit = type("Kit", (PlannerConformance,), {"factory": staticmethod(lambda: registry.factory("pathy"))})
    with pytest.raises(TandemError, match="cannot store: options.scene"):
        kit().test_its_options_check_accepts_what_it_returns()


# --- a version-1 profile's teleop default is said, not carried over silently ----------------------------


def test_a_version_1_profiles_teleop_default_is_named_in_the_migration_notice(isolated_env, caplog):
    text = (FIXTURES / "v1_ef1411f.yml").read_text()
    assert "on_robot_phase_failure: teleop" in text
    _write("old", text)
    with caplog.at_level("WARNING"):
        loaded = profiles.load("old")
    assert loaded.hitl.on_robot_phase_failure == "teleop", "kept: a migration cannot tell a choice from a default"
    assert "on_robot_phase_failure" in caplog.text and "abort" in caplog.text

    migrated = CliRunner().invoke(app, ["profile", "migrate", "old"])
    assert migrated.exit_code == 0, migrated.output
    assert "on_robot_phase_failure" in migrated.output


def test_a_version_1_profile_that_chose_abort_or_a_version_2_teleop_is_not_mentioned(isolated_env, caplog):
    _write("chose", (FIXTURES / "v1_ef1411f.yml").read_text().replace("on_robot_phase_failure: teleop", "on_robot_phase_failure: abort"))
    current = profiles.load_file(Path(profiles.__file__).parents[1] / "resources" / "profile_template.yml", name="cur")
    current.hitl.on_robot_phase_failure = "teleop"
    profiles.save(current)
    with caplog.at_level("WARNING"):
        profiles.load("chose")
        profiles.load("cur")
    assert "on_robot_phase_failure" not in caplog.text


# --- an executor's own settings ------------------------------------------------------------------------


class _Policy:
    name = "diffusion"
    segment_source = "policy"
    display_name = "A policy"
    summary = "Runs a checkpoint."

    def __init__(self, ctx) -> None:
        self.ctx = ctx

    def run(self, request, leg, *, save_root, should_stop):
        raise NotImplementedError

    def kill(self) -> None:
        return None

    @staticmethod
    def validate_options(options):
        if "checkpoint" not in options:
            raise TandemError("checkpoint is required")
        return {"horizon": 8, **options}


def test_an_executor_reads_its_own_options_from_the_profile_and_checks_them(profile):
    executors.register_human_executor("diffusion", _Policy)
    profile.hitl.human_executor_options = {"diffusion": {"checkpoint": "policy.ckpt"}}
    profiles.save(profile)
    loaded = profiles.load(profile.name)
    assert loaded.hitl.human_executor_options == {"diffusion": {"checkpoint": "policy.ckpt", "horizon": 8}}

    profile.hitl.human_executor_options = {"diffusion": {"horizon": 4}}
    with pytest.raises(ProfileError, match="human_executor_options.*checkpoint is required"):
        profiles.save(profile)


def test_the_phase_loop_hands_each_executor_its_own_block_and_teleop_none(profile, tmp_path):
    from tandem.core.phase_loop import PhaseLoop
    from tandem.executors.base import ExecutorContext
    from tandem.planning.config import PlanningConfig

    executors.register_human_executor("diffusion", _Policy)
    loop = PhaseLoop.__new__(PhaseLoop)
    loop._executors = {}
    loop.cfg = PlanningConfig(human_executor_options={"diffusion": {"checkpoint": "c"}})
    loop.executor_context = ExecutorContext(profile=profile, session_dir=tmp_path)
    assert loop._executor("diffusion").ctx.options == {"checkpoint": "c"}
    # The same context builds teleop for an operator's hand-off: it gets no policy's settings.
    assert dict(loop._executor("teleop").ctx.options) == {}


def test_an_absent_executors_options_are_kept_and_survive_switching_executors(profile):
    data = profiles.read_data(profile.profile_file())
    data["hitl"]["human_executor_options"] = {"notinstalled": {"checkpoint": "x.ckpt"}}
    profiles.save(profiles.Profile.model_validate(data))
    assert profiles.load(profile.name).hitl.human_executor_options == {"notinstalled": {"checkpoint": "x.ckpt"}}

    executors.register_human_executor("diffusion", _Policy)
    from tandem.cli import executors as executors_cli

    executors_cli.use_executor("diffusion", profile_name=profile.name)
    executors_cli.use_executor("teleop", profile_name=profile.name)
    assert profiles.load(profile.name).hitl.human_executor_options == {"notinstalled": {"checkpoint": "x.ckpt"}}


# --- a profile collected with a plugin this machine does not have -----------------------------------------


@pytest.fixture
def shelf(profile, make_trajectory):
    """A profile naming a planner and an executor nobody installed here, with one trajectory."""
    data = profiles.read_data(profile.profile_file())
    data["planner"] = {"backend": "shelfbot", "options": {"bins": ["a", "b"]}}
    data["hitl"]["human_executor"] = "armsim"
    path = _write("shelf1", "x: 1\n")
    import io

    buf = io.StringIO()
    profiles._new_yaml().dump({**data, "name": "shelf1"}, buf)
    path.write_text(buf.getvalue())
    make_trajectory(SimpleNamespace(status_dir=lambda s: path.parent / "trajectories" / s), "2026-01-01_00-00-00")
    return "shelf1"


def test_a_profile_with_an_absent_plugin_is_browsed_listed_and_exported(shelf, profile):
    loaded = profiles.load(shelf, require_installed=False)
    assert loaded.planner.backend == "shelfbot" and loaded.planner.options == {"bins": ["a", "b"]}
    assert loaded.hitl.human_executor == "armsim"

    for args in (["traj", "list", shelf], ["export", "manifest", shelf], ["profile", "show", shelf]):
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, (args, result.output)
    assert "not installed" in CliRunner().invoke(app, ["profile", "show", shelf]).output

    listed = {r["name"]: r for r in json.loads(CliRunner().invoke(app, ["profile", "list", "--json"]).output)}
    assert listed[shelf]["valid"] and listed[shelf]["missing"] == ["planner shelfbot", "executor armsim"]

    client = _client(profile)
    assert client.get("/api/trajectories", params={"profile": shelf}).status_code == 200
    assert client.get(f"/api/profiles/{shelf}").status_code == 200


def test_it_is_still_refused_wherever_the_planner_would_be_built(shelf):
    with pytest.raises(ProfileError, match="no planner named 'shelfbot' is installed on this machine"):
        profiles.load(shelf)
    # A typo is still a typo, with the nearest name.
    with pytest.raises(ValueError, match="did you mean 'tiptop'"):
        profiles.PlannerSpec(backend="tiptopp")


def test_an_invalid_profile_is_422_and_a_missing_one_404(shelf, profile):
    client = _client(profile)
    _write("broken", "hitl:\n  max_attempts: 0\n")
    assert client.get("/api/profiles/broken").status_code == 422
    assert client.get("/api/profiles/nope").status_code == 404
