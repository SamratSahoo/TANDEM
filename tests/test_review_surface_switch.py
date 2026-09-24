"""Switching a profile's planner or executor, and choosing the default: what the review found wrong.

- `planners use` / `executors use` could not repair the profile every error sent people to them for: they
  loaded it first, and a profile naming a planner or executor this machine no longer has does not load.
- A planner whose ``validate_options`` requires a setting crashed `planners use` and `profile create
  --planner` with a pydantic traceback, and the web with a 500; and nothing could supply the setting.
- A switch threw the old planner's options away for good -- the rig's robot address, a preset's TAMP
  settings -- and the web asked nothing and said nothing.
- `planners use NAME --default` was the hint for "set the default", and also switched and emptied the
  active profile.
- Every listing said only "<path> is not a valid profile:", with the reason cut off.
- `profile create --from X --planner Y` ignored --planner, even a name nothing provides.
- Settings were saved from a stale in-process copy, undoing a terminal's change; and a key nobody set was
  written, breaking an older tandem on the same machine.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomlkit
from fastapi.testclient import TestClient
from helpers import FakeFactory, isolate_registry
from typer.testing import CliRunner

from tandem.cli import executors as executors_cli
from tandem.cli import planners as planners_cli
from tandem.cli.app import app
from tandem.core import paths, presets, profiles
from tandem.core import settings as settings_mod
from tandem.core.errors import ProfileError, TandemError
from tandem.executors import base as executors
from tandem.planners import registry
from tandem.server.app import create_app

SETTINGS_PAGE = Path(__file__).resolve().parents[1] / "src/tandem/server/static/pages/settings.js"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    isolate_registry(monkeypatch)
    monkeypatch.setattr(executors, "_registered", dict(executors._registered))
    monkeypatch.setattr(executors, "metadata", SimpleNamespace(entry_points=lambda group: []))
    monkeypatch.setattr(executors, "_discovered", None)


@pytest.fixture
def active(profile):
    cfg = settings_mod.load()
    cfg.active_profile = profile.name
    settings_mod.save(cfg)
    return profile


def _run(*args: str):
    return CliRunner().invoke(app, list(args))


def _rewrite(profile, change) -> None:
    """Edit a profile's file as written, the way a person or an uninstall leaves it."""
    import io

    data = profiles.read_data(profile.profile_file())
    change(data)
    buf = io.StringIO()
    profiles._new_yaml().dump(data, buf)
    profile.profile_file().write_text(buf.getvalue())


# --- `use` is the repair ---------------------------------------------------------------------------


def test_use_switches_a_profile_away_from_a_planner_that_was_uninstalled(active):
    registry.register_backend("gone", FakeFactory("gone"))
    planners_cli.use_planner("gone")
    registry.unregister_backend("gone")
    with pytest.raises(ProfileError):
        profiles.load(active.name)

    result = planners_cli.use_planner("tiptop")
    assert result["changed"] and result["previous"] == "gone"
    assert profiles.load(active.name).planner.backend == "tiptop"


def test_use_switches_away_from_options_the_planner_no_longer_accepts(active):
    registry.register_backend("pure", FakeFactory("pure"))
    _rewrite(active, lambda d: d["planner"]["options"].update(bogus_key=1))
    with pytest.raises(ProfileError, match="bogus_key"):
        profiles.load(active.name)
    result = planners_cli.use_planner("pure")
    assert result["dropped_options"]["bogus_key"] == 1
    assert profiles.load(active.name).planner.backend == "pure"


def test_the_web_repairs_the_same_profile(active):
    registry.register_backend("gone", FakeFactory("gone"))
    planners_cli.use_planner("gone")
    registry.unregister_backend("gone")
    response = TestClient(create_app()).post("/api/planners/tiptop/use", json={"profile": active.name})
    assert response.status_code == 200, response.text
    assert profiles.load(active.name).planner.backend == "tiptop"


def test_use_still_refuses_a_profile_broken_somewhere_else(active):
    registry.register_backend("pure", FakeFactory("pure"))
    _rewrite(active, lambda d: d["hitl"].update(max_attempts=0))
    with pytest.raises(ProfileError, match="max_attempts"):
        planners_cli.use_planner("pure")
    with pytest.raises(ProfileError, match="max_attempts"):
        executors_cli.use_executor("teleop")
    assert "max_attempts: 0" in active.profile_file().read_text(), "nothing was written"


def test_executors_use_repairs_a_profile_whose_executor_was_uninstalled(active):
    _rewrite(active, lambda d: d["hitl"].update(human_executor="policyx"))
    with pytest.raises(ProfileError, match="policyx"):
        profiles.load(active.name)

    listed = _run("executors", "list", "--json")
    assert listed.exit_code == 0
    problem = json.loads(listed.output)["profile_problem"]
    assert "Unknown human executor 'policyx'" in problem, "the reason, not only the header"

    result = executors_cli.use_executor("teleop")
    assert result["changed"] and result["previous"] == "policyx"
    assert profiles.load(active.name).hitl.human_executor == "teleop"


def test_one_uninstalled_name_does_not_block_switching_the_other(active):
    # The executor is gone; switching the planner must not be refused over it, nor fix it silently.
    registry.register_backend("pure", FakeFactory("pure"))
    _rewrite(active, lambda d: d["hitl"].update(human_executor="policyx"))
    planners_cli.use_planner("pure")
    assert profiles.load(active.name, require_installed=False).planner.backend == "pure"
    assert profiles.load(active.name, require_installed=False).hitl.human_executor == "policyx"


def test_every_listing_says_why_a_profile_does_not_load(active):
    from tandem.cli.doctor import collect_checks

    _rewrite(active, lambda d: d["planner"].update(backend="nosuchplanner", options={}))
    for problem in (
        planners_cli.catalog_payload()["profile_problem"],
        executors_cli.catalog_payload()["profile_problem"],
    ):
        assert "planner.backend" in problem and "nosuchplanner" in problem
        assert not problem.rstrip().endswith(":")
    row = {c.name: c for c in collect_checks(profile_name=active.name, probe_hardware=False)}["profile"]
    assert "planner.backend" in row.detail


def test_the_listing_offers_use_to_repair_and_default_for_new_profiles(active):
    _rewrite(active, lambda d: d["planner"].update(backend="nosuchplanner", options={}))
    shown = _run("planners", "list")
    assert shown.exit_code == 0, shown.output
    assert "tandem planners use NAME" in shown.output
    assert "tandem planners default NAME" in shown.output
    assert "use NAME --default" not in shown.output


# --- a planner that requires a setting ---------------------------------------------------------------


class _NeedsHost(FakeFactory):
    OPTIONS = {"robot_ip": "the robot's address"}

    def __init__(self) -> None:
        super().__init__("req")

    def validate_options(self, options):
        options = dict(options or {})
        if "robot_ip" not in options:
            raise TandemError("planner.options.robot_ip is required: the robot's address")
        return options


def test_a_planner_that_requires_a_setting_is_refused_loudly_and_given_it_with_option(active):
    registry.register_backend("req", _NeedsHost())
    before = active.profile_file().read_text()

    refused = _run("planners", "use", "req")
    assert refused.exit_code == 1
    assert isinstance(refused.exception, TandemError), "not a pydantic traceback"
    assert "robot_ip" in refused.exception.message and "--option" in refused.exception.hint
    assert active.profile_file().read_text() == before

    given = _run("planners", "use", "req", "--option", "robot_ip=10.0.0.2")
    assert given.exit_code == 0, given.output
    assert profiles.load(active.name).planner.options == {"robot_ip": "10.0.0.2"}


def test_the_web_answers_400_not_500_and_takes_the_options(active):
    registry.register_backend("req", _NeedsHost())
    client = TestClient(create_app())
    refused = client.post("/api/planners/req/use", json={"profile": active.name})
    assert refused.status_code in (400, 404, 422) and refused.status_code != 500
    assert "robot_ip" in refused.json()["error"]
    given = client.post("/api/planners/req/use", json={"profile": active.name, "options": {"robot_ip": "10.0.0.3"}})
    assert given.status_code == 200, given.text
    assert profiles.load(active.name).planner.options == {"robot_ip": "10.0.0.3"}


def test_creating_a_profile_for_it_is_refused_loudly_and_writes_nothing(isolated_env):
    registry.register_backend("req", _NeedsHost())
    created = _run("profile", "create", "p2", "--planner", "req")
    assert created.exit_code == 1
    assert isinstance(created.exception, TandemError) and "robot_ip" in created.exception.message
    assert not profiles.exists("p2")


def test_parse_options_types_values_and_nests_dotted_keys():
    assert planners_cli.parse_options(["a=1", "b=true", "c=[x, y]", "robot.host=10.0.0.2", "s=hello"]) == {
        "a": 1,
        "b": True,
        "c": ["x", "y"],
        "robot": {"host": "10.0.0.2"},
        "s": "hello",
    }
    with pytest.raises(TandemError, match="KEY=VALUE"):
        planners_cli.parse_options(["nope"])


def test_the_conformance_kit_accepts_a_required_setting_said_as_an_error_but_not_a_crash():
    from tandem.planners.testing import ConformanceError, PlannerConformance

    registry.register_backend("req", _NeedsHost())
    kit = type("Kit", (PlannerConformance,), {"factory": staticmethod(lambda: registry.factory("req"))})
    kit().test_its_options_check_handles_no_options()

    class Crashes(_NeedsHost):
        def validate_options(self, options):
            return {"robot_ip": options["robot_ip"]}  # KeyError on {}

    registry.register_backend("crash", type("C", (Crashes,), {"__init__": lambda self: FakeFactory.__init__(self, "crash")})())
    crash = type("Kit", (PlannerConformance,), {"factory": staticmethod(lambda: registry.factory("crash"))})
    with pytest.raises(ConformanceError, match="KeyError"):
        crash().test_its_options_check_handles_no_options()


# --- a switch sets the old planner's options aside, and a switch back restores them -----------------------


def _customised(profile) -> None:
    profile.planner = profiles.PlannerSpec(
        backend="tiptop", options={"robot": {"host": "10.1.2.3"}, "tamp": {"num_particles": 999}}
    )
    profiles.save(profile)


def test_switching_away_and_back_restores_the_rigs_settings(active):
    registry.register_backend("toy", FakeFactory("toy"))
    _customised(active)

    away = planners_cli.use_planner("toy")
    assert set(away["dropped_options"]) >= {"robot", "tamp"}
    stash = Path(away["saved_to"])
    assert stash.name == "planner-options.tiptop.yml" and stash.parent == active.dir()
    assert profiles.load(active.name).planner.options == {}

    back = planners_cli.use_planner("tiptop")
    restored = profiles.load(active.name).planner.options
    assert restored["robot"]["host"] == "10.1.2.3" and restored["tamp"]["num_particles"] == 999
    assert set(back["restored_options"]) >= {"robot", "tamp"}
    assert not stash.exists(), "restored, so no longer set aside"


def test_the_paper_presets_settings_survive_a_round_trip_through_the_web(active):
    registry.register_backend("pure", FakeFactory("pure"))
    profiles.save(presets.apply(active, "paper"))
    client = TestClient(create_app())
    assert client.post("/api/planners/pure/use", json={"profile": active.name}).status_code == 200
    assert client.post("/api/planners/tiptop/use", json={"profile": active.name}).status_code == 200
    assert profiles.load(active.name).planner.options["tamp"]["vae_manifold_weight"] == 25000


def test_a_stash_the_planner_no_longer_accepts_is_kept_and_said_not_raised(active):
    registry.register_backend("toy", FakeFactory("toy"))
    _customised(active)
    planners_cli.use_planner("toy")
    stash = profiles.stash_file(active.dir(), "tiptop")
    stash.write_text("tamp:\n  not_a_tamp_key: 1\n")

    back = planners_cli.use_planner("tiptop")
    assert back["changed"] and back["restore_problem"] and "not restored" in back["restore_problem"]
    assert stash.exists(), "kept for a person to recover by hand"
    assert profiles.load(active.name).planner.backend == "tiptop"


def test_the_cli_says_where_the_options_went_and_how_they_come_back(active):
    registry.register_backend("toy", FakeFactory("toy"))
    _customised(active)
    shown = _run("planners", "use", "toy")
    assert shown.exit_code == 0, shown.output
    assert "Removed planner.options" in shown.output and "planner-options.tiptop.yml" in shown.output
    assert "tandem planners use tiptop" in shown.output


def test_the_settings_page_asks_first_and_says_what_was_set_aside(active):
    page = SETTINGS_PAGE.read_text()
    for needle in ("confirm(", "dropped_options", "restored_options", "profile_options"):
        assert needle in page, needle
    registry.register_backend("pure", FakeFactory("pure"))
    payload = TestClient(create_app()).get("/api/planners").json()
    assert payload["profile_options"] == ["perception", "robot", "tamp"]
    result = TestClient(create_app()).post("/api/planners/pure/use", json={}).json()
    assert set(result["dropped_options"]) == {"perception", "robot", "tamp"}


# --- the default, on its own ----------------------------------------------------------------------------


def test_planners_default_changes_only_the_default(active):
    registry.register_backend("ready", FakeFactory("ready"))
    _customised(active)
    before = active.profile_file().read_text()
    result = _run("planners", "default", "ready")
    assert result.exit_code == 0, result.output
    assert settings_mod.load(force=True).default_planner == "ready"
    assert active.profile_file().read_text() == before, "no profile was touched"


def test_planners_default_works_while_the_active_profile_does_not_load(active):
    registry.register_backend("pure", FakeFactory("pure"))
    _rewrite(active, lambda d: d["planner"].update(backend="nosuchplanner", options={}))
    assert _run("planners", "default", "pure").exit_code == 0
    assert settings_mod.load(force=True).default_planner == "pure"


# --- --from with flags it cannot honour -------------------------------------------------------------------


@pytest.mark.parametrize("flag", [["--planner", "no_such_planner"]])
def test_create_from_refuses_the_flags_a_clone_would_ignore(active, flag):
    result = _run("profile", "create", "eps", "--from", active.name, *flag)
    assert result.exit_code != 0
    assert isinstance(result.exception, ProfileError) and "would be ignored" in result.exception.message
    assert not profiles.exists("eps")
    assert _run("profile", "create", "eps2", "--from", active.name).exit_code == 0
    assert profiles.load("eps2").planner == profiles.load(active.name).planner


# --- settings are saved from the file, not from a stale copy ---------------------------------------------


def test_a_change_made_by_another_process_survives_the_servers_next_save(active):
    registry.register_backend("toy", FakeFactory("toy"))
    settings_mod.load()  # the server's cached copy
    path = paths.config_file()
    doc = tomlkit.parse(path.read_text())
    doc["hf_org"] = "my-lab"
    path.write_text(tomlkit.dumps(doc))  # a `tandem config set` in a terminal

    client = TestClient(create_app())
    assert client.post("/api/planners/toy/default").status_code == 200
    written = tomlkit.parse(path.read_text())
    assert written["hf_org"] == "my-lab" and written["default_planner"] == "toy"


def test_a_setting_nobody_set_is_not_written_so_an_older_tandem_still_reads_the_file(isolated_env):
    from pydantic import BaseModel

    assert _run("config", "set", "ui.port", "9000").exit_code == 0
    written = tomlkit.parse(paths.config_file().read_text())
    assert "default_planner" not in written

    class Older(BaseModel):  # ef1411f's Settings: no default_planner, and extra keys refused
        model_config = {"extra": "forbid"}
        active_profile: str = "default"
        data_root: str = ""
        runtime_dir: str = ""
        hf_org: str = ""
        ui: dict = {}
        teleop: dict = {}

    Older.model_validate(settings_mod._plain(written))

    registry.register_backend("pure", FakeFactory("pure"))
    assert _run("planners", "default", "pure").exit_code == 0
    assert tomlkit.parse(paths.config_file().read_text())["default_planner"] == "pure"


# --- the web editor and delete work on a profile collected with a plugin this machine lacks ------------------


def test_the_web_editor_saves_a_corrected_profile_and_edits_one_with_an_absent_executor(active):
    _rewrite(active, lambda d: d["hitl"].update(human_executor="droid_policy", enabled=False))
    client = TestClient(create_app())

    shown = client.get(f"/api/profiles/{active.name}")
    assert shown.status_code == 200, shown.text
    body = shown.json()["profile"]
    assert body["hitl"]["human_executor"] == "droid_policy"

    # An edit of something else keeps the executor it already named, absent here or not.
    body["task"]["prompt"] = "a new prompt"
    edited = client.put(f"/api/profiles/{active.name}", json=body)
    assert edited.status_code == 200, edited.text
    assert profiles.load(active.name, require_installed=False).task.prompt == "a new prompt"

    # A NEW name the edit introduces must be installed.
    body["hitl"]["human_executor"] = "another_absent"
    assert client.put(f"/api/profiles/{active.name}", json=body).status_code == 400

    # And the corrected body repairs it.
    body["hitl"]["human_executor"] = "teleop"
    assert client.put(f"/api/profiles/{active.name}", json=body).status_code == 200
    assert profiles.load(active.name).hitl.human_executor == "teleop"


def test_profile_delete_does_not_need_the_plugin_the_profile_was_collected_with(active):
    _rewrite(active, lambda d: d["hitl"].update(human_executor="droid_policy"))
    result = _run("profile", "delete", active.name, "--yes")
    assert result.exit_code == 0, result.output
    assert not profiles.exists(active.name) and active.trajectories_dir().is_dir()
