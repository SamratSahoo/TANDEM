"""Moving profiles written before version 3 into the current layout, and their machine settings into the rig.

The inputs are real old profiles: the template as version 1 shipped it (ef1411f and the first commit,
tests/fixtures/profiles/v1_*.yml) and a version-2 profile written by the code of 2797491, the last commit
before version 3 (v2_2797491.yml: `profiles.save` of the template with the robot's address, speed and
grasp server changed, three cameras, and phase planning on). What that same code rendered for it --
tiptop.yml, the cuRobo overrides, the TIPTOP_* environment -- is kept beside it
(tests/golden/v2_2797491_render.json), and a migrated profile on its migrated rig must render exactly that:
a session after the move plans with what it planned with before.

Nothing is deleted: every old directory ends up whole in profiles/.migrated/, with a record of what was done.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from helpers import FakeFactory, isolate_registry
from ruamel.yaml import YAML
from typer.testing import CliRunner

from tandem.cli.app import app
from tandem.core import layout, profiles, trajectories
from tandem.core import rig as rig_mod
from tandem.core import settings as settings_mod
from tandem.planners import registry
from tandem.planners.tiptop import render
from tandem.planners.tiptop.options import resolve_profile

FIXTURES = Path(__file__).parent / "fixtures" / "profiles"
GOLDEN = Path(__file__).parent / "golden" / "v2_2797491_render.json"
V1 = (FIXTURES / "v1_ef1411f.yml").read_text()
V1_FIRST = (FIXTURES / "v1_68076df.yml").read_text()
V2 = (FIXTURES / "v2_2797491.yml").read_text()
EXTRINSICS = {"pose": [0.4, 0.0, 0.6, 0.0, 0.0, 0.0], "timestamp": 0}
_yaml = YAML(typ="safe")


@pytest.fixture(autouse=True)
def _registry(monkeypatch):
    isolate_registry(monkeypatch)


def _old(name: str, text: str | None, *, calibration: dict | None = None, extra: dict | None = None) -> Path:
    """A profile in the layout before version 3: ``profiles/<name>/`` with its profile.yml and calibration."""
    directory = profiles.profiles_root() / name
    directory.mkdir(parents=True)
    if text is not None:
        (directory / "profile.yml").write_text(text)
    if calibration is not None:
        (directory / "calibration.json").write_text(json.dumps(calibration))
    for filename, content in (extra or {}).items():
        (directory / filename).write_text(content)
    return directory


def _collected(directory: Path, make_trajectory, *stamps: str, status: str = "success") -> None:
    view = SimpleNamespace(status_dir=lambda s: directory / "trajectories" / s)
    for stamp in stamps:
        make_trajectory(view, stamp, status=status)


def _files(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


# --- one profile, split -----------------------------------------------------------------------------------


@pytest.mark.parametrize("text", [V1, V1_FIRST], ids=["v1_ef1411f", "v1_68076df"])
def test_a_version_1_profile_splits_into_a_task_and_the_rig(text):
    old = _yaml.load(text)
    profile, rig, notes = layout.split_legacy(_yaml.load(text), name="default")
    assert profile["version"] == 3
    assert not {"robot", "perception", "tamp", "cameras", "name"} & set(profile)
    assert profile["planner"] == {"backend": "tiptop", "options": {"tamp": old["tamp"]}}
    assert profile["task"] == old["task"]
    assert "fps" not in profile["recording"]

    assert rig["robot"] == {"type": old["robot"]["type"], "host": old["robot"]["host"]}
    assert rig["cameras"] == old["cameras"]
    machine = rig["planners"]["tiptop"]
    assert set(machine) == {"robot", "perception"}
    assert machine["perception"] == old["perception"]
    assert machine["robot"] == {k: v for k, v in old["robot"].items() if k not in ("type", "host")}
    assert any("recording.fps dropped" in note for note in notes)
    # The migrated profile validates as it is.
    profiles.Profile.model_validate({**profile, "name": "default"}, context={profiles.ABSENT_OK: True})


def test_version_1s_teleop_default_is_said_and_kept():
    profile, _, notes = layout.split_legacy(_yaml.load(V1), name="default")
    assert profile["hitl"]["on_robot_phase_failure"] == "teleop"
    assert any("on_robot_phase_failure kept at teleop" in note for note in notes)
    chose = _yaml.load(V1.replace("on_robot_phase_failure: teleop", "on_robot_phase_failure: abort"))
    assert not any("on_robot_phase_failure" in note for note in layout.split_legacy(chose)[2])


def test_a_version_2_profile_splits_the_same_way():
    old = _yaml.load(V2)
    profile, rig, notes = layout.split_legacy(_yaml.load(V2), name="bread-box")
    assert profile["planner"]["options"] == {"tamp": old["planner"]["options"]["tamp"]}
    assert profile["hitl"] == old["hitl"] and profile["description"] == old["description"]
    assert rig["robot"] == {"type": "fr3_robotiq", "host": "10.0.0.5"}
    assert rig["cameras"]["external_2"]["serial"] == "31425515"
    assert rig["planners"]["tiptop"]["perception"]["m2t2"]["url"] == "http://gpu:8123"
    assert rig["planners"]["tiptop"]["robot"]["time_dilation_factor"] == 0.3
    assert not any("name" in note for note in notes), "the name agreed with its directory"


def test_version_1_sections_of_another_planner_are_dropped_and_said_to_be():
    registry.register_backend("toy", FakeFactory("toy"))
    data = {"name": "x", "version": 1, "robot": {"host": "10.0.0.9"}, "tamp": {"num_particles": 8},
            "planner": {"backend": "toy"}}
    profile, rig, notes = layout.split_legacy(data)
    assert profile["planner"]["options"] == {}
    assert any("robot, tamp dropped" in note and "plans with toy" in note for note in notes)
    assert "planners" not in rig


def test_a_setting_in_both_places_is_refused_rather_than_one_silently_winning():
    with pytest.raises(ValueError, match="tamp is set both at the top level"):
        layout.split_legacy({"name": "x", "tamp": {}, "planner": {"backend": "tiptop", "options": {"tamp": {}}}})


def test_a_planner_that_is_not_installed_keeps_all_its_options_in_the_profile():
    data = {"version": 2, "planner": {"backend": "shelfbot", "options": {"bins": ["a"], "host": "10.0.0.2"}}}
    profile, rig, notes = layout.split_legacy(data)
    assert profile["planner"]["options"] == {"bins": ["a"], "host": "10.0.0.2"}
    assert "planners" not in rig
    assert any("shelfbot planner is not installed here" in note for note in notes)


def test_a_relative_cache_path_is_said_to_mean_something_else_now():
    notes = layout.split_legacy({"version": 2, "hitl": {"cache_path": "proposals.sqlite"}})[2]
    assert any("now relative to profiles/" in note for note in notes)
    assert not layout.split_legacy({"version": 2, "hitl": {"cache_path": "/abs/p.sqlite"}})[2]


# --- every profile, moved ------------------------------------------------------------------------------------


@pytest.fixture
def bread_box(isolated_env, make_trajectory):
    """The version-2 bread-box profile as HEAD's code left it: trajectories, extrinsics, a stash, a backup."""
    directory = _old(
        "bread-box",
        V2,
        calibration={serial: EXTRINSICS for serial in ("14846828", "32439448", "31425515")},
        extra={"planner-options.toy.yml": "items: [duck]\n", "profile.yml.v1.bak": V1},
    )
    _collected(directory, make_trajectory, "2026-01-01_00-00-00", "2026-01-02_00-00-00")
    _collected(directory, make_trajectory, "2026-01-03_00-00-00", status="failure")
    cfg = settings_mod.load()
    cfg.active_profile = "bread-box"
    settings_mod.save(cfg)
    return directory


def test_a_version_2_profile_is_moved_and_renders_what_it_rendered_before(bread_box, tmp_path):
    before = _files(bread_box)
    report = layout.migrate_all(active="bread-box")
    assert not report.failed
    assert report.rig and "rig.yml written from bread-box" in report.rig and "fr3_robotiq at 10.0.0.5" in report.rig

    # The rig is the old profile's machine.
    rig = rig_mod.load()
    assert rig.robot.host == "10.0.0.5"
    assert {role: cam.serial for role, cam in rig.cameras.configured().items()} == {
        "hand": "14846828",
        "external": "32439448",
        "external_2": "31425515",
    }
    assert rig.missing_calibration() == []
    assert "# the robot computer: the NUC" in rig_mod.paths.rig_file().read_text(), "written from the template"

    # The profile is the task, in one file.
    profile = profiles.load("bread-box")
    assert profile.file() == profiles.profiles_root() / "bread-box.yml"
    assert profile.planner.options == {"tamp": _yaml.load(V2)["planner"]["options"]["tamp"]}
    assert profile.hitl.enabled and profile.description == "the bread box, collected on the FR3"
    assert profile.file().read_text().startswith("# bread-box: migrated from profiles/bread-box/profile.yml (version 2)")

    # What a session renders from them is what the old profile rendered.
    golden = json.loads(GOLDEN.read_text())
    options = resolve_profile(profile, rig)
    assert render.render_tiptop_config(rig, options) == golden["tiptop_yml"]
    assert render.render_tamp_overrides(profile, options) == golden["overrides"]
    env = render.render_env(profile, rig, options, events_file=tmp_path / "e.jsonl", base={})
    assert {key: env[key] for key in golden["env"]} == golden["env"]
    assert env["TIPTOP_CALIBRATION"] == str(rig.calibration_file())

    # The trajectories, moved and still found.
    assert trajectories.counts(profile) == {"eval": 0, "success": 2, "failure": 1}
    listed = CliRunner().invoke(app, ["traj", "list", "bread-box"])
    assert listed.exit_code == 0 and "2026-01-02_00-00-00" in listed.output

    # Nothing lost: the old directory, whole but for the trajectories, and a record.
    archive = profiles.profiles_root() / ".migrated" / "bread-box"
    assert not bread_box.exists()
    moved = {f for f in before if not f.startswith("trajectories/")}
    assert moved <= _files(archive)
    assert (archive / "profile.yml").read_text() == V2
    record = json.loads((archive / "migration.json").read_text())
    assert record["from_version"] == 2 and record["rig"] == "seeded rig.yml"
    assert record["profile"] == str(profile.file()) and record["trajectories"] == str(profile.trajectories_dir())
    assert any("planner-options.toy.yml stays in the archive" in note for note in record["notes"])


def test_the_session_tiptop_builds_after_the_move_is_the_one_it_built_before(bread_box, tmp_path, monkeypatch):
    """Through TiPToP's factory, as a session builds it: the tiptop.yml $TIPTOP_CONFIG points at, and the
    overrides file, are what 2797491 rendered for the version-2 profile."""
    from tandem.core import paths, secrets
    from tandem.planners.base import BackendContext
    from tandem.planners.tiptop import FACTORY

    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
    layout.migrate_all(active="bread-box")
    profile, rig = profiles.load("bread-box"), rig_mod.load()
    session_dir = paths.session_scratch_dir() / "bread-box" / "s1"
    backend = FACTORY.create(
        BackendContext(
            profile=profile,
            session_dir=session_dir,
            output_dir=profile.trajectories_dir(),
            execute=False,
            options=dict(profile.planner.options),
            task=profile.task.prompt,
            runtime_dir=tmp_path / "runtime",
            rig=rig,
            rig_options=rig_mod.planner_options(rig, "tiptop"),
        )
    )
    golden = json.loads(GOLDEN.read_text())
    assert _yaml.load(Path(backend._env["TIPTOP_CONFIG"]).read_text()) == golden["tiptop_yml"]
    assert json.loads(backend._cost_overrides_file.read_text()) == golden["overrides"]
    assert {key: backend._env[key] for key in golden["env"]} == golden["env"]


def test_a_second_run_does_nothing(bread_box):
    layout.migrate_all(active="bread-box")
    rig_text = rig_mod.paths.rig_file().read_text()
    profile_text = profiles.path_of("bread-box").read_text()
    assert layout.pending() == []
    report = layout.migrate_all(active="bread-box")
    assert report.profiles == [] and report.rig is None
    assert rig_mod.paths.rig_file().read_text() == rig_text
    assert profiles.path_of("bread-box").read_text() == profile_text


def test_the_rig_comes_from_the_active_profile_and_every_profiles_extrinsics_are_merged(isolated_env):
    _old("aaa", V2.replace("10.0.0.5", "10.0.0.1"), calibration={"14846828": EXTRINSICS, "999": EXTRINSICS})
    _old("zzz", V2, calibration={"14846828": {"pose": [1, 1, 1, 0, 0, 0]}, "32439448": EXTRINSICS})
    report = layout.migrate_all(active="zzz")
    assert "written from zzz" in report.rig
    rig = rig_mod.load()
    assert rig.robot.host == "10.0.0.5"
    calibration = rig.extrinsics()
    assert set(calibration) == {"14846828", "32439448", "999"}
    assert calibration["14846828"] == {"pose": [1, 1, 1, 0, 0, 0]}, "the active profile's own wins"
    assert any("camera 14846828: aaa's extrinsics differ from zzz's" in note for note in report.notes)
    aaa = json.loads((profiles.profiles_root() / ".migrated" / "aaa" / "migration.json").read_text())
    assert aaa["rig"].startswith("differs: ") and "robot.host" in aaa["rig"]


def test_an_existing_rig_is_never_touched_and_differences_are_said(isolated_env):
    rig_mod.update({"robot.host": "172.16.0.9", "cameras.external.serial": "32439448"})
    before = rig_mod.paths.rig_file().read_text()
    _old("bread-box", V2)
    report = layout.migrate_all(active="bread-box")
    assert report.rig is None and not report.failed
    assert rig_mod.paths.rig_file().read_text() == before
    (moved,) = report.profiles
    (note,) = [n for n in moved.notes if "differ from this machine's rig.yml" in n]
    assert "robot.host" in note and "cameras.hand" in note
    record = json.loads((profiles.profiles_root() / ".migrated" / "bread-box" / "migration.json").read_text())
    assert record["rig"].startswith("differs:")


def test_a_profile_whose_trajectories_would_land_on_others_is_refused_alone(isolated_env, make_trajectory):
    clash = _old("clash", V2)
    _collected(clash, make_trajectory, "2026-01-01_00-00-00")
    # A file, however small, is data: never merged into, never removed.
    stray = profiles.trajectories_root() / "clash" / "eval" / "notes.txt"
    stray.parent.mkdir(parents=True)
    stray.write_text("kept")
    _old("fine", V2)
    report = layout.migrate_all()
    by_name = {moved.name: moved for moved in report.profiles}
    assert not by_name["clash"].ok and "merge them by hand" in by_name["clash"].error
    assert (clash / "profile.yml").read_text() == V2 and (clash / "trajectories").is_dir(), "left as it was"
    assert stray.read_text() == "kept"
    assert not profiles.exists("clash")
    assert by_name["fine"].ok and profiles.exists("fine")


def test_a_profile_made_under_the_same_name_before_the_move_still_gets_its_trajectories(bread_box):
    """`tandem profile create bread-box` run before `tandem init`: its file is kept, and the empty
    trajectory directories it made are no reason to leave the collected ones behind."""
    profiles.save(profiles.load_file(FIXTURES / "test_v3.yml", name="bread-box"))
    (moved,) = layout.migrate_all(active="bread-box").profiles
    assert moved.ok, moved.error
    assert any("kept the existing bread-box.yml" in note for note in moved.notes)
    assert trajectories.counts(profiles.load("bread-box")) == {"eval": 0, "success": 2, "failure": 1}


def test_a_calibration_file_already_beside_the_rig_is_kept_and_merged_into(isolated_env):
    ours = {"pose": [9, 9, 9, 0, 0, 0]}
    rig_mod.paths.ensure_dir(rig_mod.paths.config_dir())
    calibration = rig_mod.paths.config_dir() / "calibration.json"
    calibration.write_text(json.dumps({"14846828": ours, "555": ours}))
    _old("bread-box", V2, calibration={"14846828": EXTRINSICS, "32439448": EXTRINSICS})
    report = layout.migrate_all(active="bread-box")
    assert json.loads(calibration.read_text()) == {"14846828": ours, "555": ours, "32439448": EXTRINSICS}
    assert any("camera 14846828: calibration.json already had extrinsics" in note for note in report.notes)


def test_a_deleted_profiles_trajectories_are_moved_too(isolated_env, make_trajectory):
    kept = _old("kept", None)
    _collected(kept, make_trajectory, "2026-01-01_00-00-00")
    report = layout.migrate_all()
    (moved,) = report.profiles
    assert moved.ok and moved.file is None
    assert (profiles.trajectories_root() / "kept" / "success" / "2026-01-01_00-00-00").is_dir()
    assert not profiles.exists("kept"), "deleted it was, deleted it stays: re-creating the name brings it back"
    assert (profiles.profiles_root() / ".migrated" / "kept" / "migration.json").is_file()


def test_an_invalid_profile_is_reported_and_left_where_it_is(isolated_env):
    broken = _old("broken", "version: 2\ntask:\n  bogus_key: 1\n")
    report = layout.migrate_all()
    (moved,) = report.profiles
    assert not moved.ok and "bogus_key" in moved.error
    assert (broken / "profile.yml").is_file() and layout.pending() == ["broken"]
    assert report.rig is None and not rig_mod.exists(), "no rig from a profile that does not load"


def test_an_archive_name_that_is_taken_gets_a_number(isolated_env):
    (profiles.profiles_root() / ".migrated" / "twice").mkdir(parents=True)
    _old("twice", V2)
    (moved,) = layout.migrate_all().profiles
    assert moved.archive.endswith("twice-2")


def test_an_existing_file_of_the_same_name_is_kept(isolated_env):
    profiles.profiles_root().mkdir(parents=True)
    profiles.path_of("both").write_text((FIXTURES / "test_v3.yml").read_text())
    _old("both", V2)
    (moved,) = layout.migrate_all().profiles
    assert moved.ok and any("kept the existing both.yml" in note for note in moved.notes)
    assert profiles.load("both").task.prompt == "Place the toys on the plate with no collisions"


# --- the commands that run it ---------------------------------------------------------------------------------


def test_profile_migrate_says_what_it_moved_and_fails_when_one_could_not_be(isolated_env):
    _old("legacy", V1)
    _old("broken", "version: 2\ntask:\n  bogus_key: 1\n")
    result = CliRunner().invoke(app, ["profile", "migrate"])
    assert result.exit_code == 1
    output = " ".join(result.output.split())
    assert "Rig written from the old profiles" in output and "legacy: moved" in output
    assert "broken: not moved" in output
    assert "1 profile(s) could not be moved: broken" in result.exception.message
    assert profiles.exists("legacy")


def test_init_moves_old_profiles_before_anything_reads_them(isolated_env):
    _old("default", V1, calibration={"14846828": EXTRINSICS, "32439448": EXTRINSICS})
    result = CliRunner().invoke(app, ["init", "--viz-only", "--yes"])
    assert result.exit_code == 0, result.output
    assert "profiles in the old layout" in result.output
    assert profiles.exists("default") and layout.pending() == []
    assert rig_mod.load().cameras.configured().keys() == {"hand", "external"}
    assert rig_mod.load().missing_calibration() == []
