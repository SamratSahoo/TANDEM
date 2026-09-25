"""The migration, in the situations a review of it found it losing or misreporting something.

- A rig.yml there before the migration (from `tandem rig set`, the web's rig card, a no-op `rig edit`)
  kept the old profiles' extrinsics out of the rig: they reached only the archive.
- A rig that could not be written from the active profile (a bad port) still archived every profile, so
  their cameras and extrinsics were in no rig, and a second run found nothing to migrate.
- A new profile could take the name of one still waiting to move, and was then given its trajectories.
- A profile moved part way was reported as "left exactly as it was".
- "Merge them by hand" could never be followed: the emptied old status directories refused it forever.
- A relative symlink (trajectories on a bigger disk) was renamed as it was, and pointed nowhere after.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from helpers import FakeFactory, isolate_registry
from ruamel.yaml import YAML
from typer.testing import CliRunner

from tandem.cli.app import app
from tandem.core import layout, paths, profiles, trajectories
from tandem.core import rig as rig_mod
from tandem.core import settings as settings_mod
from tandem.core.errors import ProfileError
from tandem.planners import registry

FIXTURES = Path(__file__).parent / "fixtures" / "profiles"
V2 = (FIXTURES / "v2_2797491.yml").read_text()
EXTRINSICS = {"pose": [0.4, 0.0, 0.6, 0.0, 0.0, 0.0], "timestamp": 0}
OTHER = {"pose": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0], "timestamp": 1}
SERIALS = ("14846828", "32439448", "31425515")
_yaml = YAML(typ="safe")


@pytest.fixture(autouse=True)
def _registry(monkeypatch):
    isolate_registry(monkeypatch)


def _old(name: str, text: str | None = V2, *, calibration: dict | None = None) -> Path:
    directory = profiles.profiles_root() / name
    directory.mkdir(parents=True)
    if text is not None:
        (directory / "profile.yml").write_text(text)
    if calibration is not None:
        (directory / "calibration.json").write_text(json.dumps(calibration))
    return directory


def _collected(directory: Path, make_trajectory, *stamps: str, status: str = "success") -> None:
    view = SimpleNamespace(status_dir=lambda s: directory / "trajectories" / s)
    for stamp in stamps:
        make_trajectory(view, stamp, status=status)


def _cli(*args: str):
    return CliRunner().invoke(app, list(args), env={"COLUMNS": "1000"})


def _calibration() -> dict:
    path = rig_mod.load(force=True).calibration_file()
    return json.loads(path.read_text()) if path.is_file() else {}


# --- the extrinsics reach the rig, whether or not rig.yml was there ----------------------------------------


def test_the_extrinsics_reach_the_rig_when_rig_yml_was_there_first(isolated_env):
    """sc/n: the machine's rig set up by hand (the same cameras) before the old profiles were moved."""
    rig_mod.update({"robot.host": "10.0.0.5", "cameras.hand.serial": "14846828", "cameras.external.serial": "32439448"})
    _old("bread-box", calibration={serial: EXTRINSICS for serial in SERIALS})

    report = layout.migrate_all(active="bread-box")
    assert not report.failed and report.rig is None
    assert _calibration() == {serial: EXTRINSICS for serial in SERIALS}
    assert rig_mod.load(force=True).missing_calibration() == [], "calibrated 2 of 2, not 0 of 2"
    assert report.calibration and "3 camera(s)' extrinsics kept" in report.calibration


def test_an_entry_the_rig_has_is_never_overwritten_and_the_difference_is_said(isolated_env):
    rig_mod.update({"cameras.hand.serial": "14846828", "cameras.external.serial": "32439448"})
    rig_mod.paths.rig_file().parent.joinpath("calibration.json").write_text(json.dumps({"14846828": OTHER}))
    _old("cloth", calibration={"14846828": EXTRINSICS, "32439448": EXTRINSICS})

    (moved,) = layout.migrate_all(active="cloth").profiles
    assert _calibration() == {"14846828": OTHER, "32439448": EXTRINSICS}
    (note,) = [n for n in moved.notes if "differ from this machine's rig" in n]
    assert "extrinsics of camera 14846828" in note and "32439448" not in note
    assert moved.archive in note, "says where its own are kept"


def test_a_profile_that_cannot_move_still_gives_the_rig_its_extrinsics(isolated_env):
    """Its profile.yml does not validate, so it stays: its calibration.json is this machine's all the same,
    and a later run (once it is fixed) would otherwise find the rig already has cameras and look no further."""
    broken = _old("broken", "version: 2\ntask:\n  bogus_key: 1\n", calibration={"555": EXTRINSICS})
    _old("fine", calibration={"14846828": EXTRINSICS})

    report = layout.migrate_all(active="broken")
    assert [m.name for m in report.failed] == ["broken"]
    assert _calibration() == {"14846828": EXTRINSICS, "555": EXTRINSICS}
    assert (broken / "calibration.json").is_file(), "left where it was"


# --- a rig that cannot be written moves nothing ------------------------------------------------------------


def _bad_port(text: str = V2) -> str:
    return text.replace("port: 5555", "port: notaport")


def test_a_rig_that_cannot_be_written_moves_nothing_and_names_the_setting_and_the_file(isolated_env, make_trajectory):
    """sc/m: the active profile's robot port is not a number."""
    bread_box = _old("bread-box", _bad_port(), calibration={serial: EXTRINSICS for serial in SERIALS})
    _collected(bread_box, make_trajectory, "2026-01-01_00-00-00")
    cloth = _old("cloth", calibration={"14846828": OTHER})

    report = layout.migrate_all(active="bread-box")
    assert isinstance(report.aborted, ProfileError)
    assert "planner.options.robot.port" in report.aborted.message, "as the old profile.yml spells it"
    assert str(bread_box / "profile.yml") in report.aborted.message
    assert report.profiles == [] and not report.notes, "nothing about a rig that does not exist"
    # Nothing moved, nothing written.
    assert (bread_box / "trajectories").is_dir() and (bread_box / "profile.yml").is_file() and cloth.is_dir()
    assert not rig_mod.exists() and not (paths.config_dir() / "calibration.json").exists()
    assert layout.pending() == ["bread-box", "cloth"], "a second run still has them to move"

    # Fixed, and run again: everything moves, and the extrinsics come along.
    (bread_box / "profile.yml").write_text(V2)
    report = layout.migrate_all(active="bread-box")
    assert report.aborted is None and not report.failed and layout.pending() == []
    assert rig_mod.load(force=True).missing_calibration() == []


def test_profile_migrate_and_init_stop_with_what_to_fix(isolated_env):
    _old("bread-box", _bad_port())
    result = _cli("profile", "migrate")
    assert result.exit_code == 1
    assert "planner.options.robot.port" in result.exception.message
    assert "Nothing was moved" in result.exception.hint
    assert layout.pending() == ["bread-box"]

    result = _cli("init", "--viz-only", "--yes")
    assert result.exit_code == 1 and "planner.options.robot.port" in result.exception.message
    assert not profiles.exists("cover-bread-rolls"), "init stops there: nothing is set up over the old profiles"


def test_a_version_1_setting_is_named_where_version_1_had_it():
    plan = layout._Plan("x", Path("p"), archive=Path("a"), version=1, backend="tiptop")
    assert layout._as_in_profile("  planners.tiptop.robot.port: bad", plan) == "  robot.port: bad"
    plan.version = 2
    assert layout._as_in_profile("  robot.host: bad", plan) == "  planner.options.robot.host: bad"
    assert layout._as_in_profile("  cameras.hand.serial: bad", plan) == "  cameras.hand.serial: bad"


# --- the rig and a new profile wait for the old ones ---------------------------------------------------------


def test_the_rig_is_not_changed_by_hand_while_old_profiles_hold_it(isolated_env, monkeypatch):
    _old("bread-box")
    for args in (("rig", "set", "robot.host", "172.16.0.9"), ("rig", "edit")):
        monkeypatch.setenv("EDITOR", "true")
        result = _cli(*args)
        assert result.exit_code == 1, result.output
        assert "`tandem profile migrate` first" in result.exception.message
        assert not rig_mod.exists()

    response = TestClient(_web()).patch("/api/rig", json={"robot.host": "172.16.0.9"})
    assert response.status_code == 400 and "`tandem profile migrate` first" in response.json()["error"]
    assert not rig_mod.exists()

    layout.migrate_all(active="bread-box")
    assert _cli("rig", "set", "robot.host", "172.16.0.9").exit_code == 0, "once moved, the rig is the person's"


def _web():
    from tandem.server.app import create_app

    return create_app()


def test_a_new_profile_cannot_take_the_name_of_one_waiting_to_move(isolated_env):
    """sc/c: the migration would keep the new file, and give it the old task's trajectories."""
    _old("bread-box")
    result = _cli("profile", "create", "bread-box", "--prompt", "something new")
    assert result.exit_code == 1 and "in the old layout" in result.exception.message
    assert not profiles.path_of("bread-box").exists()

    response = TestClient(_web()).post("/api/profiles", json={"name": "bread-box", "prompt": "something new"})
    assert response.status_code == 400 and "`tandem profile migrate` first" in response.json()["error"]


def test_a_paper_task_is_not_added_under_the_name_of_one_waiting_to_move(isolated_env):
    _old(profiles.BUILTIN[0], "version: 2\ntask:\n  bogus_key: 1\n")  # cannot move, so stays pending
    written = profiles.seed_builtins()
    assert profiles.BUILTIN[0] not in written and profiles.held_back() == [profiles.BUILTIN[0]]
    assert not profiles.path_of(profiles.BUILTIN[0]).exists()


# --- part way, and by hand ------------------------------------------------------------------------------------


def test_a_profile_moved_part_way_says_so_and_a_second_run_finishes_it(isolated_env, make_trajectory):
    """sc/f: profiles/.migrated cannot be made, so the archive step fails after the trajectories and the file."""
    cloth = _old("cloth")
    _collected(cloth, make_trajectory, "2026-01-01_00-00-00")
    blocker = profiles.profiles_root() / layout.ARCHIVE_DIR
    blocker.write_text("in the way")

    result = _cli("profile", "migrate")
    output = " ".join(result.output.split())
    assert result.exit_code == 1
    assert "cloth: partly moved" in output and "its trajectories are now in" in output
    assert f"its old directory is still at {cloth}" in output
    assert "left exactly as it was" not in result.exception.hint and "partly moved" in result.exception.hint
    assert (profiles.trajectories_root() / "cloth" / "success" / "2026-01-01_00-00-00").is_dir()

    blocker.unlink()
    report = layout.migrate_all()
    (moved,) = report.profiles
    assert moved.ok and not cloth.exists()
    assert any("was written by an earlier run" in note for note in moved.notes)
    assert not any("kept the existing" in note for note in moved.notes)
    assert trajectories.counts(profiles.load("cloth"))["success"] == 1


def test_merging_by_hand_as_the_error_says_is_then_accepted(isolated_env, make_trajectory):
    """sc/g: every run moved over by hand leaves the old, empty status directories behind."""
    cloth = _old("cloth")
    _collected(cloth, make_trajectory, "2026-01-01_00-00-00")
    theirs = profiles.trajectories_root() / "cloth" / "success" / "2026-02-02_00-00-00"
    theirs.mkdir(parents=True)
    (theirs / "meta.json").write_text("{}")

    (moved,) = layout.migrate_all().profiles
    assert not moved.ok and "`mv " in moved.error and "/success/*" in moved.error
    for run in (cloth / "trajectories" / "success").iterdir():
        os.rename(run, profiles.trajectories_root() / "cloth" / "success" / run.name)

    (moved,) = layout.migrate_all().profiles
    assert moved.ok, moved.error
    assert len(list((profiles.trajectories_root() / "cloth" / "success").iterdir())) == 2


# --- symlinks ---------------------------------------------------------------------------------------------------


def test_a_relative_symlink_to_the_trajectories_still_points_at_them_after(isolated_env, make_trajectory):
    """sc/j: trajectories on a bigger disk, through a relative link from inside the old profile directory."""
    cloth = _old("cloth")
    bigdisk = isolated_env / "bigdisk" / "cloth-traj"
    view = SimpleNamespace(status_dir=lambda s: bigdisk / s)
    make_trajectory(view, "2026-01-01_00-00-00", status="success")
    (cloth / "trajectories").symlink_to(os.path.relpath(bigdisk, cloth))

    (moved,) = layout.migrate_all().profiles
    assert moved.ok, moved.error
    new = profiles.trajectories_root() / "cloth"
    assert new.is_symlink() and new.resolve() == bigdisk.resolve()
    assert trajectories.counts(profiles.load("cloth"))["success"] == 1


def test_a_symlinked_profile_directory_is_archived_as_a_link_that_still_resolves(isolated_env):
    elsewhere = isolated_env / "elsewhere" / "cloth"
    elsewhere.mkdir(parents=True)
    (elsewhere / "profile.yml").write_text(V2)
    profiles.profiles_root().mkdir(parents=True, exist_ok=True)
    (profiles.profiles_root() / "cloth").symlink_to(os.path.relpath(elsewhere, profiles.profiles_root()))

    (moved,) = layout.migrate_all().profiles
    assert moved.ok, moved.error
    archive = Path(moved.archive)
    assert archive.is_symlink() and (archive / "profile.yml").read_text() == V2
    assert (archive / layout.RECORD).is_file()


# --- what else it said, or did not ------------------------------------------------------------------------------


def test_a_deleted_profiles_trajectories_say_where_they_went_and_how_to_have_them_back(isolated_env, make_trajectory):
    kept = _old("kept", None)
    _collected(kept, make_trajectory, "2026-01-01_00-00-00")
    result = _cli("profile", "migrate")
    output = " ".join(result.output.split())
    assert result.exit_code == 0, output
    assert str(profiles.trajectories_root() / "kept") in output
    assert '`tandem profile create kept --prompt "..."`' in output


def test_other_files_are_said_to_go_to_the_archive(isolated_env):
    directory = _old("cloth")
    (directory / "grasp_prior.pt").write_text("x")
    (moved,) = layout.migrate_all().profiles
    (note,) = [n for n in moved.notes if "grasp_prior.pt" in n]
    assert "resolves beside profiles/" in note


def test_a_stash_loses_the_machine_settings_the_rig_holds_now(isolated_env):
    directory = _old("cloth", V2.replace("backend: tiptop", "backend: toy"))
    (directory / "planner-options.tiptop.yml").write_text(
        "robot:\n  host: 10.0.0.5\ntamp:\n  num_particles: 64\n"
    )
    registry.register_backend("toy", FakeFactory("toy"))
    (moved,) = layout.migrate_all().profiles
    assert moved.ok, moved.error
    assert _yaml.load(profiles.stash_file("cloth", "tiptop").read_text()) == {"tamp": {"num_particles": 64}}
    assert any("without robot" in note and "`tandem planners use tiptop` restores it" in note for note in moved.notes)


def test_another_planners_robot_block_keeps_its_own_host(monkeypatch):
    """Only TiPToP's robot block held the arm's address: a plugin's may have a host of its own."""

    class Shelf(FakeFactory):
        OPTIONS = {"bins": "the bins"}
        RIG_OPTIONS = {"robot": "the shelf robot's own server"}

    registry.register_backend("shelf", Shelf("shelf"))
    data = {"version": 2, "planner": {"backend": "shelf", "options": {"robot": {"host": "10.9.9.9", "port": 7}}}}
    _profile, rig, _notes = layout.split_legacy(data)
    assert "robot" not in rig
    assert rig["planners"]["shelf"] == {"robot": {"host": "10.9.9.9", "port": 7}}


def test_the_active_profile_was_default_when_the_old_settings_left_it_unset(isolated_env):
    """Before version 3 the active profile defaulted to `default`, so config.toml did not say it."""
    _old("default")
    _old("other")
    settings = settings_mod.load()
    settings.active_profile = ""
    settings_mod.save(settings)
    assert _cli("profile", "migrate").exit_code == 0
    assert settings_mod.load(force=True).active_profile == "default"
