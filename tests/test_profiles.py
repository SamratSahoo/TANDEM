"""Profile loading, validation, the one-file-per-profile layout, and the TAMP-key contract.

The validation tests are the important ones. A TAMP setting that is silently ignored is the
failure mode that looks exactly like success — a whole dataset collected with the knob you
were studying doing nothing — so every one of these guards a real way that has happened.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from tandem import resources
from tandem.core import profiles
from tandem.core.errors import ProfileError, ProfileInvalid
from tandem.core.profiles import Profile
from tandem.planners.tiptop import render
from tandem.planners.tiptop.options import resolve_profile, validate_tamp


def test_template_is_valid():
    """The shipped template must load, or `tandem init` fails on a fresh machine."""
    profile = profiles.load_file(resources.path("profile_template.yml"), name="default")
    assert profile.name == "default"
    assert profile.version == profiles.LAYOUT_VERSION == 3
    assert profile.planner.backend == "tiptop"
    # The task's settings only: the machine's are the rig's.
    assert set(profile.planner.options) == {"tamp"}
    text = resources.read("profile_template.yml")
    for section in ("\ncameras:", "\nrobot:", "\nperception:", "\n    robot:", "\n    perception:", "fps:"):
        assert section not in text


def test_save_and_load_roundtrip(profile):
    loaded = profiles.load("test")
    assert loaded.task.prompt == profile.task.prompt
    assert loaded.planner.options == profile.planner.options
    for status in profiles.STATUSES:
        assert loaded.status_dir(status).is_dir()


def test_unknown_tamp_key_is_rejected_with_a_suggestion():
    with pytest.raises(ValueError) as excinfo:
        validate_tamp({"encoder_wieght": 100})  # transposed letters
    message = str(excinfo.value)
    assert "unknown TAMP setting" in message
    assert "encoder_weight" in message  # the suggestion


def test_traj_length_norm_normalises_to_the_string_inf():
    """The overrides dict round-trips through JSON, which has no Infinity literal.

    YAML `inf` parses as a string and `.inf` as a float; both have to land on the one form
    the planner accepts.
    """
    assert validate_tamp({"traj_length_norm": "inf"})["traj_length_norm"] == "inf"
    assert validate_tamp({"traj_length_norm": "INFINITY"})["traj_length_norm"] == "inf"
    assert validate_tamp({"traj_length_norm": math.inf})["traj_length_norm"] == "inf"
    assert validate_tamp({"traj_length_norm": 2})["traj_length_norm"] == 2.0


def test_retime_ops_typo_is_caught():
    """An upstream config really did ship `MoveFree. MoveHolding` — one typo'd period that
    YAML folded into a single token and the planner ignored for months."""
    with pytest.raises(ValueError) as excinfo:
        validate_tamp({"retime_ops": ["Pick", "MoveFree. MoveHolding"]})
    assert "unknown operations" in str(excinfo.value)


def test_retime_mode_is_encoder_only():
    with pytest.raises(ValueError, match="retime_mode must be one of"):
        validate_tamp({"retime_mode": "spline"})  # a mode tiptop removed
    assert validate_tamp({"retime_mode": "encoder"})["retime_mode"] == "encoder"


def test_retiming_needs_the_encoder_checkpoint():
    with pytest.raises(ValueError, match="retime_trajectory needs encoder_path"):
        validate_tamp({"retime_trajectory": True})
    assert validate_tamp({"retime_trajectory": True, "encoder_path": "e.pt"})["retime_trajectory"] is True


def test_a_profile_written_before_tiptops_rename_loads_under_the_new_names():
    """tiptop renamed the re-timing keys (vae_path -> encoder_path, blend_* -> retime_*) and removed the
    VAE clock and the spline and flow modes. A profile from before still loads, meaning the same thing."""
    old = {
        "vae_manifold_weight": 25000.0,
        "vae_path": "vae/checkpoints/vae_full_v2.pt",
        "vae_retiming": False,  # what tiptop still does: dropped
        "blend_trajectory": True,
        "blend_mode": "vae",  # the mode tiptop kept: dropped
        "blend_ops": ["Pick", "Place"],
        "blend_stretch_to_caps": True,
    }
    assert validate_tamp(old) == {
        "encoder_weight": 25000.0,
        "encoder_path": "vae/checkpoints/vae_full_v2.pt",
        "retime_trajectory": True,
        "retime_ops": ["Pick", "Place"],
        "retime_stretch_to_caps": True,
    }


@pytest.mark.parametrize(
    "setting",
    [{"vae_retiming": True}, {"blend_mode": "flow"}, {"blend_mode": "spline"}, {"blend_pace": "droid"}, {"retime_scale": 1.0}],
)
def test_a_setting_of_a_removed_mode_is_refused_not_read_as_something_else(setting):
    with pytest.raises(ValueError, match="tiptop removed it"):
        validate_tamp(setting)


def test_a_key_set_under_both_names_differently_is_refused():
    with pytest.raises(ValueError, match="old name of 'retime_smoothing'"):
        validate_tamp({"blend_smoothing": 0.001, "retime_smoothing": 0.002})
    assert validate_tamp({"blend_smoothing": 0.001, "retime_smoothing": 0.001}) == {"retime_smoothing": 0.001}


def test_scalar_types_are_enforced():
    with pytest.raises(ValueError):
        validate_tamp({"num_particles": "many"})
    with pytest.raises(ValueError):
        validate_tamp({"retime_trajectory": "yes"})  # a string, not a bool
    assert validate_tamp({"num_particles": 128})["num_particles"] == 128


def test_positive_constraints():
    with pytest.raises(ValueError):
        validate_tamp({"num_particles": 0})
    with pytest.raises(ValueError):
        validate_tamp({"time_dilation_factor_literal": 1.5})


def test_index_map_keys():
    result = validate_tamp({"smooth_weight": {0: 1.5, "1": 2}})
    assert result["smooth_weight"] == {"0": 1.5, "1": 2.0}


def test_profile_name_must_be_a_safe_path_segment():
    with pytest.raises(ValidationError):
        Profile.model_validate({"name": "../escape"})
    with pytest.raises(ValidationError):
        Profile.model_validate({"name": "Has Spaces"})


def test_the_robot_is_the_rigs_not_a_profiles():
    """TiPToP's robot and perception settings are this machine's: a profile holding them is told where they go."""
    with pytest.raises(ValidationError) as excinfo:
        Profile.model_validate({"name": "x", "planner": {"options": {"robot": {"dof": 7, "q_home": [0.0, 0.0]}}}})
    message = str(excinfo.value)
    assert "machine setting" in message
    assert "tandem rig set planners.tiptop.robot" in message


def test_a_profile_has_no_cameras():
    """The cameras are the rig's. Not a key a profile has, at the top or anywhere."""
    assert "cameras" not in Profile.model_fields
    assert "cameras" not in Profile.model_validate({"name": "viz-only"}).model_dump()


def test_missing_profile_names_the_alternatives(profile):
    with pytest.raises(ProfileError) as excinfo:
        profiles.load("nope")
    assert "test" in (excinfo.value.hint or "")


def test_render_env_sets_the_contract_variables(profile, machine_rig, tmp_path):
    events = tmp_path / "events.jsonl"
    options = resolve_profile(profile, machine_rig)
    env = render.render_env(profile, machine_rig, options, events_file=events, task="do the thing", base={})
    assert env["TIPTOP_TASK"] == "do the thing"
    assert env["TIPTOP_EVENTS_FILE"] == str(events)
    assert env["TIPTOP_STATE_PORT"] == str(options.robot.state_port)
    # The rig's extrinsics: every profile on this machine reads the one file.
    assert env["TIPTOP_CALIBRATION"] == str(machine_rig.calibration_file())
    assert env["DC_WORKSPACE"] == profile.name
    # opencv's LAPACK and torch share one libmkl_core; the threaded path corrupts a pivot
    # array and cuRobo dies inside torch.inverse.
    assert env["MKL_NUM_THREADS"] == "1"


def test_render_env_only_splits_instruction_when_the_goal_differs(profile, machine_rig, tmp_path):
    events = tmp_path / "events.jsonl"
    options = resolve_profile(profile, machine_rig)
    env = render.render_env(profile, machine_rig, options, events_file=events, base={})
    assert "TIPTOP_INSTRUCTION" not in env

    profile.task.goal = "a reduced goal the planner can express"
    env = render.render_env(profile, machine_rig, options, events_file=events, base={})
    assert env["TIPTOP_INSTRUCTION"] == profile.task.prompt
    assert env["TIPTOP_TASK"] == profile.task.goal


def test_render_tiptop_config_shape(profile, machine_rig):
    config = render.render_tiptop_config(machine_rig, resolve_profile(profile, machine_rig))
    assert set(config) == {"robot", "cameras", "perception"}
    assert config["robot"]["type"] == machine_rig.robot.type
    assert config["robot"]["host"] == machine_rig.robot.host
    assert config["cameras"]["perception"] == machine_rig.cameras.perception
    assert "depth_smoothing" in config["perception"]


def test_no_overrides_file_when_the_profile_sets_nothing(profile, machine_rig, tmp_path):
    """Passing an empty --curobo-overrides is not the same as passing none; stock behaviour
    must stay exactly stock."""
    profile.planner.options["tamp"] = {}
    options = resolve_profile(profile, machine_rig)
    assert render.write_tamp_overrides(profile, tmp_path / "o.json", options) is None


def test_overrides_json_is_written_when_set(profile, machine_rig, tmp_path):
    import json

    profile.planner.options["tamp"] = {"num_particles": 64, "traj_length_norm": "inf"}
    path = render.write_tamp_overrides(profile, tmp_path / "o.json", resolve_profile(profile, machine_rig))
    assert path is not None
    written = json.loads(path.read_text())
    assert written == {"num_particles": 64, "traj_length_norm": "inf"}


# --------------------------------------------------------------------------- the layout (version 3)


def test_a_profile_is_one_file_and_its_trajectories_are_under_trajectories(profile, isolated_env):
    data = isolated_env / "data"
    assert profile.file() == data / "profiles" / "test.yml"
    assert profile.file().is_file()
    assert profile.trajectories_dir() == data / "trajectories" / "test"
    for status in profiles.STATUSES:
        assert (data / "trajectories" / "test" / status).is_dir()
    # Nothing in profiles/ but the file: no directory per profile, no calibration of its own.
    assert sorted(p.name for p in (data / "profiles").iterdir()) == ["test.yml"]


def test_the_name_is_the_files_and_is_never_written(profile):
    text = profile.file().read_text()
    assert "\nname:" not in f"\n{text}"
    assert "version: 3" in text
    # A stale name inside a copied file does not rename the profile.
    profile.file().write_text("name: somebody-else\n" + text)
    assert profiles.load("test").name == "test"


@pytest.mark.parametrize(
    "text, said, not_said",
    [
        # Nothing to move: the version is all that is old.
        ("version: 2\ntask: {prompt: x}\n", ["says version 2", "tandem profile migrate", "set version: 3"], ["rig"]),
        # A version-3 file with a machine's section: where it lives, and no migration advice.
        (
            "version: 3\ncameras: {perception: external}\n",
            ["cameras are this machine's rig", "`tandem rig set cameras.ROLE.serial SERIAL`"],
            ["migrate", "set version"],
        ),
        ("robot: {host: 10.0.0.5}\n", ["robot.host and robot.type are the rig's", "`tandem rig set robot.host"], ["migrate"]),
        # tamp is a task's: it goes in the profile, not the rig (which refuses it).
        ("version: 3\ntamp: {num_particles: 8}\n", ["planner.options.tamp"], ["rig", "migrate"]),
    ],
)
def test_a_section_that_is_not_a_tasks_is_refused_with_where_it_lives(isolated_env, text, said, not_said):
    root = isolated_env / "data" / "profiles"
    root.mkdir(parents=True)
    (root / "old.yml").write_text(text)
    with pytest.raises(ProfileInvalid) as excinfo:
        profiles.load("old")
    message = excinfo.value.message
    for words in said:
        assert words in message
    for words in not_said:
        assert words not in message.split("is not a valid profile:")[1]


def test_a_newer_layout_is_refused_as_one(isolated_env):
    root = isolated_env / "data" / "profiles"
    root.mkdir(parents=True)
    (root / "future.yml").write_text("version: 4\n")
    with pytest.raises(ProfileInvalid, match="newer tandem"):
        profiles.load("future")


def test_recording_fps_is_refused_as_the_setting_nothing_read(isolated_env):
    with pytest.raises(ValidationError, match="nothing ever read it"):
        Profile.model_validate({"name": "x", "recording": {"enabled": True, "fps": 15}})


def test_list_names_is_the_valid_yml_files_only(profile, isolated_env):
    root = isolated_env / "data" / "profiles"
    (root / "second.yml").write_text("version: 3\n")
    (root / ".hidden.yml").write_text("version: 3\n")
    (root / "Not A Name.yml").write_text("version: 3\n")
    (root / "notes.txt").write_text("")
    (root / "test.yml.bak").write_text("")
    (root / ".planner-options").mkdir()
    (root / ".migrated").mkdir()
    assert profiles.list_names() == ["second", "test"]


def test_an_old_layout_profile_is_not_listed_but_said_to_be_there(isolated_env, caplog):
    root = isolated_env / "data" / "profiles"
    (root / "legacy").mkdir(parents=True)
    (root / "legacy" / "profile.yml").write_text("version: 2\n")
    with caplog.at_level("WARNING", logger="tandem.core.profiles"):
        assert profiles.list_names() == []
        assert profiles.list_names() == []
    notices = [r.getMessage() for r in caplog.records if "old layout" in r.getMessage()]
    assert len(notices) == 1, "said once per process, however often the profiles are listed"
    assert "legacy" in notices[0] and "tandem init" in notices[0]
    with pytest.raises(ProfileError) as excinfo:
        profiles.load("legacy")
    assert "tandem profile migrate" in (excinfo.value.hint or "")


def test_delete_keeps_the_trajectories_and_purge_removes_them(profile):
    (profile.status_dir("success") / "20260101-000000").mkdir()
    profiles.delete("test")
    assert not profile.file().exists()
    assert (profile.status_dir("success") / "20260101-000000").is_dir()
    assert not profiles.exists("test")
    # Soft-deleted: its data can still be purged, and nothing but it goes.
    profiles.delete("test", keep_data=False)
    assert not profile.trajectories_dir().exists()
    assert profile.trajectories_dir().parent.is_dir()
    with pytest.raises(ProfileError, match="does not exist"):
        profiles.delete("test")


def test_purge_refuses_anything_but_a_trajectories_directory(profile, tmp_path):
    with pytest.raises(ProfileError, match="not a profile name"):
        profiles.delete("..", keep_data=False)
    # A symlinked trajectories directory is not purged through: its target is somebody else's data.
    elsewhere = tmp_path / "precious"
    elsewhere.mkdir()
    (elsewhere / "keep.txt").write_text("x")
    import shutil

    shutil.rmtree(profile.trajectories_dir())
    profile.trajectories_dir().symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ProfileError, match="Refusing to purge"):
        profiles.delete("test", keep_data=False)
    assert (elsewhere / "keep.txt").is_file()
    assert profile.file().is_file(), "a refused purge deletes nothing, the file included"


def test_a_previous_planners_options_are_set_aside_in_a_hidden_directory(profile, isolated_env):
    assert profiles.stash_file("test", "tiptop") == (
        isolated_env / "data" / "profiles" / ".planner-options" / "test.tiptop.yml"
    )
    with pytest.raises(ProfileError):
        profiles.stash_file("../x", "tiptop")


def test_a_relative_path_is_read_beside_the_profiles_file(profile, isolated_env):
    profile.hitl.cache_path = "cache/proposals.sqlite"
    assert profiles.resolve_cache_path(profile) == str(
        (isolated_env / "data" / "profiles" / "cache" / "proposals.sqlite").resolve()
    )
    profile.hitl.cache_path = "/abs/proposals.sqlite"
    assert profiles.resolve_cache_path(profile) == "/abs/proposals.sqlite"


def test_a_pinned_profile_keeps_its_data_root(profile, isolated_env, monkeypatch, tmp_path):
    pinned = profile.pinned()
    monkeypatch.setenv("TANDEM_DATA_ROOT", str(tmp_path / "elsewhere"))
    from tandem.core import settings as settings_mod

    settings_mod._cache = None
    assert pinned.file() == isolated_env / "data" / "profiles" / "test.yml"
    assert pinned.trajectories_dir() == isolated_env / "data" / "trajectories" / "test"
    assert profile.file() == tmp_path / "elsewhere" / "profiles" / "test.yml"
