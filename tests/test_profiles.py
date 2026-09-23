"""Profile loading, validation and the TAMP-key contract.

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
from tandem.core.errors import ProfileError
from tandem.core.profiles import Profile, validate_tamp
from tandem.planners.tiptop import render


def test_template_is_valid():
    """The shipped template must load, or `tandem init` fails on a fresh machine."""
    profile = profiles.load_file(resources.path("profile_template.yml"), name="default")
    assert profile.name == "default"
    assert profile.robot.dof == 7
    assert profile.cameras.perception in {"hand", "external"}


def test_save_and_load_roundtrip(profile):
    loaded = profiles.load("test")
    assert loaded.task.prompt == profile.task.prompt
    assert loaded.tamp == profile.tamp
    for status in profiles.STATUSES:
        assert loaded.status_dir(status).is_dir()


def test_unknown_tamp_key_is_rejected_with_a_suggestion():
    with pytest.raises(ValueError) as excinfo:
        validate_tamp({"vae_manifold_weigth": 100})  # transposed letters
    message = str(excinfo.value)
    assert "unknown TAMP setting" in message
    assert "vae_manifold_weight" in message  # the suggestion


def test_traj_length_norm_normalises_to_the_string_inf():
    """The overrides dict round-trips through JSON, which has no Infinity literal.

    YAML `inf` parses as a string and `.inf` as a float; both have to land on the one form
    the planner accepts.
    """
    assert validate_tamp({"traj_length_norm": "inf"})["traj_length_norm"] == "inf"
    assert validate_tamp({"traj_length_norm": "INFINITY"})["traj_length_norm"] == "inf"
    assert validate_tamp({"traj_length_norm": math.inf})["traj_length_norm"] == "inf"
    assert validate_tamp({"traj_length_norm": 2})["traj_length_norm"] == 2.0


def test_blend_ops_typo_is_caught():
    """An upstream config really did ship `MoveFree. MoveHolding` — one typo'd period that
    YAML folded into a single token and the planner ignored for months."""
    with pytest.raises(ValueError) as excinfo:
        validate_tamp({"blend_ops": ["Pick", "MoveFree. MoveHolding"]})
    assert "unknown operations" in str(excinfo.value)


def test_blend_mode_enum_is_enforced():
    with pytest.raises(ValueError):
        validate_tamp({"blend_mode": "neural"})  # documented once, never implemented
    assert validate_tamp({"blend_mode": "flow"})["blend_mode"] == "flow"


def test_scalar_types_are_enforced():
    with pytest.raises(ValueError):
        validate_tamp({"num_particles": "many"})
    with pytest.raises(ValueError):
        validate_tamp({"blend_trajectory": "yes"})  # a string, not a bool
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


def test_joint_vector_length_must_match_dof():
    with pytest.raises(ValidationError) as excinfo:
        Profile.model_validate({"name": "x", "robot": {"dof": 7, "q_home": [0.0, 0.0]}})
    assert "q_home" in str(excinfo.value)


def test_perception_camera_must_be_configured_when_there_are_cameras():
    with pytest.raises(ValidationError) as excinfo:
        Profile.model_validate({
            "name": "x",
            "cameras": {"perception": "external", "hand": {"serial": "1"}},
        })
    assert "cameras.external is not configured" in str(excinfo.value)


def test_a_profile_with_no_cameras_is_valid():
    """On a laptop a profile is just a folder of trajectories collected elsewhere. Collection
    refuses separately, where the message can be specific."""
    assert Profile.model_validate({"name": "viz-only"}).cameras.configured() == {}


def test_time_dilation_factor_bounds():
    with pytest.raises(ValidationError):
        Profile.model_validate({"name": "x", "robot": {"time_dilation_factor": 0.0}})
    with pytest.raises(ValidationError):
        Profile.model_validate({"name": "x", "robot": {"time_dilation_factor": 2.0}})


def test_missing_profile_names_the_alternatives(profile):
    with pytest.raises(ProfileError) as excinfo:
        profiles.load("nope")
    assert "test" in (excinfo.value.hint or "")


def test_missing_calibration_is_reported(profile):
    """Extrinsics are keyed by serial, and a serial with no entry aborts at warmup — so a
    swapped camera has to be caught before a session starts, not minutes in."""
    assert profiles.missing_calibration(profile) == []

    profile.cameras.external.serial = "99999999"  # a camera was swapped for another unit
    assert profiles.missing_calibration(profile) == ["99999999"]


def test_render_env_sets_the_contract_variables(profile, tmp_path):
    events = tmp_path / "events.jsonl"
    env = render.render_env(profile, events_file=events, task="do the thing", base={})
    assert env["TIPTOP_TASK"] == "do the thing"
    assert env["TIPTOP_EVENTS_FILE"] == str(events)
    assert env["TIPTOP_STATE_PORT"] == str(profile.robot.state_port)
    assert env["TIPTOP_CALIBRATION"] == str(profile.calibration_file())
    assert env["DC_WORKSPACE"] == profile.name
    # opencv's LAPACK and torch share one libmkl_core; the threaded path corrupts a pivot
    # array and cuRobo dies inside torch.inverse.
    assert env["MKL_NUM_THREADS"] == "1"


def test_render_env_only_splits_instruction_when_the_goal_differs(profile, tmp_path):
    events = tmp_path / "events.jsonl"
    env = render.render_env(profile, events_file=events, base={})
    assert "TIPTOP_INSTRUCTION" not in env

    profile.task.goal = "a reduced goal the planner can express"
    env = render.render_env(profile, events_file=events, base={})
    assert env["TIPTOP_INSTRUCTION"] == profile.task.prompt
    assert env["TIPTOP_TASK"] == profile.task.goal


def test_render_tiptop_config_shape(profile):
    config = render.render_tiptop_config(profile)
    assert set(config) == {"robot", "cameras", "perception"}
    assert config["robot"]["type"] == profile.robot.type
    assert config["cameras"]["perception"] == profile.cameras.perception
    assert "depth_smoothing" in config["perception"]


def test_no_overrides_file_when_the_profile_sets_nothing(profile, tmp_path):
    """Passing an empty --curobo-overrides is not the same as passing none; stock behaviour
    must stay exactly stock."""
    profile.tamp = {}
    assert render.write_tamp_overrides(profile, tmp_path / "o.json") is None


def test_overrides_json_is_written_when_set(profile, tmp_path):
    import json

    profile.tamp = {"num_particles": 64, "traj_length_norm": "inf"}
    path = render.write_tamp_overrides(profile, tmp_path / "o.json")
    assert path is not None
    written = json.loads(path.read_text())
    assert written == {"num_particles": 64, "traj_length_norm": "inf"}
