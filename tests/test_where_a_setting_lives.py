"""A setting given in the wrong place is refused with where it lives.

`tandem config set robot.host ...` said only "Unknown setting"; the robot's address is the rig's.
"""

from __future__ import annotations

from typer.testing import CliRunner

from tandem.cli.app import app
from tandem.core import profiles


def _run(*args: str):
    return CliRunner().invoke(app, list(args), env={"COLUMNS": "1000"})


def test_config_set_of_a_rig_setting_points_at_the_rig(isolated_env):
    result = _run("config", "set", "robot.host", "1.2.3.4")
    assert result.exit_code == 1
    assert "this machine's rig" in result.exception.message
    assert result.exception.hint.startswith("`tandem rig set robot.host 1.2.3.4`")
    for key in ("cameras.hand.serial", "calibration", "planners.tiptop.robot.port"):
        assert "`tandem rig set" in _run("config", "get", key).exception.hint


def test_config_set_of_a_task_setting_points_at_the_profile(isolated_env):
    result = _run("config", "set", "hitl.enabled", "true")
    assert result.exit_code == 1 and "a profile's" in result.exception.message
    assert "`tandem profile edit NAME`" in result.exception.hint


def test_config_set_of_a_typo_still_points_at_the_listing(isolated_env):
    result = _run("config", "set", "data_rot", "/x")
    assert result.exit_code == 1 and "`tandem config list`" in result.exception.hint


def test_profile_show_says_what_becomes_of_a_trial_whose_check_still_fails(machine_rig):
    profiles.create("cups", prompt="stack the cups")
    said = " ".join(_run("profile", "show", "cups").output.split())
    assert "the trial ends, and is excluded from the dataset" in said and "the rollout fails" not in said
