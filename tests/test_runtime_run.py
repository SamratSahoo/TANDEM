"""`tandem runtime run` and `runtime shell`: a command in the planner's runtime, reaching this machine.

    tandem runtime run cutamp-demo --motion_plan        the command's options are the command's
    tandem runtime run calibrate-wrist-cam              tiptop reads the rig: the NUC's address, the cameras,
                                                        and writes the extrinsics into the rig's calibration.json
    tandem runtime run --raw calibrate-wrist-cam        tiptop's own config, as it ships

The runtime and the subprocess are stand-ins; the environment is the real one TiPToP's factory writes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from helpers import isolate_registry
from ruamel.yaml import YAML
from toy_planner import ToyPlanner
from typer.testing import CliRunner

from tandem.cli import runtime as runtime_cli
from tandem.cli.app import app
from tandem.core import rig as rig_mod
from tandem.core.errors import TandemError
from tandem.planners import registry

TIPTOP_VARS = ("TIPTOP_CONFIG", "TIPTOP_CALIBRATION", "TIPTOP_HAND_CAMERA_ID", "TIPTOP_EXTERNAL_CAMERA_ID")


class StandInRuntime:
    """A built runtime whose commands are recorded, not run."""

    def __init__(self, root: Path) -> None:
        self.root = self.workdir = root

    def require_ready(self) -> None:
        return None

    def command(self, args):
        return ["pixi", "run", *args]

    def shell_command(self):
        return ["pixi", "shell"]


@pytest.fixture
def calls(tmp_path, monkeypatch, machine_rig):
    """What subprocess.call was asked to run: (argv, env). It exits with $STAND_IN_EXIT, default 0."""
    made: list[tuple[list[str], dict]] = []
    runtime = StandInRuntime(tmp_path / "runtime")

    def call(argv, cwd=None, env=None):
        made.append((list(argv), dict(env or {})))
        return int((env or {}).get("STAND_IN_EXIT", "0"))

    monkeypatch.setattr(runtime_cli, "_recipe_runtime", lambda planner, profile: (planner or "tiptop", runtime))
    monkeypatch.setattr(runtime_cli.subprocess, "call", call)
    for var in (*TIPTOP_VARS, "TIPTOP_STATE_PORT"):
        monkeypatch.delenv(var, raising=False)
    return made


def _run(*args: str):
    return CliRunner().invoke(app, ["runtime", *args])


# --------------------------------------------------------------------------- the command's own options


def test_the_help_example_works_as_written(calls):
    result = _run("run", "cutamp-demo", "--motion_plan")
    assert result.exit_code == 0, result.output
    assert calls[0][0] == ["pixi", "run", "cutamp-demo", "--motion_plan"]


def test_everything_after_the_command_is_the_commands(calls):
    assert _run("run", "viz-calibration", "--camera", "external", "-h").exit_code == 0
    assert calls[-1][0][2:] == ["viz-calibration", "--camera", "external", "-h"], "-h is the command's, not tandem's"
    # `--` before the command still works.
    assert _run("run", "--", "viz-calibration", "--camera", "external").exit_code == 0
    assert calls[-1][0][2:] == ["viz-calibration", "--camera", "external"]
    # tandem's own options go before it.
    assert _run("run", "--raw", "--planner", "tiptop", "cutamp-demo", "--planner", "x").exit_code == 0
    assert calls[-1][0][2:] == ["cutamp-demo", "--planner", "x"]


def test_the_commands_exit_status_is_tandems(calls, monkeypatch):
    monkeypatch.setenv("STAND_IN_EXIT", "3")
    assert _run("run", "cutamp-demo").exit_code == 3


def test_tandems_own_help_is_still_there():
    result = _run("run", "--help")
    assert result.exit_code == 0
    # Unstyled and unboxed: typer draws rich's terminal output under GITHUB_ACTIONS, bold and dim included.
    output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    said = " ".join("".join(c for c in output if c not in "│╭╮╰╯─").split())
    assert "COMMAND [ARGS]..." in said and "--raw" in said and "cutamp-demo --motion_plan" in said


# --------------------------------------------------------------------------- the rig, for tiptop's scripts


def test_a_command_reads_this_machines_rig(calls, machine_rig):
    rig_mod.update({"robot.host": "172.16.0.5", "planners.tiptop.robot.state_port": 6007})
    rig = rig_mod.load(force=True)
    result = _run("run", "calibrate-wrist-cam")
    assert result.exit_code == 0, result.output
    env = calls[0][1]
    assert env["TIPTOP_CALIBRATION"] == str(rig.calibration_file())
    assert env["TIPTOP_HAND_CAMERA_ID"] == rig.cameras.hand.serial
    assert env["TIPTOP_EXTERNAL_CAMERA_ID"] == rig.cameras.external.serial
    assert env["TIPTOP_STATE_PORT"] == "6007"
    assert "PATH" in env, "added to the caller's environment, not instead of it"

    config = YAML(typ="safe").load(Path(env["TIPTOP_CONFIG"]).read_text())
    assert config["robot"]["host"] == "172.16.0.5" and config["robot"]["type"] == "fr3_robotiq"
    assert config["cameras"]["hand"]["serial"] == rig.cameras.hand.serial
    assert config["cameras"]["external"]["serial"] == rig.cameras.external.serial
    assert config["perception"]["foundation_stereo"]["url"] == "http://localhost:1234"
    # Which robot it will reach is said, on stderr.
    assert "172.16.0.5" in " ".join(result.output.split()) and "--raw" in result.output


def test_raw_leaves_tiptops_own_config(calls):
    assert _run("run", "--raw", "calibrate-wrist-cam").exit_code == 0
    assert not set(TIPTOP_VARS) & set(calls[0][1])


def test_the_calibration_file_is_there_for_the_script_to_write(calls, machine_rig):
    machine_rig.calibration_file().unlink()
    assert _run("run", "calibrate-wrist-cam").exit_code == 0
    assert json.loads(machine_rig.calibration_file().read_text()) == {}


def test_an_arm_tiptop_does_not_drive_is_said_and_raw_still_runs(calls):
    rig_mod.update({"robot.type": "kuka"})
    refused = _run("run", "calibrate-wrist-cam")
    assert refused.exit_code == 1 and "robot.type" in refused.exception.message
    assert "tandem rig set robot.type" in refused.exception.hint
    assert calls == []
    assert _run("run", "--raw", "calibrate-wrist-cam").exit_code == 0


def test_the_shell_reads_the_rig_too_unless_raw(calls):
    assert _run("shell").exit_code == 0
    assert calls[-1][0] == ["pixi", "shell"] and "TIPTOP_CONFIG" in calls[-1][1]
    assert _run("shell", "--raw").exit_code == 0
    assert "TIPTOP_CONFIG" not in calls[-1][1]


# --------------------------------------------------------------------------- any planner


def test_a_planner_with_nothing_to_add_adds_nothing(monkeypatch, machine_rig):
    isolate_registry(monkeypatch)
    registry.register_backend("toy", ToyPlanner)
    assert registry.runtime_env("toy", rig=machine_rig) == {}


def test_a_planner_that_returns_what_exec_cannot_take_is_refused(monkeypatch, machine_rig):
    isolate_registry(monkeypatch)

    class Numbers(ToyPlanner):
        @classmethod
        def runtime_env(cls, *, rig, settings=None):
            return {"PORT": 5555}

    registry.register_backend("toy", Numbers)
    with pytest.raises(TandemError, match="not a mapping of strings to strings") as info:
        registry.runtime_env("toy", rig=machine_rig)
    assert "--raw" in info.value.hint


def test_with_no_rig_yet_a_command_runs_with_the_planners_own_config_and_says_so(calls, machine_rig):
    """A config rendered from the rig's defaults names a robot nobody set up and no cameras."""
    rig_mod.paths.rig_file().unlink()
    machine_rig.calibration_file().unlink()
    result = _run("run", "cutamp-demo", "--motion_plan")
    assert result.exit_code == 0, result.output
    assert calls[0][0] == ["pixi", "run", "cutamp-demo", "--motion_plan"]
    assert not set(TIPTOP_VARS) & set(calls[0][1]), "as --raw"
    said = " ".join(result.output.split())
    assert "no rig on this machine yet" in said and "`tandem init`" in said
    assert "172.16.0.2" not in said, "no robot is named that nobody set up"
    assert not machine_rig.calibration_file().exists(), "nothing is created on the way"
