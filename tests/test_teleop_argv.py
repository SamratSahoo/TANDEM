"""The command line a hand-off launches the teleop driver with, checked against the REAL driver's flags.

Every hand-off test runs tests/fake_teleop.py, which used to ignore flags it did not know. The real driver
(teleop/driver.py) parses its command line from its ``Args`` dataclass (``parse_args``), which exits on one it does not know
-- so a flag renamed in teleop/child.py passed the suite and then stopped every hand-off on a robot at
the first leg. And ``--keep-pose`` is a safety flag, not a nicety: without it the driver homes the arm
mid-task, holding whatever the planner left in its gripper.

The driver is imported the way test_merge_optional_keys imports it (as the DROID interpreter runs it),
and only its ``Args`` dataclass is read: nothing starts.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tandem import teleop as teleop_pkg
from tandem.core.settings import Settings
from tandem.teleop import child as child_mod


@pytest.fixture
def driver_flags(monkeypatch) -> set[str]:
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location("tandem_teleop_driver_flags", teleop_pkg.driver_path())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The driver's spelling of a dataclass field on the command line: underscores become dashes.
    return {"--" + f.name.replace("_", "-") for f in dataclasses.fields(module.Args)}


class _Session:
    """Just enough Session for TeleopChild.start to build its command line, with all three cameras."""

    def __init__(self, tmp_path: Path) -> None:
        cams = {
            "hand": SimpleNamespace(serial="111"),
            "external": SimpleNamespace(serial="222"),
            "external_2": SimpleNamespace(serial="333"),
        }
        self._trajectory_id = "traj-abc123"
        self.instruction = "fold the cloth over the toy"
        self._files = {"session_dir": tmp_path}
        self.current = None
        self.profile = SimpleNamespace(
            trajectories_dir=lambda: tmp_path / "trajectories",
            cameras=SimpleNamespace(configured=lambda: cams),
        )
        self.logs: list[str] = []

    def _log(self, stream: str, text: str) -> None:
        self.logs.append(text)

    def _pump(self, stream, name: str) -> None:
        return None


def _argv(tmp_path, monkeypatch) -> list[str]:
    cfg = Settings()
    cfg.teleop.enabled = True
    cfg.teleop.python = sys.executable
    cfg.teleop.droid_dir = str(tmp_path)
    launched: list[list[str]] = []

    class _Proc:
        stdout = None

    def popen(args, **kwargs):
        launched.append(list(args))
        return _Proc()

    monkeypatch.setattr(child_mod.subprocess, "Popen", popen)
    monkeypatch.setattr(child_mod.events_mod, "EventTailer", lambda *a, **k: SimpleNamespace(start=lambda: None))
    child = child_mod.TeleopChild(
        _Session(tmp_path), cfg, phase_index=0, n_phases=2, phase_description="open the box"
    )
    child.start()
    assert launched, "the driver was never launched"
    # argv[0] is the interpreter and argv[1] the driver; the rest is what the driver parses.
    return launched[0][2:]


def test_every_flag_the_hand_off_passes_is_one_the_real_driver_takes(tmp_path, monkeypatch, driver_flags):
    argv = _argv(tmp_path, monkeypatch)
    passed = [token for token in argv if token.startswith("--")]
    unknown = sorted(set(passed) - driver_flags)
    assert not unknown, f"the real driver would exit on {unknown} (it takes {sorted(driver_flags)})"


def test_a_hand_off_keeps_the_pose_and_says_which_leg_and_cameras_it_records(tmp_path, monkeypatch, driver_flags):
    argv = _argv(tmp_path, monkeypatch)
    # Without it the driver homes the arm mid-task, still holding what the planner left in the gripper.
    assert "--keep-pose" in argv
    for flag, value in (
        ("--trajectory-id", "traj-abc123"),
        ("--phase-index", "0"),
        ("--n-phases", "2"),
        ("--hand-camera-id", "111"),
        ("--external-camera-id", "222"),
        ("--external-2-camera-id", "333"),
    ):
        assert flag in argv, flag
        assert argv[argv.index(flag) + 1] == value, flag
