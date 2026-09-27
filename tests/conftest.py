"""Shared fixtures.

Every test runs against an isolated config/data root, so nothing touches a real install. The run as a
whole runs under a home of its own too, and fails if anything was written under the real one anyway
(``home_guard``).
"""

from __future__ import annotations

# First, before anything reads the environment it remembers.
import home_guard  # isort: skip

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

_RUN_ROOT = pytest.StashKey[Path]()
_BEFORE = pytest.StashKey[dict]()
_WRITTEN = pytest.StashKey[list]()


def pytest_configure(config):
    """Give the run a home, XDG directories and tandem roots of its own, under a temporary directory.

    ``isolated_env`` does the same for each test, but only for the test: a thread that outlives it --
    a merge, a session still ending -- runs on after the environment is put back, and resolved the
    real ~/tandem-data. With this underneath, whatever resolves late still lands somewhere temporary.
    """
    root = Path(tempfile.mkdtemp(prefix="tandem-tests-"))
    config.stash[_RUN_ROOT] = root
    config.stash[_BEFORE] = home_guard.snapshot()
    home_guard.apply(root)


def pytest_sessionfinish(session, exitstatus):
    """Fail the run if anything was written under the real home's tandem directories while it ran."""
    config = session.config
    if _BEFORE not in config.stash:
        return
    written = home_guard.changes(config.stash[_BEFORE], home_guard.snapshot())
    if written:
        config.stash[_WRITTEN] = written
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    written = config.stash.get(_WRITTEN, None)
    if not written:
        return
    terminalreporter.section("the test run wrote into the real home", sep="!", red=True, bold=True)
    terminalreporter.line(
        "Every tandem root is a temporary directory during the run, so something resolved a path outside "
        "them -- most likely a background thread that outlived its test, or a path read before conftest "
        "moved HOME. What changed:",
        red=True,
    )
    for line in written[:40]:
        terminalreporter.line(f"  {line}", red=True)
    if len(written) > 40:
        terminalreporter.line(f"  ... and {len(written) - 40} more", red=True)


def pytest_unconfigure(config):
    home_guard.restore()
    root = config.stash.get(_RUN_ROOT, None)
    if root is not None:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Point every tandem root at a temp directory, and clear inherited credentials.

    The credential vars matter: a developer machine usually exports GEMINI_API_KEY, which
    would make a test that asserts "no key configured" pass for the wrong reason.
    """
    monkeypatch.setenv("TANDEM_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("TANDEM_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("TANDEM_SHARE_DIR", str(tmp_path / "share"))
    monkeypatch.setenv("TANDEM_DATA_ROOT", str(tmp_path / "data"))
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    # These modules cache what they loaded (the settings, the rig) and say some things once per
    # process; a stale cache leaks one test's config into the next.
    from tandem.core import profiles
    from tandem.core import rig as rig_mod
    from tandem.core import settings as settings_mod

    def forget() -> None:
        settings_mod._cache = None
        rig_mod._cache = None
        rig_mod._stamp = None
        rig_mod._noticed.clear()
        profiles._noticed_old_layout = False

    # A session starts its first rollout on its own once warm. Most tests want it held at the task
    # prompt, to look at the session before the attempt starts; those that test the default undo this.
    from tandem.core import session as session_mod

    monkeypatch.setattr(session_mod, "AUTO_START_FIRST_TASK", False)

    forget()
    yield tmp_path
    forget()


#: The rig's cameras in every test that collects: a wrist and a third-person ZED, perception reading the latter.
RIG_CAMERAS = {"hand": "14846828", "external": "32439448"}


@pytest.fixture
def machine_rig(isolated_env):
    """This machine's rig, as a workstation that collects has it: rig.yml with a wrist and an external
    camera, and extrinsics for both in its calibration.json.

    Extrinsics are filled in for every camera: collection refuses to start without them (a serial with no
    entry aborts at warmup), so a fixture missing them would make every session test fail for a reason
    that has nothing to do with what it is testing. Named machine_rig, not rig: a test module has a
    rig() helper of its own.
    """
    from tandem.core import rig as rig_mod

    rig = rig_mod.update(
        {
            "cameras.perception": "external",
            **{f"cameras.{role}": {"serial": serial} for role, serial in RIG_CAMERAS.items()},
        }
    )
    rig.calibration_file().write_text(
        json.dumps(
            {serial: {"pose": [0.4, 0.0, 0.6, 0.0, 0.0, 0.0], "timestamp": 0} for serial in RIG_CAMERAS.values()},
            indent=2,
        )
    )
    return rig_mod.load(force=True)


@pytest.fixture
def profile(machine_rig):
    """A saved profile named `test` (tests/fixtures/profiles/test_v3.yml), on the `machine_rig` rig.

    Not built from the shipped template: that holds whatever settings new profiles should start with, and
    a test's premises (phase planning off, four TAMP overrides) must not move when it does.
    """
    from tandem.core import profiles

    prof = profiles.load_file(Path(__file__).parent / "fixtures" / "profiles" / "test_v3.yml", name="test")
    profiles.save(prof)
    return prof


def write_trajectory(
    profile,
    timestamp: str,
    *,
    status: str = "success",
    n_frames: int = 60,
    instruction: str = "test task",
    trajectory_id: str | None = None,
    segment_source: str | None = None,
    with_plan: bool = True,
    record_window: tuple[float, float] | None = None,
) -> Path:
    """Write a trajectory in the real on-disk format, minus the videos."""
    directory = profile.status_dir(status) / timestamp
    directory.mkdir(parents=True, exist_ok=True)

    t0 = 1_800_000_000.0
    frame_time = t0 + np.arange(n_frames) / 15.0

    # A plausible motion: a slow sweep with two gripper events, so the non-idle filter has
    # something real to chew on.
    phase = np.linspace(0, 2 * np.pi, n_frames)
    joints = np.stack([np.sin(phase + j) * 0.3 for j in range(7)], axis=1).astype(np.float32)
    cmd_gripper = np.zeros(n_frames, dtype=np.float32)
    cmd_gripper[n_frames // 3 : 2 * n_frames // 3] = 1.0

    np.savez(
        directory / "robot_state.npz",
        joint_position=joints,
        gripper_position=cmd_gripper * 0.9,
        cmd_joint_position=joints * 1.01,
        cmd_joint_velocity=np.gradient(joints, axis=0).astype(np.float32) * 15.0,
        cmd_gripper=cmd_gripper,
        frame_time=frame_time.astype(np.float64),
    )

    start, stop = record_window or (float(frame_time[0]) - 1.0, float(frame_time[-1]) + 1.0)
    meta = {
        "instruction": instruction,
        "fps": 15,
        "n_frames": n_frames,
        "config_id": None,
        "timestamp": timestamp,
        "source": "tiptop",
        "cameras": {"observation.images.exterior_1_left": "external_cam.mp4"},
        "record_start": start,
        "record_stop": stop,
    }
    if trajectory_id:
        meta["trajectory_id"] = trajectory_id
    if segment_source:
        meta["segment_source"] = segment_source
    (directory / "_meta.json").write_text(json.dumps(meta))

    if with_plan:
        (directory / "tiptop_plan.json").write_text(json.dumps({"q_init": [0] * 7, "steps": []}))
    return directory


@pytest.fixture
def make_trajectory():
    return write_trajectory
