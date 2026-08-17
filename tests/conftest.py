"""Shared fixtures.

Every test runs against an isolated config/data root, so nothing touches a real install.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


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

    # These modules cache the loaded settings; a stale cache leaks one test's config into
    # the next.
    from tandem.core import settings as settings_mod

    settings_mod._cache = None
    yield tmp_path
    settings_mod._cache = None


@pytest.fixture
def profile(isolated_env):
    """A saved profile named `test`, built from the shipped template.

    Extrinsics are filled in for every configured camera: collection refuses to start without
    them (a serial with no entry aborts at warmup), so a fixture missing them would make every
    session test fail for a reason that has nothing to do with what it is testing.
    """
    from tandem import resources
    from tandem.core import profiles

    prof = profiles.load_file(resources.path("profile_template.yml"), name="test")
    profiles.save(prof)
    prof.calibration_file().write_text(
        json.dumps(
            {
                camera.serial: {"pose": [0.4, 0.0, 0.6, 0.0, 0.0, 0.0], "timestamp": 0}
                for camera in prof.cameras.configured().values()
            },
            indent=2,
        )
    )
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
