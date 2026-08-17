"""Reading, indexing and relabeling trajectories, and the series payload the charts use."""

from __future__ import annotations

import numpy as np
import pytest

from tandem.core import nonidle, series, trajectories
from tandem.core.errors import TandemError


def test_list_and_count(profile, make_trajectory):
    make_trajectory(profile, "2026-01-01_00-00-00", status="success")
    make_trajectory(profile, "2026-01-01_00-01-00", status="failure")
    make_trajectory(profile, "2026-01-01_00-02-00", status="eval")

    items = trajectories.list_all(profile)
    assert len(items) == 3
    # Newest first — the one you just collected is the one you want to see.
    assert items[0].id == "2026-01-01_00-02-00"
    assert trajectories.counts(profile) == {"success": 1, "failure": 1, "eval": 1}


def test_incomplete_trajectory_is_not_counted(profile, make_trajectory):
    directory = make_trajectory(profile, "2026-01-01_00-00-00")
    (directory / "robot_state.npz").unlink()
    assert trajectories.counts(profile)["success"] == 0


def test_find_accepts_a_unique_prefix(profile, make_trajectory):
    make_trajectory(profile, "2026-01-01_00-00-00")
    found = trajectories.find(profile, "2026-01-01_00-00")
    assert found.id == "2026-01-01_00-00-00"


def test_ambiguous_prefix_lists_the_candidates(profile, make_trajectory):
    make_trajectory(profile, "2026-01-01_00-00-00")
    make_trajectory(profile, "2026-01-01_00-00-30")
    with pytest.raises(TandemError) as excinfo:
        trajectories.find(profile, "2026-01-01")
    assert "2026-01-01_00-00-00" in (excinfo.value.hint or "")


def test_relabel_moves_the_directory(profile, make_trajectory):
    make_trajectory(profile, "2026-01-01_00-00-00", status="failure")
    traj = trajectories.find(profile, "2026-01-01_00-00-00")
    moved = trajectories.relabel(profile, traj, "success")
    assert moved.status == "success"
    assert moved.path.is_dir()
    assert not traj.path.exists()
    assert trajectories.counts(profile)["success"] == 1


def test_delete_refuses_a_path_outside_the_profile(profile, make_trajectory, tmp_path):
    make_trajectory(profile, "2026-01-01_00-00-00")
    traj = trajectories.find(profile, "2026-01-01_00-00-00")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    traj.path = outside
    with pytest.raises(TandemError):
        trajectories.delete(profile, traj)
    assert outside.is_dir()


def test_media_path_rejects_traversal(profile, make_trajectory):
    make_trajectory(profile, "2026-01-01_00-00-00")
    traj = trajectories.find(profile, "2026-01-01_00-00-00")
    with pytest.raises(TandemError):
        trajectories.media_path(traj, "../../../etc/passwd")


def test_series_payload_has_everything_the_charts_need(profile, make_trajectory):
    directory = make_trajectory(profile, "2026-01-01_00-00-00", n_frames=200)
    payload = series.series_for(directory)

    assert payload["n_frames"] == 200
    assert len(payload["t"]) == payload["n_plotted"]
    assert payload["t0"] is not None
    assert payload["record_start"] is not None
    for channel in ("joint_position", "cmd_joint_velocity", "cmd_gripper"):
        assert payload[channel] is not None
    assert payload["filter"] is not None
    assert payload["filter"]["n_kept"] <= 200


def test_series_downsamples_but_filters_at_full_resolution(profile, make_trajectory):
    """min_idle_len is 7 frames, so a filter run on the downsampled copy would miss idle
    runs that the real training filter catches."""
    directory = make_trajectory(profile, "2026-01-01_00-00-00", n_frames=5000)
    payload = series.series_for(directory)
    assert payload["downsampled"] is True
    assert payload["n_plotted"] == series.MAX_POINTS
    assert payload["filter"]["n_frames"] == 5000


def test_series_needs_state(profile, make_trajectory):
    directory = make_trajectory(profile, "2026-01-01_00-00-00")
    (directory / "robot_state.npz").unlink()
    with pytest.raises(TandemError):
        series.series_for(directory)


# --- the non-idle filter ---------------------------------------------------


def test_keep_ranges_drop_a_long_idle_run():
    """A constant stretch longer than min_idle_len is dropped; the moving parts survive."""
    moving = np.tile(np.arange(40, dtype=np.float32).reshape(-1, 1), (1, 7)) * 0.01
    idle = np.tile(moving[-1], (30, 1))
    more = idle[-1] + np.tile(np.arange(1, 41, dtype=np.float32).reshape(-1, 1), (1, 7)) * 0.01
    signal = np.concatenate([moving, idle, more]).astype(np.float32)

    ranges, reasons = nonidle.analyze(signal)
    assert ranges, "a 40-frame moving run should survive"
    assert np.any(reasons == nonidle.IDLE)
    # Every kept frame is inside a returned range, and vice versa.
    kept = np.zeros(len(signal), dtype=bool)
    for start, end in ranges:
        kept[start:end] = True
    assert np.array_equal(kept, reasons == nonidle.KEEP)


def test_filter_runs_on_the_clipped_action():
    """Saturation creates idle frames: 1.3 and 1.5 both become 1.0, so their difference is
    zero. Filtering the raw command reports a different — wrong — set of dropped frames."""
    raw = np.ones((30, 7), dtype=np.float32) * 1.3
    raw[15:] = 1.5
    clipped, frac = nonidle.clip_action(raw)
    assert frac == 1.0
    assert np.all(clipped == 1.0)
    _, reasons = nonidle.analyze(clipped)
    assert np.any(reasons != nonidle.KEEP)


def test_empty_input_is_handled():
    ranges, reasons = nonidle.analyze(np.zeros((0, 7), dtype=np.float32))
    assert ranges == []
    assert len(reasons) == 0


def test_summary_reports_gripper_events(profile, make_trajectory):
    make_trajectory(profile, "2026-01-01_00-00-00", n_frames=90)
    traj = trajectories.find(profile, "2026-01-01_00-00-00")
    stats = series.summary(traj)
    # The fixture closes then opens the gripper: two transitions.
    assert stats["gripper_events"] == 2
    assert stats["joint_travel_rad"] > 0
