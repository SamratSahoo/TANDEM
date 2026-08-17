"""Joining the legs of a TAMP⇄teleop hand-off into one trajectory.

The video concatenation needs ffmpeg and real mp4s, so the tests here cover the parts that
decide correctness: discovery and ordering, the refusals, and the ``video_time`` map — which
is the piece that was measured 726 camera frames (48 s) wrong when it was derived from the
wall clock instead.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tandem.core import merge as merge_mod
from tandem.core.merge import MergeError


def test_find_legs_orders_by_record_start(profile, make_trajectory):
    make_trajectory(
        profile, "2026-01-01_00-02-00", status="eval",
        trajectory_id="abc123", segment_source="tamp", record_window=(200.0, 260.0),
    )
    make_trajectory(
        profile, "2026-01-01_00-00-00", status="eval",
        trajectory_id="abc123", segment_source="tamp", record_window=(100.0, 160.0),
    )
    make_trajectory(
        profile, "2026-01-01_00-01-00", status="eval",
        trajectory_id="abc123", segment_source="teleop", record_window=(170.0, 190.0),
    )

    legs = merge_mod.find_legs(profile, "abc123")
    # Order comes from record_start, never from a counter: the two drivers are separate
    # processes and would have to agree on one.
    assert [leg["dir"].name for leg in legs] == [
        "2026-01-01_00-00-00", "2026-01-01_00-01-00", "2026-01-01_00-02-00",
    ]
    assert [leg["source"] for leg in legs] == ["tamp", "teleop", "tamp"]


def test_find_legs_ignores_other_trajectories(profile, make_trajectory):
    make_trajectory(profile, "2026-01-01_00-00-00", trajectory_id="aaa")
    make_trajectory(profile, "2026-01-01_00-01-00", trajectory_id="bbb")
    assert len(merge_mod.find_legs(profile, "aaa")) == 1


def test_pending_ids_only_lists_multi_leg_trajectories(profile, make_trajectory):
    make_trajectory(profile, "2026-01-01_00-00-00", trajectory_id="multi")
    make_trajectory(profile, "2026-01-01_00-01-00", trajectory_id="multi")
    make_trajectory(profile, "2026-01-01_00-02-00", trajectory_id="single")
    make_trajectory(profile, "2026-01-01_00-03-00")  # no lineage at all
    assert merge_mod.pending_trajectory_ids(profile) == ["multi"]


def test_a_single_leg_is_not_merged(profile, make_trajectory):
    make_trajectory(profile, "2026-01-01_00-00-00", trajectory_id="solo")
    result = merge_mod.merge(profile, "solo")
    assert result["merged"] is False
    assert result["reason"] == "single leg"


def test_an_already_merged_trajectory_is_left_alone(profile, make_trajectory):
    """A merged trajectory keeps its id, so re-running finds itself. Say so plainly rather
    than let it look like a one-leg trajectory."""
    directory = make_trajectory(profile, "2026-01-01_00-00-00", trajectory_id="done")
    meta = json.loads((directory / "_meta.json").read_text())
    meta["video_aligned"] = True
    meta["segments"] = [{"source": "tamp"}, {"source": "teleop"}]
    (directory / "_meta.json").write_text(json.dumps(meta))

    result = merge_mod.merge(profile, "done")
    assert result["merged"] is False
    assert result["reason"] == "already merged"
    assert result["n_legs"] == 2


def test_missing_trajectory_raises(profile):
    with pytest.raises(MergeError):
        merge_mod.merge(profile, "nothing-here")


def test_legs_without_state_are_skipped_not_fatal(profile, make_trajectory):
    """One bad leg out of five is not a reason to leave the operator with five loose
    episodes."""
    good = make_trajectory(profile, "2026-01-01_00-00-00", trajectory_id="mix")
    bad = make_trajectory(profile, "2026-01-01_00-01-00", trajectory_id="mix")
    (bad / "robot_state.npz").unlink()

    result = merge_mod.merge(profile, "mix")
    assert result["merged"] is False
    assert result["reason"] == "fewer than two legs have state data"
    assert result["legs_skipped"] == ["2026-01-01_00-01-00"]
    assert good.is_dir(), "the good leg must be untouched"


def test_unexpected_state_arrays_are_refused(profile, make_trajectory):
    """An unknown array is a schema change. Dropping it silently would leave a dataset that
    looks fine and is not."""
    first = make_trajectory(profile, "2026-01-01_00-00-00", trajectory_id="odd")
    make_trajectory(profile, "2026-01-01_00-01-00", trajectory_id="odd")

    with np.load(first / "robot_state.npz") as store:
        arrays = {key: store[key] for key in store.files}
    arrays["something_new"] = np.zeros(len(arrays["frame_time"]))
    np.savez(first / "robot_state.npz", **arrays)

    legs = merge_mod.find_legs(profile, "odd")
    with pytest.raises(MergeError) as excinfo:
        merge_mod._concat_state(legs, [60, 60], 15)
    assert "unexpected arrays" in str(excinfo.value)


def test_video_time_offsets_by_frame_count_not_wall_clock(profile, make_trajectory):
    """The wall-clock gaps between legs — camera teardown, the human working — are in
    frame_time but not in the video. Using them to seek is the 48-second bug."""
    make_trajectory(
        profile, "2026-01-01_00-00-00", trajectory_id="t", n_frames=30,
        record_window=(1000.0, 1002.0),
    )
    # Second leg starts a long time later in wall clock, but immediately in the video.
    directory = make_trajectory(
        profile, "2026-01-01_00-05-00", trajectory_id="t", n_frames=30,
        record_window=(1300.0, 1302.0),
    )
    with np.load(directory / "robot_state.npz") as store:
        arrays = {key: store[key] for key in store.files}
    arrays["frame_time"] = 1300.0 + np.arange(30) / 15.0
    np.savez(directory / "robot_state.npz", **arrays)

    legs = merge_mod.find_legs(profile, "t")
    state = merge_mod._concat_state(legs, [30, 30], 15)
    video_time = state["arrays"]["video_time"]

    assert len(video_time) == 60
    assert state["total_video_frames"] == 60
    # Monotonic and bounded by the joined clip's length — not by the five-minute wall gap.
    assert np.all(np.diff(video_time) >= 0)
    assert video_time[-1] <= 60 / 15.0
    # The second leg starts right where the first leg's frames end.
    assert video_time[30] == pytest.approx(30 / 15.0, abs=0.1)


def test_proportional_fallback_is_recorded_not_silent(profile, make_trajectory):
    first = make_trajectory(profile, "2026-01-01_00-00-00", trajectory_id="fb", n_frames=20)
    second = make_trajectory(profile, "2026-01-01_00-01-00", trajectory_id="fb", n_frames=20)
    meta = json.loads((second / "_meta.json").read_text())
    del meta["record_start"]
    del meta["record_stop"]
    (second / "_meta.json").write_text(json.dumps(meta))

    legs = merge_mod.find_legs(profile, "fb")
    state = merge_mod._concat_state(legs, [20, 20], 15)
    assert second.name in state["degraded"]
    assert first.name not in state["degraded"]
