"""Per-frame series for one trajectory's plots.

Everything the charts need in one JSON payload: the measured and commanded channels
downsampled for the browser, the timeline in three flavours (so the videos can be seeked
from a chart click), the hand-off segment boundaries, and which frames the DROID non-idle
filter would drop at training time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tandem.core.errors import TandemError
from tandem.core.nonidle import series_filter
from tandem.core.trajectories import DEFAULT_FPS, STATE_FILE, Trajectory, read_meta

# More points than this and the SVG charts stop being interactive without telling you
# anything new; the non-idle filter still runs at full resolution.
MAX_POINTS = 2000

CHANNELS = (
    "joint_position",
    "gripper_position",
    "cmd_joint_position",
    "cmd_joint_velocity",
    "cmd_gripper",
)


def _downsample(n: int, cap: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    return np.linspace(0, n - 1, cap).round().astype(int)


def series_for(traj_dir: Path) -> dict:
    npz_path = traj_dir / STATE_FILE
    if not npz_path.is_file():
        raise TandemError(
            f"No {STATE_FILE} in {traj_dir.name}.",
            hint="This rollout recorded no state — a planning failure writes no episode.",
        )

    with np.load(npz_path) as store:
        data = {key: np.asarray(store[key]) for key in store.files}

    meta = read_meta(traj_dir)

    ordered = [*CHANNELS, "frame_time"]
    n = next((len(data[key]) for key in ordered if key in data), 0)
    if n == 0:
        raise TandemError(f"{npz_path} has no per-frame arrays.")

    # frame_time is wall-clock epoch seconds. Stored as float32 its sub-second deltas near
    # 1.7e9 collapse to zero, so fall back to a uniform grid when the span is degenerate.
    # t0 is the absolute epoch of frame 0: the client adds it back to a row's relative time,
    # then subtracts record_start to seek the video.
    t = None
    t0 = None
    if "frame_time" in data and len(data["frame_time"]) == n:
        frame_time = data["frame_time"].astype(np.float64)
        if frame_time[-1] - frame_time[0] > 1e-6:
            t = frame_time - frame_time[0]
            t0 = float(frame_time[0])
    if t is None:
        fps = int(meta.get("fps") or DEFAULT_FPS) or DEFAULT_FPS
        t = np.arange(n, dtype=np.float64) / fps

    sel = _downsample(n, MAX_POINTS)

    def take(key: str):
        array = data.get(key)
        if array is None or len(array) != n:
            return None
        return array[sel]

    # A merged hand-off trajectory carries video_time: each frame's position in the
    # CONCATENATED clip. Its frame_time spans the wall-clock gaps between legs -- ~14 s of
    # camera teardown per hand-off plus however long the human took -- which the video does
    # not contain, so record_start is useless for seeking it.
    video_time = take("video_time")
    segments = meta.get("segments") if meta.get("video_aligned") else None

    # Run the filter on the FULL-resolution command; the downsampled copy would blur runs the
    # filter measures in single frames.
    cmd_jv_full = data.get("cmd_joint_velocity")
    filt = None
    if cmd_jv_full is not None and n >= 2 and cmd_jv_full.shape == (n, 7):
        filt = series_filter(cmd_jv_full, t, sel)

    return {
        "n_frames": n,
        "n_plotted": int(len(sel)),
        "downsampled": bool(len(sel) < n),
        "filter": filt,
        "t": t[sel].tolist(),
        "t0": t0,
        "record_start": _num(meta.get("record_start")),
        "record_stop": _num(meta.get("record_stop")),
        "video_time": _tolist(video_time),
        "segments": segments if isinstance(segments, list) else None,
        "instruction": meta.get("instruction") or "",
        "joint_position": _tolist(take("joint_position")),
        "gripper_position": _tolist(take("gripper_position")),
        "cmd_joint_position": _tolist(take("cmd_joint_position")),
        "cmd_joint_velocity": _tolist(take("cmd_joint_velocity")),
        "cmd_gripper": _tolist(take("cmd_gripper")),
    }


def summary(traj: Trajectory) -> dict:
    """Cheap headline numbers for a terminal `traj show`, without shipping every sample."""
    npz_path = traj.path / STATE_FILE
    if not npz_path.is_file():
        return {}
    with np.load(npz_path) as store:
        data = {key: np.asarray(store[key]) for key in store.files}

    out: dict[str, object] = {"channels": sorted(data)}

    jp = data.get("joint_position")
    if jp is not None and jp.ndim == 2:
        travel = float(np.abs(np.diff(jp, axis=0)).sum()) if len(jp) > 1 else 0.0
        out["joint_travel_rad"] = travel
        out["joint_range_rad"] = [float(v) for v in (jp.max(axis=0) - jp.min(axis=0))]

    grip = data.get("gripper_position")
    if grip is not None and grip.size:
        out["gripper_min"] = float(grip.min())
        out["gripper_max"] = float(grip.max())

    cmd_grip = data.get("cmd_gripper")
    if cmd_grip is not None and cmd_grip.size > 1:
        # Each 0->1 or 1->0 transition of the binary command is a grasp or a release.
        out["gripper_events"] = int(np.count_nonzero(np.diff(cmd_grip.astype(np.int8))))

    cmd_jv = data.get("cmd_joint_velocity")
    if cmd_jv is not None and cmd_jv.ndim == 2 and cmd_jv.shape[1] == 7 and len(cmd_jv) >= 2:
        from tandem.core.nonidle import analyze, clip_action

        clipped, frac = clip_action(cmd_jv)
        ranges, reasons = analyze(clipped)
        from tandem.core.nonidle import KEEP

        out["nonidle_kept"] = int(np.count_nonzero(reasons == KEEP))
        out["nonidle_ranges"] = len(ranges)
        out["frac_clipped"] = frac
        out["peak_cmd_velocity"] = float(np.abs(cmd_jv).max())

    return out


def _tolist(array):
    return None if array is None else array.tolist()


def _num(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
