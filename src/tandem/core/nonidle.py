"""The DROID non-idle frame filter.

pi0.5-droid training does not sample every frame of an episode: the streaming loader reads a
keep-ranges list and only draws anchors from inside it. Everything else is idle or near-idle
and is thrown away. Since `tandem export lerobot` writes episodes in exactly that DROID
layout, the same filter will be applied to them — so the review UI shows which frames
actually survive to training, which is usually the first surprising thing about a rollout.

``compute_keep_ranges`` is a verbatim port of openpi's function of the same name. It is the
source of truth; do not "improve" it — a divergence here silently misreports what training
sees.

Two details matter for fidelity:

* The filter runs on ``action.joint_velocity`` as the DATASET stores it, i.e. on
  ``clip(cmd_joint_velocity, -1, 1)``. Saturation creates idle frames: 1.3 and 1.5 rad/s
  both become 1.0, so their difference is 0. Filtering the raw command reports a different —
  wrong — set of dropped frames.
* It must run at FULL resolution. ``min_idle_len`` is 7, so a 7-frame idle run in a long
  episode survives plot downsampling as ~2 samples and no longer crosses the threshold.
"""

from __future__ import annotations

import numpy as np

# openpi's defaults.
MIN_IDLE_LEN = 7
MIN_NON_IDLE_LEN = 16
FILTER_LAST_N_IN_RANGES = 10

# Per-frame verdicts: KEEP is sampled during training, the rest are the three ways to be dropped.
KEEP = 0
IDLE = 1  # inside an idle run of >= min_idle_len frames
SHORT = 2  # non-idle, but its contiguous run is shorter than min_non_idle_len
TRIM = 3  # inside a kept run, but among its last filter_last_n_in_ranges frames
REASONS = {KEEP: "keep", IDLE: "idle", SHORT: "short", TRIM: "trim"}


def clip_action(cmd_jv: np.ndarray) -> tuple[np.ndarray, float]:
    """``cmd_joint_velocity`` as the dataset stores it, plus the fraction sitting at the rail.

    A high clipped fraction means the plotted line and the filter shading diverge visibly,
    which is worth saying out loud rather than leaving as a mystery.
    """
    jv = np.asarray(cmd_jv, dtype=np.float32).reshape(-1, 7)
    frac_clipped = float(np.mean(np.abs(jv) > 1.0)) if jv.size else 0.0
    return np.clip(jv, -1.0, 1.0), frac_clipped


def compute_keep_ranges(
    joint_velocities: np.ndarray,
    *,
    min_idle_len: int = MIN_IDLE_LEN,
    min_non_idle_len: int = MIN_NON_IDLE_LEN,
    filter_last_n_in_ranges: int = FILTER_LAST_N_IN_RANGES,
) -> list[tuple[int, int]]:
    """Non-idle keep-ranges for one episode. Verbatim from openpi."""
    if joint_velocities.ndim != 2 or joint_velocities.shape[0] == 0:
        return []
    num_frames = joint_velocities.shape[0]

    is_idle_array = np.hstack(
        [np.array([False]), np.all(np.abs(joint_velocities[1:] - joint_velocities[:-1]) < 1e-3, axis=1)]
    )

    # Start and end with False so idle at the first step counts as a start of motion.
    is_idle_padded = np.concatenate([[False], is_idle_array, [False]])
    is_idle_diff = np.diff(is_idle_padded.astype(int))
    idle_starts = np.where(is_idle_diff == 1)[0]
    idle_ends = np.where(is_idle_diff == -1)[0]

    long_idle = (idle_ends - idle_starts) >= min_idle_len
    idle_starts, idle_ends = idle_starts[long_idle], idle_ends[long_idle]

    keep_mask = np.ones(num_frames, dtype=bool)
    for start, end in zip(idle_starts, idle_ends, strict=True):
        keep_mask[start:end] = False

    keep_padded = np.concatenate([[False], keep_mask, [False]])
    keep_diff = np.diff(keep_padded.astype(int))
    run_starts = np.where(keep_diff == 1)[0]
    run_ends = np.where(keep_diff == -1)[0]

    long_run = (run_ends - run_starts) >= min_non_idle_len
    run_starts, run_ends = run_starts[long_run], run_ends[long_run]

    return [(int(s), int(e) - filter_last_n_in_ranges) for s, e in zip(run_starts, run_ends, strict=True)]


def _true_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Half-open [start, end) index pairs of every True run in a boolean mask."""
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(int))
    return np.where(diff == 1)[0], np.where(diff == -1)[0]


def analyze(
    joint_velocities: np.ndarray,
    *,
    min_idle_len: int = MIN_IDLE_LEN,
    min_non_idle_len: int = MIN_NON_IDLE_LEN,
    filter_last_n_in_ranges: int = FILTER_LAST_N_IN_RANGES,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """The same ranges as :func:`compute_keep_ranges`, plus a per-frame reason label.

    ``reasons == KEEP`` is exactly the union of the returned ranges.
    """
    if joint_velocities.ndim != 2 or joint_velocities.shape[0] == 0:
        return [], np.zeros(0, dtype=np.int8)
    num_frames = joint_velocities.shape[0]
    reasons = np.full(num_frames, KEEP, dtype=np.int8)

    is_idle_array = np.hstack(
        [np.array([False]), np.all(np.abs(joint_velocities[1:] - joint_velocities[:-1]) < 1e-3, axis=1)]
    )
    idle_starts, idle_ends = _true_runs(is_idle_array)
    long_idle = (idle_ends - idle_starts) >= min_idle_len

    keep_mask = np.ones(num_frames, dtype=bool)
    for start, end in zip(idle_starts[long_idle], idle_ends[long_idle], strict=True):
        keep_mask[start:end] = False
        reasons[start:end] = IDLE

    run_starts, run_ends = _true_runs(keep_mask)
    long_run = (run_ends - run_starts) >= min_non_idle_len
    for start, end in zip(run_starts[~long_run], run_ends[~long_run], strict=True):
        reasons[start:end] = SHORT

    ranges: list[tuple[int, int]] = []
    for start, end in zip(run_starts[long_run], run_ends[long_run], strict=True):
        # The trailing frames of a kept run are anchors whose action chunk is mostly idle.
        # max() only guards a caller who lowers min_non_idle_len below the trim length.
        reasons[max(int(start), int(end) - filter_last_n_in_ranges) : end] = TRIM
        ranges.append((int(start), int(end) - filter_last_n_in_ranges))
    return ranges, reasons


def series_filter(cmd_jv: np.ndarray, t: np.ndarray, sel: np.ndarray) -> dict:
    """The ``filter`` block of a series payload.

    ``cmd_jv`` is the FULL-resolution raw command [N,7], ``t`` the full-resolution relative
    timeline [N], and ``sel`` the indices the plotted series was downsampled to.

    ``drop_spans`` are in the same relative seconds as ``t``, so a chart can shade them
    knowing nothing about downsampling; ``reason``/``keep`` are per PLOTTED frame.
    """
    jv, frac_clipped = clip_action(cmd_jv)
    num_frames = jv.shape[0]
    ranges, reasons = analyze(jv)

    t = np.asarray(t, dtype=np.float64).reshape(-1)
    dt = float(np.median(np.diff(t))) if num_frames > 1 else 0.0

    def edge(i: int) -> float:
        # A span covering [a, b) is shaded to the NEXT frame's time so consecutive spans tile
        # with no seam; past the last frame, extend by one median timestep.
        return float(t[i]) if i < num_frames else float(t[num_frames - 1]) + dt

    drop_spans = []
    if num_frames:
        boundaries = np.nonzero(np.diff(reasons))[0] + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [num_frames]])
        for start, end in zip(starts, ends, strict=True):
            code = int(reasons[start])
            if code == KEEP:
                continue
            drop_spans.append(
                {"t0": edge(int(start)), "t1": edge(int(end)), "reason": REASONS[code], "n": int(end - start)}
            )

    return {
        "keep_ranges": [[start, end] for start, end in ranges],
        "drop_spans": drop_spans,
        "keep": (reasons[sel] == KEEP).tolist(),
        "reason": [REASONS[int(c)] for c in reasons[sel]],
        "n_frames": num_frames,
        "n_kept": int(np.count_nonzero(reasons == KEEP)),
        "n_dropped": {
            "idle": int(np.count_nonzero(reasons == IDLE)),
            "short": int(np.count_nonzero(reasons == SHORT)),
            "trim": int(np.count_nonzero(reasons == TRIM)),
        },
        "frac_clipped": frac_clipped,
        "params": {
            "min_idle_len": MIN_IDLE_LEN,
            "min_non_idle_len": MIN_NON_IDLE_LEN,
            "filter_last_n_in_ranges": FILTER_LAST_N_IN_RANGES,
        },
    }
