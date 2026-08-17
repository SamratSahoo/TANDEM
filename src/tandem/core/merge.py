"""Join the legs of a TAMP⇄teleop hand-off into one trajectory.

A task attempt can span any number of legs — ``tamp → teleop → tamp → …`` — and they are
**one trajectory, not N episodes**. Every leg carries the same 16-hex ``trajectory_id`` in
its ``_meta.json``. This module gathers them, concatenates the videos and the per-frame
arrays, and writes the result **into the first TAMP leg's directory**, so the merged
trajectory simply *is* an ordinary trajectory and everything downstream keeps working on it
unchanged. The raw legs are preserved under ``<primary>/segments/<NN>_<source>_<ts>/``.

Leg **order comes from each leg's ``record_start``**, never from a counter: the TAMP driver
and the teleop driver are separate processes and would have to agree on one.

The merge never partially writes. It builds in a scratch directory beside
``eval/success/failure`` and moves it into place, so a failure leaves the legs untouched on
disk and re-running once the cause is fixed is safe.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from tandem.core.profiles import STATUSES, Profile
from tandem.core.trajectories import CAMERA_FILES, META_FILE, STATE_FILE

# Per-frame arrays that concatenate along axis 0. Anything else in the npz is refused rather
# than silently dropped — an unexpected array is a schema change, and quietly losing it would
# leave a dataset that looks fine and is not.
STATE_KEYS = (
    "joint_position",
    "gripper_position",
    "cmd_joint_position",
    "cmd_joint_velocity",
    "cmd_gripper",
    "frame_time",
)


class MergeError(RuntimeError):
    """A trajectory that cannot be merged safely. Never partially written."""


# --------------------------------------------------------------------------- ffmpeg


def _tool(name: str, runtime_dir: Path | None) -> str:
    """Prefer the runtime's ffmpeg — it is the build that recorded these clips."""
    if runtime_dir is not None:
        candidate = runtime_dir / "tiptop" / ".pixi" / "envs" / "default" / "bin" / name
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(name)
    if not found:
        raise MergeError(f"{name} not found; it is needed to join the legs' videos")
    return found


def _probe(path: Path, runtime_dir: Path | None) -> dict:
    """Frame count and codec parameters for one mp4. The parameters gate the stream copy."""
    result = subprocess.run(
        [
            _tool("ffprobe", runtime_dir), "-v", "error", "-select_streams", "v:0", "-count_packets",
            "-show_entries", "stream=nb_read_packets,codec_name,width,height,pix_fmt,r_frame_rate",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    stream = (json.loads(result.stdout).get("streams") or [{}])[0]
    return {
        "n_frames": int(stream.get("nb_read_packets") or 0),
        "params": (
            stream.get("codec_name"), stream.get("width"), stream.get("height"),
            stream.get("pix_fmt"), stream.get("r_frame_rate"),
        ),
    }


def _trim(src: Path, dest: Path, n_frames: int, runtime_dir: Path | None) -> None:
    """Stream-copy the first ``n_frames`` of ``src``. Cutting only the tail drops trailing
    packets, so this decodes cleanly even with B-frames left on."""
    subprocess.run(
        [_tool("ffmpeg", runtime_dir), "-y", "-loglevel", "error", "-i", str(src),
         "-frames:v", str(n_frames), "-c", "copy", str(dest)],
        check=True, capture_output=True, text=True,
    )


# --------------------------------------------------------------------------- discovery


def find_legs(profile: Profile, trajectory_id: str) -> list[dict]:
    """Every leg of ``trajectory_id``, ordered by camera ``record_start``.

    Legs a previous merge already folded in live one level deeper
    (``<trajectory>/segments/<NN>_…``) and are therefore never picked up again — re-merging
    them would double the episode.
    """
    legs: list[dict] = []
    for status in STATUSES:
        status_dir = profile.trajectories_dir() / status
        if not status_dir.is_dir():
            continue
        for entry in sorted(status_dir.iterdir()):
            meta_path = entry / META_FILE
            if not entry.is_dir() or not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except (ValueError, OSError):
                continue
            if meta.get("trajectory_id") != trajectory_id:
                continue
            legs.append({
                "dir": entry,
                "meta": meta,
                "status": status,
                "source": meta.get("segment_source") or "tamp",
            })

    def sort_key(leg: dict):
        start = leg["meta"].get("record_start")
        # Fall back to the directory name when a leg has no recording window; every leg dir is
        # named with a second-resolution wall-clock stamp, so it still orders correctly.
        return (float(start) if isinstance(start, (int, float)) else float("inf"), leg["dir"].name)

    return sorted(legs, key=sort_key)


def pending_trajectory_ids(profile: Profile) -> list[str]:
    """Trajectory ids with more than one unmerged leg — what a merge would act on."""
    counts: dict[str, int] = {}
    for status in STATUSES:
        status_dir = profile.trajectories_dir() / status
        if not status_dir.is_dir():
            continue
        for entry in sorted(status_dir.iterdir()):
            meta_path = entry / META_FILE
            if not entry.is_dir() or not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except (ValueError, OSError):
                continue
            tid = meta.get("trajectory_id")
            if tid and not meta.get("video_aligned"):
                counts[tid] = counts.get(tid, 0) + 1
    return sorted(tid for tid, count in counts.items() if count > 1)


# --------------------------------------------------------------------------- concatenation


def _leg_video_frames(leg: dict, cameras: list[str], runtime_dir: Path | None) -> tuple[int, dict]:
    """The common length of this leg's clips, and the per-camera counts behind it.

    Two ZEDs stop a frame or so apart, so a leg's cameras routinely differ in length.
    ``video_time`` is one array shared by every camera, so those lengths must be equalised or
    each camera drifts by its own offset — and the drift ACCUMULATES across legs. The common
    length is the minimum; what gets cut is trailing camera padding recorded after execution
    ended, which has no state frame behind it.
    """
    counts = {cam: _probe(leg["dir"] / cam, runtime_dir)["n_frames"] for cam in cameras}
    if min(counts.values()) <= 0:
        raise MergeError(f"{leg['dir'].name}: a camera clip has no frames ({counts})")
    return min(counts.values()), counts


def _concat_videos(
    legs: list[dict], camera: str, leg_frames: list[int], dest: Path, scratch: Path, runtime_dir: Path | None
) -> int:
    inputs: list[Path] = []
    for i, (leg, want) in enumerate(zip(legs, leg_frames, strict=True)):
        src = leg["dir"] / camera
        if _probe(src, runtime_dir)["n_frames"] == want:
            inputs.append(src)
        else:
            trimmed = scratch / f"{i:02d}_{camera}"
            _trim(src, trimmed, want, runtime_dir)
            inputs.append(trimmed)

    params = {_probe(path, runtime_dir)["params"] for path in inputs}
    if len(params) > 1:
        # -c copy cannot join streams that disagree, and silently re-encoding here would be a
        # surprise: minutes of CPU and a quality loss nobody asked for.
        raise MergeError(
            f"{camera}: legs were recorded with different codec parameters {sorted(params)}; "
            "cannot concatenate without re-encoding"
        )

    listing = scratch / f"{camera}.concat.txt"
    quote = "'" + "\\'" + "'"
    listing.write_text("".join(f"file '{str(p).replace(chr(39), quote)}'\n" for p in inputs))
    subprocess.run(
        [_tool("ffmpeg", runtime_dir), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c", "copy", str(dest)],
        check=True, capture_output=True, text=True,
    )
    return _probe(dest, runtime_dir)["n_frames"]


def _concat_state(legs: list[dict], leg_frames: list[int], fps: int) -> dict:
    """Concatenate the legs' state arrays and build the merged ``video_time``.

    ``video_time[i]`` is where frame *i* sits in the CONCATENATED video. It cannot come from
    ``frame_time``: the wall-clock gaps between legs — roughly 14 s of camera teardown per
    hand-off plus however long the human takes — are in ``frame_time`` but not in the video.
    Measured on a three-leg trajectory, using the single linear map instead was off by 726
    camera frames (48 s). Within each leg we do use that map, then offset by the FRAME COUNTS
    of the preceding legs, which is what the joined file actually holds.
    """
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in STATE_KEYS}
    video_time: list[np.ndarray] = []
    cumulative = 0
    degraded: list[str] = []

    for leg, n_cam in zip(legs, leg_frames, strict=True):
        with np.load(leg["dir"] / STATE_FILE) as store:
            missing = [key for key in STATE_KEYS if key not in store.files]
            if missing:
                raise MergeError(f"{leg['dir'].name}: {STATE_FILE} is missing {missing}")
            extra = [key for key in store.files if key not in STATE_KEYS]
            if extra:
                raise MergeError(
                    f"{leg['dir'].name}: {STATE_FILE} has unexpected arrays {extra}; "
                    "refusing to merge rather than drop them"
                )
            leg_arrays = {key: store[key] for key in STATE_KEYS}

        frame_time = leg_arrays["frame_time"].astype(np.float64)
        n = len(frame_time)
        if any(len(leg_arrays[key]) != n for key in STATE_KEYS):
            raise MergeError(f"{leg['dir'].name}: {STATE_FILE} arrays disagree on length")
        for key in STATE_KEYS:
            arrays[key].append(leg_arrays[key])

        start, stop = leg["meta"].get("record_start"), leg["meta"].get("record_stop")
        usable = (
            isinstance(start, (int, float)) and isinstance(stop, (int, float))
            and stop > start and n_cam > 0
        )
        if usable:
            eff_fps = n_cam / (float(stop) - float(start))
            within = (frame_time - float(start)) * eff_fps
        else:
            # Recorded, never silent: the same degradation the exporter falls back to.
            degraded.append(leg["dir"].name)
            within = np.linspace(0.0, max(n_cam - 1, 0), n) if n > 1 else np.zeros(n)
        within = np.clip(within, 0.0, max(n_cam - 1, 0))
        video_time.append((cumulative + within) / float(fps))
        cumulative += n_cam

    merged = {key: np.concatenate(arrays[key], axis=0) for key in STATE_KEYS}
    merged["frame_time"] = merged["frame_time"].astype(np.float64)  # epoch seconds stay f64
    merged["video_time"] = np.concatenate(video_time).astype(np.float64)
    return {
        "arrays": merged,
        "degraded": degraded,
        "total_video_frames": cumulative,
        "leg_state_frames": [len(a) for a in arrays["frame_time"]],
    }


# --------------------------------------------------------------------------- merge


def merge(
    profile: Profile,
    trajectory_id: str,
    *,
    status: str | None = None,
    runtime_dir: Path | None = None,
) -> dict[str, Any]:
    """Join every leg of ``trajectory_id`` into the first TAMP leg's directory."""
    legs = find_legs(profile, trajectory_id)
    if not legs:
        raise MergeError(f"No legs found for trajectory {trajectory_id}.")

    # A merged trajectory keeps its id, so re-running finds the merged trajectory itself. Say
    # that plainly rather than let it look like a one-leg trajectory.
    already = [leg for leg in legs if leg["meta"].get("video_aligned")]
    if already:
        return {
            "merged": False, "reason": "already merged", "trajectory_id": trajectory_id,
            "dir": str(already[0]["dir"]), "n_legs": len(already[0]["meta"].get("segments") or []),
        }
    if len(legs) == 1:
        return {
            "merged": False, "reason": "single leg", "trajectory_id": trajectory_id,
            "dir": str(legs[0]["dir"]), "n_legs": 1,
        }

    # A leg that captured no state contributed nothing. Skip it and say so, rather than fail
    # the whole merge — one bad leg out of five is not a reason to leave five loose episodes.
    skipped = [leg["dir"].name for leg in legs if not (leg["dir"] / STATE_FILE).is_file()]
    legs = [leg for leg in legs if (leg["dir"] / STATE_FILE).is_file()]
    if len(legs) < 2:
        return {
            "merged": False, "reason": "fewer than two legs have state data",
            "trajectory_id": trajectory_id, "n_legs": len(legs), "legs_skipped": skipped,
        }

    primary = next((leg for leg in legs if leg["source"] == "tamp"), legs[0])
    if (primary["dir"] / "segments").exists():
        raise MergeError(
            f"{primary['dir']} already has a segments/ directory, so trajectory {trajectory_id} "
            "looks merged already. Refusing to merge twice."
        )

    # Only cameras every leg recorded can be joined. A camera some legs lack cannot simply be
    # dropped from those legs: the merged clip would be shorter than the others and every
    # video_time past that point would name the wrong frame.
    per_leg = [[c for c in CAMERA_FILES if (leg["dir"] / c).is_file()] for leg in legs]
    cameras = [c for c in CAMERA_FILES if all(c in cams for cams in per_leg)]
    dropped = sorted({c for cams in per_leg for c in cams} - set(cameras))
    if not cameras:
        raise MergeError(f"Trajectory {trajectory_id}: no camera is present in all {len(legs)} legs.")

    fps = int(primary["meta"].get("fps") or 15)
    probed = [_leg_video_frames(leg, cameras, runtime_dir) for leg in legs]
    leg_frames = [n for n, _ in probed]
    trimmed = {
        leg["dir"].name: {cam: count - n for cam, count in counts.items() if count != n}
        for leg, n, (_, counts) in zip(legs, leg_frames, probed, strict=True)
        if any(count != n for count in counts.values())
    }
    state = _concat_state(legs, leg_frames, fps)
    arrays = state["arrays"]

    # Build in a scratch directory that is a SIBLING of eval/success/failure, not inside one:
    # every directory inside a status dir is listed as a trajectory, and a half-built merge
    # must never show up as one.
    work = profile.trajectories_dir() / f".merge-{trajectory_id}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    scratch = work / "_scratch"
    scratch.mkdir()

    try:
        joined = {
            cam: _concat_videos(legs, cam, leg_frames, work / cam, scratch, runtime_dir)
            for cam in cameras
        }
        expected = sum(leg_frames)
        for cam, got in joined.items():
            if got != expected:
                raise MergeError(
                    f"{cam}: the joined clip has {got} frames but the legs sum to {expected}; "
                    "video_time would be misaligned"
                )
        np.savez(work / STATE_FILE, **arrays)

        n_frames = int(len(arrays["frame_time"]))
        segments = []
        cumulative = 0
        for leg, n_cam, leg_n in zip(legs, leg_frames, state["leg_state_frames"], strict=True):
            segments.append({
                "source": leg["source"],
                "timestamp": leg["dir"].name,
                "n_frames": leg_n,
                "n_video_frames": int(n_cam),
                "video_start": cumulative / float(fps),
                "video_stop": (cumulative + n_cam) / float(fps),
                "record_start": leg["meta"].get("record_start"),
                "record_stop": leg["meta"].get("record_stop"),
            })
            cumulative += n_cam

        meta = dict(primary["meta"])
        meta.update({
            "n_frames": n_frames,
            "fps": fps,
            "source": "trajectory",
            "trajectory_id": trajectory_id,
            "segment_source": None,
            "cameras": {
                key: value for key, value in (primary["meta"].get("cameras") or {}).items()
                if value in cameras
            },
            "record_start": segments[0]["record_start"],
            "record_stop": segments[-1]["record_stop"],
            # The merged clip is contiguous while frame_time is not, so consumers must align
            # on video_time; the recording window is kept for provenance only.
            "video_aligned": True,
            "total_video_frames": int(state["total_video_frames"]),
            "segments": segments,
            "cameras_dropped": dropped,
            "proportional_fallback_legs": state["degraded"],
            "frames_trimmed": trimmed,
            "legs_skipped": skipped,
        })
        if n_frames != int(len(arrays["joint_position"])):
            raise MergeError("The merged frame count disagrees with the concatenated arrays.")
        (work / META_FILE).write_text(json.dumps(meta, indent=2))

        # Commit. Park every raw leg under the merged directory, keeping the sorted order in
        # the name so the hand-off sequence stays readable on disk.
        shutil.rmtree(scratch)
        segments_dir = work / "segments"
        segments_dir.mkdir()
        primary_parked = None
        for i, leg in enumerate(legs):
            parked = segments_dir / f"{i:02d}_{leg['source']}_{leg['dir'].name}"
            shutil.move(str(leg["dir"]), str(parked))
            if leg is primary:
                primary_parked = parked

        # The primary leg's non-state artifacts moved down with it. Surface them at the top so
        # the merged directory is a complete trajectory — tiptop_plan.json in particular is
        # what makes a rollout count as collected. Logs stay with the leg that produced them:
        # the primary's is still being written when this runs, so a copy would be truncated.
        replaced = {STATE_FILE, META_FILE, "tiptop_run.log", "postprocess.log", *CAMERA_FILES}
        if primary_parked is not None:
            for item in sorted(primary_parked.iterdir()):
                if item.name in replaced or (work / item.name).exists():
                    continue
                if item.is_dir():
                    shutil.copytree(item, work / item.name)
                else:
                    shutil.copy2(item, work / item.name)

        dest_status = status or primary["status"]
        dest = profile.status_dir(dest_status) / primary["dir"].name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(work), str(dest))
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise

    return {
        "merged": True,
        "trajectory_id": trajectory_id,
        "dir": str(dest),
        "status": dest_status,
        "n_legs": len(legs),
        "n_frames": n_frames,
        "cameras": cameras,
        "cameras_dropped": dropped,
        "proportional_fallback_legs": state["degraded"],
        "frames_trimmed": trimmed,
        "legs_skipped": skipped,
        "segments": segments,
    }
