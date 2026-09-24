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
disk and re-running once the cause is fixed is safe. Everything that can be refused -- the status,
a destination already taken -- is refused before a leg moves; every copy is made before a leg
moves; and if anything fails once the legs are parked under the scratch directory, each is moved
back where it was before the scratch directory is removed.

Each leg is one phase φ_k of a phase-planned task (or a stretch of one, for conjoined robot
phases). A leg that says which — ``phase_index`` in its ``_meta.json`` — keeps saying so in the
merged ``segments[]``, which is what lets τ = ((τ_1, φ_1), …) be read back off one episode.
"""

from __future__ import annotations

import difflib
import json
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from tandem.core.profiles import STATUSES, Profile
from tandem.core.trajectories import (
    CAMERA_FILES,
    META_FILE,
    STATE_FILE,
    read_hitl,
    refile_record,
    settled_outcome,
)

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

# Per-frame arrays a leg MAY carry. Concatenated like STATE_KEYS, but a leg without one is not
# defective: its absence says when, and by which driver, the leg was captured. TipTop has written
# ``action_joint_velocity`` (the deployable DROID action, see DROID_JV_GAIN) since 1c6daf3; older
# TipTop legs and every teleop leg have nothing there.
#
# The rule, per key, is:
#   * no leg carries it  -> the merged episode does not carry it either. Nothing is invented for
#     a trajectory whose legs never recorded it, so merging legacy legs gives what it always gave;
#   * some or all legs carry it -> the merged episode carries it for EVERY frame. A carrier's
#     array is taken verbatim; a leg without one gets the value derived in that leg's own
#     provenance (_derive_action_joint_velocity).
# The derivation has to happen here, per leg, and not later on the joined array: a hand-off is
# mixed-provenance — a live-teleop leg's cmd_joint_velocity already IS the DROID action, while a
# pre-1c6daf3 TAMP leg's is the cuTAMP plan's feedforward rad/s — so once the legs are one array
# there is no single answer left to give. Dropping the key instead would silently discard what
# the capture wrote, which is the one thing this module refuses to do. (Ported from the monorepo's
# collect/merge_trajectory.py + collect/droid_action.py; the export still reads cmd_joint_velocity.)
OPTIONAL_STATE_KEYS = ("action_joint_velocity",)

# What a merged action_joint_velocity is, stated rather than left for a consumer to infer from
# the source. It is the value TipTop writes into a leg's own _meta.json for the same array.
ACTION_CONVENTION = "droid_joint_velocity"

# 1 / max_joint_delta (0.2 rad) of the DROID FR3's IK controller (droid/robot_ik/robot_ik_solver.py).
# The deploy executor runs a joint-velocity action as joint_delta = jv * max_joint_delta once per
# control step, so the action that reproduces a recorded motion is a TRACKING ERROR in those units:
#     action_joint_velocity = DROID_JV_GAIN * (cmd_joint_position - joint_position)
# Mirrors tiptop.lerobot_capture.DROID_JV_GAIN, which is where a carrier's array comes from.
DROID_JV_GAIN = 5.0

# Keys of a leg's _meta.json that describe that LEG's phase, not the trajectory. The TAMP backend
# stamps them from LegSpec; the teleop driver from --phase-index/--n-phases/--phase-description.
PHASE_KEYS = ("phase_index", "n_phases", "phase_description")


class MergeError(RuntimeError):
    """A trajectory that cannot be merged safely. Never partially written."""


# --------------------------------------------------------------------------- ffmpeg


def _tool(name: str, tools_dir: Path | None) -> str:
    """Prefer the planner runtime's ffmpeg, in ``tools_dir`` -- it is the build that recorded these clips.

    ``tools_dir`` is that runtime's environment's bin directory (``registry.tools_dir``), or None for a
    planner with no environment of its own, and then PATH's ffmpeg is the only one there is.
    """
    if tools_dir is not None:
        candidate = Path(tools_dir) / name
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(name)
    if not found:
        raise MergeError(f"{name} not found; it is needed to join the legs' videos")
    return found


def _probe(path: Path, tools_dir: Path | None) -> dict:
    """Frame count and codec parameters for one mp4. The parameters gate the stream copy."""
    result = subprocess.run(
        [
            _tool("ffprobe", tools_dir), "-v", "error", "-select_streams", "v:0", "-count_packets",
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


def _trim(src: Path, dest: Path, n_frames: int, tools_dir: Path | None) -> None:
    """Stream-copy the first ``n_frames`` of ``src``. Cutting only the tail drops trailing
    packets, so this decodes cleanly even with B-frames left on."""
    subprocess.run(
        [_tool("ffmpeg", tools_dir), "-y", "-loglevel", "error", "-i", str(src),
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


# --------------------------------------------------------------------------- the DROID action


def _derive_action_joint_velocity(leg_arrays: dict, source: str) -> tuple[np.ndarray, str | None]:
    """The DROID action for a leg that did not record one, in that leg's own provenance.

    Returns ``(array, note)``. A note says why the array is not simply correct as derived; the
    merge keeps it in ``_meta.json`` under ``action_notes``, keyed by leg.

    The leg's DECLARED source decides, not a fit of its arrays against the identity. Every leg
    tandem writes says which driver recorded it, while a regression is only as good as the leg's
    variance — an operator who barely moved the arm gives it nothing to go on.

    * Not a 7-joint arm: DROID_JV_GAIN is the FR3's, so the identity says nothing about this
      embodiment. The stored command passes through, under a note.
    * Not a TAMP leg (teleop, or a policy): its cmd_joint_velocity is the command the env
      consumed — for teleop, the IK action — which is ground truth rather than a reconstruction,
      so it passes through untouched.
    * A TAMP leg: its cmd_joint_velocity is the planner's feedforward rad/s, a different quantity
      that under-commands the arm at deploy, so the action is RECOMPUTED from the identity —
      unless the operands cannot support it, in which case the stored array passes through under
      a note saying so:
        - cmd_joint_position is constant: a placeholder, and 5 * (0 - q) is plausible-magnitude
          garbage rather than a crash, the worst way to fail here;
        - cmd_joint_position is a copy of joint_position: nothing commanded a target, and the
          identity would read ~0.
    """
    cmd_jv = np.asarray(leg_arrays["cmd_joint_velocity"], dtype=np.float32)
    cmd_jp = np.asarray(leg_arrays["cmd_joint_position"], dtype=np.float32)
    jp = np.asarray(leg_arrays["joint_position"], dtype=np.float32)

    if jp.ndim != 2 or jp.shape[1] != 7 or cmd_jp.shape != jp.shape:
        return cmd_jv, (
            f"joint_position has shape {jp.shape}, not a 7-joint arm, so the DROID identity does not "
            "apply; action_joint_velocity is the stored cmd_joint_velocity UNCHANGED."
        )
    if source != "tamp":
        return cmd_jv, None
    if len(cmd_jp) and float(np.abs(cmd_jp - cmd_jp[0]).max()) < 1e-6:
        return cmd_jv, (
            "cmd_joint_position is constant (a placeholder), so the DROID action cannot be derived; "
            "action_joint_velocity is the stored cmd_joint_velocity UNCHANGED. Do not train on this "
            "leg's actions."
        )
    if len(cmd_jp) and float(np.abs(cmd_jp - jp).max()) < 1e-6:
        return cmd_jv, (
            "cmd_joint_position is a copy of the measured joint_position (nothing commanded a target), "
            "so the DROID action cannot be derived; action_joint_velocity is the stored "
            "cmd_joint_velocity UNCHANGED."
        )
    return (DROID_JV_GAIN * (cmd_jp - jp)).astype(np.float32), (
        "no action_joint_velocity: this TAMP leg predates the capture that writes it, so its "
        "cmd_joint_velocity is the planner's feedforward rad/s, not the action the deploy executor "
        f"reads. RECOMPUTED as {DROID_JV_GAIN:g} * (cmd_joint_position - joint_position), the value "
        "the capture now writes."
    )


# How a leg that lacks an OPTIONAL_STATE_KEYS array gets one when another leg of the same
# trajectory has it. Every optional key needs an entry: without a rule, the only alternatives are
# dropping the carriers' arrays or leaving a hole in the joined one.
_DERIVE_MISSING = {"action_joint_velocity": _derive_action_joint_velocity}


# --------------------------------------------------------------------------- concatenation


def _leg_video_frames(leg: dict, cameras: list[str], tools_dir: Path | None) -> tuple[int, dict]:
    """The common length of this leg's clips, and the per-camera counts behind it.

    Two ZEDs stop a frame or so apart, so a leg's cameras routinely differ in length.
    ``video_time`` is one array shared by every camera, so those lengths must be equalised or
    each camera drifts by its own offset — and the drift ACCUMULATES across legs. The common
    length is the minimum; what gets cut is trailing camera padding recorded after execution
    ended, which has no state frame behind it.
    """
    counts = {cam: _probe(leg["dir"] / cam, tools_dir)["n_frames"] for cam in cameras}
    if min(counts.values()) <= 0:
        raise MergeError(f"{leg['dir'].name}: a camera clip has no frames ({counts})")
    return min(counts.values()), counts


def _concat_videos(
    legs: list[dict], camera: str, leg_frames: list[int], dest: Path, scratch: Path, tools_dir: Path | None
) -> int:
    inputs: list[Path] = []
    for i, (leg, want) in enumerate(zip(legs, leg_frames, strict=True)):
        src = leg["dir"] / camera
        if _probe(src, tools_dir)["n_frames"] == want:
            inputs.append(src)
        else:
            trimmed = scratch / f"{i:02d}_{camera}"
            _trim(src, trimmed, want, tools_dir)
            inputs.append(trimmed)

    params = {_probe(path, tools_dir)["params"] for path in inputs}
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
        [_tool("ffmpeg", tools_dir), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c", "copy", str(dest)],
        check=True, capture_output=True, text=True,
    )
    return _probe(dest, tools_dir)["n_frames"]


def _concat_state(legs: list[dict], leg_frames: list[int], fps: int) -> dict:
    """Concatenate the legs' state arrays and build the merged ``video_time``.

    ``video_time[i]`` is where frame *i* sits in the CONCATENATED video. It cannot come from
    ``frame_time``: the wall-clock gaps between legs — roughly 14 s of camera teardown per
    hand-off plus however long the human takes — are in ``frame_time`` but not in the video.
    Measured on a three-leg trajectory, using the single linear map instead was off by 726
    camera frames (48 s). Within each leg we do use that map, then offset by the FRAME COUNTS
    of the preceding legs, which is what the joined file actually holds.

    OPTIONAL_STATE_KEYS are joined by the rule stated beside them: absent everywhere stays
    absent, and present anywhere is filled in for every leg that lacks it.
    """
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in STATE_KEYS}
    # Per leg, the optional array it carries or None. Resolved after the loop, because whether a
    # leg's gap must be filled depends on whether ANY leg carries the key.
    optional: dict[str, list[np.ndarray | None]] = {key: [] for key in OPTIONAL_STATE_KEYS}
    video_time: list[np.ndarray] = []
    cumulative = 0
    degraded: list[str] = []

    for leg, n_cam in zip(legs, leg_frames, strict=True):
        with np.load(leg["dir"] / STATE_FILE) as store:
            missing = [key for key in STATE_KEYS if key not in store.files]
            if missing:
                raise MergeError(f"{leg['dir'].name}: {STATE_FILE} is missing {missing}")
            extra = [key for key in store.files if key not in STATE_KEYS and key not in OPTIONAL_STATE_KEYS]
            if extra:
                raise MergeError(
                    f"{leg['dir'].name}: {STATE_FILE} has unexpected arrays {extra}; "
                    "refusing to merge rather than drop them"
                )
            leg_arrays = {key: store[key] for key in STATE_KEYS}
            leg_optional = {key: store[key] if key in store.files else None for key in OPTIONAL_STATE_KEYS}

        frame_time = leg_arrays["frame_time"].astype(np.float64)
        n = len(frame_time)
        if any(len(leg_arrays[key]) != n for key in STATE_KEYS):
            raise MergeError(f"{leg['dir'].name}: {STATE_FILE} arrays disagree on length")
        for key, value in leg_optional.items():
            if value is not None and len(value) != n:
                raise MergeError(f"{leg['dir'].name}: {key} has {len(value)} rows but the leg has {n} frames")
            optional[key].append(value)
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

    notes: dict[str, str] = {}
    for key, per_leg in optional.items():
        if all(value is None for value in per_leg):
            continue
        filled: list[np.ndarray] = []
        for i, (leg, value) in enumerate(zip(legs, per_leg, strict=True)):
            if value is None:
                value, note = _DERIVE_MISSING[key]({k: arrays[k][i] for k in STATE_KEYS}, leg["source"])
                if note:
                    notes[leg["dir"].name] = note
            filled.append(np.asarray(value))
        shapes = sorted({tuple(value.shape[1:]) for value in filled})
        if len(shapes) > 1:
            raise MergeError(f"{key}: legs disagree on its per-frame shape {shapes}; cannot join them")
        merged[key] = np.concatenate(filled, axis=0).astype(np.float32)

    return {
        "arrays": merged,
        "degraded": degraded,
        "action_notes": notes,
        "total_video_frames": cumulative,
        "leg_state_frames": [len(a) for a in arrays["frame_time"]],
    }


# --------------------------------------------------------------------------- merge


def merge(
    profile: Profile,
    trajectory_id: str,
    *,
    status: str | None = None,
    tools_dir: Path | None = None,
    leg_stamps: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Join every leg of ``trajectory_id`` into the first TAMP leg's directory.

    ``status`` is where the merged trajectory is filed (default: where its primary leg already is).
    ``leg_stamps`` adds keys to a leg's stretch of ``segments[]``, by leg directory name: the session
    passes each leg's ``plan_generation``, which the recorders do not stamp.
    """
    if status is not None and status not in STATUSES:
        # Before anything is looked at, let alone moved: `status_dir` refuses it too, but only once
        # the legs were parked, and the clean-up after that refusal deleted them.
        close = difflib.get_close_matches(str(status), STATUSES, n=1)
        raise MergeError(
            f"Unknown status {status!r}; a trajectory is filed under one of {', '.join(STATUSES)}."
            + (f" Did you mean {close[0]!r}?" if close else "")
        )
    stranded = _work_dir(profile, trajectory_id) / "segments"
    if stranded.is_dir() and any(stranded.iterdir()):
        # A merge that failed after parking the legs and could not put them back. They are the only
        # copy of the trial's raw legs, invisible to find_legs down there, and removing the scratch
        # directory -- which every merge used to do first -- would delete them.
        raise MergeError(
            f"{stranded.parent} holds raw legs of trajectory {trajectory_id} from a merge that did not "
            f"finish. Move each directory in {stranded} back into eval/, success/ or failure/, named "
            "without its NN_source_ prefix, delete the empty scratch directory, and merge again."
        )
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
    # Where the merged trajectory will go, settled while every leg is still in place. A destination
    # something else already holds is refused now: moved onto later, it would have put this merge
    # INSIDE it rather than failing.
    dest_status = status or primary["status"]
    dest = profile.status_dir(dest_status) / primary["dir"].name
    if dest.exists() and dest != primary["dir"]:
        raise MergeError(
            f"{dest} already exists, so trajectory {trajectory_id} cannot be merged there.",
        )
    if dest_status == "success":
        _refuse_settled_as_success(primary, trajectory_id)

    # Only cameras every leg recorded can be joined. A camera some legs lack cannot simply be
    # dropped from those legs: the merged clip would be shorter than the others and every
    # video_time past that point would name the wrong frame.
    per_leg = [[c for c in CAMERA_FILES if (leg["dir"] / c).is_file()] for leg in legs]
    cameras = [c for c in CAMERA_FILES if all(c in cams for cams in per_leg)]
    dropped = sorted({c for cams in per_leg for c in cams} - set(cameras))
    if not cameras:
        raise MergeError(f"Trajectory {trajectory_id}: no camera is present in all {len(legs)} legs.")

    fps = int(primary["meta"].get("fps") or 15)
    probed = [_leg_video_frames(leg, cameras, tools_dir) for leg in legs]
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
    work = _work_dir(profile, trajectory_id)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    scratch = work / "_scratch"
    scratch.mkdir()

    try:
        joined = {
            cam: _concat_videos(legs, cam, leg_frames, work / cam, scratch, tools_dir)
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
            segment = {
                "source": leg["source"],
                "timestamp": leg["dir"].name,
                "n_frames": leg_n,
                "n_video_frames": int(n_cam),
                "video_start": cumulative / float(fps),
                "video_stop": (cumulative + n_cam) / float(fps),
                "record_start": leg["meta"].get("record_start"),
                "record_stop": leg["meta"].get("record_stop"),
            }
            # Which phase φ_k this stretch of the episode is (and which driver config recorded
            # it), when the leg recorded that. Only then: a leg from before phase planning, or
            # with it off, has no phase, and inventing one from the leg's position would pair
            # frames with the wrong subgoal.
            for key in ("config_id", *PHASE_KEYS):
                if leg["meta"].get(key) is not None:
                    segment[key] = leg["meta"][key]
            # Which plan that phase belongs to, from whoever ran the trial (`leg_stamps`). A replan
            # numbers its phases from 0 again, so after one, phase_index alone is ambiguous.
            segment.update((leg_stamps or {}).get(leg["dir"].name, {}))
            segments.append(segment)
            cumulative += n_cam

        meta = dict(primary["meta"])
        # The primary leg's phase fields describe that ONE leg; left at the top they would label
        # the whole trajectory as phase k. They live on in segments[]. n_phases is the plan's,
        # so it stays at the top when every leg that states it agrees -- and was stated by ONE
        # plan: after a replan, the legs' n_phases are two different plans' lengths.
        for key in PHASE_KEYS:
            meta.pop(key, None)
        stated = {leg["meta"]["n_phases"] for leg in legs if leg["meta"].get("n_phases") is not None}
        plans = {s.get("plan_generation") for s in segments if s.get("phase_index") is not None}
        if len(stated) == 1 and len(plans) <= 1:
            meta["n_phases"] = stated.pop()
        # The merged action array exists exactly when some leg carried one, and was then resolved
        # into one convention for every frame (see OPTIONAL_STATE_KEYS). Say so unconditionally
        # rather than inherit it: a leg it was built from may have declared no convention at all.
        meta.pop("action_convention", None)
        meta.pop("action_notes", None)
        if "action_joint_velocity" in arrays:
            meta["action_convention"] = ACTION_CONVENTION
            meta["action_notes"] = state["action_notes"]
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

        # The primary leg's non-state artifacts (the backend's plan, perception output, the phase
        # record, …) are surfaced at the top so the merged directory reads like the rollout it grew
        # from — `tandem traj replay` looks for the plan there. Copied from the leg where it still
        # is, BEFORE any leg moves, so a copy that fails (a full disk) fails with every leg in place.
        # Logs stay with the leg that produced them: the primary's is still being written when this
        # runs, so a copy would be truncated. Any `*.log`, whatever the planner calls its own.
        replaced = {STATE_FILE, META_FILE, *CAMERA_FILES}
        for item in sorted(primary["dir"].iterdir()):
            if item.name in replaced or item.suffix == ".log" or (work / item.name).exists():
                continue
            if item.is_dir():
                shutil.copytree(item, work / item.name)
            else:
                shutil.copy2(item, work / item.name)
        shutil.rmtree(scratch)
        segments_dir = work / "segments"
        segments_dir.mkdir()
        dest.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Nothing has moved yet: the legs are exactly where they were.
        shutil.rmtree(work, ignore_errors=True)
        raise

    # Commit: park every raw leg under the merged directory, keeping the sorted order in the name so
    # the hand-off sequence stays readable on disk, then move the whole into place. The only steps
    # left that touch a leg are renames, and each is recorded so it can be undone.
    parked_legs: list[tuple[Path, Path]] = []
    try:
        for i, leg in enumerate(legs):
            parked = segments_dir / f"{i:02d}_{leg['source']}_{leg['dir'].name}"
            shutil.move(str(leg["dir"]), str(parked))
            parked_legs.append((leg["dir"], parked))
        shutil.move(str(work), str(dest))
    except Exception as exc:
        _unpark(parked_legs, work, trajectory_id, exc)
        raise

    # A merge filed under a status of its own is a relabel by another name, and the record it carried
    # up from the primary leg must say where the trial now is -- a trial the session stopped before
    # anybody labeled it (filed_under: null) is filed exactly this way.
    record = read_hitl(dest)
    if status is not None and record and record.get("filed_under") != status:
        refile_record(dest, record, status, settled=settled_outcome(record))

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
        "action_notes": state["action_notes"],
        "frames_trimmed": trimmed,
        "legs_skipped": skipped,
        "segments": segments,
    }


def _work_dir(profile: Profile, trajectory_id: str) -> Path:
    """Where a merge of ``trajectory_id`` is built: a sibling of the status directories, never in one.

    Every directory inside a status dir is listed as a trajectory, and a half-built merge must never
    show up as one.
    """
    return profile.trajectories_dir() / f".merge-{trajectory_id}"


def _unpark(parked_legs: list[tuple[Path, Path]], work: Path, trajectory_id: str, error: Exception) -> None:
    """Put every leg a failed commit parked back where it was, then remove the scratch directory.

    The scratch directory is removed only once every leg is out of it: removing it with a leg still
    inside deletes that leg, which is the raw data the merge exists to keep. A leg that cannot be
    moved back leaves the directory in place, and the error says where the legs are.
    """
    stranded: list[str] = []
    for original, parked in reversed(parked_legs):
        if original.exists() or not parked.exists():
            continue
        try:
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(parked), str(original))
        except OSError as exc:
            stranded.append(f"{parked} ({exc})")
    if stranded:
        raise MergeError(
            f"merging trajectory {trajectory_id} failed ({type(error).__name__}: {error}), and "
            f"{len(stranded)} of its legs could not be moved back: {'; '.join(stranded)}. They are intact "
            f"under {work / 'segments'}; move them back into eval/ by hand before merging again."
        ) from error
    shutil.rmtree(work, ignore_errors=True)


def _refuse_settled_as_success(primary: dict, trajectory_id: str) -> None:
    """Refuse to file as a success a trial whose record says the phase loop settled it otherwise.

    ``tandem traj merge --status success`` is a relabel by another name: an excluded trial -- or one
    the loop ended part-way -- merged there would be exported as a demonstration the method rejected.
    ``tandem traj relabel --force`` is the one deliberate way to overrule that, and it says so in the
    record.
    """
    record = read_hitl(primary["dir"])
    settled = settled_outcome(record)
    if settled is None:
        return
    stage = record.get("failure_stage")
    raise MergeError(
        f"trajectory {trajectory_id} ended {settled}" + (f" at {stage}" if stage else "")
        + f" ({record.get('outcome_reason') or 'see its hitl.json'}), so it cannot be filed as a success. "
        "Merge it without --status, then use `tandem traj relabel <id> success --force` to overrule "
        "that on purpose."
    )
