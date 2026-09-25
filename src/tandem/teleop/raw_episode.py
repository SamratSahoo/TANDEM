"""Writing one raw episode, from the teleop side.

The teleop driver runs under the DROID environment's interpreter, not tandem's, so this
module deliberately imports nothing from tandem — numpy and the standard library only, plus
a lazy imageio import for the video.

Everything here exists to make a teleop leg **byte-identical in shape** to a TAMP leg, so
`tandem.core.merge` can stream-copy them together and the exporter treats them the same.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# The control rate, the capture rate, and the export FPS are all the same number on purpose.
CONTROL_HZ = 15

# Fixed filenames. The exporter maps them to DROID's image keys and the merge matches legs by
# them, so they are a contract, not a convention.
EXTERNAL_CAM = "external_cam.mp4"  # -> exterior_image_1_left
EXTERNAL_CAM_2 = "external_cam_2.mp4"  # -> exterior_image_2_left
HAND_CAM = "hand_cam.mp4"  # -> wrist_image_left


def emit(events_file, event: str, **fields) -> None:
    """Append one JSON event line to the file tandem tails.

    Silently ignores I/O errors: losing an event costs the UI a state update, while raising
    here would abort a recording that is otherwise fine.
    """
    if not events_file:
        return
    try:
        with open(events_file, "a") as handle:
            handle.write(json.dumps({"event": event, **fields}) + "\n")
            handle.flush()
    except OSError:
        pass


def write_robot_state_npz(
    path,
    *,
    joint_position,
    gripper_position,
    cmd_joint_position,
    cmd_joint_velocity,
    cmd_gripper,
    frame_time,
) -> int:
    """Write the per-frame arrays. Returns the frame count.

    Two invariants are enforced here rather than trusted:

    * ``cmd_gripper`` is forced **binary**. A continuous gripper action is an echo of the
      measured state, and a policy trained on that learns never to close the gripper.
    * ``frame_time`` stays **float64**. As float32, epoch seconds near 1.8e9 have ~128 s of
      resolution, which silently collapses every frame onto one timestamp.
    """
    jp = np.asarray(joint_position, dtype=np.float32).reshape(-1, 7)
    n = len(jp)
    gp = np.asarray(gripper_position, dtype=np.float32).reshape(-1)
    cjp = np.asarray(cmd_joint_position, dtype=np.float32).reshape(-1, 7)
    cjv = np.asarray(cmd_joint_velocity, dtype=np.float32).reshape(-1, 7)
    cg = np.asarray(cmd_gripper, dtype=np.float32).reshape(-1)
    cg = np.where(cg > 0.5, 1.0, 0.0).astype(np.float32)
    ft = np.asarray(frame_time, dtype=np.float64).reshape(-1)

    if not (len(gp) == len(cjp) == len(cjv) == len(cg) == len(ft) == n):
        raise ValueError(
            f"state arrays disagree on length: joint={n} gripper={len(gp)} cmd_joint={len(cjp)} "
            f"cmd_vel={len(cjv)} cmd_gripper={len(cg)} time={len(ft)}"
        )

    np.savez(
        path,
        joint_position=jp,
        gripper_position=gp,
        cmd_joint_position=cjp,
        cmd_joint_velocity=cjv,
        cmd_gripper=cg,
        frame_time=ft,
    )
    return n


def write_meta(
    path,
    *,
    instruction,
    n_frames,
    timestamp,
    cameras,
    record_start,
    record_stop,
    config_id=None,
    trajectory_id=None,
    segment_source="teleop",
    source="teleop",
    phase_index=None,
    n_phases=None,
    phase_description=None,
) -> None:
    """Write ``_meta.json``.

    ``trajectory_id`` / ``segment_source`` mark this episode as one LEG of a hand-off, which
    the merge later joins into a single trajectory. Both are None for a standalone episode.

    ``phase_index`` / ``n_phases`` / ``phase_description`` say which phase of a phase-planned
    task the leg records. Each is written only when given: an absent key means "not known",
    which a null would blur with "known to be nothing".
    """
    meta = {
        "instruction": instruction,
        "fps": CONTROL_HZ,
        "n_frames": int(n_frames),
        "config_id": config_id,
        "timestamp": timestamp,
        "source": source,
        "cameras": cameras,
        "record_start": float(record_start),
        "record_stop": float(record_stop),
        "trajectory_id": trajectory_id,
        "segment_source": segment_source,
    }
    if phase_index is not None:
        meta["phase_index"] = int(phase_index)
    if n_phases is not None:
        meta["n_phases"] = int(n_phases)
    if phase_description is not None:
        meta["phase_description"] = str(phase_description)
    Path(path).write_text(json.dumps(meta))


def write_video(frames, path) -> None:
    """Write HWC-RGB uint8 frames to an mp4.

    The encoder settings (libx264 / yuv420p / crf 20) match what the TAMP side writes, because
    a hand-off is concatenated leg-by-leg with a **stream copy**: the merge refuses legs whose
    codec, size, pixel format or frame rate disagree.

    ``macro_block_size=1`` keeps the captured resolution. imageio's default of 16 silently
    upscales any dimension that is not a multiple of 16, which both changes the recording and
    makes the leg unmergeable against a TAMP leg of the true size.
    """
    import imageio.v2 as imageio

    # imageio only starts ffmpeg on the first frame, so writing an empty sequence leaves NO
    # file and returns quietly — an episode with metadata and state but no video, which the
    # merge then cannot probe. Raise instead.
    if not len(frames):
        raise ValueError(f"refusing to write an empty video: {path}")

    writer = imageio.get_writer(
        str(path),
        fps=CONTROL_HZ,
        codec="libx264",
        pixelformat="yuv420p",
        quality=None,
        output_params=["-crf", "20"],
        macro_block_size=1,
        ffmpeg_log_level="error",
    )
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame))
    finally:
        writer.close()


# The upstream driver imports this name; keep it working.
_write_video = write_video
