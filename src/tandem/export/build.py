"""Build a LeRobot v3.0 dataset from a profile's successful trajectories.

Matches ``lerobot/droid_1.0.1``'s schema so the result feeds a π₀.₅-DROID finetune directly.
Pure Python — av, pyarrow, numpy — so this runs on a laptop with no GPU and no runtime.

The one thing that is easy to get silently wrong is the action scale. Both sources already
store ``cmd_joint_velocity`` on DROID's normalized [-1, 1] scale, so it is **clipped, never
rescaled**:

  * teleop — the IK command captured from the env's action dict, natively in [-1, 1]
  * tamp   — the cuRobo plan velocity in rad/s, which measurement shows already sits at
             ~DROID scale (action/measured ≈ 1.3 against DROID's ≈ 1.44)

Dividing it by 3 "to convert units" deploys about three times too slow.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import NamedTuple

import numpy as np

from tandem import __version__
from tandem.core import trajectories as traj_mod
from tandem.core.errors import TandemError
from tandem.core.profiles import Profile
from tandem.export.lerobot_v3 import VIDEO_KEY_MAP, V3DatasetWriter

log = logging.getLogger("tandem.export")

FPS = 15
IMG_HW = (180, 320)  # DROID LeRobot image size (H, W)
TAGS = ["droid", "panda", "real", "tamp", "tandem"]

# Written into every dataset this export builds, and kept out of what is pushed. A build only ever
# replaces a directory that has one: `--out` pointed one level too high, or a repo name that matches a
# directory of somebody's own, used to be deleted without a word before a single episode was looked at.
MARKER = ".tandem-export.json"

# Our camera filenames -> the LeRobot common image key.
REAL_CAMERAS = {
    "exterior_image_1_left": "external_cam.mp4",
    "exterior_image_2_left": "external_cam_2.mp4",
    "wrist_image_left": "hand_cam.mp4",
}


def clip_joint_velocity(cmd_jv: np.ndarray) -> tuple[np.ndarray, float]:
    frac_clipped = float(np.mean(np.abs(cmd_jv) > 1.0)) if cmd_jv.size else 0.0
    return np.clip(cmd_jv, -1.0, 1.0).astype(np.float32), frac_clipped


def _resample_indices(n_src: int, n_dst: int) -> np.ndarray:
    if n_dst <= 1 or n_src <= 1:
        return np.zeros(max(n_dst, 0), dtype=int)
    return np.clip(np.round(np.arange(n_dst) * (n_src - 1) / (n_dst - 1)), 0, n_src - 1).astype(int)


def _camera_indices(frame_time: np.ndarray, n_cam: int, record_start: float, record_stop: float) -> np.ndarray:
    """Camera-frame index per state frame, aligned by wall clock.

    Cameras start before execution and stop after it, so the recording window brackets the
    state window. Mapping each state frame's timestamp onto the video keeps images in step
    with actions; stretching the state timeline across the whole clip does not.
    """
    eff_fps = n_cam / (record_stop - record_start)
    idx = np.round((frame_time - record_start) * eff_fps)
    return np.clip(idx, 0, n_cam - 1).astype(int)


def _video_time_indices(video_time: np.ndarray, n_cam: int, video_seconds: float) -> np.ndarray:
    """Camera index per state frame for a MERGED hand-off trajectory.

    ``video_time`` already places each frame on the concatenated clip's timeline, so only the
    seconds→frames scaling is left. The wall-clock map above is meaningless here: the gaps
    between legs (camera teardown, the human working) exist in frame_time but not in the video.
    """
    idx = np.round(video_time * (n_cam / video_seconds))
    return np.clip(idx, 0, n_cam - 1).astype(int)


def _finite(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _decode_resized(path: Path, hw: tuple[int, int]) -> list[np.ndarray]:
    """Decode an mp4 to RGB frames at the target size.

    Scaling happens inside PyAV's reformatter rather than in OpenCV, which keeps the export's
    dependencies down to av + pyarrow + huggingface_hub.
    """
    import av

    height, width = hw
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.reformat(width=width, height=height, format="rgb24").to_ndarray())
    return frames


def _decode_cameras(traj_dir: Path) -> dict[str, list[np.ndarray]] | None:
    """Decoded {common_key -> frames}, or None when a required camera is unusable.

    exterior_2 is duplicated from exterior_1 on a two-camera rig. A present-but-corrupt mp4
    decodes to zero frames and is treated as missing, so one bad video skips one episode
    instead of aborting the whole build with an IndexError.
    """
    decoded: dict[str, list[np.ndarray]] = {}
    for common_key, filename in REAL_CAMERAS.items():
        mp4 = traj_dir / filename
        if not mp4.is_file():
            continue
        try:
            frames = _decode_resized(mp4, IMG_HW)
        except Exception as exc:
            log.warning("%s: %s failed to decode (%s); treating as missing", traj_dir.name, filename, exc)
            continue
        if not frames:
            log.warning("%s: %s decoded to 0 frames; treating as missing", traj_dir.name, filename)
            continue
        decoded[common_key] = frames

    if "exterior_image_1_left" not in decoded or "wrist_image_left" not in decoded:
        log.warning("%s: missing exterior_1 and/or wrist video; skipping episode", traj_dir.name)
        return None
    decoded.setdefault("exterior_image_2_left", decoded["exterior_image_1_left"])
    return decoded


def build_dataset(
    profile: Profile,
    *,
    repo_id: str,
    out_root: Path,
    push: bool = False,
    private: bool = False,
    max_episodes: int | None = None,
    token: str | None = None,
    on_episode=None,
    force: bool = False,
) -> dict:
    """Write (and optionally push) the dataset. Returns a summary dict.

    Every directory under success/ with state data is a candidate, except one whose phase record says
    the method settled the trial itself (``trajectories.settled_outcome``): excluded by verification,
    ended part-way, or aborted. The directory is only where somebody filed it -- a relabel, a merge
    with ``--status``, a move by hand -- and the record is what the method decided. Such a trial is
    listed in ``skipped`` with why, and is exported only once a forced relabel says so on its record.
    A trial with no record (phase planning off, or collected before there was one) is exported as
    before.

    A dataset already at ``out_root/repo_id`` is replaced only by a finished one. Every candidate's
    state is read first, the new dataset is built in a directory beside it, and the two are swapped
    only once the new one is complete; a build that fails, or writes nothing, leaves the old one as it
    was. The directory there is replaced only if this export built it (``MARKER``), or with ``force``.
    """
    success_dir = profile.status_dir("success")
    stated = (
        sorted(d for d in success_dir.iterdir() if d.is_dir() and (d / traj_mod.STATE_FILE).is_file())
        if success_dir.is_dir()
        else []
    )
    skipped: list[tuple[str, str]] = []
    candidates = []
    for directory in stated:
        record = traj_mod.read_hitl(directory)
        settled = traj_mod.settled_outcome(record)
        if settled is None:
            candidates.append(directory)
            continue
        stage = record.get("failure_stage")
        reason = f"{settled}" + (f" at {stage}" if stage else "") + " (hitl.json), so not a demonstration"
        skipped.append((directory.name, reason))
        log.warning("%s: %s; skipping", directory.name, reason)
    held_back = len(skipped)
    if max_episodes is not None:
        candidates = candidates[:max_episodes]
    if not candidates:
        kept_out = f" ({held_back} more are filed there, but their records keep them out)" if held_back else ""
        raise TandemError(
            f"No successful trajectories with {traj_mod.STATE_FILE} under {success_dir}{kept_out}.",
            hint="Collect some, or relabel a trial with `tandem traj relabel <id> success` -- one the method "
            "settled needs --force, and its record then says it was overruled.",
        )

    # Every candidate's state, read before anything on disk is touched: what cannot be an episode is
    # known now, not after the dataset it would have replaced is gone.
    exportable = []
    for directory in candidates:
        try:
            _read_state(directory)
        except _Unexportable as exc:
            log.log(exc.level, "%s: %s; skipping", directory.name, exc)
            skipped.append((directory.name, str(exc)))
            continue
        exportable.append(directory)
    dataset_root = Path(out_root) / repo_id
    untouched = f" {dataset_root} is as it was." if dataset_root.exists() else ""
    if not exportable:
        raise TandemError(
            f"None of the {len(candidates)} successful trajectories under {success_dir} can be exported: "
            + _reasons(skipped[held_back:]),
            hint="Nothing was written." + untouched,
        )
    _check_replaceable(dataset_root, force=force)

    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    _warn_leftovers(dataset_root)
    # Beside the destination, so the swap at the end is a rename within one directory -- never a copy
    # across filesystems that could fail half-way. Made with a plain mkdir, which honours the umask: the
    # rename makes this directory the dataset's root, and `tempfile.mkdtemp` makes every directory 0700,
    # so a dataset the exporter's group and a training container could read became readable by the
    # exporter alone.
    staging = _sibling(dataset_root, "building")
    staging.mkdir()
    written = 0
    writer = None
    try:
        writer = V3DatasetWriter(staging, FPS)
        for traj_dir in exportable:
            result = _add_episode(writer, traj_dir, profile.task.prompt)
            if result is None:
                skipped.append((traj_dir.name, _last_skip_reason))
            else:
                written += 1
            if on_episode is not None:
                on_episode(traj_dir.name, written, len(exportable), result is not None)
        if not written:
            raise TandemError(
                f"None of the {len(exportable)} trajectories could be written: " + _reasons(skipped[held_back:]),
                hint="Nothing was replaced." + untouched,
            )
        writer.finalize()
        (staging / MARKER).write_text(
            json.dumps(
                {
                    "written_by": "tandem",
                    "version": __version__,
                    "profile": profile.name,
                    "repo_id": repo_id,
                    "episodes": written,
                    "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                },
                indent=2,
            )
            + "\n"
        )
        _swap_in(staging, dataset_root)
    except BaseException:
        if writer is not None:
            writer.abort()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    log.info("Wrote %d episode(s) to %s", written, dataset_root)

    pushed = False
    if push and written:
        _upload(dataset_root, repo_id, private, token)
        pushed = True

    return {
        "repo_id": repo_id,
        "dataset_root": str(dataset_root),
        "written": written,
        # The ones held back by their record were looked at too, and are in `skipped` saying why.
        "considered": len(candidates) + held_back,
        "skipped": skipped,
        "pushed": pushed,
    }


def _reasons(skipped: list[tuple[str, str]], limit: int = 5) -> str:
    shown = "; ".join(f"{name}: {why}" for name, why in skipped[:limit])
    return shown + (f"; and {len(skipped) - limit} more" if len(skipped) > limit else "")


def _check_replaceable(dataset_root: Path, *, force: bool) -> None:
    """Refuse to replace ``dataset_root`` unless this export built it, it is empty, or ``force`` says to.

    The destination is ``--out`` joined with the repo name, and either can be wrong by one level: a
    directory of somebody's own there was deleted, whole, before a single episode had been read.
    """
    if not os.path.lexists(dataset_root):
        return
    if dataset_root.is_dir() and not dataset_root.is_symlink():
        if (dataset_root / MARKER).is_file() or not any(dataset_root.iterdir()):
            return
    if force:
        log.warning("Replacing %s, which this export did not build (--force)", dataset_root)
        return
    raise TandemError(
        f"{dataset_root} already exists and is not a dataset tandem built, so it is not replaced.",
        hint="Pick another --out or --repo, or move it aside. --force replaces it -- as a dataset built by "
        f"a tandem from before this check needs once, having no {MARKER} in it.",
    )


def _sibling(dataset_root: Path, kind: str) -> Path:
    """A fresh hidden name beside ``dataset_root`` for a dataset being built or one being replaced."""
    return dataset_root.with_name(f".{dataset_root.name}.{kind}-{uuid.uuid4().hex[:8]}")


def _warn_leftovers(dataset_root: Path) -> None:
    """Say what an export that did not finish left beside ``dataset_root``. Nothing is deleted.

    A build killed outright (the OOM killer mid-encode, a closed terminal, a power cut) leaves its
    staging directory, gigabytes of video under a name `ls` does not show, and nothing ever mentioned
    it again. Not removed here, because an export to the same place still running has one too.
    """
    prefixes = {f".{dataset_root.name}.{kind}-": kind for kind in ("building", "replaced")}
    try:
        names = sorted(os.listdir(dataset_root.parent))
    except OSError:
        return
    for name in names:
        kind = next((kind for prefix, kind in prefixes.items() if name.startswith(prefix)), None)
        if kind is None:
            continue
        path = dataset_root.parent / name
        if kind == "building":
            log.warning(
                "%s is what an export that did not finish (or one still running) left; delete it by hand",
                path,
            )
        elif kind == "replaced" and not os.path.lexists(dataset_root):
            log.warning(
                "%s is the dataset an export was replacing when it stopped, and nothing is at %s: it is the "
                "last complete one, so move it back there to keep it",
                path,
                dataset_root,
            )
        elif kind == "replaced":
            log.warning("%s is a dataset an export replaced and did not delete; delete it by hand", path)


def _swap_in(staging: Path, dataset_root: Path) -> None:
    """Put the finished dataset at ``dataset_root``, replacing whatever was there.

    Two renames within one directory, so there is never a moment with neither: until the second, the
    old dataset is whole (moved aside), and after it the new one is. Only then is the old one deleted,
    and a failure to delete it costs disk, not data -- it is left beside the new one, and said to be.
    """
    if not os.path.lexists(dataset_root):
        os.rename(staging, dataset_root)
        return
    aside = _sibling(dataset_root, "replaced")
    try:
        os.rename(dataset_root, aside)
        os.rename(staging, dataset_root)
    except BaseException:
        # How far the renames got is read off the disk, not from which line raised: Python acts on a
        # Ctrl-C as a call returns, so one lands just after a rename as readily as during it. Going by the
        # line, a Ctrl-C after the first rename (then outside this block) had the new dataset deleted as
        # a failed build and left the old one at a hidden name; one after the second tried to move the
        # old one back onto the new one, and came out as an OSError.
        if not os.path.lexists(staging):
            # Both happened: the new dataset is in place, and there is nothing to undo.
            log.warning("Interrupted, but the new dataset was already in place at %s", dataset_root)
            _discard(aside)
        elif os.path.lexists(aside) and not os.path.lexists(dataset_root):
            # Only the first: the old one goes back.
            os.rename(aside, dataset_root)
        raise
    _discard(aside)


def _discard(aside: Path) -> None:
    try:
        if aside.is_dir() and not aside.is_symlink():
            shutil.rmtree(aside)
        else:
            aside.unlink()
    except OSError as exc:
        log.warning("The dataset it replaced is still at %s (%s); delete it by hand", aside, exc)


_last_skip_reason = ""


def _skip(reason: str) -> None:
    global _last_skip_reason
    _last_skip_reason = reason


# DROID's arm: the schema this export writes has seven joints, and nothing else fits it.
ARM_JOINTS = 7


def _shape_problem(store) -> str | None:
    """Why the state arrays do not fit DROID's schema, or None. Read BEFORE anything is reshaped.

    ``reshape(-1, 7)`` on a 6-joint arm's [F,6] raised a ValueError that nothing caught, and the whole
    export died half-way with a traceback -- or, when F*6 happened to divide by 7, succeeded, and the
    episode was then skipped as "state arrays disagree on length", hiding the real reason.
    """
    for key in ("joint_position", "cmd_joint_position", "cmd_joint_velocity"):
        if key in store.files:
            shape = tuple(store[key].shape)
            if len(shape) != 2 or shape[1] != ARM_JOINTS:
                return (
                    f"{key} has shape {list(shape)}, expected [F,{ARM_JOINTS}] (the export writes DROID's "
                    f"{ARM_JOINTS}-joint schema)"
                )
    if "gripper_position" in store.files:
        shape = tuple(store["gripper_position"].shape)
        if not (len(shape) == 1 or (len(shape) == 2 and shape[1] == 1)):
            return f"gripper_position has shape {list(shape)}, expected [F] or [F,1]"
    return None


class _Unexportable(Exception):
    """Why a trajectory cannot be an episode of this dataset. It is skipped, with this as the reason.

    ``level`` is how loudly: an error for what says capture has regressed (a shape DROID's schema cannot
    hold, a gripper that is not binary), a warning for a trajectory that is simply too short or incomplete.
    """

    def __init__(self, reason: str, *, level: int = logging.WARNING) -> None:
        super().__init__(reason)
        self.level = level


class _State(NamedTuple):
    jp: np.ndarray
    gp: np.ndarray
    cmd_jp: np.ndarray
    cmd_jv: np.ndarray
    cmd_g: np.ndarray
    frame_time: np.ndarray | None
    video_time: np.ndarray | None


def _read_state(traj_dir: Path) -> _State:
    """One trajectory's state arrays, checked against DROID's schema. Raises ``_Unexportable``.

    Reads no video, so it is cheap enough to run over every candidate before anything is written.
    """
    with np.load(traj_dir / traj_mod.STATE_FILE) as store:
        problem = _shape_problem(store)
        if problem is not None:
            raise _Unexportable(problem, level=logging.ERROR)
        try:
            jp = store["joint_position"].astype(np.float32).reshape(-1, 7)
            gp = store["gripper_position"].astype(np.float32).reshape(-1)
            cmd_jp = store["cmd_joint_position"].astype(np.float32).reshape(-1, 7)
            cmd_jv = store["cmd_joint_velocity"].astype(np.float32).reshape(-1, 7)
            cmd_g_raw = store["cmd_gripper"]
        except KeyError as exc:
            raise _Unexportable(f"missing array {exc}") from None
        except ValueError as exc:
            # A backstop for a shape _shape_problem did not foresee: one bad episode is skipped, loudly,
            # rather than ending the export with every episode after it unwritten.
            raise _Unexportable(
                f"its state arrays cannot be read as DROID's schema: {exc}", level=logging.ERROR
            ) from None

        cmd_g_shape = tuple(cmd_g_raw.shape)
        cmd_g = cmd_g_raw.astype(np.float32).reshape(-1)
        # frame_time must stay float64: float32 near the current epoch (~1.8e9) has ~128 s
        # resolution and silently collapses every frame to one timestamp.
        frame_time = (
            np.asarray(store["frame_time"], dtype=np.float64).reshape(-1) if "frame_time" in store.files else None
        )
        video_time = (
            np.asarray(store["video_time"], dtype=np.float64).reshape(-1) if "video_time" in store.files else None
        )

    n = len(jp)
    if n < 2:
        raise _Unexportable(f"only {n} frame(s)")
    if not (len(gp) == len(cmd_jp) == len(cmd_jv) == len(cmd_g) == n):
        raise _Unexportable("state arrays disagree on length")

    # cmd_gripper becomes action[:, 7]. A non-[F]/[F,1] shape or a non-binary value means
    # capture regressed to writing a CONTINUOUS gripper -- the feedback trap where the action
    # is an echo of the state, which is what stopped finetuned policies ever closing the
    # gripper. Skip loudly rather than poison the dataset.
    if len(cmd_g_shape) > 2 or (len(cmd_g_shape) == 2 and cmd_g_shape[1] != 1):
        raise _Unexportable(f"cmd_gripper has shape {cmd_g_shape}, expected [F] or [F,1]", level=logging.ERROR)
    if not np.all((cmd_g == 0.0) | (cmd_g == 1.0)):
        bad = np.unique(cmd_g[(cmd_g != 0.0) & (cmd_g != 1.0)])
        raise _Unexportable(f"cmd_gripper is not binary (e.g. {bad[:4].tolist()})", level=logging.ERROR)
    return _State(jp, gp, cmd_jp, cmd_jv, cmd_g, frame_time, video_time)


def _add_episode(writer: V3DatasetWriter, traj_dir: Path, default_instruction: str):
    try:
        jp, gp, cmd_jp, cmd_jv, cmd_g, frame_time, video_time = _read_state(traj_dir)
    except _Unexportable as exc:
        _skip(str(exc))
        log.log(exc.level, "%s: %s; skipping", traj_dir.name, _last_skip_reason)
        return None
    n = len(jp)

    task = default_instruction
    record_start = record_stop = video_seconds = None
    meta = traj_mod.read_meta(traj_dir)
    if meta:
        task = meta.get("instruction", default_instruction) or default_instruction
        record_start = _finite(meta.get("record_start"))
        record_stop = _finite(meta.get("record_stop"))
        if meta.get("video_aligned"):
            total = _finite(meta.get("total_video_frames"))
            fps_meta = _finite(meta.get("fps")) or FPS
            video_seconds = (total / fps_meta) if total else None
            log.info(
                "%s: merged hand-off trajectory (%d legs); aligning cameras on video_time",
                traj_dir.name,
                len(meta.get("segments") or []),
            )

    decoded = _decode_cameras(traj_dir)
    if decoded is None:
        _skip("missing exterior_1 and/or wrist video")
        return None

    use_video_time = (
        video_time is not None and len(video_time) == n and video_seconds is not None and video_seconds > 0
    )
    use_wallclock = not use_video_time and (
        frame_time is not None
        and len(frame_time) == n
        and record_start is not None
        and record_stop is not None
        and record_stop > record_start
    )
    if not use_video_time and not use_wallclock:
        log.warning(
            "%s: no usable recording window; falling back to proportional camera resampling", traj_dir.name
        )

    aligned = {}
    for common_key in VIDEO_KEY_MAP:
        frames = decoded[common_key]
        if use_video_time:
            idx = _video_time_indices(video_time, len(frames), video_seconds)
        elif use_wallclock:
            idx = _camera_indices(frame_time, len(frames), record_start, record_stop)
        else:
            idx = _resample_indices(len(frames), n)
        aligned[common_key] = np.stack([frames[i] for i in idx])

    cmd_jv_norm, frac_clipped = clip_joint_velocity(cmd_jv)
    if frac_clipped > 0.01:
        log.warning(
            "%s: %.1f%% of joint-velocity elements exceeded [-1, 1] and were clipped",
            traj_dir.name,
            frac_clipped * 100,
        )

    actions = np.concatenate([cmd_jv_norm, cmd_g.reshape(n, 1)], axis=1).astype(np.float32)  # [N, 8]
    writer.add_episode(
        images=aligned,
        joint_position=jp,
        gripper_position=gp.reshape(n, 1),
        actions=actions,
        action_joint_position=cmd_jp,
        task=task,
    )
    return {"frames": n, "task": task}


def _upload(dataset_root: Path, repo_id: str, private: bool, token: str | None) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token or None)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    log.info("Uploading %s -> %s", dataset_root, repo_id)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(dataset_root),
        commit_message="Add LeRobot dataset (tandem)",
        # tandem's own note that it built this directory, which is nobody else's business.
        ignore_patterns=[MARKER],
    )
    # LeRobotDataset(repo_id) needs a git tag matching info.json's codebase_version or it
    # raises RevisionNotFoundError; upload_folder does not create one.
    info = dataset_root / "meta" / "info.json"
    if info.is_file():
        version = json.loads(info.read_text()).get("codebase_version")
        if version:
            api.create_tag(repo_id, tag=version, repo_type="dataset", exist_ok=True)
