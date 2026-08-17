"""Read, index, relabel and delete collected trajectories.

Deliberately dependency-light — numpy and the standard library, nothing else. This is the
module a laptop with no GPU, no robot and no cameras runs, and every import here is on the
path of `tandem ui`.

On-disk layout (unchanged from the source system, so data moves between the two):

    <profile>/trajectories/{eval,success,failure}/<timestamp>/
        external_cam.mp4  external_cam_2.mp4  hand_cam.mp4
        tiptop_plan.json          serialized TAMP plan
        robot_state.npz           per-frame measured + commanded arrays
        _meta.json                instruction, fps, n_frames, timestamps, lineage
        segments/<NN>_<source>_<ts>/   raw legs of a merged hand-off trajectory
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tandem.core.errors import TandemError
from tandem.core.profiles import STATUSES, Profile

DEFAULT_FPS = 15

# In capture order. exterior_1 / exterior_2 / wrist in DROID's naming.
CAMERA_FILES = ("external_cam.mp4", "external_cam_2.mp4", "hand_cam.mp4")
CAMERA_LABELS = {
    "external_cam.mp4": "Exterior 1",
    "external_cam_2.mp4": "Exterior 2",
    "hand_cam.mp4": "Wrist",
}

PLAN_FILE = "tiptop_plan.json"
STATE_FILE = "robot_state.npz"
META_FILE = "_meta.json"


@dataclass
class Trajectory:
    """One collected trajectory. `path` is the truth; everything else is read from it."""

    id: str
    profile: str
    status: str
    path: Path
    n_frames: int = 0
    fps: int = DEFAULT_FPS
    duration_s: float = 0.0
    instruction: str = ""
    cameras: list[str] = field(default_factory=list)
    has_state: bool = False
    has_plan: bool = False
    complete: bool = False
    trajectory_id: str | None = None
    segment_source: str | None = None
    segments: list[dict] | None = None
    recorded_at: float | None = None
    size_bytes: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "profile": self.profile,
            "status": self.status,
            "path": str(self.path),
            "n_frames": self.n_frames,
            "fps": self.fps,
            "duration_s": round(self.duration_s, 2),
            "instruction": self.instruction,
            "cameras": self.cameras,
            "camera_labels": [CAMERA_LABELS.get(c, c) for c in self.cameras],
            "has_state": self.has_state,
            "has_plan": self.has_plan,
            "complete": self.complete,
            "trajectory_id": self.trajectory_id,
            "segment_source": self.segment_source,
            "segments": self.segments,
            "merged": bool(self.segments and len(self.segments) > 1),
            "recorded_at": self.recorded_at,
            "size_bytes": self.size_bytes,
        }


# --------------------------------------------------------------------------- reading


def read_meta(traj_dir: Path) -> dict:
    """`_meta.json`, or {} when absent or unreadable. Legacy episodes have none."""
    path = traj_dir / META_FILE
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _frames_and_fps(traj_dir: Path, meta: dict) -> tuple[int, int]:
    """Prefer `_meta.json`; fall back to measuring `robot_state.npz`."""
    if meta:
        try:
            n = int(meta.get("n_frames", 0))
            fps = int(meta.get("fps", DEFAULT_FPS)) or DEFAULT_FPS
            if n:
                return n, fps
        except (TypeError, ValueError):
            pass

    npz_path = traj_dir / STATE_FILE
    if not npz_path.is_file():
        return 0, DEFAULT_FPS
    try:
        import numpy as np

        with np.load(npz_path) as data:
            key = "joint_position" if "joint_position" in data.files else "frame_time"
            n = int(len(data[key])) if key in data.files else 0
            fps = DEFAULT_FPS
            if "frame_time" in data.files and len(data["frame_time"]) > 1:
                span = float(data["frame_time"][-1] - data["frame_time"][0])
                if span > 0:
                    fps = max(1, int(round((len(data["frame_time"]) - 1) / span)))
            return n, fps
    except (ValueError, OSError):
        return 0, DEFAULT_FPS


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except OSError:
        pass
    return total


def read(traj_dir: Path, *, profile_name: str = "", status: str = "", with_size: bool = False) -> Trajectory:
    meta = read_meta(traj_dir)
    n_frames, fps = _frames_and_fps(traj_dir, meta)
    cameras = [name for name in CAMERA_FILES if (traj_dir / name).is_file()]
    has_state = (traj_dir / STATE_FILE).is_file()
    has_plan = (traj_dir / PLAN_FILE).is_file()

    # A merged hand-off trajectory carries its legs' boundaries; an ordinary rollout has none.
    segments = meta.get("segments") if meta.get("video_aligned") else None

    return Trajectory(
        id=traj_dir.name,
        profile=profile_name,
        status=status or traj_dir.parent.name,
        path=traj_dir,
        n_frames=n_frames,
        fps=fps,
        duration_s=(n_frames / fps) if fps else 0.0,
        instruction=str(meta.get("instruction") or ""),
        cameras=cameras,
        has_state=has_state,
        has_plan=has_plan,
        # tamp rollouts need both; a teleop leg has no plan of its own.
        complete=has_state and (has_plan or meta.get("segment_source") == "teleop"),
        trajectory_id=meta.get("trajectory_id"),
        segment_source=meta.get("segment_source"),
        segments=segments if isinstance(segments, list) else None,
        recorded_at=_as_float(meta.get("record_start")),
        size_bytes=_dir_size(traj_dir) if with_size else 0,
        meta=meta,
    )


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- listing


def list_all(profile: Profile, *, status: str | None = None, with_size: bool = False) -> list[Trajectory]:
    """Every trajectory under a profile, newest first."""
    out: list[Trajectory] = []
    wanted = (status,) if status else STATUSES
    for st in wanted:
        status_dir = profile.trajectories_dir() / st
        if not status_dir.is_dir():
            continue
        for entry in sorted(status_dir.iterdir()):
            # `segments/` sits one level deeper, so raw legs never appear as trajectories.
            if entry.is_dir():
                out.append(read(entry, profile_name=profile.name, status=st, with_size=with_size))
    return sorted(out, key=lambda t: t.id, reverse=True)


def counts(profile: Profile) -> dict[str, int]:
    """Completed trajectories per status. Incomplete dirs (a planning failure writes no
    state) are not counted — they are not data."""
    out = {status: 0 for status in STATUSES}
    for status in STATUSES:
        status_dir = profile.trajectories_dir() / status
        if not status_dir.is_dir():
            continue
        for entry in status_dir.iterdir():
            if entry.is_dir() and (entry / STATE_FILE).is_file():
                out[status] += 1
    return out


def find(profile: Profile, traj_id: str, *, status: str | None = None) -> Trajectory:
    """Resolve a trajectory by id, searching every status unless one is given.

    Accepts a unique prefix, because the ids are timestamps and nobody wants to type
    `2026-08-16_21-14-02` in full.
    """
    if status:
        candidate = profile.status_dir(status) / traj_id
        if candidate.is_dir():
            return read(candidate, profile_name=profile.name, status=status)

    exact: list[Trajectory] = []
    prefixed: list[Trajectory] = []
    for traj in list_all(profile):
        if traj.id == traj_id:
            exact.append(traj)
        elif traj.id.startswith(traj_id):
            prefixed.append(traj)

    if exact:
        return exact[0]
    if len(prefixed) == 1:
        return prefixed[0]
    if len(prefixed) > 1:
        raise TandemError(
            f"{traj_id!r} matches {len(prefixed)} trajectories in profile {profile.name!r}.",
            hint="Use more of the timestamp: " + ", ".join(t.id for t in prefixed[:4]),
        )
    raise TandemError(
        f"No trajectory {traj_id!r} in profile {profile.name!r}.",
        hint=f"Run `tandem traj list {profile.name}` to see what is there.",
    )


# --------------------------------------------------------------------------- mutation


def relabel(profile: Profile, traj: Trajectory, new_status: str) -> Trajectory:
    """Move a trajectory between eval / success / failure."""
    if new_status not in STATUSES:
        raise TandemError(f"Unknown status {new_status!r}.", hint=f"One of: {', '.join(STATUSES)}")
    if traj.status == new_status:
        return traj

    dest_dir = profile.status_dir(new_status)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / traj.id
    if dest.exists():
        raise TandemError(
            f"{dest} already exists.",
            hint="Two trajectories share a timestamp; move or delete one by hand.",
        )
    shutil.move(str(traj.path), str(dest))
    return read(dest, profile_name=profile.name, status=new_status)


def delete(profile: Profile, traj: Trajectory) -> Path:
    path = traj.path
    if not path.is_dir():
        raise TandemError(f"{path} is not a directory.")
    # Refuse anything that is not actually inside the profile: relabel/find take user input,
    # and rmtree is not a place to be relaxed about that.
    root = profile.trajectories_dir().resolve()
    if root not in path.resolve().parents:
        raise TandemError(f"Refusing to delete {path}: it is outside {root}.")
    shutil.rmtree(path)
    return path


# --------------------------------------------------------------------------- media


def media_path(traj: Trajectory, filename: str) -> Path:
    """Resolve a media file inside a trajectory, rejecting anything that escapes it."""
    if filename not in CAMERA_FILES and not filename.endswith((".mp4", ".json", ".gif", ".png")):
        raise TandemError(f"Refusing to serve {filename!r}.")
    candidate = (traj.path / filename).resolve()
    if traj.path.resolve() not in candidate.parents:
        raise TandemError(f"Refusing to serve {filename!r}: it escapes the trajectory directory.")
    if not candidate.is_file():
        raise TandemError(f"{filename} is not in {traj.path.name}.")
    return candidate


def plan(traj: Trajectory) -> dict | None:
    path = traj.path / PLAN_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return None
