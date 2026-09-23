"""Read, index, relabel and delete collected trajectories.

Deliberately dependency-light — numpy and the standard library, nothing else. This is the
module a laptop with no GPU, no robot and no cameras runs, and every import here is on the
path of `tandem ui`.

On-disk layout (unchanged from the source system, so data moves between the two):

    <profile>/trajectories/{eval,success,failure}/<timestamp>/
        external_cam.mp4  external_cam_2.mp4  hand_cam.mp4
        robot_state.npz           per-frame measured + commanded arrays
        _meta.json                instruction, fps, n_frames, timestamps, lineage
        tiptop_plan.json          the TipTop backend's serialized plan (backend-specific, optional)
        segments/<NN>_<source>_<ts>/   raw legs of a merged hand-off trajectory

The first three lines are the RECORDING CONTRACT every leg must meet, whichever backend or human
executor wrote it (see `is_complete`). Anything else in the directory is that recorder's own.
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
# Written when phase planning is on: the phases, the predicates the VLM invented, per-clause
# coverage, and every verification verdict. `vlm/` beside it holds each image sent and a
# rendered PNG of the reply, rejected attempts included.
HITL_FILE = "hitl.json"
VLM_DIR = "vlm"


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
    # Phase planning, when it was on for this rollout.
    has_hitl: bool = False
    n_phases: int = 0
    n_human_phases: int = 0
    has_vlm_log: bool = False

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
            "has_hitl": self.has_hitl,
            "n_phases": self.n_phases,
            "n_human_phases": self.n_human_phases,
            "has_vlm_log": self.has_vlm_log,
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


def is_complete(traj_dir: Path, meta: dict | None = None) -> bool:
    """Whether a leg or episode holds what the recording contract promises.

    That is ``_meta.json``, ``robot_state.npz``, and its camera clips: every clip ``_meta.json``
    names (``cameras`` maps a dataset key to a file in the directory), or, when it names none, at
    least one of CAMERA_FILES. It deliberately does not ask for the backend's plan — a teleop leg
    has none, and neither does a leg from a planner that is not TipTop — which is what the old
    ``tiptop_plan.json`` rule got wrong. Nor does it ask for the phase or lineage keys: a plain
    rollout is complete without them.

    Old data is still accepted: an episode written before ``_meta.json`` existed passes on its
    ``tiptop_plan.json`` instead, which is what made it complete under the rule this replaces.
    """
    if meta is None:
        meta = read_meta(traj_dir)
    if not (traj_dir / STATE_FILE).is_file():
        return False
    if not meta and not (traj_dir / PLAN_FILE).is_file():
        return False

    named = meta.get("cameras") if isinstance(meta.get("cameras"), dict) else {}
    # A bare file name only: the map comes off disk, and `traj_dir / name` must not wander.
    clips = [name for name in named.values() if isinstance(name, str) and name and Path(name).name == name]
    if clips:
        return all((traj_dir / name).is_file() for name in clips)
    return any((traj_dir / name).is_file() for name in CAMERA_FILES)


def read(traj_dir: Path, *, profile_name: str = "", status: str = "", with_size: bool = False) -> Trajectory:
    meta = read_meta(traj_dir)
    n_frames, fps = _frames_and_fps(traj_dir, meta)
    cameras = [name for name in CAMERA_FILES if (traj_dir / name).is_file()]
    has_state = (traj_dir / STATE_FILE).is_file()
    has_plan = (traj_dir / PLAN_FILE).is_file()

    # A merged hand-off trajectory carries its legs' boundaries; an ordinary rollout has none.
    segments = meta.get("segments") if meta.get("video_aligned") else None

    hitl = read_hitl(traj_dir)
    phases = hitl.get("phases") or [] if hitl else []

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
        complete=is_complete(traj_dir, meta),
        trajectory_id=meta.get("trajectory_id"),
        segment_source=meta.get("segment_source"),
        segments=segments if isinstance(segments, list) else None,
        recorded_at=_as_float(meta.get("record_start")),
        size_bytes=_dir_size(traj_dir) if with_size else 0,
        meta=meta,
        has_hitl=bool(hitl),
        n_phases=len(phases),
        n_human_phases=sum(1 for p in phases if isinstance(p, dict) and p.get("executor") == "human"),
        has_vlm_log=(traj_dir / VLM_DIR).is_dir(),
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


def read_hitl(traj_dir: Path) -> dict:
    """The phase plan recorded with a rollout, or {} when phase planning was off.

    Holds what was asked of the robot and of the person, the predicates the VLM invented to
    describe the human's part, which clauses of the instruction each phase covered, and every
    verification verdict — which together are what make a HITL episode auditable after the
    fact rather than a video someone has to re-watch.
    """
    path = traj_dir / HITL_FILE
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def plan(traj: Trajectory) -> dict | None:
    path = traj.path / PLAN_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return None
