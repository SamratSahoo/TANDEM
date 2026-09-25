"""A world of items and bins with no robot in it: the planner behind tandem's own SDK tests.

It is hosted two ways, and the point is that they are the same world. ``toy_planner.ToyPlanner`` runs
it in tandem's process, as a ``Planner``; ``toy_sidecar.py`` serves it from a child process with
``tandem_sidecar``, the way a planner that needs its own environment would, behind
``toy_planner.ToySidecarPlanner``. One conformance run against each then says something about both
hostings rather than about two toys that happen to share a name.

So it imports nothing from tandem (a sidecar cannot), only the standard library, numpy and pillow,
and speaks in the plain dicts that cross the wire. Its goal language is deliberately not cuTAMP's:
``in_bin(item, bin)``, with no table to put things on and no exclusivity, which is the vocabulary
of the ``InBin`` predicate the toy planners declare.
"""

from __future__ import annotations

import json
import tempfile
import time
import uuid
from pathlib import Path

ITEMS = ("apple", "pear", "plum")
BINS = ("blue_bin", "red_bin")
FLOOR = "floor"
WIRE = "in_bin"
# tandem.core.merge.STATE_KEYS, restated because the sidecar half cannot import tandem. A test pins
# the two together, so a toy leg is exactly what merging joins.
STATE_KEYS = (
    "joint_position",
    "gripper_position",
    "cmd_joint_position",
    "cmd_joint_velocity",
    "cmd_gripper",
    "frame_time",
)
FRAMES_PER_DROP = 6
FPS = 15
CLIP = "external_cam.mp4"


class ToyWorld:
    """Items on the floor, bins to drop them in, and custody of an imaginary arm and camera."""

    def __init__(self, *, items: tuple[str, ...] = ITEMS, step_seconds: float = 0.0) -> None:
        self.where = {item: FLOOR for item in items}
        self.scenes: dict = {}
        self.plans: dict = {}
        self.holds_hardware = True
        self.warmed = False
        # How long one recorded frame takes. Zero in-process; a little in a sidecar, so a stop asked
        # for from the other process has a step boundary to land on.
        self.step_seconds = step_seconds
        self._frames = Path(tempfile.mkdtemp(prefix="toy-frames-"))

    # ---- lifecycle and custody ---------------------------------------------------------------------

    def warm(self, *, output_dir=None, execute=True, record=True, station=None, robot_host=None) -> dict:
        self.warmed = True
        self.station, self.robot_host = station, robot_host
        return {"items": sorted(self.where)}

    def close(self) -> dict:
        self.warmed = False
        return {}

    def release_hardware(self) -> dict:
        self.holds_hardware = False
        return {}

    def reacquire_hardware(self) -> dict:
        self.holds_hardware = True
        return {}

    def home(self) -> dict:
        self._require_hardware("home")
        return {}

    def capture_frame(self, *, camera: str = "external") -> dict:
        self._require_hardware("capture_frame")
        return {"path": self._image(self._frames / f"{camera}-{uuid.uuid4().hex[:8]}.png")}

    # ---- the sub-goal cycle ------------------------------------------------------------------------

    def perceive(
        self, *, task_hint: str, save_dir: str, reset_arm: bool = True, open_gripper: bool = False
    ) -> dict:
        self._require_hardware("perceive")
        directory = Path(save_dir)
        directory.mkdir(parents=True, exist_ok=True)
        scene_id = uuid.uuid4().hex[:8]
        self.scenes[scene_id] = dict(self.where)
        return {
            "scene_id": scene_id,
            "object_labels": [*self.where, *BINS],
            "table_label": FLOOR,
            "surface_labels": list(BINS),
            "rgb_path": self._image(directory / "perception_rgb.png"),
            "detected_goal": [],
        }

    def plan(
        self,
        *,
        scene_id: str,
        goal: list,
        surfaces: list,
        save_dir: str,
        movables: list | None = None,
        return_home: bool = True,
    ) -> dict:
        scene = self.scenes.get(scene_id)
        if scene is None:
            raise ValueError(f"unknown scene {scene_id!r}; perceive again before planning")
        drops = []
        for atom in goal:
            args = list(atom.get("args") or [])
            if atom.get("predicate") != WIRE or len(args) != 2:
                return self._no_plan(f"the toy world cannot plan {atom}")
            item, bin_ = args
            if item not in scene:
                return self._no_plan(f"there is no {item} in this scene")
            if bin_ not in BINS:
                return self._no_plan(f"{bin_} is not a bin")
            drops.append((item, bin_))
        if movables is not None:
            outside = sorted({item for item, _ in drops} - set(movables))
            if outside:
                return self._no_plan(
                    f"the goal moves {', '.join(outside)}, but this leg may only pick "
                    f"{', '.join(sorted(movables)) or 'nothing'}"
                )
        directory = Path(save_dir)
        directory.mkdir(parents=True, exist_ok=True)
        plan_path = directory / "toy_plan.json"
        plan_path.write_text(json.dumps({"drops": drops, "return_home": return_home}))
        handle = uuid.uuid4().hex[:8]
        self.plans[handle] = {"drops": drops, "return_home": return_home}
        return {
            "ok": True,
            "planning_seconds": 0.01,
            "plan_handle": handle,
            "artifacts": {"plan": str(plan_path)},
            "task_plan": [f"Drop({item}, {bin_})" for item, bin_ in drops],
        }

    def execute(self, *, plan_handle: str, leg: dict, save_dir: str, should_stop=None) -> dict:
        self._require_hardware("execute")
        entry = self.plans.get(plan_handle)
        if entry is None:
            raise ValueError(f"unknown plan {plan_handle!r}")
        directory = Path(save_dir)
        directory.mkdir(parents=True, exist_ok=True)
        start = time.time()
        frames = 0
        stopped = False
        for item, bin_ in entry["drops"]:
            for _ in range(FRAMES_PER_DROP):
                if should_stop is not None and should_stop():
                    stopped = True
                    break
                frames += 1
                if self.step_seconds:
                    time.sleep(self.step_seconds)
            if stopped:
                break
            self.where[item] = bin_
        if leg.get("record", True):
            self._record(directory, leg, frames, start, time.time())
        return {
            "ok": not stopped,
            "stopped_early": stopped,
            "n_frames": frames,
            "rollout_dir": str(directory),
            "failure_reason": "stopped at the operator's request" if stopped else None,
        }

    # ---- helpers -----------------------------------------------------------------------------------

    def _record(self, directory: Path, leg: dict, frames: int, start: float, stop: float) -> None:
        """The recording contract: state, a clip, and a _meta.json that says which leg this is."""
        import numpy as np

        arrays = {
            "joint_position": np.zeros((frames, 7)),
            "gripper_position": np.zeros(frames),
            "cmd_joint_position": np.zeros((frames, 7)),
            "cmd_joint_velocity": np.zeros((frames, 7)),
            "cmd_gripper": np.zeros(frames),
            "frame_time": start + np.arange(frames) / FPS,
        }
        np.savez(directory / "robot_state.npz", **arrays)
        # A placeholder, not a video: what the contract asks is that the clip it names is there.
        (directory / CLIP).write_bytes(b"toy clip")
        meta = {
            "trajectory_id": leg.get("trajectory_id"),
            "segment_source": leg.get("segment_source") or "tamp",
            "instruction": leg.get("instruction") or "",
            "n_frames": frames,
            "fps": FPS,
            "record_start": start,
            "record_stop": stop,
            "cameras": {"exterior_image_1_left": CLIP},
        }
        if leg.get("phase_index") is not None:
            meta.update(
                phase_index=leg["phase_index"],
                n_phases=leg.get("n_phases"),
                phase_description=leg.get("phase_description") or "",
            )
        (directory / "_meta.json").write_text(json.dumps(meta, indent=2))

    def _require_hardware(self, verb: str) -> None:
        if not self.holds_hardware:
            raise RuntimeError(f"{verb} was asked for while the arm and camera were handed away")

    @staticmethod
    def _no_plan(reason: str) -> dict:
        return {"ok": False, "failure_reason": reason, "planning_seconds": 0.0}

    @staticmethod
    def _image(path: Path) -> str:
        from PIL import Image

        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), (120, 90, 60)).save(path)
        return str(path)
