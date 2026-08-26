"""A task-and-motion planner with no planner behind it.

The session engine's job is to decide who does what and in what order, and to hold the arm's
custody straight across a hand-off. None of that needs a real planner — but all of it needs a
backend that behaves like one, including the awkward parts: perception that renames objects between
passes, a plan that sometimes cannot be found, and hardware that has to be released before anyone
else can touch it.

It records every call in ``calls``, so a test can assert on the ORDER things happened in. That
ordering is the part that bites: release before the teleop child starts, reacquire only after it has
gone, and never a perception pass that parks an arm a person is still holding something with.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tandem.planners.base import ExecuteResult, GoalAtom, LegSpec, PlanResult, SceneView
from tandem.planners.tiptop.capabilities import CAPABILITIES

# The scene every pass reports, unless a test says otherwise. Chosen to match the plan the canned
# proposer returns: a movable, a container it goes into, and the table.
DEFAULT_LABELS = ("blue_toy", "white_box")
DEFAULT_TABLE = "table"


@dataclass
class FakeBackend:
    """Implements the TampBackend protocol, in-process and instantly."""

    name: str = "fake"
    labels: tuple[str, ...] = DEFAULT_LABELS
    table: str = DEFAULT_TABLE
    # Set to a reason to make every plan() fail, which is how the on_robot_phase_failure policies
    # are exercised without a robot that cannot reach something.
    plan_failure: str | None = None
    # Labels reported from the SECOND pass onwards, for the drift-rebinding path.
    drifted_labels: tuple[str, ...] | None = None
    n_frames: int = 24
    calls: list[str] = field(default_factory=list)
    legs: list[dict] = field(default_factory=list)
    warmed: bool = False
    closed: bool = False
    holds_hardware: bool = True
    perceptions: int = 0

    # Accepted by the constructor the registry hands the session, and ignored.
    def __init__(self, runtime=None, **kwargs: Any) -> None:
        self.name = "fake"
        self.labels = kwargs.pop("labels", DEFAULT_LABELS)
        self.table = DEFAULT_TABLE
        self.plan_failure = kwargs.pop("plan_failure", None)
        self.drifted_labels = kwargs.pop("drifted_labels", None)
        self.n_frames = kwargs.pop("n_frames", 24)
        self.calls = []
        self.legs = []
        self.warmed = False
        self.closed = False
        self.holds_hardware = True
        self.perceptions = 0
        self._output_dir = Path(kwargs.get("output_dir") or ".")
        self._on_log = kwargs.get("on_log")

    # ---- what this planner is ----------------------------------------------

    def capabilities(self):
        # The real declaration, so a plan validated here is one the real backend would accept.
        return CAPABILITIES

    def require_ready(self) -> None:
        self.calls.append("require_ready")

    # ---- lifecycle ---------------------------------------------------------

    def warm(self) -> None:
        self.calls.append("warm")
        self.warmed = True

    def close(self) -> None:
        self.calls.append("close")
        self.closed = True

    # ---- hardware custody --------------------------------------------------

    def release_hardware(self) -> None:
        self.calls.append("release_hardware")
        self.holds_hardware = False

    def reacquire_hardware(self) -> None:
        self.calls.append("reacquire_hardware")
        self.holds_hardware = True

    def capture_frame(self, *, camera: str = "external") -> str:
        self.calls.append(f"capture_frame:{camera}")
        assert self.holds_hardware, "a frame was asked for while the cameras were handed away"
        return self._write_image(self._output_dir / "verify.png")

    def home(self) -> None:
        self.calls.append("home")

    # ---- the sub-goal cycle ------------------------------------------------

    def perceive(self, *, task_hint: str, save_dir: Path, reset_arm: bool = True) -> SceneView:
        self.calls.append(f"perceive:{'reset' if reset_arm else 'keep'}")
        assert self.holds_hardware, "perception ran while the cameras were handed away"
        self.perceptions += 1
        labels = self.labels
        if self.drifted_labels is not None and self.perceptions > 1:
            labels = self.drifted_labels
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        return SceneView(
            object_labels=tuple(labels),
            table_label=self.table,
            surface_labels=frozenset(),
            rgb_path=self._write_image(save_dir / "perception_rgb.png"),
            scene_id=f"scene-{self.perceptions}",
            # What an ordinary rollout would have planned for, from the instruction alone.
            detected_goal=(GoalAtom("on", (labels[0], self.table)),),
        )

    def plan(self, scene_id, goal, *, surfaces=frozenset(), save_dir, reuse_skeleton=None) -> PlanResult:
        rendered = [a.to_dict() for a in goal]
        self.calls.append(f"plan:{json.dumps(rendered)}")
        if self.plan_failure:
            return PlanResult(ok=False, failure_reason=self.plan_failure)
        return PlanResult(ok=True, planning_seconds=0.5, plan_handle=f"plan-{len(self.calls)}")

    def execute(self, plan_handle, leg: LegSpec, *, save_dir: Path, should_stop=None) -> ExecuteResult:
        self.calls.append(f"execute:{leg.phase_index}")
        assert self.holds_hardware, "the arm was driven while it was handed away"
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # The backend is what stamps the leg's identity, because merging keys on it.
        (save_dir / "_meta.json").write_text(
            json.dumps(
                {
                    "n_frames": self.n_frames,
                    "instruction": leg.instruction,
                    "trajectory_id": leg.trajectory_id,
                    "segment_source": leg.segment_source,
                    "phase_index": leg.phase_index,
                }
            )
        )
        self.legs.append({"leg": leg, "dir": str(save_dir)})
        return ExecuteResult(ok=True, n_frames=self.n_frames, rollout_dir=str(save_dir))

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _write_image(path: Path) -> str:
        from PIL import Image

        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), (90, 110, 130)).save(path)
        return str(path)

    def call_names(self) -> list[str]:
        """Every call, with its arguments stripped — for asserting on ORDER."""
        return [c.split(":", 1)[0] for c in self.calls]
