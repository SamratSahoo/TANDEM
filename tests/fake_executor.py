"""A human executor with nobody on the other end.

The phase loop hands a human phase to whatever ``hitl.human_executor`` names, through the executor
registry, the way it hands a robot phase to a planner backend. This stands in for it as
``fake_backend.FakeBackend`` stands in for a planner: in-process, instant unless told to wait, and
keeping what it was asked for. Where a test needs the data layer, it writes its leg to disk in the
real format (``write_leg``), so a merge can join it to the robot's legs.

What each run was asked is kept in ``calls`` (the request, None for a hand-off the operator asked
for) and ``legs`` (the ``LegSpec``), so a test can say what the executor was told without knowing
how the loop builds it.
"""

from __future__ import annotations

import itertools
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from tandem.executors.base import CustodyError, ExecutorFactory, HumanPhaseResult

FPS = 15
# One clock for every leg a test writes, robot or human, so the merge (which orders legs by their
# recording window) sees them in the order they were recorded, however fast the test ran.
_tick = itertools.count()


def write_leg(directory: Path, leg: Any, *, n_frames: int, source: str) -> Path:
    """One leg on disk as a recorder leaves it: its state arrays, a camera clip and its _meta.json.

    Stamped from the ``LegSpec`` exactly as the real recorders stamp theirs: the trajectory id, what
    the leg is (``segment_source``), and the phase keys only when the leg carries a phase. Anything
    already in the directory's _meta.json is kept underneath.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    t0 = 1_800_000_000.0 + 600.0 * next(_tick)
    frame_time = t0 + np.arange(n_frames, dtype=np.float64) / FPS
    joints = np.tile(np.arange(7, dtype=np.float32), (n_frames, 1))
    np.savez(
        directory / "robot_state.npz",
        joint_position=joints,
        gripper_position=np.zeros(n_frames, np.float32),
        cmd_joint_position=joints,
        cmd_joint_velocity=np.zeros_like(joints),
        cmd_gripper=np.zeros(n_frames, np.float32),
        frame_time=frame_time,
    )
    (directory / "external_cam.mp4").write_bytes(b"not really a video")
    meta_path = directory / "_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    meta.update(
        {
            "instruction": leg.instruction,
            "fps": FPS,
            "n_frames": n_frames,
            "timestamp": directory.name,
            "trajectory_id": leg.trajectory_id,
            "segment_source": source,
            "cameras": {"observation.images.exterior_1_left": "external_cam.mp4"},
            "record_start": float(frame_time[0]),
            "record_stop": float(frame_time[0]) + n_frames / FPS,
        }
    )
    if leg.phase_index is not None:
        meta.update(
            phase_index=leg.phase_index, n_phases=leg.n_phases, phase_description=leg.phase_description
        )
    meta_path.write_text(json.dumps(meta))
    return directory


class FakeExecutor:
    """Implements the HumanExecutor protocol, with nobody driving the arm.

    * ``n_frames``: what each leg records. 0 is a leg that recorded nothing.
    * ``statuses``: how each run ends, one per run, the last repeating (``done`` by default).
    * ``write``: put the leg on disk under ``save_root/eval/``, as a recorder does.
    * ``wait``: do not return until ``should_stop`` says so -- a person holding the arm until they
      hand it back -- or until ``kill``.
    * ``custody``: never let go, raising ``CustodyError`` as a driver that will not exit does.
    * ``on_run(request, leg)``: called as the leg starts, for asserting on what holds the hardware.
    """

    name = "fake"
    display_name = "Nobody"
    summary = "Carries out a human phase with nobody on the other end."

    def __init__(
        self,
        ctx: Any = None,
        *,
        segment_source: str = "teleop",
        n_frames: int = 30,
        statuses: tuple[str, ...] = ("done",),
        write: bool = False,
        wait: bool = False,
        custody: bool = False,
        on_run: Callable[[Any, Any], None] | None = None,
    ) -> None:
        self.ctx = ctx
        self.segment_source = segment_source
        self.n_frames = n_frames
        self.statuses = list(statuses)
        self.write = write
        self.wait = wait
        self.custody = custody
        self.on_run = on_run
        self.calls: list = []
        self.legs: list = []
        self.written: list[Path] = []
        self.running = threading.Event()
        self.killed = threading.Event()

    def run(self, request, leg, *, save_root: Path, should_stop) -> HumanPhaseResult:
        self.calls.append(request)
        self.legs.append(leg)
        self.killed.clear()
        if self.on_run is not None:
            self.on_run(request, leg)
        if self.wait:
            self.running.set()
            deadline = time.monotonic() + 20.0
            try:
                while not should_stop() and not self.killed.is_set() and time.monotonic() < deadline:
                    time.sleep(0.01)
            finally:
                self.running.clear()
        if self.custody:
            raise CustodyError("the stand-in executor will not let go of the arm")

        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        if self.killed.is_set():
            status = "aborted"
        leg_dir = None
        if self.write and self.n_frames:
            leg_dir = Path(save_root) / "eval" / f"human-{len(self.calls):02d}"
            write_leg(leg_dir, leg, n_frames=self.n_frames, source=self.segment_source)
            self.written.append(leg_dir)
        return HumanPhaseResult(status, n_frames=self.n_frames, leg_dir=leg_dir)

    def kill(self) -> None:
        self.killed.set()


def use_fake_executor(
    monkeypatch, executor: FakeExecutor | None = None, *, name: str = "teleop", **kwargs
) -> list[FakeExecutor]:
    """Make the executor registry build a FakeExecutor as ``name``, and return every one it builds.

    Registered through ``register_human_executor``, so the loop takes exactly the path it takes in
    production: the registry, a factory, an ``ExecutorContext``. Pass ``executor`` to have that one
    instance handed out; otherwise each build is a fresh ``FakeExecutor(ctx, **kwargs)``. Whatever
    this registers is forgotten at teardown.
    """
    from tandem.executors import base

    monkeypatch.setattr(base, "_registered", dict(base._registered))
    built: list[FakeExecutor] = []

    def create(ctx) -> FakeExecutor:
        made = executor if executor is not None else FakeExecutor(ctx, **kwargs)
        if made.ctx is None:
            made.ctx = ctx
        built.append(made)
        return made

    source = executor.segment_source if executor is not None else kwargs.get("segment_source", "teleop")
    factory = ExecutorFactory(
        create=create,
        display_name=FakeExecutor.display_name,
        summary=FakeExecutor.summary,
        segment_source=source,
    )
    base.register_human_executor(name, factory, replace=True)
    return built
