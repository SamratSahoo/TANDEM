#!/usr/bin/env python3
"""tandem's half of the TiPToP backend, run inside the planner's own environment.

This file is TANDEM's code executed by the runtime's interpreter, where torch, cuRobo, cuTAMP and
TiPToP are importable and tandem is not. It answers the verbs in ``tandem.planners.base.VERBS`` over
newline-delimited JSON on stdin/stdout, and implements each of them as a call to a PUBLIC function of
an unmodified TiPToP.

That is the whole design. The phase planner used to live inside a fork of TiPToP, which meant every
planner tandem wanted to drive had to be forked and the fork kept alive against upstream. Here the
planner-specific knowledge is one file on tandem's side, and pointing tandem at a different task and
motion planner means writing another one of these -- not patching the planner.

    parent (pure python, no CUDA)                 this process (pixi runtime)
      TiptopBackend.plan(goal) ──JSON──►            create_tamp_environment(goal)
                               ◄──JSON──            run_planning(...)

It imports nothing from ``tandem``. It is passed to the interpreter by path, so there is no tandem on
this process's sys.path and adding it would mean installing tandem's dependencies into the planner's
environment for no reason. The cost is that the verb list is repeated here; ``tests/test_planners.py``
pins the two copies together so they cannot drift.

Run it by hand to debug a backend:

    pixi run --manifest-path <runtime>/tiptop/pixi.toml python .../sidecar.py
    {"id": 1, "verb": "capabilities", "args": {}}
"""

from __future__ import annotations

import json
import os
import sys
import traceback
import uuid

# ---------------------------------------------------------------------------------------- stdout
# Everything below imports libraries that print: CUDA banners, warp's version line, SAM-2's
# progress, a stray print() in a vendored tree. Any one of them on fd 1 would land in the middle of
# a JSON reply and desynchronise the channel for good. So the real stdout is taken away first and
# kept private, and fd 1 is pointed at stderr -- where the parent pumps it into the session log,
# which is where that output was always wanted anyway.
_PROTOCOL_OUT = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

# The verbs this sidecar answers. Mirrors tandem.planners.base.VERBS, which it cannot import.
VERBS = (
    "capabilities",
    "warm",
    "close",
    "release_hardware",
    "reacquire_hardware",
    "capture_frame",
    "home",
    "perceive",
    "plan",
    "execute",
)

# Margin between releasing a camera and telling another process it may open it. The SDK's teardown
# already blocks (~14s for two ZEDs, measured) and the device is claimable about a second later, so
# this is slack rather than a readiness check.
CAMERA_RELEASE_SETTLE_S = 2.0


def _emit(payload: dict) -> None:
    _PROTOCOL_OUT.write(json.dumps(payload) + "\n")
    _PROTOCOL_OUT.flush()


def _log(message: str) -> None:
    _emit({"log": message})


class Sidecar:
    """One warm planner, and the scenes and plans it has produced this session."""

    def __init__(self) -> None:
        self.container = None
        self.config = None
        self.loop = None
        self.http = None
        self.output_dir = None
        self.execute_plans = True
        self.record = True
        self.cost_overrides: dict = {}
        # scene_id -> everything a later plan() call needs, so a phase is planned against the pass
        # that reported the labels it is stated in rather than a fresh one.
        self.scenes: dict = {}
        # plan_handle -> the cuTAMP plan, where it was saved, and the scene it came from.
        self.plans: dict = {}
        self._had_external_cam_2 = False

    # ---- lifecycle ---------------------------------------------------------

    def warm(self, *, output_dir: str, execute: bool, record: bool, cost_overrides: str | None) -> dict:
        """Build the solvers, open the cameras, connect the robot. Tens of seconds."""
        import asyncio
        import logging
        from concurrent.futures import ProcessPoolExecutor

        import rerun as rr
        from tiptop import tiptop_run
        from tiptop.config import tiptop_cfg
        from tiptop.motion_planning import (
            resolve_grasp_orientation_cost,
            resolve_max_motion_refine_attempts,
            resolve_time_dilation_factor,
            resolve_traj_length_norm,
        )
        from tiptop.planning import build_tamp_config
        from tiptop.tiptop_run import get_demo_container
        from tiptop.tiptop_websocket_server import _load_curobo_overrides
        from tiptop.utils import check_cutamp_version, setup_logging

        self.output_dir = output_dir
        self.execute_plans = bool(execute)
        self.record = bool(record)
        setup_logging(level=logging.INFO)
        check_cutamp_version()

        # Same knobs an ordinary tiptop-run reads, from the same file tandem already renders for it.
        self.cost_overrides = _load_curobo_overrides(cost_overrides)
        num_particles = int(self.cost_overrides.get("num_particles") or 256)
        opt_steps = int(self.cost_overrides.get("opt_steps_per_skeleton") or 500)
        max_planning_time = float(self.cost_overrides.get("max_planning_time") or 60.0)

        cfg = tiptop_cfg()
        robot_types = tiptop_run._planning_robot_types()
        tamp_configs = {
            robot_type: build_tamp_config(
                num_particles=num_particles,
                max_planning_time=max_planning_time,
                opt_steps=opt_steps,
                robot_type=robot_type,
                # The config default is required, not optional: an override of None or 1.0 means
                # "no extra scaling" and falls back to it, and tiptop.yml ships 0.2 -- passing 1.0
                # here would run every trajectory at five times the intended speed.
                time_dilation_factor=resolve_time_dilation_factor(
                    self.cost_overrides, cfg.robot.time_dilation_factor
                ),
                collision_activation_distance=0.0,
                enable_visualizer=False,
                traj_length_norm=resolve_traj_length_norm(self.cost_overrides),
                grasp_orientation_cost=resolve_grasp_orientation_cost(self.cost_overrides),
                arm_mode=cfg.robot.get("arm_mode", "single"),
                dual_task=cfg.robot.get("dual_task", "parallel"),
                max_motion_refine_attempts=resolve_max_motion_refine_attempts(self.cost_overrides),
            )
            for robot_type in robot_types
        }
        self.config = tamp_configs[robot_types[0]]

        # Headless: there is no DISPLAY, and a viewer would be nobody's window. rerun still wants a
        # recording to log into, so give it one that goes nowhere.
        rr.init("tandem_tiptop", spawn=False)

        _log("warming the planner: cuRobo solvers, SAM-2, the cameras and the robot client")
        self.container = get_demo_container(
            num_particles,
            self.config.coll_n_spheres,
            0.0,
            self.record,
            self.cost_overrides,
            {},
            tamp_configs=tamp_configs,
        )
        self._had_external_cam_2 = self.container.external_cam_2 is not None

        # The save workers must fork from a process whose CUDA context and cameras are already up,
        # so they share the parent's context rather than each building their own ~600MB one. That
        # ordering is also why releasing the cameras means reaping them -- see release_hardware.
        if tiptop_run._executor_pool is None:
            tiptop_run._executor_pool = ProcessPoolExecutor(
                max_workers=4, initializer=tiptop_run._init_pool_worker
            )

        self.loop = asyncio.new_event_loop()
        self.http = self.loop.run_until_complete(self._make_session())
        _log("planner warm")
        return {"robot_types": list(robot_types)}

    # A hard cap on how many perception passes and plans are kept. Only the current one is ever
    # asked for again -- plan() looks up the scene perceive() just returned, execute() the plan
    # plan() just returned -- but one entry pins an Observation with several full-resolution stereo
    # frames and a dict of grasp tensors, so an unbounded dict is gigabytes across a long session in
    # the one process that also holds the CUDA context. A couple of spares cover a retry.
    _KEEP = 3

    def _remember(self, store: dict, key: str, value) -> None:
        """Record an entry, dropping the oldest once there are more than ``_KEEP``."""
        store[key] = value
        while len(store) > self._KEEP:
            store.pop(next(iter(store)))

    async def _make_session(self):
        import aiohttp

        return aiohttp.ClientSession()

    def close(self) -> dict:
        if self.http is not None and self.loop is not None:
            try:
                self.loop.run_until_complete(self.http.close())
            except Exception:
                pass
        self.http = None
        try:
            self._reap_pool()
        except Exception:
            pass
        if self.container is not None:
            self._release_cameras()
            self._release_robot()
        self.container = None
        if self.loop is not None:
            self.loop.close()
            self.loop = None
        return {}

    # ---- hardware custody --------------------------------------------------
    #
    # Upstream deletes all of this: the fork that had it is what this refactor is un-forking, and a
    # planner with no notion of handing its arm to a person has no need of it. tandem does, because
    # a human phase IS a hand-off, so it lives here -- on tandem's side, where it belongs.

    def release_hardware(self) -> dict:
        """Release the robot and every camera, and do not return until they are genuinely free."""
        import time

        serials = self._release_cameras()
        self._reap_pool()
        self._release_robot()
        stationary = self._wait_for_robot_stationary()
        if not stationary:
            _log(
                "could not confirm the arm has stopped moving: either the state port is unreachable "
                "or it is still finishing its last trajectory segment. Do not take the arm yet."
            )
        time.sleep(CAMERA_RELEASE_SETTLE_S)
        return {"cameras": serials, "stationary": stationary}

    def reacquire_hardware(self) -> dict:
        """Take the robot and the cameras back, from wherever the operator left the arm."""
        from concurrent.futures import ProcessPoolExecutor

        from tiptop import tiptop_run
        from tiptop.tiptop_run import get_external_camera, get_external_camera_2, get_hand_camera
        from tiptop.utils import get_robot_client

        if self.container is None:
            raise RuntimeError("the planner is not warm")

        # get_robot_client is cached, and the cached client is the one that was just closed.
        self._clear_robot_client_cache()
        object.__setattr__(self.container, "robot", get_robot_client())

        _log("re-opening the cameras teleop was using")
        object.__setattr__(self.container, "cam", get_hand_camera())
        object.__setattr__(self.container, "external_cam", get_external_camera())
        # This returns None both when a camera is not configured and when it failed to open, so a
        # camera that was recording before the hand-off and is None now would otherwise just vanish
        # from the next episode's videos without a word.
        external_cam_2 = get_external_camera_2()
        if external_cam_2 is None and self._had_external_cam_2:
            _log(
                "the second external camera did not come back after the hand-off; the next legs "
                "will record without it, and merging will drop it from the whole episode"
            )
        object.__setattr__(self.container, "external_cam_2", external_cam_2)

        # Re-fork the save workers now that the cameras are back, for the same reason as at warmup.
        if tiptop_run._executor_pool is None:
            tiptop_run._executor_pool = ProcessPoolExecutor(
                max_workers=4, initializer=tiptop_run._init_pool_worker
            )
        return {}

    def _release_cameras(self) -> list:
        """Close every camera this process holds. Returns their serials.

        A camera is exclusive to one process: the teleop driver opens the same serials we do, so it
        cannot start while our handles are alive. Releasing only the robot connection is not enough.
        """
        serials = []
        for attr in ("cam", "external_cam", "external_cam_2"):
            cam = getattr(self.container, attr, None)
            if cam is None:
                continue
            serial = str(getattr(cam, "serial", "") or "")
            try:
                cam.close()
                if serial:
                    serials.append(serial)
                _log(f"released camera {attr} (s/n {serial or '?'})")
            except Exception:
                _log(f"failed to close camera {attr}; teleop may not be able to open it")
            object.__setattr__(self.container, attr, None)
        return serials

    def _reap_pool(self) -> None:
        """Reap the save workers so they release the camera handles they inherited from us.

        Closing our own camera objects does NOT free the devices: the pool was forked after the
        cameras came up, so its workers still hold them, the SDK reports them as serial 0 /
        NOT AVAILABLE, and the teleop driver's open fails with CAMERA NOT DETECTED. Killing the
        workers is what actually hands the cameras over.

        Deliberately does not block on a save in flight: an operator is waiting for the arm, and the
        worst case is losing one leg's perception debug images, never episode data.
        """
        import time

        from tiptop import tiptop_run

        pool = tiptop_run._executor_pool
        if pool is None:
            return
        _log("stopping the save workers: they hold the cameras until they exit")
        workers = list((getattr(pool, "_processes", None) or {}).values())
        pool.shutdown(wait=False)
        tiptop_run._executor_pool = None
        deadline = time.monotonic() + 5.0
        for proc in workers:
            proc.join(timeout=max(0.0, deadline - time.monotonic()))
            if proc.is_alive():
                _log(f"save worker {proc.pid} still holding the cameras; terminating")
                proc.terminate()
                proc.join(timeout=2.0)

    def _clear_robot_client_cache(self) -> None:
        from tiptop.utils import get_bamboo_client

        get_bamboo_client.cache_clear()
        try:
            from tiptop.ur5.ur5_client import get_ur5_client

            get_ur5_client.cache_clear()
        except ImportError:
            pass

    def _release_robot(self) -> None:
        """Best-effort graceful release of the robot connection.

        The teleop stack and this one talk to the same server and cannot hold it at the same time.
        The cache is dropped either way, so a client built during the hand-off window is a new
        connection rather than the closed one.
        """
        client = getattr(self.container, "robot", None) if self.container is not None else None
        try:
            for name in ("close", "disconnect", "shutdown"):
                fn = getattr(client, name, None)
                if callable(fn):
                    try:
                        fn()
                        _log(f"released {type(client).__name__} via .{name}()")
                    except Exception:
                        _log(f"{type(client).__name__}.{name}() raised while releasing")
                    return
            if client is not None:
                _log(
                    f"{type(client).__name__} exposes no close/disconnect/shutdown; relying on GC. "
                    "If teleop cannot take the arm, this is the first place to look."
                )
        finally:
            self._clear_robot_client_cache()

    def _wait_for_robot_stationary(
        self,
        velocity_threshold: float = 0.02,
        settle_seconds: float = 0.4,
        timeout_seconds: float = 15.0,
        poll_hz: float = 30.0,
    ) -> bool:
        """Block until measured joint velocities confirm the arm has stopped, or give up.

        Closing our connection does not mean the arm has stopped: the controller is still finishing
        whatever trajectory segment it was mid-way through. Telling an operator "the arm is yours"
        before that has happened means handing them a moving arm.

        Uses its own short-lived connection to the state port rather than the robot client, so it
        keeps working after that client has been torn down. Returns False on timeout, meaning "could
        not confirm" -- which the caller must treat as a warning, never as "probably fine".
        """
        import time

        import msgpack
        import numpy as np
        import zmq
        from tiptop.config import tiptop_cfg

        host = tiptop_cfg().robot.host
        port = int(os.environ.get("TIPTOP_STATE_PORT", 5557))
        ctx = zmq.Context()
        sock = None

        def _connect():
            nonlocal sock
            if sock is not None:
                sock.close(linger=0)
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, 300)  # ms; a timed-out REQ socket is unusable
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(f"tcp://{host}:{port}")

        _connect()
        req = msgpack.packb({"command": "get_robot_state"})
        deadline = time.monotonic() + timeout_seconds
        stationary_since = None
        try:
            while time.monotonic() < deadline:
                now = time.monotonic()
                dq = None
                try:
                    sock.send(req)
                    reply = msgpack.unpackb(sock.recv(), raw=False)
                    data = reply.get("data") if isinstance(reply, dict) and reply.get("success") else None
                    if data:
                        dq = np.asarray(data.get("dq", []), dtype=np.float32).reshape(-1)
                except zmq.Again:
                    _connect()
                except Exception:
                    _connect()

                if dq is not None and dq.size and float(np.max(np.abs(dq))) < velocity_threshold:
                    if stationary_since is None:
                        stationary_since = now
                    elif now - stationary_since >= settle_seconds:
                        return True
                else:
                    stationary_since = None  # no reading, or still moving: restart the settle window
                time.sleep(max(0.0, 1.0 / poll_hz))
            return False
        finally:
            if sock is not None:
                sock.close(linger=0)
            ctx.term()

    def capture_frame(self, *, camera: str = "external") -> dict:
        """One frame, written to a PNG whose path is returned.

        A third-person view by default: this is what phase verification looks at, and after a
        hand-off the arm is wherever the operator left it, so a wrist camera points nowhere useful.
        """
        import tempfile

        import numpy as np
        from PIL import Image
        from tiptop.tiptop_run import perception_camera

        if self.container is None:
            raise RuntimeError("the planner is not warm")
        if camera == "external":
            cam = self.container.external_cam or perception_camera(self.container)
        elif camera == "hand":
            cam = self.container.cam
        else:
            cam = perception_camera(self.container)
        if cam is None:
            raise RuntimeError(f"the {camera} camera is not open, so no frame can be captured")

        rgb = cam.read_camera().rgb
        handle, path = tempfile.mkstemp(prefix="tandem-frame-", suffix=".png")
        os.close(handle)
        Image.fromarray(np.asarray(rgb).astype(np.uint8)).save(path)
        return {"path": path}

    def home(self) -> dict:
        """Park the arm.

        The branch mirrors the planner's own manual `home` command. It matters: `home_all_arms`
        walks the bimanual arm list and skips anything not keyed `bimanual_yam_<side>` in
        `container.solvers`, so on every single-arm robot tandem supports it does nothing at all and
        returns cleanly -- an operator would be told the arm was parked while it stood where the
        last plan left it.

        The gripper is deliberately NOT opened, unlike the planner's own pre-rollout reset: nothing
        here can know the arm is not holding something a human just handed it.
        """
        from tiptop.config import tiptop_cfg
        from tiptop.motion_planning import go_to_dual_home, go_to_home
        from tiptop.tiptop_run import configured_arms, home_all_arms

        if self.container is None:
            raise RuntimeError("the planner is not warm")
        cfg = tiptop_cfg()
        arms = configured_arms()
        if arms:
            home_all_arms(self.container)
        elif cfg.robot.type == "bimanual_yam_dual":
            go_to_dual_home(
                time_dilation_factor=cfg.robot.time_dilation_factor, motion_gen=self.container.motion_gen
            )
        else:
            go_to_home(
                time_dilation_factor=cfg.robot.time_dilation_factor, motion_gen=self.container.motion_gen
            )
        return {}

    # ---- the sub-goal cycle ------------------------------------------------

    def perceive(self, *, task_hint: str, save_dir: str, reset_arm: bool = True) -> dict:
        """Look at the workspace and report what is in it.

        ``task_hint`` steers DETECTION only. The goal arrives separately, in ``plan`` -- which is the
        whole difference from an ordinary tiptop rollout, where the instruction is translated into a
        goal by the same call that detects the objects.

        ``reset_arm`` parks the arm first, the way an ordinary rollout does. tandem turns it OFF for a
        phase resumed after a hand-off: the arm is where a person left it, quite possibly holding
        something, and driving it home would undo their step. The gripper is never opened here for the
        same reason, which is where this departs from the planner's own pre-rollout reset.
        """
        from pathlib import Path

        import rerun as rr
        from tiptop.config import tiptop_cfg
        from tiptop.motion_planning import go_to_capture
        from tiptop.tiptop_run import capture_live_observation, run_perception

        if self.container is None:
            raise RuntimeError("the planner is not warm")
        directory = Path(save_dir)
        directory.mkdir(parents=True, exist_ok=True)

        cfg = tiptop_cfg()
        if reset_arm:
            self.home()
        # A wrist camera only points at the workspace from the capture pose, and its world pose is
        # known only through forward kinematics -- so perceiving from wherever the arm happens to be
        # images the wrong thing AND fits the table plane to it. Not optional, and done even when the
        # arm was not reset, because the alternative is a scene nobody asked about.
        if self.container.perception_cam_key == "hand":
            go_to_capture(
                time_dilation_factor=cfg.robot.time_dilation_factor, motion_gen=self.container.motion_gen
            )
        elif not reset_arm:
            _log("perceiving without parking the arm first; it is where the last step left it")

        rr.init("tandem_tiptop", recording_id=directory.name, spawn=False)
        observation = capture_live_observation(self.container)

        # The goal is tandem's, and it arrives in plan(). Handing run_perception an EMPTY goal is not
        # a detail: left to itself it ends by building a cuTAMP environment from the goal the detector
        # translated out of `task_hint`, and that construction REJECTS a goal naming an object it did
        # not detect. So a hallucinated object, in a goal tandem never asked for and immediately
        # discards, would fail the whole phase. The detected atoms are kept for reporting.
        detected: list = []

        def goal_builder(_processed_scene, detected_atoms):
            detected.extend(detected_atoms or [])
            return [], None

        _, _, processed_scene, _ = self.loop.run_until_complete(
            run_perception(
                self.http,
                observation,
                task_hint,
                directory,
                depth_estimator=self.container.depth_estimator,
                log_to_rerun=False,
                goal_builder=goal_builder,
            )
        )

        scene_id = uuid.uuid4().hex[:12]
        self._remember(
            self.scenes,
            scene_id,
            {
                "observation": observation,
                "scene": processed_scene,
                "detected_atoms": list(detected),
                "save_dir": directory,
            },
        )
        return {
            "scene_id": scene_id,
            "object_labels": sorted(processed_scene.object_meshes.keys()),
            "table_label": processed_scene.table_cuboid.name,
            # What the planner would have inferred on its own, before tandem pins the split. Reported
            # so a caller can see the two differ; tandem's own answer is what plan() is given.
            "surface_labels": sorted(
                {
                    a["args"][1]
                    for a in detected
                    if a.get("predicate") == "on" and len(a.get("args", [])) == 2
                }
            ),
            "rgb_path": self._save_rgb(observation, directory),
            # What the planner's own translator made of `task_hint`. Discarded before the
            # environment was built (see the goal_builder above), but handed back so a caller
            # with no plan of its own can ask for exactly the goal an ordinary rollout would
            # have used -- the same translator, the same atoms, and no extra model call.
            "detected_goal": [
                {"predicate": a.get("predicate"), "args": list(a.get("args") or [])}
                for a in detected
                if isinstance(a, dict) and a.get("predicate")
            ],
        }

    def _save_rgb(self, observation, directory) -> str:
        import numpy as np
        from PIL import Image

        path = directory / "perception_rgb.png"
        try:
            Image.fromarray(np.asarray(observation.frame.rgb).astype(np.uint8)).save(path)
        except Exception:
            return ""
        return str(path)

    def plan(self, *, scene_id: str, goal: list, surfaces: list, save_dir: str) -> dict:
        """Find a motion plan achieving ``goal`` in an already-perceived scene.

        ``goal`` is the ``{"predicate", "args"}`` form ``create_tamp_environment`` already consumes,
        so a tandem phase goes through exactly the same unknown-object rejection and environment
        construction as a goal tiptop translated for itself. There is no second code path, and no
        change to tiptop to have one.
        """
        from pathlib import Path

        from tiptop.motion_planning import resolve_trace_cfg
        from tiptop.planning import run_planning, save_tiptop_plan, serialize_plan
        from tiptop.tiptop_run import create_tamp_environment

        entry = self.scenes.get(scene_id)
        if entry is None:
            raise RuntimeError(f"unknown scene {scene_id!r}; perceive() again before planning")
        directory = Path(save_dir)
        directory.mkdir(parents=True, exist_ok=True)
        processed_scene = entry["scene"]
        observation = entry["observation"]

        env, all_surfaces = create_tamp_environment(
            processed_scene.object_meshes,
            processed_scene.table_cuboid,
            list(goal),
            True,
            extra_surface_labels=set(surfaces),
        )
        cutamp_plan, planning_seconds, failure_reason = run_planning(
            env,
            self.config,
            q_init=observation.q_init,
            ik_solver=self.container.ik_solver,
            grasps=processed_scene.grasps,
            motion_gen=self.container.motion_gen,
            all_surfaces=all_surfaces,
            # A directory per sub-goal: two run_planning calls into one experiment directory collide
            # inside cuTAMP's own logger, and a phase-planned task makes several per episode.
            experiment_dir=directory / "cutamp",
            cost_overrides=self.cost_overrides,
        )
        if cutamp_plan is None:
            return {
                "ok": False,
                "failure_reason": failure_reason or "no plan found",
                "planning_seconds": planning_seconds,
            }

        plan_path = directory / "tiptop_plan.json"
        save_tiptop_plan(
            serialize_plan(cutamp_plan, observation.q_init, trace_cfg=resolve_trace_cfg(self.cost_overrides)),
            plan_path,
        )
        handle = uuid.uuid4().hex[:12]
        self._remember(
            self.plans,
            handle,
            {
                "plan": cutamp_plan,
                "plan_path": plan_path,
                "scene_id": scene_id,
                "env": env,
                "grasps": processed_scene.grasps,
                "save_dir": directory,
            },
        )
        return {
            "ok": True,
            "planning_seconds": planning_seconds,
            "plan_handle": handle,
            "artifacts": {"plan": str(plan_path)},
        }

    def execute(self, *, plan_handle: str, leg: dict, save_dir: str) -> dict:
        """Run the plan on the robot and record it as one leg of a trajectory."""
        from pathlib import Path

        from tiptop.execute_plan import execute_cutamp_plan
        from tiptop.recording import save_run_outputs
        from tiptop.tiptop_run import _execute_plan_recorded

        entry = self.plans.get(plan_handle)
        if entry is None:
            raise RuntimeError(f"unknown plan {plan_handle!r}")
        directory = Path(save_dir)
        directory.mkdir(parents=True, exist_ok=True)

        n_frames = 0
        failure = None
        try:
            if not self.execute_plans:
                _log("execute_plan is off: the plan was found and saved but not run")
            elif self.record:
                n_frames = _execute_plan_recorded(
                    self.container,
                    entry["plan"],
                    entry["plan_path"],
                    directory,
                    leg.get("instruction") or "",
                )
            else:
                execute_cutamp_plan(entry["plan"], client=self.container.robot)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            _log("execution failed: " + failure)
        finally:
            try:
                save_run_outputs(directory, entry["env"], entry["grasps"])
            except Exception:
                _log("could not save run outputs")

        # tandem mints the trajectory id, because only tandem sees both the planner's legs and the
        # teleop ones. Upstream's dump_raw_episode has no parameter for it and no notion of a leg, so
        # it is stamped here, after the fact -- which is also why it is stamped even when execution
        # failed: a leg that exists on disk and is not stamped files as an episode of its own.
        stamped = self._stamp_meta(directory, leg)
        return {
            "ok": failure is None,
            "failure_reason": failure,
            "n_frames": n_frames,
            "rollout_dir": str(directory),
            "stamped": stamped,
        }

    def _stamp_meta(self, directory, leg: dict) -> bool:
        """Write the leg's identity into ``_meta.json``, where merging looks for it."""
        meta_path = directory / "_meta.json"
        if not meta_path.is_file():
            return False
        try:
            meta = json.loads(meta_path.read_text())
            meta["trajectory_id"] = leg.get("trajectory_id")
            meta["segment_source"] = leg.get("segment_source") or "tamp"
            if leg.get("phase_index") is not None:
                meta["phase_index"] = leg["phase_index"]
                meta["n_phases"] = leg.get("n_phases")
                meta["phase_description"] = leg.get("phase_description") or ""
            meta_path.write_text(json.dumps(meta, indent=2))
            return True
        except Exception:
            _log(f"could not stamp {meta_path} with the trajectory id; the leg will not merge")
            return False

    # ---- verbs -------------------------------------------------------------

    def capabilities(self) -> dict:
        """What this planner can be asked for, read out of the live cuTAMP domain.

        tandem declares the same facts statically (it has no cuTAMP to ask), so this is the copy that
        is true by construction -- worth having as the thing to compare a stale declaration against.
        """
        from cutamp.tamp_domain import all_tamp_fluents, all_tamp_operators, get_initial_state

        initial = get_initial_state(movables=["m"], surfaces=["s"])
        return {
            "name": "tiptop",
            "reserved_predicate_names": sorted(f.name for f in all_tamp_fluents),
            "achievable_predicates": sorted(
                {f.name for op in all_tamp_operators for f in op.add_effects} | {a.name for a in initial}
            ),
            "operators": [op.name for op in all_tamp_operators],
        }


def main() -> int:
    sidecar = Sidecar()
    _emit({"ready": True, "pid": os.getpid(), "verbs": list(VERBS)})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError:
            _log(f"ignoring an unparseable request: {line[:200]}")
            continue

        verb = request.get("verb")
        request_id = request.get("id")
        args = request.get("args") or {}
        if verb == "quit":
            break
        if verb not in VERBS:
            _emit({"id": request_id, "ok": False, "error": f"unknown verb {verb!r}"})
            continue
        try:
            result = getattr(sidecar, verb)(**args)
            _emit({"id": request_id, "ok": True, "result": result})
        except Exception as exc:
            # The parent turns this into a BackendError an operator reads, so it says what failed in
            # words, with the traceback beside it in the session log rather than inside the message.
            _log(traceback.format_exc())
            _emit({"id": request_id, "ok": False, "error": f"{verb} failed -- {type(exc).__name__}: {exc}"})

    try:
        sidecar.close()
    except Exception:
        _log(traceback.format_exc())
    return 0


if __name__ == "__main__":
    sys.exit(main())
