# ruff: noqa
"""Teleoperation + capture driver: what a person drives the arm with for a human leg.

Drives the arm from the **VR** (Oculus) controller — same as DROID's ``scripts/main.py`` — and captures
each episode over an events file plus the stdin protocol below, which ``tandem/teleop/child.py``'s
``TeleopChild`` answers. The operator moves the end-effector with the controller (6-DOF -> Cartesian
velocity) and works the gripper with its trigger; the operator decides when each episode ends (from the
UI or the terminal).
Every episode is written in the raw episode format (``teleop/raw_episode.py``) so the export
(``tandem.export``, ``export/build.py``) builds it exactly like a tamp episode.

Unlike DROID's ``scripts/main.py`` + Tk GUI VR flow, session control (start / end / discard / label) is
driven over the stdin protocol by tandem — the controller only moves the arm.

Nothing is installed on the NUC — this is a drop-in for the existing PC-side teleop (still talks to the
NUC's ``run_server.py`` over the same ``StableRobotEnv`` -> ServerInterface path); only the controller
changes. VR uses ``droid.controllers.oculus_controller.VRPolicy``.

Protocol (stdin lines written by ``tandem/teleop/child.py``, launched once per leg by
``executors/teleop.py``):
  {"cmd":"start"}   begin an episode (at the task prompt)
  {"cmd":"end"}     stop + SAVE the current episode        {"cmd":"discard"} stop + throw it away
  {"cmd":"home"}    send the arm to its home pose (only at the task prompt, between episodes)
  {"cmd":"end_and_quit"}  stop + SAVE, then finish the session -- what "return control to TAMP"
                    sends mid-recording, since a bare "q" there DISCARDS the episode
  y | n             label the saved episode success/failure   q  finish the session

With --trajectory-id (a tamp->teleop hand-off), episodes are legs of that tamp trajectory: they are
stamped with the id, left unlabeled in eval/, and never prompt for y/n. See
docs/ADDING_A_HUMAN_EXECUTOR.md (the TeleopExecutor) and ``teleop/child.py`` for the hand-off.
With --phase-index / --n-phases / --phase-description (a human phase of a phase-planned task), each
leg's _meta.json also says which phase it records, under the keys the TAMP legs use.

The arm also returns home automatically at the END of every trajectory (after save / discard / error),
so it is clear of the workspace before the next episode; the home motion is not part of the recording.

Events (appended to $TELEOP_EVENTS_FILE): session_start, awaiting_task, rollout_start, rollout_saved,
awaiting_label, labeled, rollout_aborted, homing, homed, session_end.

Run under the DROID conda env (same as DROID's VR ``scripts/main.py``).
"""

import dataclasses
import datetime
import json
import os
import select
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Shared raw-episode helpers, from this same directory. This module runs under the DROID
# environment's interpreter rather than tandem's, so the imports are made path-relative
# instead of package-relative -- there is no `tandem` on that interpreter's sys.path.
# Neither import touches robot or policy code.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raw_episode import emit, write_meta, write_robot_state_npz, _write_video  # noqa: E402

CONTROL_HZ = 15  # matches StableRobotEnv.control_hz + the LeRobot build FPS
# The env's action_dict["joint_velocity"] is the IK-commanded joint velocity ALREADY NORMALIZED to
# [-1,1] (see droid/franka/robot.py::create_action_dict + robot_ik_solver.py) -- exactly the DROID /
# lerobot/droid_1.0.1 action convention. We record it AS-IS (no scaling); the export
# (export/build.py::clip_joint_velocity) just CLIPS it
# for the teleop kind and does NOT re-normalize it (unlike the rad/s tamp/plan path). So there is no
# double normalization: the IK value is normalized once, at its source (the robot's IK solver).
EXTERNAL_CAM, EXTERNAL_CAM_2, HAND_CAM = "external_cam.mp4", "external_cam_2.mp4", "hand_cam.mp4"

# SIGINT (force-stop, sent to the DRIVER pid only) discards the in-flight episode and halts the arm,
# keeping the session warm (the ZED background processes survive because the signal isn't sent to the
# group). SIGTERM (graceful-stop escalation) exits cleanly so `finally` halts + closes the env.
_ABORT_EP = {"v": False}


def _install_signal_handlers():
    signal.signal(signal.SIGINT, lambda *_: _ABORT_EP.__setitem__("v", True))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))


# --------------------------------------------------------------------------- #
# stdin line protocol (shared buffer across blocking + non-blocking reads)      #
# --------------------------------------------------------------------------- #
_buf = ""


def _parse(line: str):
    line = line.strip()
    if not line:
        return {}
    if line in ("q", "y", "n"):
        return line
    try:
        return json.loads(line)
    except ValueError:
        return {}


def _read_line_blocking():
    """Block until one full stdin line, returning the parsed command (or 'q' at EOF)."""
    global _buf
    while "\n" not in _buf:
        chunk = os.read(0, 4096)
        if not chunk:
            return "q"
        _buf += chunk.decode(errors="ignore")
    line, _buf = _buf.split("\n", 1)
    return _parse(line)


def _poll_line():
    """Return the next buffered stdin command without blocking, else None ('q' at EOF)."""
    global _buf
    if "\n" not in _buf:
        r, _, _ = select.select([0], [], [], 0)
        if r:
            chunk = os.read(0, 4096)
            if not chunk:
                return "q" if not _buf else None
            _buf += chunk.decode(errors="ignore")
    if "\n" not in _buf:
        return None
    line, _buf = _buf.split("\n", 1)
    return _parse(line)


class VRPolicyDriver:
    """Wraps the DROID Oculus ``VRPolicy`` as ``forward(obs) -> (action, gripper_target)``. The VR policy
    already produces the full 7-DOF Cartesian-velocity action (including gripper velocity from the
    trigger); we surface the continuous trigger value as the gripper target, which is recorded as
    ``cmd_gripper``. Only the arm is driven from VR — session control (start/end/label) stays on the
    browser stdin protocol."""

    def __init__(self, controller: str):
        from droid.controllers.oculus_controller import VRPolicy  # lazy: pulls in oculus_reader
        self.vr = VRPolicy(right_controller=(controller != "left"))

    def forward(self, obs):
        action, info = self.vr.forward(obs, include_info=True)
        gripper_target = float(info.get("target_gripper_position", 0.0))
        return np.asarray(action, dtype=np.float64).reshape(-1), gripper_target

    def reset(self):
        """Re-anchor the VR origin to the arm's current pose. Called right before each episode records
        (the arm is home by then). VRPolicy tracks the robot/controller pose at the start of a motion and
        drives toward that offset; without a fresh anchor the origin stays pinned to the previous
        episode's pose and the arm springs back there. Reusing the same call site as the canonical
        collect_trajectory, which invokes controller.reset_state() immediately before its record loop —
        doing this at record start (not at home time) also keeps the reader state fresh so the new
        episode actually picks up controller input."""
        self.vr.reset_state()

    def wait_ready(self, timeout=8.0):
        """Block until the Oculus reader is actually streaming controller poses, or ``timeout`` elapses.
        The reader launches the headset APK and streams poses over ADB/logcat; the first frames lag a
        second or two after connect, and the headset stops streaming when it sleeps (taken off / idle).
        Starting an episode in that window leaves VRPolicy with empty poses, so ``forward`` returns an
        all-zero action and the arm never moves — the intermittent "controller not recognized". Waiting
        on ``_state['poses']`` confirms the full read thread -> VRPolicy pipeline is live. Returns True
        once poses arrive, False on timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.vr._state.get("poses"):
                return True
            time.sleep(0.05)
        return False

    def close(self):
        pass


# --------------------------------------------------------------------------- #
# Camera extraction (StableRobotEnv images are BGRA, keyed by "{serial}_left")   #
# --------------------------------------------------------------------------- #
def _rgb(img):
    return None if img is None else np.ascontiguousarray(img[..., :3][..., ::-1])  # BGRA -> RGB


def _extract_cameras(obs, ext_id, ext2_id, hand_id):
    """Return (external, external_2, wrist) RGB frames by matching configured ZED serials."""
    images = obs.get("image") or {}
    if not images:
        raise RuntimeError(
            "No camera frames from the DROID env — the ZED cameras were not read. Check they are "
            "connected and NOT already open in another process (a running tiptop-run / capture holds "
            "them exclusively), and that TIPTOP_EXTERNAL_CAMERA_ID / TIPTOP_HAND_CAMERA_ID are set."
        )

    def pick(serial):
        if not serial:
            return None
        for key, val in images.items():
            if serial in key and "left" in key:
                return _rgb(val)
        return None

    return pick(ext_id), pick(ext2_id), pick(hand_id)


# --------------------------------------------------------------------------- #
# One teleop episode                                                            #
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Args:
    events_file: str = ""
    output_root: str = ""       # runs/<workspace>/teleop/<name>; episodes nest under eval/<ts> (staging)
    instruction: str = ""
    config_id: str = "teleop/teleop"
    controller: str = "right"       # VR hand: "right" or "left"
    external_camera_id: str = ""
    external_2_camera_id: str = ""
    hand_camera_id: str = ""
    keep_pose: bool = False  # skip the startup reset-to-home move; arm stays wherever it already is
    # Set by a tamp->teleop hand-off: this session's episodes are LEGS of that tamp trajectory, not
    # episodes in their own right. They are stamped with the id and left UNLABELED in eval/ -- the
    # success/failure verdict belongs to the whole trajectory and is given once, by tandem's session
    # when the trial ends, after which core/episodes.py::merge_trajectory (via core/merge.py) joins
    # every leg into one episode.
    trajectory_id: str = ""
    # Set by tandem when this session records a HUMAN PHASE of a phase-planned task: which phase φ_k
    # (0-based) out of how many, and the phase's own words. Stamped into every saved leg's _meta.json
    # under the same keys the TAMP backend writes for its legs, so the merged episode's segments[] map
    # each leg to its phase. Left unset, the keys are simply absent -- a plain teleop episode, or a
    # hand-off from before phase planning, looks exactly as it always did. Optional (not ""/-1
    # sentinels) because 0 is a real phase index. typing.Optional, not `int | None`: this runs under
    # the DROID env's interpreter, which may predate 3.10.
    phase_index: Optional[int] = None
    n_phases: Optional[int] = None
    phase_description: Optional[str] = None


def _halt(env):
    """Command zero Cartesian velocity so the arm stops (velocity control holds the last command)."""
    try:
        env.step(np.zeros(7, dtype=np.float64))
    except Exception as e:  # noqa: BLE001
        print(f"[teleop] halt failed: {e}", flush=True)


def _go_home(env, events):
    """Return the arm to the home joint configuration (a blocking joint move, via ``env.reset()``). Runs
    at the end of every trajectory and on the go-home command from the UI. Guarded so a failed reset
    surfaces an event but never tears down the warm session. The home motion is NOT recorded — it runs
    after the episode's frames are already captured.

    Note: the controller is re-anchored (``policy.reset()``) at the START of the next episode, not here.
    Re-anchoring at home time would clear the VR reader state (poses / movement flag) and then sit in
    that cleared state through the label wait, which drops the next episode's controller input. The
    canonical ``collect_trajectory`` likewise resets the controller immediately before its record loop."""
    emit(events, "homing")
    try:
        env.reset()
    except Exception as e:  # noqa: BLE001
        print(f"[teleop] go home failed: {e}", flush=True)
        emit(events, "error", message=f"go home failed: {e}")
        return
    emit(events, "homed")


def record_episode(env, policy, ep_dir, args, events):
    """Teleop until the operator ends/discards it. Returns ``(n_frames_or_None, quit)`` — n_frames on
    save (None if discarded), and ``quit`` True when the operator asked to finish the whole session."""
    ext_frames, ext2_frames, wrist_frames = [], [], []
    joint_log, grip_log, ft_log, cmd_g_log = [], [], [], []
    cmd_jv_log, cmd_jp_log = [], []  # IK-COMMANDED joint velocity / target position from env.step
    _ABORT_EP["v"] = False  # clear any SIGINT that arrived while parked at the prompt
    ended, quit_session = None, False
    try:
        while True:
            cmd = _poll_line()
            if cmd == "q":                 # finish the session (discards the in-flight episode)
                ended, quit_session = "discard", True
                break
            if isinstance(cmd, dict) and cmd.get("cmd") == "discard":
                ended = "discard"
                break
            if isinstance(cmd, dict) and cmd.get("cmd") == "end":
                ended = "end"
                break
            if isinstance(cmd, dict) and cmd.get("cmd") == "end_and_quit":
                # "Return control to TAMP" pressed mid-recording. SAVE, then leave the loop: a bare
                # "q" here would discard the operator's demonstration, and during a hand-off that
                # demonstration is a segment of the trajectory TAMP is going to finish.
                ended, quit_session = "end", True
                break
            if _ABORT_EP["v"]:             # force-stop (SIGINT): discard this episode, stay warm
                ended = "discard"
                break

            t0 = time.time()
            obs = env.get_observation()
            ext, ext2, wrist = _extract_cameras(obs, args.external_camera_id, args.external_2_camera_id, args.hand_camera_id)
            if ext is None or wrist is None:
                raise RuntimeError("external and/or wrist camera returned no frame (check the serial env vars)")
            ext_frames.append(ext)
            wrist_frames.append(wrist)
            ext2_frames.append(ext2)

            rs = obs["robot_state"]
            joint_log.append(np.asarray(rs["joint_positions"], dtype=np.float32).reshape(-1))
            grip_log.append(float(np.asarray(rs["gripper_position"]).reshape(-1)[0]))
            ft_log.append(time.time())

            action, gtarget = policy.forward(obs)
            cmd_g_log.append(1.0 if gtarget > 0.5 else 0.0)
            # env.step returns the DROID action_dict. joint_velocity is the IK command (operator
            # cartesian velocity -> IK joint_delta / max_joint_delta), NORMALIZED to [-1,1] exactly like
            # lerobot/droid_1.0.1; joint_position is the IK commanded target (measured + joint_delta).
            # Capturing these is the whole point: finite-differencing the MEASURED joints instead gives
            # the ACHIEVED motion, which undertracks the command ~4-5x and puts teleop on a different
            # scale than DROID.
            info = env.step(action)
            if not isinstance(info, dict) or "joint_velocity" not in info or "joint_position" not in info:
                raise RuntimeError(
                    "env.step did not return the DROID action_dict with joint_velocity/joint_position; "
                    "cannot capture the IK-commanded joint velocity"
                )
            cmd_jv_log.append(np.asarray(info["joint_velocity"], dtype=np.float32).reshape(7))
            cmd_jp_log.append(np.asarray(info["joint_position"], dtype=np.float32).reshape(7))

            dt = time.time() - t0
            if dt < 1.0 / CONTROL_HZ:
                time.sleep(1.0 / CONTROL_HZ - dt)
    finally:
        _halt(env)  # always stop the arm when recording ends (end / discard / error / stop)

    n = min(len(ext_frames), len(wrist_frames), len(joint_log), len(cmd_g_log), len(cmd_jv_log), len(cmd_jp_log))
    if ended == "discard" or n < 2:
        return None, quit_session

    ep_dir = Path(ep_dir)
    _write_video(ext_frames[:n], ep_dir / EXTERNAL_CAM)
    _write_video(wrist_frames[:n], ep_dir / HAND_CAM)
    cameras = {"exterior_image_1_left": EXTERNAL_CAM, "wrist_image_left": HAND_CAM}
    if all(f is not None for f in ext2_frames[:n]):
        _write_video(ext2_frames[:n], ep_dir / EXTERNAL_CAM_2)
        cameras["exterior_image_2_left"] = EXTERNAL_CAM_2

    jp = np.stack(joint_log[:n])                                  # measured joints [n,7]
    frame_time = np.asarray(ft_log[:n], dtype=np.float64)
    # cmd_joint_velocity / cmd_joint_position are the IK COMMAND captured from env.step's action_dict --
    # the SAME quantities DROID records (droid/franka/robot.py create_action_dict). joint_velocity is
    # ALREADY normalized to [-1,1] (the DROID action convention), so we store it AS-IS; the export
    # (export/build.py::clip_joint_velocity) only clips it for the teleop kind (it does NOT divide by 3 -- that is the rad/s tamp/plan path).
    # cmd_joint_position is the IK commanded target (radians). This REPLACES the old finite-difference
    # of the MEASURED joints, which captured the achieved (undertracked) motion ~4-5x below the command.
    cmd_jv = np.stack(cmd_jv_log[:n]).astype(np.float32)          # IK command, normalized [-1,1]
    cmd_jp = np.stack(cmd_jp_log[:n]).astype(np.float32)          # IK commanded target joint positions
    write_robot_state_npz(
        ep_dir / "robot_state.npz",
        joint_position=jp,
        gripper_position=np.asarray(grip_log[:n], dtype=np.float32),
        cmd_joint_position=cmd_jp,
        cmd_joint_velocity=cmd_jv,
        cmd_gripper=np.asarray(cmd_g_log[:n], dtype=np.float32),  # binary target (0 open / 1 closed)
        frame_time=frame_time,
    )
    write_meta(
        ep_dir / "_meta.json",
        instruction=args.instruction, n_frames=n, config_id=args.config_id,
        timestamp=ep_dir.name, cameras=cameras,
        record_start=float(frame_time[0]), record_stop=recording_window_stop(frame_time),
        trajectory_id=args.trajectory_id or None, segment_source="teleop",
        phase_index=args.phase_index, n_phases=args.n_phases, phase_description=args.phase_description,
    )
    return n, quit_session


def recording_window_stop(frame_time):
    """``record_stop`` for a leg whose clip holds exactly one image per state frame.

    ``record_start``/``record_stop`` are a recording WINDOW, not the first and last timestamps: the
    export and the merge place state frame t at camera frame ``(t - record_start) * n_cam /
    (record_stop - record_start)`` -- the map a TAMP leg's camera window gets -- and n frames span n
    frame periods. Stopping the window at the LAST frame's timestamp covered only n-1 of them, which
    stretched every index by n/(n-1): from the middle of each teleop leg on, a state frame was paired
    with the NEXT image, both in a standalone export (which rounds) and in a merged episode's
    video_time. Ending the window one mean frame period after the last frame makes the map the
    identity this loop actually recorded (frame i -> image i) at whatever rate the loop really ran.
    """
    frame_time = np.asarray(frame_time, dtype=np.float64)
    if len(frame_time) < 2:
        return float(frame_time[-1])
    period = (float(frame_time[-1]) - float(frame_time[0])) / (len(frame_time) - 1)
    return float(frame_time[-1]) + period


def main(args: Args):
    events = args.events_file
    # Camera serials: explicit flag > TIPTOP_*_CAMERA_ID env var > the lab's usual ZED (the default
    # baked into droid.misc.parameters, the same source tiptop.yml resolves). Falling back to the lab
    # default — instead of "" — means teleop works even when the server's environment doesn't export
    # these vars, matching how tiptop/eval find their cameras. Serials that aren't connected simply
    # don't match a frame (external_2 is optional), so this never forces a missing camera.
    from droid.misc.parameters import hand_camera_id, varied_camera_1_id, varied_camera_2_id
    args.external_camera_id = args.external_camera_id or varied_camera_1_id
    args.external_2_camera_id = args.external_2_camera_id or varied_camera_2_id
    args.hand_camera_id = args.hand_camera_id or hand_camera_id
    output_root = Path(args.output_root)
    (output_root / "eval").mkdir(parents=True, exist_ok=True)
    _install_signal_handlers()

    emit(events, "session_start")
    env = None
    try:
        controller = (args.controller or "right").strip().lower()
        policy = VRPolicyDriver(controller)  # raises (headset not reachable) -> error exit
        print(f"[teleop] VR Oculus ({controller} controller)", flush=True)
        from droid.stable_camera_env import StableRobotEnv  # lazy: DROID env only
        env = StableRobotEnv(
            action_space="cartesian_velocity", gripper_action_space=None, do_reset=not args.keep_pose
        )
        if args.keep_pose:
            print("[teleop] --keep-pose: skipped startup reset, arm stays at its current pose", flush=True)
        print("[teleop] created the DROID env", flush=True)

        while True:
            emit(events, "awaiting_task")
            cmd = _read_line_blocking()
            if cmd == "q":
                break
            if isinstance(cmd, dict) and cmd.get("cmd") == "home":
                _go_home(env, events)  # manual go-home from the UI (only at the prompt, arm idle)
                continue
            if not (isinstance(cmd, dict) and cmd.get("cmd") == "start"):
                continue

            # Re-anchor the controller to the arm's CURRENT pose right before recording — this is where
            # the canonical collect_trajectory calls controller.reset_state(). For VR this pins the
            # tracking origin to where the arm is now (home, since each trajectory ends homed -- unless
            # --keep-pose, in which case it's wherever the previous controller left it), so the arm
            # neither springs back to the previous episode's pose nor ignores the controller.
            policy.reset()

            # Don't start recording until the controller is actually streaming — otherwise the first
            # frames capture a dead arm and the operator sees "controller not recognized". If it never
            # comes up (headset asleep / off), bail back to the prompt with a clear message.
            if not policy.wait_ready():
                emit(events, "error", message=(
                    "VR controller not detected — put the headset on and wake the controller, then "
                    "press Start again."))
                continue

            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            ep_dir = output_root / "eval" / ts   # staging bucket; the label moves it to success/failure
            ep_dir.mkdir(parents=True, exist_ok=True)
            emit(events, "rollout_start", dir=str(ep_dir), instruction=args.instruction)
            try:
                n, quit_session = record_episode(env, policy, ep_dir, args, events)
            except Exception as e:  # noqa: BLE001
                _halt(env)
                shutil.rmtree(ep_dir, ignore_errors=True)
                emit(events, "error", message=f"episode failed: {e}")
                if not args.keep_pose:
                    _go_home(env, events)  # every trajectory ends with the arm back home, even a failed one
                continue
            if n is None:
                shutil.rmtree(ep_dir, ignore_errors=True)
                emit(events, "rollout_aborted", dir=str(ep_dir))
                if not args.keep_pose:
                    _go_home(env, events)  # home after a discarded trajectory
                if quit_session:
                    break
                continue
            emit(events, "rollout_saved", dir=str(ep_dir), n_frames=n)
            if not args.keep_pose:
                _go_home(env, events)  # home after a saved trajectory, while the operator labels it

            if args.trajectory_id:
                # A hand-off leg is not a standalone episode, so there is nothing to rate here: it
                # stays unlabeled in eval/ until tandem's session files the trajectory it belongs to
                # and core/episodes.py::merge_trajectory folds it in. Prompting would also stall
                # "Return control to TAMP", which is the operator's actual next action.
                print(f"[teleop] hand-off leg saved unlabeled in {ep_dir} (trajectory {args.trajectory_id})",
                      flush=True)
                if quit_session:
                    break
                continue

            # label -> move the staged episode into success/ or failure/
            emit(events, "awaiting_label", dir=str(ep_dir))
            lab = _read_line_blocking()
            if lab == "q":
                break
            success = (lab == "y") or (isinstance(lab, dict) and bool(lab.get("success")))
            status = "success" if success else "failure"
            dest = output_root / status / ts
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(ep_dir), str(dest))
            except OSError as e:
                print(f"[teleop] could not move episode to {status}: {e}", flush=True)
                dest = ep_dir
            emit(events, "labeled", dir=str(dest), success=success)
    finally:
        emit(events, "session_end")
        if env is not None:
            _halt(env)
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
