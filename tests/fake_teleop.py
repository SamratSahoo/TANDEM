#!/usr/bin/env python3
"""A teleop driver with no robot behind it, for exercising the hand-off protocol.

Mirrors the prompt/event contract of ``src/tandem/teleop/driver.py`` exactly, and nothing else: it
parks at a task prompt, records only when told to, and ends the session only from mid-recording.
Those three facts are the whole reason a leg can silently go unrecorded, so a stand-in that is more
willing than the real driver would hide the bug it exists to catch.

    --refuse-first   emit `error` on the first start, the way a controller that never woke does
"""

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "tandem" / "teleop"))
from raw_episode import emit  # noqa: E402


def read_line():
    line = sys.stdin.readline()
    if not line:
        return None
    line = line.strip()
    try:
        return json.loads(line)
    except ValueError:
        return line


def main() -> int:
    # Strict, as the real driver's tyro parser is: an unknown flag -- or an abbreviation of one -- is an
    # error, not ignored. A lenient stand-in is how a renamed flag once passed every hand-off test and
    # stopped every hand-off on a robot (tests/test_teleop_argv.py checks the real driver's own list).
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--events-file", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--instruction", default="")
    parser.add_argument("--trajectory-id", default="")
    parser.add_argument("--controller", default="right")
    parser.add_argument("--keep-pose", action="store_true")
    parser.add_argument("--hand-camera-id", default="")
    parser.add_argument("--external-camera-id", default="")
    parser.add_argument("--external-2-camera-id", default="")
    # The stand-in's own: refuse the first start, as a headset that is not on yet does.
    parser.add_argument("--refuse-first", action="store_true")
    # The driver's phase stamp, written into _meta.json only when given, as the driver does.
    parser.add_argument("--phase-index", type=int, default=None)
    parser.add_argument("--n-phases", type=int, default=None)
    parser.add_argument("--phase-description", default=None)
    args = parser.parse_args()

    events = Path(args.events_file)
    output_root = Path(args.output_root)
    emit(events, "session_start")
    refused = False

    try:
        while True:
            emit(events, "awaiting_task")
            cmd = read_line()
            if cmd is None or cmd == "q":
                break
            # Exactly the real driver's gate: anything that is not `start` is dropped on the floor,
            # `end_and_quit` included.
            if not (isinstance(cmd, dict) and cmd.get("cmd") == "start"):
                continue
            if args.refuse_first and not refused:
                refused = True
                emit(events, "error", message="VR controller not detected -- wake it and press Start again.")
                continue

            stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
            leg = output_root / "eval" / stamp
            leg.mkdir(parents=True, exist_ok=True)
            emit(events, "rollout_start", dir=str(leg), instruction=args.instruction)

            frames = 0
            saved = False
            quit_session = False
            while True:
                cmd = read_line()
                if cmd is None or cmd == "q":
                    quit_session = True
                    break
                if isinstance(cmd, dict) and cmd.get("cmd") == "discard":
                    break
                if isinstance(cmd, dict) and cmd.get("cmd") == "end":
                    saved = True
                    break
                if isinstance(cmd, dict) and cmd.get("cmd") == "end_and_quit":
                    saved, quit_session = True, True
                    break
                frames += 1
                time.sleep(0.005)

            if saved:
                meta = {
                    "n_frames": max(frames, 2),
                    "instruction": args.instruction,
                    "trajectory_id": args.trajectory_id or None,
                    "segment_source": "teleop",
                }
                for key in ("phase_index", "n_phases", "phase_description"):
                    if getattr(args, key) is not None:
                        meta[key] = getattr(args, key)
                (leg / "_meta.json").write_text(json.dumps(meta))
                emit(events, "rollout_saved", dir=str(leg), n_frames=max(frames, 2))
            else:
                emit(events, "rollout_aborted", dir=str(leg))

            # The real driver's order, and it matters which way round it is. `quit_session` is
            # honoured only INSIDE the trajectory-id branch; a leg launched WITHOUT an id falls
            # through to a success/failure prompt and blocks there, however the operator ended it.
            # A stand-in that broke out first could never reproduce that hang -- which is exactly
            # how it went unnoticed.
            if args.trajectory_id:
                if quit_session:
                    break
                continue
            emit(events, "awaiting_label", dir=str(leg))
            label = read_line()
            if label is None or label == "q":
                break
            emit(events, "labeled", dir=str(leg), success=label == "y")
            if quit_session:
                break
    finally:
        emit(events, "session_end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
