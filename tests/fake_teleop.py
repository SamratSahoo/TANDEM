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
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-file", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--instruction", default="")
    parser.add_argument("--trajectory-id", default="")
    parser.add_argument("--refuse-first", action="store_true")
    args, _ = parser.parse_known_args()

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
                (leg / "_meta.json").write_text(
                    json.dumps(
                        {
                            "n_frames": max(frames, 2),
                            "instruction": args.instruction,
                            "trajectory_id": args.trajectory_id or None,
                            "segment_source": "teleop",
                        }
                    )
                )
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
