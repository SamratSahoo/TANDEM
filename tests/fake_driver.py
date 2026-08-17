"""A stand-in for the real collection driver, for testing the session engine.

It speaks the same contract — JSONL events out, stdin commands in, SIGINT preempts a rollout
without ending the session, SIGUSR1 arms a hand-off — with none of the robot, cameras, GPU or
network. That contract is the thing worth testing; the planner itself is tested upstream.

Run as:  python fake_driver.py --output-dir DIR [--enable-recording] [...]
Reads $TIPTOP_EVENTS_FILE and $TIPTOP_TASK exactly as the real driver does.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

EVENTS = os.environ.get("TIPTOP_EVENTS_FILE", "")
_preempted = False
_teleop_requested = False


def emit(event: str, **fields) -> None:
    if not EVENTS:
        return
    with open(EVENTS, "a") as handle:
        handle.write(json.dumps({"event": event, **fields}) + "\n")
        handle.flush()


class Preempt(Exception):
    pass


def _on_sigint(_signum, _frame):
    global _preempted
    _preempted = True
    raise Preempt()


def _on_sigusr1(_signum, _frame):
    global _teleop_requested
    _teleop_requested = True


def main() -> int:
    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGUSR1, _on_sigusr1)

    args = sys.argv[1:]
    output_dir = Path(args[args.index("--output-dir") + 1]) if "--output-dir" in args else Path(".")
    task = os.environ.get("TIPTOP_TASK", "")

    emit("session_start")
    print(f"warming up for task: {task}", flush=True)
    emit("awaiting_task")

    counter = 0
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        command = line.strip()

        if command == "q":
            break
        if command == "home":
            print("going home", flush=True)
            continue
        if command == "resume":
            emit("teleop_handoff_done")
            emit("awaiting_task")
            continue

        if command:
            task = command

        counter += 1
        stamp = f"2026-01-01_00-{counter:02d}-00"
        rollout = output_dir / "eval" / stamp
        try:
            emit("rollout_start", dir=str(rollout))
            print(f"planning: {task}", flush=True)
            for _ in range(20):
                time.sleep(0.02)
                if _teleop_requested:
                    emit("teleop_switch_pending")
                    emit("teleop_handoff_start")
                    time.sleep(0.05)
                    emit("awaiting_teleop_resume")
                    _wait_for_resume()
                    emit("teleop_handoff_done")
                    break
            else:
                rollout.mkdir(parents=True, exist_ok=True)
                (rollout / "_meta.json").write_text(json.dumps({"instruction": task, "n_frames": 42}))
                emit("rollout_saved", dir=str(rollout), n_frames=42)
                emit("awaiting_label", dir=str(rollout))

                verdict = sys.stdin.readline().strip()
                success = verdict == "y"
                final = output_dir / ("success" if success else "failure") / stamp
                final.parent.mkdir(parents=True, exist_ok=True)
                rollout.rename(final)
                emit("labeled", dir=str(final), success=success)
        except Preempt:
            print("rollout aborted", flush=True)
            emit("rollout_aborted")

        emit("awaiting_task")

    emit("session_end")
    return 0


def _wait_for_resume() -> None:
    global _teleop_requested
    while True:
        line = sys.stdin.readline()
        if not line or line.strip() == "resume":
            _teleop_requested = False
            return


if __name__ == "__main__":
    sys.exit(main())
