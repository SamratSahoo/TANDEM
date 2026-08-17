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


def read_command() -> str:
    """One line from stdin, exiting the process on EOF.

    Every read goes through here. `readline()` returns "" forever once the pipe is closed —
    which happens the moment whatever launched us dies — so a loop that treats "" as
    unrecognised input spins at full speed. One did, and it wrote a 63 GB events file before
    anyone noticed. The real driver reads with input(), which raises EOFError and unwinds; this
    is the equivalent.
    """
    line = sys.stdin.readline()
    if line == "":
        raise SystemExit(0)
    return line.strip()


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

    # Phase planning is opt-in through --hitl-config, exactly as upstream: with the flag
    # absent the package is never imported and a run behaves as it always has.
    hitl = None
    if "--hitl-config" in args:
        hitl = json.loads(Path(args[args.index("--hitl-config") + 1]).read_text())

    emit("session_start")
    print(f"warming up for task: {task}", flush=True)
    emit("awaiting_task")

    counter = 0
    while True:
        command = read_command()

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
                if hitl and hitl.get("enabled"):
                    if not _run_human_phase(hitl):
                        emit("awaiting_task")
                        continue

                rollout.mkdir(parents=True, exist_ok=True)
                (rollout / "_meta.json").write_text(json.dumps({"instruction": task, "n_frames": 42}))
                emit("rollout_saved", dir=str(rollout), n_frames=42)
                emit("awaiting_label", dir=str(rollout))

                verdict = read_command()
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


def _run_human_phase(hitl: dict) -> bool:
    """One human phase: ask, then verify. Returns False when the operator gave up.

    Mirrors the real prompt's vocabulary — 'done' / 'abort', with SIGUSR1 handing the arm over
    from the prompt itself rather than at a plan-step boundary that will never arrive here.
    """
    attempts_left = int(hitl.get("verify_retries", 1))
    # The first attempt fails verification when the config leaves retries available, so the
    # retry path is exercised rather than merely present.
    verify_ok = attempts_left <= 0

    while True:
        emit(
            "awaiting_human_phase",
            description="fold the cloth over the toy",
            instructions="Fold the near edge of the cloth over the toy so the toy is covered.",
            expected=["the cloth is folded over the toy"],
            phase_index=1,
            n_phases=2,
        )
        answer = read_command().lower()

        if _teleop_requested:
            # "Switch to teleop" at the prompt: hand the arm over, then treat it as done.
            emit("teleop_handoff_start")
            emit("awaiting_teleop_resume")
            _wait_for_resume()
            emit("teleop_handoff_done")
            answer = "done"

        if answer in ("abort", "skip", "n", "no"):
            return False
        if answer in ("q", "exit", "quit"):
            raise SystemExit(0)
        if answer not in ("done", "y", "yes"):
            continue

        emit(
            "human_phase_verified",
            description="fold the cloth over the toy",
            ok=verify_ok,
            verdicts=[{
                "atom": "Folded(cloth)",
                "statement": "the cloth is folded over the toy",
                "holds": verify_ok,
                "reason": "" if verify_ok else "a corner of the toy is still visible",
            }],
        )
        if verify_ok:
            emit("hitl_phase_complete", phase_index=1, n_phases=2, message="phase 1 of 2 done")
            return True
        if attempts_left <= 0:
            return False
        attempts_left -= 1
        verify_ok = True


def _wait_for_resume() -> None:
    global _teleop_requested
    while True:
        if read_command() == "resume":
            _teleop_requested = False
            return


if __name__ == "__main__":
    sys.exit(main())
