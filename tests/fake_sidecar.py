#!/usr/bin/env python3
"""A planner sidecar with no planner behind it, for exercising the channel.

Speaks the same newline-JSON protocol as ``tandem/planners/tiptop/sidecar.py`` and answers every
verb with something plausible, so the parent half can be tested without a GPU, a robot or a camera.
It also does the one thing that is easy to get wrong and impossible to notice in review: it prints
noise on stdout, the way a real one does when torch and the CUDA runtime announce themselves. A
channel that cannot survive that would desynchronise on the first real warmup.

Run modes, chosen by argv so one file covers the failure paths too:

    fake_sidecar.py                 answers normally
    fake_sidecar.py --noisy         prints junk on fd 1 before and between replies
    fake_sidecar.py --fail-warm     refuses to warm, with a message
    fake_sidecar.py --no-hello      never announces itself (a child that dies during import)
    fake_sidecar.py --silent VERB   accepts VERB and never answers it (a wedged planner)

It keeps the arguments of the last call of every verb and hands them back to a ``last_args`` request
(a verb only this fake answers), so a test can check what the parent actually put on the wire --
``movables`` as null rather than [], say -- and not only what came back.
"""

import json
import os
import sys
import time

_PROTOCOL_OUT = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

NOISY = "--noisy" in sys.argv
FAIL_WARM = "--fail-warm" in sys.argv
NO_HELLO = "--no-hello" in sys.argv
SILENT = sys.argv[sys.argv.index("--silent") + 1] if "--silent" in sys.argv else None


def emit(payload):
    _PROTOCOL_OUT.write(json.dumps(payload) + "\n")
    _PROTOCOL_OUT.flush()


def noise():
    if NOISY:
        # Exactly the shape of the problem: a library writing to the real stdout, mid-conversation.
        print("Warp 1.4.0 initialized:", flush=True)
        _PROTOCOL_OUT.write("not json at all\n")
        _PROTOCOL_OUT.flush()


def main():
    noise()
    if not NO_HELLO:
        emit({"ready": True, "pid": os.getpid()})

    scenes = {}
    last_args = {}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        verb, request_id, args = request.get("verb"), request.get("id"), request.get("args") or {}
        if verb == "quit":
            break
        noise()
        if verb == "last_args":
            emit({"id": request_id, "ok": True, "result": last_args.get(args.get("of"))})
            continue
        last_args[verb] = args
        if verb == SILENT:
            # Longer than any caller's timeout, short enough that the suite is not held up by it.
            time.sleep(5)
            continue
        if verb == "warm" and FAIL_WARM:
            emit({"id": request_id, "ok": False, "error": "warm failed -- no CUDA device"})
            continue
        if verb == "perceive":
            scenes["s1"] = args
            emit(
                {
                    "id": request_id,
                    "ok": True,
                    "result": {
                        "scene_id": "s1",
                        "object_labels": ["blue_toy", "white_box"],
                        "table_label": "table",
                        "surface_labels": ["white_box"],
                        "rgb_path": str(args.get("save_dir", "")) + "/perception_rgb.png",
                    },
                }
            )
            continue
        if verb == "plan":
            # The real sidecar's leg semantics, as far as they show from outside: a goal that moves
            # something the leg may not pick is refused, and the task plan comes back with only the
            # operators' object arguments.
            goal = args.get("goal") or []
            movables = args.get("movables")
            moved = {a["args"][0] for a in goal if a.get("predicate") in ("on", "holding") and a.get("args")}
            outside = sorted(moved - set(movables)) if movables is not None else []
            if outside:
                result = {"ok": False, "failure_reason": f"the goal moves {', '.join(outside)}"}
            else:
                task_plan = []
                for atom in goal:
                    if atom.get("predicate") == "on" and len(atom.get("args") or []) == 2:
                        task_plan += [f"Pick({atom['args'][0]})", f"Place({', '.join(atom['args'])})"]
                result = {
                    "ok": True,
                    "planning_seconds": 1.25,
                    "plan_handle": "p1",
                    "artifacts": {},
                    "task_plan": task_plan,
                }
            emit({"id": request_id, "ok": True, "result": result})
            continue
        if verb == "execute":
            emit(
                {
                    "id": request_id,
                    "ok": True,
                    "result": {"ok": True, "n_frames": 42, "rollout_dir": args.get("save_dir")},
                }
            )
            continue
        if verb == "capture_frame":
            emit({"id": request_id, "ok": True, "result": {"path": "/tmp/frame.png"}})
            continue
        emit({"id": request_id, "ok": True, "result": {}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
