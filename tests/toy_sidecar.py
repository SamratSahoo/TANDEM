#!/usr/bin/env python3
"""The toy world served as a sidecar: what a planner's sidecar looks like written with tandem_sidecar.

``toy_planner.ToySidecarPlanner`` launches this with the test interpreter and the kit on its path,
exactly as ``SidecarPlanner`` launches a real one inside a runtime. It imports nothing from tandem.

Flags, so one script covers the failure paths too:

    --crash-on VERB        exit with status 3 when VERB arrives (a planner that segfaults mid-plan)
    --crash-on-nth VERB N  exit with status 3 when VERB arrives for the N-th time (a leg that dies after
                           an earlier one was recorded)
    --hang-up-on VERB      close the protocol stream when VERB arrives, then exit with status 4 half
                           a second later (a crash whose exit status is not in yet when tandem looks)
    --slow VERB SECONDS    sleep before answering VERB (a planner that wedges)
    --step SECONDS         how long one recorded frame takes (default 0.02)
    --only V1,V2,...       answer only these verbs
    --noisy                print on stdout, the way libraries do, at import and in every handler
    --events               send a session event after every execution
    --junk-before-hello    send a JSON object that is not the hello before it
    --bad-handler          ask serve() for a verb it has no handler for
"""

from __future__ import annotations

import os
import sys
import time

import tandem_sidecar
from tandem_sidecar import _emit, event, log, serve, should_stop
from toy_world import ToyWorld

NOISY = "--noisy" in sys.argv
if NOISY:
    print("Toy 1.0 initialized: CUDA not found, carrying on", flush=True)


def _flag(name: str, count: int = 1) -> list[str] | None:
    if name not in sys.argv:
        return None
    at = sys.argv.index(name)
    return sys.argv[at + 1 : at + 1 + count]


def main() -> int:
    crash_on = (_flag("--crash-on") or [None])[0]
    crash_on_nth = _flag("--crash-on-nth", 2)
    asked: dict = {}
    hang_up_on = (_flag("--hang-up-on") or [None])[0]
    slow = _flag("--slow", 2)
    only = set((_flag("--only") or [""])[0].split(",")) - {""}
    world = ToyWorld(step_seconds=float((_flag("--step") or ["0.02"])[0]))
    last_args: dict = {}

    def close():
        # Said, so a test can see that the planner cleaned up when tandem went away.
        log("the toy world is closed")
        return world.close()

    def execute(**kwargs):
        # The stop tandem asks for arrives as a file; the kit's should_stop() looks for it.
        return world.execute(should_stop=should_stop, **kwargs)

    handlers = {
        "warm": world.warm,
        "close": close,
        "perceive": world.perceive,
        "plan": world.plan,
        "execute": execute,
        "capture_frame": world.capture_frame,
        "home": world.home,
        "release_hardware": world.release_hardware,
        "reacquire_hardware": world.reacquire_hardware,
        # Two verbs of the toy's own, which SidecarPlanner.call reaches: where everything is, and
        # what the parent last sent for a verb (so a test can see what went over the wire).
        "where": lambda: dict(world.where),
        "last_args": lambda of: last_args.get(of),
    }
    if only:
        handlers = {verb: fn for verb, fn in handlers.items() if verb in only}

    def wrap(verb, fn):
        def handler(**kwargs):
            if verb != "last_args":
                last_args[verb] = kwargs
            asked[verb] = asked.get(verb, 0) + 1
            if verb == crash_on or (crash_on_nth and [verb, str(asked[verb])] == crash_on_nth):
                log(f"crashing on {verb} (#{asked[verb]}), as asked")
                os._exit(3)
            if verb == hang_up_on:
                tandem_sidecar._PROTOCOL_OUT.close()
                time.sleep(0.5)
                os._exit(4)
            if slow and verb == slow[0]:
                time.sleep(float(slow[1]))
            if NOISY:
                print(f"noise while answering {verb}", flush=True)
            result = fn(**kwargs)
            if "--events" in sys.argv and verb == "execute":
                event("toy_leg_recorded", n_frames=result["n_frames"], rollout_dir=result["rollout_dir"])
            return result

        return handler

    if "--junk-before-hello" in sys.argv:
        _emit({"not": "the hello"})
    verbs = [*handlers, "nonexistent"] if "--bad-handler" in sys.argv else None
    return serve({verb: wrap(verb, fn) for verb, fn in handlers.items()}, verbs=verbs)


if __name__ == "__main__":
    sys.exit(main())
