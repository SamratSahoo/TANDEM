#!/usr/bin/env python3
"""The sidecar half of a tandem planner: tandem's JSON-lines protocol, for any planner.

A planner that needs an environment of its own -- torch, CUDA kernels, a camera SDK, a robot client --
cannot run inside tandem, which installs with pip on a laptop. It runs in a *sidecar* instead: a
script tandem launches with that environment's interpreter, which answers the verbs of tandem's
planner protocol one JSON object per line on stdin and stdout. This module is all a sidecar needs to
do that. A whole sidecar is handler functions and one call::

    from tandem_sidecar import log, serve          # FIRST -- see "stdout" below

    def perceive(*, task_hint, save_dir, reset_arm=True, open_gripper=False):
        ...
        return {"scene_id": "s1", "object_labels": ["apple"], "table_label": "table"}

    def plan(*, scene_id, goal, surfaces, save_dir):
        ...
        return {"ok": True, "plan_handle": "p1", "task_plan": ["Drop(apple, bin)"]}

    def execute(*, plan_handle, leg, save_dir):
        ...
        return {"ok": True, "n_frames": 120, "rollout_dir": save_dir}

    if __name__ == "__main__":
        raise SystemExit(serve({"perceive": perceive, "plan": plan, "execute": execute}))

``serve`` also takes an object, answering every protocol verb it has a method for (or exactly the
``verbs`` it is given). The parent half is ``tandem.planners.sidecar.SidecarPlanner``, which
launches the script, maps each protocol call onto a request, and turns replies back into tandem's
types.

**Standard library only, and never tandem.** This file runs in the planner's environment, which has
no tandem in it and should not need one; installing tandem there would drag tandem's dependencies
into a solved CUDA environment for nothing. tandem puts this file's directory on the sidecar's
``PYTHONPATH``, so ``import tandem_sidecar`` works without installing anything. It is written for
Python 3.8 and later, since a planner's environment pins its own interpreter.

**stdout.** Importing this module takes the process's real stdout for the protocol and points fd 1
at stderr. Everything a planner imports prints -- CUDA banners, a library's version line, a stray
``print()`` -- and one such line in the middle of a reply desynchronises the channel for good. On
stderr it lands in tandem's session log instead, which is where it was wanted anyway. So import this
module before anything that might print, which in practice means first.

The wire protocol (tandem's side is ``tandem/planners/rpc.py``)::

    hello    {"ready": true, "pid": 4242, "verbs": ["perceive", ...], "kit": 1}    once, on start
             {"ready": false, "error": "..."}                                       could not start
    request  {"id": 3, "verb": "plan", "args": {...}}           handler(**args)
    reply    {"id": 3, "ok": true, "result": <handler's return value>}
             {"id": 3, "ok": false, "error": "plan failed -- ValueError: ..."}
    log      {"log": "...", "level": "info"}                   log(); into the session log
    event    {"event": "grasp_chosen", ...}                    event(); into the session's events file
    quit     {"verb": "quit", ...}                             serve() cleans up and returns

What each verb's handler receives, and returns (a JSON-safe value; a dict in every case below):

    warm(output_dir, execute, record, **planner-specific)     -> anything; build solvers, open hardware
    perceive(task_hint, save_dir, reset_arm, open_gripper)    -> scene: scene_id, object_labels,
                                                                 table_label, surface_labels, rgb_path,
                                                                 detected_goal
    plan(scene_id, goal, surfaces, save_dir                   -> ok, failure_reason, planning_seconds,
         [, movables][, return_home][, reuse_skeleton])          plan_handle, artifacts, task_plan
    execute(plan_handle, leg, save_dir)                       -> ok, failure_reason, n_frames,
                                                                 rollout_dir, stopped_early
    capture_frame(camera)                                     -> {"path": "<an image file>"}
    home() / release_hardware() / reacquire_hardware() / close()    -> anything

``goal`` is a list of ``{"predicate", "args"}`` in the planner's wire spelling. ``movables``,
``return_home`` and ``reuse_skeleton`` are sent only when the planner's capabilities declare the
matching support. ``leg`` carries ``trajectory_id``, ``instruction``, ``segment_source``,
``phase_index``, ``n_phases``, ``phase_description`` and ``record``, which ``execute`` stamps into the
leg's ``_meta.json`` (the recording contract is in ``tandem.planners.sdk.Planner.execute``). A verb
the sidecar does not answer falls back to tandem's default for it: nothing, for the lifecycle and
hardware verbs; a clear "not supported" for ``capture_frame``. ``perceive``, ``plan`` and
``execute`` are required.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import traceback

# ------------------------------------------------------------------------------------------ stdout
# Taken at import, before the planner's script imports anything that might print; see the docstring.
_PROTOCOL_OUT = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

#: The verbs of tandem's planner protocol. Mirrors ``tandem.planners.base.VERBS``, which this file
#: cannot import; tandem's tests pin the two together.
PROTOCOL_VERBS = (
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

#: Which revision of this helper the sidecar runs, announced in the hello.
KIT_VERSION = 1

#: Set by tandem when the planner declares cooperative stop: the file whose existence means "stop".
STOP_FILE_ENV = "TANDEM_SIDECAR_STOP_FILE"

# A planner may log from a worker thread while the main thread writes a reply; two half-lines
# interleaved on the protocol stream are one unparseable line and one lost reply.
_WRITE_LOCK = threading.Lock()

# Set once tandem has gone away: a write to it failed. Likeliest mid-execute, since that is the longest
# verb -- tandem crashed, or was killed, while the arm moved.
_PARENT_GONE = threading.Event()


def _emit(payload):
    line = json.dumps(payload) + "\n"
    with _WRITE_LOCK:
        if _PARENT_GONE.is_set():
            return
        try:
            _PROTOCOL_OUT.write(line)
            _PROTOCOL_OUT.flush()
        except (OSError, ValueError):
            # BrokenPipeError: nobody is reading. It used to escape serve() from inside its own error
            # handler, and the planner's close() -- which releases the robot and the cameras -- never ran.
            _parent_gone()


def _parent_gone():
    """tandem is not there any more: stop writing to it, and point everything that would at nowhere.

    Not only the protocol stream. The planner's own prints and stderr go to the same dead parent, and
    each would raise BrokenPipeError too -- inside the close() that is about to run because of this.
    """
    _PARENT_GONE.set()
    try:
        fd = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        return
    for target in (_PROTOCOL_OUT.fileno(), 1, 2):
        try:
            os.dup2(fd, target)
        except (OSError, ValueError):
            pass
    os.close(fd)


def log(message, level="info"):
    """One line in tandem's session log, where the operator reads it. Safe from any thread."""
    _emit({"log": str(message), "level": str(level)})


def event(name, **fields):
    """One event in the session's events file: ``{"event": name, **fields}``, fields JSON-safe.

    For milestones something downstream acts on (a UI, a tailer), as opposed to ``log`` lines, which
    are for a person. ``id`` and ``log`` are not allowed as field names: they would make the line
    read as a reply or a log line.
    """
    clash = sorted({"id", "log", "event"} & set(fields))
    if clash:
        raise ValueError("an event cannot carry the field(s) " + ", ".join(clash))
    _emit(dict({"event": str(name)}, **fields))


def should_stop():
    """Whether tandem has asked the current execution to stop. Poll it at step boundaries.

    Only ever True for a planner whose capabilities declare ``supports_cooperative_stop``: tandem
    then passes a stop file in the environment and creates it when the operator preempts. Always
    False otherwise, so polling it is harmless either way.
    """
    path = os.environ.get(STOP_FILE_ENV)
    return bool(path) and os.path.exists(path)


def serve(handlers, verbs=None, on_exit=None):
    """Answer tandem's requests until it says quit or closes stdin. Returns the exit status.

    ``handlers`` is either a mapping of verb -> callable, or an object whose methods are named for
    the verbs. ``verbs`` says which verbs to answer: by default every key of the mapping, or every
    protocol verb the object has a method for. A verb outside the protocol may be answered too --
    tandem's ``SidecarPlanner.call`` reaches it -- so a planner can expose a debugging verb of its own.

    On the way out ``on_exit`` runs, or else the ``close`` handler when there is one, so the
    planner releases its hardware even when tandem went away without asking it to -- between two
    requests (its stdin closes) or in the middle of one (the reply cannot be written): whatever ends
    the loop, the cleanup runs.
    """
    try:
        table = _resolve(handlers, verbs)
    except Exception as exc:
        # Said in the hello rather than as a traceback on stderr: tandem reports it as the reason
        # the planner did not start.
        _emit({"ready": False, "error": f"the sidecar could not start: {type(exc).__name__}: {exc}"})
        return 1

    _emit({"ready": True, "pid": os.getpid(), "verbs": list(table), "kit": KIT_VERSION})
    try:
        _answer(table)
    finally:
        cleanup = on_exit if on_exit is not None else table.get("close")
        if cleanup is not None:
            try:
                cleanup()
            except Exception:
                log(traceback.format_exc())
    return 0


def _answer(table):
    """Answer requests until tandem says quit, closes stdin, or can no longer be written to."""
    for line in sys.stdin:
        if _PARENT_GONE.is_set():
            return
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError:
            log(f"ignoring an unparseable request: {line[:200]}")
            continue
        if not isinstance(request, dict):
            log(f"ignoring a request that is not an object: {line[:200]}")
            continue

        verb = request.get("verb")
        request_id = request.get("id")
        args = request.get("args") or {}
        if verb == "quit":
            return
        handler = table.get(verb)
        if handler is None:
            _emit({"id": request_id, "ok": False, "error": f"unknown verb {verb!r}"})
            continue
        try:
            result = handler(**args)
            _emit({"id": request_id, "ok": True, "result": result})
        except Exception as exc:
            # tandem turns this into a BackendError an operator reads, so it says what failed in
            # words, with the traceback beside it in the session log rather than inside the message.
            log(traceback.format_exc())
            _emit({"id": request_id, "ok": False, "error": f"{verb} failed -- {type(exc).__name__}: {exc}"})
        if _PARENT_GONE.is_set():
            # Nobody will read another reply, nor send another request worth answering.
            return


def _resolve(handlers, verbs):
    """verb -> callable, or a ValueError naming the verb that has no handler."""
    if isinstance(handlers, dict) or hasattr(handlers, "keys"):
        names = tuple(verbs) if verbs is not None else tuple(handlers.keys())
        lookup = handlers.get
    else:
        names = (
            tuple(verbs)
            if verbs is not None
            else tuple(v for v in PROTOCOL_VERBS if callable(getattr(handlers, v, None)))
        )

        def lookup(name):
            return getattr(handlers, name, None)

    table = {}
    for name in names:
        if name == "quit":
            raise ValueError("'quit' is tandem's, not a verb a sidecar can answer")
        handler = lookup(name)
        if not callable(handler):
            raise ValueError(f"no handler for the verb {name!r}")
        table[name] = handler
    return table
