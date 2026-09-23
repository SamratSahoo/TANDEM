"""Talking to a backend that has to live in another interpreter.

A task and motion planner needs torch, CUDA kernels, a camera SDK and a robot client. tandem needs
none of those, and the whole point of ``pip install tandem-tamp`` working on a laptop is that it goes
on needing none of them. So a hosted backend runs as a child process inside the heavy environment,
and this is the channel to it: one JSON object per line, one request in flight at a time.

Deliberately dumb, for the same reason the events file is. The child holds a robot and several
cameras; when something goes wrong the useful question is "what was the last thing it was asked, and
what did it say", and a line-delimited transcript answers that from a log file. There is no
concurrency to reason about because there is none to have -- a planner with one arm answers one
question at a time.

The child speaks first, once, to say it is up -- ``{"ready": true, "pid": ..., "verbs": [...]}``, or
``{"ready": false, "error": "..."}`` -- and after that four kinds of line come back:

    {"id": 3, "ok": true, "result": {...}}      the answer to request 3
    {"id": 3, "ok": false, "error": "..."}      request 3 failed, with a message for an operator
    {"log": "...", "level": "info"}             out-of-band progress, forwarded to the session log
    {"event": "name", ...}                      a session event, handed to ``on_event`` (no "id")

An unparseable line is forwarded as a log line rather than dropped: the child's own stdout noise (a
CUDA warning, a library banner) is worth seeing and is never worth failing on.

The child half of this protocol, for any planner, is ``tandem/planners/sidecar_kit/tandem_sidecar.py``.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tandem.planners.base import BackendError

_log = logging.getLogger(__name__)

# How long to wait for the child to announce itself. Generous: it imports torch and builds CUDA
# context before it can say anything at all.
HANDSHAKE_TIMEOUT = 600.0
# How long a single request may take. `warm` and `plan` are the slow ones; a plan that has not come
# back in ten minutes is wedged, not thinking.
DEFAULT_TIMEOUT = 900.0
# Grace for a child asked to quit before it is signalled. It releases cameras on the way out, and the
# SDK teardown for two of them measures ~14s.
EXIT_GRACE = 60.0
# How often a caller waiting on a reply is given the chance to act (``request(poll=...)``): often
# enough that a stop asked for mid-execution reaches the child within a step, rarely enough to cost
# nothing.
POLL_INTERVAL = 0.1


class HostedBackendChannel:
    """A child process answering verbs over newline-delimited JSON."""

    def __init__(
        self,
        argv: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        on_log: Callable[[str, str], None] | None = None,
        on_event: Callable[[dict], None] | None = None,
    ) -> None:
        self._argv = list(argv)
        self._cwd = str(cwd) if cwd else None
        self._env = env
        self._on_log = on_log or (lambda stream, text: _log.info("%s: %s", stream, text))
        self._on_event = on_event or (lambda event: self._on_log("backend", f"event: {json.dumps(event)}"))
        # What the child said when it started: at least {"ready": true}, and the verbs it answers
        # when it says so. Empty until start() returns.
        self.hello: dict = {}
        self._proc: subprocess.Popen | None = None
        self._pgid: int | None = None
        self._next_id = 0
        self._lock = threading.RLock()
        # Replies arrive on a reader thread rather than being read inline. That is not for
        # concurrency -- there is one request in flight -- but so a timeout can actually fire: a
        # blocking readline() on a child that has stopped answering never returns, and a planner
        # that wedges holding a robot is exactly the case the timeout exists for.
        self._replies: queue.Queue = queue.Queue()
        self._reader: threading.Thread | None = None

    # ---- lifecycle ---------------------------------------------------------

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> HostedBackendChannel:
        """Launch the child and wait for it to say it is ready."""
        try:
            self._proc = subprocess.Popen(
                self._argv,
                cwd=self._cwd,
                env=self._env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                # Its own process group: the launcher is usually a wrapper (`pixi run`), and the
                # process actually holding the robot is its grandchild, so a signal to the direct
                # child never reaches it.
                start_new_session=True,
            )
        except OSError as exc:
            raise BackendError(f"could not start the planner backend: {exc}") from exc

        self._pgid = os.getpgid(self._proc.pid)
        self._pump(self._proc.stderr, "backend-stderr")
        self._reader = threading.Thread(target=self._read_forever, name="backend-stdout", daemon=True)
        self._reader.start()

        try:
            hello = self._await_reply(None, timeout=HANDSHAKE_TIMEOUT, expect_hello=True)
        except BackendError:
            # A child that was too slow, or wedged on a camera the previous process still holds, is
            # still holding the robot and the cameras. Abandoning it here leaves no handle to kill it
            # with, and the operator's retry then fails on hardware their own last attempt is sitting
            # on -- with nothing on screen connecting the two.
            self.stop()
            raise
        if not hello.get("ready"):
            self.stop()
            raise BackendError(f"the planner backend did not start: {hello.get('error') or hello}")
        self.hello = hello
        return self

    def stop(self) -> None:
        """Ask the child to quit, then make sure it has."""
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                self._send({"verb": "quit", "id": -1, "args": {}})
            except BackendError:
                pass
            try:
                proc.wait(timeout=EXIT_GRACE)
            except subprocess.TimeoutExpired:
                self._signal_group()
        self._proc = None

    # ---- requests ----------------------------------------------------------

    def call(self, verb: str, timeout: float = DEFAULT_TIMEOUT, **args: Any) -> Any:
        """Ask the child for one thing and return its result, or raise ``BackendError``."""
        return self.request(verb, args, timeout=timeout)

    def request(
        self,
        verb: str,
        args: dict[str, Any] | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        poll: Callable[[], None] | None = None,
    ) -> Any:
        """``call``, with the arguments as a dict and a ``poll`` run every POLL_INTERVAL while waiting.

        ``poll`` is how the parent acts on something while the child is busy -- a stop asked for in
        the middle of an execution -- without a second request in flight, which the protocol does not
        have. Its keyword cannot collide with a verb's arguments, which ``call``'s ``**args`` could.
        """
        with self._lock:
            if not self.alive:
                raise BackendError("the planner backend is not running")
            self._next_id += 1
            request_id = self._next_id
            self._send({"id": request_id, "verb": verb, "args": dict(args or {})})
            reply = self._await_reply(request_id, timeout=timeout, poll=poll)
        if not reply.get("ok"):
            raise BackendError(str(reply.get("error") or f"the backend refused {verb!r}"))
        return reply.get("result")

    # ---- plumbing ----------------------------------------------------------

    def _send(self, payload: dict) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            raise BackendError("the planner backend is not accepting input any more")
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise BackendError(f"the planner backend closed its input: {exc}") from exc

    def _read_forever(self) -> None:
        """Parse the child's stdout on its own thread: replies to the queue, everything else to the log."""
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            for raw in proc.stdout:
                line = raw.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except ValueError:
                    # Not ours: library banners, warnings, progress bars. Worth seeing in the session
                    # log, never worth failing on -- and never mistaken for a reply.
                    self._on_log("backend", line)
                    continue
                if not isinstance(payload, dict) or "log" in payload:
                    self._on_log(
                        "backend", str(payload.get("log", line)) if isinstance(payload, dict) else line
                    )
                    continue
                if "event" in payload and "id" not in payload:
                    # Out of band, like a log line: never a reply, whatever request is in flight.
                    try:
                        self._on_event(payload)
                    except Exception as exc:  # a broken sink must not stop replies being read
                        self._on_log("backend", f"could not record an event from the backend: {exc}")
                    continue
                self._replies.put(payload)
        except (ValueError, OSError):
            pass
        finally:
            # Wake anyone waiting: the child is gone and no reply is ever coming.
            self._replies.put(None)

    def _await_reply(
        self,
        request_id: int | None,
        *,
        timeout: float,
        expect_hello: bool = False,
        poll: Callable[[], None] | None = None,
    ) -> dict:
        """Wait for the reply to ``request_id``, discarding anything stale.

        A timeout here means the child is wedged holding a robot, so it says so in those terms rather
        than as a bare TimeoutError -- the person reading it is standing next to the arm.
        """
        deadline = time.monotonic() + timeout
        while True:
            if poll is not None:
                poll()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BackendError(
                    f"the planner backend did not answer within {timeout:.0f}s; it may be wedged "
                    "holding the robot"
                )
            try:
                payload = self._replies.get(timeout=min(remaining, POLL_INTERVAL) if poll else remaining)
            except queue.Empty:
                continue
            if payload is None:
                code = self._exit_code()
                raise BackendError(
                    f"the planner backend exited (code {code}) without answering. Its stderr is "
                    "in the session log."
                )
            if expect_hello:
                if "ready" in payload:
                    return payload
                # Only the announcement can end the handshake. A stray object with no "id" used to
                # match the handshake's own id of None and be taken for it.
                self._on_log(
                    "backend", f"ignoring a message sent before the backend announced itself: {payload}"
                )
                continue
            if payload.get("id") == request_id:
                return payload
            # A reply to a request nobody is waiting for any more: only possible after a timeout, and
            # keeping it would desynchronise every later call.
            self._on_log("backend", f"discarding a late reply to request {payload.get('id')}")

    def _exit_code(self) -> int | None:
        """The exit status of a child whose stdout just closed. It has usually not been reaped yet at
        that instant, and "exited (code None)" tells the person reading it nothing about a crash."""
        proc = self._proc
        if proc is None:
            return None
        try:
            return proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            return None

    def _pump(self, stream, name: str) -> None:
        def run() -> None:
            try:
                for raw in stream:
                    self._on_log(name, raw.rstrip("\n"))
            except (ValueError, OSError):
                pass

        threading.Thread(target=run, name=name, daemon=True).start()

    def _signal_group(self) -> None:
        import signal

        if self._pgid is None or self._proc is None or self._proc.poll() is not None:
            return
        for sig, grace in ((signal.SIGTERM, 8.0), (signal.SIGKILL, 2.0)):
            try:
                os.killpg(self._pgid, sig)
            except (ProcessLookupError, PermissionError):
                return
            try:
                self._proc.wait(timeout=grace)
                return
            except subprocess.TimeoutExpired:
                continue
