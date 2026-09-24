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

import contextlib
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
# How long whatever is left of the child's process group gets to go after SIGTERM, then after SIGKILL.
GROUP_TERM_GRACE = 8.0
GROUP_KILL_GRACE = 2.0
# How long the child may take to exit once its stdout has closed before it is taken to be still alive
# (it closed the stream and carried on) rather than gone.
EXIT_AFTER_EOF = 2.0


class BackendTimeout(BackendError):
    """The child did not answer in time. Unlike every other ``BackendError`` it may still be running --
    busy, or wedged holding the robot -- so what is done about it is the caller's decision, not a
    refusal to read past (``SidecarPlanner`` stops it, with everything it started)."""


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
        # Held while the child's process group is signalled, so the reader thread (the child died) and
        # stop() (tandem is done with it) never signal it twice over; and whether that has been done.
        self._group_lock = threading.Lock()
        self._group_done = False

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
                # Decoded leniently, never strictly. One byte that is not UTF-8 -- a camera SDK's Latin-1
                # message, a path -- raised UnicodeDecodeError in the reader, which ended it silently;
                # nothing drained the pipe after that, and once it filled the child blocked writing
                # and every verb timed out, holding the robot, with the log saying nothing. The
                # requests going the other way are json.dumps output, ASCII whatever this says.
                encoding="utf-8",
                errors="backslashreplace",
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
        except BackendTimeout:
            # A child that was too slow, or wedged on a camera the previous process still holds, is
            # still holding the robot and the cameras. Abandoning it here leaves no handle to kill it
            # with, and the operator's retry then fails on hardware their own last attempt is sitting
            # on -- with nothing on screen connecting the two. Killed rather than asked: a child that
            # has not said hello is not reading its requests yet.
            self.kill()
            raise
        except BackendError:
            self.stop()
            raise
        if not hello.get("ready"):
            self.stop()
            raise BackendError(f"the planner backend did not start: {hello.get('error') or hello}")
        self.hello = hello
        return self

    def stop(self) -> None:
        """Ask the child to quit, then make sure it -- and everything it started -- has gone."""
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
                pass
        # Every time, however the child went: quit when asked, would not, or had crashed before this
        # was called. It used to be signalled only when it outlived EXIT_GRACE, so a sidecar that exited
        # cleanly left behind whatever it had started -- a camera server, a recorder, a worker pool --
        # holding the cameras the next sidecar, or the teleop driver, then could not open.
        self._reap_group(proc)
        self._close_stdin(proc)
        self._proc = None

    def kill(self) -> None:
        """End the child and its whole process group now, without asking it to quit.

        For a child that has stopped answering: asked to quit, it would not read the request, and
        EXIT_GRACE would pass with it still holding the robot. SIGTERM first, so it can let go of what it
        holds; SIGKILL after GROUP_TERM_GRACE.
        """
        proc = self._proc
        if proc is None:
            return
        self._reap_group(proc)
        self._close_stdin(proc)
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
                    self._log_safely("backend", line)
                    continue
                if not isinstance(payload, dict) or "log" in payload:
                    self._log_safely(
                        "backend", str(payload.get("log", line)) if isinstance(payload, dict) else line
                    )
                    continue
                if "event" in payload and "id" not in payload:
                    # Out of band, like a log line: never a reply, whatever request is in flight.
                    try:
                        self._on_event(payload)
                    except Exception as exc:  # a broken sink must not stop replies being read
                        self._log_safely("backend", f"could not record an event from the backend: {exc}")
                    continue
                self._replies.put(payload)
        except (ValueError, OSError):
            pass  # the stream was closed under the reader: the child is gone
        finally:
            # Wake anyone waiting: the child is gone and no reply is ever coming.
            self._replies.put(None)
            self._after_exit(proc)

    def _after_exit(self, proc: subprocess.Popen) -> None:
        """The child's stdout has closed. If the child has gone with it, so must what it started.

        A crashed sidecar leaves its helpers orphaned, still holding the cameras, and a relaunch at the
        next warm would find them taken. A child that closed the stream and carried on is left alone
        here -- stop() deals with it.
        """
        with contextlib.suppress(Exception):
            proc.stdout.close()  # at EOF, from the thread that read it: nothing else reads it
        try:
            proc.wait(timeout=EXIT_AFTER_EOF)
        except subprocess.TimeoutExpired:
            return
        self._reap_group(proc)

    def _log_safely(self, stream: str, text: str) -> None:
        """A line to the log sink, which may fail. A reader that died with it would stop draining the
        child's pipe, and a child blocked on a full pipe answers nothing, ever."""
        try:
            self._on_log(stream, text)
        except Exception:
            _log.exception("the log sink failed on a line from the planner backend")

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
                raise BackendTimeout(
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
        """Drain ``stream`` into the log until it closes, whatever arrives on it.

        Never stops early: a drain that ends leaves the child blocked on a full pipe the first time it
        writes 64 KB more, and from then on it answers nothing. Decoding cannot fail (``start`` decodes
        leniently) and a sink that fails is logged past (``_log_safely``); only the stream closing ends
        it.
        """

        def run() -> None:
            try:
                for raw in stream:
                    self._log_safely(name, raw.rstrip("\n"))
            except (ValueError, OSError):
                pass  # closed under the drain: the child is gone
            finally:
                with contextlib.suppress(Exception):
                    stream.close()

        threading.Thread(target=run, name=name, daemon=True).start()

    def _reap_group(self, proc: subprocess.Popen | None = None) -> None:
        """SIGTERM, then SIGKILL, to whatever is left of the child's process group, until it is empty.

        The group, not the child: the launcher is usually a wrapper (`pixi run`), so what holds the
        robot is its grandchild, and a sidecar may start helpers of its own. They outlive a child that
        exits -- asked to, or not -- and waiting on the child alone never saw them. Done once per
        channel: the group id is only known to be this child's while something of it is left.
        """
        import signal

        with self._group_lock:
            pgid, proc = self._pgid, proc or self._proc
            if pgid is None or self._group_done:
                return
            for sig, grace in ((signal.SIGTERM, GROUP_TERM_GRACE), (signal.SIGKILL, GROUP_KILL_GRACE)):
                if not _group_alive(pgid, proc):
                    break
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    break
                except PermissionError:
                    continue
                deadline = time.monotonic() + grace
                while time.monotonic() < deadline and _group_alive(pgid, proc):
                    time.sleep(0.05)
            else:
                if _group_alive(pgid, proc):
                    self._log_safely(
                        "backend",
                        f"processes of the planner backend's group {pgid} outlived SIGKILL; check "
                        f"`ps -g {pgid}` before starting another",
                    )
            self._group_done = True

    @staticmethod
    def _close_stdin(proc: subprocess.Popen) -> None:
        # The end tandem writes. The ends it reads are closed by their drains, at EOF: closing one
        # from here while a drain is blocked reading it can block this thread instead.
        with contextlib.suppress(Exception):
            if proc.stdin is not None:
                proc.stdin.close()


def _group_alive(pgid: int, proc: subprocess.Popen | None) -> bool:
    """Whether anything is left in the process group ``pgid``.

    The child itself is reaped on the way (``poll``): until it is, its zombie still counts as a member,
    and the group would read as alive long after everything in it had gone.
    """
    if proc is not None:
        proc.poll()
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
