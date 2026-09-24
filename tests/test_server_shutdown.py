"""`tandem ui` exiting ends its sessions the way `tandem collect` does: arm parked, hardware let go, merges done.

A session drives the robot on daemon threads. Nothing called `SessionManager.shutdown`, so a server that
exited killed them where they stood -- the arm wherever the last plan left it, the planner still holding
the robot and the cameras, and a merge in flight cut off with the trial's legs half moved.
"""

from __future__ import annotations

import asyncio
import json
import signal
import socket
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from helpers import use_fake_backend, wait_for

from tandem.core import episodes, secrets
from tandem.core import session as session_mod
from tandem.core.errors import TandemError
from tandem.core.session import State


@pytest.fixture
def manager(monkeypatch):
    """The process's session manager, fresh for this test and put back after it."""
    monkeypatch.setattr(session_mod, "_manager", None)
    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
    return session_mod.manager()


@pytest.fixture
def slow_merge(monkeypatch):
    """A merge that takes a while, and says whether it finished."""
    started, finished = threading.Event(), threading.Event()

    def merge(*args, **kwargs):
        started.set()
        time.sleep(0.6)
        finished.set()

    monkeypatch.setattr(episodes, "merge_trajectory", merge)
    return started, finished


def test_closing_the_app_parks_the_arm_and_lets_the_merge_in_flight_finish(profile, monkeypatch, manager, slow_merge):
    from tandem.server.app import create_app

    backends = use_fake_backend(monkeypatch)
    merge_started, merge_finished = slow_merge
    with TestClient(create_app()) as client:
        created = client.post("/api/sessions", json={"profile": profile.name, "task": "pick up the block"})
        assert created.status_code == 200, created.text
        session = manager.get(created.json()["id"])
        assert wait_for(lambda: session.state is State.AWAITING_TASK), f"stuck in {session.state}"
        client.post(f"/api/sessions/{session.id}/continue", json={"more": True})
        assert wait_for(lambda: session.state is State.AWAITING_LABEL), f"stuck in {session.state}"
        client.post(f"/api/sessions/{session.id}/label", json={"success": True})
        assert merge_started.wait(5), "the trial was never filed"
    # Leaving the block is the server shutting down.

    assert merge_finished.is_set(), "the server exited under a merge in flight"
    assert session.state is State.STOPPED and not session.running
    calls = backends[-1].calls
    assert "home" in calls and "close" in calls and calls.index("home") < calls.index("close"), calls


def test_the_wait_is_bounded_and_a_second_ctrl_c_ends_it(profile, monkeypatch, manager):
    use_fake_backend(monkeypatch)
    release = threading.Event()
    monkeypatch.setattr(episodes, "merge_trajectory", lambda *a, **k: release.wait(30))
    session = manager.create(profile, task="pick up the block")
    try:
        assert wait_for(lambda: session.state is State.AWAITING_TASK)
        session.next_task()
        assert wait_for(lambda: session.state is State.AWAITING_LABEL)
        session.label(True)
        assert wait_for(lambda: bool(session._merges))

        started = time.monotonic()
        assert manager.shutdown(timeout=0.3) == [session], "a session still ending must be said to be"
        assert time.monotonic() - started < 5

        forced: list[str] = []
        monkeypatch.setattr(session, "force_stop", lambda: forced.append(session.id))
        threading.Timer(0.3, manager.abandon).start()
        started = time.monotonic()
        assert manager.shutdown(timeout=60) == [session]
        assert time.monotonic() - started < 5, "abandon did not cut the wait short"
        assert forced == [session.id], "a session given up on is force-stopped"
    finally:
        release.set()
        session.wait(timeout=5)


def test_a_session_that_would_not_start_does_not_block_the_next(profile, monkeypatch, manager):
    """It stayed in the manager as `spawning`, live for its profile, and every later start was refused."""
    use_fake_backend(monkeypatch)
    monkeypatch.setattr(secrets, "gemini_api_key", lambda: None)
    monkeypatch.setattr(profile.hitl, "enabled", True)
    with pytest.raises(TandemError, match="No Gemini API key"):
        manager.create(profile, task="pick up the block")
    assert manager.all() == []

    monkeypatch.setattr(profile.hitl, "enabled", False)
    session = manager.create(profile, task="pick up the block")
    try:
        assert wait_for(lambda: session.state is State.AWAITING_TASK)
    finally:
        manager.shutdown(timeout=10)


# --- the uvicorn server `tandem ui` runs -------------------------------------------------------------------


class _Ending:
    """A session that, once stopped, takes its time to end -- a park, a merge -- until ``released``."""

    def __init__(self, session_id: str = "s1") -> None:
        self.id = session_id
        self.profile = SimpleNamespace(name="bench")
        self.state = State.AWAITING_TASK
        self.stopped = threading.Event()
        self.released = threading.Event()
        self.forced = 0

    @property
    def alive(self) -> bool:
        return not self.stopped.is_set()

    @property
    def running(self) -> bool:
        return not self.released.is_set()

    def stop(self) -> None:
        self.stopped.set()

    def wait(self, timeout=None):
        self.released.wait(timeout)
        return None if self.running else 0

    def force_stop(self) -> None:
        self.forced += 1

    def subscribe(self, callback):
        return lambda: None

    def summary(self) -> dict:
        return {"id": self.id, "state": self.state.value}

    def logs(self, limit=None) -> list:
        return []


@pytest.fixture
def ending(manager):
    session = _Ending()
    manager._sessions[session.id] = session
    yield session
    session.released.set()


@pytest.fixture
def uvicorn_logs(monkeypatch, caplog):
    """uvicorn's own log, where the ERRORs of a cancelled connection go. Its default logging config (which
    another test may have loaded) stops it short of the handler caplog listens on."""
    import logging

    monkeypatch.setattr(logging.getLogger("uvicorn"), "propagate", True)
    caplog.set_level(logging.INFO)
    return caplog


@contextmanager
def _serving(grace: float = 5):
    """`tandem ui`'s own server and app, on a free port, from a thread: its Ctrl-C is `handle_exit`."""
    import uvicorn

    from tandem.cli import ui
    from tandem.server.app import create_app

    config = uvicorn.Config(
        create_app(),
        host="127.0.0.1",
        port=0,
        log_config=None,
        log_level="warning",
        access_log=False,
        timeout_graceful_shutdown=grace,
    )
    server = ui._server(config)
    thread = threading.Thread(target=server.run, name="test-ui-server", daemon=True)
    thread.start()
    try:
        assert wait_for(lambda: server.started, timeout=10), "the server never started"
        yield server, thread
    finally:
        server.should_exit = server.force_exit = True
        thread.join(10)


def _ctrl_c(server) -> None:
    server.handle_exit(signal.SIGINT, None)


def _stills(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "was still ending when the server exited" in r.getMessage()]


def test_a_second_ctrl_c_before_the_apps_shutdown_still_force_stops_what_is_ending(ending, uvicorn_logs):
    """A second Ctrl-C while uvicorn is still closing connections makes it skip the app's shutdown, which
    is what ended the sessions: nothing was force-stopped -- a teleop driver in a process group of its
    own kept the arm -- and nothing said the session was cut off."""
    with _serving() as (server, thread):
        _ctrl_c(server)
        _ctrl_c(server)
        thread.join(10)
        assert not thread.is_alive(), "the second Ctrl-C did not quit"
    assert ending.stopped.is_set()
    assert ending.forced == 1, "what was still ending was not force-stopped"
    assert len(_stills(uvicorn_logs)) == 1, uvicorn_logs.text
    # The app's shutdown is run, not skipped: left waiting, it was cancelled at exit with a traceback.
    errors = [r.getMessage() for r in uvicorn_logs.records if r.levelname == "ERROR"]
    assert not errors, errors


def test_a_second_ctrl_c_during_the_apps_shutdown_force_stops_once_and_says_so_once(ending, manager, caplog):
    with _serving() as (server, thread):
        _ctrl_c(server)
        assert wait_for(lambda: manager.shutting_down), "the app's shutdown never began"
        time.sleep(0.3)
        assert thread.is_alive(), "the server did not wait for the session to end"
        _ctrl_c(server)
        thread.join(10)
        assert not thread.is_alive(), "the second Ctrl-C did not cut the wait short"
    assert ending.forced == 1 and len(_stills(caplog)) == 1, caplog.text


def test_an_abandon_before_the_wait_began_is_not_lost(ending, manager):
    """`shutdown` cleared the flag as it started: a Ctrl-C that came just before ran the whole wait."""
    manager.abandon()
    started = time.monotonic()
    assert manager.shutdown(timeout=3) == [ending]
    assert time.monotonic() - started < 1, "the wait ran as if nobody had asked to quit"
    assert ending.forced == 1


def _read_stream(port: int, session_id: str, into: dict) -> None:
    """A collect page's event stream, read to its end: the bytes, and when the server closed it."""
    with socket.create_connection(("127.0.0.1", port), timeout=20) as sock:
        sock.sendall(f"GET /api/sessions/{session_id}/stream HTTP/1.1\r\nHost: localhost\r\n\r\n".encode())
        body = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            body += chunk
            into["body"] = body
    into["closed_at"] = time.monotonic()


def test_one_ctrl_c_with_a_collect_page_open_closes_its_stream_rather_than_cancelling_it(
    ending, uvicorn_logs, manager
):
    """The stream stayed open until its session ended, longer than the grace uvicorn gives connections, and
    was then cancelled: an ERROR and a CancelledError traceback on the terminal of every Ctrl-C that went
    exactly as it should."""
    from tandem.server.routes import sessions as sessions_routes

    grace = 2.0
    read: dict = {}
    with _serving(grace=grace) as (server, thread):
        port = server.servers[0].sockets[0].getsockname()[1]
        reader = threading.Thread(target=_read_stream, args=(port, ending.id, read), daemon=True)
        reader.start()
        assert wait_for(lambda: b"data:" in read.get("body", b"")), "the stream never opened"

        # The session takes longer to end than the connections are given.
        threading.Timer(grace + 0.5, ending.released.set).start()
        pressed = time.monotonic()
        _ctrl_c(server)
        reader.join(10)
        assert "closed_at" in read, "the stream was never closed"
        assert read["closed_at"] - pressed < grace, "the stream was held open until the grace ran out"
        thread.join(10)
        assert not thread.is_alive()

    assert sessions_routes.SHUTDOWN_NOTICE.encode() in read["body"]
    errors = [r.getMessage() for r in uvicorn_logs.records if r.levelname == "ERROR"]
    assert not errors, errors
    assert ending.forced == 0 and not _stills(uvicorn_logs), "a session that ended in time was cut off"


def test_a_closed_app_ends_a_stream_with_a_notice(ending):
    from tandem.server.app import close_streams, create_app
    from tandem.server.routes import sessions as sessions_routes

    app = create_app()
    close_streams(app)
    got: dict = {}
    reader = threading.Thread(
        target=lambda: got.update(response=TestClient(app).get(f"/api/sessions/{ending.id}/stream")), daemon=True
    )
    reader.start()
    reader.join(5)
    if reader.is_alive():
        ending.stop()  # ends it at its next keepalive, rather than never
        pytest.fail("the stream stayed open with the app closed")
    lines = got["response"].text.splitlines()
    frames = [json.loads(line[len("data: ") :]) for line in lines if line.startswith("data: ")]
    assert frames[0]["type"] == "state"
    assert frames[-1]["type"] == "log" and frames[-1]["text"] == sessions_routes.SHUTDOWN_NOTICE


class _Manager:
    def __init__(self) -> None:
        self.calls: list[str] = []
        # The app's shutdown ran, as it does after a single Ctrl-C.
        self.shutting_down = True

    def stop_all(self):
        self.calls.append("stop_all")
        return []

    def abandon(self) -> None:
        self.calls.append("abandon")


def test_the_server_stops_the_sessions_and_closes_the_streams_before_it_waits_for_connections(monkeypatch):
    """A collect page's event stream stays open while its session is on, so both have to go first."""
    import uvicorn

    from tandem.cli import ui
    from tandem.server.app import create_app

    fake = _Manager()
    monkeypatch.setattr(session_mod, "manager", lambda: fake)
    app = create_app()

    async def uvicorns_own(self, sockets=None):
        fake.calls.append(f"connections (streams closed: {app.state.closing.is_set()}), then the app's shutdown")

    monkeypatch.setattr(uvicorn.Server, "shutdown", uvicorns_own)
    server = ui._server(uvicorn.Config(app, log_config=None))
    asyncio.run(server.shutdown())
    assert fake.calls == ["stop_all", "connections (streams closed: True), then the app's shutdown"]

    server.handle_exit(signal.SIGINT, None)
    assert server.should_exit and fake.calls[-1] != "abandon", "the first Ctrl-C shuts down gracefully"
    server.handle_exit(signal.SIGINT, None)
    assert server.force_exit and fake.calls[-1] == "abandon"
