"""`tandem ui` exiting ends its sessions the way `tandem collect` does: arm parked, hardware let go, merges done.

A session drives the robot on daemon threads. Nothing called `SessionManager.shutdown`, so a server that
exited killed them where they stood -- the arm wherever the last plan left it, the planner still holding
the robot and the cameras, and a merge in flight cut off with the trial's legs half moved.
"""

from __future__ import annotations

import asyncio
import signal
import threading
import time

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


class _Manager:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def stop_all(self):
        self.calls.append("stop_all")
        return []

    def abandon(self) -> None:
        self.calls.append("abandon")


def test_the_server_stops_the_sessions_before_it_waits_for_connections(monkeypatch):
    """A collect page's event stream closes when its session ends, so the sessions have to go first."""
    import uvicorn

    from tandem.cli import ui

    fake = _Manager()
    monkeypatch.setattr(session_mod, "manager", lambda: fake)

    async def uvicorns_own(self, sockets=None):
        fake.calls.append("connections, then the app's shutdown")

    monkeypatch.setattr(uvicorn.Server, "shutdown", uvicorns_own)
    server = ui._server(uvicorn.Config(lambda scope, receive, send: None))
    asyncio.run(server.shutdown())
    assert fake.calls == ["stop_all", "connections, then the app's shutdown"]

    server.handle_exit(signal.SIGINT, None)
    assert server.should_exit and fake.calls[-1] != "abandon", "the first Ctrl-C shuts down gracefully"
    server.handle_exit(signal.SIGINT, None)
    assert server.force_exit and fake.calls[-1] == "abandon"

