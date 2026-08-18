"""The human-in-the-loop session engine, driven against a stand-in for the real driver.

These are the tests that matter most: the state machine, the signal semantics and the stdin
protocol are what stand between an operator and a robot that is already moving.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from helpers import FakeRuntime, wait_for

from tandem.core import secrets
from tandem.core import session as session_mod
from tandem.core.errors import SessionConflict
from tandem.core.session import Session, State


@pytest.fixture
def live_session(profile, tmp_path, monkeypatch):
    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
    runtime = FakeRuntime(tmp_path / "runtime")
    session = Session(profile, runtime, task="pick up the block")
    session.start()
    assert wait_for(lambda: session.state is State.AWAITING_TASK), f"stuck in {session.state}"
    yield session
    if session.alive:
        session.stop(park=False)
        session.wait(timeout=5)


def test_a_full_rollout_reaches_labeled(live_session):
    session = live_session
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_LABEL)

    session.label(True)
    assert wait_for(lambda: session.labeled_count == 1)
    assert session.success_count == 1
    assert session.rollouts[-1].status == "success"
    assert wait_for(lambda: session.state is State.AWAITING_TASK)


def test_a_failure_label_is_recorded_as_such(live_session):
    session = live_session
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_LABEL)
    session.label(False)
    assert wait_for(lambda: session.labeled_count == 1)
    assert session.success_count == 0
    assert session.rollouts[-1].status == "failure"


def test_a_new_task_replaces_the_old_one(live_session):
    session = live_session
    session.next_task("stack the cubes")
    assert session.task == "stack the cubes"
    assert wait_for(lambda: session.state is State.AWAITING_LABEL)


def test_preempt_aborts_the_rollout_and_keeps_the_session_warm(live_session):
    """The whole point: a bad episode costs one preempt, not a two-minute re-warm."""
    session = live_session
    session.next_task()
    assert wait_for(lambda: session.state is State.ROLLING)

    session.preempt()
    assert wait_for(lambda: session.state is State.AWAITING_TASK)
    assert session.alive
    assert session.end_reason is None, "preempt must not look like a stop"
    assert session.labeled_count == 0

    # And the warm session can still collect.
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_LABEL)


def test_preempt_is_refused_during_a_handoff(live_session):
    """SIGINT there raises out of the driver's hand-off wait, leaving it with a closed robot
    client and no cameras. Returning control is the way out."""
    session = live_session
    session.next_task()
    assert wait_for(lambda: session.state is State.ROLLING)

    session.request_teleop()
    assert wait_for(lambda: session.state is State.TELEOP_HANDOFF, timeout=10)

    with pytest.raises(SessionConflict) as excinfo:
        session.preempt()
    assert "hand-off" in excinfo.value.message


def test_returning_control_replans_the_same_task(live_session):
    session = live_session
    session.next_task("fold the cloth")
    assert wait_for(lambda: session.state is State.ROLLING)

    session.request_teleop()
    assert wait_for(lambda: session.state is State.TELEOP_HANDOFF, timeout=10)

    session.resume_from_teleop()
    assert wait_for(lambda: session.state is State.AWAITING_TASK, timeout=10)
    assert session.task == "fold the cloth", "the hand-off must not change the task"


def test_labeling_out_of_turn_is_refused(live_session):
    with pytest.raises(SessionConflict) as excinfo:
        live_session.label(True)
    assert "awaiting_label" in (excinfo.value.hint or "")


def test_stop_ends_the_session(live_session):
    session = live_session
    session.stop()
    assert wait_for(lambda: not session.alive, timeout=session_mod.HOME_EXIT_GRACE)
    assert session.end_reason == "stop"
    assert session.state is State.STOPPED


def test_stop_parks_the_arm_before_quitting(live_session):
    """The driver's reset runs at the START of a rollout, so ending a session otherwise
    leaves the arm wherever the last plan put it."""
    session = live_session
    session.stop(park=True)
    assert wait_for(lambda: not session.alive, timeout=session_mod.HOME_EXIT_GRACE)
    assert any("going home" in line["text"] for line in session.logs())


def test_max_episodes_stops_the_session(profile, tmp_path, monkeypatch):
    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
    session = Session(profile, FakeRuntime(tmp_path / "rt"), max_episodes=1)
    session.start()
    assert wait_for(lambda: session.state is State.AWAITING_TASK)
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_LABEL)
    session.label(True)
    assert wait_for(lambda: not session.alive, timeout=session_mod.HOME_EXIT_GRACE)


def test_subscribers_see_state_and_log_messages(live_session):
    received = []
    live_session.subscribe(received.append)
    live_session.next_task()
    assert wait_for(lambda: any(m.get("type") == "state" for m in received))
    assert wait_for(lambda: any(m.get("type") == "log" for m in received))


def test_a_broken_subscriber_cannot_take_down_the_session(live_session):
    """A disconnected browser must never stop the process driving a physical robot."""

    def explode(_message):
        raise RuntimeError("boom")

    live_session.subscribe(explode)
    live_session.next_task()
    assert wait_for(lambda: live_session.state is State.AWAITING_LABEL)


def test_manager_refuses_a_second_session_for_a_profile(profile, tmp_path, monkeypatch):
    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
    manager = session_mod.SessionManager()
    runtime = FakeRuntime(tmp_path / "rt")
    first = manager.create(profile, runtime)
    try:
        assert wait_for(lambda: first.state is State.AWAITING_TASK)
        with pytest.raises(SessionConflict):
            manager.create(profile, runtime)
    finally:
        first.stop(park=False)
        first.wait(timeout=5)


def test_the_driver_exits_when_its_input_closes(profile, tmp_path, monkeypatch):
    """A driver whose launcher dies must stop, not spin.

    readline() returns "" forever once the pipe is closed, so a prompt loop that treats that
    as unrecognised input runs at full speed. One did, and wrote a 63 GB events file before
    anyone noticed. The real driver reads with input(), which raises EOFError.
    """
    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
    session = Session(profile, FakeRuntime(tmp_path / "rt"))
    session.start()
    assert wait_for(lambda: session.state is State.AWAITING_TASK)

    session._proc.stdin.close()
    assert session.wait(timeout=10) is not None, "the driver kept running after stdin closed"

    events = Path(session.summary()["events_file"])
    settled = events.stat().st_size
    time.sleep(0.5)
    assert events.stat().st_size == settled, "the events file is still growing"


def test_missing_gemini_key_is_caught_before_spawning(profile, tmp_path, monkeypatch):
    monkeypatch.setattr(secrets, "gemini_api_key", lambda: None)
    session = Session(profile, FakeRuntime(tmp_path / "rt"))
    with pytest.raises(Exception) as excinfo:
        session.start()
    assert "Gemini" in str(excinfo.value)


def test_a_profile_without_cameras_cannot_collect(tmp_path, monkeypatch):
    from tandem.core import profiles

    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
    bare = profiles.Profile.model_validate({"name": "bare"})
    profiles.save(bare)
    session = Session(bare, FakeRuntime(tmp_path / "rt"))
    with pytest.raises(Exception) as excinfo:
        session.start()
    assert "no cameras" in str(excinfo.value)
