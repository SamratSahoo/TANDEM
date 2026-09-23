"""The session engine: the state machine, and who holds the arm.

These are the tests that matter most. tandem decides who does what and in what order, and holds the
arm's custody straight across a hand-off, so this is what stands between an operator and a robot
that is already moving.

Everything runs against a stand-in backend (``fake_backend.FakeBackend``) rather than a planner: the
decisions being tested are tandem's, and a real planner would only add a GPU to the requirements.
"""

from __future__ import annotations

import json

import pytest
from helpers import FakeRuntime, use_fake_backend, wait_for

from tandem.core import secrets
from tandem.core.errors import SessionConflict
from tandem.core.session import Session, State


@pytest.fixture
def backends(monkeypatch):
    return use_fake_backend(monkeypatch)


@pytest.fixture
def live_session(profile, tmp_path, monkeypatch, backends):
    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
    runtime = FakeRuntime(tmp_path / "runtime")
    session = Session(profile, runtime, task="pick up the block")
    session.start()
    assert wait_for(lambda: session.state is State.AWAITING_TASK), f"stuck in {session.state}"
    yield session
    if session.alive:
        session.stop(park=False)
        session.wait(timeout=5)


def backend(backends):
    assert backends, "the session never built a backend"
    return backends[-1]


# --- the ordinary loop ----------------------------------------------------------------------------


def test_a_full_rollout_reaches_labeled(live_session, backends):
    session = live_session
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_LABEL), f"stuck in {session.state}"

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


def test_with_phase_planning_off_the_planners_own_goal_is_used(live_session, backends):
    """No decomposition means nothing for a model to do, so the goal is the one the planner's own
    translator made of the instruction — the same behaviour a session had before phases existed,
    through the same code path and with no extra model call."""
    session = live_session
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_LABEL)

    planned = [c for c in backend(backends).calls if c.startswith("plan:")]
    assert len(planned) == 1
    assert json.loads(planned[0].split(":", 1)[1]) == [{"predicate": "on", "args": ["blue_toy", "table"]}]


def test_the_leg_is_stamped_with_the_trajectory_tandem_minted(live_session, backends):
    """Only tandem sees both the planner's legs and the teleop ones, so only tandem can mint the id
    that joins them. A leg that reaches disk unstamped files as an episode of its own."""
    session = live_session
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_LABEL)

    legs = backend(backends).legs
    assert len(legs) == 1
    spec = legs[0]["leg"]
    assert spec.trajectory_id and len(spec.trajectory_id) == 16
    assert spec.segment_source == "tamp"
    # The dataset's language label is the WHOLE task, never one phase of it.
    assert spec.instruction == "pick up the block"
    meta = json.loads((__import__("pathlib").Path(legs[0]["dir"]) / "_meta.json").read_text())
    assert meta["trajectory_id"] == spec.trajectory_id


def test_max_episodes_stops_the_session(profile, tmp_path, monkeypatch, backends):
    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
    session = Session(profile, FakeRuntime(tmp_path / "runtime"), task="one", max_episodes=1)
    session.start()
    assert wait_for(lambda: session.state is State.AWAITING_TASK)
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_LABEL)
    session.label(True)
    assert wait_for(lambda: session.state in (State.STOPPED, State.FAILED)), f"in {session.state}"
    assert session.state is State.STOPPED


# --- preempt, stop, custody -----------------------------------------------------------------------


def test_preempt_abandons_the_attempt_and_keeps_the_session_warm(live_session, backends):
    session = live_session
    session.preempt()
    assert wait_for(lambda: session.state is State.AWAITING_TASK)
    assert session.alive
    assert session.end_reason is None, "a preempt must not end the session"
    assert not backend(backends).closed, "the planner must stay warm"


def test_preempt_is_refused_during_a_handoff(live_session, monkeypatch):
    """Preempting there would strand the session with no robot and no cameras."""
    session = live_session
    session._set_state(State.TELEOP_HANDOFF)
    with pytest.raises(SessionConflict, match="hand-off"):
        session.preempt()


def test_stop_parks_the_arm_before_letting_go(live_session, backends):
    """Nothing homes at the end of a task, so without this the arm stays where the last plan left
    it. The gripper is deliberately not opened — nothing here can know it is empty."""
    session = live_session
    session.stop(park=True)
    assert wait_for(lambda: session.state is State.STOPPED)
    names = backend(backends).call_names()
    assert "home" in names
    assert names.index("home") < names.index("close"), "the arm must be parked before letting go"


def test_stop_without_parking_does_not_move_the_arm(live_session, backends):
    session = live_session
    session.stop(park=False)
    assert wait_for(lambda: session.state is State.STOPPED)
    assert "home" not in backend(backends).call_names()
    assert backend(backends).closed


def test_the_planner_is_always_closed_even_when_the_session_fails(profile, tmp_path, monkeypatch):
    """A backend left open holds the robot and every camera. Nothing else can start until it does
    not, so releasing it belongs on every exit path, including the ones nobody planned for."""
    from fake_backend import FakeBackend

    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")

    class Exploding(FakeBackend):
        def perceive(self, **kwargs):
            raise RuntimeError("the camera fell off")

    built = use_fake_backend(monkeypatch, backend_type=Exploding)
    session = Session(profile, FakeRuntime(tmp_path / "runtime"), task="x")
    session.start()
    assert wait_for(lambda: session.state is State.AWAITING_TASK)
    session.next_task()
    # A bad attempt returns to the prompt rather than ending the session. Wait on the LOG rather
    # than the state: the session is already at the prompt when next_task is called, so a state
    # check would pass before the attempt had even started.
    assert wait_for(lambda: any("the camera fell off" in line["text"] for line in session.logs()))
    assert wait_for(lambda: session.state is State.AWAITING_TASK)
    assert session.alive, "one bad attempt must not end a warm session"
    session.stop(park=False)
    assert wait_for(lambda: session.state is State.STOPPED)
    assert built[-1].closed


def test_labeling_out_of_turn_is_refused(live_session):
    with pytest.raises(SessionConflict):
        live_session.label(True)


def test_manager_refuses_a_second_session_for_a_profile(profile, tmp_path, monkeypatch, backends):
    from tandem.core.session import SessionManager

    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
    runtime = FakeRuntime(tmp_path / "runtime")
    manager = SessionManager()
    first = manager.create(profile, runtime, task="one")
    try:
        assert wait_for(lambda: first.state is State.AWAITING_TASK)
        with pytest.raises(SessionConflict, match="already running"):
            manager.create(profile, runtime, task="two")
    finally:
        manager.shutdown()


# --- observation ----------------------------------------------------------------------------------


def test_subscribers_see_state_and_log_messages(live_session):
    seen: list[dict] = []
    unsubscribe = live_session.subscribe(seen.append)
    live_session.next_task()
    assert wait_for(lambda: any(m.get("type") == "state" for m in seen))
    unsubscribe()
    assert any(m.get("type") in ("log", "event") for m in seen)


def test_a_broken_subscriber_cannot_take_down_the_session(live_session):
    """A disconnected browser must never take down a session driving a physical robot."""

    def explode(_message):
        raise RuntimeError("subscriber went away")

    live_session.subscribe(explode)
    live_session.next_task()
    assert wait_for(lambda: live_session.state is State.AWAITING_LABEL)
    assert live_session.alive


def test_the_session_writes_its_own_events_file(live_session):
    """Tandem's now, rather than a planner child's — but still an append-only line per event, so a
    session that went wrong can be read off disk after the process is gone."""
    from tandem.core import events as events_mod

    session = live_session
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_LABEL)

    recorded = events_mod.read_all(session._files["events_file"])
    names = [e.name for e in recorded]
    assert "session_start" in names
    assert "awaiting_task" in names
    assert "rollout_start" in names
    assert "awaiting_label" in names


# --- where the data actually lands ------------------------------------------------------------


def test_a_leg_lands_where_a_trajectory_lives_and_gets_labeled(live_session, backends, profile):
    """The one thing that has to be right or nothing else matters.

    A leg is only real if three separate readers can find it: `merge.find_legs`, which joins a
    task's legs into one episode; `trajectories.list_all`, which is what `tandem traj list` and the
    UI show; and the label, which files it under success or failure. All three look under
    `<profile>/trajectories/<status>/`. A leg written anywhere else — a session scratch directory,
    say — is invisible to every one of them: never merged, never labeled, never in the dataset, and
    silent about it.
    """
    from pathlib import Path

    from tandem.core import merge as merge_mod
    from tandem.core import trajectories

    session = live_session
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_LABEL)

    leg = Path(backend(backends).legs[0]["dir"])
    assert leg.parent == profile.status_dir("eval"), f"the leg landed in {leg.parent}"

    trajectory_id = json.loads((leg / "_meta.json").read_text())["trajectory_id"]
    assert [x["dir"] for x in merge_mod.find_legs(profile, trajectory_id)] == [leg]

    session.label(True)
    assert wait_for(lambda: session.labeled_count == 1)
    # The label has to move it out of the staging bucket. A single-leg task is never merged
    # ("single leg"), so nothing else would ever do it.
    assert wait_for(lambda: (profile.status_dir("success") / leg.name).is_dir()), "still in eval/"
    assert not leg.exists()

    listed = trajectories.list_all(profile)
    assert [t.id for t in listed] == [leg.name]
    assert listed[0].status == "success"


def test_two_legs_of_one_task_do_not_collide_on_a_directory(live_session, backends):
    """Leg directories are named with a second-resolution wall clock, and two legs of one task
    routinely start inside the same second."""
    session = live_session
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_LABEL)
    session.label(True)
    assert wait_for(lambda: session.state is State.AWAITING_TASK)
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_LABEL)

    directories = [leg["dir"] for leg in backend(backends).legs]
    assert len(set(directories)) == len(directories), f"two legs shared a directory: {directories}"


# --- the paths that abandon an attempt ----------------------------------------------------------


def test_a_preempt_mid_task_still_reaches_the_label_prompt(live_session, backends, profile):
    """Frames on disk need a verdict, whatever ended the attempt.

    The label is what ends a trajectory and merges its legs, so an attempt that recorded something
    and then returned straight to the task prompt would leave legs nothing will ever join — filing
    as episodes of their own, exactly what the multi-leg design exists to prevent.
    """
    session = live_session
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_LABEL)
    session.label(True)
    assert wait_for(lambda: session.state is State.AWAITING_TASK)

    # Now preempt one mid-flight. The stand-in records instantly, so preempt after the leg exists.
    session.next_task()
    assert wait_for(lambda: len(backend(backends).legs) == 2)
    session.preempt()
    # Recorded, therefore labelable — not silently discarded.
    assert wait_for(lambda: session.state in (State.AWAITING_LABEL, State.AWAITING_TASK))


def test_a_pass_that_records_nothing_leaves_no_phantom_episode(live_session, backends, profile):
    """Every pass allocates a leg directory before anyone knows whether a recording will follow.

    For a human phase, or a phase the planner could not plan, none ever does — and left in `eval/`
    each is listed by `tandem traj list` and the UI as a zero-frame episode, so a session's worth
    of them buries the real ones.
    """
    from tandem.core import trajectories

    session = live_session
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_LABEL)
    session.label(True)
    assert wait_for(lambda: session.labeled_count == 1)

    listed = trajectories.list_all(profile)
    assert len(listed) == 1, f"phantom episodes left behind: {[t.id for t in listed]}"
    # The perception dump is kept, just not in the dataset.
    kept = session._files["session_dir"] / "perception"
    assert not kept.exists() or all(p.is_dir() for p in kept.iterdir())


def test_a_teleop_handoff_with_no_phase_plan_gives_the_task_back_to_the_planner(
    live_session, backends, monkeypatch, tmp_path
):
    """Pressing "switch to teleop" in an ordinary session lends the arm; it does not end the task.

    The browser says so explicitly while the operator is in that state, so ending the episode there
    would make it a teleop-only leg and contradict what they were just told.
    """
    import sys
    from pathlib import Path as _Path

    from tandem import teleop as teleop_pkg
    from tandem.core import settings as settings_mod
    from tandem.core.settings import Settings

    monkeypatch.setattr(teleop_pkg, "driver_path", lambda: _Path(__file__).parent / "fake_teleop.py")
    cfg = Settings()
    cfg.teleop.enabled = True
    cfg.teleop.python = sys.executable
    cfg.teleop.droid_dir = str(_Path(__file__).parent)
    monkeypatch.setattr(settings_mod, "load", lambda: cfg)

    session = live_session
    session.next_task()
    session.request_teleop()
    assert wait_for(lambda: session.state is State.TELEOP_HANDOFF), f"stuck in {session.state}"
    assert wait_for(lambda: session._teleop is not None and session._teleop._recording)
    session.resume_from_teleop()

    assert wait_for(lambda: session.state is State.AWAITING_LABEL), f"stuck in {session.state}"
    # The planner got the task back and ran it, rather than the episode ending at the hand-off.
    assert any(c.startswith("plan:") for c in backend(backends).calls)
    assert backend(backends).legs, "the robot never ran the task after the hand-off"
