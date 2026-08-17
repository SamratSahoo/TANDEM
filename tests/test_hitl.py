"""Phase planning — the deep human-in-the-loop mode.

A VLM splits the instruction into robot and human phases; the human ones are handed over with
written instructions and checked from a photo afterwards. These cover the parts tandem owns:
the config that turns it on, and the prompt protocol it drives.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from tandem.core import render, secrets
from tandem.core.errors import SessionConflict
from tandem.core.profiles import Profile
from tandem.core.session import Session, State
from tests.test_session import FakeRuntime, wait_for

# --- config ----------------------------------------------------------------


def test_phase_planning_is_off_by_default(profile):
    """Disabled, the driver never imports the package at all — so a profile that has never
    heard of this behaves exactly as it always did."""
    assert profile.hitl.enabled is False
    assert render.write_hitl_config(profile, profile.dir() / "hitl.json") is None


def test_enabling_writes_the_whole_block(profile, tmp_path):
    """Emitted whole rather than sparsely: the reader rejects unknown keys loudly, so the file
    doubles as a complete record of what the run was configured with."""
    profile.hitl.enabled = True
    path = render.write_hitl_config(profile, tmp_path / "hitl.json")
    assert path is not None

    written = json.loads(path.read_text())
    assert written["enabled"] is True
    assert written["proposal_model"] == "gemini-2.5-pro"
    assert written["vlm_model"] == "gemini-2.5-flash"
    assert written["verify_enforced"] is True
    # Every field of the upstream dataclass, so nothing is left to a default that may drift.
    assert set(written) == set(Profile.model_fields["hitl"].annotation.model_fields)


def test_unknown_hitl_key_is_rejected():
    """A misspelled key here silently disables the feature the config was written to turn on,
    so it is a hard error rather than a dropped key."""
    with pytest.raises(ValidationError) as excinfo:
        Profile.model_validate({"name": "x", "hitl": {"enabled": True, "verify_retires": 2}})
    assert "verify_retires" in str(excinfo.value)


def test_negative_retries_are_rejected():
    with pytest.raises(ValidationError):
        Profile.model_validate({"name": "x", "hitl": {"verify_retries": -1}})


def test_cache_path_resolves_against_the_profile(profile):
    profile.hitl.enabled = True
    profile.hitl.cache_path = "proposals.sqlite"
    rendered = render.render_hitl_config(profile)
    assert rendered["cache_path"] == str(profile.dir() / "proposals.sqlite")


# --- the prompt protocol ---------------------------------------------------


@pytest.fixture
def hitl_session(profile, tmp_path, monkeypatch):
    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
    profile.hitl.enabled = True
    profile.hitl.verify_retries = 0  # verify first time, so the happy path is the short one
    from tandem.core import profiles as profiles_mod

    profiles_mod.save(profile)

    session = Session(profile, FakeRuntime(tmp_path / "runtime"), task="fold the cloth over the toy")
    session.start()
    assert wait_for(lambda: session.state is State.AWAITING_TASK), f"stuck in {session.state}"
    yield session
    if session.alive:
        session.stop(park=False)
        session.wait(timeout=5)


def test_the_driver_is_told_phase_planning_is_on(hitl_session):
    assert hitl_session.hitl_enabled is True
    assert any("--hitl-config" in line["text"] for line in hitl_session.logs())


def test_a_human_phase_carries_its_instructions_and_expectations(hitl_session):
    """The expectations are the same list the model will be asked about — an operator judged
    against a standard they were never shown stops trusting the verification."""
    session = hitl_session
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE, timeout=10)

    phase = session.human_phase
    assert phase is not None
    assert phase.description == "fold the cloth over the toy"
    assert "Fold the near edge" in phase.instructions
    assert phase.expected == ["the cloth is folded over the toy"]
    assert (phase.index, phase.total) == (1, 2)
    assert session.summary()["human_phase"]["instructions"] == phase.instructions


def test_completing_a_phase_verifies_and_carries_on(hitl_session):
    session = hitl_session
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE, timeout=10)

    session.complete_human_phase()
    # Verified, so the phase clears and the run continues in the same trajectory.
    assert wait_for(lambda: session.human_phase is None, timeout=10)
    assert wait_for(lambda: session.phase_progress == (1, 2), timeout=10)
    assert wait_for(lambda: session.state is State.AWAITING_LABEL, timeout=10)


def test_aborting_a_phase_abandons_the_attempt(hitl_session):
    session = hitl_session
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE, timeout=10)

    session.abort_human_phase()
    assert wait_for(lambda: session.state is State.AWAITING_TASK, timeout=10)
    assert session.human_phase is None
    assert session.labeled_count == 0


def test_phase_actions_are_refused_outside_the_prompt(hitl_session):
    with pytest.raises(SessionConflict) as excinfo:
        hitl_session.complete_human_phase()
    assert "awaiting_human_phase" in (excinfo.value.hint or "")
    with pytest.raises(SessionConflict):
        hitl_session.abort_human_phase()


def test_a_failed_check_says_what_is_still_missing(profile, tmp_path, monkeypatch):
    """A demonstration is not thrown away on one bad classifier call, and the operator is told
    what the model thought was wrong rather than just 'no'."""
    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
    profile.hitl.enabled = True
    profile.hitl.verify_retries = 1  # the fake driver fails the first check when retries remain
    from tandem.core import profiles as profiles_mod

    profiles_mod.save(profile)

    session = Session(profile, FakeRuntime(tmp_path / "rt"), task="fold the cloth")
    session.start()
    try:
        assert wait_for(lambda: session.state is State.AWAITING_TASK)
        session.next_task()
        assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE, timeout=10)

        session.complete_human_phase()
        # Re-asked, with what the model said was still missing and the attempt counter moved on.
        assert wait_for(lambda: session.human_phase is not None and session.human_phase.attempt == 2, timeout=10)
        phase = session.human_phase
        assert phase.missing == ["the cloth is folded over the toy — a corner of the toy is still visible"]

        session.complete_human_phase()
        assert wait_for(lambda: session.state is State.AWAITING_LABEL, timeout=10)
    finally:
        if session.alive:
            session.stop(park=False)
            session.wait(timeout=5)


def test_a_disabled_profile_never_reaches_a_human_phase(profile, tmp_path, monkeypatch):
    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
    session = Session(profile, FakeRuntime(tmp_path / "rt"))
    session.start()
    try:
        assert wait_for(lambda: session.state is State.AWAITING_TASK)
        session.next_task()
        assert wait_for(lambda: session.state is State.AWAITING_LABEL, timeout=10)
        assert session.human_phase is None
        assert not any("--hitl-config" in line["text"] for line in session.logs())
    finally:
        if session.alive:
            session.stop(park=False)
            session.wait(timeout=5)


# --- the recorded plan -----------------------------------------------------


def test_hitl_json_is_surfaced_on_a_trajectory(profile, make_trajectory):
    from tandem.core import trajectories

    directory = make_trajectory(profile, "2026-01-01_00-00-00")
    (directory / "hitl.json").write_text(json.dumps({
        "instruction": "put the toy on the cloth, then fold it",
        "phases": [
            {"executor": "robot", "description": "put the toy on the cloth", "atoms": ["On(toy, cloth)"]},
            {"executor": "human", "description": "fold the cloth", "instructions": "Fold it over."},
        ],
        "verifications": [{"atom": "Folded(cloth)", "holds": True}],
    }))
    (directory / "vlm").mkdir()

    traj = trajectories.find(profile, "2026-01-01_00-00-00")
    assert traj.has_hitl is True
    assert traj.n_phases == 2
    assert traj.n_human_phases == 1
    assert traj.has_vlm_log is True

    plan = trajectories.read_hitl(directory)
    assert plan["phases"][1]["executor"] == "human"


def test_a_rollout_without_phase_planning_reports_none(profile, make_trajectory):
    from tandem.core import trajectories

    make_trajectory(profile, "2026-01-01_00-00-00")
    traj = trajectories.find(profile, "2026-01-01_00-00-00")
    assert traj.has_hitl is False
    assert traj.n_phases == 0
