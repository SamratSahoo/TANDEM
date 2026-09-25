"""A session resolves where it writes once, when it starts, and never again from a thread of its own.

A merge runs on a thread of its own, and can outlive the environment the session was started in. One
that outlived its test resolved the data root again after the test had put its environment back, and
wrote a trajectory into the real ~/tandem-data. Every leg, record and merge of a session lands where
its profile was when the session started, whatever the environment says by the time it runs.
"""

from __future__ import annotations

from pathlib import Path

from helpers import use_fake_backend, wait_for

from tandem.core import secrets
from tandem.core.session import Session, State


def _moved_elsewhere(monkeypatch, tmp_path) -> Path:
    """What the environment says once the test that started the session has put its own back."""
    elsewhere = tmp_path / "elsewhere"
    monkeypatch.setenv("TANDEM_DATA_ROOT", str(elsewhere / "data"))
    monkeypatch.setenv("HOME", str(elsewhere / "home"))
    return elsewhere


def test_a_merge_after_the_environment_moved_files_where_the_session_started(profile, tmp_path, monkeypatch):
    backends = use_fake_backend(monkeypatch)
    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
    session = Session(profile, task="pick up the block").start()
    try:
        assert wait_for(lambda: session.state is State.AWAITING_TASK)
        session.next_task()
        assert wait_for(lambda: session.state is State.AWAITING_LABEL)
        leg = Path(backends[-1].legs[0]["dir"])
        assert leg.parent == profile.status_dir("eval")
        filed = profile.status_dir("success") / leg.name

        elsewhere = _moved_elsewhere(monkeypatch, tmp_path)
        session.label(True)
        assert wait_for(lambda: filed.is_dir()), "the trial was not filed where the session started"
        assert not elsewhere.exists(), f"the merge wrote into {sorted(elsewhere.rglob('*'))}"
    finally:
        session.stop(park=False)
        session.wait(timeout=5)


def test_the_proposal_cache_is_where_it_was_when_the_session_started(profile, tmp_path, monkeypatch):
    use_fake_backend(monkeypatch)
    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    profile.hitl.cache_path = "~/proposals.sqlite"
    session = Session(profile, task="pick up the block").start()
    try:
        _moved_elsewhere(monkeypatch, tmp_path)
        assert session._planning_config().cache_path == str(tmp_path / "home" / "proposals.sqlite")
    finally:
        session.stop(park=False)
        session.wait(timeout=5)


def test_a_pinned_profile_keeps_its_directory_and_is_still_the_same_profile(profile, tmp_path, monkeypatch):
    pinned = profile.pinned()
    before, trajectories = pinned.file(), pinned.trajectories_dir()
    _moved_elsewhere(monkeypatch, tmp_path)

    assert pinned.file() == before and pinned.trajectories_dir() == trajectories
    assert trajectories == before.parent.parent / "trajectories" / profile.name
    assert profile.file() == tmp_path / "elsewhere" / "data" / "profiles" / f"{profile.name}.yml", (
        "only the copy is pinned"
    )
    # Where it lives is not a setting: nothing of the pin reaches what is written into the profile's file.
    assert pinned.model_dump() == profile.model_dump()
