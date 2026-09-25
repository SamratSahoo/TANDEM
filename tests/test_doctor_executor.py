"""`tandem doctor` says whether a profile's human steps can be carried out -- and recorded -- here.

Where two items meet: the executor catalog (`tandem executors`) says what each executor still needs on
this machine, and the phase loop now completes a human step while recording only through a leg of
``hitl.human_executor``. Put together, an executor that needs setup leaves every human step with one
answer, give up, unless ``hitl.allow_unrecorded_human_phase`` is set; and one that will not load stops
the session before it warms. doctor is where "will this work?" is answered before anyone is standing
at the arm, so it says both.
"""

from __future__ import annotations

import pytest

from tandem.cli.doctor import collect_checks
from tandem.core import profiles
from tandem.core import settings as settings_mod
from tandem.executors import base as executors


class ReadyExecutor:
    """An executor that needs nothing on this machine."""

    segment_source = "policy"
    display_name = "Ready policy"
    summary = "Always ready."

    def __init__(self, ctx) -> None:
        self.ctx = ctx


@pytest.fixture(autouse=True)
def only_registered_executors(monkeypatch):
    monkeypatch.setattr(executors, "_registered", dict(executors._registered))


def _row(profile):
    profiles.save(profile)
    rows = [
        c
        for c in collect_checks(profile_name=profile.name, probe_hardware=False)
        if c.name == "human executor"
    ]
    assert len(rows) <= 1
    return rows[0] if rows else None


def test_no_row_when_there_are_no_human_steps(profile):
    profile.hitl.enabled = False
    assert _row(profile) is None


def test_teleop_this_machine_lacks_is_a_warning_that_says_what_it_costs(profile):
    profile.hitl.enabled = True
    assert not settings_mod.load().teleop.enabled
    row = _row(profile)
    assert row.state == "warn"
    assert row.detail.startswith("teleop needs setup: ") and "teleop is not enabled" in row.detail
    assert "give up" in row.hint and "tandem executors list" in row.hint
    assert "hitl.allow_unrecorded_human_phase" in row.hint


def test_with_unrecorded_steps_allowed_the_warning_says_they_will_stand(profile):
    profile.hitl.enabled = True
    profile.hitl.allow_unrecorded_human_phase = True
    row = _row(profile)
    assert row.state == "warn"
    assert "give up" not in row.hint and "unrecorded" in row.hint


def test_a_ready_executor_is_ok(profile):
    executors.register_human_executor("ready", ReadyExecutor)
    profile.hitl.enabled = True
    profile.hitl.human_executor = "ready"
    row = _row(profile)
    assert row.state == "ok" and row.detail == "ready · Ready policy"


def test_an_executor_that_will_not_load_fails_as_the_session_would(profile):
    executors.register_human_executor("broken", "no_such_module_anywhere:FACTORY")
    profile.hitl.enabled = True
    profile.hitl.human_executor = "broken"
    row = _row(profile)
    assert row.state == "fail" and "could not be loaded" in row.detail
