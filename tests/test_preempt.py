"""A preempt that lands while an attempt is still running: the path a stand-in that finishes instantly
never takes.

The session's own tests preempt either at the task prompt, where no attempt runs, or after the stand-in
has already finished the task -- so the branch that unwinds a running attempt (``_Preempted``) was never
executed, and a change that dropped the filing of a preempted trial passed the whole suite. Here the
attempt is held at a human phase, after its first robot leg is on disk, and preempted there: the leg
must still be filed -- under failure/, as an aborted trial, with no label prompt, because a plan cut
off part-way is not something a label could make a demonstration of -- the planner must stay warm, and
the session must say the attempt was aborted.
"""

from __future__ import annotations

import json

import test_hitl
from helpers import wait_for

from tandem.core.session import State

# The phase-planning tests' own session fixture, shared rather than copied.
phase_session = test_hitl.phase_session


def test_a_preempt_at_a_human_phase_files_the_leg_already_recorded_as_aborted(phase_session, profile):
    session, backends, _ = phase_session()
    events: list[dict] = []
    session.subscribe(events.append)
    states: list[str] = []
    session.subscribe(lambda e: states.append(e.get("state")) if e.get("type") == "state" else None)

    session.next_task()
    # The robot phase ran and recorded its leg; the human phase is waiting for the operator.
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE), f"stuck in {session.state}"
    assert len(backends[-1].legs) == 1

    session.preempt()
    # Straight back to the task prompt, with the leg filed: only filing merges it, and a leg left in
    # eval/ is joined to nothing, for ever.
    trajectories = profile.trajectories_dir()
    assert wait_for(lambda: session.state is State.AWAITING_TASK), f"stuck in {session.state}"
    assert State.AWAITING_LABEL.value not in states, "an unfinished plan was offered for a label"
    assert len(backends[-1].legs) == 1, "the second robot phase never runs after a preempt"
    # Preempting abandons an attempt, not the session: the planner is still warm.
    assert session.alive and session.end_reason is None
    assert not backends[-1].closed
    assert wait_for(
        lambda: any(e.get("type") == "event" and e.get("event") == "rollout_aborted" for e in events)
    ), "the attempt was not reported as aborted"

    assert wait_for(lambda: not any((trajectories / "eval").iterdir())), "the leg was left unfiled in eval/"
    assert not (trajectories / "success").is_dir() or not any((trajectories / "success").iterdir())
    # Waited for, not only checked: the record is written before the background merge, which must
    # finish inside this test's temporary data root rather than after it is torn down.
    failure = trajectories / "failure"
    assert wait_for(lambda: failure.is_dir() and any((d / "hitl.json").is_file() for d in failure.iterdir()))
    (record,) = [json.loads(path.read_text()) for path in failure.glob("*/hitl.json")]
    assert (record["outcome"], record["failure_stage"]) == ("aborted", None)
    assert session.aborted_count == 1 and session.labeled_count == 0
