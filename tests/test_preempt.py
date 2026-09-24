"""A preempt that lands while an attempt is still running: the path a stand-in that finishes instantly
never takes.

The session's own tests preempt either at the task prompt, where no attempt runs, or after the stand-in
has already finished the task -- so the branch that unwinds a running attempt (``_Preempted``) was never
executed, and a change that dropped the label for a preempted trial passed the whole suite. Here the
attempt is held at a human phase, after its first robot leg is on disk, and preempted there: the leg
must still be labeled and filed, the planner must stay warm, and the session must say the attempt was
aborted.
"""

from __future__ import annotations

import test_hitl
from helpers import wait_for

from tandem.core.session import State

# The phase-planning tests' own session fixture, shared rather than copied.
phase_session = test_hitl.phase_session


def test_a_preempt_at_a_human_phase_still_labels_and_files_the_leg_already_recorded(phase_session, profile):
    session, backends, _ = phase_session()
    events: list[dict] = []
    session.subscribe(events.append)

    session.next_task()
    # The robot phase ran and recorded its leg; the human phase is waiting for the operator.
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE), f"stuck in {session.state}"
    assert len(backends[-1].legs) == 1

    session.preempt()
    # Exactly the label prompt: a leg is on disk, and only a label files and merges it. Straight back to
    # the task prompt would leave it in eval/ for ever, joined to nothing.
    assert wait_for(lambda: session.state is State.AWAITING_LABEL), f"stuck in {session.state}"
    assert len(backends[-1].legs) == 1, "the second robot phase never runs after a preempt"

    session.label(True)
    assert wait_for(lambda: session.state is State.AWAITING_TASK), f"stuck in {session.state}"
    # Preempting abandons an attempt, not the session: the planner is still warm.
    assert session.alive and session.end_reason is None
    assert not backends[-1].closed
    assert wait_for(
        lambda: any(e.get("type") == "event" and e.get("event") == "rollout_aborted" for e in events)
    ), "the attempt was not reported as aborted"

    trajectories = profile.trajectories_dir()
    assert wait_for(lambda: not any((trajectories / "eval").iterdir())), "the leg was left unlabeled in eval/"
    assert any((trajectories / "success").iterdir()), "the labeled leg was not filed under success/"
    # Waited for, not only checked: the record is written by the background merge, which must finish
    # inside this test's temporary data root rather than after it is torn down.
    assert wait_for(lambda: any((d / "hitl.json").is_file() for d in (trajectories / "success").iterdir()))
