"""Phase planning, end to end through the session engine.

A vision model splits the instruction into robot and human phases; the human ones are handed over
with written instructions and checked from a photo afterwards. tandem owns all of that now, so these
tests drive the real session engine against a stand-in backend and a canned model — no planner, no
GPU, no robot, and no network.

The proposal validator itself is covered in test_planning.py; what is tested here is the part only
the session can do: dispatching each phase to the right executor, holding the arm's custody across
the hand-off, retrying a step that did not verify, and leaving an audit trail behind.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest
from helpers import FakeGemini, use_fake_backend, wait_for
from pydantic import ValidationError

from tandem.core import profiles as profiles_mod
from tandem.core import secrets
from tandem.core.errors import SessionConflict
from tandem.core.profiles import Profile
from tandem.core.session import Session, State

# The three-phase task that forced the whole design: the box has to be OPENED before anything can be
# put in it, which a final-state goal cannot say and a one-way ordering cannot express.
PLAN = {
    "new_predicates": [
        {"name": "IsOpen", "instructions": "the container {0} is open, so its interior is visible"}
    ],
    "phases": [
        {
            "executor": "robot",
            "description": "take the toy off the box",
            "atoms": [{"predicate": "On", "args": ["blue_toy", "table"]}],
        },
        {
            "executor": "human",
            "description": "open the box",
            "instructions": "Open the white_box and fold its flaps back.",
            "atoms": [{"predicate": "IsOpen", "args": ["white_box"]}],
            "operator": {
                "name": "Open",
                "args": ["white_box"],
                "preconditions": [{"predicate": "HandEmpty", "args": []}],
                "add_effects": [{"predicate": "IsOpen", "args": ["white_box"]}],
                "delete_effects": [],
            },
        },
        {
            "executor": "robot",
            "description": "put the toy inside the box",
            "atoms": [{"predicate": "On", "args": ["blue_toy", "white_box"]}],
        },
    ],
}
HOLDS = json.dumps({"holds": True, "reason": "the flaps are folded back"})
DOES_NOT_HOLD = json.dumps({"holds": False, "reason": "a flap is still closed over the opening"})


# --- config ---------------------------------------------------------------------------------------


def test_phase_planning_is_off_by_default(profile):
    assert profile.hitl.enabled is False
    # The paper counts a TAMP failure as a trial failure, so that is what a fresh profile does.
    assert profile.hitl.on_robot_phase_failure == "abort"
    assert profile.planner.backend == "tiptop"


def test_unknown_hitl_key_is_rejected():
    """A misspelled key here silently disables the feature the config was written to turn on,
    so it is a hard error rather than a dropped key."""
    with pytest.raises(ValidationError) as excinfo:
        Profile.model_validate({"name": "x", "hitl": {"enabled": True, "verify_retires": 2}})
    assert "verify_retires" in str(excinfo.value)


def test_negative_retries_are_rejected():
    with pytest.raises(ValidationError):
        Profile.model_validate({"name": "x", "hitl": {"verify_retries": -1}})


def test_a_max_attempts_of_zero_is_refused_where_the_user_can_see_it():
    """The two validation layers have to agree.

    `max_attempts: 0` means never asking the model at all, which is not a configuration of the
    feature. It used to pass here and then raise out of the planner package as a bare ValueError,
    with the traceback the CLI's error boundary exists to suppress."""
    with pytest.raises(ValidationError, match="max_attempts"):
        Profile.model_validate({"name": "x", "hitl": {"max_attempts": 0}})
    assert Profile.model_validate({"name": "x", "hitl": {"max_attempts": 1}}).hitl.max_attempts == 1
    # verify_retries keeps its own, different floor: zero retries is a real choice.
    assert Profile.model_validate({"name": "x", "hitl": {"verify_retries": 0}}).hitl.verify_retries == 0


def test_a_relative_cache_path_reads_the_same_from_every_command(profile):
    """A session, `tandem plan --profile` and `tandem doctor` all read this key.

    Resolving it differently in any of them means they open DIFFERENT SQLite files, so the proposal
    cache never hits across the very loop it exists for — and since opening one creates its parent
    directories, the odd one out silently litters a second cache wherever it was run from."""
    profile.hitl.enabled = True
    profile.hitl.cache_path = "caches/proposals.sqlite"

    resolved = profiles_mod.resolve_cache_path(profile)
    assert Path(resolved).is_absolute()
    assert Path(resolved).parent.parent == profile.dir()
    assert profile.hitl.to_planning_config(cache_path=resolved).cache_path == resolved
    assert profiles_mod.resolve_cache_path(Profile(name="none")) is None


# --- walking a phase plan -------------------------------------------------------------------------


@pytest.fixture
def phase_session(profile, tmp_path, monkeypatch):
    """A live session with phase planning on, a stand-in backend, and a canned proposer."""
    from tandem.planning import llm

    def build(*verdicts, backend_kwargs=None, **profile_changes):
        profile.hitl.enabled = True
        # These tests answer a human phase "done", for a step done by hand. While recording that is
        # refused unless allowed (tests/test_executor_integration.py).
        profile.hitl.allow_unrecorded_human_phase = True
        for key, value in profile_changes.items():
            setattr(profile.hitl, key, value)
        monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
        # The plan is answered however many times it is asked for; the verdicts are consumed in
        # order and the last one repeats.
        client = FakeGemini(json.dumps(PLAN), list(verdicts) or [HOLDS])
        monkeypatch.setattr(llm, "gemini_client", lambda: client)
        backends = use_fake_backend(monkeypatch, **(backend_kwargs or {}))

        session = Session(profile, task="put the toy in the box")
        session.start()
        assert wait_for(lambda: session.state is State.AWAITING_TASK), f"stuck in {session.state}"
        made.append(session)
        return session, backends, client

    made: list[Session] = []
    yield build
    for session in made:
        if session.alive:
            session.stop(park=False)
            session.wait(timeout=5)


def test_the_plan_reaches_the_human_phase_with_its_instructions(phase_session):
    session, backends, _ = phase_session()
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE), f"stuck in {session.state}"

    phase = session.human_phase
    assert phase is not None
    assert phase.description == "open the box"
    assert phase.instructions.startswith("Open the white_box")
    # What will be checked afterwards, in the words the person is shown — nobody is judged against
    # a hidden standard.
    assert phase.expected == ["the container white_box is open, so its interior is visible"]
    # And where in the plan it is. The old integration never sent these, so "step N of M" was
    # always the fallback and usually wrong.
    assert (phase.index, phase.total) == (1, 3)
    assert session.summary()["phase_progress"] == [1, 3]


def test_the_robot_phase_before_it_was_planned_and_run(phase_session):
    session, backends, _ = phase_session()
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE)

    backend = backends[-1]
    planned = [json.loads(c.split(":", 1)[1]) for c in backend.calls if c.startswith("plan:")]
    assert planned == [[{"predicate": "on", "args": ["blue_toy", "table"]}]]
    assert backend.legs[0]["leg"].phase_index == 0
    assert backend.legs[0]["leg"].n_phases == 3


def test_completing_a_phase_verifies_it_and_the_robot_carries_on(phase_session):
    session, backends, _ = phase_session()
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE)

    session.complete_human_phase()
    assert wait_for(lambda: session.state is State.AWAITING_LABEL), f"stuck in {session.state}"

    backend = backends[-1]
    # The check ran against a fresh frame from the third-person camera.
    assert "capture_frame:external" in backend.calls
    # And the third phase — the part the previous design lost — actually ran.
    planned = [json.loads(c.split(":", 1)[1]) for c in backend.calls if c.startswith("plan:")]
    assert planned[-1] == [{"predicate": "on", "args": ["blue_toy", "white_box"]}]
    assert [leg["leg"].phase_index for leg in backend.legs] == [0, 2]


def test_every_leg_of_the_task_shares_one_trajectory(phase_session):
    """That shared id is the only thing that joins the planner's legs and the person's into one
    episode, and only tandem sees both kinds."""
    session, backends, _ = phase_session()
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE)
    session.complete_human_phase()
    assert wait_for(lambda: session.state is State.AWAITING_LABEL)

    ids = {leg["leg"].trajectory_id for leg in backends[-1].legs}
    assert len(ids) == 1 and next(iter(ids))


def test_a_step_that_does_not_verify_is_retried_before_it_is_given_up_on(phase_session):
    """One bad classifier call should not cost a demonstration: the person is told what is still
    missing and gets another go."""
    session, _, _ = phase_session(DOES_NOT_HOLD, HOLDS, verify_retries=1)
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE)

    session.complete_human_phase()
    # Re-prompted, with what is still expected.
    assert wait_for(lambda: session.human_phase is not None and session.human_phase.attempt == 2)
    assert session.human_phase.missing
    assert "a flap is still closed" in session.human_phase.missing[0]

    session.complete_human_phase()
    assert wait_for(lambda: session.state is State.AWAITING_LABEL), f"stuck in {session.state}"


def test_a_step_that_never_verifies_ends_the_attempt(phase_session):
    # `label` keeps the operator in the loop for a trial the check stopped; the default, `exclude`,
    # files it without asking (tests/test_trial_outcomes.py).
    session, _, _ = phase_session(DOES_NOT_HOLD, verify_retries=0, on_verification_failure="label")
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE)
    session.complete_human_phase()
    # The first phase put frames on disk, and the label is what ends a trajectory and merges its
    # legs — so an abandoned attempt still has to reach the prompt, or those legs are orphaned.
    assert wait_for(lambda: session.state is State.AWAITING_LABEL), f"stuck in {session.state}"
    assert session.human_phase is None


def test_a_failed_check_can_be_recorded_without_failing_the_run(phase_session):
    """What you want while calibrating the classifier prompts: the verdict is still recorded, the
    run just carries on."""
    session, backends, _ = phase_session(DOES_NOT_HOLD, verify_enforced=False, verify_retries=0)
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE)
    session.complete_human_phase()
    assert wait_for(lambda: session.state is State.AWAITING_LABEL), f"stuck in {session.state}"
    assert [leg["leg"].phase_index for leg in backends[-1].legs] == [0, 2]


def test_aborting_a_phase_abandons_the_attempt(phase_session):
    session, _, _ = phase_session()
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE)
    session.abort_human_phase()
    assert wait_for(lambda: session.state is State.AWAITING_LABEL), f"stuck in {session.state}"
    assert session.human_phase is None
    assert session.alive
    session.label(False)
    assert wait_for(lambda: session.state is State.AWAITING_TASK)


def test_phase_actions_are_refused_outside_the_prompt(phase_session):
    session, _, _ = phase_session()
    with pytest.raises(SessionConflict):
        session.complete_human_phase()
    with pytest.raises(SessionConflict):
        session.abort_human_phase()


def test_a_phase_the_planner_cannot_plan_is_offered_to_the_person(phase_session):
    """The capability the old design could not express at all, now opt-in.

    Who did what was decided at proposal time inside the planner's process, so a phase it turned
    out not to be able to plan could only end the attempt. tandem owns the split, so the sub-goal
    is described to the operator, checked exactly as any other human step, and the task carries on.
    """
    session, _, _ = phase_session(
        HOLDS,
        backend_kwargs={"plan_failure": "no collision-free grasp"},
        on_robot_phase_failure="teleop",
    )
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE), f"stuck in {session.state}"

    phase = session.human_phase
    # Phase 0 is a ROBOT phase in the plan; it is the person's now because the planner could not
    # do it, and what they are asked for is what was going to be asked of the planner.
    assert phase.index == 0
    assert "blue_toy is resting on top of table" in phase.instructions
    assert any("no collision-free grasp" in line["text"] for line in session.logs())


def test_a_phase_that_cannot_be_planned_ends_the_attempt_by_default(phase_session):
    """The paper counts a TAMP failure as a trial failure, and so does a profile nobody tuned.

    A teleop fallback here would turn the robot's phase into the operator's without anyone having
    asked, crediting the method with a trial it did not complete and inflating the human effort
    the dataset cost."""
    session, _, _ = phase_session(backend_kwargs={"plan_failure": "no collision-free grasp"})
    session.next_task()
    # Nothing was recorded, so there is nothing to label: straight back to the task prompt.
    assert wait_for(
        lambda: any("could not plan this phase" in line["text"] for line in session.logs())
    )
    assert wait_for(lambda: session.state is State.AWAITING_TASK), f"stuck in {session.state}"
    assert session.human_phase is None


def test_the_audit_record_lands_beside_the_finished_episode(phase_session, monkeypatch):
    """Written after the merge and into the MERGED directory, not into a leg.

    The merge surfaces only the first planner leg's extra files, and that copy is the earliest
    snapshot — a low phase index and no verifications at all — so a record written per leg reads as
    though the task barely started."""
    session, _, _ = phase_session()
    merged: dict = {}

    from tandem.core import merge as merge_mod

    def fake_merge(profile, trajectory_id, *, status=None, tools_dir=None):
        directory = Path(profile.status_dir(status or "success")) / trajectory_id
        directory.mkdir(parents=True, exist_ok=True)
        merged["dir"] = directory
        return {"merged": True, "n_legs": 2, "n_frames": 48, "dir": str(directory)}

    monkeypatch.setattr(merge_mod, "merge", fake_merge)

    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE)
    session.complete_human_phase()
    assert wait_for(lambda: session.state is State.AWAITING_LABEL)
    session.label(True)
    assert wait_for(lambda: "dir" in merged and (merged["dir"] / "hitl.json").is_file())

    record = json.loads((merged["dir"] / "hitl.json").read_text())
    assert [p["executor"] for p in record["phases"]] == ["robot", "human", "robot"]
    assert record["planner"] == "tiptop"
    assert record["phases"][1]["atoms"] == ["IsOpen(white_box)"]
    assert record["verifications"], "the verdicts belong in the record, not only in the log"


def test_a_disabled_profile_never_reaches_a_human_phase(profile, tmp_path, monkeypatch):
    """The degradation guarantee: with it off, a session behaves exactly as it always has."""
    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
    use_fake_backend(monkeypatch)
    assert profile.hitl.enabled is False

    session = Session(profile, task="pick up the block")
    session.start()
    try:
        assert wait_for(lambda: session.state is State.AWAITING_TASK)
        session.next_task()
        assert wait_for(lambda: session.state is State.AWAITING_LABEL)
        assert session.human_phase is None
        assert session.summary()["hitl_enabled"] is False
    finally:
        session.stop(park=False)
        session.wait(timeout=5)


# --- what a finished trajectory carries -----------------------------------------------------------


def test_hitl_json_is_surfaced_on_a_trajectory(profile, make_trajectory):
    from tandem.core import trajectories

    directory = make_trajectory(profile, "2026-01-01_00-00-00")
    (directory / "hitl.json").write_text(
        json.dumps(
            {
                "instruction": "put the toy in the box",
                "phases": [
                    {"index": 0, "executor": "robot", "description": "a", "atoms": []},
                    {"index": 1, "executor": "human", "description": "b", "atoms": []},
                ],
                "verifications": [{"atom": "IsOpen(box)", "holds": True, "statement": "", "reason": ""}],
            }
        )
    )
    record = trajectories.read(directory)
    assert record.has_hitl
    assert record.to_dict()["has_hitl"] is True


def test_a_rollout_without_phase_planning_reports_none(profile, make_trajectory):
    from tandem.core import trajectories

    record = trajectories.read(make_trajectory(profile, "2026-01-01_00-00-01"))
    assert record.has_hitl is False


# --- a real hand-off, through the session -----------------------------------------------------


class Driver:
    """What the teleop driver has announced, as any subscriber to the session sees it.

    The driver runs inside the teleop executor, behind the phase loop's hand-off, so the session no
    longer holds it. The one thing these tests need from it -- has the person started recording? --
    is on the session's message bus, which is where the web page reads it too.
    """

    def __init__(self, session) -> None:
        self.started = 0
        session.subscribe(self._on_message)

    def _on_message(self, message: dict) -> None:
        if message.get("type") == "teleop_event" and message.get("event") == "rollout_start":
            self.started += 1

    def recording(self, legs: int = 1):
        """A wait_for predicate: the driver has started recording its ``legs``-th leg."""
        return lambda: self.started >= legs


@pytest.fixture
def teleop_enabled(monkeypatch, tmp_path):
    """Point the hand-off at the stand-in driver and turn teleop on."""
    import sys

    from tandem import teleop as teleop_pkg
    from tandem.core import settings as settings_mod
    from tandem.core.settings import Settings

    monkeypatch.setattr(teleop_pkg, "driver_path", lambda: Path(__file__).parent / "fake_teleop.py")
    cfg = Settings()
    cfg.teleop.enabled = True
    cfg.teleop.python = sys.executable
    cfg.teleop.droid_dir = str(Path(__file__).parent)
    monkeypatch.setattr(settings_mod, "load", lambda: cfg)
    return cfg


def test_a_teleop_leg_counts_as_part_of_the_episode(phase_session, teleop_enabled):
    """The demonstration a person just gave has to reach the label prompt.

    The frame count arrives on the tailer thread when the driver emits `rollout_saved`, which it
    does only after muxing its videos — and the wait for the driver to exit is what drains that
    event. Asking before the wait is asking too early every single time, and the answer is
    "nothing was recorded": the task returns to the prompt, nothing is ever labeled or merged, and
    a recorded demonstration is stranded in eval/ while the operator is told it never happened.
    """
    session, backends, _ = phase_session()
    driver = Driver(session)
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE), f"stuck in {session.state}"

    # Take the arm through the rig rather than doing it by hand.
    session.request_teleop()
    assert wait_for(lambda: session.state is State.TELEOP_HANDOFF), f"stuck in {session.state}"
    # The planner must have let go before anything else opens a camera.
    assert "release_hardware" in backends[-1].calls
    assert not backends[-1].holds_hardware

    # Wait until the driver is actually recording, which is what an operator taking the arm and
    # demonstrating amounts to. Returning control before then is a leg nobody drove.
    assert wait_for(driver.recording())
    session.resume_from_teleop()
    assert wait_for(lambda: session.state is State.AWAITING_LABEL), f"stuck in {session.state}"

    # Taken back only after the driver had gone.
    names = backends[-1].call_names()
    assert names.index("reacquire_hardware") > names.index("release_hardware")
    assert backends[-1].holds_hardware
    assert session._legs_recorded >= 2, "the person's leg is part of this episode too"
    assert any("teleop leg is part of this episode" in line["text"] for line in session.logs())


def test_the_teleop_leg_is_stamped_with_the_same_trajectory_as_the_planners(phase_session, teleop_enabled):
    """One shared id is the only thing that joins the planner's legs and the person's."""
    from tandem.core import merge as merge_mod

    session, backends, _ = phase_session()
    driver = Driver(session)
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE)
    session.request_teleop()
    assert wait_for(lambda: session.state is State.TELEOP_HANDOFF)
    assert wait_for(driver.recording())
    session.resume_from_teleop()
    assert wait_for(lambda: session.state is State.AWAITING_LABEL)

    trajectory_id = backends[-1].legs[0]["leg"].trajectory_id
    legs = merge_mod.find_legs(session.profile, trajectory_id)
    sources = sorted(leg["source"] for leg in legs)
    assert "teleop" in sources, f"the person's leg was not found under this trajectory: {sources}"
    assert sources.count("tamp") >= 1



def test_the_teleop_leg_of_a_human_phase_is_stamped_with_that_phase(phase_session, teleop_enabled):
    """The planner's legs carry their phase, and so must the person's.

    The merge maps each leg to its phase in segments[] from the leg's own _meta.json, so a teleop
    leg launched without the phase is a gap in the one record of who did which step.
    """
    from tandem.core import merge as merge_mod

    session, backends, _ = phase_session()
    driver = Driver(session)
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE)
    session.request_teleop()
    assert wait_for(lambda: session.state is State.TELEOP_HANDOFF)
    assert wait_for(driver.recording())
    session.resume_from_teleop()
    assert wait_for(lambda: session.state is State.AWAITING_LABEL)

    trajectory_id = backends[-1].legs[0]["leg"].trajectory_id
    teleop_legs = [leg for leg in merge_mod.find_legs(session.profile, trajectory_id) if leg["source"] == "teleop"]
    assert len(teleop_legs) == 1
    meta = json.loads((Path(teleop_legs[0]["dir"]) / "_meta.json").read_text())
    # 0-based, the way the planner's legs count: the robot's phase 0, then the person's.
    assert (meta["phase_index"], meta["n_phases"]) == (1, 3)
    assert meta["phase_description"] == "open the box"

def test_replan_actually_re_plans_rather_than_quietly_aborting(phase_session):
    """`replan` is a documented policy, and it has to differ from `abort`.

    It drops the plan and goes round again, so the next pass perceives afresh and decomposes the
    task against the scene as it now stands. Bounded by max_attempts, because a goal the planner
    genuinely cannot reach fails the same way every time and an unbounded retry would perceive and
    re-propose forever with an operator watching an arm that never moves.
    """
    # One proposal per attempt: the original plus every re-plan.
    session, backends, client = phase_session(
        backend_kwargs={"plan_failure": "no collision-free grasp"},
        on_robot_phase_failure="replan",
    )
    replans = session.profile.hitl.max_attempts
    session.next_task()
    # Wait on the LOG, not the state: the session is already at the prompt when next_task is
    # called, so a state check would pass before the attempt had even started.
    assert wait_for(lambda: any("out of re-planning attempts" in x["text"] for x in session.logs()))
    # Nothing was ever recorded, so the attempt ends back at the task prompt.
    assert wait_for(lambda: session.state is State.AWAITING_TASK), f"stuck in {session.state}"

    # It tried again rather than giving up on the first failure — one perception pass per attempt,
    # plus the re-planned ones.
    assert backends[-1].perceptions == replans + 1, (
        f"expected {replans + 1} perception passes, got {backends[-1].perceptions}"
    )
    assert client.plan_calls == replans + 1
    assert any("re-planning the task" in line["text"] for line in session.logs())
    assert any("out of re-planning attempts" in line["text"] for line in session.logs())


def test_a_second_return_control_click_does_not_end_the_next_handoff(phase_session, teleop_enabled):
    """The state stays TELEOP_HANDOFF for the whole exit sequence, so the button stays live.

    Clicking it twice is natural — the exit takes seconds while the driver saves and the planner
    takes the hardware back. The second click used to leave the flag set with nobody waiting, and
    the NEXT hand-off then returned instantly: the driver was launched and told to quit before the
    person could touch the arm, while the UI was still telling them the arm was theirs.
    """
    import threading

    # The first attempt at the step does not verify, so the person is offered it again — which is
    # what gives us a second hand-off to check.
    session, backends, _ = phase_session(DOES_NOT_HOLD, HOLDS, verify_retries=1)
    driver = Driver(session)
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE)

    session.request_teleop()
    assert wait_for(lambda: session.state is State.TELEOP_HANDOFF)
    assert wait_for(driver.recording())

    # Hold the teardown open, which is what makes the second click land where it did in practice.
    # An instantaneous fake closes the window entirely and the bug cannot reproduce at all.
    release, taking_back = threading.Event(), threading.Event()
    original = backends[-1].reacquire_hardware
    backends[-1].reacquire_hardware = lambda: (taking_back.set(), release.wait(timeout=10.0), original())[2]

    session.resume_from_teleop()
    assert wait_for(taking_back.is_set), "the teardown never started"
    with contextlib.suppress(SessionConflict):
        session.resume_from_teleop()  # the second click, mid-teardown
    release.set()

    # The step did not verify, so it comes back round. This second hand-off is the one the stale
    # flag used to end before it began.
    assert wait_for(lambda: session.human_phase is not None and session.human_phase.attempt == 2)
    session.request_teleop()
    assert wait_for(lambda: session.state is State.TELEOP_HANDOFF), f"stuck in {session.state}"
    assert wait_for(driver.recording(2)), "the second hand-off ended before the person could record anything"


def test_a_preempt_does_not_answer_the_next_human_phase_on_the_persons_behalf(phase_session):
    """`preempt` used to set the same flag a person's "done" sets, and nothing cleared it.

    It only bites when nobody is waiting on that flag — a preempt at the task prompt, or during a
    robot phase — so the flag survives to the NEXT human phase and answers it instantly. The person
    is never asked to do the step, and it is verified anyway: a camera capture and a model call
    spent to be told that a step nobody performed did not happen, and a retry burned doing it.
    """
    import time as _time

    session, backends, client = phase_session()

    # Preempted where nothing is waiting for an answer. Realistic: the operator changes their mind
    # at the prompt, then starts a task.
    session.preempt()
    verdicts_before = client.verdict_calls

    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE), f"stuck in {session.state}"

    # It has to STAY there. The bug does not stop it reaching the prompt — it answers it, so the
    # state flickers through and a verification runs on a step nobody was asked to do.
    _time.sleep(0.4)
    assert session.state is State.AWAITING_HUMAN_PHASE, (
        f"the phase was answered without anyone being asked (now {session.state})"
    )
    assert client.verdict_calls == verdicts_before, "a step nobody performed was verified"
    assert session.human_phase is not None and session.human_phase.attempt == 1
