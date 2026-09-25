"""How a trial ends: the camera checks around each phase, the exclusion rule, and the record of both.

The paper's contract for a human phase (Sec. IV-D): after the person is done, a fresh image is taken
and the phase's operator is checked -- add effects must hold, delete effects must NOT. Preconditions
can be checked before the hand-off too. A trial whose check still fails once its retries are spent
is terminated and EXCLUDED from the dataset: filed under failure/ with ``excluded: true``, its failing
verdicts and raw legs kept, and no label prompt (``on_verification_failure: exclude``). The same
classifiers can watch a robot leg (``check_tamp_*``), which only ever records unless told otherwise.

Most of this is the phase loop on its own (``tandem.core.phase_loop``), driven against a stand-in
backend and a canned model. The parts only the session can do -- skipping the label, filing, merging,
the events file -- are driven through a real session.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from fake_backend import FakeBackend
from fake_executor import FakeExecutor, use_fake_executor
from helpers import use_fake_backend, wait_for

from tandem.core import events as events_mod
from tandem.core import secrets
from tandem.core.episodes import LegDirs
from tandem.core.phase_loop import PhaseLoop
from tandem.core.session import Session, State
from tandem.executors.base import ExecutorContext
from tandem.planning.config import PlanningConfig

# What the camera is asked about, in the words it is asked in.
OPEN = "the container white_box is open, so its interior is visible"
CLOSED = "the container white_box is closed, with its flaps down"
TOY_ON_TABLE = "blue_toy is resting on top of table"
TOY_IN_BOX = "blue_toy is resting on top of white_box"

TAKE_OFF = {
    "executor": "robot",
    "description": "take the toy off the box",
    "atoms": [{"predicate": "On", "args": ["blue_toy", "table"]}],
}
PUT_IN = {
    "executor": "robot",
    "description": "put the toy inside the box",
    "atoms": [{"predicate": "On", "args": ["blue_toy", "white_box"]}],
}


def open_the_box(*, preconditions=None, delete=None) -> dict:
    """The person's step. By default it needs the box CLOSED, and leaves it open and no longer closed."""
    pre = [("IsClosed", ["white_box"]), ("HandEmpty", [])] if preconditions is None else preconditions
    gone = [("IsClosed", ["white_box"])] if delete is None else delete
    return {
        "executor": "human",
        "description": "open the box",
        "instructions": "Open the white_box and fold its flaps back.",
        "atoms": [{"predicate": "IsOpen", "args": ["white_box"]}],
        "operator": {
            "name": "Open",
            "args": ["white_box"],
            "preconditions": [{"predicate": p, "args": a} for p, a in pre],
            "add_effects": [{"predicate": "IsOpen", "args": ["white_box"]}],
            "delete_effects": [{"predicate": p, "args": a} for p, a in gone],
        },
    }


INVENTED = {
    "IsOpen": "the container {0} is open, so its interior is visible",
    "IsClosed": "the container {0} is closed, with its flaps down",
}


def proposal(*phases) -> dict:
    """A model's answer: these phases, and every invented predicate they use (an unused one is refused)."""
    used = json.dumps(phases)
    return {
        "new_predicates": [
            {"name": name, "instructions": text} for name, text in INVENTED.items() if f'"{name}"' in used
        ],
        "phases": list(phases),
    }


# Robot, then the person, then the robot again: the person's step is in the MIDDLE, so everything
# after it depends on the check.
PLAN = proposal(TAKE_OFF, open_the_box(), PUT_IN)
# The person's step is the LAST one, which is what verify_final_phase is about.
ENDS_WITH_A_PERSON = proposal(TAKE_OFF, open_the_box())


class Camera:
    """The vision model, answering each classifier question by the STATEMENT it asks about.

    Keyed on the statement rather than on call order, because a phase's add and delete effects are
    asked about concurrently and the order they arrive in is not the test's business. Each answer is
    a bool, or a list of them consumed one call at a time with the last one repeating. A statement
    nobody gave an answer for fails the test: the camera was asked something it was not meant to be.
    """

    PLAN_MARKER = "ORDERED list of phases"

    def __init__(self, plan, answers: dict | None = None, *, fail: str | None = None) -> None:
        self.plan = plan if isinstance(plan, str) else json.dumps(plan)
        self.answers = {
            k: list(v) if isinstance(v, (list, tuple)) else [v] for k, v in (answers or {}).items()
        }
        self.fail = fail
        self.asked: list[str] = []
        self.plan_calls = 0
        self.aio = self

    @property
    def models(self):
        return self

    async def generate_content(self, model, contents, config):
        prompt = contents[-1]
        if "Statement:" in prompt and self.PLAN_MARKER not in prompt:
            statement = prompt.split("Statement:", 1)[1].split("\n", 1)[0].strip()
            self.asked.append(statement)
            if self.fail:
                raise RuntimeError(self.fail)
            if statement not in self.answers:
                raise AssertionError(f"the camera was asked about {statement!r}, which no test answer covers")
            queue = self.answers[statement]
            holds = queue.pop(0) if len(queue) > 1 else queue[0]
            reason = "it is plainly visible" if holds else "it is not what the image shows"
            return mock.Mock(text=json.dumps({"holds": holds, "reason": reason}))
        self.plan_calls += 1
        return mock.Mock(text=self.plan)


class Sink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []
        self.lines: list[str] = []

    def event(self, name: str, **payload) -> None:
        self.events.append((name, payload))

    def log(self, text: str) -> None:
        self.lines.append(text)

    def named(self, name: str) -> list[dict]:
        return [payload for event, payload in self.events if event == name]


class Operator:
    """The person at the prompts. Answers "done" to every human phase unless told otherwise."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.prompts = 0
        self.shown: list = []
        self.human_phase = None
        self.progress = None

    def check_preempt(self) -> None:
        return None

    def take_handoff_request(self) -> bool:
        return False

    def rolling(self) -> None:
        return None

    def show_progress(self, progress) -> None:
        self.progress = progress

    def show_unrepresented(self, clauses) -> None:
        return None

    def show_human_phase(self, phase) -> None:
        self.human_phase = phase
        if phase is not None:
            self.shown.append(phase)

    def await_human_phase(self) -> str:
        self.prompts += 1
        return self.answers.pop(0) if self.answers else "done"

    def rollout_started(self, save_dir) -> None:
        return None

    def rollout_saved(self, n_frames: int) -> None:
        return None

    def handing_off(self) -> None:
        return None

    def arm_lent(self):
        return lambda: True

    def arm_returned(self) -> None:
        return None


@pytest.fixture
def rig(profile, tmp_path, monkeypatch):
    """A phase loop against a stand-in backend and a scripted camera. Nothing runs until `.run()`."""
    from tandem.planning import llm

    def make(answers=None, *operator_answers, plan=PLAN, fail=None, **cfg):
        camera = Camera(plan, answers, fail=fail)
        monkeypatch.setattr(llm, "gemini_client", lambda: camera)
        backend = FakeBackend(None, output_dir=tmp_path / "frames")
        # The human executor with nobody on the other end: every leg "records" 30 frames.
        sink, operator, hands = Sink(), Operator(*operator_answers), FakeExecutor()
        use_fake_executor(monkeypatch, hands)
        loop = PhaseLoop(
            backend,
            backend.capabilities(),
            # The operator answers "done" for a step done by hand, which while recording is refused
            # unless allowed (tests/test_executor_integration.py); these tests are about the checks.
            PlanningConfig(
                **{"enabled": True, "save_vlm_io": False, "allow_unrecorded_human_phase": True, **cfg}
            ),
            events=sink,
            operator=operator,
            executor_context=ExecutorContext(profile=profile, session_dir=tmp_path / "session"),
            legs=LegDirs(profile, tmp_path / "session", log=sink.log),
        )

        def run():
            return loop.run(
                task="put the toy in the box", instruction="put the toy in the box", trajectory_id="t" * 16
            )

        return SimpleNamespace(
            loop=loop, backend=backend, sink=sink, operator=operator, hands=hands, camera=camera, run=run
        )

    return make


def ran(backend) -> list:
    """The phase index of every robot leg that was executed."""
    return [leg["leg"].phase_index for leg in backend.legs]


def frames(backend) -> int:
    return sum(1 for call in backend.calls if call.startswith("capture_frame"))


def record_of(outcome) -> dict:
    return json.loads(json.dumps(outcome.plan.to_json(), default=str))


# --- the effects of a human phase -----------------------------------------------------------------


def test_a_delete_effect_still_true_is_retried_and_then_excludes_the_trial(rig):
    """The box looks open, but it also still looks closed: the step undid nothing it said it would."""
    r = rig({OPEN: True, CLOSED: True}, verify_retries=1)
    outcome = r.run()

    # One retry, and the person was told what is wrong -- the other way round, since what is missing
    # is that something STOPPED being true.
    assert r.operator.prompts == 2
    retry = r.operator.shown[1]
    assert retry.attempt == 2
    assert retry.missing and retry.missing[0].startswith(f"{CLOSED} -- and it should no longer be")
    assert all(OPEN not in line for line in retry.missing), "the add effect held; it is not missing"

    # Then the trial stopped there, excluded, and the robot never ran the phase after it.
    assert (outcome.outcome, outcome.failure_stage) == ("excluded", "verification")
    assert "open the box" in outcome.reason
    assert ran(r.backend) == [0]
    assert r.operator.human_phase is None

    # The record says so, and carries the failing verdicts it was excluded over.
    record = record_of(outcome)
    assert (record["outcome"], record["failure_stage"], record["excluded"]) == (
        "excluded",
        "verification",
        True,
    )
    by_role = {v["role"]: v for v in record["verifications"]}
    assert set(by_role) == {"effect", "effect (deleted)"}, "the attempt that settled it, and only that one"
    assert by_role["effect (deleted)"]["satisfied"] is False and by_role["effect (deleted)"]["holds"] is True
    assert by_role["effect"]["satisfied"] is True
    assert {v["phase"] for v in record["verifications"]} == {1}

    # And the loop said so, in its own events.
    assert [p["ok"] for p in r.sink.named("human_phase_verified")] == [False, False]
    (ended,) = r.sink.named("trial_outcome")
    assert (ended["outcome"], ended["failure_stage"], ended["excluded"]) == ("excluded", "verification", True)
    assert ended["phase_index"] == 1


def test_a_step_that_passes_on_its_retry_records_the_attempt_that_passed(rig):
    r = rig({OPEN: True, CLOSED: [True, False]})
    outcome = r.run()

    assert r.operator.prompts == 2
    assert ran(r.backend) == [0, 2]
    assert (outcome.outcome, outcome.failure_stage) == (None, None), "the label decides a finished trial"
    record = record_of(outcome)
    assert [v["satisfied"] for v in record["verifications"]] == [True, True]
    assert record["excluded"] is False


def test_with_verify_enforced_off_a_failed_check_is_recorded_and_the_trial_carries_on(rig):
    r = rig({OPEN: True, CLOSED: True}, verify_enforced=False)
    outcome = r.run()

    assert r.operator.prompts == 1, "no retry: the check does not decide anything"
    assert ran(r.backend) == [0, 2]
    assert outcome.outcome is None
    assert any(not v["satisfied"] for v in record_of(outcome)["verifications"])


@pytest.mark.parametrize(
    ("plan", "cfg", "checked"),
    [
        (ENDS_WITH_A_PERSON, {}, True),
        (ENDS_WITH_A_PERSON, {"verify_final_phase": False}, False),
        # Only the LAST phase is left to the label; a person's step in the middle is still checked.
        (PLAN, {"verify_final_phase": False}, True),
        (PLAN, {"check_human_effects": False}, False),
    ],
    ids=["final-checked", "final-skipped", "middle-still-checked", "effects-off"],
)
def test_which_human_phases_are_put_to_the_camera(rig, plan, cfg, checked):
    r = rig({OPEN: True, CLOSED: False}, plan=plan, **cfg)
    outcome = r.run()

    assert bool(r.camera.asked) is checked
    assert frames(r.backend) == (1 if checked else 0)
    (verified,) = r.sink.named("human_phase_verified")
    if checked:
        assert verified["ok"] is True
    else:
        # Not checked is not passed: the trail says which it was.
        assert verified["ok"] is None and verified["skipped"]
        assert r.operator.shown[-1].verified is None
        assert record_of(outcome)["verifications"] == []
    assert outcome.outcome is None
    assert outcome.plan.finished


# --- a check that cannot run ----------------------------------------------------------------------


def test_a_camera_that_fails_leaves_the_step_accepted_but_on_the_record_as_unchecked(rig, monkeypatch):
    r = rig({OPEN: True, CLOSED: False})

    def broken(*, camera="external"):
        raise OSError("the external camera is not answering")

    monkeypatch.setattr(r.backend, "capture_frame", broken)
    outcome = r.run()

    # One unreachable camera must not cost the person their demonstration...
    assert ran(r.backend) == [0, 2]
    assert outcome.outcome is None
    # ...but a phase nobody checked must not read as one that passed.
    record = record_of(outcome)
    assert record["checks"]["unchecked_phases"] == [1]
    assert record["phases"][1]["unchecked"].startswith("effect check: OSError")
    assert record["verifications"] == []
    (verified,) = r.sink.named("human_phase_verified")
    assert verified["ok"] is None and "not answering" in verified["unchecked"]
    assert r.operator.shown[-1].verified is None


def test_a_classifier_that_fails_is_unchecked_even_where_preconditions_are_enforced(rig):
    """An error is a check that did not happen, not an unmet precondition: nothing is stopped for it."""
    r = rig(fail="the model is overloaded", check_human_preconditions=True, precondition_enforced=True)
    outcome = r.run()

    assert ran(r.backend) == [0, 2]
    assert r.operator.prompts == 1
    assert outcome.outcome is None
    unchecked = record_of(outcome)["phases"][1]["unchecked"]
    # Both checks of the phase, side by side, rather than the second hiding the first.
    assert "human phase precondition check: RuntimeError: the model is overloaded" in unchecked
    assert "effect check: RuntimeError: the model is overloaded" in unchecked
    assert record_of(outcome)["checks"]["unchecked_phases"] == [1]
    (checked,) = r.sink.named("phase_preconditions_checked")
    assert checked["ok"] is None and checked["unchecked"]


# --- the preconditions of a human phase -----------------------------------------------------------


def test_an_unmet_precondition_stops_the_trial_before_the_arm_is_handed_over_when_enforced(rig):
    r = rig({CLOSED: False}, "teleop", check_human_preconditions=True, precondition_enforced=True)
    outcome = r.run()

    # Nobody was asked to do a step the workspace was not ready for.
    assert r.operator.prompts == 0
    assert r.hands.calls == []
    assert ran(r.backend) == [0]
    assert (outcome.outcome, outcome.failure_stage) == ("excluded", "verification")
    assert "preconditions" in outcome.reason and CLOSED in outcome.reason

    (checked,) = r.sink.named("phase_preconditions_checked")
    assert (checked["what"], checked["ok"], checked["enforced"], checked["phase_index"]) == (
        "human phase",
        False,
        True,
        1,
    )
    record = record_of(outcome)
    assert [(v["role"], v["satisfied"], v["phase"]) for v in record["verifications"]] == [
        ("precondition", False, 1)
    ]
    assert record["excluded"] is True


def test_an_unmet_precondition_is_recorded_and_the_step_goes_ahead_by_default(rig):
    # The box was never closed, so the precondition fails -- and the delete effect is satisfied.
    r = rig({OPEN: True, CLOSED: False}, check_human_preconditions=True)
    outcome = r.run()

    assert ran(r.backend) == [0, 2]
    assert outcome.outcome is None
    (checked,) = r.sink.named("phase_preconditions_checked")
    assert (checked["ok"], checked["enforced"]) == (False, False)
    roles = [(v["role"], v["satisfied"]) for v in record_of(outcome)["verifications"]]
    assert roles[0] == ("precondition", False)
    assert ("effect (deleted)", True) in roles


def test_preconditions_are_checked_once_however_many_attempts_the_step_takes(rig):
    r = rig({OPEN: [False, True], CLOSED: [True, False]}, check_human_preconditions=True)
    r.run()

    assert r.operator.prompts == 2
    assert len(r.sink.named("phase_preconditions_checked")) == 1
    # The precondition frame, then one per attempt.
    assert frames(r.backend) == 3


def test_preconditions_no_camera_can_settle_take_no_frame(rig):
    """HandEmpty() alone: the gripper is the robot's to know, and there is nothing to look at."""
    plan = proposal(TAKE_OFF, open_the_box(preconditions=[("HandEmpty", [])], delete=[]), PUT_IN)
    r = rig({OPEN: True}, plan=plan, check_human_preconditions=True, precondition_enforced=True)
    r.run()

    assert ran(r.backend) == [0, 2]
    assert r.sink.named("phase_preconditions_checked") == []
    assert frames(r.backend) == 1, "only the effects frame"


# --- the robot's legs -----------------------------------------------------------------------------


def test_the_robot_leg_checks_only_record(rig):
    """The camera disagrees with the arm about every placement, and nothing stops for it."""
    r = rig(
        {TOY_ON_TABLE: False, TOY_IN_BOX: False, OPEN: True, CLOSED: False},
        check_tamp_preconditions=True,
        check_tamp_effects=True,
    )
    outcome = r.run()

    assert ran(r.backend) == [0, 2]
    assert outcome.outcome is None

    # Before the second leg: what the earlier phases were responsible for, asked of THIS pass's
    # perception image. The first leg had nothing earlier to be responsible for, so it was not asked.
    (before,) = r.sink.named("phase_preconditions_checked")
    assert (before["what"], before["phase_index"], before["ok"], before["enforced"]) == (
        "robot leg",
        2,
        False,
        False,
    )
    assert {v["atom"] for v in before["verdicts"]} == {"On(blue_toy, table)", "IsOpen(white_box)"}

    # After each leg: its own goal, on a fresh frame.
    after = r.sink.named("phase_effects_checked")
    assert [(e["phase_index"], e["ok"], e["enforced"]) for e in after] == [
        (0, False, False),
        (2, False, False),
    ]
    # Two effect frames and the person's step; the precondition check reused the perception image.
    assert frames(r.backend) == 3

    entries = [
        (v["phase"], v["role"], v["atom"], v["satisfied"]) for v in record_of(outcome)["verifications"]
    ]
    assert (0, "effect", "On(blue_toy, table)", False) in entries
    assert (2, "precondition", "On(blue_toy, table)", False) in entries
    assert (2, "effect", "On(blue_toy, white_box)", False) in entries


def test_an_enforced_robot_leg_precondition_stops_before_the_planner_is_asked(rig):
    r = rig(
        {TOY_ON_TABLE: False, OPEN: True, CLOSED: False},
        check_tamp_preconditions=True,
        precondition_enforced=True,
    )
    outcome = r.run()

    planned = [call for call in r.backend.calls if call.startswith("plan:")]
    assert len(planned) == 1, "the second leg was never planned"
    assert ran(r.backend) == [0]
    assert len(r.sink.named("rollout_start")) == 1, "a leg its preconditions stop never starts"
    assert (outcome.outcome, outcome.failure_stage) == ("excluded", "verification")


# --- how every trial ends -------------------------------------------------------------------------


def test_a_proposal_that_never_validates_ends_the_trial_at_invention(rig):
    r = rig(plan=json.dumps({"phases": []}), max_attempts=3)
    outcome = r.run()

    assert r.camera.plan_calls == 3, "the repair loop had its three attempts"
    assert (outcome.outcome, outcome.failure_stage) == ("failure", "invention")
    assert outcome.plan is None and outcome.legs_recorded == 0
    assert r.backend.legs == []
    # With no plan there is no hitl.json to say it, so the event is the record.
    (ended,) = r.sink.named("trial_outcome")
    assert (ended["outcome"], ended["failure_stage"]) == ("failure", "invention")
    assert "could not decompose" in ended["reason"]


def test_an_abandoned_step_is_on_the_plan_and_in_the_events(rig):
    r = rig({}, "abort")
    outcome = r.run()

    assert (outcome.outcome, outcome.failure_stage) == ("aborted", None)
    assert record_of(outcome)["outcome"] == "aborted"
    (ended,) = r.sink.named("trial_outcome")
    assert ended["outcome"] == "aborted" and "abandoned" in ended["reason"]
    assert r.camera.asked == [], "an abandoned step is not checked"


def test_the_deprecated_verify_phase_alias_is_no_longer_what_the_loop_calls():
    from tandem.core import phase_loop

    source = Path(phase_loop.__file__).read_text()
    assert "verify_phase" not in source
    assert "verify_effects" in source and "verify_preconditions" in source


# --- through a session: no label, filed as excluded, legs kept ------------------------------------


@pytest.fixture
def session_for(profile, tmp_path, monkeypatch):
    from tandem.planning import llm

    made: list[Session] = []

    def build(answers, *, plan=PLAN, **hitl):
        profile.hitl.enabled = True
        # These tests answer a human phase "done", for a step done by hand. While recording that is
        # refused unless allowed (tests/test_executor_integration.py).
        profile.hitl.allow_unrecorded_human_phase = True
        for key, value in hitl.items():
            setattr(profile.hitl, key, value)
        monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
        camera = Camera(plan, answers)
        monkeypatch.setattr(llm, "gemini_client", lambda: camera)
        backends = use_fake_backend(monkeypatch)
        session = Session(profile, task="put the toy in the box")
        states: list[str] = []
        session.subscribe(
            lambda message: states.append(message["state"]) if message.get("type") == "state" else None
        )
        session.start()
        assert wait_for(lambda: session.state is State.AWAITING_TASK), f"stuck in {session.state}"
        made.append(session)
        return session, backends, states

    yield build
    for session in made:
        if session.alive:
            session.stop(park=False)
            session.wait(timeout=5)


@pytest.fixture
def merges(monkeypatch):
    """Every merge the session asks for, passed through to the real one."""
    from tandem.core import merge as merge_mod

    real = merge_mod.merge
    calls: list[tuple[str, str | None]] = []

    def spy(profile, trajectory_id, *, status=None, tools_dir=None):
        calls.append((trajectory_id, status))
        return real(profile, trajectory_id, status=status, tools_dir=tools_dir)

    monkeypatch.setattr(merge_mod, "merge", spy)
    return calls


def session_events(session) -> list:
    return events_mod.read_all(Path(session.summary()["events_file"]))


def filed_record(profile, status: str) -> dict | None:
    records = sorted(profile.status_dir(status).glob("*/hitl.json"))
    return json.loads(records[0].read_text()) if records else None


def test_an_excluded_trial_is_filed_under_failure_without_a_label(session_for, merges, profile):
    session, backends, states = session_for({OPEN: True, CLOSED: True}, verify_retries=0)
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE), f"stuck in {session.state}"
    session.complete_human_phase()

    assert wait_for(lambda: session.excluded_count == 1), f"stuck in {session.state}"
    assert wait_for(lambda: session.state is State.AWAITING_TASK)
    # Never asked: a label the operator could answer "success" would put it back in the dataset.
    assert "awaiting_label" not in states
    assert session.labeled_count == 0 and session.success_count == 0
    assert session.rollouts[-1].status == "excluded" and session.rollouts[-1].success is None

    # Filed and merged all the same, so the raw legs survive for whoever audits the exclusion.
    trajectory_id = backends[-1].legs[0]["leg"].trajectory_id
    assert wait_for(lambda: filed_record(profile, "failure") is not None), "no hitl.json under failure/"
    # The record is written before the merge starts (so a merge that never finishes still leaves it).
    assert wait_for(lambda: merges == [(trajectory_id, "failure")]), merges
    assert not any(profile.status_dir("eval").glob("*/_meta.json")), "a leg was left unfiled in eval/"

    record = filed_record(profile, "failure")
    assert (record["outcome"], record["excluded"], record["failure_stage"]) == (
        "excluded",
        True,
        "verification",
    )
    assert record["filed_under"] == "failure"
    assert "open the box" in record["outcome_reason"]
    deleted = [v for v in record["verifications"] if v["role"] == "effect (deleted)"]
    assert deleted and deleted[0]["satisfied"] is False and deleted[0]["phase"] == 1

    # The events file says how it ended, and that it was filed without a label.
    names = [event.name for event in session_events(session)]
    assert "trial_outcome" in names and "trial_excluded" in names
    assert "awaiting_label" not in names and "labeled" not in names
    (excluded,) = [e.payload for e in session_events(session) if e.name == "trial_excluded"]
    assert (excluded["outcome"], excluded["failure_stage"], excluded["trajectory_id"]) == (
        "excluded",
        "verification",
        trajectory_id,
    )

    # And the operator is told, since nothing asked them anything.
    summary = session.summary()
    assert summary["excluded"] == 1
    assert summary["last_trial"]["excluded"] is True
    assert any("excluded from the dataset" in line["text"] for line in session.logs())


def test_an_excluded_trial_does_not_count_towards_the_episode_target(session_for):
    session, _, _ = session_for({OPEN: True, CLOSED: True}, verify_retries=0)
    session.max_episodes = 1
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE)
    session.complete_human_phase()
    assert wait_for(lambda: session.excluded_count == 1)
    assert wait_for(lambda: session.state is State.AWAITING_TASK)
    assert session.alive, "an excluded trial is not an episode, so the target is not reached"


def test_with_on_verification_failure_label_the_operator_is_asked_and_decides(session_for, merges, profile):
    session, _, states = session_for(
        {OPEN: True, CLOSED: True}, verify_retries=0, on_verification_failure="label"
    )
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE)
    session.complete_human_phase()

    assert wait_for(lambda: session.state is State.AWAITING_LABEL), f"stuck in {session.state}"
    # The prompt can say why the trial stopped before the operator answers it.
    assert session.summary()["last_trial"]["failure_stage"] == "verification"
    session.label(True)
    assert wait_for(lambda: filed_record(profile, "success") is not None)

    record = filed_record(profile, "success")
    # The operator overruled the check -- which is what this setting is for -- and the stage the
    # loop stopped at stays on the record beside their answer.
    assert (record["outcome"], record["excluded"], record["failure_stage"]) == (
        "success",
        False,
        "verification",
    )
    assert any(not v["satisfied"] for v in record["verifications"])
    (labeled,) = [e.payload for e in session_events(session) if e.name == "labeled"]
    assert (labeled["outcome"], labeled["failure_stage"], labeled["success"]) == (
        "success",
        "verification",
        True,
    )
    assert session.excluded_count == 0


def test_a_trial_with_nothing_recorded_still_says_how_it_ended(session_for):
    session, _, _ = session_for({}, plan=json.dumps({"phases": []}))
    session.next_task()
    assert wait_for(lambda: any(e.name == "rollout_discarded" for e in session_events(session)))

    (discarded,) = [e.payload for e in session_events(session) if e.name == "rollout_discarded"]
    assert (discarded["outcome"], discarded["failure_stage"]) == ("failure", "invention")
    assert "could not decompose" in discarded["reason"]
    assert wait_for(lambda: session.state is State.AWAITING_TASK)


def test_the_terminal_says_why_a_trial_came_back_without_a_label():
    """An excluded trial goes straight back to the task prompt, so the prompt has to say why."""
    from rich.text import Text

    from tandem.cli.collect import _excluded_note, _stopped_note

    note = _excluded_note({"excluded": True, "reason": "the step [open the box] did not verify"})
    # The reason is in the model's and the plan's words, so it is escaped before Rich reads it.
    assert Text.from_markup(note).plain.startswith(
        "Excluded, not labeled: the step [open the box] did not verify"
    )
    assert _excluded_note({"excluded": False, "reason": "labeled as usual"}) == ""
    stopped = _stopped_note({"failure_stage": "verification", "reason": "still closed"})
    assert Text.from_markup(stopped).plain.startswith("Stopped at verification: still closed")
    assert _stopped_note({"failure_stage": None, "reason": None}) == ""
