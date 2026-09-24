"""A human phase carried out by a human executor, as one leg of the trajectory.

The paper gives every magic operator an executor, pi_omega_Delta, that brings its effects about; TANDEM's
own is a person teleoperating the arm. The phase loop now asks for it by name (``hitl.human_executor``)
through the executor registry, and owns everything around it:

* the custody transfer: the planner lets go of the arm and the cameras, the executor carries out the
  leg, the leg is counted as soon as it is on disk, and the planner takes the arm back last -- except
  from an executor that will not let go, whose cameras nothing may reach for;
* how the leg ended: ``done`` is checked by the camera, ``ended_by_operator`` goes ahead unchecked and
  says so, ``aborted`` ends the trial at ``human_policy``;
* what a step with no leg is worth: while recording, "done" by hand is refused unless
  ``allow_unrecorded_human_phase`` says otherwise;
* the operator's own hand-off between phases, which is always teleop;
* the demonstration: every leg carries the phase it carried out, so the merged episode reads back as
  tau = ((tau_1, phi_1), .., (tau_N, phi_N)).

Most of it runs the phase loop on its own against a stand-in backend, a canned model and a stand-in
executor (``fake_executor``). What only the session can show -- the prompt refusing "done", a forced
stop reaching the leg, the teleop driver's events file per leg -- runs through a real session.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fake_backend import FakeBackend
from fake_executor import FakeExecutor, use_fake_executor, write_leg
from helpers import FakeGemini, use_fake_backend, wait_for

from tandem.core import episodes, secrets
from tandem.core import merge as merge_mod
from tandem.core.episodes import LegDirs
from tandem.core.errors import SessionConflict
from tandem.core.phase_loop import PhaseLoop
from tandem.core.session import Session, State
from tandem.executors.base import CustodyError, ExecutorContext
from tandem.planning.config import PlanningConfig

HOLDS = json.dumps({"holds": True, "reason": "the flaps are folded back"})
DOES_NOT_HOLD = json.dumps({"holds": False, "reason": "a flap is still closed over the opening"})

TAKE_OFF = {
    "executor": "robot",
    "description": "take the toy off the box",
    "atoms": [{"predicate": "On", "args": ["blue_toy", "table"]}],
}
BLOCK_OFF = {
    "executor": "robot",
    "description": "put the block on the table",
    "atoms": [{"predicate": "On", "args": ["red_block", "table"]}],
}
OPEN_THE_BOX = {
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
}
PUT_IN = {
    "executor": "robot",
    "description": "put the toy inside the box",
    "atoms": [{"predicate": "On", "args": ["blue_toy", "white_box"]}],
}
IS_OPEN = {"name": "IsOpen", "instructions": "the container {0} is open, so its interior is visible"}

# Robot, the person, the robot again.
PLAN = {"new_predicates": [IS_OPEN], "phases": [TAKE_OFF, OPEN_THE_BOX, PUT_IN]}
TASK = "put the toy in the box"


class Sink:
    def __init__(self, timeline: list) -> None:
        self.events: list[tuple[str, dict]] = []
        self.lines: list[str] = []
        self._timeline = timeline

    def event(self, name: str, **payload) -> None:
        self.events.append((name, payload))
        if name == "human_leg_ended":
            self._timeline.append("leg_ended")

    def log(self, text: str) -> None:
        self.lines.append(text)

    def named(self, name: str) -> list[dict]:
        return [payload for event, payload in self.events if event == name]


class Operator:
    """The person at the prompts. Hands every human phase to the executor unless told otherwise.

    Records the hand-off's transitions on the shared timeline, next to the backend's and the
    executor's, so a test can read off the order the custody transfer happened in. A loop that keeps
    asking fails the test at the 20th prompt instead of hanging the suite.
    """

    def __init__(self, timeline: list, *answers: str, handoffs: int = 0, holds_arm: bool = False) -> None:
        self.answers = list(answers)
        self.handoffs = handoffs
        self.holds_arm = holds_arm
        self.timeline = timeline
        self.prompts = 0
        self.shown: list = []
        self.human_phase = None

    def check_preempt(self) -> None:
        return None

    def take_handoff_request(self) -> bool:
        if self.handoffs:
            self.handoffs -= 1
            return True
        return False

    def rolling(self) -> None:
        return None

    def show_progress(self, progress) -> None:
        return None

    def show_unrepresented(self, clauses) -> None:
        return None

    def show_human_phase(self, phase) -> None:
        self.human_phase = phase
        if phase is not None:
            self.shown.append(phase)

    def await_human_phase(self) -> str:
        self.prompts += 1
        if self.prompts > 20:
            raise AssertionError("the operator was asked 20 times; the loop is not moving on")
        return self.answers.pop(0) if self.answers else "teleop"

    def rollout_started(self, save_dir) -> None:
        return None

    def rollout_saved(self, n_frames: int) -> None:
        return None

    def handing_off(self) -> None:
        self.timeline.append("handing_off")

    def arm_lent(self):
        self.timeline.append("arm_lent")
        # Nobody hands the arm back unless the test says a person is holding it.
        return (lambda: False) if self.holds_arm else (lambda: True)

    def arm_returned(self) -> None:
        self.timeline.append("arm_returned")


class Backend(FakeBackend):
    """The stand-in backend, putting each custody call on the shared timeline, and optionally writing
    each robot leg in the real on-disk format so a merge can join it."""

    def __init__(self, runtime=None, **kwargs) -> None:
        self.timeline = kwargs.pop("timeline")
        self.write = kwargs.pop("write", False)
        self.release_fails = kwargs.pop("release_fails", None)
        self.legs_counted_at_reacquire: list[int] = []
        self.loop = None
        super().__init__(runtime, **kwargs)

    def release_hardware(self) -> None:
        self.timeline.append("release")
        if self.release_fails:
            self.calls.append("release_hardware")
            raise RuntimeError(self.release_fails)
        super().release_hardware()

    def reacquire_hardware(self) -> None:
        self.timeline.append("reacquire")
        if self.loop is not None:
            self.legs_counted_at_reacquire.append(self.loop.outcome.legs_recorded)
        super().reacquire_hardware()

    def execute(self, plan_handle, leg, *, save_dir: Path, should_stop=None):
        result = super().execute(plan_handle, leg, save_dir=save_dir, should_stop=should_stop)
        if self.write:
            write_leg(save_dir, leg, n_frames=self.n_frames, source="tamp")
        return result


@pytest.fixture
def rig(profile, tmp_path, monkeypatch):
    """A phase loop wired to the stand-ins above. Nothing runs until `.run()`."""
    from tandem.planning import llm

    def make(
        *answers,
        plan=PLAN,
        verdicts=(HOLDS,),
        executor: FakeExecutor | None = None,
        executor_name: str = "teleop",
        teleop: FakeExecutor | None = None,
        backend_kwargs=None,
        record: bool = True,
        handoffs: int = 0,
        holds_arm: bool = False,
        **cfg,
    ):
        client = FakeGemini(json.dumps(plan), list(verdicts))
        monkeypatch.setattr(llm, "gemini_client", lambda: client)
        timeline: list = []
        backend = Backend(None, output_dir=tmp_path / "frames", timeline=timeline, **(backend_kwargs or {}))
        executor = executor or FakeExecutor()
        executor.on_run = lambda request, leg: timeline.append(
            ("run", "released" if not backend.holds_hardware else "HELD")
        )
        built = use_fake_executor(monkeypatch, executor, name=executor_name)
        if teleop is not None:
            # A separate stand-in for the operator's own hand-off, when the human phases go elsewhere.
            teleop.on_run = executor.on_run
            use_fake_executor(monkeypatch, teleop, name="teleop")
        problems: list[str] = []
        sink = Sink(timeline)
        operator = Operator(timeline, *answers, handoffs=handoffs, holds_arm=holds_arm)
        loop = PhaseLoop(
            backend,
            backend.capabilities(),
            PlanningConfig(**{"enabled": True, "save_vlm_io": False, "human_executor": executor_name, **cfg}),
            events=sink,
            operator=operator,
            executor_context=ExecutorContext(
                profile=profile, session_dir=tmp_path / "session", on_problem=problems.append
            ),
            legs=LegDirs(profile, tmp_path / "session", log=sink.log),
            record=record,
        )
        backend.loop = loop

        def run():
            return loop.run(task=TASK, instruction=TASK, trajectory_id="t" * 16)

        return SimpleNamespace(
            loop=loop,
            backend=backend,
            sink=sink,
            operator=operator,
            executor=executor,
            teleop=teleop,
            built=built,
            client=client,
            timeline=timeline,
            problems=problems,
            run=run,
        )

    return make


def ran(backend) -> list:
    return [leg["leg"].phase_index for leg in backend.legs]


def record_of(outcome) -> dict:
    return json.loads(json.dumps(outcome.plan.to_json(), default=str))


# --- which executor, built how often --------------------------------------------------------------


def test_the_profiles_executor_carries_out_the_phase_and_is_built_once(rig):
    """Looked up by name through the registry, like a planner, and kept: building one may start a
    process, and a session runs attempt after attempt through one loop."""
    r = rig(executor=FakeExecutor(segment_source="policy"), executor_name="diffusion-policy")
    first = r.run()
    second = r.run()

    assert [request.description for request in r.executor.calls] == ["open the box", "open the box"]
    assert len(r.built) == 1, "the executor was built again for the second attempt"
    assert first.legs_recorded == second.legs_recorded == 3
    assert ran(r.backend) == [0, 2, 0, 2]


def test_the_executor_is_asked_for_what_the_operator_was_shown(rig):
    r = rig()
    r.run()

    (request,) = r.executor.calls
    shown = r.operator.shown[0]
    assert (request.phase_index, request.n_phases) == (shown.index, shown.total) == (1, 3)
    assert request.description == "open the box"
    assert request.instructions == "Open the white_box and fold its flaps back."
    assert (
        request.expected == shown.expected == ["the container white_box is open, so its interior is visible"]
    )
    assert (request.attempt, request.missing) == (1, [])
    # The magic operator goes along, for an executor that conditions on it.
    assert request.operator["name"] == "Open"
    assert request.operator["add_effects"] == ["IsOpen(white_box)"]

    (leg,) = r.executor.legs
    # One trajectory with the robot's legs, labelled with the WHOLE task, stamped with its phase.
    assert leg.trajectory_id == "t" * 16
    assert leg.instruction == TASK
    assert leg.segment_source == "teleop"
    assert (leg.phase_index, leg.n_phases, leg.phase_description) == (1, 3, "open the box")
    assert leg.record is True
    assert shown.executor == "teleop"


def test_a_retry_asks_the_executor_again_aimed_at_what_was_missed(rig):
    r = rig(verdicts=(DOES_NOT_HOLD, HOLDS), verify_retries=1)
    outcome = r.run()

    assert [request.attempt for request in r.executor.calls] == [1, 2]
    assert r.executor.calls[0].missing == []
    assert "a flap is still closed" in r.executor.calls[1].missing[0]
    assert outcome.legs_recorded == 4, "both of the person's legs are on disk, and so both are counted"
    assert outcome.plan.finished


# --- the custody transfer -------------------------------------------------------------------------


def test_the_arm_is_released_before_the_leg_and_taken_back_after_it_is_counted(rig):
    r = rig()
    r.run()

    human = r.timeline[r.timeline.index("handing_off") :]
    assert human[:6] == [
        "handing_off",
        "release",
        "arm_lent",
        ("run", "released"),
        "reacquire",
        "leg_ended",
    ]
    assert human[6] == "arm_returned"
    # Counted BEFORE the arm is taken back: the robot's first leg and the person's.
    assert r.backend.legs_counted_at_reacquire == [2]


def test_an_arm_the_planner_cannot_take_back_ends_the_session_with_the_leg_counted(rig, monkeypatch):
    r = rig()

    def refuse() -> None:
        r.backend.calls.append("reacquire_hardware")
        raise RuntimeError("the robot client timed out")

    monkeypatch.setattr(r.backend, "reacquire_hardware", refuse)
    with pytest.raises(CustodyError, match="could not take the robot back: the robot client timed out"):
        r.run()
    assert r.loop.outcome.legs_recorded == 2, "the person's leg reached disk and must still be filed"


def test_an_executor_that_will_not_let_go_ends_the_session_without_reaching_for_the_arm(rig):
    """It still holds the cameras. Reaching for them would fail like broken hardware rather than say
    what is true, so the error goes up untouched and nothing is reacquired."""
    r = rig(executor=FakeExecutor(custody=True))
    with pytest.raises(CustodyError, match="will not let go"):
        r.run()
    assert "release_hardware" in r.backend.calls
    assert "reacquire_hardware" not in r.backend.calls
    assert "arm_returned" not in r.timeline


def test_a_release_that_fails_is_shown_and_the_leg_still_goes_ahead(rig):
    """The person may already have their hands on the arm; only they can say whether to carry on."""
    r = rig(backend_kwargs={"release_fails": "the gripper did not answer"})
    outcome = r.run()

    assert r.problems == ["the planner could not release the robot: the gripper did not answer"]
    assert r.sink.named("teleop_handoff_warning")
    assert len(r.executor.calls) == 1
    assert "reacquire_hardware" in r.backend.calls
    assert outcome.plan.finished


def test_kill_reaches_the_leg_in_flight_and_the_trial_ends_at_human_policy(rig):
    executor = FakeExecutor(wait=True)
    r = rig(executor=executor, holds_arm=True)
    results: list = []
    runner = threading.Thread(target=lambda: results.append(r.run()), daemon=True)
    runner.start()
    assert wait_for(executor.running.is_set), "the leg never started"

    r.loop.kill()
    runner.join(timeout=10.0)
    assert not runner.is_alive()
    (outcome,) = results
    assert (outcome.outcome, outcome.failure_stage) == ("failure", "human_policy")
    # The arm is still taken back after a killed leg.
    assert r.backend.calls.count("reacquire_hardware") == 1


# --- how the leg ended ----------------------------------------------------------------------------


def test_an_aborted_leg_ends_the_trial_at_human_policy(rig):
    r = rig(executor=FakeExecutor(statuses=("aborted",)))
    outcome = r.run()

    assert (outcome.outcome, outcome.failure_stage) == ("failure", "human_policy")
    assert "open the box" in outcome.reason
    # Nothing the arm did may be read as the phase being done: no check, and no later robot phase.
    assert r.client.verdict_calls == 0
    assert ran(r.backend) == [0]
    assert outcome.plan.index == 1, "the plan advanced past a phase nobody carried out"
    # Its frames are on disk all the same, and are filed with the rest.
    assert outcome.legs_recorded == 2
    record = record_of(outcome)
    assert (record["outcome"], record["failure_stage"]) == ("failure", "human_policy")


def test_a_leg_the_operator_ended_goes_ahead_unchecked_and_says_so(rig):
    r = rig(executor=FakeExecutor(statuses=("ended_by_operator",)))
    outcome = r.run()

    assert r.client.verdict_calls == 0, "the camera was asked about a leg the operator cut short"
    assert not any(call.startswith("capture_frame") for call in r.backend.calls)
    (verified,) = r.sink.named("human_phase_verified")
    assert verified["ok"] is None and "ended the teleop leg" in verified["skipped"]
    # On the record as a phase that went through without a check -- not as one that passed.
    record = record_of(outcome)
    assert "ended the teleop leg" in record["phases"][1]["unchecked"]
    assert ran(r.backend) == [0, 2] and outcome.plan.finished


# --- a step with no leg ---------------------------------------------------------------------------


def test_done_by_hand_is_refused_while_recording_and_the_step_asked_again(rig):
    r = rig("done", "teleop")
    outcome = r.run()

    assert r.operator.prompts == 2
    first, second = r.operator.shown
    assert first.by_hand is False and second.by_hand is False
    assert second.attempt == 1, "a refusal is not a retry: nothing was checked"
    (refused,) = r.sink.named("human_phase_refused")
    assert refused["phase_index"] == 1
    assert any("being recorded" in line and "allow_unrecorded_human_phase" in line for line in r.sink.lines)
    assert len(r.executor.calls) == 1
    assert outcome.plan.finished and outcome.legs_recorded == 3


def test_done_by_hand_stands_when_nothing_is_recorded(rig):
    r = rig("done", record=False)
    outcome = r.run()

    assert r.operator.shown[0].by_hand is True
    assert r.executor.calls == [], "nobody was asked to carry the step out"
    assert r.client.verdict_calls == 1, "a step done by hand is still checked"
    assert outcome.plan.finished and outcome.legs_recorded == 2


def test_done_by_hand_stands_while_recording_when_the_profile_allows_it(rig):
    r = rig("done", allow_unrecorded_human_phase=True)
    outcome = r.run()

    assert r.operator.shown[0].by_hand is True
    assert r.executor.calls == [] and not r.sink.named("human_phase_refused")
    assert outcome.plan.finished


def test_an_executor_leg_that_recorded_nothing_is_refused_while_recording(rig):
    """Teleop with no driver running still lends the arm and comes back ``done`` with no frames. That is
    the same step with no demonstration, whoever drove the arm."""
    r = rig("teleop", "abort", executor=FakeExecutor(n_frames=0))
    outcome = r.run()

    assert len(r.executor.calls) == 1
    assert r.sink.named("human_phase_refused")[0]["reason"] == "the teleop leg recorded nothing"
    assert r.client.verdict_calls == 0
    assert outcome.outcome == "aborted"


def test_the_retry_line_offers_done_only_where_it_would_be_accepted(rig):
    from tandem.planning.plan import retry_message

    assert "or say it IS done" in retry_message(["the box is open"], 1, by_hand=True)
    assert "say it IS done" not in retry_message(["the box is open"], 1, by_hand=False)

    # The single retry most profiles have is offered with its instruction: the line counts the goes
    # left with the one about to start, not after it (which left the default retry with no line).
    r = rig(verdicts=(DOES_NOT_HOLD, HOLDS), verify_retries=1)
    r.run()
    (line,) = [text for text in r.sink.lines if text.startswith("The workspace does not look like")]
    assert "Take the arm again and finish it." in line and "say it IS done" not in line

    # And an executor that is told how many goes it has is told them all: two retries, then one.
    r = rig(
        verdicts=(DOES_NOT_HOLD, DOES_NOT_HOLD, HOLDS),
        verify_retries=2,
        executor=FakeExecutor(segment_source="policy"),
        executor_name="diffusion-policy",
    )
    r.run()
    lines = [text for text in r.sink.lines if text.startswith("The workspace does not look like")]
    assert ["(2 attempt(s) left)" in lines[0], "(1 attempt(s) left)" in lines[1]] == [True, True]


def test_an_executor_leg_that_recorded_nothing_stands_when_nothing_is_recorded(rig):
    r = rig(executor=FakeExecutor(n_frames=0), record=False)
    outcome = r.run()
    assert not r.sink.named("human_phase_refused")
    assert outcome.plan.finished and outcome.legs_recorded == 2


# --- the operator's own hand-off ------------------------------------------------------------------


def test_a_handoff_the_operator_asks_for_is_teleop_whatever_does_the_human_phases(rig):
    """Only an executor a person drives can lend the arm with no phase attached."""
    policy, teleop = FakeExecutor(segment_source="policy"), FakeExecutor()
    r = rig(executor=policy, executor_name="diffusion-policy", teleop=teleop, handoffs=1)
    r.run()

    assert teleop.calls[0] is None
    (lent,) = teleop.legs[:1]
    assert lent.segment_source == "teleop"
    assert (lent.phase_index, lent.n_phases, lent.phase_description) == (None, None, "")
    assert lent.trajectory_id == "t" * 16
    # The human phase itself still went to the profile's executor, recorded as its kind of leg.
    assert [request.description for request in policy.calls] == ["open the box"]
    assert policy.legs[0].segment_source == "policy"


# --- the demonstration ----------------------------------------------------------------------------


@pytest.fixture
def fake_video(monkeypatch):
    """Stand in for ffprobe/ffmpeg: every clip holds as many frames as its leg has states."""

    def leg_video_frames(leg, cameras, runtime_dir):
        with np.load(leg["dir"] / "robot_state.npz") as store:
            n = len(store["frame_time"])
        return n, {cam: n for cam in cameras}

    def concat_videos(legs, camera, leg_frames, dest, scratch, runtime_dir):
        dest.write_bytes(b"joined")
        return sum(leg_frames)

    monkeypatch.setattr(merge_mod, "_leg_video_frames", leg_video_frames)
    monkeypatch.setattr(merge_mod, "_concat_videos", concat_videos)


def test_the_merged_episode_maps_every_stretch_back_to_its_phase(rig, profile, fake_video):
    """tau = ((tau_1, phi_1), .., (tau_N, phi_N)), read back off one merged episode.

    The merge copies each leg's phase stamp into segments[], and hitl.json beside it holds the phases.
    Two robot phases planned as one goal are one leg, stamped with the first and recorded as covering
    both (``covers_phases``). Every phase must come back exactly once, in order, from the stretch of
    frames that carried it out, and by the kind of executor the plan gave it to.
    """
    plan = {"new_predicates": [IS_OPEN], "phases": [TAKE_OFF, BLOCK_OFF, OPEN_THE_BOX, PUT_IN]}
    r = rig(
        plan=plan,
        executor=FakeExecutor(write=True, n_frames=12),
        backend_kwargs={"write": True, "labels": ("blue_toy", "red_block", "white_box")},
    )
    outcome = r.run()
    assert outcome.plan.finished and outcome.legs_recorded == 3

    episodes.merge_trajectory(
        profile,
        "t" * 16,
        "success",
        outcome.plan,
        tools_dir=None,
        vlm_dir=None,
        log=lambda text: None,
        emit=lambda message: None,
    )
    (merged,) = [d for d in profile.status_dir("success").iterdir() if (d / "hitl.json").is_file()]
    meta = json.loads((merged / "_meta.json").read_text())
    record = json.loads((merged / "hitl.json").read_text())
    segments, phases = meta["segments"], record["phases"]

    assert [s["phase_index"] for s in segments] == [0, 2, 3]
    assert [s["source"] for s in segments] == ["tamp", "teleop", "tamp"]
    assert meta["n_phases"] == len(phases) == 4

    kind = {"tamp": "robot", "teleop": "human"}
    walked = []
    for segment in segments:
        covered = phases[segment["phase_index"]].get("covers_phases", [segment["phase_index"]])
        assert {phases[k]["executor"] for k in covered} == {kind[segment["source"]]}
        assert segment["phase_description"] == "; ".join(phases[k]["description"] for k in covered)
        walked += covered
    assert walked == [0, 1, 2, 3], "a phase is missing from the demonstration, or claimed twice"

    # And the stretches tile the episode: every frame belongs to exactly one phase's leg.
    assert sum(s["n_frames"] for s in segments) == meta["n_frames"]
    for before, after in zip(segments, segments[1:], strict=False):
        assert after["video_start"] == before["video_stop"]


# --- through a session ----------------------------------------------------------------------------


@pytest.fixture
def session_for(profile, tmp_path, monkeypatch):
    """A live session with phase planning on, a stand-in backend, a canned model, and a stand-in
    human executor that holds the arm until control is returned."""
    from tandem.planning import llm

    made: list[Session] = []

    def build(*verdicts, record=None, executor_kwargs=None, **hitl):
        profile.hitl.enabled = True
        for key, value in hitl.items():
            setattr(profile.hitl, key, value)
        monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
        client = FakeGemini(json.dumps(PLAN), list(verdicts) or [HOLDS])
        monkeypatch.setattr(llm, "gemini_client", lambda: client)
        backends = use_fake_backend(monkeypatch)
        executors = use_fake_executor(monkeypatch, **{"wait": True, **(executor_kwargs or {})})
        session = Session(profile, task=TASK, record=record)
        session.start()
        assert wait_for(lambda: session.state is State.AWAITING_TASK), f"stuck in {session.state}"
        made.append(session)
        return session, backends, executors

    yield build
    for session in made:
        if session.alive:
            session.stop(park=False)
            session.wait(timeout=5)


def test_the_prompt_refuses_done_while_recording_and_the_executor_records_the_step(session_for):
    session, backends, executors = session_for()
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE), f"stuck in {session.state}"

    shown = session.summary()["human_phase"]
    assert shown["by_hand"] is False and shown["executor"] == "teleop"
    with pytest.raises(SessionConflict, match="being recorded") as refused:
        session.complete_human_phase()
    assert "allow_unrecorded_human_phase" in refused.value.hint
    assert session.state is State.AWAITING_HUMAN_PHASE, "a refused answer must leave the prompt up"

    session.request_teleop()
    assert wait_for(lambda: session.state is State.TELEOP_HANDOFF), f"stuck in {session.state}"
    (executor,) = executors
    assert wait_for(executor.running.is_set)
    assert not backends[-1].holds_hardware, "the executor ran while the planner held the arm"
    assert session.summary()["can_preempt"] is False

    session.resume_from_teleop()
    assert wait_for(lambda: session.state is State.AWAITING_LABEL), f"stuck in {session.state}"
    assert backends[-1].holds_hardware
    assert session._legs_recorded == 3, "the robot's two legs and the person's"
    names = backends[-1].call_names()
    assert names.index("release_hardware") < names.index("reacquire_hardware")


def test_the_session_builds_its_executor_once_for_every_task(session_for):
    """One loop per session, holding the executors it built: a policy's server is started once, not
    at every task."""
    session, _, executors = session_for()
    for task in (1, 2):
        session.next_task()
        assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE), f"stuck in {session.state}"
        session.request_teleop()
        assert wait_for(lambda t=task: len(executors[0].calls) == t if executors else False)
        assert wait_for(executors[0].running.is_set)
        session.resume_from_teleop()
        assert wait_for(lambda: session.state is State.AWAITING_LABEL), f"stuck in {session.state}"
        session.label(True)
        assert wait_for(lambda: session.state is State.AWAITING_TASK)
    assert len(executors) == 1, "the executor was built again for the next task"


def test_with_recording_off_the_prompt_accepts_done(session_for):
    session, _, executors = session_for(record=False)
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE)
    assert session.summary()["human_phase"]["by_hand"] is True

    session.complete_human_phase()
    assert wait_for(lambda: session.state is State.AWAITING_LABEL), f"stuck in {session.state}"
    assert not executors, "nobody should have been asked to carry the step out"


def test_the_summary_says_who_does_a_human_step_and_whether_they_can(session_for, profile):
    session, _, _ = session_for(human_executor="teleop")
    summary = session.summary()
    assert summary["human_executor"]["name"] == "teleop"
    assert summary["human_executor"]["display_name"] == FakeExecutor.display_name
    assert summary["human_executor"]["ready"] is True
    assert summary["teleop_available"] is True


def test_teleop_that_this_machine_lacks_is_not_offered(profile, tmp_path, monkeypatch):
    """Read from the executor registry now, not from one settings flag: a DROID interpreter that is not
    on this machine is teleop nobody can take, whatever teleop.enabled says."""
    from tandem.core import settings as settings_mod
    from tandem.core.settings import Settings

    cfg = Settings()
    cfg.teleop.enabled = True
    cfg.teleop.python = str(tmp_path / "no-such-python")
    monkeypatch.setattr(settings_mod, "load", lambda: cfg)
    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
    use_fake_backend(monkeypatch)
    session = Session(profile, task=TASK)
    session.start()
    try:
        assert wait_for(lambda: session.state is State.AWAITING_TASK)
        summary = session.summary()
        assert summary["teleop_available"] is False
        assert any("teleop.python" in item for item in summary["human_executor"]["unmet"])
    finally:
        session.stop(park=False)
        session.wait(timeout=5)


def test_an_executor_the_profile_names_that_cannot_load_stops_the_session_before_it_warms(
    profile, tmp_path, monkeypatch
):
    from tandem.core.errors import TandemError
    from tandem.executors import base

    monkeypatch.setattr(base, "_registered", dict(base._registered))
    base.register_human_executor("broken", "no_such_module_anywhere:FACTORY")
    profile.hitl.enabled = True
    profile.hitl.human_executor = "broken"
    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
    backends = use_fake_backend(monkeypatch)
    session = Session(profile, task=TASK)
    with pytest.raises(TandemError, match="could not be loaded"):
        session.start()
    assert not backends, "a planner was built for a session that could never carry out a human phase"


def test_a_forced_stop_kills_the_executor_in_flight(session_for):
    session, backends, executors = session_for()
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE)
    session.request_teleop()
    assert wait_for(lambda: len(executors) == 1)
    executor = executors[0]
    assert wait_for(executor.running.is_set)

    session.force_stop()
    assert executor.killed.is_set(), "a forced stop left the executor driving the arm"
    assert wait_for(lambda: not session.alive), f"stuck in {session.state}"
    # Cut off, not handed back: nothing the arm did is read as the step being done.
    assert session.last_trial["failure_stage"] == "human_policy"


def _rendered(renderable) -> str:
    from rich.console import Console

    console = Console(record=True, width=120)
    console.print(renderable)
    return console.export_text()


def test_the_terminal_prompt_offers_only_the_answers_that_will_be_accepted():
    from tandem.cli import collect

    phase = {"description": "open the box", "instructions": "Open it.", "index": 1, "total": 3, "attempt": 1}
    teleop = {"name": "teleop", "display_name": "Teleoperation", "ready": True, "unmet": [], "error": None}
    policy = {"name": "diffusion-policy", "display_name": "Diffusion policy", "ready": True, "unmet": []}

    recorded = collect._footer(
        State.AWAITING_HUMAN_PHASE, {"human_phase": {**phase, "by_hand": False}, "human_executor": teleop}
    ).plain
    assert "I did it" not in recorded and "take the arm" in recorded
    by_hand = collect._footer(
        State.AWAITING_HUMAN_PHASE, {"human_phase": {**phase, "by_hand": True}, "human_executor": policy}
    ).plain
    assert "I did it" in by_hand and "run Diffusion policy" in by_hand

    # Neither answer is possible: say what this machine lacks rather than leave only "give up".
    lacking = {**teleop, "ready": False, "unmet": ["teleop.python is not an interpreter on this machine"]}
    stuck = collect._footer(
        State.AWAITING_HUMAN_PHASE, {"human_phase": {**phase, "by_hand": False}, "human_executor": lacking}
    ).plain
    assert "I did it" not in stuck and "take the arm" not in stuck and "give up" in stuck
    panel = _rendered(collect._human_phase_panel({**phase, "by_hand": False}, lacking))
    assert "teleop.python is not an interpreter" in panel


# --- the teleop driver, one events file per leg ---------------------------------------------------


@pytest.fixture
def teleop_driver(monkeypatch):
    """Teleop on, with the stand-in DROID driver behind the real TeleopExecutor."""
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


def test_each_teleop_handoff_follows_its_own_events_file(profile, tmp_path, monkeypatch, teleop_driver):
    """The session used to point every hand-off at one teleop-events.jsonl, and the driver's events
    are followed from the first byte -- so each later hand-off replayed the earlier legs' events and
    reported their frames as its own. Now every leg has its own file, and its own count."""
    from tandem.planning import llm

    profile.hitl.enabled = True
    profile.hitl.verify_retries = 1
    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
    client = FakeGemini(json.dumps(PLAN), [DOES_NOT_HOLD, HOLDS])
    monkeypatch.setattr(llm, "gemini_client", lambda: client)
    backends = use_fake_backend(monkeypatch)
    ended: list[dict] = []
    started: list[dict] = []

    def on_message(message: dict) -> None:
        if message.get("type") == "event" and message.get("event") == "human_leg_ended":
            ended.append(message)
        if message.get("type") == "teleop_event" and message.get("event") == "rollout_start":
            started.append(message)

    session = Session(profile, task=TASK)
    session.subscribe(on_message)
    session.start()
    try:
        assert wait_for(lambda: session.state is State.AWAITING_TASK)
        session.next_task()
        for attempt in (1, 2):
            assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE), f"stuck in {session.state}"
            assert wait_for(
                lambda a=attempt: session.human_phase is not None and session.human_phase.attempt == a
            )
            session.request_teleop()
            assert wait_for(lambda a=attempt: len(started) >= a, timeout=20.0), "the driver never recorded"
            session.resume_from_teleop()
        assert wait_for(lambda: session.state is State.AWAITING_LABEL, timeout=20.0), (
            f"stuck in {session.state}"
        )
    finally:
        session.stop(park=False)
        session.wait(timeout=10)

    teleop_legs = [
        leg
        for leg in merge_mod.find_legs(profile, backends[-1].legs[0]["leg"].trajectory_id)
        if leg["source"] == "teleop"
    ]
    assert len(teleop_legs) == 2
    # Each hand-off counted exactly what its own leg saved, not what an earlier one did.
    assert [event["n_frames"] for event in ended] == [leg["meta"]["n_frames"] for leg in teleop_legs]
    assert all(leg["meta"]["phase_index"] == 1 for leg in teleop_legs)
    scratch = session._files["session_dir"]
    assert not (scratch / "teleop-events.jsonl").exists(), "a session-wide events file is back"
    per_leg = sorted((scratch / "teleop").glob("leg-*/teleop-events.jsonl"))
    assert len(per_leg) == 2
