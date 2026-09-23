"""The phase loop on its own, with no session around it.

The loop is given everything it touches: the backend, the settings, an event sink, the person at the
prompts, and the seam a human phase goes through. That is what lets the trial algorithm be changed,
and tested, without a state machine or a session thread. These tests hold it to that. Nothing here
builds a Session: each dependency is a plain object that records what the loop asked of it.

The end-to-end behaviour through a real session is covered in test_hitl.py and test_session.py.
"""

from __future__ import annotations

import json

import pytest
from fake_backend import FakeBackend
from helpers import FakeGemini

from tandem.core.episodes import LegDirs
from tandem.core.phase_loop import PhaseLoop
from tandem.planning.config import PlanningConfig

# Robot, then a person, then the robot again: the smallest plan that exercises every branch of a
# walk. The box has to be opened before the toy can go in it.
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


class Sink:
    """The event sink: every event and log line, in order."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []
        self.lines: list[str] = []

    def event(self, name: str, **payload) -> None:
        self.events.append((name, payload))

    def log(self, text: str) -> None:
        self.lines.append(text)

    def names(self) -> list[str]:
        return [name for name, _ in self.events]


class Operator:
    """The person at the prompts, scripted: answers to human phases are given in order."""

    def __init__(self, *answers: str, handoff_requests: int = 0) -> None:
        self.answers = list(answers)
        self.handoff_requests = handoff_requests
        self.prompts = 0
        self.shown: list = []
        self.human_phase = None
        self.progress = None
        self.unrepresented: list = []
        self.rollouts: list[dict] = []

    def check_preempt(self) -> None:
        return None

    def take_handoff_request(self) -> bool:
        if self.handoff_requests:
            self.handoff_requests -= 1
            return True
        return False

    def rolling(self) -> None:
        return None

    def show_progress(self, progress) -> None:
        self.progress = progress

    def show_unrepresented(self, clauses) -> None:
        self.unrepresented = clauses

    def show_human_phase(self, phase) -> None:
        self.human_phase = phase
        if phase is not None:
            self.shown.append(phase)

    def await_human_phase(self) -> str:
        self.prompts += 1
        return self.answers.pop(0)

    def rollout_started(self, save_dir) -> None:
        self.rollouts.append({"dir": save_dir, "n_frames": None})

    def rollout_saved(self, n_frames: int) -> None:
        self.rollouts[-1]["n_frames"] = n_frames


class Hands:
    """The human-leg seam with nobody on the other end: each call "records" one leg."""

    def __init__(self, n_frames: int = 30, fail_after_recording: str | None = None) -> None:
        self.n_frames = n_frames
        self.fail_after_recording = fail_after_recording
        self.calls: list = []

    def __call__(self, phase, recorded) -> None:
        self.calls.append(phase)
        if self.n_frames:
            recorded(self.n_frames)
        if self.fail_after_recording:
            raise RuntimeError(self.fail_after_recording)


@pytest.fixture
def build(profile, tmp_path, monkeypatch):
    """A loop wired to a stand-in backend, a canned model, and the recorders above."""
    from tandem.planning import llm

    def make(*answers, verdicts=(HOLDS,), backend_kwargs=None, hands=None, handoff_requests=0, **cfg):
        client = FakeGemini(json.dumps(PLAN), list(verdicts))
        monkeypatch.setattr(llm, "gemini_client", lambda: client)
        backend = FakeBackend(None, output_dir=tmp_path / "frames", **(backend_kwargs or {}))
        sink = Sink()
        operator = Operator(*answers, handoff_requests=handoff_requests)
        hands = hands or Hands()
        loop = PhaseLoop(
            backend,
            backend.capabilities(),
            PlanningConfig(**{"enabled": True, "save_vlm_io": False, **cfg}),
            events=sink,
            operator=operator,
            human_leg=hands,
            legs=LegDirs(profile, tmp_path / "session", log=sink.log),
        )
        return loop, backend, sink, operator, hands

    return make


def run(loop):
    return loop.run(task="put the toy in the box", instruction="put the toy in the box", trajectory_id="t" * 16)


def test_a_whole_plan_is_walked_robot_person_robot(build):
    loop, backend, sink, operator, hands = build("teleop")
    outcome = run(loop)

    # Both robot phases were planned and executed, stamped with the trajectory the caller minted.
    assert [leg["leg"].phase_index for leg in backend.legs] == [0, 2]
    assert {leg["leg"].trajectory_id for leg in backend.legs} == {"t" * 16}
    # The person was asked once, took the arm through the seam, and the step was checked.
    assert operator.prompts == 1
    assert [phase.description for phase in hands.calls] == ["open the box"]
    assert "capture_frame:external" in backend.calls
    assert operator.shown[-1].verified is True
    assert operator.human_phase is None, "the prompt must be cleared once the step is over"
    assert operator.progress == (3, 3)
    # Two robot legs and the person's: all three are on disk, so all three need a label.
    assert outcome.legs_recorded == 3
    assert outcome.plan is not None and outcome.plan.finished
    assert (outcome.outcome, outcome.failure_stage) == (None, None), "the label decides a finished trial"
    assert outcome is loop.outcome


def test_with_phase_planning_off_the_backends_goal_is_one_robot_leg(build):
    loop, backend, sink, operator, hands = build(enabled=False)
    outcome = run(loop)

    planned = [json.loads(c.split(":", 1)[1]) for c in backend.calls if c.startswith("plan:")]
    assert planned == [[{"predicate": "on", "args": ["blue_toy", "table"]}]]
    assert outcome.legs_recorded == 1
    assert outcome.plan is None
    assert not hands.calls and operator.prompts == 0
    # The operator was shown the leg as it started and told how long it came out.
    assert [rollout["n_frames"] for rollout in operator.rollouts] == [backend.n_frames]
    assert str(operator.rollouts[0]["dir"]) == backend.legs[0]["dir"]


def test_a_handoff_the_operator_asked_for_goes_through_the_seam_with_no_phase(build):
    """Lending the arm is not a phase: nothing advances, and the task is planned again after."""
    loop, backend, sink, operator, hands = build(enabled=False, handoff_requests=1)
    outcome = run(loop)

    assert hands.calls == [None]
    assert len(backend.legs) == 1, "the robot still ran the task once the arm came back"
    assert outcome.legs_recorded == 2


def test_aborting_the_step_ends_the_attempt_and_keeps_the_plan_for_the_record(build):
    loop, backend, sink, operator, hands = build("abort")
    outcome = run(loop)

    assert [leg["leg"].phase_index for leg in backend.legs] == [0], "nothing ran after the abort"
    assert operator.human_phase is None
    assert outcome.outcome == "aborted"
    # The plan is dropped from the walk but kept for the audit record, which is the episode whose
    # provenance is most worth having.
    assert outcome.plan is not None and not outcome.plan.finished
    assert outcome.legs_recorded == 1


def test_a_step_that_never_verifies_ends_the_attempt_at_verification(build):
    loop, backend, sink, operator, hands = build("done", verdicts=(DOES_NOT_HOLD,), verify_retries=0)
    outcome = run(loop)

    assert outcome.failure_stage == "verification"
    assert [leg["leg"].phase_index for leg in backend.legs] == [0]
    verified = [payload for name, payload in sink.events if name == "human_phase_verified"]
    assert verified and verified[0]["ok"] is False
    assert "a flap is still closed" in operator.shown[-1].missing[0]


def test_a_phase_that_cannot_be_planned_ends_the_attempt_at_tamp_planning(build):
    loop, backend, sink, operator, hands = build(backend_kwargs={"plan_failure": "no grasp"})
    outcome = run(loop)

    assert outcome.failure_stage == "tamp_planning"
    assert outcome.legs_recorded == 0
    assert ("phase_plan_failed", {"reason": "no grasp", "policy": "abort"}) in sink.events


def test_a_leg_on_disk_is_counted_even_when_the_arm_cannot_be_taken_back(build):
    """Taking the arm back can fail and end the session, after the person's leg reached disk.

    That leg still has to be labeled and merged on the way out, so the seam reports it BEFORE it
    takes the arm back. The loop has to have counted it by the time the failure reaches the caller.
    """
    loop, backend, sink, operator, hands = build(
        "teleop", hands=Hands(fail_after_recording="the planner could not take the robot back")
    )
    with pytest.raises(RuntimeError, match="take the robot back"):
        run(loop)

    assert loop.outcome.legs_recorded == 2, "the robot's leg and the person's"
    assert loop.outcome.plan is not None


def test_legs_nothing_was_recorded_into_are_taken_out_of_the_dataset(build, profile, tmp_path):
    """Every pass allocates a leg directory; the human phase's and the planning failure's are empty."""
    loop, backend, sink, operator, hands = build(backend_kwargs={"plan_failure": "no grasp"})
    run(loop)

    assert not list(profile.status_dir("eval").iterdir()), "an empty leg was left in the dataset"
    kept = list((tmp_path / "session" / "perception").iterdir())
    assert len(kept) == 1 and (kept[0] / "perception_rgb.png").is_file()
