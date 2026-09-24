"""What a robot leg is asked for, what it is told first, and how it ends the trial when it cannot go on.

A robot phase's subgoal goes to the TAMP system with the current scene (TANDEM Sec. IV-D), through
``TampBackend.plan`` and ``execute``. Around that one call the loop owes the planner and the dataset
several things, each learned from a demonstration that came out wrong:

* the leg is told which objects it may pick (``movables``) and whether it ends at home
  (``return_home``) -- but only where the planner declares it can be told;
* the scene is perceived afresh before every robot leg and never before a person's step, and the
  first pass after a person had the arm opens the gripper;
* a leg that could not be PLANNED follows ``on_robot_phase_failure`` (abort by default, teleop, or
  replan with the failure fed back to the model); a leg that was planned and could not EXECUTE
  always ends the trial, and the plan never advances past it;
* a leg with nothing to plan for ends the trial instead of perceiving and proposing forever.

Driven through the phase loop on its own against a stand-in backend and a canned model, and through a
real session where only the session can show it (the label prompt, the record on disk).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from fake_backend import FakeBackend
from fake_executor import FakeExecutor, use_fake_executor
from helpers import use_fake_backend, wait_for

from tandem.core import secrets
from tandem.core.episodes import LegDirs
from tandem.core.phase_loop import PhaseLoop
from tandem.core.session import Session, State
from tandem.executors.base import ExecutorContext
from tandem.planners.base import ExecuteResult, PlanResult
from tandem.planners.tiptop.capabilities import CAPABILITIES
from tandem.planning.config import PlanningConfig

# What the camera is asked about, in the words it is asked in.
OPEN = "the container white_box is open, so its interior is visible"
TOY_ON_TABLE = "blue_toy is resting on top of table"
TOY_IN_BOX = "blue_toy is resting on top of white_box"

TAKE_OFF = {
    "executor": "robot",
    "description": "take the toy off the box",
    "atoms": [{"predicate": "On", "args": ["blue_toy", "table"]}],
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
BLOCK_IN = {
    "executor": "robot",
    "description": "put the block inside the box",
    "atoms": [{"predicate": "On", "args": ["red_block", "white_box"]}],
}

INVENTED = {"IsOpen": "the container {0} is open, so its interior is visible"}


def proposal(*phases) -> dict:
    """A model's answer: these phases, and every invented predicate they use (an unused one is refused)."""
    used = json.dumps(phases)
    return {
        "new_predicates": [
            {"name": name, "instructions": text} for name, text in INVENTED.items() if f'"{name}"' in used
        ],
        "phases": list(phases),
    }


# Robot, then the person, then the robot again.
PLAN = proposal(TAKE_OFF, OPEN_THE_BOX, PUT_IN)


class Camera:
    """The model: the plan for a proposal, and a verdict by STATEMENT for a classifier question.

    Keyed on the statement rather than on call order, so a test says what the camera sees rather
    than how many times the loop will ask. A statement no test answered fails the test: the camera
    was asked something it was not meant to be.
    """

    PLAN_MARKER = "ORDERED list of phases"

    def __init__(self, plan, answers: dict | None = None) -> None:
        self.plan = plan if isinstance(plan, str) else json.dumps(plan)
        self.answers = dict(answers or {})
        self.asked: list[str] = []
        self.plan_prompts: list[str] = []
        self.aio = self

    @property
    def models(self):
        return self

    async def generate_content(self, model, contents, config):
        prompt = contents[-1]
        if "Statement:" in prompt and self.PLAN_MARKER not in prompt:
            statement = prompt.split("Statement:", 1)[1].split("\n", 1)[0].strip()
            self.asked.append(statement)
            if statement not in self.answers:
                raise AssertionError(f"the camera was asked about {statement!r}, which no test answer covers")
            holds = self.answers[statement]
            return mock.Mock(text=json.dumps({"holds": holds, "reason": "as the test says"}))
        self.plan_prompts.append(prompt)
        return mock.Mock(text=self.plan)


class Backend(FakeBackend):
    """The stand-in backend, plus the two ways a robot leg fails, each on chosen legs only.

    ``plan_failures`` and ``execute_failures`` map the n-th call (from 0) of that verb to the reason
    it fails with. A failed execution still recorded its frames: the arm moved before it stopped.
    ``no_goal`` makes perception's own goal translation come back empty.
    """

    def __init__(self, runtime=None, **kwargs) -> None:
        self.plan_failures = dict(kwargs.pop("plan_failures", {}))
        self.execute_failures = dict(kwargs.pop("execute_failures", {}))
        self.no_goal = kwargs.pop("no_goal", False)
        super().__init__(runtime, **kwargs)
        self.planned = 0
        self.executed = 0

    def perceive(self, **kwargs):
        scene = super().perceive(**kwargs)
        return dataclasses.replace(scene, detected_goal=()) if self.no_goal else scene

    def plan(self, scene_id, goal, **kwargs) -> PlanResult:
        n, self.planned = self.planned, self.planned + 1
        result = super().plan(scene_id, goal, **kwargs)
        if n in self.plan_failures:
            return PlanResult(ok=False, failure_reason=self.plan_failures[n])
        return result

    def execute(self, plan_handle, leg, *, save_dir: Path, should_stop=None) -> ExecuteResult:
        n, self.executed = self.executed, self.executed + 1
        result = super().execute(plan_handle, leg, save_dir=save_dir, should_stop=should_stop)
        if n in self.execute_failures:
            return dataclasses.replace(result, ok=False, failure_reason=self.execute_failures[n])
        return result


# A planner that declares neither keyword -- and so, as the protocol allows, does not take them.
PLAIN = dataclasses.replace(CAPABILITIES, supports_movable_restriction=False, supports_return_home=False)


class PlainBackend(Backend):
    def capabilities(self):
        return PLAIN

    def plan(self, scene_id, goal, *, surfaces=frozenset(), save_dir, reuse_skeleton=None) -> PlanResult:
        return super().plan(scene_id, goal, surfaces=surfaces, save_dir=save_dir)


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
    """The person at the prompts: "done" to every human phase unless told otherwise.

    Also a watchdog. Every step boundary passes through `check_preempt`, so a loop that goes round
    and round without ending -- the bug an empty goal used to cause -- fails the test here instead
    of hanging the suite.
    """

    BOUNDARIES = 40

    def __init__(self, *answers: str, handoff_after_human: bool = False) -> None:
        self.answers = list(answers)
        self.prompts = 0
        self.shown: list = []
        self.human_phase = None
        self.boundaries = 0
        self.handoff_after_human = handoff_after_human
        self._handoff = False

    def check_preempt(self) -> None:
        self.boundaries += 1
        if self.boundaries > self.BOUNDARIES:
            raise AssertionError(f"the loop went round {self.BOUNDARIES} times without ending the trial")

    def take_handoff_request(self) -> bool:
        asked, self._handoff = self._handoff, False
        return asked

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
        self._handoff = self.handoff_after_human
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

    def make(
        answers=None,
        *operator_answers,
        plan=PLAN,
        backend_type=Backend,
        backend_kwargs=None,
        handoff_after_human=False,
        **cfg,
    ):
        camera = Camera(plan, {OPEN: True} if answers is None else answers)
        monkeypatch.setattr(llm, "gemini_client", lambda: camera)
        backend = backend_type(None, output_dir=tmp_path / "frames", **(backend_kwargs or {}))
        # The human executor with nobody on the other end: every leg "records" 30 frames.
        sink, hands = Sink(), FakeExecutor()
        use_fake_executor(monkeypatch, hands)
        operator = Operator(*operator_answers, handoff_after_human=handoff_after_human)
        loop = PhaseLoop(
            backend,
            backend.capabilities(),
            # The operator answers "done" for a step done by hand, which while recording is refused
            # unless allowed (tests/test_executor_integration.py); these tests are about the robot.
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


def goals(backend) -> list[list]:
    """The goal of every plan asked for, as [predicate, *args] per atom."""
    return [[[a["predicate"], *a["args"]] for a in r["goal"]] for r in backend.plan_requests]


def record_of(outcome) -> dict:
    return json.loads(json.dumps(outcome.plan.to_json(), default=str))


# --- a leg that does not execute ------------------------------------------------------------------


def test_a_leg_that_fails_to_execute_ends_the_trial_and_the_plan_never_advances(rig):
    r = rig(backend_kwargs={"execute_failures": {0: "the arm hit a joint limit"}}, check_tamp_effects=True)
    outcome = r.run()

    assert (outcome.outcome, outcome.failure_stage) == ("failure", "tamp_execution")
    assert "take the toy off the box" in outcome.reason and "joint limit" in outcome.reason
    # Stopped where it failed: the person was never asked to open a box the robot had not cleared,
    # and the robot never went on to the last phase.
    assert r.operator.prompts == 0 and r.hands.calls == []
    assert ran(r.backend) == [0]
    assert outcome.plan.index == 0, "the plan advanced past a leg that did not execute"
    # Nobody asked the camera to confirm motion that did not finish.
    assert r.camera.asked == []
    assert not any(call.startswith("capture_frame") for call in r.backend.calls)
    # The frames it did record are on disk and still need a label.
    assert outcome.legs_recorded == 1

    record = record_of(outcome)
    assert (record["outcome"], record["failure_stage"], record["excluded"]) == (
        "failure",
        "tamp_execution",
        False,
    )
    # What was planned for the leg that failed is on the record: it is what the failure is audited by.
    assert record["phases"][0]["task_plan"] == ["Pick(blue_toy)", "Place(blue_toy, table)"]
    (ended,) = r.sink.named("trial_outcome")
    assert (ended["failure_stage"], ended["phase_index"]) == ("tamp_execution", 0)


def test_a_later_leg_that_fails_to_execute_stops_there_too(rig):
    r = rig(None, "teleop", backend_kwargs={"execute_failures": {1: "the gripper slipped"}})
    outcome = r.run()

    assert ran(r.backend) == [0, 2]
    assert (outcome.outcome, outcome.failure_stage) == ("failure", "tamp_execution")
    assert outcome.plan.index == 2 and not outcome.plan.finished
    # Every leg is on disk -- the robot's two and the person's -- so the operator still labels it.
    assert outcome.legs_recorded == 3


# --- what the leg is told -------------------------------------------------------------------------


def test_each_leg_may_pick_only_what_a_robot_phase_moves_and_only_the_last_goes_home(rig):
    # The screwdriver is on the table, detected, and nobody's business but the person's.
    r = rig(backend_kwargs={"labels": ("blue_toy", "white_box", "screwdriver")})
    outcome = r.run()

    assert ran(r.backend) == [0, 2]
    asked = [(req["movables"], req["return_home"]) for req in r.backend.plan_requests]
    assert asked == [
        # The box is placed INTO and the screwdriver is left alone: neither is ever a thing to pick.
        (frozenset({"blue_toy"}), False),  # a person continues from where this leg stops
        (frozenset({"blue_toy"}), True),  # nothing follows the last leg
    ]
    started = r.sink.named("rollout_start")
    assert [(e["movables"], e["return_home"]) for e in started] == [
        (["blue_toy"], False),
        (["blue_toy"], True),
    ]
    # And the record says what the planner ran for each robot phase.
    record = record_of(outcome)
    assert record["phases"][0]["task_plan"] == ["Pick(blue_toy)", "Place(blue_toy, table)"]
    assert record["phases"][2]["task_plan"] == ["Pick(blue_toy)", "Place(blue_toy, white_box)"]


@pytest.mark.parametrize(
    ("conjoin", "home", "phases"),
    [(True, [True], [0]), (False, [False, True], [0, 1])],
    ids=["one-leg", "a-leg-per-phase"],
)
def test_consecutive_robot_phases_go_home_only_at_the_end_of_the_task(rig, conjoin, home, phases):
    plan = proposal(PUT_IN, BLOCK_IN)
    r = rig(
        plan=plan,
        conjoin_robot_phases=conjoin,
        backend_kwargs={"labels": ("blue_toy", "red_block", "white_box")},
    )
    outcome = r.run()

    assert outcome.plan.finished
    assert ran(r.backend) == phases
    assert [req["return_home"] for req in r.backend.plan_requests] == home
    assert {req["movables"] for req in r.backend.plan_requests} == {frozenset({"blue_toy", "red_block"})}
    if conjoin:
        # One plan for both phases, recorded against each of them.
        assert record_of(outcome)["phases"][1]["covers_phases"] == [0, 1]
    else:
        # Robot after robot: nobody touched the gripper, so the second pass leaves it alone.
        assert [req["open_gripper"] for req in r.backend.perceive_requests] == [False, False]


def test_a_planner_that_declares_neither_is_never_handed_either(rig):
    """Its plan() takes neither keyword, as the protocol allows: handing it one is a TypeError."""
    r = rig(backend_type=PlainBackend)
    outcome = r.run()

    assert ran(r.backend) == [0, 2]
    assert outcome.outcome is None
    # The defaults, filled in by the stand-in's own plan(): nothing reached it from the loop.
    assert [(req["movables"], req["return_home"]) for req in r.backend.plan_requests] == [(None, True)] * 2
    assert all("movables" not in e and "return_home" not in e for e in r.sink.named("rollout_start"))


def test_with_phase_planning_off_the_leg_is_an_ordinary_rollout(rig):
    r = rig(enabled=False)
    r.run()

    (request,) = r.backend.plan_requests
    assert (request["movables"], request["return_home"]) == (None, True)


# --- when the scene is looked at ------------------------------------------------------------------


def test_the_scene_is_perceived_before_each_robot_leg_and_never_before_a_person(rig):
    r = rig()
    r.run()

    assert r.backend.call_names() == [
        "perceive",  # the proposal needs it, and so does the first leg
        "plan",
        "execute",
        "capture_frame",  # the person's step is judged on a fresh frame, not a perception pass
        "perceive",  # the scene the person left, before the robot plans on it
        "plan",
        "execute",
        # Nothing after the last leg: the task is over.
    ]


def test_a_plan_that_opens_and_ends_with_people_takes_one_pass_per_robot_leg(rig):
    plan = proposal(OPEN_THE_BOX, PUT_IN, {**OPEN_THE_BOX, "description": "check the box is still open"})
    r = rig(plan=plan)
    outcome = r.run()

    assert outcome.plan.finished
    assert r.backend.call_names() == [
        "perceive",  # for the proposal; the first step is the person's
        "capture_frame",
        "perceive",
        "plan",
        "execute",
        "capture_frame",
    ]


def test_the_first_robot_leg_after_a_person_opens_the_gripper_first(rig):
    r = rig()
    r.run()

    assert [(req["reset_arm"], req["open_gripper"]) for req in r.backend.perceive_requests] == [
        (True, False),  # nothing is mid-task yet: park the arm, and leave the hand as it is
        (False, True),  # a person had the arm: do not move it, but open the fingers they may have closed
    ]


def test_a_handoff_the_operator_asked_for_leaves_the_gripper_as_they_left_it(rig):
    # The operator asks for the arm right after the person's step, before the robot's next leg.
    r = rig(handoff_after_human=True)
    r.run()

    assert r.hands.calls[-1] is None, "the operator's own hand-off, with no phase"
    assert ran(r.backend) == [0, 2]
    assert [(req["reset_arm"], req["open_gripper"]) for req in r.backend.perceive_requests] == [
        (True, False),
        (False, True),  # after the human phase
        (False, False),  # after the operator's hand-off: what the arm holds is theirs to decide
    ]


def test_drifted_labels_are_rebound_on_the_robot_leg_after_the_person(rig):
    r = rig(backend_kwargs={"drifted_labels": ("small_blue_toy", "white_box")})
    outcome = r.run()

    assert outcome.plan.finished
    # The person's step was shown and checked in the plan's own names; no pass came between.
    assert r.operator.shown[0].description == "open the box"
    # The leg after it was planned in the names this pass produced.
    assert goals(r.backend)[-1] == [["on", "small_blue_toy", "white_box"]]
    assert r.backend.plan_requests[-1]["movables"] == frozenset({"small_blue_toy"})


def test_an_object_the_next_leg_needs_that_is_gone_ends_the_trial(rig):
    r = rig(backend_kwargs={"drifted_labels": ("red_cube", "white_box")})
    outcome = r.run()

    assert ran(r.backend) == [0]
    assert (outcome.outcome, outcome.failure_stage) == ("failure", "tamp_planning")
    assert "blue_toy" in outcome.reason
    assert record_of(outcome)["failure_stage"] == "tamp_planning"


# --- a leg that cannot be planned -----------------------------------------------------------------


def test_by_default_a_phase_that_cannot_be_planned_ends_the_trial_at_tamp_planning(rig):
    r = rig(backend_kwargs={"plan_failures": {1: "no collision-free grasp"}})
    outcome = r.run()

    assert ran(r.backend) == [0]
    assert (outcome.outcome, outcome.failure_stage) == ("failure", "tamp_planning")
    assert "no collision-free grasp" in outcome.reason
    assert len(r.camera.plan_prompts) == 1, "abort does not propose again"
    record = record_of(outcome)
    assert (record["outcome"], record["failure_stage"], record["phase_index"]) == (
        "failure",
        "tamp_planning",
        2,
    )
    # With the phase it was about, so the events file ties the failure to a phase of the episode.
    assert r.sink.named("phase_plan_failed") == [
        {"reason": "no collision-free grasp", "policy": "abort", "phase_index": 2}
    ]


def test_teleop_hands_the_phase_to_a_person_who_is_checked_on_its_atoms(rig):
    r = rig(
        {OPEN: True, TOY_ON_TABLE: True},
        backend_kwargs={"plan_failures": {0: "no grasp"}},
        on_robot_phase_failure="teleop",
    )
    outcome = r.run()

    # Phase 0 was the robot's; the person did it instead, then their own step, then the robot the last.
    assert [phase.description for phase in r.operator.shown] == ["take the toy off the box", "open the box"]
    assert ran(r.backend) == [2]
    handed = outcome.plan.phases[0]
    assert handed.is_human and handed.operator is None, "no operator is made up for a handed-over phase"
    # Checked on exactly what the planner was going to be asked for.
    assert TOY_ON_TABLE in r.camera.asked
    assert outcome.outcome is None, "the trial carries on, and the label decides it"
    # The robot's first leg follows people, so the gripper is opened before it looks.
    assert r.backend.perceive_requests[-1]["open_gripper"] is True


def test_replan_tells_the_model_why_the_last_plan_failed(rig):
    reason = "no collision-free grasp on blue_toy"
    r = rig(backend_kwargs={"plan_failures": {0: reason}}, on_robot_phase_failure="replan")
    outcome = r.run()

    first, again = r.camera.plan_prompts
    assert "could not be planned" not in first
    # What failed, why, and what that phase asked for: the model never sees the plan it replaces.
    assert f"phase 0 could not be planned: {reason}" in again
    assert "take the toy off the box" in again and "On(blue_toy, table)" in again
    # The second plan ran to the end.
    assert outcome.plan.finished and outcome.outcome is None
    assert ran(r.backend) == [0, 2]
    assert r.backend.perceptions == 3, "a pass for each proposal, and one after the person"


def test_replan_is_bounded_by_max_attempts_and_remembers_every_failure(rig):
    failures = {n: f"no plan, attempt {n}" for n in range(10)}
    r = rig(backend_kwargs={"plan_failures": failures}, on_robot_phase_failure="replan", max_attempts=2)
    outcome = r.run()

    # The original proposal and two re-plans, then it gives up.
    assert len(r.camera.plan_prompts) == 3
    assert r.backend.planned == 3 and r.backend.legs == []
    assert (outcome.outcome, outcome.failure_stage) == ("failure", "tamp_planning")
    assert "after 2 re-plan(s)" in outcome.reason
    # The last proposal was told about both earlier failures, not only the latest.
    last = r.camera.plan_prompts[-1]
    assert "no plan, attempt 0" in last and "no plan, attempt 1" in last
    assert any("out of re-planning attempts" in line for line in r.sink.lines)


# --- a leg with nothing to plan for ---------------------------------------------------------------


def test_with_phase_planning_off_an_empty_goal_ends_the_trial_instead_of_looping(rig):
    r = rig(enabled=False, backend_kwargs={"no_goal": True})
    outcome = r.run()

    assert (outcome.outcome, outcome.failure_stage) == ("failure", "tamp_planning")
    assert "found no goal" in outcome.reason
    assert r.backend.perceptions == 1, "looked once, and did not go round again"
    assert r.backend.plan_requests == []
    (ended,) = r.sink.named("trial_outcome")
    assert ended["failure_stage"] == "tamp_planning"


EMPTY_HANDED = {
    "executor": "robot",
    "description": "let go of whatever it holds",
    "atoms": [{"predicate": "HandEmpty", "args": []}],
}


def test_a_robot_phase_the_planner_can_be_given_nothing_for_is_sent_back_to_the_proposer(rig):
    """HandEmpty() is achievable, and TipTop supplies it for itself: no goal survives rendering.

    That used to be found out by the loop, as the leg came up -- after the robot's earlier legs had
    run -- and the trial ended at invention with the proposer never told. It is refused inside the
    repair loop now, so the model is told why on every attempt; a model that never fixes it still
    ends the trial at invention, having planned nothing.
    """
    # A person's step after it, so the phase is a leg of its own rather than conjoined with the next.
    r = rig(plan=proposal(EMPTY_HANDED, OPEN_THE_BOX, PUT_IN))
    outcome = r.run()

    assert (outcome.outcome, outcome.failure_stage) == ("failure", "invention")
    assert "HandEmpty()" in outcome.reason and "let go of whatever it holds" in outcome.reason
    assert len(r.camera.plan_prompts) == 3, "the repair loop had its three attempts"
    assert "cannot be given as a goal" in r.camera.plan_prompts[1], "and the model was told why"
    assert r.backend.plan_requests == [] and r.backend.legs == []
    assert outcome.plan is None


def test_an_empty_goal_that_gets_past_the_proposal_still_ends_the_trial_instead_of_looping(rig, monkeypatch):
    """The loop's own check stays as the backstop, for a leg the proposal-time check never saw."""
    from tandem.planning import proposal as proposal_mod

    monkeypatch.setattr(proposal_mod, "check_plan", lambda spec, cfg, caps: None)
    r = rig(plan=proposal(EMPTY_HANDED, OPEN_THE_BOX, PUT_IN))
    outcome = r.run()

    assert (outcome.outcome, outcome.failure_stage) == ("failure", "invention")
    assert "HandEmpty()" in outcome.reason and "let go of whatever it holds" in outcome.reason
    assert len(r.camera.plan_prompts) == 1, "not proposed again"
    assert r.backend.plan_requests == []
    assert record_of(outcome)["failure_stage"] == "invention"


# --- through a session ----------------------------------------------------------------------------


@pytest.fixture
def session_for(profile, tmp_path, monkeypatch):
    from tandem.planning import llm

    made: list[Session] = []

    def build(answers=None, *, plan=PLAN, backend_kwargs=None, **hitl):
        profile.hitl.enabled = True
        # These tests answer a human phase "done", for a step done by hand. While recording that is
        # refused unless allowed (tests/test_executor_integration.py).
        profile.hitl.allow_unrecorded_human_phase = True
        for key, value in hitl.items():
            setattr(profile.hitl, key, value)
        monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
        camera = Camera(plan, {OPEN: True} if answers is None else answers)
        monkeypatch.setattr(llm, "gemini_client", lambda: camera)
        backends = use_fake_backend(monkeypatch, backend_type=Backend, **(backend_kwargs or {}))
        session = Session(profile, task="put the toy in the box")
        session.start()
        assert wait_for(lambda: session.state is State.AWAITING_TASK), f"stuck in {session.state}"
        made.append(session)
        return session, backends

    yield build
    for session in made:
        if session.alive:
            session.stop(park=False)
            session.wait(timeout=5)


def test_a_failed_execution_reaches_the_label_prompt_saying_where_it_stopped(session_for, profile):
    session, backends = session_for(backend_kwargs={"execute_failures": {0: "the arm hit a joint limit"}})
    session.next_task()

    # The leg recorded frames before it stopped, so the operator labels it -- with the stage shown.
    assert wait_for(lambda: session.state is State.AWAITING_LABEL), f"stuck in {session.state}"
    assert session.human_phase is None, "the person was never asked to carry on from it"
    last = session.summary()["last_trial"]
    assert (last["failure_stage"], last["outcome"]) == ("tamp_execution", "failure")
    session.label(False)

    def filed():
        records = sorted(profile.status_dir("failure").glob("*/hitl.json"))
        return json.loads(records[0].read_text()) if records else None

    assert wait_for(lambda: filed() is not None), "no hitl.json under failure/"
    record = filed()
    assert (record["outcome"], record["failure_stage"], record["excluded"]) == (
        "failure",
        "tamp_execution",
        False,
    )
    assert "joint limit" in record["outcome_reason"]
    assert [leg["leg"].phase_index for leg in backends[-1].legs] == [0]


def test_a_persons_step_straight_after_another_reaches_the_page(session_for):
    """No perception pass comes between two people's steps any more, and it used to be what changed
    the session's state between them. The web page repaints on a change of state, so without one the
    second step's prompt never reaches it and the operator is left looking at the first."""
    fold = {**OPEN_THE_BOX, "description": "fold the flaps flat"}
    session, backends = session_for(plan=proposal(OPEN_THE_BOX, fold, PUT_IN))
    frames: list[tuple[str, str | None]] = []

    def on_message(message: dict) -> None:
        if message.get("type") == "state":
            frames.append((message["state"], (message.get("human_phase") or {}).get("description")))

    session.subscribe(on_message)
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE), f"stuck in {session.state}"
    session.complete_human_phase()

    assert wait_for(lambda: ("awaiting_human_phase", "fold the flaps flat") in frames), frames
    # And the person's two steps were never separated by a look at the scene.
    assert backends[-1].perceptions == 1
