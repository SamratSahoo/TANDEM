"""The phase loop and the session around it, held to what the review found they got wrong.

* A stop is a step boundary. After it nothing is perceived, planned, executed or checked, and what
  was recorded is filed -- or, at the label prompt, kept unlabeled in eval/ with its phase record.
* A trial the loop ended itself is never labeled "success": an unfinished plan is filed under
  failure/ with no prompt (aborted, or failed at its stage), and a planner verb or an executor that
  raises is recorded at its stage before anyone is asked anything.
* A preempt stops a planner that declares cooperative stop part-way, and such a leg never advances
  the plan.
* A forced stop that lands while the arm is being released never starts the leg.
* A hand-off asked for while a person's step is due is that step's own, stamped leg.
* A person's step with nothing a camera can settle is recorded as unchecked, not as passed.
* A planner that cannot capture a frame is refused at start when the checks need one.
* The record of a replanned trial keeps the plans it replaced, and which plan each leg belongs to.
* Human executors are closed when the session ends; merges are waited for.

Loop-level tests drive ``PhaseLoop`` against the stand-in backend, a canned model and a stand-in
executor; session-level ones run a real ``Session`` on the same stand-ins.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fake_backend import FakeBackend
from fake_executor import FakeExecutor, use_fake_executor, write_leg
from helpers import FakeGemini, isolate_registry, use_fake_backend, wait_for

from tandem.core import episodes, secrets
from tandem.core import events as events_mod
from tandem.core import merge as merge_mod
from tandem.core.episodes import LegDirs
from tandem.core.errors import TandemError
from tandem.core.phase_loop import PhaseLoop
from tandem.core.session import Session, State
from tandem.executors.base import CustodyError, ExecutorContext, HumanPhaseResult
from tandem.planners.base import BackendError, ExecuteResult
from tandem.planning.config import PlanningConfig

TASK = "put the toy in the box"
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


def proposal(*phases, **extra) -> dict:
    used = json.dumps(phases)
    return {"new_predicates": [IS_OPEN] if '"IsOpen"' in used else [], "phases": list(phases), **extra}


# Robot, the person, the robot again.
PLAN = proposal(TAKE_OFF, OPEN_THE_BOX, PUT_IN)
# Two robot phases in a row, planned as two legs when conjoin_robot_phases is off.
TWO_ROBOT = proposal(TAKE_OFF, BLOCK_OFF)
LABELS = ("blue_toy", "red_block", "white_box")


class Hooked(FakeBackend):
    """The stand-in backend, with hooks a test can hang a stop, a crash or a hand-off request on.

    * ``on_execute(leg)``: called as a leg starts executing.
    * ``execute_raises`` / ``perceive_raises``: the n-th call (from 0) raises this instead of answering.
    * ``cooperative``: declare ``supports_cooperative_stop``, and poll ``should_stop`` through the leg.
    * ``write``: put each robot leg on disk in the real format, so a merge can join it.
    * ``release_gate``: an Event ``release_hardware`` blocks on, for a test to act mid-release.
    """

    def __init__(self, runtime=None, **kwargs) -> None:
        self.execute_raises = dict(kwargs.pop("execute_raises", {}))
        self.perceive_raises = dict(kwargs.pop("perceive_raises", {}))
        self.cooperative = kwargs.pop("cooperative", False)
        self.write = kwargs.pop("write", False)
        self.release_gate: threading.Event | None = kwargs.pop("release_gate", None)
        self.on_execute = None
        self.executed = 0
        self.perceived = 0
        self.execute_kwargs: list[dict] = []
        super().__init__(runtime, **kwargs)

    def capabilities(self):
        caps = super().capabilities()
        return replace(caps, supports_cooperative_stop=True) if self.cooperative else caps

    def release_hardware(self) -> None:
        if self.release_gate is not None:
            self.release_gate.wait(10.0)
        super().release_hardware()

    def perceive(self, **kwargs):
        n, self.perceived = self.perceived, self.perceived + 1
        if n in self.perceive_raises:
            self.calls.append("perceive:raised")
            raise self.perceive_raises[n]
        return super().perceive(**kwargs)

    def execute(self, plan_handle, leg, *, save_dir: Path, **kwargs):
        self.execute_kwargs.append(dict(kwargs))
        n, self.executed = self.executed, self.executed + 1
        if self.on_execute is not None:
            self.on_execute(leg)
        should_stop = kwargs.get("should_stop")
        stopped = False
        if self.cooperative and should_stop is not None:
            # A long leg that looks at should_stop at every step boundary, as the protocol asks.
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if should_stop():
                    stopped = True
                    break
                time.sleep(0.01)
        result = super().execute(plan_handle, leg, save_dir=save_dir, should_stop=should_stop)
        if self.write:
            write_leg(save_dir, leg, n_frames=self.n_frames, source="tamp")
        if n in self.execute_raises:
            raise self.execute_raises[n]
        if stopped:
            return replace(result, ok=False, stopped_early=True, failure_reason="stopped when asked")
        return result


# --------------------------------------------------------------------------- the loop on its own


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
    """The person at the prompts. Takes every human phase through the executor unless told otherwise.

    ``pending`` is a hand-off request waiting to be taken, as the session's flag is. The loop going
    round without end fails the test at the 30th prompt instead of hanging the suite.
    """

    def __init__(self, *answers: str, preempt: Exception | None = None) -> None:
        self.answers = list(answers)
        self.pending = False
        self.prompts = 0
        self.preempt = preempt
        self.shown: list = []
        self.unrepresented: list = []
        self.on_handing_off = None
        self.stop_requested = False

    def check_preempt(self) -> None:
        if self.preempt is not None:
            raise self.preempt

    def leg_should_stop(self):
        return lambda: self.stop_requested

    def take_handoff_request(self) -> bool:
        taken, self.pending = self.pending, False
        return taken

    def rolling(self) -> None:
        return None

    def show_progress(self, progress) -> None:
        return None

    def show_unrepresented(self, clauses) -> None:
        self.unrepresented = clauses

    def show_human_phase(self, phase) -> None:
        if phase is not None:
            self.shown.append(phase)

    def await_human_phase(self) -> str:
        self.prompts += 1
        if self.prompts > 30:
            raise AssertionError("the operator was asked 30 times; the loop is not moving on")
        if self.pending:
            self.pending = False
            return "teleop"
        return self.answers.pop(0) if self.answers else "teleop"

    def rollout_started(self, save_dir) -> None:
        return None

    def rollout_saved(self, n_frames: int) -> None:
        return None

    def handing_off(self) -> None:
        if self.on_handing_off is not None:
            self.on_handing_off()

    def arm_lent(self):
        return lambda: True

    def arm_returned(self) -> None:
        return None


@pytest.fixture
def rig(profile, tmp_path, monkeypatch):
    """A phase loop over the stand-ins above. Nothing runs until `.run()`."""
    from tandem.planning import llm

    def make(*answers, plan=PLAN, verdicts=(HOLDS,), executor=None, backend_kwargs=None, record=True, **cfg):
        client = FakeGemini(json.dumps(plan), list(verdicts))
        monkeypatch.setattr(llm, "gemini_client", lambda: client)
        backend = Hooked(None, output_dir=tmp_path / "frames", **{"labels": LABELS, **(backend_kwargs or {})})
        hands = executor or FakeExecutor()
        use_fake_executor(monkeypatch, hands)
        sink, operator = Sink(), Operator(*answers)
        loop = PhaseLoop(
            backend,
            backend.capabilities(),
            PlanningConfig(**{"enabled": True, "save_vlm_io": False, **cfg}),
            events=sink,
            operator=operator,
            executor_context=ExecutorContext(profile=profile, session_dir=tmp_path / "session"),
            legs=LegDirs(profile, tmp_path / "session", log=sink.log),
            record=record,
        )

        def run():
            return loop.run(task=TASK, instruction=TASK, trajectory_id="t" * 16)

        return SimpleNamespace(
            loop=loop, backend=backend, sink=sink, operator=operator, hands=hands, client=client, run=run
        )

    return make


def ran(backend) -> list:
    return [leg["leg"].phase_index for leg in backend.legs]


class Crashing(FakeExecutor):
    """A human executor whose policy server died in the middle of the leg."""

    def run(self, request, leg, *, save_root, should_stop):
        self.calls.append(request)
        raise RuntimeError("policy server died")


# --- a planner or an executor that raises -------------------------------------------------------


def test_an_executor_that_raises_ends_the_trial_at_human_policy_on_the_record(rig):
    r = rig(executor=Crashing())
    with pytest.raises(RuntimeError, match="policy server died"):
        r.run()

    outcome = r.loop.outcome
    assert (outcome.outcome, outcome.failure_stage) == ("failure", "human_policy")
    assert "policy server died" in outcome.reason
    assert outcome.plan.to_json()["failure_stage"] == "human_policy"
    assert len(r.sink.named("trial_outcome")) == 1
    # The arm is still taken back after an executor that raised.
    assert "reacquire_hardware" in r.backend.calls


def test_an_execute_that_raises_is_a_tamp_execution_failure_and_its_leg_is_still_counted(rig):
    """The arm moved and recorded before the planner died: its leg, stamped, is on disk."""
    r = rig(backend_kwargs={"execute_raises": {1: BackendError("the sidecar died mid-leg")}})
    with pytest.raises(BackendError):
        r.run()
    outcome = r.loop.outcome
    assert (outcome.outcome, outcome.failure_stage) == ("failure", "tamp_execution")
    assert "the sidecar died mid-leg" in outcome.reason
    assert outcome.legs_recorded == 3, "the robot's first leg, the person's, and the one that crashed"


def test_a_perceive_that_raises_is_a_tamp_planning_failure(rig):
    r = rig(backend_kwargs={"perceive_raises": {1: BackendError("the camera went away")}})
    with pytest.raises(BackendError):
        r.run()
    assert (r.loop.outcome.outcome, r.loop.outcome.failure_stage) == ("failure", "tamp_planning")


def test_a_preempt_still_unwinds_with_the_outcome_left_to_whoever_catches_it(rig):
    class Preempted(Exception):
        pass

    r = rig()
    r.operator.preempt = Preempted()
    with pytest.raises(Preempted):
        r.run()
    assert r.loop.outcome.outcome is None, "the loop recorded an interrupt it cannot see the cause of"
    r.loop.interrupted("preempted by the operator")
    assert (r.loop.outcome.outcome, r.loop.outcome.reason) == ("aborted", "preempted by the operator")

    # One that lands once the plan has run to the end interrupted nothing: the label still decides.
    r = rig()
    outcome = r.run()
    r.loop.interrupted("preempted by the operator")
    assert outcome.plan.finished and outcome.outcome is None


# --- a hand-off asked for as a person's step comes up --------------------------------------------


def test_a_handoff_asked_for_during_the_robot_leg_before_a_persons_step_is_that_steps_leg(rig):
    """It was taken as an unstamped hand-off, and the step then needed a second, stamped, leg."""
    r = rig(allow_unrecorded_human_phase=False)
    r.backend.on_execute = lambda leg: setattr(r.operator, "pending", True) if leg.phase_index == 0 else None
    outcome = r.run()

    assert [request.description for request in r.hands.calls] == ["open the box"]
    (leg,) = r.hands.legs
    assert (leg.phase_index, leg.n_phases) == (1, 3)
    assert outcome.legs_recorded == 3
    assert not r.sink.named("human_phase_refused")
    assert r.operator.prompts == 0, "the request was the answer; nobody had to be asked"


def test_a_handoff_pending_as_a_plan_opens_with_a_persons_step_is_that_steps_leg(rig):
    r = rig(plan=proposal(OPEN_THE_BOX, PUT_IN), allow_unrecorded_human_phase=False)
    r.operator.pending = True
    r.run()

    (leg,) = r.hands.legs
    assert (leg.phase_index, leg.n_phases) == (0, 2)


# --- a forced stop while the arm is being released ------------------------------------------------


def test_a_kill_while_the_arm_is_being_released_never_starts_the_leg(rig):
    """It reached nothing then -- no leg was running -- so the leg started, came back "done", and
    was verified as a step carried out."""
    r = rig()
    r.operator.on_handing_off = r.loop.kill
    outcome = r.run()

    assert r.hands.calls == [], "the executor was started after a forced stop"
    (ended,) = r.sink.named("human_leg_ended")
    assert ended["status"] == "aborted"
    assert not r.sink.named("human_phase_verified")
    assert (outcome.outcome, outcome.failure_stage) == ("failure", "human_policy")
    assert "reacquire_hardware" in r.backend.calls


def test_the_teleop_executor_killed_before_it_runs_does_not_start_the_driver(profile, tmp_path, monkeypatch):
    from tandem.core.settings import Settings
    from tandem.executors import teleop as teleop_mod
    from tandem.planners.base import LegSpec

    settings = Settings()
    settings.teleop.enabled = True
    started: list = []
    monkeypatch.setattr(teleop_mod.TeleopChild, "start", lambda self: started.append(self) or self)
    executor = teleop_mod.TeleopExecutor(
        ExecutorContext(profile=profile, session_dir=tmp_path / "session", settings=settings)
    )
    leg = LegSpec(trajectory_id="traj-1", instruction=TASK, segment_source="teleop")

    executor.kill()
    # Bounded, so a driver that was started anyway (the kill thrown away on entry) ends the leg
    # "done" after a moment rather than holding the arm for good.
    deadline = time.monotonic() + 1.0
    result = executor.run(None, leg, save_root=tmp_path, should_stop=lambda: time.monotonic() > deadline)
    assert result.status == "aborted"
    assert started == [], "the driver was launched after the leg was killed"
    # The kill ended exactly that leg: the next one runs.
    executor.run(None, leg, save_root=tmp_path, should_stop=lambda: True)
    assert len(started) == 1


# --- a planner that stops when asked --------------------------------------------------------------


def test_should_stop_is_passed_only_to_a_planner_that_declares_cooperative_stop(rig):
    r = rig()
    r.run()
    assert all("should_stop" not in kwargs for kwargs in r.backend.execute_kwargs)

    r = rig(backend_kwargs={"cooperative": True})
    r.operator.stop_requested = False
    r.run()
    assert all(callable(kwargs.get("should_stop")) for kwargs in r.backend.execute_kwargs)


def test_a_leg_stopped_part_way_never_advances_the_plan(rig):
    """Asked to stop and not unwound by it (a planner that stopped by itself), it is still not done."""
    r = rig(backend_kwargs={"cooperative": True})
    r.operator.stop_requested = True
    outcome = r.run()
    assert ran(r.backend) == [0]
    assert outcome.plan.index == 0, "a leg stopped part-way advanced the plan"
    assert outcome.failure_stage == "tamp_execution"


# --- a person's step no camera can settle ---------------------------------------------------------


HOLD_IT = {
    "executor": "human",
    "description": "hand the toy to the robot",
    "instructions": "Put the blue_toy in the gripper.",
    "atoms": [{"predicate": "Holding", "args": ["blue_toy"]}],
    "operator": {
        "name": "HandOver",
        "args": ["blue_toy"],
        "preconditions": [],
        "add_effects": [{"predicate": "Holding", "args": ["blue_toy"]}],
        "delete_effects": [{"predicate": "HandEmpty", "args": []}],
    },
}


def test_a_persons_step_with_nothing_a_camera_can_settle_is_recorded_unchecked_not_passed(rig):
    """TiPToP declares only On checkable: Holding and HandEmpty are the robot's to know. The step used to
    be put to the camera anyway, and came back "ok" with no verdicts and nothing on the record."""
    r = rig(plan=proposal(TAKE_OFF, HOLD_IT, PUT_IN), verdicts=())
    outcome = r.run()

    assert not any(call.startswith("capture_frame") for call in r.backend.calls), "a frame was taken"
    (verified,) = r.sink.named("human_phase_verified")
    assert verified["ok"] is None and "camera can settle" in verified["skipped"]
    assert outcome.plan.checks()["unchecked_phases"] == [1]
    assert "camera can settle" in outcome.plan.phase_record(1)["unchecked"]
    assert r.operator.shown[-1].verified is not True
    assert outcome.plan.finished


# --- a planner with no camera frame ---------------------------------------------------------------


def test_a_planner_that_can_never_capture_a_frame_does_not_have_its_steps_accepted_unchecked(rig, monkeypatch):
    from tandem.planners.sdk import UnsupportedVerb

    r = rig(verdicts=(DOES_NOT_HOLD,))

    def cannot(*, camera="external"):
        raise UnsupportedVerb("The toy planner cannot capture a camera frame.", hint="Implement it.")

    monkeypatch.setattr(r.backend, "capture_frame", cannot)
    with pytest.raises(UnsupportedVerb):
        r.run()
    outcome = r.loop.outcome
    assert (outcome.outcome, outcome.failure_stage) == ("excluded", "verification")
    assert "cannot be verified" in outcome.reason
    assert not outcome.plan.unchecked, "a missing verb was recorded as a check that could not run"


# --- the parts of the instruction the plan leaves out ---------------------------------------------


DROPPED = {"clause": "pick another toy", "reason": "only one toy was detected"}
WITH_A_CLAUSE_DROPPED = proposal(
    TAKE_OFF,
    OPEN_THE_BOX,
    PUT_IN,
    coverage=[
        {"clause": "take the toy off", "phase": 0},
        {"clause": "open the box", "phase": 1},
        {"clause": "put the toy in", "phase": 2},
        {"clause": "pick another toy", "phase": -1},
    ],
    unrepresented=[DROPPED],
)


def test_a_clause_the_plan_leaves_out_is_said_loudly(rig):
    r = rig(plan=WITH_A_CLAUSE_DROPPED)
    r.run()

    assert r.operator.unrepresented == [DROPPED]
    assert ("instruction_not_fully_represented", {"unrepresented": [DROPPED]}) in r.sink.events
    assert any("NOT part of the plan" in line and "pick another toy" in line for line in r.sink.lines)


# --------------------------------------------------------------------------- through a session


@pytest.fixture
def session_for(profile, tmp_path, monkeypatch):
    """A live session with phase planning on, the hooked backend, a canned model, and a stand-in
    human executor that holds the arm until control is returned."""
    from tandem.planning import llm

    made: list[Session] = []

    def build(
        *verdicts,
        plan=PLAN,
        record=None,
        executor_kwargs=None,
        executor=None,
        backend_kwargs=None,
        max_episodes=None,
        **hitl,
    ):
        profile.hitl.enabled = True
        for key, value in hitl.items():
            setattr(profile.hitl, key, value)
        monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
        client = FakeGemini(json.dumps(plan), list(verdicts) or [HOLDS])
        monkeypatch.setattr(llm, "gemini_client", lambda: client)
        backends = use_fake_backend(
            monkeypatch, backend_type=Hooked, **{"labels": LABELS, **(backend_kwargs or {})}
        )
        if executor is not None:
            executors = use_fake_executor(monkeypatch, executor)
        else:
            executors = use_fake_executor(monkeypatch, **{"wait": True, **(executor_kwargs or {})})
        session = Session(profile, task=TASK, record=record, max_episodes=max_episodes)
        states: list[str] = []
        session.subscribe(lambda m: states.append(m["state"]) if m.get("type") == "state" else None)
        session.start()
        assert wait_for(lambda: session.state is State.AWAITING_TASK), f"stuck in {session.state}"
        made.append(session)
        return SimpleNamespace(
            session=session, backends=backends, executors=executors, client=client, states=states
        )

    yield build
    for session in made:
        if session.alive:
            session.stop(park=False)
            session.wait(timeout=10)


def session_events(session) -> list:
    return events_mod.read_all(Path(session.summary()["events_file"]))


def filed_records(profile, status: str) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(profile.status_dir(status).glob("*/hitl.json"))]


def _in_the_persons_leg(s) -> None:
    session = s.session
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_HUMAN_PHASE), f"stuck in {session.state}"
    session.request_teleop()
    assert wait_for(lambda: s.executors and s.executors[0].running.is_set()), "the leg never started"


# --- a stop is a step boundary --------------------------------------------------------------------


def test_a_stop_while_a_person_holds_the_arm_runs_nothing_after_it_and_files_the_trial(session_for, profile):
    """Stop pressed while a person held the arm used to end the leg as "done", then check it, perceive,
    plan and EXECUTE the next robot leg -- seconds after the person let go -- and file nothing."""
    s = session_for()
    _in_the_persons_leg(s)
    backend = s.backends[-1]
    mark = len(backend.calls)

    s.session.stop(park=False)
    assert wait_for(lambda: not s.session.alive, timeout=10), f"stuck in {s.session.state}"

    after = [call.split(":")[0] for call in backend.calls[mark:]]
    assert not {"perceive", "plan", "execute", "capture_frame"} & set(after), after
    assert "reacquire_hardware" in after
    assert "human_phase_verified" not in [event.name for event in session_events(s.session)]

    (record,) = filed_records(profile, "failure")
    assert (record["outcome"], record["filed_under"]) == ("aborted", "failure")
    assert "stopped" in record["outcome_reason"]
    assert not list(profile.status_dir("eval").glob("*/_meta.json")), "a leg was left unfiled in eval/"
    assert (s.session.aborted_count, s.session.labeled_count) == (1, 0)


def test_a_stop_during_a_robot_leg_runs_no_later_leg(session_for, profile):
    s = session_for(plan=TWO_ROBOT, conjoin_robot_phases=False)
    backend = s.backends[-1]
    backend.on_execute = lambda leg: s.session.stop(park=False) if leg.phase_index == 0 else None
    s.session.next_task()
    assert wait_for(lambda: not s.session.alive, timeout=10), f"stuck in {s.session.state}"

    assert "execute:1" not in backend.calls
    assert sum(1 for call in backend.calls if call.startswith("perceive")) == 1
    (record,) = filed_records(profile, "failure")
    assert record["outcome"] == "aborted"


def test_a_stop_at_the_label_prompt_keeps_the_phase_record_with_the_unmerged_legs(
    session_for, profile, monkeypatch
):
    """The plan exists only in memory; returning without it lost the record for good, and a later
    `tandem traj merge` could join the legs but never recreate their verdicts."""

    def leg_video_frames(leg, cameras, tools_dir):
        with np.load(leg["dir"] / "robot_state.npz") as store:
            n = len(store["frame_time"])
        return n, {cam: n for cam in cameras}

    def concat_videos(legs, camera, leg_frames, dest, scratch, tools_dir):
        dest.write_bytes(b"joined")
        return sum(leg_frames)

    monkeypatch.setattr(merge_mod, "_leg_video_frames", leg_video_frames)
    monkeypatch.setattr(merge_mod, "_concat_videos", concat_videos)

    s = session_for(
        backend_kwargs={"write": True}, executor_kwargs={"write": True, "n_frames": 12}, save_vlm_io=True
    )
    _in_the_persons_leg(s)
    s.session.resume_from_teleop()
    assert wait_for(lambda: s.session.state is State.AWAITING_LABEL), f"stuck in {s.session.state}"
    trajectory_id = s.backends[-1].legs[0]["leg"].trajectory_id

    s.session.stop(park=False)
    assert wait_for(lambda: not s.session.alive, timeout=10)

    (kept,) = sorted(profile.status_dir("eval").glob("*/hitl.json"))
    record = json.loads(kept.read_text())
    assert record["filed_under"] is None and record["outcome"] is None
    assert len(record["phases"]) == 3 and record["verifications"]
    assert (kept.parent / "vlm").is_dir(), "the model's audit trail was left in the session scratch dir"
    assert s.session.labeled_count == 0
    assert "trial_unlabeled" in [event.name for event in session_events(s.session)]

    merged = merge_mod.merge(profile, trajectory_id, status="success")
    assert (Path(merged["dir"]) / "hitl.json").is_file(), "the merge dropped the record it was left"
    # And filing it that way is its label: the record says where it now is.
    filed = json.loads((Path(merged["dir"]) / "hitl.json").read_text())
    assert (filed["filed_under"], filed["outcome"]) == ("success", "success")


# --- an unfinished plan is never a success --------------------------------------------------------


def test_a_step_given_up_on_is_filed_as_aborted_without_a_label(session_for, profile):
    s = session_for()
    s.session.next_task()
    assert wait_for(lambda: s.session.state is State.AWAITING_HUMAN_PHASE)
    s.session.abort_human_phase()
    assert wait_for(lambda: filed_records(profile, "failure")), "nothing was filed"
    assert "awaiting_label" not in s.states

    (record,) = filed_records(profile, "failure")
    assert (record["outcome"], record["filed_under"], record["excluded"]) == ("aborted", "failure", False)
    assert s.session.success_count == 0 and s.session.aborted_count == 1


def test_a_preempt_part_way_is_filed_as_aborted_without_a_label(session_for, profile):
    """A preempt at phase 1 of 3 used to leave no outcome at all: labeled success, the record read
    exactly like a finished one."""
    s = session_for()
    s.session.next_task()
    assert wait_for(lambda: s.session.state is State.AWAITING_HUMAN_PHASE)
    s.session.preempt()
    assert wait_for(lambda: filed_records(profile, "failure")), "nothing was filed"
    assert wait_for(lambda: s.session.state is State.AWAITING_TASK)
    assert "awaiting_label" not in s.states

    (record,) = filed_records(profile, "failure")
    assert record["outcome"] == "aborted" and record["outcome_reason"] == "preempted by the operator"
    assert record["phase_index"] == 1


def test_a_preempt_that_lands_after_the_last_leg_leaves_the_trial_to_its_label(session_for, profile):
    s = session_for()
    backend = s.backends[-1]
    backend.on_execute = lambda leg: s.session.preempt() if leg.phase_index == 2 else None
    _in_the_persons_leg(s)
    s.session.resume_from_teleop()
    assert wait_for(lambda: s.session.state is State.AWAITING_LABEL), f"stuck in {s.session.state}"
    s.session.label(True)
    assert wait_for(lambda: filed_records(profile, "success"))
    (record,) = filed_records(profile, "success")
    assert (record["outcome"], record["failure_stage"]) == ("success", None)


def test_an_executor_that_raises_is_said_and_filed_before_anyone_is_asked(session_for, profile):
    """It reached the label prompt with the error still in flight: "did that work?" before the
    operator was told it failed, and a "success" answer wrote failure_stage null."""
    s = session_for(executor=Crashing())
    s.session.next_task()
    assert wait_for(lambda: s.session.state is State.AWAITING_HUMAN_PHASE)
    s.session.request_teleop()
    assert wait_for(lambda: filed_records(profile, "failure")), f"stuck in {s.session.state}"
    assert "awaiting_label" not in s.states

    assert s.session.last_trial["failure_stage"] == "human_policy"
    (record,) = filed_records(profile, "failure")
    assert (record["outcome"], record["failure_stage"]) == ("failure", "human_policy")
    lines = [line["text"] for line in s.session.logs()]
    failed = next(i for i, line in enumerate(lines) if "the task attempt failed" in line)
    filed = next(i for i, line in enumerate(lines) if "filed under failure/ without one" in line)
    assert failed < filed, "the operator was told how the attempt ended after it was filed"


def test_a_preempt_stops_a_cooperative_planner_part_way_and_the_leg_is_not_a_failure(session_for, profile):
    s = session_for(backend_kwargs={"cooperative": True})
    backend = s.backends[-1]
    started = threading.Event()
    backend.on_execute = lambda leg: started.set()
    s.session.next_task()
    assert wait_for(started.is_set)
    began = time.monotonic()
    s.session.preempt()
    assert wait_for(lambda: filed_records(profile, "failure")), f"stuck in {s.session.state}"
    assert time.monotonic() - began < 2.5, "the leg ran to the end of its segment"

    assert backend.executed == 1
    (record,) = filed_records(profile, "failure")
    assert (record["outcome"], record["failure_stage"]) == ("aborted", None)
    assert record["phase_index"] == 0, "a leg stopped part-way advanced the plan"
    assert s.session.alive and wait_for(lambda: s.session.state is State.AWAITING_TASK)


# --- a forced stop mid-release --------------------------------------------------------------------


def test_a_forced_stop_while_the_arm_is_being_released_never_starts_the_leg(session_for):
    gate = threading.Event()
    s = session_for(backend_kwargs={"release_gate": gate})
    s.session.next_task()
    assert wait_for(lambda: s.session.state is State.AWAITING_HUMAN_PHASE)
    s.session.request_teleop()
    assert wait_for(lambda: s.session.state is State.HANDING_OFF)
    s.session.force_stop()
    gate.set()
    assert wait_for(lambda: not s.session.alive, timeout=10), f"stuck in {s.session.state}"

    assert not s.executors or s.executors[0].calls == [], "the executor was started after a forced stop"
    assert s.session.last_trial["failure_stage"] == "human_policy"


# --- the executors are closed, the merges waited for ----------------------------------------------


class Closing(FakeExecutor):
    """A stand-in executor that started something when it was built, and says when it is closed."""

    def __init__(self, backend_of, closed: list, *, fails: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self._backend_of = backend_of
        self.closed = closed
        self.fails = fails

    def close(self) -> None:
        self.closed.append(("close", self._backend_of().closed))
        if self.fails:
            raise RuntimeError("the policy server would not shut down")


@pytest.mark.parametrize("how", ["stop", "force_stop", "custody"])
def test_the_executor_is_closed_once_on_every_way_the_session_ends(session_for, how):
    closed: list = []
    holder: dict = {}
    # A driver that will not let go does so as soon as it starts; the others hold the arm until told.
    executor = Closing(lambda: holder["backend"], closed, wait=how != "custody", custody=how == "custody")
    s = session_for(executor=executor)
    holder["backend"] = s.backends[-1]
    _in_the_persons_leg(s) if how != "custody" else _start_a_custody_failure(s)

    if how == "stop":
        s.session.stop(park=False)
    elif how == "force_stop":
        s.session.force_stop()
    assert wait_for(lambda: not s.session.alive, timeout=10), f"stuck in {s.session.state}"
    assert closed == [("close", True)], "not closed exactly once, after the planner let go"


def _start_a_custody_failure(s) -> None:
    s.session.next_task()
    assert wait_for(lambda: s.session.state is State.AWAITING_HUMAN_PHASE)
    s.session.request_teleop()
    assert wait_for(lambda: s.session.state is State.FAILED, timeout=10), f"stuck in {s.session.state}"


def test_an_executor_whose_close_fails_is_logged_and_the_session_still_ends(session_for):
    closed: list = []
    holder: dict = {}
    s = session_for(executor=Closing(lambda: holder["backend"], closed, fails=True, wait=True))
    holder["backend"] = s.backends[-1]
    _in_the_persons_leg(s)
    s.session.stop(park=False)
    assert wait_for(lambda: not s.session.alive, timeout=10)
    assert closed
    assert any("would not shut down" in line["text"] for line in s.session.logs())
    assert "session_end" in [event.name for event in session_events(s.session)]


def test_an_executor_refused_at_build_is_closed_before_the_refusal(profile, tmp_path, monkeypatch):
    from tandem import executors
    from tandem.executors.base import ExecutorFactory

    closed: list = []

    class Liar(FakeExecutor):
        def close(self):
            closed.append(True)

    monkeypatch.setattr(executors.base, "_registered", dict(executors.base._registered))
    factory = ExecutorFactory(
        create=lambda ctx: Liar(ctx, segment_source="policy"),
        display_name="liar",
        summary="says teleop, records policy",
        segment_source="teleop",
    )
    executors.register_human_executor("liar", factory)
    with pytest.raises(TandemError, match="declares its legs as 'teleop'"):
        executors.create("liar", ExecutorContext(profile=profile, session_dir=tmp_path))
    assert closed == [True]


def test_a_session_ending_waits_for_its_merge_and_the_record_is_on_disk_before_it(
    session_for, profile, monkeypatch
):
    """`tandem collect --episodes 1` exits as soon as the session ends, and killed the merge that the
    only copy of the phase record waited on."""
    release = threading.Event()
    real = merge_mod.merge

    def slow_merge(*args, **kwargs):
        release.wait(10.0)
        return real(*args, **kwargs)

    monkeypatch.setattr(merge_mod, "merge", slow_merge)
    s = session_for(max_episodes=1)
    _in_the_persons_leg(s)
    s.session.resume_from_teleop()
    assert wait_for(lambda: s.session.state is State.AWAITING_LABEL)
    s.session.label(True)

    assert wait_for(lambda: filed_records(profile, "success")), "the record waited for the merge"
    time.sleep(0.3)
    assert s.session.alive, "the session ended with its merge still running"
    release.set()
    assert wait_for(lambda: not s.session.alive, timeout=10)


# --- a planner that cannot capture a frame, at start ----------------------------------------------


def _toy_session(profile, monkeypatch, planner_cls, **hitl):
    from toy_planner import TOY_CAPABILITIES  # noqa: F401  (the toy's own declaration)

    from tandem.core.profiles import PlannerSpec
    from tandem.planners import registry

    isolate_registry(monkeypatch)
    registry.register_backend("toy", planner_cls)
    profile.planner = PlannerSpec(backend="toy", options={})
    profile.hitl.enabled = True
    for key, value in hitl.items():
        setattr(profile.hitl, key, value)
    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
    session = Session(profile, task=TASK)
    session.start()
    return session


@pytest.fixture
def blind_toy():
    from toy_planner import ToyPlanner

    from tandem.planners.sdk import Planner

    class BlindToy(ToyPlanner):
        """The toy with the SDK's default capture_frame, as a scaffolded planner ships."""

        capture_frame = Planner.capture_frame

    return BlindToy


def test_a_planner_without_capture_frame_is_refused_when_the_checks_need_one(profile, monkeypatch, blind_toy):
    session = _toy_session(profile, monkeypatch, blind_toy)
    try:
        assert wait_for(lambda: session.state is State.FAILED), f"stuck in {session.state}"
        assert "capture a camera frame" in session.error and "hitl.check_human_effects" in session.error
    finally:
        session.stop(park=False)
        session.wait(timeout=5)


def test_a_planner_without_capture_frame_runs_with_the_camera_checks_off(profile, monkeypatch, blind_toy):
    session = _toy_session(
        profile,
        monkeypatch,
        blind_toy,
        check_human_effects=False,
        check_human_preconditions=False,
        check_tamp_effects=False,
    )
    try:
        assert wait_for(lambda: session.state is State.AWAITING_TASK), f"stuck in {session.state}"
    finally:
        session.stop(park=False)
        session.wait(timeout=5)


def test_a_sidecar_that_does_not_answer_capture_frame_cannot_capture_one():
    from toy_planner import ToyPlanner, ToySidecarPlanner

    from tandem.core.session import _can_capture_frame

    class Listed(ToySidecarPlanner):
        verbs: frozenset | None = None

        @property
        def sidecar_verbs(self):
            return self.verbs

    sidecar = Listed()
    sidecar.verbs = frozenset({"warm", "perceive", "plan", "execute"})
    assert not _can_capture_frame(sidecar)
    sidecar.verbs = frozenset({"warm", "perceive", "plan", "execute", "capture_frame"})
    assert _can_capture_frame(sidecar)
    sidecar.verbs = None  # a sidecar that lists nothing is taken to answer the whole protocol
    assert _can_capture_frame(sidecar)
    assert _can_capture_frame(ToyPlanner())
    assert _can_capture_frame(FakeBackend())


# --- the record of a replanned trial --------------------------------------------------------------


def test_the_record_of_a_replanned_trial_keeps_the_plan_it_replaced(profile):
    """Straight through the episode writer, from a loop outcome, without a session."""
    from tandem.core.phase_loop import TrialOutcome

    superseded = [{"phases": [{"index": 0}], "plan_generation": 0, "superseded_because": "phase 0: no"}]
    outcome = TrialOutcome(trajectory_id="t", superseded_plans=superseded, leg_generations={"leg-a": 0})
    directory = profile.status_dir("failure") / "leg-a"
    directory.mkdir(parents=True)

    class Plan:
        def to_json(self):
            return {"outcome": None, "failure_stage": None, "phases": [{"index": 0}]}

    episodes.write_phase_record(
        Plan(),
        directory,
        status="failure",
        vlm_dir=None,
        log=print,
        superseded=outcome.superseded_plans,
        leg_generations=outcome.leg_generations,
    )
    record = json.loads((directory / "hitl.json").read_text())
    assert record["plan_generation"] == 1
    assert record["superseded_plans"] == superseded
    assert record["leg_plan_generations"] == {"leg-a": 0}


# --- the deprecated runtime argument --------------------------------------------------------------


def test_a_task_passed_where_the_deprecated_runtime_goes_is_refused(profile):
    """`Session(profile, "put the toy in the box")` took the string as the runtime and planned every
    episode with the profile's default task."""
    with pytest.raises(TypeError, match="task="):
        Session(profile, TASK)
    assert Session(profile, task=TASK).task == TASK


def test_the_summary_carries_the_clauses_the_plan_leaves_out(session_for):
    s = session_for(plan=WITH_A_CLAUSE_DROPPED)
    s.session.next_task()
    assert wait_for(lambda: s.session.state is State.AWAITING_HUMAN_PHASE)
    assert s.session.summary()["unrepresented"] == [DROPPED]


def test_a_crashed_executor_leg_result_is_still_a_result():
    # HumanPhaseResult's new field is optional: an executor written before it still constructs one.
    assert HumanPhaseResult("done", 3, Path("x")).leg_dirs == ()
    assert ExecuteResult(ok=True).stopped_early is False
    with pytest.raises(CustodyError):
        raise CustodyError("x")
