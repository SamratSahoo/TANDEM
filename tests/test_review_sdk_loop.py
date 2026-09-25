"""A planner that raises instead of answering: the trial ends there, on the record, and the next one starts.

The likeliest way a real planner fails mid-trial is not ``ok=False``. It is its sidecar crashing (a
segfault in a CUDA kernel, the robot client dying) or wedging until the verb times out -- and both
arrive as an exception. The phase loop used to let that unwind straight out of the trial: no outcome,
no failure stage, no ``trial_outcome`` event, and the session then sent the legs already on disk to the
label prompt as though the trial had simply ended. An operator answering "success" there filed a
demonstration that stops half-way, with ``failure_stage: null``, into the dataset.

Now each of the planner's three verbs is guarded where the loop calls it. A raise ends the trial like
the failure it is -- perception or planning at ``tamp_planning``, execution at ``tamp_execution``,
where the plan never advances -- and the session warms the planner again before the next task, which
for a sidecar that died means a fresh one.

Driven on the toy world (items dropped in bins, two robot phases, each its own leg): in tandem's own
process through the real phase loop, and behind a real sidecar through a real session.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from helpers import isolate_registry, wait_for
from toy_planner import TOY_CAPABILITIES, ToyPlanner, ToySidecarPlanner

from tandem.core import profiles, secrets
from tandem.core.episodes import LegDirs
from tandem.core.phase_loop import PhaseLoop
from tandem.core.profiles import PlannerSpec
from tandem.core.session import Session, State
from tandem.executors.base import ExecutorContext
from tandem.planners import PlannerInfo, registry
from tandem.planners.base import BackendError
from tandem.planning.config import PlanningConfig

CRASH = "the planner backend exited (code 3) without answering. Its stderr is in the session log."


def robot(description: str, item: str, bin_: str) -> dict:
    return {
        "executor": "robot",
        "description": description,
        "atoms": [{"predicate": "InBin", "args": [item, bin_]}],
    }


# Two robot phases. The toy declares no clean initial state, so they are two legs: the second one is
# where the planner dies, with the first already recorded.
PLAN = {
    "new_predicates": [],
    "phases": [
        robot("put the apple in the red bin", "apple", "red_bin"),
        robot("put the pear in the blue bin", "pear", "blue_bin"),
    ],
    "coverage": [
        {"clause": "put the apple in the red bin", "phase": 0},
        {"clause": "put the pear in the blue bin", "phase": 1},
    ],
    "unrepresented": [],
}
TASK = "put the apple in the red bin, then the pear in the blue bin"


class Model:
    """The model: the plan for a proposal, and "yes" to anything the camera is asked."""

    PLAN_MARKER = "ORDERED list of phases"

    def __init__(self) -> None:
        self.plan_prompts: list[str] = []
        self.aio = self

    @property
    def models(self):
        return self

    async def generate_content(self, model, contents, config):
        prompt = contents[-1]
        if "Statement:" in prompt and self.PLAN_MARKER not in prompt:
            return mock.Mock(text=json.dumps({"holds": True, "reason": "as seen"}))
        self.plan_prompts.append(prompt)
        return mock.Mock(text=json.dumps(PLAN))


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
    """Nobody at the prompts: a robot-only plan never asks. A loop that never ends fails the test."""

    def __init__(self) -> None:
        self.boundaries = 0

    def check_preempt(self) -> None:
        self.boundaries += 1
        if self.boundaries > 40:
            raise AssertionError("the loop went round 40 times without ending the trial")

    def take_handoff_request(self) -> bool:
        return False

    def __getattr__(self, name):
        # rolling, show_progress, show_unrepresented, show_human_phase, rollout_started, ...
        return lambda *args, **kwargs: None


class Raising(ToyPlanner):
    """The toy, whose ``raises[verb]``-th call of that verb (from 1) raises what a dead sidecar raises.

    ``stamp_first`` has the failing execute stamp its leg before it goes, as the recording contract
    asks a planner to: that leg is on disk, and belongs to the trial.
    """

    def __init__(self, ctx=None, *, raises: dict | None = None, stamp_first: bool = False) -> None:
        super().__init__(ctx)
        self.raises = dict(raises or {})
        self.stamp_first = stamp_first
        self.called: dict[str, int] = {}

    def _maybe_raise(self, verb: str) -> None:
        self.called[verb] = self.called.get(verb, 0) + 1
        if self.raises.get(verb) == self.called[verb]:
            raise BackendError(CRASH)

    def perceive(self, **kwargs):
        self._maybe_raise("perceive")
        return super().perceive(**kwargs)

    def plan(self, scene_id, goal, **kwargs):
        self._maybe_raise("plan")
        return super().plan(scene_id, goal, **kwargs)

    def execute(self, plan_handle, leg, *, save_dir, should_stop=None):
        if self.stamp_first and self.raises.get("execute") == self.called.get("execute", 0) + 1:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            (Path(save_dir) / "_meta.json").write_text(json.dumps({"trajectory_id": leg.trajectory_id}))
        self._maybe_raise("execute")
        return super().execute(plan_handle, leg, save_dir=save_dir, should_stop=should_stop)


@pytest.fixture
def loop_on(profile, tmp_path, monkeypatch):
    """The real phase loop over a toy planner that raises where it is told to."""
    from tandem.planning import llm

    def make(cfg: dict | None = None, **raising):
        model = Model()
        monkeypatch.setattr(llm, "gemini_client", lambda: model)
        planner = Raising(**raising)
        sink = Sink()
        loop = PhaseLoop(
            planner,
            planner.capabilities(),
            PlanningConfig(**{"enabled": True, "save_vlm_io": False, **(cfg or {})}),
            events=sink,
            operator=Operator(),
            executor_context=ExecutorContext(profile=profile, session_dir=tmp_path / "session"),
            legs=LegDirs(profile, tmp_path / "session", log=sink.log),
        )

        def run():
            return loop.run(task=TASK, instruction=TASK, trajectory_id="c" * 16)

        return SimpleNamespace(planner=planner, sink=sink, loop=loop, run=run)

    return make


def _record(outcome) -> dict:
    return json.loads(json.dumps(outcome.plan.to_json(), default=str))


def test_a_planner_that_dies_mid_execution_ends_the_trial_at_tamp_execution(loop_on):
    r = loop_on(raises={"execute": 2})
    outcome = r.run()  # does not raise

    assert (outcome.outcome, outcome.failure_stage) == ("failure", "tamp_execution")
    assert "put the pear in the blue bin" in outcome.reason and "exited (code 3)" in outcome.reason
    assert outcome.planner_raised
    # The plan stopped where the robot did, and the record says so.
    assert outcome.plan.index == 1
    assert _record(outcome)["failure_stage"] == "tamp_execution"
    assert outcome.legs_recorded == 1, "the first leg is the trial's; the dead one recorded nothing"
    (event,) = r.sink.named("trial_outcome")
    assert (event["outcome"], event["failure_stage"]) == ("failure", "tamp_execution")


def test_a_leg_the_planner_stamped_before_it_died_is_counted_as_the_trials(loop_on):
    # Left uncounted, the stamped leg stays on disk (LegDirs keeps any leg with a _meta.json) and is
    # never filed with the trial's other legs: an orphan episode of its own.
    r = loop_on(raises={"execute": 2}, stamp_first=True)
    outcome = r.run()
    assert outcome.failure_stage == "tamp_execution"
    assert outcome.legs_recorded == 2


@pytest.mark.parametrize(
    ("verb", "complaint"),
    [("plan", "failed while planning 'put the pear in the blue bin'"), ("perceive", "could not perceive")],
)
def test_a_planner_that_dies_while_perceiving_or_planning_ends_the_trial_at_tamp_planning(
    loop_on, verb, complaint
):
    r = loop_on(raises={verb: 2})
    outcome = r.run()
    assert (outcome.outcome, outcome.failure_stage) == ("failure", "tamp_planning")
    assert complaint in outcome.reason and "exited (code 3)" in outcome.reason
    assert outcome.planner_raised and outcome.legs_recorded == 1
    assert _record(outcome)["failure_stage"] == "tamp_planning"
    # Not handed to on_robot_phase_failure: a planner that is not running can neither lend its arm to
    # a person nor perceive for a re-plan.
    assert r.sink.named("phase_plan_failed") == []


@pytest.mark.parametrize("policy", ["replan", "teleop"])
def test_a_planner_that_raises_is_not_offered_to_teleop_or_replan(loop_on, policy):
    r = loop_on(cfg={"on_robot_phase_failure": policy}, raises={"plan": 1})
    outcome = r.run()
    assert (outcome.outcome, outcome.failure_stage) == ("failure", "tamp_planning")
    assert r.planner.called["plan"] == 1, "no re-plan was attempted on a planner that raised"
    assert r.planner.called["perceive"] == 1, "and nothing was perceived through it again"


# --- through a session, on a sidecar that really dies -----------------------------------------------------


class CrashingToy(ToySidecarPlanner):
    """The toy behind a real sidecar, launched with whatever FLAGS say when it is (re)started."""

    info = PlannerInfo(name="toy-crash", display_name="Toy (crashing)", summary="Dies when told to.")
    CAPABILITIES = replace(TOY_CAPABILITIES, name="toy-crash")
    FLAGS: tuple[str, ...] = ()

    def launch_command(self) -> list[str]:
        return [*super().launch_command(), *type(self).FLAGS]


@pytest.fixture
def crashing_session(profile, monkeypatch):
    from tandem.planning import llm

    isolate_registry(monkeypatch)
    registry.register_backend("toy-crash", CrashingToy)
    monkeypatch.setattr(CrashingToy, "FLAGS", ("--crash-on-nth", "execute", "2"))
    profile.planner = PlannerSpec(backend="toy-crash", options={})
    profile.hitl.enabled = True
    profiles.save(profile)
    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
    model = Model()
    monkeypatch.setattr(llm, "gemini_client", lambda: model)
    session = Session(profiles.load(profile.name), task=TASK)
    session.start()
    assert wait_for(lambda: session.state is State.AWAITING_TASK, timeout=30), f"stuck in {session.state}"
    yield session
    if session.alive:
        session.stop(park=False)
        session.wait(timeout=10)


def test_a_sidecar_that_dies_mid_trial_is_filed_as_the_failure_it_is_and_replaced(crashing_session, profile):
    session = crashing_session
    first = session._backend._channel.hello["pid"]
    # Read only when a sidecar is launched, so the running one still dies on its 2nd execute; the
    # relaunch must not die the same way: it is the sidecar the next task gets. Cleared up front,
    # because the relaunch follows the failed attempt at once -- a trial the loop ended itself is
    # filed with no label prompt to wait at.
    CrashingToy.FLAGS = ()
    session.next_task()

    def filed():
        records = sorted(profile.status_dir("failure").glob("*/hitl.json"))
        return json.loads(records[0].read_text()) if records else None

    assert wait_for(lambda: filed() is not None, timeout=30), "no hitl.json under failure/"
    record = filed()
    assert (record["outcome"], record["failure_stage"]) == ("failure", "tamp_execution")
    assert "exited (code 3)" in record["outcome_reason"]
    last = session.summary()["last_trial"]
    assert (last["outcome"], last["failure_stage"]) == ("failure", "tamp_execution")
    assert "exited (code 3)" in last["reason"]

    # Warmed again before the next task: a fresh sidecar, not a session that fails every task from
    # here on with "the planner backend is not running".
    assert wait_for(lambda: session.state is State.AWAITING_TASK, timeout=30), f"stuck in {session.state}"
    fresh = session._backend._channel.hello["pid"]
    assert fresh != first and session._backend._channel.alive
    session.next_task()
    assert wait_for(lambda: session.state is State.AWAITING_LABEL, timeout=30), f"stuck in {session.state}"
    assert session.summary()["last_trial"]["failure_stage"] is None, "the next trial ran to the end"
    assert session._backend._channel.hello["pid"] == fresh
