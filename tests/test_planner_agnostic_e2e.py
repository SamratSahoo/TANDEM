"""TANDEM on a planner it was not written for, end to end: the toy world through the real loop.

The paper's claim is that TANDEM "only requires an interface for specifying subgoals and executing the
resulting plans". Every other part of the method -- segmenting the instruction, inventing predicates
and magic operators, verifying a person's step, merging the legs into one demonstration -- is
tandem's, so it must hold for ANY planner behind that interface, not only for the one it grew up
with. The unit tests show each piece is written against ``Capabilities``; this file shows the pieces
still hold together when nothing TipTop-shaped is anywhere in the run.

The planner is ``tests/toy_planner.py``: items dropped into bins, ``InBin(?obj: item, ?bin:
container)``, no table, no exclusivity, and ``initial_state_is_clean=False`` -- an item can be
re-binned, so two drops are ordered and never one goal. It is registered the way a plugin is
(``register_backend``) and driven through the REAL proposal (the prompt rendered from its
``Capabilities``, the parser, the contract check, the repair loop), the real ``PhasePlan``, the real
``PhaseLoop``, and the real episode writer and merge. Only the model (a scripted camera), the person
(a stand-in executor) and the video tools are stand-ins. The whole trial runs twice: with the toy in
tandem's own process (``ToyPlanner``), and behind a sidecar in a child process
(``ToySidecarPlanner``), which is the hosting a planner with an environment of its own gets.

The scenario is a pick-and-place-shaped task in words that are not cuTAMP's: put one item in the red
bin, have the person close its lid, then put two more in the blue bin. Robot, person, robot, robot --
so the last two robot phases are consecutive and would be conjoined into one goal by a planner that
declared a clean initial state. The toy does not, so each is its own leg.

Whether the planner itself meets the protocol is the conformance kit's question, and
``tests/test_conformance.py`` already runs ``tandem.planners.testing`` against both toy hostings.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from fake_executor import FakeExecutor, use_fake_executor
from helpers import isolate_registry, wait_for
from ruamel.yaml import YAML
from toy_planner import TOY_CAPABILITIES, ToyPlanner, ToySidecarPlanner

from tandem.core import episodes, profiles, secrets, trajectories
from tandem.core import merge as merge_mod
from tandem.core.episodes import LegDirs
from tandem.core.phase_loop import PhaseLoop
from tandem.core.profiles import PlannerSpec
from tandem.core.session import Session, State
from tandem.executors.base import ExecutorContext, HumanPhaseResult
from tandem.planners import GoalAtom, registry
from tandem.planners.base import BackendContext
from tandem.planning.config import PlanningConfig

TRAJECTORY = "7" * 16

# The two predicates the model invents for the person's step, in the words the camera is asked them in.
LID_CLOSED = "the lid of {0} is shut, covering its opening"
LID_OPEN = "the lid of {0} is open, so the inside of {0} is visible"
SHUT = LID_CLOSED.format("red_bin")
AJAR = LID_OPEN.format("red_bin")

# What the camera sees after the person's step: the lid shut, and so no longer open. Add effect true,
# delete effect false -- the only answer that passes.
DONE = {SHUT: True, AJAR: False}
# The person pressed the lid down, but it sprang back: shut by one reading and open by the other.
SPRUNG_BACK = {SHUT: True, AJAR: True}


# --- the scenario, in whichever items the world has --------------------------------------------------


def robot(description: str, item: str, bin_: str) -> dict:
    return {
        "executor": "robot",
        "description": description,
        "atoms": [{"predicate": "InBin", "args": [item, bin_]}],
    }


def close_the_lid(item: str) -> dict:
    """The person's step, as the model states it: the magic operator Close(red_bin).

    It needs the item already in the bin (a goal predicate an earlier ROBOT phase establishes) and the
    lid open; it makes the lid shut and ends it being open. Both halves of the effect are put to the
    camera afterwards.
    """
    return {
        "executor": "human",
        "description": "close the red bin's lid",
        "instructions": f"Close the lid of the red_bin, with the {item} inside it.",
        "atoms": [{"predicate": "LidClosed", "args": ["red_bin"]}],
        "operator": {
            "name": "Close",
            "args": ["red_bin"],
            "preconditions": [
                {"predicate": "InBin", "args": [item, "red_bin"]},
                {"predicate": "LidOpen", "args": ["red_bin"]},
            ],
            "add_effects": [{"predicate": "LidClosed", "args": ["red_bin"]}],
            "delete_effects": [{"predicate": "LidOpen", "args": ["red_bin"]}],
        },
    }


def proposal(*phases: dict) -> dict:
    """The model's answer: these phases, the predicates they invent, and one clause per phase."""
    used = json.dumps(phases)
    invented = {"LidClosed": LID_CLOSED, "LidOpen": LID_OPEN}
    return {
        "new_predicates": [
            {"name": name, "instructions": text} for name, text in invented.items() if f'"{name}"' in used
        ],
        "phases": list(phases),
        "coverage": [{"clause": phase["description"], "phase": i} for i, phase in enumerate(phases)],
        "unrepresented": [],
    }


@dataclass(frozen=True)
class Scene:
    """The scenario in one world's item names.

    ``first`` goes in the red bin and is shut in by the person; ``second`` and ``third`` go in the blue
    bin, one straight after the other. ``bystander`` is on the floor and named by nothing: the object
    ``movables`` exists to keep the planner's hands off.
    """

    first: str
    second: str
    third: str
    bystander: str | None = None

    @property
    def items(self) -> tuple[str, ...]:
        return (self.first, self.second, self.third, *((self.bystander,) if self.bystander else ()))

    @property
    def moved(self) -> frozenset[str]:
        return frozenset({self.first, self.second, self.third})

    @property
    def instruction(self) -> str:
        return (
            f"put the {self.first} in the red bin, close its lid, put the {self.second} in the blue bin, "
            f"then put the {self.third} in the blue bin"
        )

    def phases(self, *, third_into: str = "blue_bin") -> tuple[dict, ...]:
        return (
            robot(f"put the {self.first} in the red bin", self.first, "red_bin"),
            close_the_lid(self.first),
            robot(f"put the {self.second} in the blue bin", self.second, "blue_bin"),
            robot(f"put the {self.third} in the blue bin", self.third, third_into),
        )

    def plan(self, **kwargs) -> dict:
        return proposal(*self.phases(**kwargs))

    def placed(self, where: dict) -> dict:
        """Where the world put the three items, as {item: bin}, to compare with `instructed`."""
        return {item: where[item] for item in (self.first, self.second, self.third)}

    @property
    def instructed(self) -> dict:
        return {self.first: "red_bin", self.second: "blue_bin", self.third: "blue_bin"}


# The toy's items are a planner option in tandem's process, so a bystander can be put on the floor. The
# sidecar builds its world in its own process from the toy's default items.
HOSTINGS = {
    "in-process": ("toy", ToyPlanner, Scene("duck", "ball", "cube", bystander="sponge")),
    "sidecar": ("toy-sidecar", ToySidecarPlanner, Scene("apple", "pear", "plum")),
}


# --- the stand-ins ----------------------------------------------------------------------------------


class Model:
    """The model: a plan for a proposal, and a verdict by STATEMENT for a classifier question.

    ``replan`` is the answer to a proposal that carries the replan section (the planner's failure fed
    back), so a test can tell a re-proposal from the first one by what the model was told. A statement
    no answer covers fails the test: the camera was asked something the plan never said. An answer that
    is a list is what the camera sees each time it is asked, the last one repeating: a lid that is open
    before the person's step and shut after it.
    """

    PLAN_MARKER = "ORDERED list of phases"
    REPLAN_MARKER = "An earlier plan for this task could not be carried out"

    def __init__(self, plan: dict, answers: dict, *, replan: dict | None = None) -> None:
        self.plan = json.dumps(plan)
        self.replan = json.dumps(replan) if replan is not None else None
        self.answers = {k: list(v) if isinstance(v, (list, tuple)) else [v] for k, v in answers.items()}
        self.plan_prompts: list[str] = []
        self.asked: list[str] = []
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
                raise AssertionError(f"the camera was asked about {statement!r}, which no answer covers")
            seen = self.answers[statement]
            holds = seen.pop(0) if len(seen) > 1 else seen[0]
            return mock.Mock(text=json.dumps({"holds": holds, "reason": "as seen"}))
        self.plan_prompts.append(prompt)
        again = self.replan is not None and self.REPLAN_MARKER in prompt
        return mock.Mock(text=self.replan if again else self.plan)


FPS = 15


def record_leg(directory: Path, leg, n_frames: int, start: float) -> Path:
    """A person's leg on disk, recorded the way the toy world records the robot's.

    On the wall clock, which is the point: the merge orders a trajectory's legs by when each was
    recorded, and ``fake_executor.write_leg`` stamps its legs on a clock of its own, years from now --
    right for joining its legs to each other, and wrong beside the toy's, where every person's leg
    would sort after every robot leg.
    """
    directory.mkdir(parents=True, exist_ok=True)
    joints = np.zeros((n_frames, 7))
    np.savez(
        directory / "robot_state.npz",
        joint_position=joints,
        gripper_position=np.zeros(n_frames),
        cmd_joint_position=joints,
        cmd_joint_velocity=joints,
        cmd_gripper=np.zeros(n_frames),
        frame_time=start + np.arange(n_frames) / FPS,
    )
    (directory / "external_cam.mp4").write_bytes(b"a person's clip")
    meta = {
        "trajectory_id": leg.trajectory_id,
        "segment_source": leg.segment_source,
        "instruction": leg.instruction,
        "n_frames": n_frames,
        "fps": FPS,
        "record_start": start,
        "record_stop": time.time(),
        "cameras": {"exterior_image_1_left": "external_cam.mp4"},
    }
    if leg.phase_index is not None:
        meta.update(
            phase_index=leg.phase_index, n_phases=leg.n_phases, phase_description=leg.phase_description
        )
    (directory / "_meta.json").write_text(json.dumps(meta))
    return directory


class Person(FakeExecutor):
    """The human executor: somebody at the arm who carries the step out and whose leg is recorded."""

    def __init__(self, ctx=None, **kwargs) -> None:
        super().__init__(ctx, n_frames=kwargs.pop("n_frames", 12), **kwargs)

    def run(self, request, leg, *, save_root: Path, should_stop) -> HumanPhaseResult:
        start = time.time()
        result = super().run(request, leg, save_root=save_root, should_stop=should_stop)
        if result.status == "aborted" or not result.n_frames or not leg.record:
            return result
        directory = record_leg(
            Path(save_root) / "eval" / f"person-{len(self.calls):02d}", leg, result.n_frames, start
        )
        return HumanPhaseResult(result.status, n_frames=result.n_frames, leg_dir=directory)


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


class Stopped(Exception):
    """The operator's stop, raised at the next step boundary as the session's own is."""


class Operator:
    """The person at the prompts: hands every human phase to the executor. A loop that never ends
    fails the test at the 40th step boundary instead of hanging the suite.

    ``stop_legs`` is a stop the operator asks for while a robot leg runs: the leg's ``should_stop``
    turns true, and the next step boundary unwinds the attempt (`Stopped`)."""

    def __init__(self) -> None:
        self.boundaries = 0
        self.shown: list = []
        self.stop_legs = False
        self.asked_to_stop = False

    def check_preempt(self) -> None:
        self.boundaries += 1
        if self.boundaries > 40:
            raise AssertionError("the loop went round 40 times without ending the trial")
        if self.asked_to_stop:
            raise Stopped()

    def leg_should_stop(self):
        def should_stop() -> bool:
            if self.stop_legs:
                self.asked_to_stop = True
            return self.stop_legs

        return should_stop

    def take_handoff_request(self) -> bool:
        return False

    def rolling(self) -> None:
        return None

    def show_progress(self, progress) -> None:
        return None

    def show_unrepresented(self, clauses) -> None:
        return None

    def show_human_phase(self, phase) -> None:
        if phase is not None:
            self.shown.append(phase)

    def await_human_phase(self) -> str:
        return "teleop"

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


# The verbs the loop reaches the planner through, watched in the order they arrive.
WATCHED = ("perceive", "plan", "execute", "release_hardware", "reacquire_hardware", "capture_frame")


def watch(planner) -> list[SimpleNamespace]:
    """Every call the loop makes of ``planner``, in order, with exactly the arguments it passed.

    Wrapped on the instance, so the planner registered is the toy class itself and what is recorded is
    what crossed the interface: a keyword the loop left out is absent here, not a default.
    """
    calls: list[SimpleNamespace] = []
    for verb in WATCHED:
        real = getattr(planner, verb)

        def spy(*args, _verb=verb, _real=real, **kwargs):
            calls.append(SimpleNamespace(verb=_verb, args=args, kwargs=kwargs))
            return _real(*args, **kwargs)

        setattr(planner, verb, spy)
    return calls


def whereabouts(planner) -> dict:
    """Where every item is in the toy's world, from whichever process it lives in."""
    if isinstance(planner, ToySidecarPlanner):
        return planner.call("where")
    return dict(planner.world.where)


def of(calls: list[SimpleNamespace], verb: str) -> list[SimpleNamespace]:
    return [call for call in calls if call.verb == verb]


@pytest.fixture
def fake_video(monkeypatch):
    """Stand in for ffprobe/ffmpeg: every clip holds as many frames as its leg has states."""

    def leg_video_frames(leg, cameras, tools_dir):
        with np.load(leg["dir"] / "robot_state.npz") as store:
            n = len(store["frame_time"])
        return n, {cam: n for cam in cameras}

    def concat_videos(legs, camera, leg_frames, dest, scratch, tools_dir):
        dest.write_bytes(b"joined")
        return sum(leg_frames)

    monkeypatch.setattr(merge_mod, "_leg_video_frames", leg_video_frames)
    monkeypatch.setattr(merge_mod, "_concat_videos", concat_videos)


@pytest.fixture
def trial(profile, tmp_path, monkeypatch):
    """The real phase loop over a toy planner built by the registry. Nothing runs until `.run()`.

    ``caps`` stands in for the planner's own declaration where a test varies it, to show that what the
    loop does follows the declaration and nothing else.
    """
    from tandem.planning import llm

    isolate_registry(monkeypatch)
    built: list = []

    def make(answers=DONE, *, hosting="in-process", plan=None, replan=None, caps=None, planner_cls=None, **cfg):
        name, cls, scene = HOSTINGS[hosting]
        cls = planner_cls or cls
        registry.register_backend(name, cls)
        planner = registry.create(
            name,
            BackendContext(
                profile=profile,
                session_dir=tmp_path / "session",
                output_dir=profile.trajectories_dir(),
                options={"items": list(scene.items)} if cls is ToyPlanner else {},
            ),
        )
        built.append(planner)
        planner.warm()
        calls = watch(planner)
        model = Model(plan or scene.plan(), answers, replan=replan)
        monkeypatch.setattr(llm, "gemini_client", lambda: model)
        person = Person()
        use_fake_executor(monkeypatch, person)
        sink, operator = Sink(), Operator()
        loop = PhaseLoop(
            planner,
            caps or planner.capabilities(),
            PlanningConfig(**{"enabled": True, "save_vlm_io": False, **cfg}),
            events=sink,
            operator=operator,
            executor_context=ExecutorContext(profile=profile, session_dir=tmp_path / "session"),
            legs=LegDirs(profile, tmp_path / "session", log=sink.log),
        )

        def run():
            return loop.run(task=scene.instruction, instruction=scene.instruction, trajectory_id=TRAJECTORY)

        return SimpleNamespace(
            scene=scene,
            planner=planner,
            calls=calls,
            model=model,
            person=person,
            sink=sink,
            operator=operator,
            loop=loop,
            run=run,
        )

    yield make
    for planner in built:
        planner.close()


def merged_under(profile, status: str) -> bool:
    """Whether the trial's merged episode is filed under ``status`` yet.

    Its hitl.json is not enough to go by: the record is written into the leg the trial is filed under
    before the merge starts, so that a merge that never finishes still leaves it.
    """
    for directory in profile.status_dir(status).glob("*"):
        try:
            if (directory / "hitl.json").is_file() and trajectories.read_meta(directory).get("video_aligned"):
                return True
        except OSError:
            continue
    return False


def merged_episode(profile, status: str) -> tuple[Path, dict, dict]:
    """The one merged episode filed under ``status``: its directory, its _meta.json and its hitl.json."""
    (directory,) = [d for d in profile.status_dir(status).iterdir() if (d / "hitl.json").is_file()]
    meta = json.loads((directory / "_meta.json").read_text())
    return directory, meta, json.loads((directory / "hitl.json").read_text())


# --- the proposal, in the planner's own language ------------------------------------------------------


def test_the_segmentation_prompt_is_the_toys_language_and_nothing_else(trial):
    """The prompt is rendered from the toy's ``Capabilities``: its one predicate, its one action, its
    item names. A word of cuTAMP's in it would ask the model for a goal this planner cannot take."""
    r = trial()
    r.run()

    (prompt,) = r.model.plan_prompts
    for tiptop in ("On(", "Holding", "HandEmpty"):
        assert tiptop not in prompt, f"the toy's prompt asks for {tiptop}"
    assert "- InBin(?obj: item, ?bin: container): {0} is inside {1}" in prompt
    assert "The robot can do one thing: drop an item into a bin." in prompt
    assert "A ROBOT phase's atoms may use ONLY InBin." in prompt
    assert "Write them with InBin or a predicate you invented." in prompt
    # What the toy perceived, and nothing else: the prompt's object list is the planner's own labels.
    listed = prompt.split("no others:\n", 1)[1].split("\n\n", 1)[0].splitlines()
    assert listed == [f"- {name}" for name in sorted((*r.scene.items, "blue_bin", "red_bin"))]


def test_the_plan_is_typed_in_the_toys_types_and_the_persons_step_is_an_operator(trial):
    r = trial()
    outcome = r.run()

    spec = outcome.plan.spec
    # Typed by the toy's declaration: a bin is a container and what goes in one an item. The floor is
    # what the toy reports as its table, which is always somewhere things are put, so a container too.
    assert spec.scene_types.surfaces == {"floor", "red_bin", "blue_bin"}
    assert spec.scene_types.movables == set(r.scene.items)
    human = outcome.plan.phases[1]
    assert human.is_human and human.operator is not None
    assert human.operator.signature == "Close(x0: container)"
    assert {str(a) for a in human.operator.preconditions} == {
        f"InBin({r.scene.first}, red_bin)",
        "LidOpen(red_bin)",
    }
    assert {str(a) for a in human.add_effects} == {"LidClosed(red_bin)"}
    assert {str(a) for a in human.delete_effects} == {"LidOpen(red_bin)"}
    # Invented over a bin, so typed as one: the scene's type, never a word the model made up.
    assert [v.predicate.types for v in spec.invented] == [("container",), ("container",)]


# --- the trial, through the real loop ---------------------------------------------------------------


@pytest.mark.parametrize("hosting", sorted(HOSTINGS))
def test_a_whole_trial_runs_on_the_toy_one_leg_per_robot_phase(trial, hosting):
    """Robot, person, robot, robot -- four legs, in that order, and the world ends as instructed.

    Every call the loop makes of the planner is here, in order. That order is the method: the scene is
    perceived before each robot leg and never before the person's step (which is judged on a fresh
    frame from the verification camera instead), the arm is lent for the person's leg and taken back
    before anything else is asked of it, and the first look after the person opens the gripper.
    """
    r = trial(hosting=hosting)
    scene = r.scene
    outcome = r.run()

    # Ran to the end: the loop did not end it, so the operator's label is its verdict.
    assert (outcome.outcome, outcome.failure_stage) == (None, None)
    assert outcome.plan.finished and outcome.legs_recorded == 4
    assert [call.verb for call in r.calls] == [
        "perceive", "plan", "execute",  # phase 0, the robot
        "release_hardware", "reacquire_hardware", "capture_frame",  # phase 1, the person, then its check
        "perceive", "plan", "execute",  # phase 2, the robot
        "perceive", "plan", "execute",  # phase 3, the robot again: its own leg, its own look
    ]  # fmt: skip

    # Each robot phase is one goal, in the toy's wire language, and nothing else -- including the two
    # in a row, which a planner that plans from a clean state would have been handed as one.
    assert [call.args[1] for call in of(r.calls, "plan")] == [
        [GoalAtom("in_bin", (scene.first, "red_bin"))],
        [GoalAtom("in_bin", (scene.second, "blue_bin"))],
        [GoalAtom("in_bin", (scene.third, "blue_bin"))],
    ]
    assert [call.args[1].phase_index for call in of(r.calls, "execute")] == [0, 2, 3]
    assert not any("covers_phases" in record for record in outcome.plan.plans.values())

    # The toy declares both, so both are passed: only what a robot phase moves may be picked (never
    # the bystander), and only the task's last leg goes home.
    planned = [call.kwargs for call in of(r.calls, "plan")]
    assert [kwargs["movables"] for kwargs in planned] == [scene.moved] * 3
    assert [kwargs["return_home"] for kwargs in planned] == [False, False, True]
    assert all(kwargs["surfaces"] == {"floor", "red_bin", "blue_bin"} for kwargs in planned)

    # The arm is parked for the first look only, and the gripper opened for the first after the person.
    looks = [call.kwargs for call in of(r.calls, "perceive")]
    assert [kwargs["reset_arm"] for kwargs in looks] == [True, False, False]
    assert [kwargs.get("open_gripper") for kwargs in looks] == [None, True, None]
    assert "open_gripper" not in looks[0] and "open_gripper" not in looks[2]

    where = whereabouts(r.planner)
    assert scene.placed(where) == scene.instructed
    if scene.bystander:
        assert where[scene.bystander] == "floor"


def test_what_a_leg_is_told_follows_the_declaration_not_the_planner(trial):
    """The same toy, declared differently, is handed exactly what the declaration says.

    Without movable restriction or return home declared, neither keyword is passed at all -- a planner
    that does not take them may leave them out of its signature. With a clean initial state declared,
    the two robot phases in a row become one goal and one leg: the per-phase legs above are the toy's
    ``initial_state_is_clean=False`` at work, not something the loop does to every planner.
    """
    undeclared = replace(TOY_CAPABILITIES, supports_movable_restriction=False, supports_return_home=False)
    r = trial(caps=undeclared)
    r.run()
    for call in of(r.calls, "plan"):
        assert "movables" not in call.kwargs and "return_home" not in call.kwargs

    r = trial(caps=replace(TOY_CAPABILITIES, initial_state_is_clean=True))
    outcome = r.run()
    scene = r.scene
    assert [call.args[1] for call in of(r.calls, "plan")] == [
        [GoalAtom("in_bin", (scene.first, "red_bin"))],
        [GoalAtom("in_bin", (scene.second, "blue_bin")), GoalAtom("in_bin", (scene.third, "blue_bin"))],
    ]
    assert len(of(r.calls, "perceive")) == 2
    assert outcome.plan.plans[2]["covers_phases"] == [2, 3]


def test_the_persons_step_carries_its_operator_and_is_checked_both_ways(trial):
    """The person is handed the magic operator, and the camera is asked about both halves of its
    effect: the lid must now be shut (add), and must no longer be open (delete)."""
    r = trial()
    outcome = r.run()

    (request,) = r.person.calls
    assert request.phase_index == 1 and request.n_phases == 4
    assert request.operator["name"] == "Close"
    assert request.operator["add_effects"] == ["LidClosed(red_bin)"]
    assert request.operator["delete_effects"] == ["LidOpen(red_bin)"]
    assert r.person.legs[0].phase_index == 1 and r.person.legs[0].segment_source == "teleop"
    # One fresh frame from the verification camera, both statements asked of it, nothing else.
    (frame,) = of(r.calls, "capture_frame")
    assert frame.kwargs == {"camera": "external"}
    assert sorted(r.model.asked) == sorted([SHUT, AJAR])

    verdicts = {str(v.atom): v for v in outcome.plan.verdicts}
    assert (verdicts["LidClosed(red_bin)"].holds, verdicts["LidClosed(red_bin)"].expected) == (True, True)
    assert (verdicts["LidOpen(red_bin)"].holds, verdicts["LidOpen(red_bin)"].expected) == (False, False)
    assert all(v.satisfied for v in verdicts.values())
    (verified,) = r.sink.named("human_phase_verified")
    assert verified["ok"] is True and verified["phase_index"] == 1


def test_with_every_check_on_the_camera_reads_the_toys_own_predicate_in_its_own_words(trial):
    """The toy declares ``InBin`` checkable, so every check that may put a goal atom to the camera
    does, phrased by the toy's own description -- "duck is inside red_bin" -- and nothing else is asked.

    TipTop declares only ``On`` checkable, which is what keeps ``Holding`` and ``HandEmpty`` away from a
    third-person camera; which atoms a camera may settle is the planner's declaration, not tandem's rule.
    """
    scene = HOSTINGS["in-process"][2]
    inside = {f"{item} is inside {bin_}" for item, bin_ in scene.instructed.items()}
    # The lid is open when the person's preconditions are read, and gone by the time the effects are.
    seen = {**{statement: True for statement in inside}, SHUT: True, AJAR: [True, False]}
    r = trial(seen, check_human_preconditions=True, check_tamp_preconditions=True, check_tamp_effects=True)
    outcome = r.run()

    assert outcome.plan.finished and (outcome.outcome, outcome.failure_stage) == (None, None)
    assert set(r.model.asked) == inside | {SHUT, AJAR}
    # After the leg that put it there, before the person's step that needs it, and before each robot leg
    # after that (the plan still believes it).
    assert r.model.asked.count(f"{scene.first} is inside red_bin") == 4
    assert all(v.satisfied for v in outcome.plan.verdicts)
    assert {v.role for v in outcome.plan.verdicts} == {"precondition", "effect", "effect (deleted)"}
    before = r.sink.named("phase_preconditions_checked")
    assert [(e["what"], e["phase_index"], e["ok"]) for e in before] == [
        ("human phase", 1, True),
        ("robot leg", 2, True),
        ("robot leg", 3, True),
    ]
    after = r.sink.named("phase_effects_checked")
    assert [(e["phase_index"], e["ok"]) for e in after] == [(0, True), (2, True), (3, True)]


@pytest.mark.parametrize("hosting", sorted(HOSTINGS))
def test_the_merged_episode_is_the_demonstration_and_hitl_json_says_who_did_what(
    trial, hosting, profile, fake_video
):
    """tau = ((tau_1, phi_1), .., (tau_N, phi_N)), read back off one merged episode of the toy's legs,
    with the record beside it naming the toy wherever the planner is meant."""
    r = trial(hosting=hosting)
    scene = r.scene
    outcome = r.run()
    episodes.merge_trajectory(
        profile,
        TRAJECTORY,
        "success",
        outcome.plan,
        tools_dir=None,
        vlm_dir=None,
        log=lambda text: None,
        emit=lambda message: None,
    )
    directory, meta, record = merged_episode(profile, "success")

    # Every stretch of the episode maps back to the phase it carried out, by who the plan gave it to.
    segments = meta["segments"]
    assert [s["phase_index"] for s in segments] == [0, 1, 2, 3]
    assert [s["source"] for s in segments] == ["tamp", "teleop", "tamp", "tamp"]
    assert [s["phase_description"] for s in segments] == [p["description"] for p in record["phases"]]
    assert meta["n_phases"] == len(record["phases"]) == 4
    kind = {"tamp": "robot", "teleop": "human"}
    assert [kind[s["source"]] for s in segments] == [p["executor"] for p in record["phases"]]
    # The toy's own record of each leg says what it was told: only the last one went home.
    parked = sorted((directory / "segments").iterdir())
    told = [
        json.loads((leg / "toy_plan.json").read_text()) for leg in parked if leg.name.split("_")[1] == "tamp"
    ]
    assert [plan["return_home"] for plan in told] == [False, False, True]

    # How the trial ended: filed under success, and not by the loop's say-so.
    assert (record["outcome"], record["excluded"], record["failure_stage"]) == ("success", False, None)
    assert record["filed_under"] == "success"
    assert record["planner"] == HOSTINGS[hosting][0]

    # Who did what: the robot's operators are the toy's declaration, the person's the model's.
    provenance = record["provenance"]
    assert provenance["robot_operators"]["signatures"] == ["Drop(obj: item, bin: container)"]
    assert provenance["robot_operators"]["by"].startswith(f"{record['planner']} -- ")
    assert provenance["human_operators"]["signatures"] == ["Close(x0: container)"]
    operator = record["phases"][1]["operator"]
    assert operator["instance"] == "Close(red_bin)"
    assert operator["preconditions"] == [f"InBin({scene.first}, red_bin)", "LidOpen(red_bin)"]
    assert operator["add_effects"] == ["LidClosed(red_bin)"]
    assert operator["delete_effects"] == ["LidOpen(red_bin)"]

    # Each robot phase: the goal as the toy took it, and the plan the toy found for it alone.
    for index, (item, bin_) in zip((0, 2, 3), scene.instructed.items(), strict=True):
        phase = record["phases"][index]
        assert phase["goal"] == [{"predicate": "in_bin", "args": [item, bin_]}]
        assert phase["goal_description"] == [f"{item} is inside {bin_}"]
        assert phase["task_plan"] == [f"Drop({item}, {bin_})"]
        assert "covers_phases" not in phase

    # The verdicts the person's step passed on, both halves, filed against phase 1.
    checked = {v["atom"]: v for v in record["verifications"]}
    assert set(checked) == {"LidClosed(red_bin)", "LidOpen(red_bin)"}
    assert all(v["phase"] == 1 and v["satisfied"] for v in checked.values())
    assert (
        checked["LidClosed(red_bin)"]["holds"] is True and checked["LidClosed(red_bin)"]["role"] == "effect"
    )
    assert checked["LidOpen(red_bin)"]["holds"] is False
    assert checked["LidOpen(red_bin)"]["role"] == "effect (deleted)"
    assert record["checks"]["human_effects"] is True and record["checks"]["unchecked_phases"] == []


# --- when the toy cannot plan --------------------------------------------------------------------------

# The model asks for the last item "in" the floor. The floor is the toy's table, so the goal is well
# typed -- the parser has no reason to refuse it -- and only the planner can say it is not a bin.
INTO_THE_FLOOR = {"third_into": "floor"}


def test_a_goal_the_toy_cannot_plan_ends_the_trial_at_tamp_planning_by_default(trial):
    r = trial(plan=HOSTINGS["in-process"][2].plan(**INTO_THE_FLOOR))
    scene = r.scene
    outcome = r.run()

    assert (outcome.outcome, outcome.failure_stage) == ("failure", "tamp_planning")
    assert "floor is not a bin" in outcome.reason
    (failed,) = r.sink.named("phase_plan_failed")
    assert failed["policy"] == "abort" and "floor is not a bin" in failed["reason"]
    # The legs before it ran and are on disk; the one it could not plan never executed.
    assert [call.args[1].phase_index for call in of(r.calls, "execute")] == [0, 2]
    assert outcome.legs_recorded == 3
    assert outcome.plan.index == 3 and not outcome.plan.finished
    assert whereabouts(r.planner)[scene.third] == "floor"
    record = outcome.plan.to_json()
    assert (record["outcome"], record["failure_stage"]) == ("failure", "tamp_planning")


def test_replan_feeds_the_toys_failure_back_and_the_new_plan_is_carried_out(trial):
    """``on_robot_phase_failure: replan``. The model is asked again with what the TOY said was wrong
    -- in the toy's words, with the phase in its own atoms -- and the plan it answers with runs."""
    scene = HOSTINGS["in-process"][2]
    rest = proposal(robot(f"put the {scene.third} in the blue bin", scene.third, "blue_bin"))
    r = trial(plan=scene.plan(**INTO_THE_FLOOR), replan=rest, on_robot_phase_failure="replan")
    outcome = r.run()

    first, again = r.model.plan_prompts
    assert Model.REPLAN_MARKER not in first
    assert Model.REPLAN_MARKER in again
    assert "phase 3 could not be planned: floor is not a bin" in again
    assert f"put the {scene.third} in the blue bin, with the goal InBin({scene.third}, floor)" in again
    # The re-proposal is planned against the scene as it now stands: one more look, then the new plan.
    assert [call.verb for call in r.calls][-5:] == ["perceive", "plan", "perceive", "plan", "execute"]
    assert of(r.calls, "plan")[-1].args[1] == [GoalAtom("in_bin", (scene.third, "blue_bin"))]
    assert (outcome.outcome, outcome.failure_stage) == (None, None)
    assert (
        outcome.plan.finished and outcome.plan.spec.phases[0].description == rest["phases"][0]["description"]
    )
    assert whereabouts(r.planner)[scene.third] == "blue_bin"


def test_a_replanned_trial_keeps_every_plan_and_says_which_plan_each_leg_carried_out(trial, profile, fake_video):
    """The record used to be the last plan alone: the person's step, checked and passed under the
    first plan, vanished from hitl.json, and two segments claimed phase 0 for different subgoals."""
    scene = HOSTINGS["in-process"][2]
    rest = proposal(robot(f"put the {scene.third} in the blue bin", scene.third, "blue_bin"))
    r = trial(plan=scene.plan(**INTO_THE_FLOOR), replan=rest, on_robot_phase_failure="replan")
    outcome = r.run()
    assert (outcome.outcome, outcome.plan.finished) == (None, True)
    episodes.merge_trajectory(
        profile,
        TRAJECTORY,
        "success",
        outcome.plan,
        tools_dir=None,
        vlm_dir=None,
        log=lambda text: None,
        emit=lambda message: None,
        superseded=outcome.superseded_plans,
        leg_generations=outcome.leg_generations,
    )
    _, meta, record = merged_episode(profile, "success")

    (first,) = record["superseded_plans"]
    assert record["plan_generation"] == 1 and first["plan_generation"] == 0
    assert "floor is not a bin" in first["superseded_because"]
    assert len(first["phases"]) == 4
    assert first["provenance"]["human_operators"]["signatures"] == ["Close(x0: container)"]
    assert first["phases"][2]["task_plan"] == [f"Drop({scene.second}, blue_bin)"]
    checked = {v["atom"]: v for v in first["verifications"]}
    assert set(checked) == {"LidClosed(red_bin)", "LidOpen(red_bin)"}
    assert all(v["phase"] == 1 and v["satisfied"] for v in checked.values())

    # Every stretch names its phase as (plan, index), and that phase is the one it carried out.
    segments = meta["segments"]
    assert [(s["plan_generation"], s["phase_index"]) for s in segments] == [(0, 0), (0, 1), (0, 2), (1, 0)]
    plans = {0: first, 1: record}
    for segment in segments:
        phase = plans[segment["plan_generation"]]["phases"][segment["phase_index"]]
        assert segment["phase_description"] == phase["description"]
    assert "n_phases" not in meta, "two plans' lengths, read as one"


class SlowToySidecar(ToySidecarPlanner):
    """The toy next door, slow enough that a stop reaches it before a drop is done."""

    def launch_command(self) -> list[str]:
        return [*super().launch_command(), "--step", "0.2"]


@pytest.mark.parametrize("hosting", sorted(HOSTINGS))
def test_a_stop_reaches_the_toy_part_way_through_a_leg_and_the_plan_does_not_advance(trial, hosting):
    """The toy declares cooperative stop, and the loop never passed it a should_stop: a preempt waited
    for the whole leg. Now the stop reaches it -- across the process boundary as a stop file, for the
    sidecar -- and the leg it cut short neither advances the plan nor counts as a failed execution."""
    r = trial(hosting=hosting, planner_cls=SlowToySidecar if hosting == "sidecar" else None)
    if hosting == "in-process":
        r.planner.world.step_seconds = 0.2
    r.operator.stop_legs = True

    with pytest.raises(Stopped):
        r.run()
    (executed,) = of(r.calls, "execute")
    assert callable(executed.kwargs["should_stop"])
    assert whereabouts(r.planner)[r.scene.first] == "floor", "the drop the stop was meant to cut short ran"
    assert r.loop.outcome.plan.index == 0
    assert r.loop.outcome.failure_stage is None


def test_a_planner_that_does_not_declare_cooperative_stop_is_not_handed_one(trial):
    r = trial(caps=replace(TOY_CAPABILITIES, supports_cooperative_stop=False))
    r.run()
    assert all("should_stop" not in call.kwargs for call in of(r.calls, "execute"))


# --- through a session, on a profile that names the toy -------------------------------------------------


@pytest.fixture
def toy_session(profile, tmp_path, monkeypatch, fake_video):
    """A live session on a profile saved with ``planner: {backend: toy}`` and loaded back from disk.

    Nothing about the planner is handed to the session -- not even a runtime: the profile names it, the
    profile loader checks the name against the registry and the options against the toy's own
    ``OPTIONS`` (``validate_options``), and the session builds it from the registry with the profile's
    ``planner.options``, the way it builds any planner.
    """
    from tandem.planning import llm

    made: list[Session] = []

    def build(answers=DONE):
        isolate_registry(monkeypatch)
        registry.register_backend("toy", ToyPlanner)
        scene = HOSTINGS["in-process"][2]
        profile.planner = PlannerSpec(backend="toy", options={"items": list(scene.items)})
        profile.hitl.enabled = True
        saved = YAML(typ="safe").load(profiles.save(profile).read_text())
        # The file on disk is the toy's profile and nobody else's: TiPToP's robot, perception and tamp
        # are its planner options, so a profile planning with the toy carries none of them anywhere.
        assert saved["planner"] == {"backend": "toy", "options": {"items": list(scene.items)}}
        assert not {"robot", "perception", "tamp"} & set(saved)
        loaded = profiles.load(profile.name)
        assert (loaded.planner.backend, loaded.planner.options) == ("toy", {"items": list(scene.items)})

        monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
        model = Model(scene.plan(), answers)
        monkeypatch.setattr(llm, "gemini_client", lambda: model)
        person = Person()
        use_fake_executor(monkeypatch, person)
        session = Session(loaded, task=scene.instruction)
        states: list[str] = []
        session.subscribe(lambda m: states.append(m["state"]) if m.get("type") == "state" else None)
        session.start()
        assert wait_for(lambda: session.state is State.AWAITING_TASK), f"stuck in {session.state}"
        made.append(session)
        return SimpleNamespace(session=session, scene=scene, model=model, person=person, states=states)

    yield build
    for session in made:
        if session.alive:
            session.stop(park=False)
            session.wait(timeout=5)


def _carry_out_the_persons_step(session, attempt: int) -> None:
    assert wait_for(
        lambda: (
            session.state is State.AWAITING_HUMAN_PHASE
            and session.human_phase is not None
            and session.human_phase.attempt == attempt
        )
    ), f"stuck in {session.state}"
    session.request_teleop()


def test_a_session_on_the_toy_runs_the_trial_and_files_the_demonstration(toy_session, profile):
    t = toy_session()
    session, scene = t.session, t.scene
    # The planner the profile names, built with the profile's options: the world holds its items.
    assert isinstance(session._backend, ToyPlanner)
    assert set(whereabouts(session._backend)) == set(scene.items)
    # No runtime was located for it: TiPToP's is TiPToP's factory's to find, and the toy declares none.
    assert session._backend.ctx.runtime_dir is None

    session.next_task()
    _carry_out_the_persons_step(session, attempt=1)
    assert wait_for(lambda: session.state is State.AWAITING_LABEL), f"stuck in {session.state}"
    assert session._legs_recorded == 4
    assert scene.placed(whereabouts(session._backend)) == scene.instructed

    session.label(True)
    assert wait_for(lambda: merged_under(profile, "success")), "nothing was filed"
    _, meta, record = merged_episode(profile, "success")
    assert [(s["phase_index"], s["source"]) for s in meta["segments"]] == [
        (0, "tamp"),
        (1, "teleop"),
        (2, "tamp"),
        (3, "tamp"),
    ]
    assert record["planner"] == "toy" and record["outcome"] == "success"
    assert record["provenance"]["robot_operators"]["signatures"] == ["Drop(obj: item, bin: container)"]
    assert {v["atom"]: v["satisfied"] for v in record["verifications"]} == {
        "LidClosed(red_bin)": True,
        "LidOpen(red_bin)": True,
    }
    assert "On(" not in t.model.plan_prompts[0]


def test_a_persons_step_that_never_verifies_is_excluded_without_a_label(toy_session, profile):
    """The paper's rule on a planner it never met: the lid reads open after every attempt, so the
    trial is terminated, kept out of the dataset, and filed with its failing verdicts and raw legs --
    without the operator being asked for a label they could answer "success"."""
    t = toy_session(SPRUNG_BACK)
    session, scene = t.session, t.scene

    session.next_task()
    _carry_out_the_persons_step(session, attempt=1)
    # The default single retry: the person is asked again, told what is still wrong.
    _carry_out_the_persons_step(session, attempt=2)
    assert wait_for(lambda: session.excluded_count == 1), f"stuck in {session.state}"
    assert wait_for(lambda: session.state is State.AWAITING_TASK)
    assert "awaiting_label" not in t.states
    assert session.labeled_count == 0
    assert len(t.person.calls) == 2

    assert wait_for(lambda: merged_under(profile, "failure")), "nothing was filed"
    _, meta, record = merged_episode(profile, "failure")
    ended = (record["outcome"], record["excluded"], record["failure_stage"], record["filed_under"])
    assert ended == ("excluded", True, "verification", "failure")
    assert "close the red bin's lid" in record["outcome_reason"]
    # The failing verdict is on the record, against the phase it was about: still open.
    (still_open,) = [v for v in record["verifications"] if v["atom"] == "LidOpen(red_bin)"]
    assert (still_open["holds"], still_open["expected"], still_open["satisfied"]) == (True, False, False)
    assert still_open["phase"] == 1
    # Every raw leg kept: the robot's one, and both of the person's attempts at phase 1. Nothing after
    # them ran.
    assert [(s["phase_index"], s["source"]) for s in meta["segments"]] == [
        (0, "tamp"),
        (1, "teleop"),
        (1, "teleop"),
    ]
    where = whereabouts(session._backend)
    assert (where[scene.second], where[scene.third]) == ("floor", "floor")
