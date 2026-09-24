"""Regressions from the method review: each test here failed before the fix it is named for.

What they share is the method's two promises. A plan the proposer can repair is sent back to it
rather than failing a trial the proposer never hears about: a template ``describe`` cannot render, a
robot leg with no goal, a goal that holds one object in two places, a precondition the plan's own
placement rules out. And ``hitl.json`` says what actually happened: a robot phase handed to a person,
a step staged by hand with no leg, a phase whose check had nothing to ask, every image a model was
shown (a cached proposal included), and one spelling for every operator signature.

The loop-level tests drive the phase loop against the stand-in backend and the scripted camera from
tests/test_trial_outcomes.py, with a model that can answer the proposal differently each time it is
asked -- which is what a repair looks like from the outside.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from types import SimpleNamespace

import pytest
from fake_backend import FakeBackend
from fake_executor import FakeExecutor, use_fake_executor
from test_trial_outcomes import (
    CLOSED,
    OPEN,
    PLAN,
    PUT_IN,
    TAKE_OFF,
    Camera,
    Operator,
    Sink,
    open_the_box,
    proposal,
)

from tandem.core.episodes import LegDirs
from tandem.core.phase_loop import PhaseLoop
from tandem.executors.base import ExecutorContext
from tandem.planners import registry
from tandem.planners.sdk import capability_problems
from tandem.planning import feasibility, llm
from tandem.planning.cache import ProposalCache
from tandem.planning.config import PlanningConfig
from tandem.planning.contracts import check_plan_effects, exclusive_conflicts
from tandem.planning.plan import PhasePlan, operator_signature
from tandem.planning.proposal import check_plan, parse_plan_response
from tandem.planning.record import recording_to
from tandem.planning.structs import HumanOperator, VLMPredicate
from tandem.planning.symbols import Atom, Parameter, Predicate, ProposalError, describe

CAPS = registry.capabilities("tiptop")
NO_DISPLACEMENT = dataclasses.replace(CAPS, exclusive_arguments={})
OBJECTS = ["blue_toy", "white_box"]
TABLE = "table"
CFG = PlanningConfig(enabled=True)


def _atoms(*pairs):
    return [{"predicate": name, "args": list(args)} for name, args in pairs]


def _op(name, args, *, pre=(), add=(), dele=()):
    return {
        "name": name,
        "args": list(args),
        "preconditions": _atoms(*pre),
        "add_effects": _atoms(*add),
        "delete_effects": _atoms(*dele),
    }


def _robot(description, *atoms):
    return {"executor": "robot", "description": description, "atoms": _atoms(*atoms)}


def _human(description, atoms, operator):
    return {
        "executor": "human",
        "description": description,
        "instructions": f"Please {description}.",
        "atoms": _atoms(*atoms),
        "operator": operator,
    }


def parse(response, caps=CAPS):
    return parse_plan_response(response, "do the thing", OBJECTS, TABLE, caps)


# --- the loop, with a model that can change its answer ---------------------------------------------


class Model(Camera):
    """test_trial_outcomes' camera, answering the Nth proposal with the Nth plan (the last repeats)."""

    def __init__(self, plans, answers=None) -> None:
        super().__init__(plans[0], answers)
        self.plans = [p if isinstance(p, str) else json.dumps(p) for p in plans]
        self.plan_prompts: list[str] = []

    async def generate_content(self, model, contents, config):
        prompt = contents[-1]
        if not ("Statement:" in prompt and self.PLAN_MARKER not in prompt):
            self.plan = self.plans.pop(0) if len(self.plans) > 1 else self.plans[0]
            self.plan_prompts.append(prompt)
        return await super().generate_content(model, contents, config)


@pytest.fixture
def loop_for(profile, tmp_path, monkeypatch):
    """A phase loop against a stand-in backend and a scripted model. Nothing runs until `.run()`."""

    def make(plans, answers=None, *operator_answers, executor=None, record=True, **cfg):
        model = Model(plans if isinstance(plans, list) else [plans], answers)
        monkeypatch.setattr(llm, "gemini_client", lambda: model)
        backend = FakeBackend(None, output_dir=tmp_path / "frames")
        sink, operator, hands = Sink(), Operator(*operator_answers), executor or FakeExecutor()
        use_fake_executor(monkeypatch, hands)
        loop = PhaseLoop(
            backend,
            backend.capabilities(),
            PlanningConfig(
                **{"enabled": True, "save_vlm_io": False, "allow_unrecorded_human_phase": True, **cfg}
            ),
            events=sink,
            operator=operator,
            executor_context=ExecutorContext(profile=profile, session_dir=tmp_path / "session"),
            legs=LegDirs(profile, tmp_path / "session", log=sink.log),
            record=record,
        )

        def run(vlm_dir=None):
            return loop.run(
                task="put the toy in the box",
                instruction="put the toy in the box",
                trajectory_id="t" * 16,
                vlm_dir=vlm_dir,
            )

        return SimpleNamespace(
            loop=loop, backend=backend, sink=sink, operator=operator, hands=hands, model=model, run=run
        )

    return make


def ran(backend) -> list:
    return [leg["leg"].phase_index for leg in backend.legs]


def frames(backend) -> int:
    return sum(1 for call in backend.calls if call.startswith("capture_frame"))


def record_of(outcome) -> dict:
    return json.loads(json.dumps(outcome.plan.to_json(), default=str))


# --- 1. an invented predicate's template is checked the way describe() reads it -------------------


IS_OPEN = Predicate("IsOpen", (Parameter("x0", "surface"),))


@pytest.mark.parametrize(
    "template",
    [
        "the lid of {box} is open",  # a named field: the slip from {0} this is mostly about
        "{ 0 } is open",
        "{0.x} is open",
        "{0[1]} is open",  # renders against a stand-in name, and raises against a one-letter one
        "{0!r} is open",
        "{0:>5} is open",
        "{0} is open {",
        "{0} is open }",
    ],
)
def test_a_template_describe_cannot_render_is_refused_for_the_proposer(template):
    with pytest.raises(ProposalError, match=r"Instructions for IsOpen .*\{0\}, \{1\}"):
        VLMPredicate(IS_OPEN, template)


def test_an_auto_numbered_field_is_refused_even_with_no_arguments():
    with pytest.raises(ProposalError, match="not a placeholder"):
        VLMPredicate(Predicate("IsTidy"), "the workspace {} is tidy")


def test_literal_braces_and_bare_placeholders_are_accepted_and_render():
    braces = VLMPredicate(IS_OPEN, "literal {{braces}} around {0}")
    assert describe(Atom("IsOpen", ("white_box",)), {"IsOpen": braces.instructions}) == (
        "literal {braces} around white_box"
    )
    VLMPredicate(Predicate("On2", (Parameter("a", "m"), Parameter("b", "s"))), "{1} holds {0}")


def _with_template(plan: dict, template: str) -> dict:
    bad = json.loads(json.dumps(plan))
    for entry in bad["new_predicates"]:
        if entry["name"] == "IsOpen":
            entry["instructions"] = template
    return bad


def test_the_parser_refuses_a_plan_whose_template_names_its_argument():
    with pytest.raises(ProposalError, match=r"uses \{box\}, which is not a placeholder"):
        parse(_with_template(PLAN, "the container {box} is open"))


def test_a_bad_template_is_repaired_by_the_proposer_instead_of_crashing_the_human_phase(loop_for):
    """It used to be accepted, the first robot leg ran, and describe() raised KeyError('box') out of
    PhaseLoop.run at the hand-off -- a trial with legs on disk and no outcome on its record."""
    r = loop_for([_with_template(PLAN, "the container {box} is open"), PLAN], {OPEN: True, CLOSED: False})
    outcome = r.run()

    assert r.model.plan_calls == 2, "the repair loop asked again"
    assert "which is not a placeholder" in r.model.plan_prompts[1], "and said why"
    assert ran(r.backend) == [0, 2]
    assert outcome.outcome is None and outcome.plan.finished


def test_a_planner_template_is_held_to_the_same_rule_and_reported_not_raised():
    # describe() renders the planner's own phrasings too. A trial format against stand-in names let
    # `{0.x}` escape the declaration check as an AttributeError, and `{0[1]}` pass it.
    for template in ("{0.x} is resting on top of {1}", "{0[1]} is resting on top of {1}"):
        caps = dataclasses.replace(
            CAPS, predicate_descriptions={**CAPS.predicate_descriptions, "On": template}
        )
        problems = capability_problems(caps)
        assert any("predicate_descriptions['On']" in p for p in problems), template
    assert capability_problems(CAPS) == []


# --- 2. a precondition the plan's own placement rules out -----------------------------------------


WIPE_UNDER_THE_TOY = _human(
    "wipe under the toy",
    [("IsWiped", ["table"])],
    _op("Wipe", ["table"], pre=[("On", ["blue_toy", "table"])], add=[("IsWiped", ["table"])]),
)
WIPED = {"name": "IsWiped", "instructions": "the surface {0} has been wiped"}


def test_a_placement_rules_out_a_precondition_no_earlier_phase_established():
    """The toy goes into the box; then a person needs it on the table. Nothing put it on the table,
    but nothing has to: the plan itself put it somewhere else, whatever the workspace started as."""
    spec = parse(
        {
            "new_predicates": [WIPED],
            "phases": [_robot("toy into the box", ("On", ["blue_toy", "white_box"])), WIPE_UNDER_THE_TOY],
        }
    )
    broken = check_plan_effects(spec, caps=CAPS)
    assert broken is not None
    assert "On(blue_toy, table)" in broken and "an earlier phase deletes it" in broken
    # A backend that never said an object rests on one thing at a time gets no such conclusion.
    assert check_plan_effects(spec, caps=NO_DISPLACEMENT) is None


def test_a_persons_placement_rules_it_out_too():
    spec = parse(
        {
            "new_predicates": [WIPED],
            "phases": [
                _human(
                    "put the toy in the box",
                    [("On", ["blue_toy", "white_box"])],
                    _op("Put", ["blue_toy"], add=[("On", ["blue_toy", "white_box"])]),
                ),
                WIPE_UNDER_THE_TOY,
            ],
        }
    )
    assert "On(blue_toy, table)" in (check_plan_effects(spec, caps=CAPS) or "")


def test_a_placement_undone_in_between_rules_nothing_out():
    # Once a phase declares the toy is no longer in the box, the plan no longer knows where it is.
    spec = parse(
        {
            "new_predicates": [WIPED, {"name": "IsHeld", "instructions": "a person is holding {0}"}],
            "phases": [
                _robot("toy into the box", ("On", ["blue_toy", "white_box"])),
                _human(
                    "take the toy out",
                    [("IsHeld", ["blue_toy"])],
                    _op(
                        "TakeOut",
                        ["blue_toy"],
                        add=[("IsHeld", ["blue_toy"])],
                        dele=[("On", ["blue_toy", "white_box"])],
                    ),
                ),
                WIPE_UNDER_THE_TOY,
            ],
        }
    )
    assert check_plan_effects(spec, caps=CAPS) is None


# --- 5. one object in two places at once ----------------------------------------------------------


def test_a_phase_asking_for_one_object_in_two_places_is_sent_back():
    everywhere = {
        "phases": [_robot("toy everywhere", ("On", ["blue_toy", "white_box"]), ("On", ["blue_toy", "table"]))]
    }
    # With the contract check off too: this is a goal no planner can reach, and cuTAMP would search
    # for it until its timeout.
    with pytest.raises(ProposalError, match="blue_toy can be On only one thing at a time"):
        check_plan(parse(everywhere), dataclasses.replace(CFG, check_plan_effects=False), CAPS)
    # Without the declaration there is nothing to say it is impossible.
    check_plan(parse(everywhere, caps=NO_DISPLACEMENT), CFG, NO_DISPLACEMENT)


def test_a_persons_operator_adding_both_is_sent_back_too():
    spec = parse(
        {
            "phases": [
                _human(
                    "put the toy in two places",
                    [("On", ["blue_toy", "white_box"])],
                    _op(
                        "Put",
                        ["blue_toy"],
                        add=[("On", ["blue_toy", "white_box"]), ("On", ["blue_toy", "table"])],
                    ),
                )
            ]
        }
    )
    with pytest.raises(
        ProposalError, match="asks for both On\\(blue_toy, table\\) and On\\(blue_toy, white_box\\)"
    ):
        check_plan(spec, CFG, CAPS)


def test_exclusive_conflicts_pairs_only_the_same_slot():
    atoms = [Atom("On", ("a", "x")), Atom("On", ("a", "y")), Atom("On", ("b", "x")), Atom("Holding", ("a",))]
    assert exclusive_conflicts(atoms, caps=CAPS) == [(Atom("On", ("a", "x")), Atom("On", ("a", "y")))]
    assert exclusive_conflicts(atoms, caps=NO_DISPLACEMENT) == []


TOY_TO_TABLE = _robot("toy to the table", ("On", ["blue_toy", "table"]))
TOY_TO_BOX = _robot("toy into the box", ("On", ["blue_toy", "white_box"]))


@pytest.mark.parametrize(
    ("caps", "run"),
    [
        # A clean-state planner that may pick an object twice: only exclusivity stops the run, and it
        # must, since {On(toy, table), On(toy, box)} is one goal no workspace can satisfy.
        (dataclasses.replace(CAPS, one_pick_per_object=False), 1),
        (dataclasses.replace(CAPS, one_pick_per_object=False, exclusive_arguments={}), 2),
        # TipTop: one pick per object already splits at the same place.
        (CAPS, 1),
    ],
    ids=["multi-pick-exclusive", "multi-pick-no-exclusivity", "tiptop"],
)
def test_consecutive_phases_holding_one_object_in_two_places_are_not_conjoined(caps, run):
    spec = parse({"phases": [TOY_TO_TABLE, TOY_TO_BOX]}, caps=caps)
    walk = PhasePlan(cfg=CFG, caps=caps, instruction="x", trajectory_id="t", spec=spec)
    assert len(walk.robot_run()) == run
    assert feasibility.conjoinable_run(spec.phases, caps) == run


# --- 6. a robot leg with nothing the planner can be given -----------------------------------------


FOLD = _human(
    "fold the cloth while the robot holds the toy",
    [("IsFolded", ["white_box"])],
    _op("Fold", ["white_box"], pre=[("Holding", ["blue_toy"])], add=[("IsFolded", ["white_box"])]),
)
FOLDED = {"name": "IsFolded", "instructions": "{0} has been folded over on itself"}
LET_GO = _robot("let go of the toy", ("HandEmpty", []))


@pytest.mark.parametrize("conjoin", [True, False])
def test_a_leg_of_only_handempty_is_sent_back(conjoin):
    spec = parse(
        {
            "new_predicates": [FOLDED],
            "phases": [_robot("hold the toy", ("Holding", ["blue_toy"])), FOLD, LET_GO],
        }
    )
    with pytest.raises(
        ProposalError, match=r"phase 2 \('let go of the toy'\) asks the robot only for HandEmpty\(\)"
    ):
        check_plan(spec, dataclasses.replace(CFG, conjoin_robot_phases=conjoin), CAPS)


def test_a_conjoined_leg_with_no_goal_names_every_phase_in_it():
    spec = parse({"phases": [LET_GO, _robot("keep the gripper empty", ("HandEmpty", []))]})
    assert feasibility.robot_leg_without_a_goal(spec, CAPS, conjoin=True).startswith(
        "phases 0 to 1 ('let go of the toy'; 'keep the gripper empty'), planned together, ask the robot "
        "only for HandEmpty()"
    )


def test_handempty_beside_a_placement_is_fine_only_when_the_two_are_planned_together():
    spec = parse({"phases": [LET_GO, TOY_TO_BOX]})
    check_plan(spec, CFG, CAPS)  # conjoined: the leg's goal is the placement
    with pytest.raises(ProposalError, match="cannot be given as a goal"):
        check_plan(spec, dataclasses.replace(CFG, conjoin_robot_phases=False), CAPS)


def test_an_empty_leg_is_repaired_by_the_proposer_instead_of_ending_the_trial(loop_for):
    bad = proposal(LET_GO, open_the_box(), PUT_IN)
    r = loop_for([bad, PLAN], {OPEN: True, CLOSED: False})
    outcome = r.run()

    assert r.model.plan_calls == 2
    assert "cannot be given as a goal" in r.model.plan_prompts[1]
    assert "state what the robot must achieve with Holding or On" in r.model.plan_prompts[1]
    assert (outcome.outcome, outcome.failure_stage) == (None, None)
    assert ran(r.backend) == [0, 2]


# --- 3 and 10. every image a model was shown stays on the trail ------------------------------------


def _png(colour):
    from PIL import Image

    return Image.new("RGB", (8, 8), colour)


def _rows(directory):
    return [json.loads(line) for line in (directory / "index.jsonl").read_text().splitlines()]


def test_a_second_recorder_into_one_directory_continues_the_numbering(tmp_path):
    from PIL import Image

    for colour in ("red", "blue"):
        with recording_to(tmp_path) as recorder:
            recorder.record(
                label="classify IsOpen(box)",
                attempt=1,
                model="m",
                prompt="p",
                response='{"holds": true}',
                image=_png(colour),
            )
    first, second = _rows(tmp_path)
    assert first["input_image"] != second["input_image"], "the second check overwrote the first's frame"
    assert (first["seq"], second["seq"]) == (1, 2)
    assert Image.open(tmp_path / first["input_image"]).getpixel((0, 0)) == (255, 0, 0)
    assert Image.open(tmp_path / second["input_image"]).getpixel((0, 0)) == (0, 0, 255)


def test_a_retried_check_keeps_the_frame_the_person_was_sent_back_over(loop_for, tmp_path):
    from PIL import Image

    r = loop_for(PLAN, {OPEN: True, CLOSED: [True, False]})
    colours = iter(["red", "blue"])
    r.loop._verification_frame = lambda: _png(next(colours))
    vlm = tmp_path / "vlm"
    r.run(vlm_dir=vlm)

    rows = _rows(vlm)
    names = [row["input_image"] for row in rows]
    assert len(names) == len(set(names)), f"rows share image files: {names}"
    closed = [row for row in rows if row["label"] == "classify IsClosed(white_box)"]
    assert [json.loads(row["response"])["holds"] for row in closed] == [True, False]
    assert [Image.open(vlm / row["input_image"]).getpixel((0, 0)) for row in closed] == [
        (255, 0, 0),
        (0, 0, 255),
    ]


class _CountingModel:
    """A model client that answers every proposal the same, and counts the calls."""

    def __init__(self, text):
        self.text = text
        self.calls = 0
        self.aio = self

    @property
    def models(self):
        return self

    async def generate_content(self, model, contents, config):
        from unittest import mock

        self.calls += 1
        return mock.Mock(text=self.text)


def test_a_proposal_served_from_the_cache_is_on_the_trail_marked_as_a_replay(tmp_path, monkeypatch):
    client = _CountingModel(json.dumps({"answer": 42}))
    monkeypatch.setattr(llm, "gemini_client", lambda: client)
    cache = ProposalCache(tmp_path / "c.sqlite")

    def ask():
        return asyncio.run(
            llm.query_json(
                "plan this", lambda data: data, model="m", schema={}, image=_png("green"), cache=cache
            )
        )

    with recording_to(tmp_path / "a"):
        assert ask() == {"answer": 42}
    with recording_to(tmp_path / "b"):
        assert ask() == {"answer": 42}

    assert client.calls == 1, "the second came from the cache"
    (live,) = _rows(tmp_path / "a")
    (replayed,) = _rows(tmp_path / "b")
    assert live["cached"] is False
    assert replayed["cached"] is True
    assert (replayed["prompt"], replayed["response"]) == ("plan this", live["response"])
    assert (tmp_path / "b" / replayed["input_image"]).is_file()
    assert (tmp_path / "b" / replayed["output_image"]).is_file()


# --- 7. how a human phase was carried out ---------------------------------------------------------


def test_a_step_staged_by_hand_is_on_the_record_as_having_no_leg(loop_for):
    r = loop_for(PLAN, {OPEN: True, CLOSED: False}, "done", allow_unrecorded_human_phase=True)
    outcome = r.run()

    record = record_of(outcome)
    assert record["phases"][1]["carried_out"] == [
        {"attempt": 1, "carried_out_by": "by_hand", "status": None, "leg_recorded": False, "n_frames": 0}
    ]
    assert record["checks"]["unrecorded_human_phases"] == [1]
    assert r.sink.named("human_phase_by_hand") == [{"phase_index": 1, "attempt": 1}]
    assert "carried_out" not in record["phases"][0], "a robot phase has no such record"


def test_a_teleoperated_step_is_on_the_record_with_its_leg(loop_for):
    r = loop_for(PLAN, {OPEN: True, CLOSED: False}, "teleop")
    outcome = r.run()

    record = record_of(outcome)
    assert record["phases"][1]["carried_out"] == [
        {"attempt": 1, "carried_out_by": "teleop", "status": "done", "leg_recorded": True, "n_frames": 30}
    ]
    assert record["checks"]["unrecorded_human_phases"] == []
    assert r.sink.named("human_phase_by_hand") == []


def test_an_executor_that_recorded_nothing_is_on_the_record_as_such(loop_for):
    r = loop_for(
        PLAN,
        {OPEN: True, CLOSED: False},
        "teleop",
        executor=FakeExecutor(n_frames=0),
        allow_unrecorded_human_phase=True,
    )
    record = record_of(r.run())
    (attempt,) = record["phases"][1]["carried_out"]
    assert (attempt["carried_out_by"], attempt["leg_recorded"]) == ("teleop", False)
    assert record["checks"]["unrecorded_human_phases"] == [1]


def test_each_attempt_is_kept_and_one_recorded_leg_is_enough(loop_for):
    # Teleoperated, did not verify, then finished by hand: the episode has the first attempt's leg.
    r = loop_for(PLAN, {OPEN: True, CLOSED: [True, False]}, "teleop", "done")
    record = record_of(r.run())
    assert [
        (a["attempt"], a["carried_out_by"], a["leg_recorded"]) for a in record["phases"][1]["carried_out"]
    ] == [
        (1, "teleop", True),
        (2, "by_hand", False),
    ]
    assert record["checks"]["unrecorded_human_phases"] == []


def test_the_two_ways_of_carrying_a_step_out_no_longer_record_the_same(loop_for):
    by_hand = record_of(loop_for(PLAN, {OPEN: True, CLOSED: False}, "done").run())
    teleop = record_of(loop_for(PLAN, {OPEN: True, CLOSED: False}, "teleop").run())
    assert by_hand["phases"] != teleop["phases"]
    assert by_hand["checks"] != teleop["checks"]


# --- 8. a check with nothing to ask is not a check that passed ------------------------------------


LET_GO_BY_HAND = {
    "executor": "human",
    "description": "take the toy out of the gripper",
    "instructions": "Take the toy out of the gripper.",
    "atoms": _atoms(("HandEmpty", [])),
    "operator": _op("Release", ["blue_toy"], add=[("HandEmpty", [])], dele=[("Holding", ["blue_toy"])]),
}


def test_a_human_phase_whose_effects_no_camera_can_settle_is_recorded_unchecked(loop_for):
    r = loop_for(proposal(TAKE_OFF, LET_GO_BY_HAND, PUT_IN), {})
    outcome = r.run()

    assert r.model.asked == [] and frames(r.backend) == 0, "nothing to look at, so no frame"
    assert ran(r.backend) == [0, 2]
    (verified,) = r.sink.named("human_phase_verified")
    assert verified["ok"] is None and verified["skipped"]
    record = record_of(outcome)
    assert record["checks"]["unchecked_phases"] == [1]
    assert record["phases"][1]["unchecked"].startswith("effect check: not run")
    assert record["verifications"] == []
    assert r.operator.shown[-1].verified is None


def test_preconditions_no_camera_can_settle_are_recorded_unchecked_when_the_check_is_on(loop_for):
    plan = proposal(TAKE_OFF, open_the_box(preconditions=[("HandEmpty", [])], delete=[]), PUT_IN)
    record = record_of(loop_for(plan, {OPEN: True}, check_human_preconditions=True).run())
    assert record["checks"]["unchecked_phases"] == [1]
    assert record["phases"][1]["unchecked"].startswith("human phase precondition check: not run")
    # Off, nothing was going to be checked, so nothing went unchecked.
    assert record_of(loop_for(plan, {OPEN: True}).run())["checks"]["unchecked_phases"] == []


# --- 4. a robot phase handed to a person ----------------------------------------------------------


def test_a_handed_over_robot_phase_says_so_in_the_record(loop_for, monkeypatch):
    r = loop_for(
        PLAN,
        {OPEN: True, CLOSED: False, "blue_toy is resting on top of table": True},
        on_robot_phase_failure="teleop",
    )
    real_plan = r.backend.plan
    failures = iter(["no collision-free grasp on blue_toy"])

    def plan_once_failing(*args, **kwargs):
        reason = next(failures, None)
        if reason is not None:
            from tandem.planners.base import PlanResult

            return PlanResult(ok=False, failure_reason=reason)
        return real_plan(*args, **kwargs)

    monkeypatch.setattr(r.backend, "plan", plan_once_failing)
    outcome = r.run()

    record = record_of(outcome)
    handed, proposed = record["phases"][0], record["phases"][1]
    assert handed["executor"] == "human"
    assert handed["proposed_executor"] == "robot"
    assert handed["handed_over_because"] == "no collision-free grasp on blue_toy"
    assert handed["planned_by"] != "vlm" and "could not plan it" in handed["planned_by"]
    assert handed["instructions_by"] == "tandem"
    assert record["handed_over_phases"] == [0]
    # The model's own human phase is not marked.
    assert "proposed_executor" not in proposed and proposed["planned_by"] == "vlm"
    (failed,) = r.sink.named("phase_plan_failed")
    assert failed["phase_index"] == 0


# --- 9. one spelling for every operator signature -------------------------------------------------


def test_every_declared_spelling_is_recorded_one_way_and_reads_back():
    caps = dataclasses.replace(
        CAPS, robot_operators=("Pick(?obj:movable)", "Place( ?obj : movable,?surface: surface)")
    )
    assert capability_problems(caps) == [], "the SDK accepts both"
    spec = parse(PLAN)
    record = PhasePlan(cfg=CFG, caps=caps, instruction="x", trajectory_id="t", spec=spec).to_json()
    signatures = record["provenance"]["robot_operators"]["signatures"]
    assert signatures == ["Pick(obj: movable)", "Place(obj: movable, surface: surface)"]
    for signature in signatures:
        name = signature.split("(", 1)[0]
        args = ["a"] * signature.count(":")
        assert (
            HumanOperator.from_json({"name": name, "args": args, "signature": signature}).signature
            == signature
        )
    # The human side's spelling, which was already this one, is unchanged.
    assert record["provenance"]["human_operators"]["signatures"] == ["Open(x0: surface)"]


def test_a_signature_that_does_not_read_is_recorded_as_written():
    assert operator_signature("Pick (?obj: movable)") == "Pick (?obj: movable)"
