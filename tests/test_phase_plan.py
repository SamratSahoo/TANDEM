"""The phase plan as a cursor and as a record: which leg runs next, what it may touch, what it leaves.

No robot, no GPU, no network. A plan is parsed from the JSON a proposer would send, and walked by
hand. The running example is the planning suite's three-phase task -- the toy off the box, the box
opened by a person, the toy into the box -- plus two more: the sorting task, whose two robot phases
share one leg, and the screwdriver task, whose human phase uses a tool the robot must never pick up.

Ported in part from the suite that lived inside the planner fork (the last-leg, expected-now,
checks-on-the-record and screwdriver tests). What is new is what only a planner-agnostic record
needs: what a robot phase moves comes from the backend's declaration, both operator lists are
written in one spelling, and the record says how the trial ended.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from unittest import mock

import pytest
from helpers import FakeGemini

from tandem.core import episodes
from tandem.core.errors import TandemError
from tandem.planners import registry
from tandem.planners.base import Capabilities, PlanResult
from tandem.planning import feasibility, grounding, llm, proposal
from tandem.planning.config import PlanningConfig
from tandem.planning.grounding import Verdict
from tandem.planning.plan import (
    FAILURE_STAGES,
    OUTCOMES,
    PhasePlan,
    build_plan,
    operator_signature,
    phase_summary,
    retry_message,
)
from tandem.planning.proposal import parse_plan_response
from tandem.planning.structs import HumanOperator, Phase, SceneTypes, TaskSpecification, VLMPredicate
from tandem.planning.symbols import Atom, Parameter, Predicate, ProposalError

CAPS = registry.capabilities("tiptop")
CFG = PlanningConfig(enabled=True)
OBJECTS = ["blue_toy", "white_box"]
TABLE = "table"


def _atoms(*pairs):
    return [{"predicate": name, "args": list(args)} for name, args in pairs]


def _op(name, args, *, pre=(), add=(), dele=()):
    """A human phase's `operator` entry, as the proposer writes it."""
    return {
        "name": name,
        "args": list(args),
        "preconditions": _atoms(*pre),
        "add_effects": _atoms(*add),
        "delete_effects": _atoms(*dele),
    }


PLAN_RESPONSE = {
    "new_predicates": [
        {"name": "IsOpen", "instructions": "the container {0} is open, so its interior is visible"}
    ],
    "phases": [
        {
            "executor": "robot",
            "description": "take the toy off the box and put it on the table",
            "atoms": _atoms(("On", ["blue_toy", "table"])),
        },
        {
            "executor": "human",
            "description": "open the box",
            "instructions": "Open the white_box and fold its flaps back.",
            "atoms": _atoms(("IsOpen", ["white_box"])),
            "operator": _op("Open", ["white_box"], pre=[("HandEmpty", [])], add=[("IsOpen", ["white_box"])]),
        },
        {
            "executor": "robot",
            "description": "put the toy inside the box",
            "atoms": _atoms(("On", ["blue_toy", "white_box"])),
        },
    ],
}

# Two robot phases back to back, then a human one: the two sort into ONE leg.
SORT_OBJECTS = ["blue_bowl", "blue_cloth", "blue_toy", "green_bowl", "green_toy"]
SORT_RESPONSE = {
    "new_predicates": [{"name": "AreCoveredBy", "instructions": "the {2} is draped over both {0} and {1}"}],
    "phases": [
        {
            "executor": "robot",
            "description": "put the blue toy in the blue bowl",
            "atoms": _atoms(("On", ["blue_toy", "blue_bowl"])),
        },
        {
            "executor": "robot",
            "description": "put the green toy in the green bowl",
            "atoms": _atoms(("On", ["green_toy", "green_bowl"])),
        },
        {
            "executor": "human",
            "description": "cover both bowls with the cloth",
            "instructions": "Drape the blue_cloth over both bowls.",
            "atoms": _atoms(("AreCoveredBy", ["blue_bowl", "green_bowl", "blue_cloth"])),
            "operator": _op(
                "Cover",
                ["blue_cloth"],
                pre=[("HandEmpty", [])],
                add=[("AreCoveredBy", ["blue_bowl", "green_bowl", "blue_cloth"])],
            ),
        },
    ],
}

# The screwdriver task (LJ 9_pp_screwdriver_jenga), minus the deferred object tandem does not port:
# the person pushes a block out of the tower WITH THE SCREWDRIVER, and the robot puts the block back
# on top. Perception detects the screwdriver because the instruction names it.
JENGA_OBJECTS = ["jenga_tower", "screwdriver", "white_paper", "wooden_block"]
JENGA_RESPONSE = {
    "phases": [
        {
            "executor": "human",
            "description": "push a block out of the tower onto the paper",
            "instructions": "Push a block out of the jenga_tower with the screwdriver onto the white_paper.",
            "atoms": _atoms(("On", ["wooden_block", "white_paper"])),
            "operator": _op(
                "PushOut",
                ["wooden_block", "screwdriver"],
                pre=[("HandEmpty", [])],
                add=[("On", ["wooden_block", "white_paper"])],
            ),
        },
        {
            "executor": "robot",
            "description": "put the block on top of the tower",
            "atoms": _atoms(("On", ["wooden_block", "jenga_tower"])),
        },
    ],
}


def _open_close():
    """Three human phases; the middle one CLOSES the box, deleting what the first one opened."""
    return {
        "new_predicates": [
            {"name": "IsOpen", "instructions": "the container {0} is open"},
            {"name": "IsClosed", "instructions": "the container {0} is shut"},
        ],
        "phases": [
            {
                "executor": "human",
                "description": "open the box",
                "instructions": "open it",
                "atoms": _atoms(("IsOpen", ["white_box"])),
                "operator": _op("Open", ["white_box"], add=[("IsOpen", ["white_box"])]),
            },
            {
                "executor": "human",
                "description": "close the box",
                "instructions": "shut it",
                "atoms": _atoms(("IsClosed", ["white_box"])),
                "operator": _op(
                    "Close",
                    ["white_box"],
                    pre=[("IsOpen", ["white_box"])],
                    add=[("IsClosed", ["white_box"])],
                    dele=[("IsOpen", ["white_box"])],
                ),
            },
            {
                "executor": "human",
                "description": "drop the toy in",
                "instructions": "put it in",
                "atoms": _atoms(("On", ["blue_toy", "white_box"])),
                "operator": _op(
                    "Insert",
                    ["blue_toy", "white_box"],
                    pre=[("IsOpen", ["white_box"])],
                    add=[("On", ["blue_toy", "white_box"])],
                ),
            },
        ],
    }


def parse(response=None, objects=OBJECTS, caps=CAPS):
    """Parse a reply the way the proposal stage does; the parser builds each human phase's operator."""
    return parse_plan_response(response or PLAN_RESPONSE, "do the thing", objects, TABLE, caps)


def walk(response=None, objects=OBJECTS, *, caps=CAPS, cfg=CFG):
    spec = parse(response, objects, caps)
    return PhasePlan(cfg=cfg, caps=caps, instruction=spec.instruction, trajectory_id="traj-1", spec=spec)


def shown(atoms):
    return {str(a) for a in atoms}


# --- which leg is the last one --------------------------------------------------------------------


def test_only_the_last_leg_of_a_task_ends_by_driving_the_arm_home():
    # A planner ends every plan at home. That is right for a plan that IS the episode and wrong for a
    # leg of one: the bug this guards had the arm drive home after the robot phase of "place the
    # bread on the plate, then open the box and place the bread in the box" -- a return to home
    # recorded into the middle of the demonstration, and the person then handed an arm parked at
    # home rather than where the plan stopped. is_last_leg is what goes to plan(return_home=).
    plan = walk()  # robot, human, robot
    assert not plan.is_last_leg(), "two phases still follow this one"
    plan.advance()
    assert not plan.is_last_leg(), "the human phase is followed by a robot one"
    plan.advance()
    assert plan.is_last_leg(), "the final robot phase parks the arm"
    plan.advance()
    assert plan.finished and plan.is_last_leg(), "a finished plan answers conservatively"


def test_the_last_leg_is_the_whole_run_of_robot_phases_it_covers():
    # A leg can cover several phases (robot_run), so "is this the last phase" is the wrong question:
    # the run below covers phases 0 and 1 of 3, and only the leg after it ends the task.
    plan = walk(SORT_RESPONSE, SORT_OBJECTS)  # robot, robot, human
    assert len(plan.robot_run()) == 2
    assert not plan.is_last_leg(), "the human phase still follows the pair"
    plan.advance()
    assert plan.index == 2 and plan.is_last_leg()


def test_the_final_phase_is_a_question_about_one_phase_not_a_leg():
    plan = walk(SORT_RESPONSE, SORT_OBJECTS)
    # The first leg covers phase 1 too, and is still not the final PHASE.
    assert not plan.is_final_phase()
    plan.advance()
    assert plan.is_final_phase(), "the cover phase is the last one"
    plan.advance()
    assert plan.finished and not plan.is_final_phase(), "a finished plan has no current phase"


def test_without_conjoining_every_robot_phase_is_a_leg_of_its_own():
    # conjoin_robot_phases: false is the paper's "re-perceive after every phase" read strictly.
    plan = walk(SORT_RESPONSE, SORT_OBJECTS, cfg=PlanningConfig(enabled=True, conjoin_robot_phases=False))
    assert [p.description for p in plan.robot_run()] == ["put the blue toy in the blue bowl"]
    assert [a.to_dict() for a in plan.goal()] == [{"predicate": "on", "args": ["blue_toy", "blue_bowl"]}]
    assert not plan.is_last_leg()
    plan.advance()
    assert plan.index == 1, "one phase per leg"
    assert [p.description for p in plan.robot_run()] == ["put the green toy in the green bowl"]
    assert not plan.is_last_leg(), "the human phase still follows"
    plan.advance()
    assert plan.next_is_human() and plan.robot_run() == ()


# --- what a leg is allowed to conjoin -------------------------------------------------------------

# A goal language in which the thing placed ONTO is itself something the robot can pick up: blocks
# stacked on blocks. What a phase moves is the TOP argument only.
STACKED = Predicate("Stacked", (Parameter("top", "block"), Parameter("bottom", "block")))
STACKER = Capabilities(
    name="stacker",
    goal_predicates={"Stacked": STACKED},
    robot_description="stack one block on another",
    goal_predicate_wire_names={"Stacked": "stacked"},
    achievable_predicates=frozenset({"Stacked"}),
    reserved_predicate_names=frozenset({"Stacked"}),
    movable_type="block",
    surface_type="block",
    moved_arguments={"Stacked": 0},
    one_pick_per_object=True,
    # Stated, not inherited: a run is conjoined only for a planner that promises a clean state, and
    # these tests are about what splits a run that could otherwise be one.
    initial_state_is_clean=True,
)


def _stack(top, bottom):
    return Phase("robot", f"stack {top} on {bottom}", frozenset({Atom("Stacked", (top, bottom))}))


def test_a_run_is_split_by_what_its_phases_move_not_by_what_they_name():
    # Both phases name `base`; neither moves it. Deciding by the names split this into two legs --
    # a perception pass and a stop in the middle, for a plan that picks nothing twice.
    phases = [_stack("a", "base"), _stack("b", "base")]
    assert feasibility.conjoinable_run(phases, STACKER) == 2
    # Moving `a` a second time IS a second pick of it, and one plan picks each object once.
    assert feasibility.conjoinable_run([_stack("a", "base"), _stack("a", "b")], STACKER) == 1


def test_a_backend_that_does_not_say_what_moves_is_assumed_to_move_everything_named():
    # The missing declaration is read conservatively: splitting costs a perception pass, while
    # conjoining two picks of one object costs the plan.
    silent = dataclasses.replace(STACKER, moved_arguments={})
    assert feasibility.conjoinable_run([_stack("a", "base"), _stack("b", "base")], silent) == 1


def test_two_toys_placed_on_one_table_still_share_a_leg():
    # TipTop's declaration: the toy moves, the surface does not.
    phases = [
        Phase("robot", "a on table", frozenset({Atom("On", ("toy_a", "table"))})),
        Phase("robot", "b on table", frozenset({Atom("On", ("toy_b", "table"))})),
    ]
    assert feasibility.conjoinable_run(phases, CAPS) == 2


# --- what the robot may pick up -------------------------------------------------------------------


def test_the_humans_screwdriver_is_never_something_the_robot_picks_up():
    # Observed on the rig, 9_pp_screwdriver_jenga: perception detects the screwdriver because the
    # INSTRUCTION names it, every non-surface detection is a movable, and so three of the four
    # skeletons cuTAMP enumerated for "put the block back on the tower" opened with
    # Pick(screwdriver) -- one of them placing it on the tower. No robot phase ever moves it: it is
    # the human's tool, and the plan already says so.
    plan = walk(JENGA_RESPONSE, JENGA_OBJECTS)
    assert plan.robot_movables() == {"wooden_block"}
    assert "screwdriver" in plan.spec.scene_types.movables, "still a movable to the SPEC..."
    assert "screwdriver" in plan.phases[0].operator.args, "...and named by the human's operator"


def test_the_surface_a_robot_phase_places_onto_is_not_offered_as_a_pick():
    plan = walk()
    assert plan.robot_movables() == {"blue_toy"}, "the box is placed INTO, never picked"


def test_the_movables_span_every_robot_phase_not_just_the_next_leg():
    # An object pickable in one leg and an obstacle in the next would change what the search may
    # do partway through one task.
    response = {
        "new_predicates": PLAN_RESPONSE["new_predicates"],
        "phases": [
            {
                "executor": "robot",
                "description": "toy onto the table",
                "atoms": _atoms(("On", ["blue_toy", "table"])),
            },
            PLAN_RESPONSE["phases"][1],
            {
                "executor": "robot",
                "description": "cup into the box",
                "atoms": _atoms(("On", ["cup", "white_box"])),
            },
        ],
    }
    plan = walk(response, [*OBJECTS, "cup"])
    first = plan.robot_movables()
    assert first == {"blue_toy", "cup"}
    plan.advance()
    plan.advance()
    assert plan.robot_movables() == first, "the same set on every leg"


def test_a_plan_with_no_robot_phase_offers_nothing_to_pick():
    assert walk(_open_close()).robot_movables() == frozenset()


def test_a_backend_that_cannot_say_what_moves_is_refused_rather_than_told_to_pick_nothing():
    # An empty set handed to plan(movables=) forbids every pick. That is not an answer to give quietly.
    caps = dataclasses.replace(CAPS, moved_arguments={})
    with pytest.raises(TandemError, match="moved_arguments") as excinfo:
        walk(caps=caps).robot_movables()
    assert "supports_movable_restriction" in excinfo.value.hint


# --- what a robot leg inherits --------------------------------------------------------------------


def test_a_robot_leg_inherits_only_what_earlier_phases_established():
    # The robot-side precondition set: every atom an earlier phase was responsible for and no later
    # one undid. Both of these are beliefs that can have gone stale by now -- the human phase may
    # not have opened the box it was verified as opening, and it may have knocked the toy off the
    # table while it was at it.
    plan = walk()
    plan.index = 2  # the last robot phase, after the human one
    assert shown(plan.expected_now()) == {"On(blue_toy, table)", "IsOpen(white_box)"}
    # HandEmpty() is nobody's add effect -- it is the human operator's PRECONDITION -- so no earlier
    # phase can be held to it and it is not put to a camera here.
    assert "HandEmpty()" not in shown(plan.expected_now())


def test_nothing_is_inherited_before_the_first_phase_runs():
    assert walk().expected_now() == frozenset()


def test_a_finished_plan_expects_nothing():
    plan = walk()
    while not plan.finished:
        plan.advance()
    assert plan.expected_now() == frozenset()


# --- what the planner found -----------------------------------------------------------------------


def test_the_task_plan_the_planner_ran_is_on_the_record():
    plan = walk(SORT_RESPONSE, SORT_OBJECTS)
    result = PlanResult(
        ok=True,
        planning_seconds=3.25,
        task_plan=(
            "Pick(blue_toy)",
            "Place(blue_toy, blue_bowl)",
            "Pick(green_toy)",
            "Place(green_toy, green_bowl)",
        ),
    )
    plan.record_plan(0, result)
    record = plan.to_json()
    # One plan for both phases: each records it, and says it was shared.
    for phase in record["phases"][:2]:
        assert phase["task_plan"][:2] == ["Pick(blue_toy)", "Place(blue_toy, blue_bowl)"]
        assert phase["covers_phases"] == [0, 1]
        assert phase["planning_seconds"] == 3.25
    assert "task_plan" not in record["phases"][2], "a human phase has no planner record"
    # Each phase got its own copy.
    plan.plans[0]["task_plan"].append("GoHome()")
    assert "GoHome()" not in plan.plans[1]["task_plan"]


def test_a_planner_that_did_not_say_what_it_ran_records_no_task_plan():
    # Not an empty list, which would claim the plan had no operators in it.
    plan = walk()
    plan.record_plan(0, PlanResult(ok=True, planning_seconds=1.0))
    assert "task_plan" not in plan.to_json()["phases"][0]
    # A stand-in result with no such attribute is the same planner that did not say.
    plan.record_plan(0, mock.Mock(planning_seconds=1.0, skeleton_reused=False))
    assert "task_plan" not in plan.to_json()["phases"][0]


# --- the record -----------------------------------------------------------------------------------


def test_which_halves_of_the_contract_were_checked_is_on_the_record():
    # A phase with no verdicts is otherwise ambiguous between "checked and fine" and "never checked".
    checks = walk().to_json()["checks"]
    assert checks["human_effects"] is True
    assert checks["human_preconditions"] is False
    assert checks["tamp_preconditions"] is False
    assert checks["tamp_effects"] is False
    assert checks["plan_effects"] is True
    assert checks["verify_final_phase"] is True
    assert checks["precondition_enforced"] is False
    # Nothing measured the start, so nothing was re-checked against it.
    assert checks["initial_state_classified"] is False
    assert checks["plan_effects_rechecked"] is False
    assert checks["plan_effects_warning"] is None
    assert checks["unchecked_phases"] == []


def test_the_record_follows_the_configuration():
    cfg = PlanningConfig(enabled=True, check_human_preconditions=True, verify_final_phase=False)
    checks = walk(cfg=cfg).to_json()["checks"]
    assert checks["human_preconditions"] is True
    assert checks["verify_final_phase"] is False


def test_a_phase_accepted_unchecked_is_told_apart_from_one_that_passed():
    plan = walk()
    plan.record_unchecked(1, "the camera did not answer")
    record = plan.to_json()
    assert record["checks"]["unchecked_phases"] == [1]
    assert record["phases"][1]["unchecked"] == "the camera did not answer"
    assert "unchecked" not in record["phases"][0]


def test_both_operator_lists_are_written_in_one_spelling():
    # The human operators come from HumanOperator.signature, the robot's from the backend's
    # declaration, which writes its parameters PDDL-style. One record, one spelling: the one the
    # record is read back with.
    provenance = walk().to_json()["provenance"]
    assert provenance["human_operators"]["signatures"] == ["Open(x0: surface)"]
    assert provenance["robot_operators"]["signatures"] == [
        "Pick(obj: movable)",
        "Place(obj: movable, surface: surface)",
    ]
    assert CAPS.robot_operators[0].startswith("Pick(?"), "the declaration itself is left as it is"
    assert provenance["human_operators"]["by"].startswith("vlm")
    assert provenance["robot_operators"]["by"].startswith("tiptop")
    # The spelling HumanOperator.from_json reads: the robot's signatures parse into clean
    # parameters, not ones named "?obj".
    read = HumanOperator.from_json(
        {"name": "Place", "args": ["t", "b"], "signature": "Place(obj: movable, surface: surface)"}
    )
    assert [p.name for p in read.parameters] == ["obj", "surface"]


def test_operator_signature_writes_one_spelling():
    # Read and written out again, so every spacing the planner SDK's check accepts comes out the
    # same -- not just the `?` removed (tests/test_review_method.py has the record-level check).
    assert operator_signature("Pick(?obj: movable)") == "Pick(obj: movable)"
    assert operator_signature("Place(?a: movable,?b: surface)") == "Place(a: movable, b: surface)"
    assert operator_signature("Pick(?obj:movable)") == "Pick(obj: movable)"
    assert operator_signature("Pick( ?obj : movable )") == "Pick(obj: movable)"
    assert operator_signature("Open(x0: surface)") == "Open(x0: surface)"
    assert operator_signature("Wave()") == "Wave()"


def test_the_same_human_operator_is_listed_once():
    response = {
        "new_predicates": [{"name": "IsOpen", "instructions": "the container {0} is open"}],
        "phases": [
            {
                "executor": "human",
                "description": f"open the {box}",
                "instructions": "open it",
                "atoms": _atoms(("IsOpen", [box])),
                "operator": _op("Open", [box], add=[("IsOpen", [box])]),
            }
            for box in ("white_box", "red_box")
        ],
    }
    plan = walk(response, ["white_box", "red_box"])
    record = plan.to_json()
    assert record["provenance"]["human_operators"]["signatures"] == ["Open(x0: movable)"]
    # The instances stay per phase.
    assert [p["operator"]["instance"] for p in record["phases"]] == ["Open(white_box)", "Open(red_box)"]


def test_each_human_phase_record_carries_its_operator():
    record = walk().to_json()
    assert record["phases"][1]["operator"]["instance"] == "Open(white_box)"
    assert record["phases"][1]["operator"]["preconditions"] == ["HandEmpty()"]
    assert "operator" not in record["phases"][0], "a robot phase's operators are the planner's"


def test_a_verdict_is_recorded_with_what_was_expected_and_the_phase_it_was_about():
    plan = walk(_open_close())
    closed = Verdict(Atom("IsClosed", ("white_box",)), "the container white_box is shut", True, "lid on")
    still_open = Verdict(
        Atom("IsOpen", ("white_box",)),
        "the container white_box is open",
        True,
        "a flap is up",
        expected=False,
        role="effect (deleted)",
    )
    plan.record_verdicts(1, [closed, still_open])
    entries = plan.to_json()["verifications"]
    assert [e["phase"] for e in entries] == [1, 1]
    assert (entries[1]["expected"], entries[1]["holds"], entries[1]["satisfied"]) == (False, True, False)
    assert entries[1]["role"] == "effect (deleted)"
    assert entries[0]["satisfied"] is True
    # Appended directly, as older callers do: kept, just without a phase.
    plan.verdicts.append(closed)
    assert "phase" not in plan.to_json()["verifications"][2]


def test_how_the_trial_ended_is_on_the_record():
    plan = walk()
    record = plan.to_json()
    assert (record["outcome"], record["failure_stage"], record["excluded"]) == (None, None, False)

    plan.set_outcome("excluded", "verification")
    record = plan.to_json()
    assert (record["outcome"], record["failure_stage"], record["excluded"]) == (
        "excluded",
        "verification",
        True,
    )


def test_an_outcome_outside_the_known_ones_is_refused_with_the_nearest_name():
    plan = walk()
    with pytest.raises(ValueError, match="'excluded'"):
        plan.set_outcome("exclude")
    with pytest.raises(ValueError, match="'tamp_planning'"):
        plan.set_outcome("failure", "tamp_plan")
    with pytest.raises(ValueError, match="no failure stage"):
        plan.set_outcome("success", "verification")
    assert plan.outcome is None, "a refused outcome changes nothing"
    assert set(OUTCOMES) == {"success", "failure", "excluded", "aborted"}
    assert "human_policy" in FAILURE_STAGES


def test_the_record_names_who_carried_out_the_human_phases():
    assert walk().to_json()["provenance"]["human_steps"].startswith("the teleoperator")
    other = walk(cfg=PlanningConfig(enabled=True, human_executor="diffusion"))
    record = other.to_json()
    assert record["human_executor"] == "diffusion"
    assert "'diffusion' human executor" in record["provenance"]["human_steps"]


def test_the_record_survives_json():
    plan = walk()
    plan.record_plan(0, PlanResult(ok=True, task_plan=("Pick(blue_toy)", "Place(blue_toy, table)")))
    plan.record_verdicts(1, [Verdict(Atom("IsOpen", ("white_box",)), "open", True, "")])
    plan.set_outcome("failure", "tamp_execution")
    assert json.loads(json.dumps(plan.to_json()))["phases"][0]["task_plan"][0] == "Pick(blue_toy)"


# --- the human phase as the session shows it ------------------------------------------------------


def test_the_phase_event_carries_the_operator_and_whether_it_is_the_last():
    plan = walk()
    plan.advance()
    summary = phase_summary(plan, plan.current)
    assert summary["operator"]["instance"] == "Open(white_box)"
    assert summary["is_last_phase"] is False, "a robot phase still follows"

    sort = walk(SORT_RESPONSE, SORT_OBJECTS)
    sort.advance()
    assert phase_summary(sort, sort.current)["is_last_phase"] is True


def test_the_phase_event_keeps_what_must_hold_apart_from_what_must_stop_holding():
    # One list would state the delete effect as something to bring about -- the opposite of the step.
    plan = walk(_open_close())
    plan.advance()
    summary = phase_summary(plan, plan.current)
    assert summary["expected_atoms"] == ["IsClosed(white_box)"]
    assert summary["expected_deleted_atoms"] == ["IsOpen(white_box)"]
    assert any(line.startswith("NO LONGER:") for line in summary["expected"])


def test_a_phase_without_an_operator_expects_its_own_atoms():
    # A robot phase handed to a person carries no operator, and falls back to its atoms.
    plan = walk()
    handed = plan.hand_current_to_human()
    summary = phase_summary(plan, handed)
    assert summary["expected_atoms"] == ["On(blue_toy, table)"]
    assert summary["expected_deleted_atoms"] == []
    assert "operator" not in summary


def test_the_retry_line_speaks_to_whoever_gets_another_go():
    person = retry_message(["the box is open"], 1)
    assert "Take the arm again" in person
    policy = retry_message(["the box is open"], 2, by="diffusion")
    assert "Take the arm" not in policy
    assert "Running the diffusion executor on this phase again (2 attempt(s) left)." in policy
    # Nothing to say to anyone once there is no other go.
    last = retry_message(["the box is open"], 0, by="diffusion")
    assert "again" not in last and "the box is open" in last


# --- building the plan ----------------------------------------------------------------------------


def _image():
    from PIL import Image

    return Image.new("RGB", (32, 32), "white")


def test_feedback_reaches_the_model_and_never_touches_the_cache(tmp_path, monkeypatch):
    # `replan` was abort under another name while the failure was not fed back: the same question of
    # the same scene gets the same plan. And a cached answer would replay it without even asking.
    client = FakeGemini(json.dumps(PLAN_RESPONSE))
    monkeypatch.setattr(llm, "gemini_client", lambda: client)
    cfg = PlanningConfig(enabled=True, cache_path=str(tmp_path / "cache.sqlite"))
    image = _image()

    def build(**kwargs):
        plan = asyncio.run(
            build_plan(image, "put the toy in the box", OBJECTS, TABLE, cfg, CAPS, "t", **kwargs)
        )
        # The plan itself, never a (plan, failure) pair: a failure is only ever raised.
        assert isinstance(plan, PhasePlan)
        return plan

    build()
    build()
    assert client.plan_calls == 1, "the second proposal came from the cache"

    reason = "phase 2 could not be planned: no collision-free placement inside white_box"
    build(feedback=reason)
    assert client.plan_calls == 2, "a proposal with feedback always asks the model"
    assert reason in client.prompts[-1]
    assert "could not be carried out" in client.prompts[-1]
    build(feedback=reason)
    assert client.plan_calls == 3, "and is never cached, so the same feedback asks again"

    build()
    assert client.plan_calls == 3, "the ordinary entry is still the one the cache holds"
    assert reason not in client.prompts[0], "and the ordinary prompt never carried the feedback"


def test_blank_feedback_is_no_feedback(monkeypatch):
    client = FakeGemini(json.dumps(PLAN_RESPONSE))
    monkeypatch.setattr(llm, "gemini_client", lambda: client)
    asyncio.run(build_plan(_image(), "x", OBJECTS, TABLE, CFG, CAPS, None, feedback="   "))
    assert "could not be carried out" not in client.prompts[-1]


def test_a_robot_phase_nothing_can_achieve_is_still_refused_now_the_repair_loop_checks_it(monkeypatch):
    # build_plan no longer runs feasibility.check_robot_phases itself: the proposal's parse closure
    # does (proposal.check_plan), so the model is told why and gets another go. Pinned across that
    # seam, feedback included, because each side alone would pass with the check in neither place --
    # and the planner would then be handed a goal no robot operator can reach, and search forever.
    no_holding = dataclasses.replace(CAPS, achievable_predicates=CAPS.achievable_predicates - {"Holding"})
    hold_the_toy = {
        "phases": [
            {
                "executor": "robot",
                "description": "pick up the toy",
                "atoms": [{"predicate": "Holding", "args": ["blue_toy"]}],
            }
        ]
    }
    client = FakeGemini(json.dumps(hold_the_toy))
    monkeypatch.setattr(llm, "gemini_client", lambda: client)
    reason = "phase 0 could not be planned: no collision-free placement"
    with pytest.raises(ProposalError, match="This plan cannot be carried out"):
        asyncio.run(build_plan(_image(), "x", OBJECTS, TABLE, CFG, no_holding, None, feedback=reason))
    assert client.plan_calls == CFG.max_attempts
    assert all(reason in prompt for prompt in client.prompts), "the replan section rides every repair"
    assert "Holding(blue_toy), which no robot operator can achieve" in client.prompts[-1]


def _needs_an_unlocked_box():
    """Open(white_box) requires IsUnlocked(white_box), which no phase makes true.

    Built directly rather than parsed: IsUnlocked is named only in a precondition, and whether that
    counts as a use of an invented predicate is the parser's business, not this test's.
    """
    is_open = VLMPredicate(Predicate("IsOpen", (Parameter("x0", "movable"),)), "the container {0} is open")
    unlocked = VLMPredicate(Predicate("IsUnlocked", (Parameter("x0", "movable"),)), "the box {0} is unlocked")
    opened = Atom("IsOpen", ("white_box",))
    operator = HumanOperator(
        name="Open",
        args=("white_box",),
        parameters=(Parameter("x0", "movable"),),
        preconditions=frozenset({Atom("IsUnlocked", ("white_box",))}),
        add_effects=frozenset({opened}),
    )
    return TaskSpecification(
        instruction="open the box",
        phases=(Phase("human", "open the box", frozenset({opened}), "open it", operator),),
        scene_types=SceneTypes(surfaces=frozenset({TABLE}), movables=frozenset(OBJECTS)),
        invented=(is_open, unlocked),
    )


def _built(monkeypatch, spec, *, measured, **cfg):
    async def fake_propose(*args, **kwargs):
        return spec

    async def fake_classify(image, spec, cfg, caps):
        return frozenset(measured)

    monkeypatch.setattr(proposal, "propose_plan", fake_propose)
    monkeypatch.setattr(grounding, "classify_initial_state", fake_classify)
    config = PlanningConfig(enabled=True, **cfg)
    return asyncio.run(build_plan(None, "open the box", OBJECTS, TABLE, config, CAPS, "t"))


def test_a_plan_the_measured_scene_contradicts_runs_with_the_gap_on_the_record(monkeypatch, caplog):
    # The measurement found nothing true -- which is exactly when a never-established precondition
    # is provable. Keying the re-check on "something was found true" would skip this very case.
    with caplog.at_level(logging.WARNING, logger="tandem.planning.plan"):
        plan = _built(monkeypatch, _needs_an_unlocked_box(), measured=(), classify_initial=True)
    assert plan is not None, "a warning, not a refusal: the proposer is out of the loop by now"
    assert plan.initial_state_known and plan.plan_effects_rechecked
    assert "IsUnlocked(white_box)" in plan.inconsistency
    assert any("does not hang together" in r.getMessage() for r in caplog.records)
    checks = plan.to_json()["checks"]
    assert checks["initial_state_classified"] is True
    assert "IsUnlocked(white_box)" in checks["plan_effects_warning"]


def test_a_scene_that_satisfies_the_precondition_leaves_nothing_to_report(monkeypatch):
    plan = _built(
        monkeypatch,
        _needs_an_unlocked_box(),
        measured={Atom("IsUnlocked", ("white_box",))},
        classify_initial=True,
    )
    assert plan.plan_effects_rechecked and plan.inconsistency is None
    assert plan.initially_true == {Atom("IsUnlocked", ("white_box",))}


def test_nothing_is_rechecked_without_a_measurement(monkeypatch):
    plan = _built(monkeypatch, _needs_an_unlocked_box(), measured=())
    assert not plan.initial_state_known and not plan.plan_effects_rechecked
    assert plan.inconsistency is None


def test_the_recheck_is_off_with_the_check_it_repeats(monkeypatch):
    plan = _built(
        monkeypatch, _needs_an_unlocked_box(), measured=(), classify_initial=True, check_plan_effects=False
    )
    assert plan.initial_state_known and not plan.plan_effects_rechecked


# --- the record on disk ---------------------------------------------------------------------------


def test_an_excluded_trial_says_so_whatever_it_was_filed_under(tmp_path):
    plan = walk()
    plan.set_outcome("excluded", "verification")
    episodes.write_phase_record(plan, tmp_path, status="failure", vlm_dir=None, log=print)
    record = json.loads((tmp_path / "hitl.json").read_text())
    assert (record["outcome"], record["excluded"], record["failure_stage"]) == (
        "excluded",
        True,
        "verification",
    )
    assert record["filed_under"] == "failure"


def test_a_trial_that_ran_to_the_end_takes_the_operators_label(tmp_path):
    episodes.write_phase_record(walk(), tmp_path, status="success", vlm_dir=None, log=print)
    record = json.loads((tmp_path / "hitl.json").read_text())
    assert (record["outcome"], record["excluded"], record["failure_stage"]) == ("success", False, None)


@pytest.mark.parametrize(
    ("loop", "stage", "filed", "outcome"),
    [
        ("excluded", "verification", "failure", "excluded"),  # no label was asked; the filing is not its verdict
        ("aborted", None, "failure", "aborted"),
        # on_verification_failure: label -- the operator overrules the check, which is what it is for.
        ("failure", "verification", "success", "success"),
        ("failure", None, "success", "success"),  # a record from before stages: the label decides
        ("failure", None, None, "failure"),  # never filed: the loop's own word stands
        (None, None, "failure", "failure"),
        (None, None, None, None),
        # A plan that did not finish is never a success, wherever it ends up filed; nor is an
        # aborted one. Only the directory ("filed_under") says where somebody put it.
        ("failure", "tamp_execution", "success", "failure"),
        ("failure", "tamp_planning", "success", "failure"),
        ("failure", "human_policy", "success", "failure"),
        ("aborted", None, "success", "aborted"),
        ("excluded", "verification", "success", "excluded"),
    ],
)
def test_the_outcome_is_the_loops_word_then_the_label(loop, stage, filed, outcome):
    fields = episodes.trial_outcome(loop, filed, failure_stage=stage)
    assert fields == {"outcome": outcome, "excluded": outcome == "excluded", "filed_under": filed}


def test_the_merge_files_the_record_with_where_the_episode_went(tmp_path, monkeypatch):
    from tandem.core import merge as merge_mod

    merged = tmp_path / "merged"
    merged.mkdir()
    monkeypatch.setattr(merge_mod, "find_legs", lambda *a, **k: [])
    monkeypatch.setattr(merge_mod, "merge", lambda *a, **k: {"merged": False, "dir": str(merged)})
    plan = walk()
    plan.set_outcome("excluded", "verification")
    episodes.merge_trajectory(
        object(), "traj-1", "failure", plan, tools_dir=None, vlm_dir=None, log=print, emit=lambda e: None
    )
    record = json.loads((merged / "hitl.json").read_text())
    assert record["filed_under"] == "failure" and record["excluded"] is True
