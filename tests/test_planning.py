"""Phase planning: proposal validation, the phase walk, and what reaches the planner.

No robot, no GPU, no network -- the model's job here is to produce JSON, so the tests drive the
parser with the JSON directly. The running example is the task that forced the phase model:
"pick the toy off the box and place it on the table, open the box, place the toy inside the box".

Ported from the suite that lived inside the planner fork. What is new here is everything that only
became possible once tandem owned the plan: the goal language coming from a declaration rather than
an import, a robot phase being handed to a person when it cannot be planned, and the phase event
carrying where in the plan it is.
"""

from __future__ import annotations

import asyncio
import json
import sys
from unittest import mock

import pytest

from tandem.planners import registry
from tandem.planners.base import to_goal_atoms
from tandem.planning import feasibility, grounding, llm
from tandem.planning.config import PlanningConfig
from tandem.planning.drift import match_drifted_names
from tandem.planning.grounding import Verdict
from tandem.planning.plan import PhasePlan, handoff_message, phase_summary
from tandem.planning.proposal import parse_plan_response
from tandem.planning.symbols import ProposalError, describe

CAPS = registry.capabilities("tiptop")
OBJECTS = ["blue_toy", "white_box"]
TABLE = "table"
CFG = PlanningConfig(enabled=True)

# The plan the proposer should produce for the three-phase task. The ordering is the whole point:
# the box is opened BEFORE anything is placed in it, which the previous design could not express.
PLAN_RESPONSE = {
    "new_predicates": [
        {"name": "IsOpen", "instructions": "the container {0} is open, so its interior is visible"}
    ],
    "phases": [
        {
            "executor": "robot",
            "description": "take the toy off the box and put it on the table",
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


def parse(response=None, objects=OBJECTS, instruction="do the thing"):
    return parse_plan_response(response or PLAN_RESPONSE, instruction, objects, TABLE, CAPS)


def _plan_response(**changes):
    """PLAN_RESPONSE with some part replaced."""
    return {**PLAN_RESPONSE, **changes}


def _phases(*phases):
    """A plan of just these phases, with no invented predicates left over to be unused."""
    return {"phases": list(phases)}


def goal_of(phase):
    return [a.to_dict() for a in to_goal_atoms(sorted(phase.atoms, key=str), CAPS)]


def _walk(spec=None, trajectory_id="traj-1"):
    spec = spec or parse()
    return PhasePlan(cfg=CFG, caps=CAPS, instruction=spec.instruction, trajectory_id=trajectory_id, spec=spec)


def test_the_prompt_actually_contains_the_instruction():
    # It did not, for one round of testing, and the model planned from the IMAGE alone -- inventing a
    # plausible-looking task for the scene and ignoring what was asked. Nothing else catches that:
    # the response parses, validates, and plans perfectly well; it is just answering another question.
    from tandem.planning.prompts import plan_prompt

    prompt = plan_prompt("open the box and put the toy in it", ["blue_toy", "white_box"], caps=CAPS)
    assert "open the box and put the toy in it" in prompt
    assert "blue_toy" in prompt and "white_box" in prompt


def test_the_goal_language_in_the_prompt_comes_from_the_backend():
    # The predicates shown to the proposer are the backend's declaration, not a constant in the
    # prompt. A backend whose goal language differs cannot silently be asked for predicates it does
    # not have -- which is the whole reason this layer no longer imports the planner's domain.
    menu = CAPS.predicate_menu()
    assert menu.splitlines() == [
        "- On(?obj: movable, ?surface: surface): {0} is resting on top of {1}",
        "- Holding(?obj: movable): the robot's gripper is holding {0}",
        "- HandEmpty(): the robot's gripper is empty",
    ]


# --- the phase plan -------------------------------------------------------------------------------


def test_a_human_phase_can_come_before_robot_work():
    # The defect that forced this design: ordering used to flow only one way, so "open the box, THEN
    # put the toy in" planned the placement first and the run ended after the human step.
    spec = parse()
    assert [p.executor for p in spec.phases] == ["robot", "human", "robot"]
    assert [p.description for p in spec.phases][1] == "open the box"
    assert spec.needs_human


def test_an_object_may_be_picked_up_in_more_than_one_phase():
    # One pick per object per PLAN is why the toy could not be moved out of the box and back in under
    # a single-plan design. Each phase is its own problem from a fresh perception pass, so the toy
    # appears in phases 0 and 2.
    spec = parse()
    assert goal_of(spec.phases[0]) == [{"predicate": "on", "args": ["blue_toy", "table"]}]
    assert goal_of(spec.phases[2]) == [{"predicate": "on", "args": ["blue_toy", "white_box"]}]


def test_the_scene_types_are_fixed_once_across_every_phase():
    # Inferring per phase would make white_box a surface in phase 2 and a movable in phase 0, so the
    # world geometry the planner plans against would change mid-task.
    spec = parse()
    assert spec.scene_types.surfaces == frozenset({"table", "white_box"})
    assert spec.scene_types.movables == frozenset({"blue_toy"})
    assert list(spec.invented[0].predicate.types) == ["surface"]


def test_a_robot_phase_cannot_be_asked_for_an_invented_predicate():
    # The rule that decides what is a human phase: the planner has no operator that can make an
    # invented predicate true, so a robot phase asking for one would plan forever and never arrive.
    response = _plan_response(
        phases=[
            {
                "executor": "robot",
                "description": "open the box",
                "atoms": [{"predicate": "IsOpen", "args": ["white_box"]}],
            }
        ]
    )
    with pytest.raises(ProposalError, match="A robot phase cannot achieve 'IsOpen'"):
        parse(response)


def test_every_robot_phase_is_checked_for_achievability():
    spec = parse()
    assert feasibility.check_robot_phases(spec, CAPS) is None
    assert feasibility.unachievable_atoms(spec.phases[0].atoms, CAPS) == []
    # An invented atom is unachievable by the robot; this is the guard that keeps it out of the
    # planner's unbounded search rather than discovering it there.
    assert feasibility.unachievable_atoms(spec.phases[1].atoms, CAPS) == sorted(spec.phases[1].atoms, key=str)


@pytest.mark.parametrize(
    "response, expected",
    [
        (_phases(), "at least one phase"),
        (
            _phases({"executor": "sidekick", "description": "x", "atoms": []}),
            "executor must be 'robot' or 'human'",
        ),
        (_phases({"executor": "robot", "description": "nothing", "atoms": []}), "has no atoms"),
        (
            _plan_response(
                phases=[
                    {
                        "executor": "human",
                        "description": "open it",
                        "atoms": [{"predicate": "IsOpen", "args": ["white_box"]}],
                    }
                ]
            ),
            "needs `instructions`",
        ),
        (
            _phases(
                {
                    "executor": "robot",
                    "description": "x",
                    "atoms": [{"predicate": "On", "args": ["blue_toy", "moon"]}],
                }
            ),
            "not an object in this scene",
        ),
        (
            _phases(
                {
                    "executor": "robot",
                    "description": "x",
                    "atoms": [{"predicate": "Sideways", "args": ["blue_toy"]}],
                }
            ),
            "Unknown predicate",
        ),
        (
            _phases(
                {
                    "executor": "robot",
                    "description": "x",
                    "atoms": [{"predicate": "On", "args": ["blue_toy"]}],
                }
            ),
            "takes 2 argument",
        ),
        (
            {**PLAN_RESPONSE, "new_predicates": [{"name": "On", "instructions": "{0} on {1}"}]},
            "already exists",
        ),
        # Reserved is broader than the goal language: a predicate the planner uses for its own
        # bookkeeping is taken even though no goal can be stated over it.
        (
            {**PLAN_RESPONSE, "new_predicates": [{"name": "HasNotPickedUp", "instructions": "{0}"}]},
            "already exists",
        ),
        (
            {
                **PLAN_RESPONSE,
                "new_predicates": [
                    *PLAN_RESPONSE["new_predicates"],
                    {"name": "Tidy", "instructions": "{0} tidy"},
                ],
            },
            "no phase uses it",
        ),
        (
            {**PLAN_RESPONSE, "new_predicates": [{"name": "IsOpen", "instructions": "{0} and {1}"}]},
            "placeholders",
        ),
        (
            {**PLAN_RESPONSE, "new_predicates": [{"name": "Is Open!", "instructions": "{0} open"}]},
            "not a valid predicate name",
        ),
    ],
)
def test_plan_rejections(response, expected):
    with pytest.raises(ProposalError, match=expected):
        parse(response)


def test_an_all_robot_plan_needs_no_human():
    # The degradation guarantee: a task the planner already handles produces one robot phase and
    # behaves exactly as it did before any of this existed.
    response = {
        "phases": [
            {
                "executor": "robot",
                "description": "put the toy in the box",
                "atoms": [{"predicate": "On", "args": ["blue_toy", "white_box"]}],
            }
        ]
    }
    spec = parse(response)
    assert not spec.needs_human
    assert feasibility.check_robot_phases(spec, CAPS) is None
    assert goal_of(spec.phases[0]) == [{"predicate": "on", "args": ["blue_toy", "white_box"]}]


def test_a_predicate_the_planner_supplies_itself_is_shown_but_not_sent():
    # HandEmpty is statable, so a proposal may say it and a human is told about it -- but the planner
    # adds it to every goal that holds nothing, so sending it would be noise. It is dropped on the
    # way out by the backend's declaration, not by a special case in the phase planner.
    spec = parse(
        _phases(
            {
                "executor": "robot",
                "description": "put the toy down and let go",
                "atoms": [
                    {"predicate": "On", "args": ["blue_toy", "white_box"]},
                    {"predicate": "HandEmpty", "args": []},
                ],
            }
        )
    )
    assert goal_of(spec.phases[0]) == [{"predicate": "on", "args": ["blue_toy", "white_box"]}]
    shown = grounding.describe_expectations(spec.phases[0], grounding.descriptions_for((), CAPS))
    assert any("gripper is empty" in line for line in shown)


def test_a_clause_that_cannot_be_expressed_is_reported_not_dropped():
    # Observed on the rig: "... pick ANOTHER toy, and place it in the box" against a scene holding
    # exactly one toy. The plan came back missing that clause and nothing said so, so the run did most
    # of the task and reported success.
    response = {
        **PLAN_RESPONSE,
        "unrepresented": [{"clause": "pick another toy", "reason": "only one toy was detected"}],
    }
    spec = parse(response)
    assert len(spec.phases) == 3, "the clauses that COULD be expressed are still planned"
    assert spec.unrepresented == ({"clause": "pick another toy", "reason": "only one toy was detected"},)
    assert spec.to_json()["unrepresented"][0]["reason"] == "only one toy was detected"


def test_a_fully_expressible_instruction_reports_nothing_unrepresented():
    assert parse().unrepresented == ()


def test_an_invented_predicate_used_inconsistently_is_rejected():
    # Its signature is read off its uses, so the uses have to agree.
    response = {
        **PLAN_RESPONSE,
        "phases": [
            *PLAN_RESPONSE["phases"],
            {
                "executor": "human",
                "description": "and the toy",
                "instructions": "open the toy",
                "atoms": [{"predicate": "IsOpen", "args": ["blue_toy"]}],
            },
        ],
    }
    with pytest.raises(ProposalError, match="inconsistent arguments"):
        parse(response)


def test_an_invented_predicate_keeps_its_own_name_in_two_plans():
    # The planner's symbolic layer interned ground atoms process-globally on (name, values) and
    # compared them by that string alone, so two tasks in one process could be served the FIRST
    # task's parameter types from the cache. The workaround was to suffix every invented name per
    # session -- and the suffix then leaked into the audit trail and into what the model was shown.
    # tandem's atoms do not intern, so the name is just the name, in every plan.
    first, second = parse().invented[0], parse().invented[0]
    assert first.name == second.name == "IsOpen"
    assert str(sorted(parse().phases[1].atoms)[0]) == "IsOpen(white_box)"


# --- walking the plan -----------------------------------------------------------------------------


def test_the_plan_is_walked_in_order():
    walk = _walk()
    assert not walk.next_is_human() and walk.goal()
    walk.advance()
    assert walk.next_is_human(), "phase 1 is the human's"
    walk.advance()
    assert not walk.next_is_human(), "phase 2 is the robot's again -- the part that used to be lost"
    assert [a.to_dict() for a in walk.goal()] == [{"predicate": "on", "args": ["blue_toy", "white_box"]}]
    walk.advance()
    assert walk.finished


def test_the_handoff_message_says_work_remains():
    walk = _walk()
    walk.advance()
    message = handoff_message(walk, walk.current)
    assert "Open the white_box" in message
    assert "1 more phase(s) follow" in message, "the operator must know the robot is not done"


def test_the_phase_event_says_where_in_the_plan_it_is():
    # The old integration emitted only the description and the expectations while the consumer read
    # phase_index/n_phases off the same event, so "step N of M" in the UI was always the fallback and
    # usually wrong -- and the test rig sent the missing keys, so nothing caught it. One source now.
    walk = _walk()
    walk.advance()
    summary = phase_summary(walk, walk.current)
    assert summary["phase_index"] == 1
    assert summary["n_phases"] == 3
    assert summary["description"] == "open the box"
    assert summary["expected"] == ["the container white_box is open, so its interior is visible"]


def test_a_robot_phase_that_cannot_be_planned_can_be_handed_to_a_person():
    # Only possible because tandem owns the executor split. It used to be decided at proposal time
    # inside the planner's process, so a phase the planner turned out not to be able to plan could
    # only end the attempt.
    walk = _walk()
    assert not walk.current.is_human
    handed = walk.hand_current_to_human()
    assert handed.is_human
    assert walk.spec.phases[0].is_human
    assert walk.next_is_human()
    # What the person is asked for is what will be checked afterwards: the phase's own atoms.
    assert handed.atoms == parse().phases[0].atoms
    assert "blue_toy is resting on top of table" in handed.instructions
    # And it is no longer a robot leg, so nothing tries to plan it.
    assert walk.robot_run() == ()


def test_a_renamed_object_re_binds_instead_of_throwing_the_plan_away():
    # Observed on the rig: phase 0 ran against objects the detector called "toy" and "box"; the next
    # pass called the same two things "blue_toy" and "cardboard_box". The plan was discarded and the
    # whole task re-planned from a scene already half rearranged, which asked the human to redo the
    # phase they had just finished.
    assert match_drifted_names(["toy", "box"], ["blue_toy", "cardboard_box"]) == {
        "toy": "blue_toy",
        "box": "cardboard_box",
    }
    # It works the other way round too (the detector dropping an adjective).
    assert match_drifted_names(["blue_toy"], ["toy"]) == {"blue_toy": "toy"}
    # Ambiguity is refused rather than guessed: the caller then re-plans, as before.
    assert match_drifted_names(["toy"], ["blue_toy", "red_toy"]) is None
    assert match_drifted_names(["toy"], ["cardboard_box"]) is None

    walk = _walk()
    walk.advance()
    walk.advance()  # phases 0 and 1 done; phase 2 is the robot's again
    walk.rebind({"blue_toy": "green_toy", "white_box": "brown_box"})
    assert [a.to_dict() for a in walk.goal()] == [{"predicate": "on", "args": ["green_toy", "brown_box"]}]
    assert walk.index == 2, "progress through the plan survives the rename"
    assert walk.spec.scene_types.surfaces == frozenset({"table", "brown_box"})


def test_the_plan_is_dropped_when_the_trajectory_changes():
    walk = _walk()
    assert walk.matches(walk.instruction, "traj-1")
    assert not walk.matches(walk.instruction, "traj-2")
    assert not walk.matches("a different task", "traj-1")


# The second running example: two ROBOT phases back to back, then a human one. The running example
# above always separates its robot phases with a human phase, which is exactly why the old rollout
# loop could ship for months only knowing how to continue a trajectory across a robot->human
# boundary -- a robot->robot boundary ended the episode, and the arm sorted one toy and stopped.
SORT_OBJECTS = ["blue_bowl", "blue_cloth", "blue_toy", "green_bowl", "green_toy"]
SORT_RESPONSE = {
    "new_predicates": [{"name": "AreCoveredBy", "instructions": "the {2} is draped over both {0} and {1}"}],
    "phases": [
        {
            "executor": "robot",
            "description": "put the blue toy in the blue bowl",
            "atoms": [{"predicate": "On", "args": ["blue_toy", "blue_bowl"]}],
        },
        {
            "executor": "robot",
            "description": "put the green toy in the green bowl",
            "atoms": [{"predicate": "On", "args": ["green_toy", "green_bowl"]}],
        },
        {
            "executor": "human",
            "description": "cover both bowls with the cloth",
            "instructions": "Drape the blue_cloth over both bowls.",
            "atoms": [{"predicate": "AreCoveredBy", "args": ["blue_bowl", "green_bowl", "blue_cloth"]}],
            "operator": {
                "name": "Drape",
                "args": ["blue_cloth", "blue_bowl", "green_bowl"],
                "preconditions": [{"predicate": "HandEmpty", "args": []}],
                "add_effects": [
                    {"predicate": "AreCoveredBy", "args": ["blue_bowl", "green_bowl", "blue_cloth"]}
                ],
                "delete_effects": [],
            },
        },
    ],
}


def _sort_walk():
    spec = parse(SORT_RESPONSE, objects=SORT_OBJECTS, instruction="sort the toys into same color bowls")
    return PhasePlan(cfg=CFG, caps=CAPS, instruction=spec.instruction, trajectory_id="traj-1", spec=spec)


def test_consecutive_robot_phases_are_planned_and_run_as_one_goal():
    # Both toys are sorted by ONE plan and one continuous motion, which is what an ordinary planner
    # run does with the same two-clause instruction. Planning them separately worked, but the arm
    # stopped between them to re-perceive and re-plan -- and that second perception pass is where the
    # object labels drift.
    walk = _sort_walk()
    assert feasibility.check_robot_phases(walk.spec, CAPS) is None

    assert [p.description for p in walk.robot_run()] == [
        "put the blue toy in the blue bowl",
        "put the green toy in the green bowl",
    ], "the human phase ends the run"
    assert [a.to_dict() for a in walk.goal()] == [
        {"predicate": "on", "args": ["blue_toy", "blue_bowl"]},
        {"predicate": "on", "args": ["green_toy", "green_bowl"]},
    ]

    walk.record_plan(0, mock.Mock(planning_seconds=1.5, skeleton_reused=False))
    walk.advance()
    assert walk.index == 2, "one leg covered both robot phases"
    assert walk.next_is_human()

    record = walk.to_json()
    assert [p["executor"] for p in record["phases"]] == ["robot", "robot", "human"]
    # Both phases record the plan, and say they shared it -- otherwise the audit trail reads as
    # though each had been solved on its own.
    for phase in record["phases"][:2]:
        assert phase["covers_phases"] == [0, 1]


def test_a_run_stops_at_a_phase_moving_an_object_the_run_already_moved():
    # One pick per object per plan, so On(blue_toy, table) and On(blue_toy, white_box) at once is
    # unsatisfiable, not slow. Phases like these are genuinely sequential and stay separate legs.
    spec = parse(
        _phases(
            {
                "executor": "robot",
                "description": "take the toy off the box",
                "atoms": [{"predicate": "On", "args": ["blue_toy", "table"]}],
            },
            {
                "executor": "robot",
                "description": "put the toy back on the box",
                "atoms": [{"predicate": "On", "args": ["blue_toy", "white_box"]}],
            },
        )
    )
    walk = PhasePlan(cfg=CFG, caps=CAPS, instruction=spec.instruction, trajectory_id="traj-1", spec=spec)
    assert len(walk.robot_run()) == 1, "the second phase moves blue_toy again"
    assert [a.to_dict() for a in walk.goal()] == [{"predicate": "on", "args": ["blue_toy", "table"]}]

    walk.record_plan(0, mock.Mock(planning_seconds=1.0, skeleton_reused=False))
    walk.advance()
    assert walk.index == 1 and not walk.finished and not walk.next_is_human()
    assert [a.to_dict() for a in walk.goal()] == [{"predicate": "on", "args": ["blue_toy", "white_box"]}]
    # Planned on its own, so no shared-plan marker.
    assert "covers_phases" not in walk.to_json()["phases"][0]


def test_a_backend_that_allows_repeated_picks_conjoins_them_anyway():
    # The stopping rule is the backend's declaration, not a fact about every planner. One that can
    # pick the same object twice in a plan says so, and the same two phases become one goal.
    import dataclasses

    spec = parse(
        _phases(
            {
                "executor": "robot",
                "description": "take the toy off the box",
                "atoms": [{"predicate": "On", "args": ["blue_toy", "table"]}],
            },
            {
                "executor": "robot",
                "description": "put the toy back on the box",
                "atoms": [{"predicate": "On", "args": ["blue_toy", "white_box"]}],
            },
        )
    )
    caps = dataclasses.replace(CAPS, one_pick_per_object=False)
    walk = PhasePlan(cfg=CFG, caps=caps, instruction=spec.instruction, trajectory_id="t", spec=spec)
    assert len(walk.robot_run()) == 2


def test_a_leg_is_not_gated_on_an_object_only_a_later_human_phase_names():
    # objects_named() spans every remaining phase, so it includes the cloth -- which no robot phase
    # names. Perception missing the cloth used to destroy the whole plan before the planner was ever
    # called, and the operator's next attempt re-sorted the toys from the start.
    walk = _sort_walk()

    assert "blue_cloth" in walk.objects_named()
    assert "blue_cloth" not in walk.objects_needed_now()
    # What the leg genuinely cannot proceed without: every object in the run it is about to plan,
    # plus the surfaces, which are pinned for the whole task and would otherwise turn into movables
    # mid-plan.
    assert walk.objects_needed_now() == {"blue_toy", "blue_bowl", "green_toy", "green_bowl", "table"}

    # The human phase's own leg does need the cloth -- the relaxation is per-leg, not permanent.
    walk.advance()
    assert "blue_cloth" in walk.objects_needed_now()


def test_a_name_the_plan_already_owns_stays_the_plans_after_its_phase_is_done():
    # The facts the re-binding pool rests on. The pool is `detected - scene_types.all_names`, NOT
    # `detected - objects_named()`: objects_named() covers only the phases still to come, so an object
    # named solely by a COMPLETED phase drops out of it while remaining a plan object -- and offering it
    # as a target lets match_drifted_names fold two objects into one, pointing this leg at the thing the
    # robot has already put away. The pool itself is tested through the loop, in tests/test_rebind_pool.py.
    walk = _sort_walk()
    walk.advance()  # both robot phases done; only the human phase remains

    owned = walk.spec.scene_types.all_names
    assert {"blue_toy", "green_toy"} <= owned, "the sorted toys are still the plan's objects"
    assert not ({"blue_toy", "green_toy"} & walk.objects_named()), "but no remaining phase names them"
    # The rule the guard is there for: a nested pair would otherwise match.
    assert match_drifted_names(["green_toy"], ["toy"]) == {"green_toy": "toy"}


def test_a_finished_plan_still_reports_its_pinned_surfaces():
    # objects_needed_now() is read on any leg, including one that finds the plan already complete.
    walk = _sort_walk()
    while not walk.finished:
        walk.advance()
    assert walk.objects_needed_now() == {"blue_bowl", "green_bowl", "table"}


def test_the_audit_record_says_what_the_planner_was_handed():
    walk = _walk()
    walk.record_plan(0, mock.Mock(planning_seconds=2.0, skeleton_reused=False))
    record = walk.to_json()

    assert record["planner"] == "tiptop"
    assert [p["executor"] for p in record["phases"]] == ["robot", "human", "robot"]
    assert record["phases"][0]["goal"] == [{"predicate": "on", "args": ["blue_toy", "table"]}]
    assert record["phases"][0]["goal_description"] == ["blue_toy is resting on top of table"]
    assert record["phases"][0]["planning_seconds"] == 2.0
    assert "goal" not in record["phases"][1]
    assert record["phases"][1]["instructions"].startswith("Open the white_box")
    assert set(record["provenance"]) >= {"phases_and_their_order", "robot_phases", "who_does_what"}
    assert record["phases"][1]["atoms"] == ["IsOpen(white_box)"]


# --- verification ---------------------------------------------------------------------------------


def test_only_camera_settleable_atoms_are_put_to_the_model():
    # A human phase may also mention the robot's own state. HandEmpty/Holding must not be judged from
    # a photo: the frame is a third-person view chosen because the arm is wherever the operator left
    # it, so the gripper is often out of shot, and the classifier answers false when it cannot see
    # the statement to be true. Which of them are checkable is the backend's declaration.
    response = _plan_response(
        phases=[
            {
                "executor": "human",
                "description": "open the box",
                "instructions": "open it",
                "atoms": [
                    {"predicate": "IsOpen", "args": ["white_box"]},
                    {"predicate": "On", "args": ["blue_toy", "table"]},
                    {"predicate": "HandEmpty", "args": []},
                ],
                "operator": {
                    "name": "Open",
                    "args": ["white_box"],
                    "preconditions": [{"predicate": "HandEmpty", "args": []}],
                    "add_effects": [
                        {"predicate": "IsOpen", "args": ["white_box"]},
                        {"predicate": "On", "args": ["blue_toy", "table"]},
                        {"predicate": "HandEmpty", "args": []},
                    ],
                    "delete_effects": [],
                },
            }
        ]
    )
    spec = parse(response)
    phase = spec.phases[0]
    asked = []

    async def fake_classify_all(image, atoms, descriptions, cfg, *, expected=True, role="effect"):
        asked.extend(atoms)
        return [Verdict(a, describe(a, descriptions), True, "", expected=expected, role=role) for a in atoms]

    with mock.patch.object(grounding, "classify_all", fake_classify_all):
        ok, _ = asyncio.run(grounding.verify_effects(None, phase, spec.invented, CFG, CAPS))
    assert ok
    assert {str(a) for a in asked} == {"IsOpen(white_box)", "On(blue_toy, table)"}

    # But the human is still TOLD about all of it, so they know what is expected.
    shown = grounding.describe_expectations(phase, grounding.descriptions_for(spec.invented, CAPS))
    assert any("gripper is empty" in line for line in shown)


# --- the reprompt loop ----------------------------------------------------------------------------


class _FakeGemini:
    """A client that returns canned responses in order and records the prompts it saw."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []
        self.aio = self

    @property
    def models(self):
        return self

    async def generate_content(self, model, contents, config):
        self.prompts.append(contents[-1])
        return mock.Mock(text=self._responses.pop(0))


def test_a_rejected_proposal_is_reprompted_with_the_reason():
    # The mechanism the design leans on: a proposal that parses but does not validate is a correction,
    # not a failed run. This is what recovered the errors seen live (a hallucinated object, an atom
    # applied to the wrong type, a robot phase asking for an invented predicate).
    bad = json.dumps(
        _plan_response(
            phases=[
                {
                    "executor": "robot",
                    "description": "open it",
                    "atoms": [{"predicate": "IsOpen", "args": ["white_box"]}],
                }
            ]
        )
    )
    client = _FakeGemini([bad, json.dumps(PLAN_RESPONSE)])
    with mock.patch.object(llm, "gemini_client", lambda: client):
        spec = asyncio.run(llm.query_json("PROMPT", lambda d: parse(d), model="m", schema={}, max_attempts=3))
    assert len(client.prompts) == 2, "the second attempt should have been made"
    assert "A robot phase cannot achieve" in client.prompts[1], "the reprompt must say what was wrong"
    assert [p.executor for p in spec.phases] == ["robot", "human", "robot"]


def test_a_proposal_that_never_validates_raises_the_last_reason():
    bad = json.dumps(
        {
            "phases": [
                {
                    "executor": "robot",
                    "description": "x",
                    "atoms": [{"predicate": "On", "args": ["blue_toy", "moon"]}],
                }
            ]
        }
    )
    client = _FakeGemini([bad, bad, bad])
    with mock.patch.object(llm, "gemini_client", lambda: client):
        with pytest.raises(ProposalError, match="not an object in this scene"):
            asyncio.run(llm.query_json("PROMPT", lambda d: parse(d), model="m", schema={}, max_attempts=3))
    assert len(client.prompts) == 3, "every attempt should have been used"


# --- config ---------------------------------------------------------------------------------------


def test_config_defaults_to_off_and_rejects_a_bad_failure_policy():
    assert PlanningConfig().enabled is False
    assert PlanningConfig(enabled=True).on_robot_phase_failure == "abort"
    with pytest.raises(ValueError, match="on_robot_phase_failure must be one of"):
        PlanningConfig(on_robot_phase_failure="panic")
    with pytest.raises(ValueError, match="max_attempts must be at least 1"):
        PlanningConfig(max_attempts=0)


def test_the_profile_is_the_definition_of_these_settings():
    # HitlSpec used to mirror a config that lived inside the planner and had to be kept in step with
    # it by hand. It is now the only definition, so `extra: forbid` is the only validation layer --
    # and a value it would have to translate for the planner is checked here rather than there.
    from pydantic import ValidationError

    from tandem.core.profiles import HitlSpec

    resolved = HitlSpec(enabled=True, verify_retries=2).to_planning_config()
    assert resolved.enabled and resolved.verify_retries == 2
    assert resolved.on_robot_phase_failure == "abort"
    with pytest.raises(ValidationError, match="enable"):
        HitlSpec(enable=True)
    with pytest.raises(ValidationError, match="on_robot_phase_failure"):
        HitlSpec(on_robot_phase_failure="panic")


# --- the dry run ----------------------------------------------------------------------------------


def test_tandem_plan_decomposes_a_task_with_no_planner_present(tmp_path, monkeypatch):
    """End to end through the CLI, with the model stubbed and nothing else.

    This is the command the whole move buys: the decomposition can be checked before anyone goes
    near the arm. It used to be answerable only from inside a warm planner process, which meant the
    answer arrived with an operator already standing next to a robot -- and the remedy for a plan
    that covers less than it was asked (put the missing object on the table, reword the instruction)
    is only available before that.
    """
    from typer.testing import CliRunner

    from tandem.cli.app import app

    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow is a base dependency
        pytest.skip("Pillow is not installed")

    photo = tmp_path / "workspace.png"
    Image.new("RGB", (64, 48), (200, 200, 200)).save(photo)

    client = _FakeGemini([json.dumps(PLAN_RESPONSE)])
    monkeypatch.setattr(llm, "gemini_client", lambda: client)

    # A snapshot, not a bare `"cutamp" not in sys.modules`: another test imports the vendored
    # symbolic layer on purpose to check the capability declaration against it, and whether that has
    # already run must not decide whether this passes.
    before = set(sys.modules)
    result = CliRunner().invoke(
        app,
        [
            "plan", "do the thing",
            "--image", str(photo),
            "--object", "blue_toy",
            "--object", "white_box",
            "--save-vlm-io", str(tmp_path / "vlm"),
        ],
    )
    assert result.exit_code == 0, result.output

    assert "0  robot" in result.output and "1  human" in result.output and "2  robot" in result.output
    assert "IsOpen" in result.output, "the invented predicate and its classifier are shown"
    assert '"predicate": "on"' in result.output, "and what the planner would actually be handed"

    # No planner was consulted for any of it.
    planner_modules = {m for m in set(sys.modules) - before if m.split(".")[0] in ("cutamp", "tiptop", "torch")}
    assert not planner_modules, f"`tandem plan` imported a planner: {sorted(planner_modules)}"

    # The audit trail is written where it was asked for, rejected attempts and all.
    index = tmp_path / "vlm" / "index.jsonl"
    assert index.is_file()
    recorded = [json.loads(line) for line in index.read_text().splitlines()]
    assert [r["label"] for r in recorded] == ["task plan"]


def test_tandem_plan_says_loudly_when_the_plan_covers_less_than_it_was_asked(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from tandem.cli.app import app

    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        pytest.skip("Pillow is not installed")

    photo = tmp_path / "workspace.png"
    Image.new("RGB", (32, 32), (10, 10, 10)).save(photo)

    response = {
        **PLAN_RESPONSE,
        "unrepresented": [{"clause": "pick another toy", "reason": "only one toy was detected"}],
    }
    client = _FakeGemini([json.dumps(response)])
    monkeypatch.setattr(llm, "gemini_client", lambda: client)

    result = CliRunner().invoke(
        app, ["plan", "do the thing", "--image", str(photo), "-o", "blue_toy", "-o", "white_box"]
    )
    assert result.exit_code == 0, result.output
    assert "pick another toy" in result.output
    assert "only one toy was detected" in result.output


def test_tandem_plan_json_output_actually_parses(tmp_path, monkeypatch):
    """`--json` is for piping into something. Any prose on stdout ahead of it breaks that."""
    from typer.testing import CliRunner

    from tandem.cli.app import app

    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        pytest.skip("Pillow is not installed")

    photo = tmp_path / "workspace.png"
    Image.new("RGB", (32, 32), (128, 128, 128)).save(photo)
    monkeypatch.setattr(llm, "gemini_client", lambda: _FakeGemini([json.dumps(PLAN_RESPONSE)]))

    result = CliRunner().invoke(
        app, ["plan", "do the thing", "--image", str(photo), "-o", "blue_toy", "-o", "white_box", "--json"]
    )
    assert result.exit_code == 0, result.output

    payload = json.loads(result.stdout)
    assert [p["executor"] for p in payload["phases"]] == ["robot", "human", "robot"]
    assert payload["planner"] == "tiptop"
