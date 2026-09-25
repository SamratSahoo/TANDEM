"""The magic operator: the explicit contract behind a human phase.

A human phase used to say only what should be true afterwards. ``HumanOperator`` adds what must hold
first and what the step undoes, and ``Phase`` exposes all three whether or not an operator is present,
so the checks built on top never need to ask which kind of phase they have. These are the struct-level
tests; the proposal parser that builds operators from a model's reply is tested with it.

The running example is the same three-phase task the planning tests use: the toy off the box, the
box opened by a person, the toy into the box. Here the person's step carries ``Open(white_box)``.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from tandem.planners import registry
from tandem.planning.config import PlanningConfig
from tandem.planning.plan import PhasePlan
from tandem.planning.proposal import parse_plan_response
from tandem.planning.structs import HumanOperator, Phase
from tandem.planning.symbols import Atom, Parameter, ProposalError

CAPS = registry.capabilities("tiptop")
CFG = PlanningConfig(enabled=True)

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

IS_OPEN = Atom("IsOpen", ("white_box",))
IS_CLOSED = Atom("IsClosed", ("white_box",))
HAND_EMPTY = Atom("HandEmpty")
TOY_ON_TABLE = Atom("On", ("blue_toy", "table"))


def open_box(**changes) -> HumanOperator:
    """``Open(white_box)``, typed the way the scene types it: the box is placed into, so a surface."""
    fields = {
        "name": "Open",
        "args": ("white_box",),
        "parameters": (Parameter("x0", "surface"),),
        "preconditions": frozenset({HAND_EMPTY}),
        "add_effects": frozenset({IS_OPEN}),
        "delete_effects": frozenset(),
    }
    return HumanOperator(**{**fields, **changes})


def close_box() -> HumanOperator:
    """``Close(white_box)``: needs the box open, and undoes exactly that."""
    return HumanOperator(
        "Close",
        ("white_box",),
        (Parameter("x0", "surface"),),
        preconditions={IS_OPEN},
        add_effects={IS_CLOSED},
        delete_effects={IS_OPEN},
    )


def spec_with_operator(operator=None):
    """The three-phase plan, with the human phase carrying ``operator``."""
    spec = parse_plan_response(PLAN_RESPONSE, "do the thing", ["blue_toy", "white_box"], "table", CAPS)
    human = spec.phases[1]
    return spec.replace_phase(1, dataclasses.replace(human, operator=operator or open_box()))


# --- the operator itself --------------------------------------------------------------------------


def test_a_human_phase_carries_its_operator_grounded_to_this_scene():
    # Struct-level half of the LJ test of the same name. The parser half is in test_parser_operators.py.
    phase = spec_with_operator().phases[1]
    operator = phase.operator
    assert operator.display == "Open(white_box)"
    # Typed from the args' scene types, like an invented predicate's parameters are.
    assert operator.signature == "Open(x0: surface)"
    assert {str(a) for a in operator.preconditions} == {"HandEmpty()"}
    assert {str(a) for a in operator.add_effects} == {"IsOpen(white_box)"}
    assert operator.delete_effects == frozenset()
    # And the phase answers through it.
    assert phase.preconditions == operator.preconditions
    assert phase.add_effects == operator.add_effects
    assert phase.delete_effects == frozenset()


def test_a_robot_phase_never_carries_an_operator():
    # The robot's operators are the planner's and they are fixed; `preconditions` on a robot phase
    # falls back to the empty set rather than to something invented for it.
    robot = spec_with_operator().phases[0]
    assert robot.operator is None
    assert robot.preconditions == frozenset()
    # With no operator, the phase's own atoms ARE its add effects -- the pre-operator behaviour.
    assert robot.add_effects == robot.atoms
    assert robot.delete_effects == frozenset()


def test_an_operator_on_a_robot_phase_is_refused():
    # Almost always a human step written as a robot phase, which is the misjudgement worth catching.
    with pytest.raises(ProposalError, match="Only a human phase"):
        Phase("robot", "shove it", frozenset({TOY_ON_TABLE}), operator=open_box())


def test_a_human_phase_without_an_operator_keeps_the_atoms_only_contract():
    phase = Phase("human", "open the box", frozenset({IS_OPEN}), "open it")
    assert phase.preconditions == frozenset()
    assert phase.add_effects == frozenset({IS_OPEN})
    assert phase.delete_effects == frozenset()
    assert "operator" not in phase.summary()


def test_an_arity_mismatch_is_a_proposal_error_not_an_assert():
    # An assert would vanish under `python -O` and let the signature silently drop an argument.
    with pytest.raises(ProposalError, match="exactly one object per parameter"):
        open_box(args=("white_box", "blue_toy"))
    with pytest.raises(ProposalError, match="exactly one object per parameter"):
        open_box(args=())


def test_apply_removes_delete_effects_then_adds_add_effects():
    close = close_box()
    state = frozenset({IS_OPEN, TOY_ON_TABLE})
    assert close.apply(state) == frozenset({IS_CLOSED, TOY_ON_TABLE})
    # Add effects win a tie, for a state that already held something both deleted and added.
    both = dataclasses.replace(close, add_effects=frozenset({IS_OPEN}))
    assert IS_OPEN in both.apply(state)


def test_unmet_names_exactly_the_preconditions_the_state_lacks():
    operator = open_box(preconditions=frozenset({HAND_EMPTY, TOY_ON_TABLE}))
    assert operator.unmet(frozenset({TOY_ON_TABLE})) == frozenset({HAND_EMPTY})
    assert operator.unmet(frozenset({HAND_EMPTY, TOY_ON_TABLE, IS_OPEN})) == frozenset()


def test_an_operator_built_from_lists_equals_one_built_from_sets():
    # The parser and a test build operators differently; they must still compare and hash alike,
    # because the contract check keys on them.
    from_lists = HumanOperator(
        "Open",
        ["white_box"],
        [Parameter("x0", "surface")],
        preconditions=[HAND_EMPTY],
        add_effects=[IS_OPEN],
        delete_effects=[],
    )
    assert from_lists == open_box()
    assert hash(from_lists) == hash(open_box())
    phase = Phase("human", "open", frozenset({IS_OPEN}), "open it", from_lists)
    assert phase in {Phase("human", "open", frozenset({IS_OPEN}), "open it", open_box())}


# --- rebinding ------------------------------------------------------------------------------------


def test_rebind_relabels_the_operator_with_the_phase():
    # A later perception pass calls the box something else. The operator's checks must follow, or
    # they are stated over a name the scene no longer produces.
    operator = open_box(
        preconditions=frozenset({HAND_EMPTY, Atom("On", ("white_box", "table"))}),
        delete_effects=frozenset({IS_CLOSED}),
    )
    phase = Phase("human", "open the box", frozenset({IS_OPEN}), "open it", operator)
    moved = phase.rebind({"white_box": "cardboard_box"})

    assert moved.atoms == frozenset({Atom("IsOpen", ("cardboard_box",))})
    assert moved.operator.display == "Open(cardboard_box)"
    assert moved.preconditions == frozenset({HAND_EMPTY, Atom("On", ("cardboard_box", "table"))})
    assert moved.add_effects == frozenset({Atom("IsOpen", ("cardboard_box",))})
    assert moved.delete_effects == frozenset({Atom("IsClosed", ("cardboard_box",))})
    # A rebind renames objects; it never changes their types, so the signature is untouched.
    assert moved.operator.signature == operator.signature
    # Nothing that was not renamed moves.
    assert phase.rebind({"green_toy": "toy"}) == phase


def test_a_specification_rebind_carries_every_operator_through():
    spec = spec_with_operator().rebind({"white_box": "cardboard_box"})
    assert [op.display for op in spec.operators] == ["Open(cardboard_box)"]
    assert spec.phases[1].operator.add_effects == frozenset({Atom("IsOpen", ("cardboard_box",))})


# --- the specification ----------------------------------------------------------------------------


def test_operators_are_listed_in_phase_order_and_recorded_with_their_phase():
    close = close_box()
    spec = spec_with_operator()
    spec = spec.replace_phase(2, Phase("human", "close the box", frozenset({IS_CLOSED}), "shut it", close))
    assert [op.display for op in spec.operators] == ["Open(white_box)", "Close(white_box)"]

    recorded = spec.to_json()["human_operators"]
    assert [(r["phase"], r["instance"]) for r in recorded] == [
        (1, "Open(white_box)"),
        (2, "Close(white_box)"),
    ]
    assert recorded[1] == {
        "phase": 2,
        "name": "Close",
        "args": ["white_box"],
        "signature": "Close(x0: surface)",
        "instance": "Close(white_box)",
        "preconditions": ["IsOpen(white_box)"],
        "add_effects": ["IsClosed(white_box)"],
        "delete_effects": ["IsOpen(white_box)"],
    }


def test_replacing_one_phase_leaves_the_other_operators_alone():
    spec = spec_with_operator()
    replaced = spec.replace_phase(0, Phase("robot", "something else", frozenset({TOY_ON_TABLE})))
    assert replaced.phases[1].operator == open_box()


def test_an_all_robot_plan_records_no_operators():
    spec = spec_with_operator()
    spec = spec.replace_phase(1, spec.phases[0])
    assert spec.operators == ()
    assert spec.to_json()["human_operators"] == []


def test_a_robot_phase_handed_to_a_person_gets_no_operator():
    # The teleop fallback. The model never wrote an operator for this step, so the hand-off must not
    # make one up: the phase keeps the atoms-only contract the planner was going to be held to.
    spec = spec_with_operator()
    walk = PhasePlan(cfg=CFG, caps=CAPS, instruction=spec.instruction, trajectory_id="t", spec=spec)
    handed = walk.hand_current_to_human()
    assert handed.is_human and handed.operator is None
    assert handed.add_effects == handed.atoms
    # The operator the model DID write for the real human phase is still there.
    assert walk.spec.operators == (open_box(),)


# --- the audit record -----------------------------------------------------------------------------


def test_an_operator_round_trips_through_json():
    operator = open_box(
        preconditions=frozenset({HAND_EMPTY, Atom("On", ("white_box", "table"))}),
        delete_effects=frozenset({IS_CLOSED}),
    )
    assert HumanOperator.from_json(json.loads(json.dumps(operator.to_json()))) == operator


def test_a_nullary_operator_round_trips_through_json():
    tidy = HumanOperator("Tidy", (), (), add_effects={Atom("IsTidy", ())}, delete_effects={HAND_EMPTY})
    record = json.loads(json.dumps(tidy.to_json()))
    assert record["signature"] == "Tidy()" and record["delete_effects"] == ["HandEmpty()"]
    assert HumanOperator.from_json(record) == tidy


def test_the_hitl_record_round_trips_every_operator():
    # What hitl.json holds, read back: both the plan-level list and the per-phase record must give
    # back the operator the run was checked against.
    spec = spec_with_operator()
    walk = PhasePlan(cfg=CFG, caps=CAPS, instruction=spec.instruction, trajectory_id="t", spec=spec)
    record = json.loads(json.dumps(walk.to_json()))

    listed = record["specification"]["human_operators"]
    assert [HumanOperator.from_json(r) for r in listed] == [open_box()]
    assert [r["phase"] for r in listed] == [1]
    assert HumanOperator.from_json(record["phases"][1]["operator"]) == open_box()
    assert "operator" not in record["phases"][0]
    assert "operator" not in record["phases"][2]


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"name": "Close"}, "but its signature is"),
        ({"instance": "Open(blue_toy)"}, "its name and args make"),
        ({"args": "white_box"}, "must be a list"),
        ({"add_effects": ["IsOpen white_box"]}, "Cannot read the atom"),
        ({"add_effects": ["On(blue_toy, )"]}, "empty argument"),
        ({"signature": "Open(x0)"}, "expected 'name: type'"),
    ],
)
def test_a_damaged_operator_record_is_refused(change, expected):
    record = {**open_box().to_json(), **change}
    with pytest.raises(ValueError, match=expected):
        HumanOperator.from_json(record)
