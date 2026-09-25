"""The plan-time contract check: phases whose operators undo what a later phase needs.

No robot, no GPU, no network, no model: the check is set arithmetic over a parsed plan, so the tests
drive it with the JSON a proposer would send. The running example is the planning suite's three-phase
task -- the toy off the box, the box opened by a person, the toy into the box -- plus LJ's
open/close/insert plan, whose middle phase closes what the last one needs open.

Ported from the suite that lived inside the planner fork. What is new here is the planner-agnostic
half: displacement and "what a phase moves" come from the backend's declaration, so a backend that
declares neither gets a weaker check, and a backend with a different goal language gets the same one.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import subprocess
import sys

import pytest

from tandem.core.errors import TandemError
from tandem.planners import registry
from tandem.planners.base import Capabilities
from tandem.planning import contracts
from tandem.planning.contracts import (
    PhaseTrace,
    check_plan_effects,
    displaced_by,
    expected_before,
    phase_moves,
    simulate_phases,
    wasted_robot_move,
)
from tandem.planning.proposal import parse_plan_response
from tandem.planning.symbols import Atom, Parameter, Predicate

CAPS = registry.capabilities("tiptop")
OBJECTS = ["blue_toy", "white_box"]
TABLE = "table"

# TipTop's goal language with no free delete effect declared: a backend that never said an object
# rests on one thing at a time.
NO_DISPLACEMENT = dataclasses.replace(CAPS, exclusive_arguments={})


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


# The plan the proposer should produce for the three-phase task. The ordering is the whole point:
# the box is opened BEFORE anything is placed in it.
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


def _open_close(**changes):
    """A three-phase plan whose middle human phase CLOSES what the first one opened."""
    plan = {
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
    return {**plan, **changes}


def _phases(*phases):
    """A plan of just these phases, with no invented predicates left over to be unused."""
    return {"phases": list(phases)}


def parse(response=None, objects=OBJECTS, caps=CAPS):
    """Parse a reply the way the proposal stage does. Every human phase arrives carrying its operator."""
    return parse_plan_response(response or PLAN_RESPONSE, "do the thing", objects, TABLE, caps)


def shown(atoms):
    return {str(a) for a in atoms}


# --- displacement ---------------------------------------------------------------------------------


def test_a_placement_displaces_only_the_same_objects_other_placements():
    toy_on_table = Atom("On", ("blue_toy", "table"))
    cup_on_table = Atom("On", ("cup", "table"))
    holding_toy = Atom("Holding", ("blue_toy",))
    state = {toy_on_table, cup_on_table, holding_toy}

    displaced = displaced_by({Atom("On", ("blue_toy", "shelf"))}, state, CAPS.exclusive_arguments)
    # The toy is no longer on the table. The cup is, and nothing about where the toy rests says
    # anything about another predicate the toy appears in.
    assert displaced == {toy_on_table}


def test_asserting_what_already_holds_displaces_nothing():
    toy_on_table = Atom("On", ("blue_toy", "table"))
    assert displaced_by({toy_on_table}, {toy_on_table}, CAPS.exclusive_arguments) == frozenset()


def test_a_predicate_nobody_declared_exclusive_displaces_nothing():
    state = {Atom("On", ("blue_toy", "table"))}
    assert displaced_by({Atom("On", ("blue_toy", "shelf"))}, state, {}) == frozenset()


def test_the_trace_replaces_an_On_atom_rather_than_accumulating_both():
    # An object rests on one thing at a time, so the second placement displaces the first. Without
    # this the trace would believe the toy is on the table AND in the box.
    traces = simulate_phases(parse(), caps=CAPS)
    assert shown(traces[1].before) == {"On(blue_toy, table)"}
    assert shown(traces[2].after) == {"IsOpen(white_box)", "On(blue_toy, white_box)"}


def test_a_backend_that_declares_no_exclusive_arguments_accumulates_both():
    # Nothing in its declaration says the toy cannot be in two places, so nothing is displaced. The
    # check that follows from it is weaker, never wrong.
    traces = simulate_phases(parse(), caps=NO_DISPLACEMENT)
    assert shown(traces[2].after) == {
        "IsOpen(white_box)",
        "On(blue_toy, table)",
        "On(blue_toy, white_box)",
    }


# --- the trace ------------------------------------------------------------------------------------


def test_the_trace_walks_every_phase_in_order():
    spec = parse()
    traces = simulate_phases(spec, caps=CAPS)
    assert [t.index for t in traces] == [0, 1, 2]
    assert all(isinstance(t, PhaseTrace) for t in traces)
    assert [t.phase for t in traces] == list(spec.phases)
    # Each phase is entered in the state the one before it left.
    assert traces[0].before == frozenset()
    assert all(earlier.after == later.before for earlier, later in zip(traces, traces[1:], strict=False))


def test_unmet_means_nothing_established_it_not_that_it_is_false():
    # HandEmpty() is nobody's add effect, so the trace cannot show it holding before the human phase.
    # That is "unmet" in the trace's sense only; whether it is a problem is check_plan_effects' call.
    traces = simulate_phases(parse(), caps=CAPS)
    assert shown(traces[1].unmet) == {"HandEmpty()"}
    assert traces[0].unmet == traces[2].unmet == frozenset()


def test_the_trace_starts_from_what_is_known_to_hold():
    hand_empty = Atom("HandEmpty")
    traces = simulate_phases(parse(), {hand_empty}, caps=CAPS)
    assert hand_empty in traces[0].before
    assert traces[1].unmet == frozenset()


def test_a_declared_delete_effect_leaves_the_state():
    traces = simulate_phases(parse(_open_close()), caps=CAPS)
    assert shown(traces[1].after) == {"IsClosed(white_box)"}
    assert shown(traces[2].unmet) == {"IsOpen(white_box)"}


# --- check_plan_effects ---------------------------------------------------------------------------


def test_a_plan_that_deletes_what_a_later_phase_needs_is_refused():
    # The check the operator contract exists for: phase 1 closes the box, phase 2 needs it open.
    spec = parse(_open_close())
    assert "IsOpen(white_box)" in check_plan_effects(spec, caps=CAPS)
    assert "an earlier phase deletes it" in check_plan_effects(spec, caps=CAPS)


def test_the_refusal_is_the_evaluated_wording():
    # This sentence goes back to the proposer in the repair prompt, so it is part of the method: it is
    # LJ's, word for word, with the atom rendered the way tandem renders every atom.
    assert check_plan_effects(parse(_open_close()), caps=CAPS) == (
        "phase 2 ('drop the toy in') (Insert(blue_toy, white_box)) requires IsOpen(white_box), but an "
        "earlier phase deletes it and no phase puts it back. Either reorder the phases so it still "
        "holds, or drop it from that operator's preconditions."
    )


def test_reordering_the_same_phases_makes_the_plan_consistent():
    phases = _open_close()["phases"]
    fixed = parse(_open_close(phases=[phases[0], phases[2], phases[1]]))
    assert check_plan_effects(fixed, caps=CAPS) is None


def test_a_precondition_put_back_before_it_is_needed_is_fine():
    # Deleted, then re-established by a later phase, then needed: the plan is consistent.
    phases = _open_close()["phases"]
    reopen = {
        "executor": "human",
        "description": "open the box again",
        "instructions": "open it again",
        "atoms": _atoms(("IsOpen", ["white_box"])),
        "operator": _op(
            "Open",
            ["white_box"],
            pre=[("IsClosed", ["white_box"])],
            add=[("IsOpen", ["white_box"])],
            dele=[("IsClosed", ["white_box"])],
        ),
    }
    spec = parse(_open_close(phases=[phases[0], phases[1], reopen, phases[2]]))
    assert check_plan_effects(spec, caps=CAPS) is None


def test_a_placement_that_invalidates_a_later_precondition_is_caught():
    # A placement deletes whatever the object rested on before, without ever declaring it: the robot
    # moving the toy into the box is what makes On(blue_toy, table) false. A later phase requiring it
    # is as broken as one requiring a declared delete effect, and has to be caught the same way.
    broken = check_plan_effects(parse(_wipe_under_the_toy()), caps=CAPS)
    assert "On(blue_toy, table)" in broken and "an earlier phase deletes it" in broken


def _wipe_under_the_toy():
    return {
        "new_predicates": [{"name": "IsWiped", "instructions": "the surface {0} has been wiped"}],
        "phases": [
            {
                "executor": "robot",
                "description": "toy to the table",
                "atoms": _atoms(("On", ["blue_toy", "table"])),
            },
            {
                "executor": "robot",
                "description": "toy into the box",
                "atoms": _atoms(("On", ["blue_toy", "white_box"])),
            },
            {
                "executor": "human",
                "description": "wipe under the toy",
                "instructions": "wipe it",
                "atoms": _atoms(("IsWiped", ["table"])),
                "operator": _op(
                    "Wipe", ["table"], pre=[("On", ["blue_toy", "table"])], add=[("IsWiped", ["table"])]
                ),
            },
        ],
    }


def test_without_exclusive_arguments_a_placement_deletes_nothing():
    # The same plan against a backend that never declared that an object rests on one thing: the
    # displacement is not known, so it is not held against the plan. Sound, not complete.
    assert check_plan_effects(parse(_wipe_under_the_toy()), caps=NO_DISPLACEMENT) is None


def test_a_precondition_nothing_establishes_is_left_alone_when_the_scene_is_unmeasured():
    # HandEmpty() is nobody's add effect, and On(toy, box) may simply have been true to begin with.
    # Refusing either would reject almost every plan that is in fact fine, so the check is sound and
    # not complete: it only concludes something is wrong when the PLAN made it wrong.
    assert check_plan_effects(parse(), caps=CAPS) is None


def _needs_an_unlocked_box():
    """Open(white_box) requires IsUnlocked(white_box), which no phase makes true.

    IsUnlocked is named only in a precondition. The parser counts that as a use, so it is typed and
    accepted like any other invented predicate.
    """
    return parse(
        _phases(
            {
                "executor": "human",
                "description": "open the box",
                "instructions": "open it",
                "atoms": _atoms(("IsOpen", ["white_box"])),
                "operator": _op(
                    "Open",
                    ["white_box"],
                    pre=[("IsUnlocked", ["white_box"])],
                    add=[("IsOpen", ["white_box"])],
                ),
            }
        )
        | {
            "new_predicates": [
                {"name": "IsOpen", "instructions": "the container {0} is open"},
                {"name": "IsUnlocked", "instructions": "the box {0} is unlocked"},
            ]
        }
    )


def test_an_invented_precondition_nothing_establishes_passes_while_the_scene_is_unmeasured():
    # The box may well have started unlocked. Without a measurement, refusing would be a guess.
    assert check_plan_effects(_needs_an_unlocked_box(), caps=CAPS) is None


def test_an_invented_precondition_nothing_establishes_is_refused_once_the_scene_is_measured():
    broken = check_plan_effects(_needs_an_unlocked_box(), frozenset(), initial_state_known=True, caps=CAPS)
    assert "IsUnlocked(white_box)" in broken
    assert "no phase makes it true and it is not true in the workspace to begin with" in broken


def test_an_invented_precondition_the_measurement_found_true_is_fine():
    unlocked = Atom("IsUnlocked", ("white_box",))
    spec = _needs_an_unlocked_box()
    assert check_plan_effects(spec, {unlocked}, initial_state_known=True, caps=CAPS) is None


def test_a_measured_scene_still_does_not_judge_the_planners_own_predicates():
    # Only invented atoms are measured, so `initially_true` says nothing about HandEmpty(): holding
    # it to "never established" would refuse the flagship plan over a gripper that was empty.
    assert check_plan_effects(parse(), frozenset(), initial_state_known=True, caps=CAPS) is None


# --- expected_before ------------------------------------------------------------------------------


def test_a_robot_leg_inherits_only_what_earlier_phases_established():
    # The robot-side precondition set: every atom an earlier phase was responsible for and no later
    # one undid. Both of these are beliefs that can have gone stale by now -- the human phase may
    # not have opened the box it was verified as opening, and it may have knocked the toy off the
    # table while it was at it.
    expected = expected_before(parse(), 2, caps=CAPS)
    assert shown(expected) == {"On(blue_toy, table)", "IsOpen(white_box)"}
    # HandEmpty() is nobody's add effect -- it is the human operator's PRECONDITION -- so no earlier
    # phase can be held to it and it is not put to a camera here.
    assert "HandEmpty()" not in shown(expected)


def test_nothing_is_inherited_before_the_first_phase_runs():
    assert expected_before(parse(), 0, caps=CAPS) == frozenset()


def test_a_displaced_atom_is_no_longer_expected():
    # After the last phase the toy is in the box, and no longer on the table the first phase put it
    # on. Expecting both would send the camera looking for a toy in two places.
    assert shown(expected_before(parse(), 3, caps=CAPS)) == {"IsOpen(white_box)", "On(blue_toy, white_box)"}


def test_a_deleted_atom_is_no_longer_expected():
    assert shown(expected_before(parse(_open_close()), 2, caps=CAPS)) == {"IsClosed(white_box)"}


def test_what_was_true_from_the_start_is_never_inherited():
    # No earlier phase is on the hook for it, so it is never something the plan depends on.
    measured = Atom("IsOpen", ("white_box",))
    assert measured not in expected_before(parse(), 1, {measured}, caps=CAPS)


@pytest.mark.parametrize("index", [-1, 4])
def test_there_is_no_phase_outside_the_plan_to_enter(index):
    with pytest.raises(IndexError, match="expected 0 to 3"):
        expected_before(parse(), index, caps=CAPS)


# --- wasted_robot_move ----------------------------------------------------------------------------


def _cloth_then_puzzle():
    return parse(
        _phases(
            {
                "executor": "robot",
                "description": "place the toy on the cloth",
                "atoms": _atoms(("On", ["pink_toy", "yellow_cloth"])),
            },
            {
                "executor": "robot",
                "description": "solve the puzzle",
                "atoms": _atoms(("On", ["pink_toy", "puzzle_board"])),
            },
        ),
        objects=["pink_toy", "yellow_cloth", "puzzle_board"],
    )


def test_a_repeated_robot_move_is_reported_but_not_refused():
    """Observed on the rig: "place the toy on the cloth and solve the puzzle" planned as TWO robot
    phases -- On(pink_toy, yellow_cloth) then On(pink_toy, puzzle_board).

    The proposer read "solve the puzzle" as a pick-and-place, because On is the only thing a robot
    phase can say and putting the toy on the board looks like putting the toy on the board. The run
    did TAMP twice: the second phase picked the toy straight back up and dropped it on the board,
    undoing the first to achieve nothing, and the puzzle was of course not solved.

    Reported, NOT refused: consecutive robot phases that move the same object are split into two
    legs, and that robot-to-robot continuation is a supported shape. The prompt is where this is
    prevented; this is what makes a recurrence visible in the log.
    """
    reason = wasted_robot_move(_cloth_then_puzzle().phases, caps=CAPS)
    assert reason and "pink_toy" in reason
    assert "HUMAN phase" in reason, "the message must say what to do about it, not just that it happened"

    # The plan it should have produced -- "solve the puzzle" as the human's -- says nothing.
    fixed = parse(
        {
            "new_predicates": [
                {"name": "IsSolved", "instructions": "every piece of {0} sits flush in its own cut-out"}
            ],
            **_phases(
                {
                    "executor": "robot",
                    "description": "place the toy on the cloth",
                    "atoms": _atoms(("On", ["pink_toy", "yellow_cloth"])),
                },
                {
                    "executor": "human",
                    "description": "solve the puzzle",
                    "atoms": _atoms(("IsSolved", ["puzzle_board"])),
                    "instructions": "Fit each piece into its matching cut-out.",
                    "operator": _op(
                        "Solve",
                        ["puzzle_board"],
                        pre=[("HandEmpty", [])],
                        add=[("IsSolved", ["puzzle_board"])],
                    ),
                },
            ),
        },
        objects=["pink_toy", "yellow_cloth", "puzzle_board"],
    )
    assert wasted_robot_move(fixed.phases, caps=CAPS) is None


def test_the_report_describes_the_robot_from_its_declaration():
    # "Not really a pick-and-place" is a statement about one planner. The report says what THIS
    # robot can do, in the words the proposer was shown.
    reason = wasted_robot_move(_cloth_then_puzzle().phases, caps=CAPS)
    assert f"the robot can only {CAPS.robot_description}" in reason


def test_a_human_phase_between_two_robot_ones_makes_the_repeat_legitimate():
    # The canonical plan: the toy IS placed twice, but the world changed in between and the first
    # placement is what made the opening possible. Flagging this would cry wolf on the flagship plan.
    assert wasted_robot_move(parse().phases, caps=CAPS) is None


def test_two_robot_phases_sharing_only_a_SURFACE_are_not_flagged():
    # The false positive worth guarding: "put toy_a on the table" and "put toy_b on the table" name
    # `table` in common while moving nothing in common. Asking Phase.objects instead of phase_moves
    # would flag the commonest plan there is.
    spec = parse(
        _phases(
            {
                "executor": "robot",
                "description": "put toy_a on the table",
                "atoms": _atoms(("On", ["toy_a", "table"])),
            },
            {
                "executor": "robot",
                "description": "put toy_b on the table",
                "atoms": _atoms(("On", ["toy_b", "table"])),
            },
        ),
        objects=["toy_a", "toy_b"],
    )
    assert phase_moves(spec.phases[0], caps=CAPS) == {"toy_a"}, "the surface is not something the phase moves"
    assert wasted_robot_move(spec.phases, caps=CAPS) is None


def test_a_human_operator_moves_what_its_add_effects_place():
    insert = parse(_open_close()).phases[2]
    assert phase_moves(insert, caps=CAPS) == {"blue_toy"}


def test_a_backend_that_declares_no_moved_arguments_gets_no_report():
    quiet = dataclasses.replace(CAPS, moved_arguments={})
    assert phase_moves(_cloth_then_puzzle().phases[0], caps=quiet) == frozenset()
    assert wasted_robot_move(_cloth_then_puzzle().phases, caps=quiet) is None


# --- a backend with another goal language ---------------------------------------------------------

# A planner that drops items into bins: one goal predicate, its own type names, and the same two
# declarations TipTop makes about On. Nothing in the contract check may assume On exists.
IN_BIN = Predicate("InBin", (Parameter("item", "item"), Parameter("bin", "bin")))
TOY = Capabilities(
    name="toy",
    goal_predicates={"InBin": IN_BIN},
    robot_description="drop an item into a bin",
    goal_predicate_wire_names={"InBin": "in_bin"},
    achievable_predicates=frozenset({"InBin"}),
    reserved_predicate_names=frozenset({"InBin"}),
    movable_type="item",
    surface_type="bin",
    predicate_descriptions={"InBin": "{0} is lying inside {1}"},
    exclusive_arguments={"InBin": 0},
    moved_arguments={"InBin": 0},
)

TOY_OBJECTS = ["apple", "red_bin", "blue_bin"]

# The apple goes into the red bin, then the blue one, and a person then lids the red bin "with the
# apple in it" -- which it no longer is.
TOY_RESPONSE = {
    "new_predicates": [{"name": "IsLidded", "instructions": "the bin {0} has its lid on"}],
    "phases": [
        {
            "executor": "robot",
            "description": "apple into the red bin",
            "atoms": _atoms(("InBin", ["apple", "red_bin"])),
        },
        {
            "executor": "robot",
            "description": "apple into the blue bin",
            "atoms": _atoms(("InBin", ["apple", "blue_bin"])),
        },
        {
            "executor": "human",
            "description": "lid the red bin over the apple",
            "instructions": "lid it",
            "atoms": _atoms(("IsLidded", ["red_bin"])),
            "operator": _op(
                "Lid", ["red_bin"], pre=[("InBin", ["apple", "red_bin"])], add=[("IsLidded", ["red_bin"])]
            ),
        },
    ],
}


def test_a_backend_with_its_own_goal_language_gets_the_same_contract_check():
    spec = parse(TOY_RESPONSE, objects=TOY_OBJECTS, caps=TOY)
    traces = simulate_phases(spec, caps=TOY)
    assert shown(traces[2].before) == {"InBin(apple, blue_bin)"}

    broken = check_plan_effects(spec, caps=TOY)
    assert broken.startswith(
        "phase 2 ('lid the red bin over the apple') (Lid(red_bin)) requires InBin(apple, red_bin)"
    )
    assert "an earlier phase deletes it" in broken


def test_a_backend_with_its_own_goal_language_is_told_about_wasted_moves_in_its_own_terms():
    spec = parse(TOY_RESPONSE, objects=TOY_OBJECTS, caps=TOY)
    reason = wasted_robot_move(spec.phases, caps=TOY)
    assert "both move apple" in reason
    assert "the robot can only drop an item into a bin" in reason
    assert "pick-and-place" not in reason


def test_the_same_backend_without_exclusivity_proves_nothing_about_the_bin():
    unexclusive = dataclasses.replace(TOY, exclusive_arguments={})
    spec = parse(TOY_RESPONSE, objects=TOY_OBJECTS, caps=unexclusive)
    assert check_plan_effects(spec, caps=unexclusive) is None
    assert shown(expected_before(spec, 2, caps=unexclusive)) == {
        "InBin(apple, blue_bin)",
        "InBin(apple, red_bin)",
    }


# --- the declaration it reads ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,hint",
    [
        # The wire spelling: the likeliest slip, and not "close" to On by difflib's ratio.
        ("on", "Did you mean 'On'?"),
        ("Holdng", "Did you mean 'Holding'?"),
        ("Stacked", "Its goal predicates are: On, Holding, HandEmpty."),
    ],
)
def test_a_misspelt_predicate_in_the_declaration_is_refused_with_a_suggestion(name, hint):
    # Ignored, "on" would match no atom and the check would quietly lose the displacement.
    misspelt = dataclasses.replace(CAPS, exclusive_arguments={name: 0})
    with pytest.raises(TandemError, match="not one of its goal predicates") as raised:
        check_plan_effects(parse(), caps=misspelt)
    assert raised.value.hint == hint


@pytest.mark.parametrize(
    "positions,expected",
    [({"On": 2}, "On takes 2 argument"), ({"HandEmpty": 0}, "HandEmpty takes 0 argument")],
)
def test_a_position_past_the_predicates_arity_is_refused(positions, expected):
    bad = dataclasses.replace(CAPS, moved_arguments=positions)
    with pytest.raises(TandemError, match=expected):
        wasted_robot_move(parse().phases, caps=bad)


def test_tiptops_declaration_is_accepted():
    spec = parse()
    simulate_phases(spec, caps=CAPS)
    phase_moves(spec.phases[0], caps=CAPS)


# --- what the module is -----------------------------------------------------------------------------


def test_the_contract_check_names_no_predicate():
    # Every predicate it reasons about comes from Capabilities. A literal "On" in here is the
    # hard-coding this module was ported to remove.
    tree = ast.parse(inspect.getsource(contracts))
    literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)}
    assert not literals & {"On", "Holding", "HandEmpty"}


def test_the_contract_check_imports_nothing_heavy():
    # It runs inside the proposal's repair loop, on the laptop path with no GPU.
    script = (
        "import sys, tandem.planning.contracts\n"
        "heavy = {'torch', 'cv2', 'pyzed', 'open3d', 'curobo', 'cutamp', 'tiptop', 'warp', 'rerun_sdk',"
        " 'google', 'PIL', 'numpy'}\n"
        "print(sorted(heavy & {m.split('.')[0] for m in sys.modules}))\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
