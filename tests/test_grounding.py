"""Checking a phase against a camera: preconditions, add effects, and delete effects.

A human phase is checked against its operator's contract. Its preconditions are read off a frame
before the hand-off. After it, the add effects must hold and the delete effects must NOT. The second
half is the reason ``Verdict`` keeps ``holds`` (what the model saw) apart from ``expected`` (what the
plan said). For a delete effect "yes, it holds" is the failure, and a check that branched on
``holds`` would pass exactly the phases it should fail.

No network. The model is a fake client that answers from a table keyed by the statement it is
asked about. Going through the real ``query_json``, ``classify`` and the classifier prompt, rather
than patching ``classify`` itself, is what lets these tests pin down WHICH question the model is
asked, not just what is done with the answer.

The running example is a box that the person first opens and then closes again: ``Close(white_box)``
needs the box open, makes it closed, and undoes the open.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from unittest import mock

import pytest

from tandem.planners import registry
from tandem.planning import grounding, llm
from tandem.planning.config import PlanningConfig
from tandem.planning.prompts import classifier_prompt
from tandem.planning.structs import HumanOperator, Phase, SceneTypes, TaskSpecification, VLMPredicate
from tandem.planning.symbols import Atom, Parameter, Predicate

CAPS = registry.capabilities("tiptop")
CFG = PlanningConfig(enabled=True)

IS_OPEN = Atom("IsOpen", ("white_box",))
IS_CLOSED = Atom("IsClosed", ("white_box",))
HAND_EMPTY = Atom("HandEmpty")
HOLDING_TOY = Atom("Holding", ("blue_toy",))
TOY_ON_TABLE = Atom("On", ("blue_toy", "table"))
TOY_IN_BOX = Atom("On", ("blue_toy", "white_box"))

OPEN_TEXT = "the container white_box is open"
CLOSED_TEXT = "the container white_box is shut"
TOY_ON_TABLE_TEXT = "blue_toy is resting on top of table"
TOY_IN_BOX_TEXT = "blue_toy is resting on top of white_box"

INVENTED = (
    VLMPredicate(Predicate("IsOpen", (Parameter("x0", "surface"),)), "the container {0} is open"),
    VLMPredicate(Predicate("IsClosed", (Parameter("x0", "surface"),)), "the container {0} is shut"),
)
SCENE = SceneTypes(surfaces=frozenset({"white_box", "table"}), movables=frozenset({"blue_toy"}))
BOX = (Parameter("x0", "surface"),)


def open_box() -> HumanOperator:
    """``Open(white_box)``: needs an empty hand, which no camera is asked about, and opens the box."""
    return HumanOperator("Open", ("white_box",), BOX, preconditions={HAND_EMPTY}, add_effects={IS_OPEN})


def close_box() -> HumanOperator:
    """``Close(white_box)``: needs the box open, makes it closed, and undoes the open."""
    return HumanOperator(
        "Close",
        ("white_box",),
        BOX,
        preconditions={IS_OPEN, HAND_EMPTY},
        add_effects={IS_CLOSED},
        delete_effects={IS_OPEN},
    )


def human(description, atoms, operator=None) -> Phase:
    return Phase("human", description, frozenset(atoms), "do it", operator)


OPEN_PHASE = human("open the box", {IS_OPEN}, open_box())
CLOSE_PHASE = human("close the box", {IS_CLOSED}, close_box())


class _FakeVLM:
    """A client that answers each classifier question from ``answers``, keyed by the statement.

    Every prompt is recorded, so a test can check the question as well as the answer. A statement
    missing from the table is an error rather than a default answer: a question the test did not
    expect is exactly what these tests exist to catch (HandEmpty put to a camera, a negated
    statement).
    """

    def __init__(self, answers: dict[str, bool]):
        self.answers = dict(answers)
        self.prompts: list[str] = []
        self.aio = self

    @property
    def models(self):
        return self

    async def generate_content(self, model, contents, config):
        prompt = contents[-1]
        self.prompts.append(prompt)
        statement = re.search(r"^Statement: (.*)$", prompt, re.MULTILINE).group(1)
        if statement not in self.answers:
            raise AssertionError(f"the model was asked about something unexpected: {statement!r}")
        return mock.Mock(text=json.dumps({"holds": self.answers[statement], "reason": "looked"}))

    @property
    def statements(self) -> list[str]:
        return sorted(re.search(r"^Statement: (.*)$", p, re.MULTILINE).group(1) for p in self.prompts)


def run(coro_fn, answers, *args, **kwargs):
    """Run one grounding coroutine against a fake model. Returns ``(result, client)``."""
    client = _FakeVLM(answers)
    with mock.patch.object(llm, "gemini_client", lambda: client):
        return asyncio.run(coro_fn(*args, **kwargs)), client


def check(verify, phase, answers):
    """``verify(image, phase, INVENTED, CFG, CAPS)`` against a fake model answering ``answers``."""
    return run(verify, answers, None, phase, INVENTED, CFG, CAPS)


# --- effects: add must hold, delete must not ------------------------------------------------------


def test_a_delete_effect_that_is_still_true_fails_the_phase():
    # The reason Verdict separates `holds` from `expected`: for a delete effect "holds: true" IS the
    # failure, and branching on `holds` would silently invert the check.
    (ok, verdicts), _ = check(grounding.verify_effects, CLOSE_PHASE, {CLOSED_TEXT: True, OPEN_TEXT: True})
    assert not ok
    deleted = next(v for v in verdicts if v.atom == IS_OPEN)
    assert deleted.holds and not deleted.expected and not deleted.satisfied
    assert deleted.role == "effect (deleted)"
    # And the operator is told the right way round, not asked to redo what they already did.
    assert grounding.missing_statements(verdicts) == [f"{OPEN_TEXT} -- and it should no longer be"]


def test_a_phase_passes_when_its_add_effects_hold_and_its_delete_effects_do_not():
    (ok, verdicts), _ = check(grounding.verify_effects, CLOSE_PHASE, {CLOSED_TEXT: True, OPEN_TEXT: False})
    assert ok
    assert all(v.satisfied for v in verdicts)
    # Add effects first, then the deleted ones, each labelled with what it was checked as.
    assert [(v.atom, v.role) for v in verdicts] == [(IS_CLOSED, "effect"), (IS_OPEN, "effect (deleted)")]
    assert grounding.missing_statements(verdicts) == []


def test_an_add_effect_that_never_became_true_still_fails_the_phase():
    (ok, verdicts), _ = check(grounding.verify_effects, CLOSE_PHASE, {CLOSED_TEXT: False, OPEN_TEXT: False})
    assert not ok
    assert grounding.missing_statements(verdicts) == [CLOSED_TEXT]


def test_the_classifier_is_only_ever_asked_the_positive_statement():
    # Confirming a negative ("the box is NOT open") reads as a double negative and got worse
    # answers, so a delete effect is asked exactly like an add effect -- the same Appendix-B prompt
    # about the same positive sentence -- and `expected` is applied to the answer afterwards.
    _, client = check(grounding.verify_effects, CLOSE_PHASE, {CLOSED_TEXT: True, OPEN_TEXT: False})
    assert sorted(client.prompts) == sorted([classifier_prompt(OPEN_TEXT), classifier_prompt(CLOSED_TEXT)])


def test_a_phase_without_an_operator_is_checked_on_its_atoms_alone():
    # Every plan proposed before operators existed, and a robot phase handed to a person: the atoms
    # are the add effects and nothing is deleted, which is exactly the check such a phase always got.
    plain = human("put the toy in the box", {TOY_IN_BOX, HAND_EMPTY})
    (ok, verdicts), client = run(
        grounding.verify_effects, {TOY_IN_BOX_TEXT: True}, None, plain, (), CFG, CAPS
    )
    assert ok
    assert [(v.atom, v.expected, v.role) for v in verdicts] == [(TOY_IN_BOX, True, "effect")]
    assert client.statements == [TOY_IN_BOX_TEXT], "HandEmpty is not put to a camera"


def test_verify_phase_is_still_there_and_now_checks_delete_effects_too():
    # The phase loop calls it by its old name until it moves over to verify_effects.
    (ok, verdicts), _ = check(grounding.verify_phase, CLOSE_PHASE, {CLOSED_TEXT: True, OPEN_TEXT: True})
    assert not ok
    assert {v.role for v in verdicts} == {"effect", "effect (deleted)"}


# --- preconditions ---------------------------------------------------------------------------------


def test_preconditions_are_checked_as_their_own_role():
    (ok, verdicts), client = check(grounding.verify_preconditions, CLOSE_PHASE, {OPEN_TEXT: True})
    assert ok
    assert [v.role for v in verdicts] == ["precondition"]
    assert verdicts[0].atom == IS_OPEN and verdicts[0].expected
    # HandEmpty() is a precondition too, but the gripper is the robot's to know, not the camera's.
    assert client.statements == [OPEN_TEXT]


def test_an_unmet_precondition_fails_the_check():
    (ok, verdicts), _ = check(grounding.verify_preconditions, CLOSE_PHASE, {OPEN_TEXT: False})
    assert not ok
    assert grounding.missing_statements(verdicts) == [OPEN_TEXT]


def test_a_phase_with_nothing_checkable_before_it_passes_without_asking():
    # No operator: no preconditions at all. Open(white_box): only HandEmpty(), which no camera
    # judges. Either way nothing is put to the model, so an empty fake that raises on any question
    # proves none was asked.
    robot = Phase("robot", "toy to the table", frozenset({TOY_ON_TABLE}))
    for phase in (robot, OPEN_PHASE):
        (ok, verdicts), client = check(grounding.verify_preconditions, phase, {})
        assert ok and verdicts == [] and client.prompts == []


# --- what the camera may judge ---------------------------------------------------------------------


def test_invented_predicates_are_always_checkable_and_base_ones_only_when_declared():
    atoms = [TOY_ON_TABLE, HAND_EMPTY, HOLDING_TOY, IS_OPEN]
    # TipTop declares On, and only On: a third-person frame cannot settle the gripper.
    assert grounding.checkable(atoms, INVENTED, CAPS) == [IS_OPEN, TOY_ON_TABLE]
    # A planner that declares nothing checkable leaves only what the proposer invented.
    silent = dataclasses.replace(CAPS, checkable_predicates=frozenset())
    assert grounding.checkable(atoms, INVENTED, silent) == [IS_OPEN]
    # The backend's declaration decides, not a predicate name written into tandem. A planner whose
    # camera can see its own gripper may declare Holding.
    wristy = dataclasses.replace(CAPS, checkable_predicates=frozenset({"Holding"}))
    assert grounding.checkable(atoms, INVENTED, wristy) == [HOLDING_TOY, IS_OPEN]
    # An invented predicate this plan did not invent is nobody's to judge.
    assert grounding.checkable([IS_OPEN], (), silent) == []


def test_an_atom_expected_both_to_hold_and_not_is_refused_rather_than_failed():
    # Such a check fails whatever the workspace looks like. The parser refuses an operator that
    # adds and deletes the same atom, so this is a bug upstream and must not read as a botched step.
    with pytest.raises(ValueError, match=r"IsOpen\(white_box\) both holding and not holding"):
        run(
            grounding.verify_atoms,
            {},
            None,
            INVENTED,
            CFG,
            CAPS,
            expect_true=[IS_OPEN],
            expect_false=[IS_OPEN],
        )


# --- what the operator is told ---------------------------------------------------------------------


def test_missing_statements_reads_satisfied_and_turns_a_delete_effect_around():
    verdicts = [
        grounding.Verdict(IS_CLOSED, CLOSED_TEXT, False, "lid up", expected=True),
        grounding.Verdict(IS_OPEN, OPEN_TEXT, True, "lid up", expected=False, role="effect (deleted)"),
        # Satisfied either way round, so neither is mentioned.
        grounding.Verdict(TOY_IN_BOX, TOY_IN_BOX_TEXT, True, "", expected=True),
        grounding.Verdict(TOY_ON_TABLE, TOY_ON_TABLE_TEXT, False, "", expected=False),
    ]
    assert grounding.missing_statements(verdicts) == [
        CLOSED_TEXT,
        f"{OPEN_TEXT} -- and it should no longer be",
    ]


def test_the_phase_loop_tells_the_operator_what_missing_statements_does():
    # The loop builds the retry message from the verdicts' summary dicts, as the events carry them,
    # not from the Verdicts. Reading `holds` there would list the delete effect that was undone and
    # drop the one still true, so the operator would be sent to redo what they got right.
    from tandem.core.phase_loop import _missing_from

    verdicts = [
        grounding.Verdict(IS_CLOSED, CLOSED_TEXT, False, "", expected=True),
        grounding.Verdict(IS_OPEN, OPEN_TEXT, True, "", expected=False, role="effect (deleted)"),
        grounding.Verdict(TOY_ON_TABLE, TOY_ON_TABLE_TEXT, False, "", expected=False),
    ]
    summaries = [v.summary() for v in verdicts]
    assert _missing_from(summaries) == grounding.missing_statements(verdicts)
    # The model's reason still rides along, and a summary from before verdicts had an expectation
    # (only `holds`) still reads as it always did.
    assert _missing_from([{**summaries[0], "reason": "lid up"}]) == [f"{CLOSED_TEXT} — lid up"]
    assert _missing_from([{"statement": CLOSED_TEXT, "holds": False}]) == [CLOSED_TEXT]
    assert _missing_from([{"statement": CLOSED_TEXT, "holds": True}]) == []


def test_the_hand_off_names_what_should_stop_being_true():
    # A step whose whole point is that something stops being the case reads as a missing
    # instruction if only the add effects are shown. Everything is listed, checkable or not.
    descriptions = grounding.descriptions_for(INVENTED, CAPS)
    assert grounding.describe_expectations(CLOSE_PHASE, descriptions) == [
        CLOSED_TEXT,
        f"NO LONGER: {OPEN_TEXT}",
    ]
    plain = human("put the toy in the box", {TOY_IN_BOX, HAND_EMPTY})
    assert grounding.describe_expectations(plain, descriptions) == [
        "the robot's gripper is empty",
        TOY_IN_BOX_TEXT,
    ]


def test_a_verdict_records_what_was_expected_and_whether_it_was_met():
    verdict = grounding.Verdict(IS_OPEN, OPEN_TEXT, True, "lid up", expected=False, role="effect (deleted)")
    assert verdict.summary() == {
        "atom": "IsOpen(white_box)",
        "statement": OPEN_TEXT,
        "holds": True,
        "expected": False,
        "satisfied": False,
        "role": "effect (deleted)",
        "reason": "lid up",
    }
    # The defaults are the add-effect check every verdict used to be.
    plain = grounding.Verdict(IS_OPEN, OPEN_TEXT, True, "")
    assert (plain.expected, plain.role, plain.satisfied) == (True, "effect", True)


# --- the starting state ----------------------------------------------------------------------------


def test_the_starting_state_covers_atoms_named_only_inside_an_operator():
    # "The box starts open, and the person closes it": IsOpen is named only as Close's precondition
    # and delete effect, never in any phase's atoms, and it is exactly the atom whose starting value
    # the plan cannot derive. Base predicates (On) are the planner's to perceive, not asked here.
    spec = TaskSpecification(
        instruction="close the box, then put the toy on the table",
        phases=(CLOSE_PHASE, Phase("robot", "toy to the table", frozenset({TOY_ON_TABLE}))),
        scene_types=SCENE,
        invented=INVENTED,
    )
    initially_true, client = run(
        grounding.classify_initial_state, {OPEN_TEXT: True, CLOSED_TEXT: False}, None, spec, CFG, CAPS
    )
    assert initially_true == frozenset({IS_OPEN})
    assert client.statements == sorted([CLOSED_TEXT, OPEN_TEXT])
