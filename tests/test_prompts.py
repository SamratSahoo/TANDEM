"""The phase-segmentation prompt: the paper's Appendix B for TipTop, and planner-neutral for anything else.

The prompt is part of the method. The one in the paper -- LJ tiptop cf75a68 -- is what produced
101/101 consistent plans over five tasks, and a word changed in it is an unevaluated prompt. So the
TipTop render is held to that file byte for byte, not to a paraphrase of it.

The other half is that none of it may leak into another planner's prompt. Everything that says On,
Holding or HandEmpty is a slot TipTop fills, and a backend that fills none of them has to get a
prompt that asks only for the goal language it declared. Otherwise the proposer takes the prompt's
word over the predicate menu's and writes goals the planner has never heard of.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from tandem.core.errors import TandemError
from tandem.planners.base import Capabilities
from tandem.planners.tiptop.capabilities import CAPABILITIES as TIPTOP
from tandem.planning import llm
from tandem.planning.prompts import (
    PLAN_SCHEMA,
    PROMPT_SLOTS,
    classifier_prompt,
    plan_prompt,
    prompt_fragments,
)
from tandem.planning.symbols import Parameter, Predicate

# tests/golden/plan_prompt_tiptop.txt is LJ tiptop cf75a68's OWN render, generated once and committed,
# so this suite never needs that checkout. Its prompts.py imports nothing, so it loads on its own:
#
#   import importlib.util
#   spec = importlib.util.spec_from_file_location("lj_prompts", "<tiptop@cf75a68>/tiptop/hitl/prompts.py")
#   lj = importlib.util.module_from_spec(spec)
#   spec.loader.exec_module(lj)
#   GOLDEN.write_bytes(lj.plan_prompt(INSTR, OBJS).encode("utf-8"))
#
# Bytes both ways, never text: the render has no trailing newline, and a text-mode round trip that
# added one or translated line endings would make a failure here about the file, not the prompt.
GOLDEN = Path(__file__).parent / "golden" / "plan_prompt_tiptop.txt"
# The paper's "Store Bread in Closed Box" task. Unsorted on purpose: the prompt lists them sorted.
INSTR = "place the bread on the plate, and then open the box and place the bread in the box"
OBJS = ["plate", "cardboard_box", "bread"]

# A planner with a different goal language that has written none of the prompt's slots: what the next
# backend looks like on the day it is registered.
IN_BIN = Predicate("InBin", (Parameter("obj", "item"), Parameter("bin", "bin")))
SORTER = Capabilities(
    name="sorter",
    goal_predicates={"InBin": IN_BIN},
    robot_description="drop an item into a bin",
    predicate_descriptions={"InBin": "{0} is inside {1}"},
    movable_type="item",
    surface_type="bin",
)
TIPTOP_WORDS = re.compile(r"\bOn\b|\bHolding\b|\bHandEmpty\b|pick-and-place")


def _render(caps: Capabilities) -> str:
    return plan_prompt(INSTR, OBJS, caps=caps)


# --- TipTop: the evaluated prompt -----------------------------------------------------------------


def test_tiptops_render_is_the_evaluated_prompt_byte_for_byte():
    expected = GOLDEN.read_bytes().decode("utf-8")
    assert _render(TIPTOP) == expected


def test_tiptop_fills_every_slot_in_its_own_words():
    # A slot TipTop left out would render the generic paragraph, which the golden would catch -- but
    # only for the one render it pins. This says it structurally: no part of TipTop's prompt is text
    # nobody evaluated it with.
    assert set(TIPTOP.prompt_fragments) == set(PROMPT_SLOTS)
    assert prompt_fragments(TIPTOP) == dict(TIPTOP.prompt_fragments)


def test_the_worked_example_and_new_objects_are_gone():
    # The worked examples were near-copies of the tasks the prompt ran on, and taught the tasks rather
    # than the rule; "WHAT On MEANS" replaced them. Deferred objects are not ported.
    prompt = _render(TIPTOP)
    assert "Worked example" not in prompt and "worked example" not in prompt
    assert "new_objects" not in prompt and "NEW OBJECTS" not in prompt
    assert "new_objects" not in PLAN_SCHEMA["properties"]
    assert "exactly these objects RIGHT NOW" in prompt
    # The placeholders have to survive the f-string -- and the slot splice -- as single braces.
    assert "On({0}, {1}) means {0}" in prompt and "{{" not in prompt


# --- any other planner: nothing of TipTop's -------------------------------------------------------


def test_a_planner_that_wrote_no_fragments_is_never_asked_for_tiptops_predicates():
    prompt = _render(SORTER)
    assert "On(" not in prompt
    assert TIPTOP_WORDS.search(prompt) is None, TIPTOP_WORDS.search(prompt)
    assert "MEANS" not in prompt, "the placement paragraph has no generic form and must be omitted"
    # What it is asked for instead is its own declaration.
    assert "The robot can do one thing: drop an item into a bin. That is all." in prompt
    assert "- InBin(?obj: item, ?bin: bin): {0} is inside {1}\n\nThe human can do anything" in prompt
    assert "Write them with InBin or a predicate you invented." in prompt
    assert "A ROBOT phase's atoms may use ONLY InBin. The robot cannot achieve" in prompt


def test_the_generic_prompt_is_the_same_prompt_with_the_planner_taken_out():
    # Everything that is about phase planning rather than about a planner reaches every backend: the
    # operator block, the contract-check paragraph, coverage, the rules, unrepresented.
    generic, tiptop = _render(SORTER), _render(TIPTOP)
    for line in tiptop.splitlines():
        if not TIPTOP_WORDS.search(line) and "robot can do one thing" not in line:
            assert line in generic.splitlines(), line


def test_the_robot_may_use_exactly_the_goal_language():
    # Not a slot: it is the menu restated, and a backend overriding it could only make the prompt
    # disagree with the parser that enforces it.
    lidded = Predicate("Lidded", (Parameter("bin", "bin"),))
    stacked = Predicate("Stacked", (Parameter("top", "item"), Parameter("bottom", "item")))
    two = dataclasses.replace(SORTER, goal_predicates={"InBin": IN_BIN, "Lidded": lidded})
    three = dataclasses.replace(two, goal_predicates={**two.goal_predicates, "Stacked": stacked})
    assert "may use ONLY InBin and Lidded." in _render(two)
    assert "may use ONLY InBin, Lidded and Stacked." in _render(three)
    assert "Write them with InBin, Lidded, Stacked or a predicate you invented." in _render(three)


def test_the_moved_twice_rule_comes_with_the_map_that_checks_it():
    # The rule is the prompt side of the wasted-robot-move check, which reads moved_arguments. A
    # planner that declared none gets no rule it could never be held to.
    rule = "Two ROBOT phases in a row must not move the same object twice."
    assert rule not in _render(SORTER)
    moving = dataclasses.replace(SORTER, moved_arguments={"InBin": 0})
    assert rule in _render(moving)


def test_a_fragment_replaces_its_own_slot_and_nothing_else():
    meaning = "WHAT InBin MEANS. InBin({0}, {1}) means {0} was dropped into {1}."
    caps = dataclasses.replace(SORTER, prompt_fragments={"placement_semantics": meaning})
    prompt = _render(caps)
    assert f"{{0}} is inside {{1}}\n\n{meaning}\n\nThe human can do anything" in prompt
    assert prompt.replace(f"\n\n{meaning}", "") == _render(SORTER)


def test_an_empty_slot_leaves_no_trace():
    # "" is how a backend says "nothing here". It must not leave a dangling bullet, a doubled space or
    # an extra blank line where the slot was.
    emptied = dataclasses.replace(SORTER, prompt_fragments=dict.fromkeys(PROMPT_SLOTS, ""))
    for prompt in (_render(SORTER), _render(emptied), _render(TIPTOP)):
        assert "  " not in prompt
        assert "\n\n\n" not in prompt
        assert not any(line.rstrip() != line or line.strip() == "-" for line in prompt.splitlines())
    assert "able to do it.\n- `add_effects`" in _render(emptied)
    assert "most often forgotten. An empty list is fine" in _render(emptied)


def test_a_misspelt_slot_is_refused_rather_than_rendered_generic():
    # Ignored, the typo would quietly swap the backend's evaluated paragraph for the generic one.
    caps = dataclasses.replace(SORTER, prompt_fragments={"placement_semantic": "WHAT InBin MEANS."})
    with pytest.raises(TandemError) as caught:
        _render(caps)
    assert "placement_semantic" in caught.value.message and "sorter" in caught.value.message
    assert caught.value.hint == "Did you mean 'placement_semantics'?"


# --- the schema -----------------------------------------------------------------------------------


def test_a_phase_can_carry_an_operator_whose_contract_is_all_required():
    phase = PLAN_SCHEMA["properties"]["phases"]["items"]
    operator = phase["properties"]["operator"]
    assert operator["required"] == ["name", "args", "preconditions", "add_effects", "delete_effects"]
    for effects in ("preconditions", "add_effects", "delete_effects"):
        assert operator["properties"][effects]["items"] is phase["properties"]["atoms"]["items"]
    # Optional at the phase level: a robot phase has none, and the parser is what makes it mandatory
    # on a human phase -- with a message the model can act on, which a schema failure is not.
    assert "operator" not in phase["required"]


# --- the other two Appendix-B prompts, which were already right -----------------------------------


def test_the_repair_prompt_is_appendix_b():
    assert llm._REPROMPT == (
        "Your previous response was rejected.\n"
        "\n"
        "Your response was:\n"
        "{response}\n"
        "\n"
        "The problem was:\n"
        "{error}\n"
        "\n"
        "Try again, fixing exactly that problem and keeping everything else that was correct."
    )


def test_the_classifier_prompt_is_appendix_b():
    assert classifier_prompt("IsOpen(cardboard_box)") == (
        "You are the perception system of a robot. Look at the image of the robot's workspace and "
        "decide whether the following statement is true right now.\n"
        "\n"
        "Statement: IsOpen(cardboard_box)\n"
        "\n"
        "Judge only what you can see. If the workspace does not clearly show the statement to be true, "
        "it is false. Give a one-sentence reason for your answer."
    )
