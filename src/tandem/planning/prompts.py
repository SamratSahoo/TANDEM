"""The two prompts: plan the task into phases, and check one statement against an image.

The planner is described to the model in ABSTRACT terms — "pick an object up and place it on a
surface" — rather than with its real operator signatures. Those carry motion-level parameters
(``conf``, ``traj``, ``grasp``) and bookkeeping predicates (``At``, ``CanMove``, ``JustMoved``) that
a proposer has no business reasoning about, and shown them it writes goals over the alternation lock.
Restricting the vocabulary to the few state predicates a sub-goal can be phrased in keeps every phase
groundable by construction.

The phase-segmentation prompt is the one the method was evaluated with: the paper's Appendix B, which
is LJ tiptop ``cf75a68``'s ``tiptop/hitl/prompts.py``. It was written for one planner, and parts of
it are statements about cuTAMP's goal language rather than about phase planning -- what On means,
which predicates a precondition may be written in. Those parts are SLOTS, filled from
``Capabilities.prompt_fragments`` (``PROMPT_SLOTS`` is the contract), and the robot's one-line
description and the predicate menu come from the same declaration, so the template itself names no
planner predicate.

Everything else is wording, and the wording is the record of what went wrong before: each rule below
is in the prompt because a proposal without it produced a plan that was wrong in that exact way.
``tests/golden/plan_prompt_tiptop.txt`` holds the TipTop render to LJ's byte for byte, so a change
here is a change to the method and gets made on purpose.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from typing import TYPE_CHECKING

from tandem.core.errors import TandemError

if TYPE_CHECKING:
    from tandem.planners.base import Capabilities

# The backend-specific parts of the phase-segmentation prompt, by the key a backend declares them
# under in ``Capabilities.prompt_fragments``. Each value is spliced in VERBATIM, so it is written the
# way it should read: no leading or trailing whitespace, a bullet as one line. A slot the backend
# leaves out gets the generic text from ``_generic_fragments``; one it declares as "" renders nothing.
#
#   placement_semantics         A whole paragraph after the predicate menu saying what the placement
#                               predicate does and does not cover -- TipTop's "WHAT On MEANS.", which
#                               is what replaced the worked examples. Generic: none; there is no
#                               honest way to say what a predicate means without knowing it.
#   precondition_vocabulary     The sentences that finish the operator's `preconditions` bullet,
#                               after "...to be able to do it.". Generic: "Write them with <the goal
#                               predicates> or a predicate you invented."
#   delete_effect_example       The example in the `delete_effects` bullet, between "most often
#                               forgotten." and "An empty list is fine...". Generic: an invented
#                               predicate's example only.
#   work_division               Bullet line(s) opening "How to divide the work:", saying which steps
#                               are the robot's. Generic: "every step it can do".
#   intermediate_state_example  The bullet line on intermediate states, after ORDER MATTERS.
#                               Generic: the same point with no predicate in it.
#   robot_phase_rules           Bullet line(s) in the rules, after the one saying which predicates a
#                               robot phase may use. Generic: the moved-twice rule when the backend
#                               declares moved_arguments (the contract check reads the same map), and
#                               nothing otherwise.
#
# Which predicates a ROBOT phase may use is not a slot: it is the goal language, rendered from
# goal_predicates, and a backend overriding it could only make the prompt disagree with the parser.
PROMPT_SLOTS = (
    "placement_semantics",
    "precondition_vocabulary",
    "delete_effect_example",
    "work_division",
    "intermediate_state_example",
    "robot_phase_rules",
)

_PLACEHOLDER_NOTE = (
    "Write the text with {0}, {1}, ... standing in for the arguments, in the order they are declared."
)

_INVENTION_NOTE = (
    "Invent a new predicate only for something no existing predicate can express. A new predicate is "
    "evaluated by a vision-language model looking at a camera image of the workspace, so it needs "
    "`instructions`: a description of what must be VISIBLE in the image for it to be true. "
    + _PLACEHOLDER_NOTE
)

_ATOM_ITEM = {
    "type": "object",
    "properties": {
        "predicate": {"type": "string"},
        "args": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["predicate", "args"],
}

# A human phase's operator. `name`/`args` are the action itself (`Push`, ["box"]); the three atom
# lists are its contract. Only `delete_effects` is genuinely optional -- plenty of human steps take
# nothing away -- but it is asked for explicitly so an empty list is a STATEMENT rather than an
# omission, which is the difference between "this undoes nothing" and "nobody said".
_OPERATOR_ITEM = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "args": {"type": "array", "items": {"type": "string"}},
        "preconditions": {"type": "array", "items": _ATOM_ITEM},
        "add_effects": {"type": "array", "items": _ATOM_ITEM},
        "delete_effects": {"type": "array", "items": _ATOM_ITEM},
    },
    "required": ["name", "args", "preconditions", "add_effects", "delete_effects"],
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        # No argument types are asked for: they are read off how the predicate is USED, where every
        # argument is a real object. Asking for them produced placeholder answers -- "container",
        # "cover_object" -- naming nothing in the scene.
        "new_predicates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "instructions": {"type": "string"}},
                "required": ["name", "instructions"],
            },
        },
        "phases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "executor": {"type": "string", "enum": ["robot", "human"]},
                    "description": {"type": "string"},
                    "atoms": {"type": "array", "items": _ATOM_ITEM},
                    "instructions": {"type": "string"},
                    # The explicit operator a HUMAN phase is, stated the way a planner states its
                    # own: what must hold first, what becomes true, what stops being true. Asked for
                    # per phase rather than as a lifted library because the proposer has already
                    # decided this step happens here, to these objects -- and the grounded instance
                    # is the only form a camera can be asked about. Optional at this level because a
                    # robot phase has none; the parser is what makes it mandatory on a human phase.
                    "operator": _OPERATOR_ITEM,
                },
                "required": ["executor", "description", "atoms"],
            },
        },
        # Forces the instruction to be enumerated clause by clause and each clause pinned to a phase.
        # Without it the model plans the first clause, over-decomposes it, and stops -- observed
        # answering a three-clause instruction with two phases, both for clause one.
        "coverage": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "clause": {"type": "string"},
                    "phase": {"type": "integer"},
                },
                "required": ["clause", "phase"],
            },
        },
        # Where a dropped clause goes. Without somewhere to put it, the only way to answer at all is
        # to leave it out, and the run then does most of the task and reports success.
        "unrepresented": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"clause": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["clause", "reason"],
            },
        },
    },
    "required": ["phases"],
}

CLASSIFIER_SCHEMA = {
    "type": "object",
    "properties": {"holds": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["holds", "reason"],
}


def _and_list(names: Sequence[str]) -> str:
    """``A``, ``A and B``, ``A, B and C`` -- the evaluated prompt's own spelling, no Oxford comma."""
    if not names:
        return "the predicates listed above"
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _generic_fragments(caps: Capabilities) -> dict[str, str]:
    """What each slot says for a backend that has not written its own.

    Rendered from the declaration and never from TipTop's wording: a generic prompt that mentioned
    On would ask a planner with a different goal language for something it does not have, and the
    proposer would take the prompt's word over the menu's.
    """
    names = list(caps.goal_predicates)
    vocabulary = f"{', '.join(names)} or " if names else ""
    moved_twice = (
        "- Two ROBOT phases in a row must not move the same object twice. The second throws the first "
        "one away, so the first is wasted motion -- and it almost always means a step the robot cannot "
        "really do was given to it. If a human phase belongs between them, put it there; if the second "
        "phase is the one the robot cannot do, make IT the human phase."
    )
    return {
        "placement_semantics": "",
        "precondition_vocabulary": f"Write them with {vocabulary}a predicate you invented.",
        "delete_effect_example": (
            "If the person unplugs what an earlier phase plugged in, the IsPluggedIn(...) is a delete effect."
        ),
        "work_division": (
            "- Give the robot every step it can do. A human phase that does something the robot could "
            "have done is taking the robot's work away from it."
        ),
        "intermediate_state_example": (
            "- Intermediate states are fine and often necessary: a phase may leave an object somewhere "
            "the instruction never mentions so that a later phase becomes possible, and the same object "
            "may be handled in more than one phase."
        ),
        "robot_phase_rules": moved_twice if caps.moved_arguments else "",
    }


def prompt_fragments(caps: Capabilities) -> dict[str, str]:
    """Every slot's text for this backend: its own where it declared one, the generic one elsewhere.

    A key that is not a slot is refused rather than ignored. Ignored, a misspelt key renders the
    generic paragraph in place of the backend's evaluated wording, and nothing anywhere says the
    prompt the planner was tuned against is not the one being sent.
    """
    for key in caps.prompt_fragments:
        if key not in PROMPT_SLOTS:
            suggestion = difflib.get_close_matches(key, PROMPT_SLOTS, n=1, cutoff=0.6)
            hint = (
                f"Did you mean {suggestion[0]!r}?"
                if suggestion
                else f"The slots are: {', '.join(PROMPT_SLOTS)}."
            )
            raise TandemError(
                f"The {caps.name!r} planner declares a prompt fragment for {key!r}, which is not a "
                "slot of the phase-planning prompt.",
                hint=hint,
            )
    return {**_generic_fragments(caps), **caps.prompt_fragments}


def _line(text: str) -> str:
    """A bullet slot, followed by the newline the next bullet needs -- or nothing at all."""
    return f"{text}\n" if text else ""


def _sentence(text: str) -> str:
    """A sentence slot, with the space that separates it from the one before -- or nothing at all."""
    return f" {text}" if text else ""


# No worked examples and no `new_objects`. The shipped examples were near-copies of the tasks this
# prompt is run on, so they were answering those tasks rather than teaching the rule behind them --
# with them removed, "put the bread in the (opened) box" went to the human 12 times in 20. Stating
# the loose-placement / precise-fit line as a rule (TipTop's "WHAT On MEANS") replaced them: 101/101
# plans consistent with the ones that succeeded on the robot, over five tasks (LJ tiptop's
# offline_eval on the tamp_v3 runs, 2026-09-22). The inline illustrations are deliberately unlike
# those tasks too.
def plan_prompt(instruction: str, objects: Sequence[str], *, caps: Capabilities) -> str:
    """Turn an instruction into an ordered list of robot and human phases."""
    fragment = prompt_fragments(caps)
    object_list = "\n".join(f"- {name}" for name in sorted(objects))
    # Worked out here rather than inline: an f-string expression may not reuse the string's own
    # quote character before Python 3.12, and the slot names are easier to audit in one place.
    placement = f"\n\n{fragment['placement_semantics']}" if fragment["placement_semantics"] else ""
    preconditions = _sentence(fragment["precondition_vocabulary"])
    delete_example = _sentence(fragment["delete_effect_example"])
    work_division = _line(fragment["work_division"])
    intermediate_states = _line(fragment["intermediate_state_example"])
    robot_rules = _line(fragment["robot_phase_rules"])
    robot_vocabulary = _and_list(list(caps.goal_predicates))
    return f"""\
A robot and a human share a workspace. Break the instruction below into an ORDERED list of phases. \
Each phase is done either by the robot or by the human, and they happen in the order you give.

THE INSTRUCTION:
{instruction}

The image shows the workspace as it is right now. Plan the INSTRUCTION -- the picture is context for \
where things are, not a task in itself.

The workspace contains exactly these objects RIGHT NOW, and no others:
{object_list}

The robot can do one thing: {caps.robot_description}. That is all. It plans and executes each of its \
phases itself; you only say what must be TRUE when the phase is finished, using these predicates:
{caps.predicate_menu()}{placement}

The human can do anything the robot cannot -- open, close, fold, unfold, tie, flatten, rotate, \
manipulate cloth. Give a human phase `instructions` addressed to the person, and `atoms` saying what \
should be true afterwards. That is what a camera will be used to check, so it must be visible.

EVERY HUMAN PHASE IS AN OPERATOR, AND YOU MUST SAY WHAT IT IS. Give the phase an `operator` with:
- `name`: what the action is, one word, capitalised -- Push, Unplug, Staple, Insert, Tie, Wipe.
- `args`: the objects it acts on, from the object list, in the order the name reads.
- `preconditions`: what must ALREADY be true for the person to be able to do it.{preconditions}
- `add_effects`: what becomes true. Every atom you put in the phase's `atoms` must appear here.
- `delete_effects`: what STOPS being true. This is the one most often forgotten.{delete_example} An \
empty list is fine and means "this takes nothing away" -- but say it.

The preconditions and effects are CHECKED against a camera image, before and after the person acts, \
and they are checked against each other across your whole plan. So they have to be true statements \
about the world, not decoration: do not list a precondition that nothing in your plan makes true, \
and do not delete something a later phase still needs.

How to divide the work:
{work_division}- Use a human phase only for something the robot genuinely cannot do.
- ORDER MATTERS AND IS YOURS TO SET. If a jar must be unscrewed before anything can go in it, the \
human's "unscrew the lid" phase comes BEFORE the robot's "put the coin in the jar" phase. If a rubber \
band goes around a bundle of pencils, the robot places the pencils first and the human puts the band \
on afterwards. Think about what has to be true for the next phase to be physically possible.
{intermediate_states}- Do not add phases the instruction does not ask for, and do not merge two of its steps into one.

PLAN THE WHOLE INSTRUCTION. Work through it clause by clause and give every clause a phase, in the \
order stated. Fill in `coverage` with one entry per clause, naming the phase that carries it out \
(the index in your `phases` list), or -1 if you had to leave it out. Phases are numbered from 0: the \
first phase is 0, and the last is one less than the number of phases. The most common mistake is to \
plan the first clause carefully and stop; the last phase must leave the workspace as the END of the \
instruction describes.

Rules, all of which are checked:
- A ROBOT phase's atoms may use ONLY {robot_vocabulary}. The robot cannot achieve a predicate \
you invent -- if a phase needs one, it is a human phase.
{robot_rules}- Every phase needs at least one atom, and a human phase needs `instructions` and an `operator` too.
- An operator's `add_effects` must include every atom in its phase's `atoms`, and no atom may be in \
both `add_effects` and `delete_effects`.
- An operator's `preconditions` must be reachable: either true in the workspace to begin with, or \
made true by an earlier phase. Do not require something no phase establishes, and do not delete \
something a later phase's preconditions still need.
- Every object name must be one of the objects listed above, spelled exactly. Do not name an object \
any other way.
- {_INVENTION_NOTE}

If some clause CANNOT be expressed, put it in `unrepresented` with the reason, and leave it out of \
the phases. The usual reason is that it refers to something not in the object list and nothing in \
your plan produces it -- "pick another cup" when only one cup was detected, and no phase makes a \
second one. Never invent an object to satisfy a clause, and never bind it to a different object that \
happens to be present. Saying you could not do it is always better than quietly doing something \
else: a human is watching, and can put the missing object on the table and start again."""


def classifier_prompt(statement: str) -> str:
    """Ask whether one statement holds in one image of the workspace."""
    return f"""\
You are the perception system of a robot. Look at the image of the robot's workspace and decide \
whether the following statement is true right now.

Statement: {statement}

Judge only what you can see. If the workspace does not clearly show the statement to be true, it is \
false. Give a one-sentence reason for your answer."""
