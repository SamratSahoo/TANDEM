"""The proposal parser's half of the magic operator, and the plan checks inside the repair loop.

Three things are held here. The parser builds each human phase's operator from the model's reply and
refuses one that does not hold together, in words the model can act on (the rules are LJ's, ported
from ``tiptop/hitl/proposal.py`` at cf75a68). A plan whose phases are each fine but which fails as a
whole -- a robot phase no robot operator can achieve, a phase that needs what an earlier one undid --
goes back to the model with the reason instead of failing the trial. And an API failure that says
nothing about the answer (a 429, a 503, a timeout) is retried underneath the repair loop, so it
never spends one of the model's attempts.

No network: the model is a fake client that plays back canned replies, or raises, in order. Going
through the real ``query_json`` and ``propose_plan`` rather than calling the parser directly is what
lets these tests pin down what the model is TOLD when a plan is refused.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
from unittest import mock

import pytest

from tandem.core.errors import TandemError
from tandem.planners import registry
from tandem.planning import llm
from tandem.planning.config import PlanningConfig
from tandem.planning.proposal import check_plan, parse_plan_response, propose_plan
from tandem.planning.symbols import ProposalError

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


OPEN = _op("Open", ["white_box"], pre=[("HandEmpty", [])], add=[("IsOpen", ["white_box"])])

# The three-phase task: the toy off the box, the box opened by a person, the toy into the box.
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
            "operator": OPEN,
        },
        {
            "executor": "robot",
            "description": "put the toy inside the box",
            "atoms": _atoms(("On", ["blue_toy", "white_box"])),
        },
    ],
}


def _plan(**changes):
    """PLAN_RESPONSE with some part replaced."""
    return {**PLAN_RESPONSE, **changes}


def _open_close(**changes):
    """LJ's open/close/insert plan: the middle phase CLOSES the box the last one needs open."""
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


def _reordered():
    """The same three phases in an order that works: the toy goes in before the box is closed."""
    phases = _open_close()["phases"]
    return _open_close(phases=[phases[0], phases[2], phases[1]])


def parse(response=None, objects=OBJECTS, caps=CAPS):
    return parse_plan_response(response or PLAN_RESPONSE, "do the thing", objects, TABLE, caps)


def _one_human_phase(operator, atoms=(("IsOpen", ["white_box"]),), **extra):
    """A plan of one human phase, "open the box", carrying ``operator``."""
    phase = {
        "executor": "human",
        "description": "open the box",
        "instructions": "open it",
        "atoms": _atoms(*atoms),
        "operator": operator,
    }
    return _plan(phases=[phase], **extra)


# --- the operator the parser builds ---------------------------------------------------------------


def test_a_human_phase_carries_its_operator_grounded_to_this_scene():
    operator = parse().phases[1].operator
    assert operator.display == "Open(white_box)"
    # Typed from the args' scene types, like an invented predicate's parameters are: white_box has
    # something put on it in phase 2, so it is a surface for the whole task.
    assert operator.signature == "Open(x0: surface)"
    assert {str(a) for a in operator.preconditions} == {"HandEmpty()"}
    assert {str(a) for a in operator.add_effects} == {"IsOpen(white_box)"}
    assert operator.delete_effects == frozenset()


def test_an_operators_parameters_follow_each_argument_in_order():
    insert = parse(_reordered()).phases[1].operator
    assert insert.signature == "Insert(x0: movable, x1: surface)"
    assert insert.display == "Insert(blue_toy, white_box)"


def test_a_robot_phase_never_carries_an_operator():
    robot = parse().phases[0]
    assert robot.operator is None
    assert robot.preconditions == frozenset()
    assert robot.add_effects == robot.atoms
    assert robot.delete_effects == frozenset()


def test_the_record_carries_each_human_phases_operator():
    record = parse().to_json()
    assert [o["instance"] for o in record["human_operators"]] == ["Open(white_box)"]
    assert record["human_operators"][0]["phase"] == 1
    assert record["human_operators"][0]["preconditions"] == ["HandEmpty()"]


@pytest.mark.parametrize(
    "operator,expected",
    [
        # LJ's six (tests/test_hitl.py test_operator_rejections), in LJ's words.
        (None, "needs an `operator`"),
        (_op("Open", ["white_box"], add=[("On", ["blue_toy", "table"])]), "does not make that true"),
        (
            _op("Open", ["white_box"], add=[("IsOpen", ["white_box"])], dele=[("IsOpen", ["white_box"])]),
            "both adds and deletes IsOpen(white_box)",
        ),
        (_op("Open", ["white_box"]), "has no `add_effects`"),
        (_op("Open", ["green_crate"], add=[("IsOpen", ["white_box"])]), "not an object in this scene"),
        (
            _op("Open", ["white_box"], pre=[("IsShiny", ["white_box"])], add=[("IsOpen", ["white_box"])]),
            "Unknown predicate 'IsShiny'",
        ),
        # And the rest of the rules.
        ("open the box", "needs an `operator`"),
        ({"args": ["white_box"], "add_effects": _atoms(("IsOpen", ["white_box"]))}, "required key 'name'"),
        (_op("  ", ["white_box"], add=[("IsOpen", ["white_box"])]), "needs a `name`"),
        (_op("Open box!", ["white_box"], add=[("IsOpen", ["white_box"])]), "not a valid operator name"),
        (_op("Open#1", ["white_box"], add=[("IsOpen", ["white_box"])]), "not a valid operator name"),
        (
            {**_op("Open", [], add=[("IsOpen", ["white_box"])]), "args": "white_box"},
            "must be a list of strings",
        ),
        (
            {**_op("Open", ["white_box"]), "add_effects": {"predicate": "IsOpen"}},
            "'add_effects' must be a list",
        ),
        (
            _op("Open", ["white_box"], add=[("IsOpen", ["white_box"]), ("On", ["blue_toy", "moon"])]),
            "'moon' is not an object in this scene",
        ),
        (
            _op("Open", ["white_box"], add=[("IsOpen", ["white_box"]), ("On", ["blue_toy"])]),
            "On takes 2 argument(s)",
        ),
    ],
)
def test_operator_rejections(operator, expected):
    with pytest.raises(ProposalError, match=re.escape(expected)):
        parse(_one_human_phase(operator))


def test_an_operator_may_not_delete_what_its_own_phase_promises():
    # Checked BEFORE "atoms must be add effects", so the model hears the specific contradiction
    # rather than the generic rule it would otherwise be told first.
    operator = _op("Open", ["white_box"], add=[("IsClosed", ["white_box"])], dele=[("IsOpen", ["white_box"])])
    response = _one_human_phase(
        operator,
        new_predicates=[
            {"name": "IsOpen", "instructions": "the container {0} is open"},
            {"name": "IsClosed", "instructions": "the container {0} is shut"},
        ],
    )
    with pytest.raises(ProposalError) as raised:
        parse(response)
    assert str(raised.value) == (
        "The operator Open deletes IsOpen(white_box), which the phase 'open the box' says must be TRUE "
        "afterwards. Take it out of `delete_effects`, or out of the phase's `atoms`."
    )


def test_every_operator_rejection_names_what_to_change():
    # These messages go back to the model verbatim, so a bare "invalid" would waste an attempt.
    with pytest.raises(ProposalError) as raised:
        parse(_one_human_phase(_op("Open", ["white_box"], add=[("On", ["blue_toy", "table"])])))
    assert str(raised.value) == (
        "The phase 'open the box' says IsOpen(white_box) must be true afterwards, but its operator Open "
        "does not make that true. Every atom in a phase's `atoms` must appear in its operator's "
        "`add_effects`."
    )


def test_an_operator_on_a_robot_phase_is_refused():
    # Almost always a human step written as a robot phase, which is the misjudgement worth catching.
    response = _plan(
        phases=[
            {
                "executor": "robot",
                "description": "shove it",
                "atoms": _atoms(("On", ["blue_toy", "table"])),
                "operator": _op("Shove", ["blue_toy"], add=[("On", ["blue_toy", "table"])]),
            }
        ]
    )
    with pytest.raises(ProposalError, match="Only a human phase") as raised:
        parse(response, caps=CAPS)
    # Told what the robot CAN do, in the words of this backend's declaration.
    assert f"it can only {CAPS.robot_description}" in str(raised.value)
    assert "The ROBOT phase 'shove it' declares an `operator`" in str(raised.value)


@pytest.mark.parametrize("empty", [None, {}])
def test_an_empty_operator_on_a_robot_phase_is_not_a_declaration(empty):
    # Structured output may fill the optional key with null or {}. That says nothing, so it is not
    # held against the phase (LJ refuses only a truthy one).
    response = {
        "phases": [
            {
                "executor": "robot",
                "description": "put the toy on the table",
                "atoms": _atoms(("On", ["blue_toy", "table"])),
                "operator": empty,
            }
        ]
    }
    assert parse(response).phases[0].operator is None


def test_an_invented_predicate_used_only_inside_an_operator_is_still_typed():
    # IsUnlocked is only a precondition and IsClosed only a delete effect; neither is in any phase's
    # atoms. Without counting uses inside operators both would be refused as "invented but unused",
    # and "the person opens what was shut" could not be said at all.
    operator = _op(
        "Open",
        ["white_box"],
        pre=[("IsUnlocked", ["white_box"])],
        add=[("IsOpen", ["white_box"])],
        dele=[("IsClosed", ["white_box"])],
    )
    response = _one_human_phase(
        operator,
        new_predicates=[
            {"name": "IsOpen", "instructions": "the container {0} is open"},
            {"name": "IsClosed", "instructions": "the container {0} is shut"},
            {"name": "IsUnlocked", "instructions": "the box {0} is unlocked"},
        ],
    )
    spec = parse(response)
    assert {p.name for p in spec.invented} == {"IsOpen", "IsClosed", "IsUnlocked"}
    # Typed from the use, like any other: nothing is put on white_box here, so it is a movable.
    types = {p.name: list(p.predicate.types) for p in spec.invented}
    assert types["IsUnlocked"] == types["IsClosed"] == ["movable"]
    assert {str(a) for a in spec.phases[0].preconditions} == {"IsUnlocked(white_box)"}
    assert {str(a) for a in spec.phases[0].delete_effects} == {"IsClosed(white_box)"}


def test_an_invented_predicate_used_differently_inside_an_operator_is_refused():
    # Uses inside an operator are uses: they have to agree with the phase atoms on the signature. The
    # table is a surface and white_box (nothing is put on it here) a movable.
    operator = _op("Open", ["white_box"], pre=[("IsOpen", ["table"])], add=[("IsOpen", ["white_box"])])
    with pytest.raises(ProposalError, match="inconsistent arguments"):
        parse(_one_human_phase(operator))


def test_a_predicate_nobody_uses_anywhere_is_still_refused():
    response = _plan(
        new_predicates=[
            *PLAN_RESPONSE["new_predicates"],
            {"name": "IsUnlocked", "instructions": "the box {0} is unlocked"},
        ]
    )
    with pytest.raises(ProposalError, match="no phase uses it"):
        parse(response)


def test_the_scene_types_come_from_the_phases_atoms_never_from_an_operator():
    # A precondition must not quietly make white_box a surface -- and with it, a static obstacle the
    # planner may not move -- when no phase ever puts anything on it. It is typed against the split
    # instead, and refused with the reason.
    operator = _op(
        "Open", ["white_box"], pre=[("On", ["blue_toy", "white_box"])], add=[("IsOpen", ["white_box"])]
    )
    response = _one_human_phase(operator)
    with pytest.raises(ProposalError, match="'white_box', which is a movable in this scene"):
        parse(response)


# --- the plan as a whole --------------------------------------------------------------------------


def test_check_plan_passes_a_plan_that_hangs_together():
    assert check_plan(parse(), CFG, CAPS) is None
    assert check_plan(parse(_reordered()), CFG, CAPS) is None


def test_check_plan_refuses_a_plan_that_undoes_what_a_later_phase_needs():
    with pytest.raises(ProposalError) as raised:
        check_plan(parse(_open_close()), CFG, CAPS)
    assert str(raised.value) == (
        "This plan does not hang together: phase 2 ('drop the toy in') (Insert(blue_toy, white_box)) "
        "requires IsOpen(white_box), but an earlier phase deletes it and no phase puts it back. Either "
        "reorder the phases so it still holds, or drop it from that operator's preconditions."
    )


def test_the_contract_check_can_be_turned_off():
    check_plan(parse(_open_close()), dataclasses.replace(CFG, check_plan_effects=False), CAPS)


# A backend whose robot cannot reach one of the predicates it lets a goal name. TipTop's can reach
# all three, so this is the only way to get a robot phase past the parser and into the check.
NO_HOLDING = dataclasses.replace(CAPS, achievable_predicates=CAPS.achievable_predicates - {"Holding"})

HOLD_THE_TOY = {
    "phases": [
        {
            "executor": "robot",
            "description": "pick up the toy",
            "atoms": _atoms(("Holding", ["blue_toy"])),
        }
    ]
}


def test_check_plan_refuses_a_robot_phase_no_robot_operator_can_achieve():
    # Even with the contract check off: this one is not optional, the planner would search forever.
    contract_check_off = dataclasses.replace(CFG, check_plan_effects=False)
    with pytest.raises(ProposalError) as raised:
        check_plan(parse(HOLD_THE_TOY, caps=NO_HOLDING), contract_check_off, NO_HOLDING)
    # Only On is offered: HandEmpty is achievable too, but it has no wire name, and a phase stated
    # with it alone is refused as a leg with no goal (tests/test_review_method.py).
    assert str(raised.value) == (
        "This plan cannot be carried out: phase 0 ('pick up the toy') asks the robot for "
        "Holding(blue_toy), which no robot operator can achieve. The robot can only pick an object up "
        "and place it on a surface. Either state that phase with On, or make it a human phase "
        "with an operator."
    )


# --- inside the repair loop -----------------------------------------------------------------------


class _FakeGemini:
    """A client that plays back canned replies in order and records every prompt it was sent.

    A reply that is an exception is raised instead of returned, which is how an API failure looks.
    """

    def __init__(self, replies):
        self._replies = list(replies)
        self.prompts: list[str] = []
        self.aio = self

    @property
    def models(self):
        return self

    async def generate_content(self, model, contents, config):
        self.prompts.append(contents[-1])
        reply = self._replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return mock.Mock(text=reply if isinstance(reply, str) else json.dumps(reply))


@pytest.fixture
def no_waiting(monkeypatch):
    """Take the time out of the transient-error backoff, and record every wait it asked for."""
    waits: list[float] = []

    async def sleep(delay):
        waits.append(delay)

    monkeypatch.setattr(llm, "_sleep", sleep)
    return waits


def propose(replies, *, cfg=CFG, caps=CAPS, objects=OBJECTS):
    """Run the real ``propose_plan`` against a fake model. Returns ``(spec, client)``."""
    client = _FakeGemini(replies)
    with mock.patch.object(llm, "gemini_client", lambda: client):
        spec = asyncio.run(propose_plan(None, "do the thing", objects, TABLE, cfg, caps))
    return spec, client


def test_a_plan_that_does_not_hang_together_is_reprompted_with_the_reason():
    # The whole point of running the contract check inside `parse`: the model hears exactly which
    # phase needs what an earlier one deleted, and fixes the order instead of the trial failing.
    spec, client = propose([_open_close(), _reordered()])
    assert len(client.prompts) == 2, "the second attempt should have been made"
    assert (
        "This plan does not hang together: phase 2 ('drop the toy in') (Insert(blue_toy, white_box)) "
        "requires IsOpen(white_box), but an earlier phase deletes it"
    ) in client.prompts[1]
    assert [p.description for p in spec.phases] == ["open the box", "drop the toy in", "close the box"]


def test_a_plan_that_never_hangs_together_fails_with_the_contract_reason():
    with pytest.raises(ProposalError, match="This plan does not hang together"):
        propose([_open_close()] * 3)


def test_with_the_contract_check_off_the_same_plan_is_accepted_first_time():
    spec, client = propose([_open_close()], cfg=dataclasses.replace(CFG, check_plan_effects=False))
    assert len(client.prompts) == 1
    assert [p.description for p in spec.phases] == ["open the box", "close the box", "drop the toy in"]


def test_a_robot_phase_no_robot_operator_can_achieve_is_repaired_too():
    # It used to be checked after the proposal was accepted, where a failure could only end the
    # attempt -- and the model that wrote the phase never heard why.
    fixed = _plan()
    spec, client = propose([HOLD_THE_TOY, fixed], caps=NO_HOLDING)
    assert len(client.prompts) == 2
    assert "This plan cannot be carried out: phase 0 ('pick up the toy')" in client.prompts[1]
    assert "make it a human phase with an operator" in client.prompts[1]
    assert [p.executor for p in spec.phases] == ["robot", "human", "robot"]


def test_a_human_phase_without_an_operator_is_repaired():
    missing = _plan(phases=[{**p, "operator": None} for p in PLAN_RESPONSE["phases"]])
    spec, client = propose([missing, PLAN_RESPONSE])
    assert "The human phase 'open the box' needs an `operator` object" in client.prompts[1]
    assert spec.phases[1].operator.display == "Open(white_box)"


def test_a_bad_capabilities_declaration_is_not_the_models_to_repair():
    # A declaration that can never match an atom is the backend author's mistake. Reprompting the
    # model would only burn its attempts on something it cannot fix, so it is raised at once.
    misdeclared = dataclasses.replace(CAPS, exclusive_arguments={"on": 0})
    client = _FakeGemini([PLAN_RESPONSE, PLAN_RESPONSE, PLAN_RESPONSE])
    with mock.patch.object(llm, "gemini_client", lambda: client):
        with pytest.raises(TandemError, match="not one of its goal predicates"):
            asyncio.run(propose_plan(None, "do the thing", OBJECTS, TABLE, CFG, misdeclared))
    assert len(client.prompts) == 1


def test_an_accepted_plan_logs_each_operator_and_any_repeated_robot_move(caplog):
    repeated = _plan(
        phases=[
            {
                "executor": "robot",
                "description": "put the toy on the table",
                "atoms": _atoms(("On", ["blue_toy", "table"])),
            },
            {
                "executor": "robot",
                "description": "put the toy on the box",
                "atoms": _atoms(("On", ["blue_toy", "white_box"])),
            },
            PLAN_RESPONSE["phases"][1],
        ]
    )
    with caplog.at_level(logging.INFO, logger="tandem.planning.proposal"):
        propose([repeated])
    messages = [r.getMessage() for r in caplog.records]
    assert (
        "phase 2 operator Open(white_box): pre ['HandEmpty()'] -> add ['IsOpen(white_box)'] del none"
        in messages
    )
    # A warning, not a rejection: the plan was accepted first time.
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        w.startswith("this plan repeats work -- phases 0") and "both move blue_toy" in w for w in warnings
    )


# --- transient API failures -----------------------------------------------------------------------


def _server_error(code=503):
    from google.genai import errors

    return errors.ServerError(
        code, {"error": {"code": code, "message": "overloaded", "status": "UNAVAILABLE"}}
    )


def _client_error(code):
    from google.genai import errors

    return errors.ClientError(code, {"error": {"code": code, "message": "no", "status": "BAD"}})


def test_a_transient_api_error_does_not_use_up_a_repair_attempt(no_waiting):
    # max_attempts=1 leaves no room for a second answer, so if either failure counted as an attempt
    # this would raise instead of returning the plan.
    spec, client = propose(
        [_server_error(503), _client_error(429), PLAN_RESPONSE],
        cfg=dataclasses.replace(CFG, max_attempts=1),
    )
    assert [p.executor for p in spec.phases] == ["robot", "human", "robot"]
    assert len(client.prompts) == 3
    assert no_waiting == [llm.TRANSIENT_DELAYS[0], llm.TRANSIENT_DELAYS[1]]


def test_the_repair_attempts_survive_transient_errors_around_them(no_waiting):
    # 503, a bad answer, 429, the good answer: two model answers, so max_attempts=2 is enough, and
    # the repair prompt still carries the reason the first answer was refused.
    spec, client = propose(
        [_server_error(503), _open_close(), _client_error(429), _reordered()],
        cfg=dataclasses.replace(CFG, max_attempts=2),
    )
    assert len(client.prompts) == 4
    assert "This plan does not hang together" in client.prompts[3]
    # The retry after a transient failure is the SAME request, not a repair prompt.
    assert client.prompts[2] == client.prompts[3]
    assert "Your previous response was rejected" not in client.prompts[0]
    assert client.prompts[0] == client.prompts[1]
    assert [p.description for p in spec.phases] == ["open the box", "drop the toy in", "close the box"]


def test_a_request_the_api_refuses_outright_is_raised_at_once(no_waiting):
    # A bad key or a malformed request fails the same way however long one waits.
    from google.genai import errors

    with pytest.raises(errors.ClientError):
        propose([_client_error(400), PLAN_RESPONSE])
    assert no_waiting == []


def test_the_backoff_is_bounded_and_says_so_when_it_gives_up(no_waiting):
    always = [_server_error(503)] * (len(llm.TRANSIENT_DELAYS) + 1)
    with pytest.raises(TandemError, match="kept failing: ServerError") as raised:
        propose(always)
    assert no_waiting == list(llm.TRANSIENT_DELAYS)
    assert sum(no_waiting) <= 90, "an operator is standing at the robot while this waits"
    assert "rate-limited" in raised.value.hint
    assert isinstance(raised.value.__cause__, Exception)


def test_backoff_covers_the_classifier_too(no_waiting):
    # query_json is shared, so a camera check hit by a 503 is retried the same way rather than
    # recorded as a failed verification.
    from tandem.planning import grounding
    from tandem.planning.symbols import Atom

    client = _FakeGemini([_server_error(502), json.dumps({"holds": True, "reason": "open"})])
    with mock.patch.object(llm, "gemini_client", lambda: client):
        verdict = asyncio.run(
            grounding.classify(None, Atom("IsOpen", ("white_box",)), {"IsOpen": "{0} is open"}, CFG)
        )
    assert verdict.holds
    assert no_waiting == [llm.TRANSIENT_DELAYS[0]]


def _transient_cases():
    import httpx

    request = httpx.Request("POST", "https://example.invalid")
    return [
        (_server_error(500), True),
        (_server_error(502), True),
        (_server_error(503), True),
        (_server_error(504), True),
        (_client_error(429), True),
        (_client_error(408), True),
        (asyncio.TimeoutError(), True),
        (TimeoutError(), True),
        (ConnectionResetError(), True),
        (httpx.ReadTimeout("slow", request=request), True),
        (httpx.ConnectError("refused", request=request), True),
        (httpx.RemoteProtocolError("dropped", request=request), True),
        (_client_error(400), False),
        (_client_error(401), False),
        (_client_error(403), False),
        (_client_error(404), False),
        (ValueError("no"), False),
        (ProposalError("the model's fault, not the API's"), False),
        (httpx.UnsupportedProtocol("ftp", request=request), False),
    ]


@pytest.mark.parametrize(
    "exc,transient",
    _transient_cases(),
    ids=lambda v: type(v).__name__ if isinstance(v, BaseException) else str(v),
)
def test_what_counts_as_transient(exc, transient):
    assert llm.is_transient(exc) is transient


# --- `tandem plan` --------------------------------------------------------------------------------


def _photo(tmp_path):
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow is a base dependency
        pytest.skip("Pillow is not installed")
    photo = tmp_path / "workspace.png"
    Image.new("RGB", (32, 32), (128, 128, 128)).save(photo)
    return photo


def _run_plan(tmp_path, monkeypatch, replies, cfg=None):
    from typer.testing import CliRunner

    from tandem.cli import plan as plan_cmd
    from tandem.cli.app import app

    monkeypatch.setattr(llm, "gemini_client", lambda: _FakeGemini(replies))
    if cfg is not None:
        monkeypatch.setattr(plan_cmd, "_config_for", lambda name: cfg)
    result = CliRunner().invoke(
        app, ["plan", "do the thing", "--image", str(_photo(tmp_path)), "-o", "blue_toy", "-o", "white_box"]
    )
    assert result.exit_code == 0, result.output
    return result.output


def _flat(output):
    """The output with its line wrapping undone, for a sentence the console may have broken."""
    return " ".join(output.split())


def test_tandem_plan_prints_each_human_phases_operator(tmp_path, monkeypatch):
    output = _run_plan(tmp_path, monkeypatch, [PLAN_RESPONSE])
    lines = [line.strip() for line in output.splitlines()]
    assert "operator Open(x0: surface)  as Open(white_box)" in lines
    assert "preconditions  HandEmpty()" in lines
    assert "add effects    IsOpen(white_box)" in lines
    # An empty list is printed as a statement, not left out.
    assert "delete effects none" in lines
    assert output.count("operator ") == 1, "a robot phase has no operator to print"


def test_tandem_plan_reports_the_contract_check(tmp_path, monkeypatch):
    output = _flat(_run_plan(tmp_path, monkeypatch, [PLAN_RESPONSE]))
    assert "the phases' contracts hang together checked in the repair loop" in output
    assert "repeats work" not in output


def test_tandem_plan_says_what_the_check_would_have_refused_when_it_is_off(tmp_path, monkeypatch):
    off = PlanningConfig(enabled=True, check_plan_effects=False)
    output = _flat(_run_plan(tmp_path, monkeypatch, [_open_close()], cfg=off))
    assert "the phases' contracts do not hang together (check_plan_effects is off)" in output
    assert "phase 2 ('drop the toy in') (Insert(blue_toy, white_box)) requires IsOpen(white_box)" in output


def _needs_an_unlocked_box():
    """Opening the box needs it unlocked, and no phase unlocks it: fine until the start is measured."""
    return _plan(
        new_predicates=[
            {"name": "IsOpen", "instructions": "the container {0} is open, so its interior is visible"},
            {"name": "IsUnlocked", "instructions": "the latch on {0} is undone"},
        ],
        phases=[
            {
                "executor": "human",
                "description": "open the box",
                "instructions": "Open the white_box.",
                "atoms": _atoms(("IsOpen", ["white_box"])),
                "operator": _op(
                    "Open",
                    ["white_box"],
                    pre=[("IsUnlocked", ["white_box"])],
                    add=[("IsOpen", ["white_box"])],
                ),
            },
        ],
    )


@pytest.mark.parametrize("unlocked_at_start", [False, True])
def test_tandem_plan_reports_the_recheck_against_the_measured_scene(tmp_path, monkeypatch, unlocked_at_start):
    # The repair loop accepted this plan: with the start unmeasured, the box may already be unlocked.
    # Once `classify_initial` finds it is not, the plan provably cannot run -- and `tandem plan` is
    # the one place that can still be acted on, so it must say so rather than only log it.
    from tandem.planning import grounding
    from tandem.planning.symbols import Atom

    async def classify(image, spec, cfg, caps):
        return frozenset({Atom("IsUnlocked", ("white_box",))}) if unlocked_at_start else frozenset()

    monkeypatch.setattr(grounding, "classify_initial_state", classify)
    cfg = PlanningConfig(enabled=True, classify_initial=True)
    output = _flat(_run_plan(tmp_path, monkeypatch, [_needs_an_unlocked_box()], cfg=cfg))
    assert "the phases' contracts hang together checked in the repair loop" in output
    if unlocked_at_start:
        assert "the plan holds against the scene in the photo" in output
    else:
        assert "the plan does not hang together against the scene in the photo" in output
        assert "IsUnlocked(white_box)" in output
