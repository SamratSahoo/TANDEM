"""What `import tandem` offers a Python caller, and `tandem.plan_task`: a decomposition with no robot.

Two promises are pinned here. The surface is import-light: `import tandem` must cost nothing on a
laptop, and resolving any public name must not reach for a GPU, a camera or a robot client. And
`plan_task` is exactly what `tandem plan` runs -- the real proposal, parser, contract check and
repair loop -- so a script and the command cannot disagree about what a plan is. Only the model is a
stand-in, answering by the kind of question it is asked.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import subprocess
import sys
from unittest import mock

import numpy as np
import pytest
from test_packaging import FORBIDDEN
from toy_planner import TOY_CAPABILITIES, ToyPlanner

import tandem
from tandem.core.errors import TandemError
from tandem.planning import llm
from tandem.planning.config import PlanningConfig
from tandem.planning.plan import PhasePlan

OBJECTS = ["blue_toy", "white_box"]


def _atom(name, *args):
    return {"predicate": name, "args": list(args)}


# The three-phase task in TiPToP's goal language: the toy off the box, the box opened by a person,
# the toy into the box.
PLAN = {
    "new_predicates": [
        {"name": "IsOpen", "instructions": "the container {0} is open, so its interior is visible"}
    ],
    "phases": [
        {
            "executor": "robot",
            "description": "take the toy off the box and put it on the table",
            "atoms": [_atom("On", "blue_toy", "table")],
        },
        {
            "executor": "human",
            "description": "open the box",
            "instructions": "Open the white_box and fold its flaps back.",
            "atoms": [_atom("IsOpen", "white_box")],
            "operator": {
                "name": "Open",
                "args": ["white_box"],
                "preconditions": [_atom("HandEmpty")],
                "add_effects": [_atom("IsOpen", "white_box")],
                "delete_effects": [],
            },
        },
        {
            "executor": "robot",
            "description": "put the toy inside the box",
            "atoms": [_atom("On", "blue_toy", "white_box")],
        },
    ],
}

# A plan the parser refuses every time: a human phase with no operator.
NO_OPERATOR = {**PLAN, "phases": [{**p, "operator": None} for p in PLAN["phases"]]}

# The same shape of task in the toy's goal language, which is not cuTAMP's.
TOY_PLAN = {
    "new_predicates": [{"name": "LidClosed", "instructions": "the lid of {0} is shut"}],
    "phases": [
        {
            "executor": "robot",
            "description": "put the duck in the red bin",
            "atoms": [_atom("InBin", "duck", "red_bin")],
        },
        {
            "executor": "human",
            "description": "close the red bin's lid",
            "instructions": "Close the red bin's lid.",
            "atoms": [_atom("LidClosed", "red_bin")],
            "operator": {
                "name": "Close",
                "args": ["red_bin"],
                "preconditions": [_atom("InBin", "duck", "red_bin")],
                "add_effects": [_atom("LidClosed", "red_bin")],
                "delete_effects": [],
            },
        },
    ],
}


class Model:
    """The model: the object names when asked to name objects, a plan otherwise.

    Each list is played in order and its last answer repeats, so "it never gives a usable answer" is
    one entry rather than a guess at how many attempts the config allows.
    """

    OBJECTS_MARKER = "List the objects on the table"

    def __init__(self, plans=(PLAN,), objects=({"objects": OBJECTS},)):
        self.plans = list(plans)
        self.objects = list(objects)
        self.prompts: list[str] = []
        self.aio = self

    @property
    def models(self):
        return self

    @property
    def plan_prompts(self) -> list[str]:
        return [p for p in self.prompts if not p.startswith(self.OBJECTS_MARKER)]

    @property
    def object_prompts(self) -> list[str]:
        return [p for p in self.prompts if p.startswith(self.OBJECTS_MARKER)]

    async def generate_content(self, model, contents, config):
        prompt = contents[-1]
        self.prompts.append(prompt)
        answers = self.objects if prompt.startswith(self.OBJECTS_MARKER) else self.plans
        reply = answers.pop(0) if len(answers) > 1 else answers[0]
        return mock.Mock(text=json.dumps(reply))


@pytest.fixture
def model(monkeypatch):
    def use(**kwargs) -> Model:
        client = Model(**kwargs)
        monkeypatch.setattr(llm, "gemini_client", lambda: client)
        return client

    return use


@pytest.fixture
def photo(tmp_path):
    from PIL import Image

    path = tmp_path / "workspace.png"
    Image.new("RGB", (32, 32), (128, 128, 128)).save(path)
    return path


# --- the surface ------------------------------------------------------------------------------------


def test_import_tandem_imports_nothing_but_itself():
    """`import tandem` is free: no tandem submodule, and none of the libraries they pull in."""
    script = (
        "import sys, tandem\n"
        "print(sorted(m for m in sys.modules if m.startswith('tandem.')))\n"
        "print(sorted(m for m in ('numpy', 'PIL', 'pydantic', 'typer', 'google.genai') if m in sys.modules))\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["[]", "[]"]


def test_every_public_name_is_its_home_modules_own_object():
    for name, module in tandem._EXPORTS.items():
        assert getattr(tandem, name) is getattr(importlib.import_module(module), name), name


def test_the_public_names_are_exactly_the_listed_ones():
    assert set(tandem.__all__) == set(tandem._EXPORTS) | {"__version__"}
    assert set(dir(tandem)) == set(tandem.__all__)
    assert tandem.__version__ == importlib.import_module("tandem").__version__


def test_an_unknown_name_is_an_attribute_error_and_submodules_still_import():
    with pytest.raises(AttributeError, match="no attribute 'plan_everything'"):
        tandem.plan_everything  # noqa: B018
    # `from tandem import planners` asks the module for the attribute first, and must fall back to the
    # submodule on exactly that AttributeError.
    from tandem import executors, planners

    assert planners.Planner is tandem.Planner
    assert executors.register_human_executor is tandem.register_human_executor


def test_resolving_every_public_name_imports_nothing_heavy():
    """The whole surface is on the laptop path, not only the bare import."""
    script = f"""
import builtins
forbidden = {FORBIDDEN!r}
real_import = builtins.__import__

def guard(name, *args, **kwargs):
    if name.split(".")[0] in forbidden:
        raise AssertionError(f"resolving the public API imported a heavy dependency: {{name}}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guard
import tandem
for name in tandem.__all__:
    getattr(tandem, name)
print("ok")
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "ok"


def test_a_planner_registered_from_the_top_level_is_one_a_plan_can_name(model, photo):
    from helpers import isolate_registry

    with pytest.MonkeyPatch.context() as patch:
        isolate_registry(patch)
        tandem.register_backend("toy", ToyPlanner)
        model(plans=[TOY_PLAN])
        plan = tandem.plan_task("put the duck away", photo, planner="toy", objects=["duck", "red_bin"])
    assert plan.caps.name == "toy"
    assert plan.spec.scene_types.movable_type == "item"


class Scripted:
    """Replays a recorded policy for a human phase."""

    # A class registered as an executor carries what a listing shows as class attributes.
    segment_source = "policy"
    display_name = "Scripted policy"

    def __init__(self, ctx) -> None:
        self.ctx = ctx


def test_a_human_executor_registered_from_the_top_level_is_listed(monkeypatch):
    from tandem.executors import base

    monkeypatch.setattr(base, "_registered", dict(base._registered))
    tandem.register_human_executor("scripted", Scripted)
    assert "scripted" in base.available()
    described = base.info("scripted")
    assert (described.display_name, described.segment_source) == ("Scripted policy", "policy")
    assert described.summary == "Replays a recorded policy for a human phase."


# --- plan_task ------------------------------------------------------------------------------------------


def test_plan_task_returns_the_plan_a_session_would_walk(model, photo):
    client = model()
    plan = tandem.plan_task("do the thing", photo, objects=OBJECTS)

    assert isinstance(plan, PhasePlan)
    assert plan.caps.name == "tiptop", "the machine's default planner"
    assert plan.cfg.enabled is True
    assert [p.executor for p in plan.phases] == ["robot", "human", "robot"]
    assert plan.phases[1].operator.display == "Open(white_box)"
    assert plan.spec.unrepresented == ()
    # The objects were pinned, so the model was asked for the plan and nothing else.
    assert client.object_prompts == []
    assert len(client.plan_prompts) == 1 and "do the thing" in client.plan_prompts[0]
    record = plan.to_json()
    assert record["planner"] == "tiptop" and len(record["phases"]) == 3


def test_plan_task_names_the_objects_first_when_none_are_given(model, photo):
    client = model()
    plan = tandem.plan_task("do the thing", photo)
    assert len(client.object_prompts) == 1
    assert "do the thing" in client.object_prompts[0]
    assert plan.spec.scene_types.all_names == {"blue_toy", "white_box", "table"}


def test_plan_task_takes_an_image_in_memory(model):
    from PIL import Image

    model()
    for image in (np.zeros((24, 24, 3), np.uint8), Image.new("RGB", (24, 24))):
        plan = tandem.plan_task("do the thing", image, objects=OBJECTS)
        assert len(plan.phases) == 3


def test_plan_task_plans_for_a_planner_that_is_only_a_declaration(model, photo):
    """A ``Capabilities`` is all the phase planner reads, so an unregistered planner can be planned for."""
    client = model(plans=[TOY_PLAN])
    plan = tandem.plan_task(
        "put the duck in the red bin and close it",
        photo,
        planner=TOY_CAPABILITIES,
        objects=["duck", "red_bin"],
    )
    assert plan.caps is TOY_CAPABILITIES
    assert "InBin(?obj: item, ?bin: container)" in client.plan_prompts[0]
    assert "On(" not in client.plan_prompts[0], "the toy's prompt must not carry cuTAMP's goal language"
    assert plan.spec.scene_types.movable_type == "item"


def test_a_profile_supplies_the_planner_and_its_settings_with_planning_forced_on(model, photo, profile):
    from tandem.core import profiles

    assert profile.hitl.enabled is False, "the template has phase planning off"
    saved = profiles.load("test")
    saved.hitl.max_attempts = 1
    profiles.save(saved)

    client = model(plans=[NO_OPERATOR])
    with pytest.raises(TandemError):
        tandem.plan_task("do the thing", photo, profile="test", objects=OBJECTS)
    assert len(client.plan_prompts) == 1, "the profile's max_attempts, not the default 3"

    model()
    plan = tandem.plan_task("do the thing", photo, profile="test", objects=OBJECTS)
    assert plan.cfg.enabled is True and plan.cfg.max_attempts == 1
    assert plan.caps.name == profile.planner.backend


def test_a_config_given_directly_wins_and_is_turned_on(model, photo):
    client = model(plans=[NO_OPERATOR])
    with pytest.raises(TandemError):
        tandem.plan_task(
            "do the thing", photo, objects=OBJECTS, config=PlanningConfig(enabled=False, max_attempts=2)
        )
    assert len(client.plan_prompts) == 2


def test_a_model_that_never_gives_a_usable_plan_is_a_tandem_error_with_the_last_refusal(model, photo):
    client = model(plans=[NO_OPERATOR])
    with pytest.raises(TandemError) as raised:
        tandem.plan_task("do the thing", photo, objects=OBJECTS)
    assert raised.value.message == "The model could not produce a usable plan for that instruction."
    assert "needs an `operator` object" in raised.value.hint
    assert len(client.plan_prompts) == 3, "the paper's at most three attempts"


def test_a_model_that_never_names_the_objects_is_a_tandem_error_not_a_traceback(model, photo):
    client = model(objects=[{"objects": []}])
    with pytest.raises(TandemError) as raised:
        tandem.plan_task("do the thing", photo)
    assert raised.value.message == "The model could not name the objects in the photo."
    assert "Pin them instead" in raised.value.hint
    assert len(client.object_prompts) == 3 and client.plan_prompts == []


def test_tandem_plan_reports_objects_it_could_not_name_as_a_tandem_error(model, photo):
    """The same step `tandem plan` takes: a ProposalError from naming the objects used to escape raw."""
    from typer.testing import CliRunner

    from tandem.cli.app import app

    model(objects=[{"objects": []}])
    result = CliRunner().invoke(app, ["plan", "do the thing", "--image", str(photo)])
    assert result.exit_code != 0
    assert isinstance(result.exception, TandemError), repr(result.exception)
    assert result.exception.message == "The model could not name the objects in the photo."


def test_plan_task_and_tandem_plan_give_the_same_plan(model, photo):
    from typer.testing import CliRunner

    from tandem.cli.app import app

    model()
    result = CliRunner().invoke(
        app, ["plan", "do the thing", "--image", str(photo), "--json", "-o", OBJECTS[0], "-o", OBJECTS[1]]
    )
    assert result.exit_code == 0, result.output
    model()
    assert json.loads(result.stdout) == tandem.plan_task("do the thing", photo, objects=OBJECTS).to_json()


def test_plan_task_writes_what_the_model_saw_and_said_when_asked(model, photo, tmp_path):
    model()
    tandem.plan_task("do the thing", photo, objects=OBJECTS, save_vlm_io=tmp_path / "vlm")
    lines = (tmp_path / "vlm" / "index.jsonl").read_text().splitlines()
    assert [json.loads(line)["label"] for line in lines] == ["task plan"]


def test_plan_task_cannot_block_a_running_event_loop_and_the_async_form_runs_there(model, photo):
    model()

    async def in_a_notebook():
        with pytest.raises(TandemError, match="running event loop") as raised:
            tandem.plan_task("do the thing", photo, objects=OBJECTS)
        assert "plan_task_async" in raised.value.hint
        return await tandem.plan_task_async("do the thing", photo, objects=OBJECTS)

    plan = asyncio.run(in_a_notebook())
    assert len(plan.phases) == 3


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"planner": "tiptoe"}, "tiptop"),
        ({"planner": 3}, "planner must be a planner's name or a Capabilities"),
        ({"objects": "blue_toy"}, "objects must be a list of labels"),
    ],
    ids=["unknown planner", "not a planner", "a string for objects"],
)
def test_plan_task_refuses_what_it_cannot_use_before_asking_the_model(model, photo, kwargs, message):
    client = model()
    with pytest.raises(TandemError) as raised:
        tandem.plan_task("do the thing", photo, **{"objects": OBJECTS, **kwargs})
    assert message in f"{raised.value.message} {raised.value.hint}"
    assert client.prompts == []


def test_plan_task_says_so_when_the_photo_is_missing_or_not_an_image(model, tmp_path):
    client = model()
    with pytest.raises(TandemError, match="There is no image at"):
        tandem.plan_task("do the thing", tmp_path / "nowhere.png", objects=OBJECTS)
    text = tmp_path / "notes.png"
    text.write_text("not a picture")
    with pytest.raises(TandemError, match="could not be read as an image"):
        tandem.plan_task("do the thing", text, objects=OBJECTS)
    assert client.prompts == []
