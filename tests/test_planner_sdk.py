"""The planner SDK's base class: what a new planner has to write, and what it gets for nothing.

A new planner is supposed to be a declaration, three methods and one line of registration. These
tests hold the SDK to that: the class is its own factory and registers as itself; every verb the
planner did not write has a default that is either harmless or loudly unsupported; and a declaration
that would load and then quietly do the wrong thing is refused when the class is defined.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import pytest
from helpers import isolate_registry
from toy_planner import IN_BIN, TOY_CAPABILITIES, ToyPlanner

import tandem.planners as planners
from tandem.core.errors import RuntimeNotReady, TandemError
from tandem.planners import (
    BackendContext,
    Capabilities,
    Planner,
    PlannerInfo,
    RuntimeRecipe,
    SidecarPlanner,
    Source,
    SourcePin,
    register_backend,
    registry,
)
from tandem.planners.base import BackendFactory, TampBackend
from tandem.planners.runtime import RecipeRuntime
from tandem.planners.sdk import UnsupportedVerb, capability_problems, factory_problems
from tandem.planners.tiptop import FACTORY as TIPTOP
from tandem.planners.tiptop.backend import TiptopBackend
from tandem.planners.tiptop.capabilities import CAPABILITIES as TIPTOP_CAPABILITIES
from tandem.planners.tiptop.factory import INFO as TIPTOP_INFO


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    isolate_registry(monkeypatch)


def _ctx(tmp_path, **options) -> BackendContext:
    logs: list = []
    return BackendContext(
        profile=None,
        session_dir=tmp_path,
        output_dir=tmp_path / "legs",
        options=options,
        on_log=lambda stream, text: logs.append((stream, text)),
    )


class _Minimal(Planner):
    """A planner that wrote only what the SDK says it must."""

    info = PlannerInfo(name="minimal")
    CAPABILITIES = replace(TOY_CAPABILITIES, name="minimal")

    def perceive(self, *, task_hint, save_dir, reset_arm=True, open_gripper=False):
        raise NotImplementedError

    def plan(
        self,
        scene_id,
        goal,
        *,
        surfaces=frozenset(),
        movables=None,
        return_home=True,
        save_dir,
        reuse_skeleton=None,
    ):
        raise NotImplementedError

    def execute(self, plan_handle, leg, *, save_dir, should_stop=None):
        raise NotImplementedError


# --- one place to import from -----------------------------------------------------------------------------


def test_everything_an_author_needs_is_importable_from_tandem_planners():
    assert sorted(planners.__all__) == sorted(planners._EXPORTS)
    for name in planners.__all__:
        assert getattr(planners, name) is not None
    assert planners.Planner is Planner and planners.register_backend is registry.register_backend
    with pytest.raises(AttributeError):
        planners.NoSuchThing  # noqa: B018


def test_importing_the_protocol_does_not_import_the_sdk():
    # tandem.planners.base is on the phase planner's path. Its package must not drag the SDK, the
    # registry and the runtime builder in with it -- they are resolved when first asked for.
    script = (
        "import sys\n"
        "import tandem.planners.base\n"
        "loaded = [m for m in ('tandem.planners.sdk', 'tandem.planners.registry', 'tandem.planners.runtime') if m in sys.modules]\n"
        "assert not loaded, loaded\n"
        "from tandem.planners import Planner\n"
        "assert 'tandem.planners.sdk' in sys.modules\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0 and "ok" in result.stdout, result.stdout + result.stderr


# --- the class is its own factory -----------------------------------------------------------------------


def test_a_planner_class_is_its_own_factory():
    assert isinstance(ToyPlanner, BackendFactory)
    assert ToyPlanner.capabilities() is TOY_CAPABILITIES
    assert ToyPlanner.runtime() is None, "a planner with no recipe is pure Python and installs with pip"
    assert ToyPlanner.name == "toy", "the backend's name follows its info"
    assert factory_problems(ToyPlanner) == []


def test_one_line_registers_a_planner_class_and_the_registry_never_instantiates_it(tmp_path):
    register_backend("toy", ToyPlanner)
    assert registry.factory("toy") is ToyPlanner, "the class itself is the factory"
    assert registry.info("toy").display_name == "Toy"
    (entry,) = [e for e in registry.catalog() if e.name == "toy"]
    assert entry.ok and entry.origin == "registered"

    first = registry.create("toy", _ctx(tmp_path))
    second = registry.create("toy", _ctx(tmp_path))
    assert isinstance(first, ToyPlanner) and isinstance(first, TampBackend)
    assert first is not second, "every session gets its own backend"
    assert first.ctx.output_dir == tmp_path / "legs"


def test_an_installed_package_may_point_its_entry_point_at_a_planner_class(tmp_path):
    register_backend("toy", "toy_planner:ToyPlanner")
    assert registry.factory("toy") is ToyPlanner
    assert isinstance(registry.create("toy", _ctx(tmp_path)), ToyPlanner)


def test_a_base_for_planners_is_not_a_planner_to_register(tmp_path):
    register_backend("sidecar", SidecarPlanner)
    with pytest.raises(TandemError, match="a base for planners, not a planner: it is declared abstract"):
        registry.factory("sidecar")

    class Unfinished(Planner):
        info = PlannerInfo(name="unfinished")
        CAPABILITIES = replace(TOY_CAPABILITIES, name="unfinished")

        def perceive(self, **kwargs): ...

    register_backend("unfinished", Unfinished)
    with pytest.raises(TandemError, match="leaves execute, plan unimplemented"):
        registry.factory("unfinished")


def test_options_a_planner_does_not_read_are_refused_not_ignored(tmp_path):
    with pytest.raises(TandemError) as excinfo:
        ToyPlanner.create(_ctx(tmp_path, itmes=["apple"]))
    assert "does not read planner.options itmes" in excinfo.value.message
    assert excinfo.value.hint == "Did you mean 'items'?"

    with pytest.raises(TandemError) as excinfo:
        _Minimal.create(_ctx(tmp_path, speed=2))
    assert "reads no options" in excinfo.value.hint

    planner = ToyPlanner.create(_ctx(tmp_path, items=["kiwi"]))
    assert planner.options == {"items": ["kiwi"]} and set(planner.world.where) == {"kiwi"}


def test_a_planner_logs_into_the_session_log(tmp_path):
    seen: list = []
    ctx = replace(_ctx(tmp_path), on_log=lambda stream, text: seen.append((stream, text)))
    ToyPlanner(ctx).log("picked a bin")
    assert seen == [("tandem", "picked a bin")]
    ToyPlanner().log("no session: goes to Python's logging, and does not fail")


# --- what a planner gets for nothing ------------------------------------------------------------------------


def test_every_verb_a_planner_did_not_write_has_a_default(tmp_path):
    planner = _Minimal()
    planner.require_ready()
    planner.warm()
    planner.release_hardware()
    planner.release_hardware()
    planner.reacquire_hardware()
    planner.home()
    planner.close()
    planner.close()


def test_a_frame_or_a_motion_nobody_implemented_is_refused_loudly():
    planner = _Minimal()
    with pytest.raises(UnsupportedVerb) as excinfo:
        planner.capture_frame(camera="external")
    assert isinstance(excinfo.value, TandemError)
    assert "_Minimal.capture_frame is not implemented" in excinfo.value.message
    assert "hitl.check_human_effects" in excinfo.value.hint
    with pytest.raises(UnsupportedVerb, match="move_to_joints is not implemented"):
        planner.move_to_joints([0.0] * 7)


_PIN = SourcePin("toy-src", "https://example.invalid/toy.git", "0123456789abcdef0123456789abcdef01234567")


def _recipe(planner: str = "built") -> RuntimeRecipe:
    return RuntimeRecipe(planner=planner, sources=(Source(_PIN),))


def test_a_planner_with_a_recipe_checks_its_runtime_before_a_session_starts():
    class Built(_Minimal):
        info = PlannerInfo(name="built")
        CAPABILITIES = replace(TOY_CAPABILITIES, name="built")
        recipe = _recipe()

    assert Built.info.sources == (_PIN,), "the catalog's pins are filled in from the recipe"
    runtime = Built.runtime()
    assert isinstance(runtime, RecipeRuntime) and runtime.root.name == "built"
    with pytest.raises(RuntimeNotReady, match="not ready"):
        Built().require_ready()


# --- declarations are checked when the class is defined ------------------------------------------------


def _declare(caps: Capabilities | None = None, **attributes):
    body = {
        "info": PlannerInfo(name="toy"),
        "CAPABILITIES": caps if caps is not None else TOY_CAPABILITIES,
        **attributes,
    }
    return type("Declared", (_Minimal,), body)


@pytest.mark.parametrize(
    ("change", "complaint"),
    [
        ({"moved_arguments": {"InBin": 1}}, r"moved_arguments\['InBin'\] points at \?bin, a container"),
        (
            {"goal_predicate_wire_names": {"Inbin": "in_bin"}},
            r"'Inbin', which is not a goal predicate \(did you mean 'InBin'\?\)",
        ),
        ({"achievable_predicates": frozenset()}, "not in achievable_predicates"),
        ({"reserved_predicate_names": frozenset()}, "could invent a predicate of the same name"),
        ({"prompt_fragments": {"placement_semantic": "..."}}, r"did you mean 'placement_semantics'\?"),
        ({"moved_arguments": {}}, "every leg would be told it may pick nothing"),
        ({"predicate_descriptions": {"InBin": "{0} is inside {2}"}}, "does not format with 2 argument"),
        ({"robot_operators": ("Drop(?obj: fruit)",)}, "uses the type 'fruit'"),
        ({"robot_operators": ("Drop an item",)}, "is not a signature"),
        ({"supports_return_home": "false"}, "supports_return_home is 'false', not True or False"),
        ({"exclusive_arguments": {"InBin": 2}}, r"exclusive_arguments\['InBin'\] = 2, but InBin takes 2"),
        ({"checkable_predicates": frozenset({"Glowing"})}, "checkable_predicates names Glowing"),
        ({"surface_type": "item"}, "both 'item'"),
        ({"goal_predicate_wire_names": {}}, "every goal would reach the planner empty"),
        (
            {"goal_predicates": {"InBin": IN_BIN, "Held": IN_BIN}},
            r"goal_predicates\['Held'\] is a predicate named 'InBin'",
        ),
    ],
)
def test_a_capability_declaration_that_would_mislead_is_refused_at_definition(change, complaint):
    with pytest.raises(TandemError, match=complaint):
        _declare(replace(TOY_CAPABILITIES, **change))


def test_every_problem_in_a_declaration_is_named_at_once():
    broken = replace(
        TOY_CAPABILITIES, achievable_predicates=frozenset(), moved_arguments={}, robot_operators=("x",)
    )
    problems = capability_problems(broken)
    assert len(problems) == 3, problems
    with pytest.raises(TandemError) as excinfo:
        _declare(broken)
    assert all(problem in excinfo.value.message for problem in problems)


@pytest.mark.parametrize(
    ("attributes", "complaint"),
    [
        ({"info": PlannerInfo(name="other")}, "CAPABILITIES.name is 'toy' but info.name is 'other'"),
        (
            {
                "info": PlannerInfo(name="Toy Planner"),
                "CAPABILITIES": replace(TOY_CAPABILITIES, name="Toy Planner"),
            },
            "not a usable planner name",
        ),
        (
            {"info": PlannerInfo(name="toy", sources=(replace(_PIN, commit="main"),))},
            "not a full 40-character commit",
        ),
        ({"recipe": _recipe("someone-else")}, "its recipe is for the planner 'someone-else'"),
        (
            {
                "recipe": _recipe("toy"),
                "info": PlannerInfo(name="toy", sources=(replace(_PIN, commit="f" * 40),)),
            },
            "differs from the commits its recipe fetches",
        ),
        ({"recipe": "tiptop"}, "recipe is a str"),
        ({"OPTIONS": {"items": 3}}, "OPTIONS must map each option name"),
        ({"name": "toy2"}, "name is 'toy2' but info.name is 'toy'"),
        ({"CAPABILITIES": "On, Holding"}, "CAPABILITIES is a str"),
    ],
)
def test_the_rest_of_a_declaration_is_refused_at_definition_too(attributes, complaint):
    with pytest.raises(TandemError, match=complaint):
        _declare(**attributes)


def test_a_planner_with_nothing_declared_is_told_to_declare_or_say_it_is_a_base():
    with pytest.raises(TandemError, match="declares no info and no CAPABILITIES") as excinfo:

        class Bare(Planner):
            def perceive(self, **kwargs): ...
            def plan(self, *args, **kwargs): ...
            def execute(self, *args, **kwargs): ...

    assert "abstract=True" in excinfo.value.message

    class Base(Planner, abstract=True):
        """Shared plumbing for a family of planners; declares nothing, is never registered."""

        def perceive(self, **kwargs): ...
        def plan(self, *args, **kwargs): ...
        def execute(self, *args, **kwargs): ...

    class Member(Base):
        info = PlannerInfo(name="member")
        CAPABILITIES = replace(TOY_CAPABILITIES, name="member")

    assert Base._planner_base and not Member._planner_base and Member.name == "member"


def test_a_sidecar_planner_must_say_where_its_sidecar_is(tmp_path):
    with pytest.raises(TandemError, match="declares no SIDECAR"):

        class Nowhere(SidecarPlanner):
            info = PlannerInfo(name="nowhere")
            CAPABILITIES = replace(TOY_CAPABILITIES, name="nowhere")

    with pytest.raises(TandemError, match="does not exist"):

        class Missing(SidecarPlanner):
            info = PlannerInfo(name="missing")
            CAPABILITIES = replace(TOY_CAPABILITIES, name="missing")
            SIDECAR = "no_such_sidecar.py"


# --- TiPToP is built with the SDK too ----------------------------------------------------------------------


def test_tiptop_is_a_sidecar_planner_with_a_sound_declaration():
    assert issubclass(TiptopBackend, SidecarPlanner)
    assert TiptopBackend.info is TIPTOP_INFO and TiptopBackend.CAPABILITIES is TIPTOP_CAPABILITIES
    assert capability_problems(TIPTOP_CAPABILITIES) == []
    assert factory_problems(TIPTOP) == [] and factory_problems(TiptopBackend) == []
    # The registry still holds TiPToP's factory object; the class answers the same questions.
    assert registry.factory("tiptop") is TIPTOP
    assert TiptopBackend.capabilities() is TIPTOP.capabilities()
    assert TiptopBackend.info.sources == TIPTOP.info.sources
