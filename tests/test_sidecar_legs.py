"""A robot plan as one LEG of a longer task: which objects it may pick, where it ends, what it did.

Three things separate a leg from a whole rollout, and every one of them is implemented on tandem's
side of the TiPToP boundary, around tiptop's public functions rather than inside them:

- ``movables``: only the named objects may be picked. The rest stay in the world as obstacles, so a
  robot phase never picks up the screwdriver the person's phase is about.
- ``return_home=False``: no drive home at the end of a leg something else continues from.
- ``task_plan``: the operators the plan ran, in the terms a person reads (``Pick(bread)``).

The sidecar that does this runs in a GPU environment with a robot attached, so it cannot run here.
What can be tested is split in two. The decisions are pure functions of plain data, loaded out of the
sidecar's source and run against stub objects. The calls into tiptop and cuTAMP are checked
statically against the planner's sources, the same way ``test_planners.py`` checks the rest of the
sidecar -- a renamed helper or a new constructor argument is an ImportError or a silently different
world forty seconds into a warm-up otherwise.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from fake_backend import FakeBackend
from planner_sources import planner_sources
from ruamel.yaml import YAML

from tandem.planners.base import GoalAtom, PlanResult, TampBackend
from tandem.planners.tiptop.backend import TiptopBackend, sidecar_path

FAKE_SIDECAR = Path(__file__).parent / "fake_sidecar.py"
LEG_HELPERS = ("goal_moves_outside", "restrict_movables", "task_plan_labels", "return_target")


# --- loading the sidecar's pure helpers ------------------------------------------------------------


def _sidecar_tree() -> ast.Module:
    return ast.parse(sidecar_path().read_text())


def _sidecar_helper(name: str) -> ast.FunctionDef:
    """One top-level function of the sidecar, as source."""
    found = [n for n in _sidecar_tree().body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert found, f"the sidecar no longer defines {name}"
    return found[0]


def _sidecar_functions(*names: str):
    """Top-level functions of the sidecar, compiled from its source without running the file.

    Importing it is not an option: the first thing it does is take fd 1 away from its host, which in
    a test process is pytest's own capture. The leg helpers are written to need nothing but their
    arguments (see test_the_leg_helpers_need_nothing_but_their_arguments), so compiling them alone into
    an empty namespace is enough to run them.
    """
    found = {n.name: n for n in _sidecar_tree().body if isinstance(n, ast.FunctionDef) and n.name in names}
    missing = set(names) - set(found)
    assert not missing, f"the sidecar no longer defines {sorted(missing)}"
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *found.values()], type_ignores=[]))
    namespace: dict = {}
    exec(compile(module, str(sidecar_path()), "exec"), namespace)
    return tuple(namespace[name] for name in names)


def test_the_leg_helpers_need_nothing_but_their_arguments():
    """The loader above works only while this holds, and nothing else would notice it stopped.

    A helper that reached for a module global -- ``_log``, a constant, tiptop -- would still run in the
    real sidecar and fail only here, or worse, only on the one branch no test takes.
    """
    for node in _sidecar_tree().body:
        if not (isinstance(node, ast.FunctionDef) and node.name in LEG_HELPERS):
            continue
        local = {a.arg for a in ast.walk(node.args) if isinstance(a, ast.arg)}
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                local.add(sub.id)
            elif isinstance(sub, ast.comprehension):
                local.update(n.id for n in ast.walk(sub.target) if isinstance(n, ast.Name))
        loaded = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        foreign = sorted(loaded - local - set(dir(builtins)))
        assert not foreign, f"{node.name} reads {foreign} from outside its arguments"


# --- stubs standing in for cuTAMP's objects --------------------------------------------------------


@dataclass(frozen=True)
class Obj:
    """An obstacle, as far as the rebuild is concerned: something with a name."""

    name: str


class StubEnvironment:
    """cuTAMP's TAMPEnvironment constructor, down to the one check the rebuild relies on."""

    def __init__(self, name, movables, statics, type_to_objects, goal_state, pick_transparent=()):
        self.name = name
        self.movables = movables
        self.statics = statics
        self.type_to_objects = type_to_objects
        self.goal_state = goal_state
        self.pick_transparent = tuple(pick_transparent)
        both = {o.name for o in movables} & {o.name for o in statics}
        if both:
            raise ValueError(f"Objects cannot be both movable and static: {both}")


def _kitchen() -> StubEnvironment:
    """What create_tamp_environment builds for "put the bread on the plate" with a person's tool about."""
    bread, screwdriver, toy = Obj("bread"), Obj("screwdriver"), Obj("toy")
    table, plate = Obj("table"), Obj("plate")
    return StubEnvironment(
        name="tiptop_cutamp",
        movables=[bread, screwdriver, toy],
        statics=[Obj("workspace_back"), table, plate],
        type_to_objects={"Movable": [bread, screwdriver, toy], "Surface": [table, plate]},
        goal_state=frozenset({"On(bread, plate)", "HandEmpty()"}),
        pick_transparent=("plate",),
    )


# --- movables ---------------------------------------------------------------------------------------


def test_a_restricted_leg_keeps_every_other_movable_as_an_obstacle():
    (restrict_movables,) = _sidecar_functions("restrict_movables")
    env = _kitchen()
    rebuilt, demoted = restrict_movables(env, frozenset({"bread"}), StubEnvironment)

    assert [o.name for o in rebuilt.movables] == ["bread"]
    assert rebuilt.type_to_objects["Movable"] == rebuilt.movables
    # Demoted, not deleted: the arm still has to route around the screwdriver.
    assert [o.name for o in rebuilt.statics] == ["workspace_back", "table", "plate", "screwdriver", "toy"]
    assert demoted == ["screwdriver", "toy"]
    # Everything that is not about which objects move is the world it was.
    assert rebuilt.type_to_objects["Surface"] is env.type_to_objects["Surface"]
    assert rebuilt.goal_state == env.goal_state
    assert rebuilt.pick_transparent == ("plate",)
    assert rebuilt.name == env.name
    # And the environment create_tamp_environment returned is left exactly as it was.
    assert [o.name for o in env.movables] == ["bread", "screwdriver", "toy"]
    assert len(env.statics) == 3


def test_a_restriction_that_demotes_nothing_hands_back_the_same_environment():
    (restrict_movables,) = _sidecar_functions("restrict_movables")
    env = _kitchen()
    rebuilt, demoted = restrict_movables(env, {"bread", "screwdriver", "toy"}, StubEnvironment)
    assert rebuilt is env and demoted == []


def test_names_that_are_not_movables_here_are_not_kept_or_duplicated():
    # A pinned surface and a label from another pass. Neither is a movable of THIS environment, and
    # neither may end up in the world twice -- the constructor would refuse a surface that is both.
    (restrict_movables,) = _sidecar_functions("restrict_movables")
    rebuilt, demoted = restrict_movables(_kitchen(), {"bread", "plate", "blue_toy"}, StubEnvironment)
    assert [o.name for o in rebuilt.movables] == ["bread"]
    assert [o.name for o in rebuilt.statics].count("plate") == 1
    assert demoted == ["screwdriver", "toy"]


def test_an_empty_restriction_leaves_nothing_to_pick():
    # frozenset() and None are different requests; only None means "no restriction".
    (restrict_movables,) = _sidecar_functions("restrict_movables")
    rebuilt, demoted = restrict_movables(_kitchen(), frozenset(), StubEnvironment)
    assert rebuilt.movables == [] and rebuilt.type_to_objects["Movable"] == []
    assert demoted == ["bread", "screwdriver", "toy"]


def test_what_the_planner_hangs_on_an_environment_survives_the_rebuild():
    # A planner that attaches something after construction -- per-surface support points, say -- is
    # describing the scene. Losing it would change the plan with nothing to say it had.
    (restrict_movables,) = _sidecar_functions("restrict_movables")
    env = _kitchen()
    env.support_points = {"plate": [(0.0, 0.0, 0.0)]}
    rebuilt, _ = restrict_movables(env, {"bread"}, StubEnvironment)
    assert rebuilt.support_points is env.support_points


def test_a_goal_that_moves_something_the_leg_may_not_pick_is_named():
    (goal_moves_outside,) = _sidecar_functions("goal_moves_outside")
    goal = [{"predicate": "on", "args": ["bread", "plate"]}, {"predicate": "holding", "args": ["toy"]}]
    assert goal_moves_outside(goal, {"bread", "toy"}) == []
    assert goal_moves_outside(goal, {"bread"}) == ["toy"]
    assert goal_moves_outside(goal, set()) == ["bread", "toy"]
    # The surface is not moved by being placed on, and a predicate the planner never builds a goal
    # from moves nothing either.
    assert goal_moves_outside([{"predicate": "on", "args": ["bread", "plate"]}], {"bread"}) == []
    assert goal_moves_outside([{"predicate": "inside", "args": ["bread", "box"]}], set()) == []


# --- task_plan --------------------------------------------------------------------------------------


def _pick_and_place(obj: str, surface: str, n: int) -> list[dict]:
    """The steps cuTAMP's motion solver emits for one pick and one place, labels and all."""
    pick = f"Pick({obj}, grasp{n}, q{2 * n + 1})"
    place = f"Place({obj}, grasp{n}, placement{n}, {surface}, q{2 * n + 2})"
    return [
        {"type": "trajectory", "label": pick},
        {"type": "trajectory", "label": pick},
        {"type": "gripper", "action": "close", "label": pick},
        {"type": "trajectory", "label": place},
        {"type": "trajectory", "label": place},
        {"type": "trajectory", "label": place},
        {"type": "gripper", "action": "open", "label": place},
    ]


# cuTAMP closes every plan with two segments under this one label: the retract, then the drive home.
GO_HOME = [{"type": "trajectory", "label": "GoToInitial(q0)"}] * 2
SCENE = ["bread", "plate", "toy", "screwdriver", "table"]


def test_the_task_plan_is_the_operators_in_the_terms_a_person_reads():
    (task_plan_labels,) = _sidecar_functions("task_plan_labels")
    labels = task_plan_labels(_pick_and_place("bread", "plate", 0) + GO_HOME, SCENE)
    # The paper's figure, exactly: motion-level arguments (grasp, placement, conf) gone, one label per
    # operator rather than per trajectory segment, and no drive home.
    assert labels == ["Pick(bread)", "Place(bread, plate)"]
    assert json.loads(json.dumps(labels)) == labels


def test_only_consecutive_repeats_collapse():
    (task_plan_labels,) = _sidecar_functions("task_plan_labels")
    steps = _pick_and_place("bread", "plate", 0) + _pick_and_place("toy", "table", 1) + GO_HOME
    assert task_plan_labels(steps, SCENE) == [
        "Pick(bread)",
        "Place(bread, plate)",
        "Pick(toy)",
        "Place(toy, table)",
    ]


def test_a_blended_or_unlabelled_step_does_not_invent_an_operator():
    # Blending merges a run of segments into one step carrying the FIRST segment's label, so a
    # blended plan reads the same; a step with no label says nothing and is skipped.
    (task_plan_labels,) = _sidecar_functions("task_plan_labels")
    steps = [
        {"type": "trajectory", "label": "Pick(bread, grasp0, q1)"},
        {"type": "gripper", "action": "close"},
        {"type": "gripper", "action": "close", "label": "Pick(bread, grasp0, q1)"},
        {"type": "trajectory", "label": "Place(bread, grasp0, placement0, plate, q2)"},
        {"type": "trajectory", "label": ""},
    ]
    assert task_plan_labels(steps, SCENE) == ["Pick(bread)", "Place(bread, plate)"]


# --- where the plan ends ----------------------------------------------------------------------------

Q_HOME = [0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.0]


def test_a_leg_started_from_a_chosen_pose_ends_where_an_ordinary_rollout_would():
    # Parked at home, or at the wrist camera's capture pose: cuTAMP's own default, back where it
    # started, is what a rollout has always done, and is left alone.
    (return_target,) = _sidecar_functions("return_target")
    assert return_target(arm_placed=True, q_home=Q_HOME, n_joints=7, arm_mode="single") == (None, None)


def test_a_leg_started_wherever_the_arm_was_left_is_sent_home_instead():
    # After a person teleoperated, "back where it started" is back where THEY left the arm.
    (return_target,) = _sidecar_functions("return_target")
    target, why_not = return_target(arm_placed=False, q_home=tuple(Q_HOME), n_joints=7, arm_mode="single")
    assert target == Q_HOME and why_not is None
    assert all(isinstance(q, float) for q in target), "handed to cuTAMP as plain floats"


def test_a_home_that_cannot_be_used_is_said_out_loud_not_guessed_at():
    (return_target,) = _sidecar_functions("return_target")
    target, why_not = return_target(arm_placed=False, q_home=Q_HOME, n_joints=14, arm_mode="dual")
    assert target is None and "dual" in why_not, "cuTAMP raises on a return pose for a dual-arm plan"
    target, why_not = return_target(arm_placed=False, q_home=Q_HOME[:6], n_joints=7, arm_mode="single")
    assert target is None and "6 joints" in why_not
    target, why_not = return_target(arm_placed=False, q_home=None, n_joints=7, arm_mode="single")
    assert target is None and "q_home" in why_not


# --- the protocol ------------------------------------------------------------------------------------


def test_a_plan_result_carries_its_task_plan_as_plain_strings():
    result = PlanResult.from_dict({"ok": True, "task_plan": ["Pick(bread)", "Place(bread, plate)"]})
    assert result.task_plan == ("Pick(bread)", "Place(bread, plate)")
    assert PlanResult.from_dict({"ok": True}).task_plan == ()
    assert PlanResult.from_dict({"ok": True, "task_plan": None}).task_plan == ()
    assert PlanResult().task_plan == ()


@pytest.mark.parametrize("backend", [TiptopBackend, FakeBackend], ids=["tiptop", "fake"])
@pytest.mark.parametrize("verb", ["perceive", "plan"])
def test_a_backend_takes_every_keyword_the_protocol_names(backend, verb):
    """Method presence is all runtime_checkable checks. A missing keyword is a TypeError at the first
    robot phase after a human one -- the one thing every earlier test skipped."""
    wanted = inspect.signature(getattr(TampBackend, verb)).parameters
    got = inspect.signature(getattr(backend, verb)).parameters
    for name, parameter in wanted.items():
        if name == "self":
            continue
        assert name in got, f"{backend.__name__}.{verb} does not take {name}"
        assert got[name].default == parameter.default, f"{backend.__name__}.{verb}({name}=) default differs"


def test_the_fake_backend_records_what_each_leg_was_asked_for(tmp_path):
    backend = FakeBackend(output_dir=tmp_path)
    scene = backend.perceive(task_hint="t", save_dir=tmp_path / "p", reset_arm=False, open_gripper=True)
    goal = [GoalAtom("on", ("blue_toy", "white_box"))]
    result = backend.plan(
        scene.scene_id,
        goal,
        surfaces=frozenset({"white_box"}),
        movables=frozenset({"blue_toy"}),
        return_home=False,
        save_dir=tmp_path / "l",
    )
    assert result.ok and result.task_plan == ("Pick(blue_toy)", "Place(blue_toy, white_box)")
    assert backend.perceive_requests == [{"task_hint": "t", "reset_arm": False, "open_gripper": True}]
    (request,) = backend.plan_requests
    assert request["movables"] == frozenset({"blue_toy"}) and request["return_home"] is False
    # The ordering record every existing test reads is unchanged by any of it.
    assert backend.calls == ["perceive:keep", 'plan:[{"predicate": "on", "args": ["blue_toy", "white_box"]}]']

    # Unasked, the defaults an ordinary rollout has always had.
    backend.plan(scene.scene_id, goal, save_dir=tmp_path / "l2")
    assert backend.plan_requests[-1]["movables"] is None and backend.plan_requests[-1]["return_home"] is True


def test_the_fake_backend_refuses_a_goal_outside_its_movables_as_the_sidecar_does(tmp_path):
    backend = FakeBackend(output_dir=tmp_path)
    scene = backend.perceive(task_hint="t", save_dir=tmp_path / "p")
    result = backend.plan(
        scene.scene_id,
        [GoalAtom("on", ("blue_toy", "white_box"))],
        movables=frozenset({"white_box"}),
        save_dir=tmp_path / "l",
    )
    assert not result.ok and "blue_toy" in result.failure_reason


class _FakeSidecarRuntime:
    """Just enough of a Runtime for TiptopBackend.warm to launch the fake sidecar in place of the real
    one. Unlike helpers.FakeRuntime, launching something is the point here."""

    def __init__(self, tmp_path: Path) -> None:
        self.tiptop_dir = tmp_path

    def command(self, argv: list[str]) -> list[str]:
        return [sys.executable, str(FAKE_SIDECAR)]

    def require_ready(self) -> None:
        pass


def test_the_tiptop_backend_puts_the_leg_semantics_on_the_wire(tmp_path):
    backend = TiptopBackend(_FakeSidecarRuntime(tmp_path), env=dict(os.environ), output_dir=tmp_path)
    backend.warm()
    try:
        wire = backend._channel
        backend.perceive(task_hint="t", save_dir=tmp_path, reset_arm=False, open_gripper=True)
        sent = wire.call("last_args", of="perceive")
        assert sent["open_gripper"] is True and sent["reset_arm"] is False

        goal = [GoalAtom("on", ("blue_toy", "white_box"))]
        result = backend.plan(
            "s1", goal, surfaces=frozenset({"white_box"}), movables=frozenset({"blue_toy"}),
            return_home=False, save_dir=tmp_path,
        )
        assert result.ok and result.task_plan == ("Pick(blue_toy)", "Place(blue_toy, white_box)")
        sent = wire.call("last_args", of="plan")
        assert sent["movables"] == ["blue_toy"] and sent["return_home"] is False

        # No restriction travels as null, not as an empty list -- which would be "pick nothing".
        backend.plan("s1", goal, save_dir=tmp_path)
        sent = wire.call("last_args", of="plan")
        assert sent["movables"] is None and sent["return_home"] is True

        refused = backend.plan("s1", goal, movables=frozenset(), save_dir=tmp_path)
        assert not refused.ok and "blue_toy" in refused.failure_reason
    finally:
        backend.close()


# --- the calls into the planner, checked against its sources -----------------------------------------


def _planner_sources() -> Path:
    """Where tiptop and cuTAMP's sources are, found the one way every static check finds them.

    ``$TANDEM_PLANNER_SOURCES`` pointed at other commits checks the sidecar against a bump before the
    pins move (see tests/planner_sources.py).
    """
    return planner_sources("tiptop/tiptop/tiptop_run.py", "cuTAMP/cutamp")


def _function(path: Path, name: str, *, cls: str | None = None) -> ast.FunctionDef:
    body = ast.parse(path.read_text()).body
    if cls is not None:
        classes = [n for n in body if isinstance(n, ast.ClassDef) and n.name == cls]
        assert classes, f"{path.name} no longer defines class {cls}"
        body = classes[0].body
    found = [n for n in body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert found, f"{path.name} no longer defines {cls + '.' if cls else ''}{name}"
    return found[0]


def _sidecar_method(name: str) -> ast.FunctionDef:
    return _function(sidecar_path(), name, cls="Sidecar")


def _calls(node: ast.AST, callee: str) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == callee
    ]


def _parameters(fn: ast.FunctionDef) -> tuple[set[str], set[str]]:
    """(every parameter name, the required ones), leaving out self."""
    positional = [a.arg for a in fn.args.posonlyargs + fn.args.args]
    required = positional[: len(positional) - len(fn.args.defaults)]
    required += [a.arg for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults, strict=True) if d is None]
    every = set(positional) | {a.arg for a in fn.args.kwonlyargs}
    return every - {"self"}, set(required) - {"self"}


def test_the_sidecar_imports_the_leg_symbols_from_where_the_planner_defines_them():
    root = _planner_sources()
    imported = {
        (node.module, alias.name)
        for node in ast.walk(_sidecar_tree())
        if isinstance(node, ast.ImportFrom) and node.module
        for alias in node.names
    }
    assert ("tiptop.goal_clearing", "drop_return_to_initial") in imported
    assert ("cutamp.envs.utils", "TAMPEnvironment") in imported

    drop = _function(root / "tiptop" / "tiptop" / "goal_clearing.py", "drop_return_to_initial")
    every, required = _parameters(drop)
    assert required == {"plan"} and every == {"plan"}, "the trim takes the plan and nothing else"
    utils = ast.parse((root / "cuTAMP" / "cutamp" / "envs" / "utils.py").read_text())
    assert any(isinstance(n, ast.ClassDef) and n.name == "TAMPEnvironment" for n in utils.body)


def test_the_environment_is_rebuilt_with_everything_the_planner_built_it_with():
    """The rebuild is only faithful while it passes what create_tamp_environment passes.

    A tiptop that starts building its environment with one more argument -- support points, a
    placement region -- would otherwise have that argument silently dropped from every restricted
    leg, and plan in a different world from the one it perceived.
    """
    root = _planner_sources()
    init = _function(root / "cuTAMP" / "cutamp" / "envs" / "utils.py", "__init__", cls="TAMPEnvironment")
    accepted, required = _parameters(init)

    restrict = _sidecar_helper("restrict_movables")
    (rebuild,) = _calls(restrict, "environment_cls")
    passed = {k.arg for k in rebuild.keywords}
    assert not rebuild.args, "pass the constructor's arguments by name, so this check can read them"
    assert passed <= accepted, f"TAMPEnvironment does not take {sorted(passed - accepted)}"
    assert required <= passed, f"the rebuild leaves out {sorted(required - passed)}"

    builder = _function(root / "tiptop" / "tiptop" / "tiptop_run.py", "create_tamp_environment")
    (built,) = _calls(builder, "TAMPEnvironment")
    assert {k.arg for k in built.keywords} == passed, "create_tamp_environment builds with other arguments"

    # And every attribute the rebuild reads off the old environment is one the constructor sets.
    assigned = {
        t.attr
        for n in ast.walk(init)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self"
    }
    read = {
        n.attr
        for n in ast.walk(restrict)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "env"
    }
    assert read <= assigned, f"TAMPEnvironment no longer sets {sorted(read - assigned)}"


def test_the_trimmed_plan_is_the_one_saved_executed_and_recorded():
    # _execute_plan_recorded builds the episode from the plan FILE and drives the plan OBJECT, so a
    # trim applied after serialising would record one plan and execute another.
    plan = _sidecar_method("plan")
    (trim,) = _calls(plan, "drop_return_to_initial")
    (serialise,) = _calls(plan, "serialize_plan")
    assert trim.lineno < serialise.lineno
    assert isinstance(serialise.args[0], ast.Name) and serialise.args[0].id == "cutamp_plan"
    remembered = [
        v
        for n in ast.walk(plan)
        if isinstance(n, ast.Dict)
        for k, v in zip(n.keys, n.values, strict=True)
        if isinstance(k, ast.Constant) and k.value == "plan"
    ]
    assert [v.id for v in remembered if isinstance(v, ast.Name)] == ["cutamp_plan"]


def test_the_return_pose_reaches_the_planner_through_an_argument_it_has():
    root = _planner_sources()
    (call,) = _calls(_sidecar_method("plan"), "run_planning")
    assert "q_return" in {k.arg for k in call.keywords}
    every, _ = _parameters(_function(root / "tiptop" / "tiptop" / "planning.py", "run_planning"))
    assert "q_return" in every

    # q_home is read from the robot config the planner itself homes to (go_to_home drives there).
    config = YAML(typ="safe").load((root / "tiptop" / "tiptop" / "config" / "tiptop.yml").read_text())
    assert len(config["robot"]["q_home"]) == 7


def _strings(node: ast.AST) -> set[str]:
    return {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def test_the_drive_home_is_called_the_same_thing_everywhere_it_is_named():
    """task_plan_labels leaves GoToInitial out by name and drop_return_to_initial trims by it; both only
    mean anything while cuTAMP's motion solver still calls the closing drive that."""
    root = _planner_sources()
    solver = ast.parse((root / "cuTAMP" / "cutamp" / "motion_solver.py").read_text())
    assert "GoToInitial(q0)" in _strings(solver)
    trim = _function(root / "tiptop" / "tiptop" / "goal_clearing.py", "drop_return_to_initial")
    assert "GoToInitial" in _strings(trim)
    assert "GoToInitial" in _strings(_sidecar_helper("task_plan_labels"))


def test_the_gripper_is_opened_through_calls_every_robot_client_answers():
    root = _planner_sources()
    ur5 = _function(root / "tiptop" / "tiptop" / "ur5" / "ur5_client.py", "open_gripper", cls="UR5Client")
    assert not (_parameters(ur5)[1]), "a bare open_gripper() must be enough for a one-handed robot"
    yam = _function(root / "tiptop" / "tiptop" / "yam" / "yam_client.py", "open_gripper", cls="YamClient")
    every, required = _parameters(yam)
    assert "arm" in every and not required, "the dual-arm branch names the hand"
