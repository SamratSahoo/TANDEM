"""A planner in its own environment: SidecarPlanner on tandem's side, tandem_sidecar on the planner's.

What TiPToP's backend and sidecar used to do for TiPToP alone -- launch a script in the planner's
runtime, speak JSON lines to it, survive what its libraries print, time it out, notice it died, hand
its hardware over -- is now generic, and any planner gets it by writing handler functions. These tests
drive the toy world's sidecar (``tests/toy_sidecar.py``) through it, including every way it can
misbehave, and then the REAL TiPToP sidecar, which runs here as far as it can without tiptop: far
enough to show it speaks through the kit exactly as it spoke before.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from helpers import wait_for
from toy_planner import ToySidecarPlanner

from tandem.core.errors import TandemError
from tandem.planners import GoalAtom, LegSpec, PlannerInfo, sidecar_kit
from tandem.planners.base import VERBS, BackendContext, BackendError
from tandem.planners.sdk import UnsupportedVerb
from tandem.planners.sidecar import STOP_FILE_ENV
from tandem.planners.tiptop.backend import TiptopBackend, sidecar_path

GOAL = [GoalAtom("in_bin", ("apple", "red_bin"))]


def _logs():
    seen: list = []
    return seen, lambda stream, text: seen.append((stream, text))


@pytest.fixture
def toy(tmp_path):
    """A factory for toy sidecar planners, every one of which is closed at the end of the test."""
    built: list = []

    def make(*flags: str, cls=ToySidecarPlanner, ctx=None):
        seen, on_log = _logs()
        planner = cls(ctx, flags=flags, on_log=on_log)
        planner.seen = seen
        built.append(planner)
        return planner

    yield make
    for planner in built:
        planner.close()


# --- the kit itself ---------------------------------------------------------------------------------------


def _kit_tree() -> ast.Module:
    return ast.parse(sidecar_kit.path().read_text())


def test_the_kit_knows_exactly_the_protocols_verbs():
    # It cannot import tandem.planners.base, so it restates them. Pinned, like TiPToP's sidecar's copy.
    (declared,) = [
        tuple(el.value for el in node.value.elts)
        for node in _kit_tree().body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "PROTOCOL_VERBS" for t in node.targets)
    ]
    assert declared == VERBS


def test_nothing_but_the_kit_is_put_on_every_sidecars_path():
    # The whole directory goes on the path, so anything else in it could shadow a planner's own module
    # of the same name. (That the kit needs only the standard library is test_packaging.py's.)
    shipped = {p.name for p in sidecar_kit.DIRECTORY.iterdir() if p.name != "__pycache__"}
    assert shipped == {"__init__.py", "tandem_sidecar.py"}


def test_the_kit_is_first_on_the_sidecars_path_and_nothing_else_changes(toy):
    planner = toy()
    planner._env = {"PYTHONPATH": os.pathsep.join(["/opt/planner", str(sidecar_kit.DIRECTORY)]), "HOME": "/h"}
    env = planner.launch_env()
    assert env["PYTHONPATH"].split(os.pathsep) == [str(sidecar_kit.DIRECTORY), "/opt/planner"]
    assert env["HOME"] == "/h" and planner._env["PYTHONPATH"].startswith("/opt"), (
        "the caller's env is not mutated"
    )


# --- the cycle, over the wire ----------------------------------------------------------------------------


def test_a_sidecar_planner_runs_the_sub_goal_cycle_in_another_process(toy, tmp_path):
    planner = toy()
    planner.warm()
    assert planner._channel.hello["pid"] != os.getpid()
    assert planner.sidecar_verbs >= {"perceive", "plan", "execute", "where", "last_args"}

    scene = planner.perceive(task_hint="bin the apple", save_dir=tmp_path / "p")
    assert "apple" in scene.object_labels and scene.surface_labels == {"blue_bin", "red_bin"}
    result = planner.plan(
        scene.scene_id, GOAL, surfaces=frozenset(scene.surface_labels), save_dir=tmp_path / "l"
    )
    assert result.ok and result.task_plan == ("Drop(apple, red_bin)",)
    leg = LegSpec("t-1", instruction="bin the apple", phase_index=0, n_phases=1)
    executed = planner.execute(result.plan_handle, leg, save_dir=tmp_path / "l")
    assert executed.ok and executed.n_frames > 0
    assert json.loads((tmp_path / "l" / "_meta.json").read_text())["trajectory_id"] == "t-1"
    # A verb of the planner's own, reached through call().
    assert planner.call("where")["apple"] == "red_bin"
    assert Path(planner.capture_frame()).is_file()


def test_what_the_planner_declared_goes_on_the_wire_and_nothing_else(toy, tmp_path):
    planner = toy()
    planner.warm()
    scene = planner.perceive(task_hint="x", save_dir=tmp_path / "p")
    planner.plan(scene.scene_id, GOAL, save_dir=tmp_path / "l")
    sent = planner.call("last_args", of="plan")
    # Declared: no restriction travels as null (not [] -- "pick nothing"), and the leg goes home.
    assert sent["movables"] is None and sent["return_home"] is True
    assert "reuse_skeleton" not in sent, "skeleton reuse is not declared, so the sidecar is never handed it"

    class Plain(ToySidecarPlanner):
        info = PlannerInfo(name="plain")
        CAPABILITIES = replace(
            ToySidecarPlanner.CAPABILITIES,
            name="plain",
            supports_movable_restriction=False,
            supports_return_home=False,
            supports_cooperative_stop=False,
        )

    plain = toy(cls=Plain)
    plain.warm()
    scene = plain.perceive(task_hint="x", save_dir=tmp_path / "p2")
    assert plain.plan(scene.scene_id, GOAL, save_dir=tmp_path / "l2").ok
    assert set(plain.call("last_args", of="plan")) == {"scene_id", "goal", "surfaces", "save_dir"}
    # Asking an undeclared planner for either is an error, not a plan quietly made without it.
    with pytest.raises(TandemError, match="capabilities do not declare supports_movable_restriction"):
        plain.plan(scene.scene_id, GOAL, movables=frozenset({"apple"}), save_dir=tmp_path / "l3")
    with pytest.raises(TandemError, match="capabilities do not declare supports_return_home"):
        plain.plan(scene.scene_id, GOAL, return_home=False, save_dir=tmp_path / "l3")
    assert STOP_FILE_ENV not in plain.launch_env(), "no stop file is offered to a planner that cannot stop"


def test_library_noise_on_the_sidecars_stdout_goes_to_the_log_not_the_protocol(toy, tmp_path):
    planner = toy("--noisy")
    planner.warm()
    for _ in range(3):
        assert planner.perceive(task_hint="x", save_dir=tmp_path / "p").scene_id
    noise = [text for stream, text in planner.seen if stream == "backend-stderr"]
    assert "Toy 1.0 initialized: CUDA not found, carrying on" in noise, "printed at import, before the hello"
    assert "noise while answering perceive" in noise


def test_a_stop_asked_for_mid_execution_crosses_into_the_sidecar(toy, tmp_path):
    # Slow frames (six to a drop), so the stop asked for on the second poll lands mid-drop.
    planner = toy("--step", "0.2")
    planner.warm()
    scene = planner.perceive(task_hint="x", save_dir=tmp_path / "p")
    result = planner.plan(scene.scene_id, GOAL, save_dir=tmp_path / "l")
    asks: list = []

    def should_stop() -> bool:
        asks.append(True)
        return len(asks) > 1  # not at once: let the leg start, then stop it

    executed = planner.execute(
        result.plan_handle, LegSpec("t-1"), save_dir=tmp_path / "l", should_stop=should_stop
    )
    assert executed.stopped_early and not executed.ok and len(asks) >= 2
    assert planner.call("where")["apple"] == "floor", "a stopped drop did not happen"
    assert not planner._stop_file.exists(), (
        "the stop is cleared, so the next leg is not stopped before it starts"
    )
    again = planner.plan(scene.scene_id, GOAL, save_dir=tmp_path / "m")
    assert planner.execute(
        again.plan_handle, LegSpec("t-2"), save_dir=tmp_path / "m", should_stop=lambda: False
    ).ok


def test_events_the_sidecar_sends_land_in_the_sessions_events_file(toy, tmp_path):
    events_file = tmp_path / "events.jsonl"
    ctx = BackendContext(
        profile=None, session_dir=tmp_path, output_dir=tmp_path / "legs", events_file=events_file
    )
    planner = toy("--events", ctx=ctx)
    planner.warm()
    scene = planner.perceive(task_hint="x", save_dir=tmp_path / "p")
    result = planner.plan(scene.scene_id, GOAL, save_dir=tmp_path / "l")
    executed = planner.execute(result.plan_handle, LegSpec("t-1"), save_dir=tmp_path / "l")
    (event,) = [json.loads(line) for line in events_file.read_text().splitlines()]
    assert event == {
        "event": "toy_leg_recorded",
        "n_frames": executed.n_frames,
        "rollout_dir": executed.rollout_dir,
    }


# --- the ways a sidecar goes wrong -----------------------------------------------------------------------------


def test_a_sidecar_that_crashes_is_reported_with_its_exit_code_and_replaced_at_the_next_warm(toy, tmp_path):
    planner = toy("--crash-on", "plan")
    planner.warm()
    first = planner._channel.hello["pid"]
    scene = planner.perceive(task_hint="x", save_dir=tmp_path / "p")
    with pytest.raises(BackendError, match=r"exited \(code 3\) without answering"):
        planner.plan(scene.scene_id, GOAL, save_dir=tmp_path / "l")
    with pytest.raises(BackendError, match="not running"):
        planner.perceive(task_hint="x", save_dir=tmp_path / "p")

    planner.warm()
    assert planner._channel.hello["pid"] != first
    assert planner.perceive(task_hint="x", save_dir=tmp_path / "p").scene_id
    assert any("no longer running; starting a new one" in text for _, text in planner.seen)


def test_the_exit_status_of_a_sidecar_that_hung_up_is_waited_for(toy, tmp_path):
    # Its output closes before it is reaped. Read at that instant the status is None, and "exited
    # (code None)" tells the person next to the arm nothing about what happened.
    planner = toy("--hang-up-on", "perceive")
    planner.warm()
    with pytest.raises(BackendError, match=r"exited \(code 4\)"):
        planner.perceive(task_hint="x", save_dir=tmp_path)


def test_the_sidecar_cleans_up_whether_it_is_asked_to_quit_or_tandem_just_goes_away(toy):
    """What releases a real planner's cameras. A tandem that crashed never says quit; the sidecar sees
    its stdin close, and must clean up all the same."""
    closed = ("backend", "the toy world is closed")
    asked = toy()
    asked.warm()
    asked.close()
    assert wait_for(lambda: closed in asked.seen, timeout=5.0)

    abandoned = toy()
    abandoned.warm()
    abandoned._channel._proc.stdin.close()
    assert wait_for(lambda: closed in abandoned.seen, timeout=5.0)


def test_a_wedged_sidecar_is_reported_as_one_that_may_hold_the_robot(toy, tmp_path):
    class Impatient(ToySidecarPlanner):
        TIMEOUTS = {**ToySidecarPlanner.TIMEOUTS, "plan": 0.3}

    planner = toy("--slow", "plan", "1.0", cls=Impatient)
    planner.warm()
    scene = planner.perceive(task_hint="x", save_dir=tmp_path / "p")
    with pytest.raises(BackendError, match="did not answer within 0s; it may be wedged holding the robot"):
        planner.plan(scene.scene_id, GOAL, save_dir=tmp_path / "l")
    # The late answer is thrown away rather than read as the reply to the next question.
    assert planner.perceive(task_hint="x", save_dir=tmp_path / "p2").object_labels
    assert any("discarding a late reply" in text for _, text in planner.seen)


def test_a_verb_the_sidecar_does_not_answer_falls_back_to_the_default(toy):
    planner = toy("--only", "warm,perceive,plan,execute")
    planner.warm()
    planner.home()
    planner.release_hardware()
    planner.reacquire_hardware()
    with pytest.raises(UnsupportedVerb, match="cannot capture a camera frame"):
        planner.capture_frame()


def test_a_sidecar_missing_a_verb_every_planner_needs_is_refused_before_anything_relies_on_it(toy):
    planner = toy("--only", "warm,perceive,execute")
    with pytest.raises(BackendError, match="does not answer plan, which every planner must"):
        planner.warm()
    assert planner._channel is None, "and it is not left running"


def test_a_sidecar_that_cannot_start_says_why(toy):
    planner = toy("--bad-handler")
    with pytest.raises(
        BackendError,
        match="did not start: the sidecar could not start: ValueError: no handler for the verb 'nonexistent'",
    ):
        planner.warm()


def test_only_the_hello_ends_the_handshake(toy):
    # A JSON object with no "id" used to match the handshake's own id and be taken for the hello.
    planner = toy("--junk-before-hello")
    planner.warm()
    assert planner._channel.hello.get("ready") is True
    assert any("before the backend announced itself" in text for _, text in planner.seen)


def test_every_verb_before_warm_is_an_error_and_close_is_not(toy, tmp_path):
    planner = toy()
    for call in (
        lambda: planner.perceive(task_hint="x", save_dir=tmp_path),
        planner.release_hardware,
        planner.home,
        lambda: planner.call("where"),
    ):
        with pytest.raises(BackendError, match="toy-sidecar backend has not been warmed"):
            call()
    planner.close()
    planner.close()


# --- TiPToP's sidecar, through the kit ------------------------------------------------------------------------


class _ThisInterpreter:
    """A runtime that launches TiPToP's REAL sidecar with this interpreter: no pixi, no tiptop, no GPU."""

    def __init__(self, root: Path) -> None:
        self.tiptop_dir = root
        self.commands: list = []

    def command(self, argv: list[str]) -> list[str]:
        self.commands.append(list(argv))
        return [sys.executable, *argv[1:]]

    def require_ready(self) -> None:
        pass


def test_tiptops_sidecar_speaks_through_the_kit_exactly_as_it_spoke_before(tmp_path):
    """Everything short of importing tiptop: the handshake, the verbs, the failure format, the exit.

    The sidecar imports tiptop inside each verb, so without it every verb fails -- which is the
    point: it fails over the channel, in the words the parent has always shown an operator, and the
    channel carries on.
    """
    runtime = _ThisInterpreter(tmp_path)
    seen, on_log = _logs()
    backend = TiptopBackend(runtime, env=dict(os.environ), output_dir=tmp_path, on_log=on_log)
    with pytest.raises(BackendError, match=r"^warm failed -- ModuleNotFoundError"):
        backend.warm()
    try:
        assert runtime.commands == [["python", str(sidecar_path())]]
        hello = backend._channel.hello
        assert hello["verbs"] == list(VERBS) and hello["ready"] is True
        assert backend.sidecar_verbs == frozenset(VERBS)
        with pytest.raises(
            BackendError, match=r"^home failed -- ModuleNotFoundError: No module named 'tiptop'"
        ):
            backend.home()
        with pytest.raises(BackendError, match=r"^capabilities failed -- ModuleNotFoundError"):
            backend.call("capabilities")
        with pytest.raises(BackendError, match=r"^unknown verb 'last_args'"):
            backend.call("last_args", of="plan")
        assert any("Traceback" in text for _, text in seen), "the traceback is in the session log"
    finally:
        backend.close()
    assert backend._channel is None


def test_tiptop_is_launched_and_warmed_exactly_as_before(tmp_path):
    runtime = _ThisInterpreter(tmp_path / "runtime")
    overrides = tmp_path / "curobo-overrides.json"
    env = {"TIPTOP_CONFIG": "/p/tiptop.yml", "PYTHONPATH": "/opt/x"}
    backend = TiptopBackend(
        runtime,
        env=env,
        output_dir=tmp_path / "out",
        execute=False,
        record=True,
        cost_overrides_file=overrides,
    )
    assert backend.launch_command() == [sys.executable, str(sidecar_path())]
    assert backend.launch_cwd() == runtime.tiptop_dir
    launched = backend.launch_env()
    assert launched["TIPTOP_CONFIG"] == "/p/tiptop.yml"
    assert launched["PYTHONPATH"] == os.pathsep.join([str(sidecar_kit.DIRECTORY), "/opt/x"])
    assert STOP_FILE_ENV not in launched, "TiPToP cannot stop mid-plan, so it is offered no stop file"
    assert backend.warm_args() == {
        "output_dir": str(tmp_path / "out"),
        "execute": False,
        "record": True,
        "cost_overrides": str(overrides),
    }
    assert {verb: backend.timeout(verb) for verb in ("warm", "perceive", "plan", "execute", "home")} == {
        "warm": 900.0,
        "perceive": 300.0,
        "plan": 900.0,
        "execute": 1800.0,
        "home": 180.0,
    }
