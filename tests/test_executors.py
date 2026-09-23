"""Human executors: who carries out a human phase, found by name, and teleop as the one that ships.

The registry half is about names. An unknown one is loud and suggests the nearest, an installed package
adds one through the entry point, and two packages claiming one name is an error rather than a guess.
The teleop half drives the real ``TeleopExecutor``, and the real ``TeleopChild`` inside it, against the
stand-in driver in ``fake_teleop.py``. That stand-in reproduces the real driver's gate: it records
only when told to start, and it quits from mid-recording only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomlkit
from pydantic import ValidationError

from tandem.core.errors import TandemError
from tandem.core.phase_loop import HumanPhase
from tandem.core.profiles import HitlSpec, Profile
from tandem.core.settings import Settings
from tandem.executors import base
from tandem.executors import teleop as teleop_mod
from tandem.executors.base import (
    CustodyError,
    ExecutorContext,
    ExecutorFactory,
    HumanExecutor,
    HumanPhaseRequest,
    HumanPhaseResult,
)
from tandem.planners.base import LegSpec

FAKE_TELEOP = Path(__file__).parent / "fake_teleop.py"
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    """Nothing registered in this process, and no installed package declaring an executor.

    Each test that wants an entry point installs its own through `installed`. The real scan would make
    the outcome depend on what happens to be pip-installed on the machine running the suite.
    """
    monkeypatch.setattr(base, "_registered", {})
    monkeypatch.setattr(base, "metadata", SimpleNamespace(entry_points=lambda group: []))
    monkeypatch.setattr(base, "_discovered", None)


@pytest.fixture
def installed(monkeypatch):
    """Pretend the given entry points are installed: (name, "module:attr", distribution name)."""

    def install(*entries: tuple[str, str, str | None]) -> None:
        points = [
            SimpleNamespace(
                name=name,
                value=value,
                group=base.ENTRY_POINT_GROUP,
                dist=SimpleNamespace(name=dist) if dist else None,
            )
            for name, value, dist in entries
        ]

        def entry_points(group: str):
            assert group == base.ENTRY_POINT_GROUP
            return points

        monkeypatch.setattr(base, "metadata", SimpleNamespace(entry_points=entry_points))
        base.refresh()

    return install


@pytest.fixture
def plugin_module(tmp_path, monkeypatch):
    """Write a module a test's entry point can name, importable for this test only."""
    made: list[str] = []

    def write(name: str, source: str) -> str:
        (tmp_path / f"{name}.py").write_text(source)
        made.append(name)
        return name

    monkeypatch.syspath_prepend(str(tmp_path))
    yield write
    for name in made:
        sys.modules.pop(name, None)


class FakeExecutor:
    """Stands in for a policy that carries out a phase."""

    name = "fake"
    segment_source = "policy"
    display_name = "Fake policy"
    summary = "Pretends to carry out a phase."
    requirements = ("a checkpoint",)

    def __init__(self, ctx: ExecutorContext) -> None:
        self.ctx = ctx
        self.runs: list = []

    def run(self, request, leg, *, save_root, should_stop):
        self.runs.append((request, leg, save_root))
        return HumanPhaseResult("done", n_frames=3)

    def kill(self) -> None:
        return None


def _this_tree() -> dict:
    """The environment for a subprocess that must import THIS checkout's tandem."""
    return {**os.environ, "PYTHONPATH": str(REPO / "src")}


def _ctx(tmp_path: Path, **kwargs) -> ExecutorContext:
    profile = Profile.model_validate({"name": "x"})
    return ExecutorContext(profile=profile, session_dir=tmp_path / "session", **kwargs)


# --- names ----------------------------------------------------------------------------------------


def test_teleop_ships_and_is_listed_without_being_imported():
    # Listing has to work on a laptop, so it may not import an executor to learn its name.
    script = (
        "import sys, tandem.executors as ex\n"
        "assert 'teleop' in ex.available(), ex.available()\n"
        "assert 'tandem.executors.teleop' not in sys.modules\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=_this_tree())
    assert result.returncode == 0, result.stdout + result.stderr


def test_teleop_is_described_without_being_built():
    described = base.info("teleop", settings=Settings())
    assert described.display_name == "Teleoperation"
    assert described.segment_source == "teleop"
    assert described.origin == base.BUILTIN_ORIGIN
    assert described.requirements, "a person choosing an executor has to be told what it needs"
    assert set(described.to_dict()) >= {"name", "display_name", "summary", "requirements", "ready"}


def test_an_unknown_name_is_refused_with_the_nearest_one(tmp_path):
    for attempt in (
        lambda: base.info("teleopp"),
        lambda: base.create("teleopp", _ctx(tmp_path)),
        lambda: base.check_name("teleopp"),
    ):
        with pytest.raises(TandemError, match="Unknown human executor 'teleopp'") as caught:
            attempt()
        assert caught.value.hint == "Did you mean 'teleop'?"


def test_an_unknown_name_with_nothing_close_lists_the_known_ones_and_the_entry_point():
    with pytest.raises(TandemError) as caught:
        base.info("diffusion")
    assert "teleop" in caught.value.hint
    assert base.ENTRY_POINT_GROUP in caught.value.hint


def test_a_class_registered_at_runtime_is_listed_described_built_and_removed(tmp_path):
    base.register_human_executor("fake", FakeExecutor)
    assert base.available() == ["fake", "teleop"]

    described = base.info("fake", settings=Settings())
    assert (described.display_name, described.summary) == ("Fake policy", "Pretends to carry out a phase.")
    assert described.segment_source == "policy"
    assert described.requirements == ("a checkpoint",)
    assert described.origin == base.REGISTERED_ORIGIN

    built = base.create("fake", _ctx(tmp_path))
    assert isinstance(built, FakeExecutor) and isinstance(built, HumanExecutor)

    base.unregister_human_executor("fake")
    assert base.available() == ["teleop"]
    with pytest.raises(TandemError, match="was registered in this process"):
        base.unregister_human_executor("teleop")


def test_a_taken_name_is_shadowed_only_on_purpose(tmp_path):
    with pytest.raises(TandemError, match="already exists"):
        base.register_human_executor("teleop", FakeExecutor)
    base.register_human_executor("teleop", FakeExecutor, replace=True)
    assert isinstance(base.create("teleop", _ctx(tmp_path)), FakeExecutor)


@pytest.mark.parametrize("name", ["", "1fake", "my executor", "pkg.mod:Cls", "../fake"])
def test_a_name_no_profile_could_hold_is_refused_at_registration(name):
    with pytest.raises(TandemError, match="cannot be the name of a human executor"):
        base.register_human_executor(name, FakeExecutor)


def test_a_factory_that_builds_something_else_is_refused_when_built(tmp_path):
    class NoKill:
        name = "half"
        segment_source = "policy"
        display_name = "Half"
        summary = "Cannot be stopped."

        def __init__(self, ctx):
            pass

        def run(self, *args, **kwargs):
            return HumanPhaseResult("done")

    base.register_human_executor("half", NoKill)
    with pytest.raises(TandemError, match="which has no kill"):
        base.create("half", _ctx(tmp_path))


def test_an_executor_may_not_record_its_legs_as_the_planners(tmp_path):
    # The merge treats "tamp" legs as the planner's, so an executor claiming it would be read as one.
    with pytest.raises(ValueError, match="teleop, policy"):
        ExecutorFactory(create=FakeExecutor, display_name="x", summary="x", segment_source="tamp")

    class Pretender(FakeExecutor):
        segment_source = "tamp"

    base.register_human_executor("pretender", Pretender)
    with pytest.raises(TandemError, match="does not say what its legs are"):
        base.info("pretender")


def test_an_executor_whose_legs_are_not_what_it_declared_is_refused(tmp_path):
    class Liar(FakeExecutor):
        def __init__(self, ctx):
            super().__init__(ctx)
            self.segment_source = "teleop"

    base.register_human_executor(
        "liar", ExecutorFactory(create=Liar, display_name="Liar", summary="x", segment_source="policy")
    )
    with pytest.raises(TandemError, match="records them as 'teleop'"):
        base.create("liar", _ctx(tmp_path))


def test_a_factory_that_fails_is_reported_with_its_name(tmp_path):
    def broken(ctx):
        raise RuntimeError("no checkpoint at /nowhere")

    base.register_human_executor(
        "broken", ExecutorFactory(create=broken, display_name="Broken", summary="x", segment_source="policy")
    )
    with pytest.raises(TandemError, match="Could not build the human executor 'broken'.*no checkpoint"):
        base.create("broken", _ctx(tmp_path))


# --- installed packages ---------------------------------------------------------------------------

_TOY = '''
from tandem.executors.base import ExecutorFactory, HumanPhaseResult


class ToyExecutor:
    name = "toy"
    segment_source = "policy"
    display_name = "Toy policy"
    summary = "A policy from a package installed beside tandem."

    def __init__(self, ctx):
        self.ctx = ctx

    def run(self, request, leg, *, save_root, should_stop):
        return HumanPhaseResult("ended_by_operator", n_frames=7)

    def kill(self):
        pass


FACTORY = ExecutorFactory(
    create=ToyExecutor,
    display_name=ToyExecutor.display_name,
    summary=ToyExecutor.summary,
    segment_source="policy",
    requirements=("a toy checkpoint",),
    check=lambda settings: ["no toy checkpoint on this machine"],
)
'''


def test_an_installed_package_adds_an_executor_through_the_entry_point(tmp_path, installed, plugin_module):
    module = plugin_module("toy_executor_ok", _TOY)
    installed(("toy", f"{module}:FACTORY", "tandem-toy"))

    assert base.available() == ["teleop", "toy"]
    assert module not in sys.modules, "listing must not import a plugin"

    described = base.info("toy", settings=Settings())
    assert described.origin == "installed by tandem-toy"
    assert described.unmet == ("no toy checkpoint on this machine",)
    assert not described.ready

    built = base.create("toy", _ctx(tmp_path))
    leg = LegSpec(trajectory_id="t", segment_source="policy")
    assert built.run(None, leg, save_root=tmp_path, should_stop=lambda: False).status == "ended_by_operator"

    # And a profile may name it, since this machine has it.
    assert HitlSpec(human_executor="toy").human_executor == "toy"


def test_tandems_own_entry_point_for_teleop_is_the_built_in_not_a_rival(installed):
    installed(("teleop", "tandem.executors.teleop:FACTORY", "tandem-tamp"))
    assert base.available() == ["teleop"]
    assert base.info("teleop", settings=Settings()).origin == base.BUILTIN_ORIGIN


def test_two_claims_to_one_name_are_an_error_not_a_guess(installed, plugin_module):
    module = plugin_module("toy_executor_rival", _TOY)
    installed(
        ("teleop", f"{module}:FACTORY", "tandem-rival"),
        ("toy", f"{module}:FACTORY", "tandem-toy"),
        ("toy", f"{module}:FACTORY", "tandem-toy"),  # one package seen twice on sys.path is one claim
    )
    with pytest.raises(TandemError, match="More than one package provides a human executor named 'teleop'"):
        base.info("teleop")
    assert base.info("toy", settings=Settings()).origin == "installed by tandem-toy"


def test_a_plugin_that_will_not_import_is_listed_with_the_reason(installed, plugin_module):
    module = plugin_module("toy_executor_broken", "import torch_that_is_not_installed\n")
    installed(("broken", f"{module}:FACTORY", "tandem-broken"))

    with pytest.raises(TandemError, match="could not be loaded"):
        base.info("broken")
    listed = {entry.name: entry for entry in base.catalog(settings=Settings())}
    assert set(listed) == {"broken", "teleop"}, "one broken plugin must not hide the others"
    assert "torch_that_is_not_installed" in listed["broken"].error
    assert not listed["broken"].ready
    assert listed["teleop"].error is None


def test_the_pyproject_declares_every_built_in_under_the_entry_point_group():
    # tandem registers its own in code too, so the two must not drift apart: an installed tandem
    # whose metadata named another path would be a second claim to "teleop", and an error.
    project = tomlkit.parse((REPO / "pyproject.toml").read_text())["project"]
    declared = dict(project["entry-points"][base.ENTRY_POINT_GROUP])
    assert declared == base._BUILTIN


# --- readiness ------------------------------------------------------------------------------------


def test_readiness_says_what_teleop_still_needs_on_this_machine(tmp_path):
    fresh = Settings()
    unmet = base.info("teleop", settings=fresh).unmet
    assert any("teleop is not enabled" in item for item in unmet)
    assert any("teleop.python" in item for item in unmet)
    assert any("teleop.droid_dir" in item for item in unmet)

    ready = _settings()
    assert base.info("teleop", settings=ready).ready


# --- profiles -------------------------------------------------------------------------------------


def test_a_profile_naming_an_executor_nobody_installed_is_refused_with_a_suggestion():
    with pytest.raises(ValidationError, match=r"human_executor[\s\S]*Did you mean 'teleop'\?"):
        Profile.model_validate({"name": "x", "hitl": {"human_executor": "teleopp"}})


def test_a_profile_may_name_an_executor_registered_in_this_process():
    with pytest.raises(ValidationError, match="Unknown human executor 'fake'"):
        HitlSpec(human_executor="fake")
    base.register_human_executor("fake", FakeExecutor)
    assert HitlSpec(human_executor="fake").to_planning_config().human_executor == "fake"


# --- what a leg is asked for, and answers -------------------------------------------------------


def test_a_request_is_what_the_operator_is_shown():
    from tandem.planning.structs import HumanOperator
    from tandem.planning.symbols import Atom, Parameter

    view = HumanPhase(
        description="open the box",
        instructions="Lift the lid until it stays open.",
        expected=["the box is open"],
        index=0,
        total=3,
        attempt=2,
        missing=["the box is open — the lid is still down"],
    )
    operator = HumanOperator(
        name="Open",
        args=("box",),
        parameters=(Parameter("x0", "surface"),),
        add_effects=frozenset({Atom("IsOpen", ("box",))}),
    )
    request = HumanPhaseRequest.from_view(view, operator=operator)
    assert (request.phase_index, request.n_phases, request.attempt) == (0, 3, 2)
    assert request.description == "open the box"
    assert request.expected == ["the box is open"]
    assert request.missing == ["the box is open — the lid is still down"]
    assert request.operator == operator.to_json(), "the operator travels in its JSON form"


def test_the_leg_of_a_request_carries_its_phase_even_phase_zero():
    request = HumanPhaseRequest(phase_index=0, n_phases=3, description="open the box", instructions="")
    leg = request.leg_spec(
        trajectory_id="traj-1", instruction="put the toy in the box", segment_source="teleop"
    )
    assert (leg.phase_index, leg.n_phases, leg.phase_description) == (0, 3, "open the box")
    assert leg.instruction == "put the toy in the box", "the label is the whole task, not the phase"
    assert (leg.trajectory_id, leg.segment_source) == ("traj-1", "teleop")


def test_a_request_with_no_plan_behind_it_stamps_nothing():
    request = HumanPhaseRequest(phase_index=0, n_phases=0, description="", instructions="")
    leg = request.leg_spec(trajectory_id="traj-1", instruction="x", segment_source="teleop")
    assert (leg.phase_index, leg.n_phases, leg.phase_description) == (None, None, "")


def test_a_result_is_one_of_three_endings():
    assert HumanPhaseResult("done", n_frames=4, leg_dir="/x").leg_dir == Path("/x")
    assert not HumanPhaseResult("done").recorded
    with pytest.raises(ValueError, match="done, aborted, ended_by_operator"):
        HumanPhaseResult("finished")
    with pytest.raises(ValueError, match="negative"):
        HumanPhaseResult("done", n_frames=-1)


# --- the teleop executor, against the stand-in driver ---------------------------------------------


def _settings(**teleop) -> Settings:
    cfg = Settings()
    cfg.teleop.enabled = True
    cfg.teleop.python = sys.executable
    cfg.teleop.droid_dir = str(Path(__file__).parent)
    for key, value in teleop.items():
        setattr(cfg.teleop, key, value)
    return cfg


class _Operator:
    """The session's side of a leg: its log, its UI messages, the problems it shows."""

    def __init__(self) -> None:
        self.logs: list[str] = []
        self.emitted: list[dict] = []
        self.problems: list[str] = []

    def events(self, name: str) -> list[dict]:
        return [p for p in self.emitted if p.get("type") == "teleop_event" and p.get("event") == name]

    def context(self, tmp_path: Path, settings: Settings) -> ExecutorContext:
        return _ctx(
            tmp_path,
            settings=settings,
            on_log=lambda stream, text: self.logs.append(text),
            on_emit=self.emitted.append,
            on_problem=self.problems.append,
        )


@pytest.fixture
def teleop(tmp_path, monkeypatch):
    """A TeleopExecutor whose driver is the stand-in, and the operator watching it."""
    from tandem import teleop as teleop_pkg

    monkeypatch.setattr(teleop_pkg, "driver_path", lambda: FAKE_TELEOP)
    operator = _Operator()
    built: list[teleop_mod.TeleopExecutor] = []

    def build(settings: Settings | None = None) -> tuple[teleop_mod.TeleopExecutor, _Operator]:
        executor = base.create("teleop", operator.context(tmp_path, settings or _settings()))
        built.append(executor)
        return executor, operator

    yield build
    for executor in built:
        executor.kill()


def _when(condition, timeout: float = 20.0):
    """A `should_stop` that turns true once `condition()` does, or after `timeout` as a backstop."""
    deadline = time.monotonic() + timeout

    def should_stop() -> bool:
        return condition() or time.monotonic() > deadline

    return should_stop


def _recording(executor) -> bool:
    child = executor._child
    return child is not None and child._recording


def _meta(result: HumanPhaseResult) -> dict:
    assert result.leg_dir is not None
    return json.loads((result.leg_dir / "_meta.json").read_text())


def _leg(request: HumanPhaseRequest | None = None) -> LegSpec:
    stamp = {
        "trajectory_id": "traj-abc123",
        "instruction": "put the toy in the box",
        "segment_source": "teleop",
    }
    return LegSpec(**stamp) if request is None else request.leg_spec(**stamp)


def test_the_teleop_executor_is_what_the_registry_builds(teleop):
    executor, _ = teleop()
    assert isinstance(executor, teleop_mod.TeleopExecutor)
    assert isinstance(executor, HumanExecutor)
    assert executor.segment_source == "teleop"


def test_a_teleop_leg_is_recorded_and_stamped_with_its_phase(teleop, tmp_path):
    executor, operator = teleop()
    request = HumanPhaseRequest(
        phase_index=0, n_phases=3, description="open the box", instructions="Lift it."
    )
    save_root = tmp_path / "trajectories"

    result = executor.run(
        request, _leg(request), save_root=save_root, should_stop=_when(lambda: _recording(executor))
    )

    assert result.status == "done"
    assert result.n_frames >= 2, "the leg saved no frames"
    assert result.leg_dir.parent == save_root / "eval", "legs go where merge.find_legs looks"
    meta = _meta(result)
    assert meta["trajectory_id"] == "traj-abc123"
    assert meta["segment_source"] == "teleop"
    assert meta["instruction"] == "put the toy in the box"
    assert (meta["phase_index"], meta["n_phases"], meta["phase_description"]) == (0, 3, "open the box")
    assert not operator.problems
    assert any("the teleop leg is part of this episode" in line for line in operator.logs)
    assert executor._child is None, "the executor must not keep hold of a finished driver"


def test_a_leg_lent_with_no_phase_claims_none(teleop, tmp_path):
    executor, _ = teleop()
    result = executor.run(
        None, _leg(), save_root=tmp_path / "trajectories", should_stop=_when(lambda: _recording(executor))
    )
    assert result.status == "done"
    assert not {"phase_index", "n_phases", "phase_description"} & set(_meta(result))


def test_every_recording_in_one_handoff_is_counted(teleop, tmp_path):
    # The driver starts another recording whenever one ends short of quitting, and each one is a leg of
    # this trajectory. The child keeps only the last count, so the executor adds them up itself.
    executor, operator = teleop()
    ended_first = []

    def should_stop() -> bool:
        child = executor._child
        starts = len(operator.events("rollout_start"))
        if child is not None and child._recording and starts == 1 and not ended_first:
            child._write({"cmd": "end"})
            ended_first.append(True)
            return False
        return starts >= 2 and _recording(executor)

    result = executor.run(None, _leg(), save_root=tmp_path / "trajectories", should_stop=_when(should_stop))

    saved = operator.events("rollout_saved")
    assert len(saved) == 2
    assert result.n_frames == sum(int(event["n_frames"]) for event in saved)
    assert result.leg_dir == Path(saved[-1]["dir"])


def test_consecutive_legs_do_not_replay_each_others_events(teleop, tmp_path):
    # One events file per session would be followed from its first byte by every later leg, reporting the
    # previous leg's recording as the new one's.
    executor, _ = teleop()
    save_root = tmp_path / "trajectories"
    first_request = HumanPhaseRequest(phase_index=0, n_phases=3, description="open the box", instructions="")
    second_request = HumanPhaseRequest(
        phase_index=2, n_phases=3, description="close the box", instructions=""
    )

    first = executor.run(first_request, _leg(first_request), save_root=save_root,
                         should_stop=_when(lambda: _recording(executor)))
    second = executor.run(second_request, _leg(second_request), save_root=save_root,
                          should_stop=_when(lambda: _recording(executor)))

    assert first.leg_dir != second.leg_dir
    assert second.n_frames == _meta(second)["n_frames"], "the second leg counted the first one's frames"
    assert _meta(second)["phase_index"] == 2
    assert len(list((tmp_path / "session" / "teleop").iterdir())) == 2


def test_teleop_that_is_not_configured_still_lends_the_arm(teleop, tmp_path):
    # The arm is already released and may be in someone's hands: the leg waits to be handed back, and the
    # person can do the step by hand. It is unrecorded, which is the loop's to judge.
    executor, operator = teleop(_settings(enabled=False))
    result = executor.run(None, _leg(), save_root=tmp_path / "trajectories", should_stop=lambda: True)
    assert result == HumanPhaseResult("done", n_frames=0, leg_dir=None)
    assert any("Teleop is not configured" in problem for problem in operator.problems)


def test_a_driver_that_cannot_start_is_reported_and_the_leg_still_waits(teleop, tmp_path):
    executor, operator = teleop(_settings(python=str(tmp_path / "no-such-python")))
    asked = []

    def should_stop() -> bool:
        asked.append(True)
        return len(asked) >= 3

    result = executor.run(None, _leg(), save_root=tmp_path / "trajectories", should_stop=should_stop)
    assert result.status == "done" and not result.recorded
    assert len(asked) >= 3, "the leg ended before anyone handed the arm back"
    assert any("Could not start the teleop driver" in problem for problem in operator.problems)


def test_a_leg_with_no_trajectory_id_is_refused_before_anything_starts(teleop, tmp_path):
    executor, operator = teleop()
    with pytest.raises(TandemError, match="needs the trajectory id"):
        executor.run(None, LegSpec(trajectory_id=""), save_root=tmp_path, should_stop=lambda: True)
    assert not operator.emitted


def test_a_leg_stamped_for_another_phase_is_refused(teleop, tmp_path):
    executor, _ = teleop()
    request = HumanPhaseRequest(phase_index=1, n_phases=3, description="fold it", instructions="")
    wrong = HumanPhaseRequest(phase_index=0, n_phases=3, description="open it", instructions="")
    with pytest.raises(TandemError, match="stamped as phase 0 of 3, but the request is for phase 1 of 3"):
        executor.run(request, _leg(wrong), save_root=tmp_path, should_stop=lambda: True)


def test_a_leg_built_for_the_planner_is_refused(teleop, tmp_path):
    # LegSpec's default segment_source is the planner's "tamp"; a caller that forgot to set it has built
    # the leg for something else.
    executor, _ = teleop()
    with pytest.raises(TandemError, match="asked to record as 'tamp'"):
        executor.run(None, LegSpec(trajectory_id="traj-1"), save_root=tmp_path, should_stop=lambda: True)


def test_whatever_unwinds_the_wait_still_ends_the_driver(teleop, tmp_path):
    # The caller reaches for the robot and the cameras next, and a driver left running still holds them.
    executor, _ = teleop()
    seen = []

    def should_stop() -> bool:
        child = executor._child
        if child is not None and child._recording:
            seen.append(child)
            raise RuntimeError("the operator's connection dropped")
        return False

    with pytest.raises(RuntimeError, match="connection dropped"):
        executor.run(None, _leg(), save_root=tmp_path / "trajectories", should_stop=_when(should_stop))
    assert seen and seen[0].proc.poll() is not None, "the driver outlived the leg"
    assert executor._child is None


def test_kill_cuts_a_leg_off_as_aborted(teleop, tmp_path):
    executor, _ = teleop()
    results: list[HumanPhaseResult] = []
    runner = threading.Thread(
        target=lambda: results.append(
            executor.run(None, _leg(), save_root=tmp_path / "trajectories", should_stop=_when(lambda: False))
        ),
        daemon=True,
    )
    runner.start()
    deadline = time.monotonic() + 20.0
    while not _recording(executor) and time.monotonic() < deadline:
        time.sleep(0.02)
    child = executor._child
    assert child is not None and child._recording

    executor.kill()
    runner.join(timeout=20.0)
    assert not runner.is_alive(), "a killed leg did not return"
    assert results[0].status == "aborted"
    assert child.proc.poll() is not None, "the driver is still running"


def test_a_driver_that_will_not_exit_is_a_custody_error(teleop, tmp_path, monkeypatch):
    killed = []

    class Stuck:
        def __init__(self, host, cfg, **stamp):
            self.proc = SimpleNamespace(poll=lambda: None)

        def start(self):
            return self

        def finish(self):
            pass

        def wait(self, timeout):
            return False

        def kill(self):
            killed.append(True)

    monkeypatch.setattr(teleop_mod, "TeleopChild", Stuck)
    executor, _ = teleop()
    with pytest.raises(CustodyError, match="still holds the robot and cameras"):
        executor.run(None, _leg(), save_root=tmp_path, should_stop=lambda: True)
    assert killed, "it was never killed before giving up"
    assert executor._child is None


# --- the base install -----------------------------------------------------------------------------


def test_the_executor_package_imports_nothing_heavy():
    # test_packaging's list, plus DROID: the teleop executor launches the driver but never imports it.
    forbidden = (
        "torch", "cv2", "pyzed", "open3d", "curobo", "cutamp", "tiptop", "warp", "rerun_sdk", "droid",
    )
    script = f"""
import builtins
forbidden = {forbidden!r}
real_import = builtins.__import__

def guard(name, *args, **kwargs):
    if name.split(".")[0] in forbidden:
        raise AssertionError(f"imported a heavy dependency: {{name}}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guard
import tandem.executors, tandem.executors.base, tandem.executors.teleop
print("ok")
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=_this_tree())
    assert result.returncode == 0, result.stdout + result.stderr
