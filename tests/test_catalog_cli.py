"""The planner and executor catalogs from the command line: what is there, what is installed, what is used.

Four promises are tested here, each through the real commands:

- **The listing tells the truth, all of it.** Every planner is listed -- built in, registered, installed
  as a package, and one that will not load, with why -- and for each, whether this machine has it
  (installed, not installed, outdated, or needing nothing) is kept apart from whether the profile uses
  it. Outdated is its own state: a runtime built from commits its planner no longer pins.
- **Choosing is never blocked by installing.** `use` switches a profile (and, with `--default`, every
  new one) to a planner this machine has not built, and says so; it refuses only a name nothing
  provides, or a plugin that will not load.
- **Installing is idempotent and asks first.** Nothing is rebuilt that is built; pixi is never put into
  the home directory without a yes.
- **One name rule, and loud unknowns.** Planners and executors accept the same lowercase names, and an
  unknown one names the nearest and the command that lists them all.

Nothing here needs a GPU, pixi, git or the network: runtimes are stand-ins whose state a test sets.
"""

from __future__ import annotations

import importlib.metadata
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from helpers import FakeFactory, isolate_registry
from typer.testing import CliRunner

from tandem.cli import planners as planners_cli
from tandem.cli.app import app
from tandem.core import names, profiles
from tandem.core import settings as settings_mod
from tandem.core.errors import TandemError
from tandem.executors import base as executors
from tandem.planners import registry
from tandem.planners.base import RuntimeStatus, SourcePin

PINS = (SourcePin("solver", "https://example.com/solver.git", "a" * 40),)
OLD = (SourcePin("solver", "https://example.com/solver.git", "b" * 40),)


class StubRuntime:
    """A planner's runtime whose state a test sets, and which records what it was asked to do."""

    def __init__(self, root: Path, *, installed: bool = False, pins=(), problems=("not built yet",)) -> None:
        self.root = root
        self.calls: list[tuple] = []
        self._status = RuntimeStatus(
            installed=installed,
            path=str(root),
            pins=tuple(pins),
            detail="built" if installed else "sources missing",
            problems=() if installed else tuple(problems),
        )

    def status(self) -> RuntimeStatus:
        return self._status

    def install(self, *, on_progress=None, sources_dir=None, force=False) -> None:
        self.calls.append(("install", sources_dir, force))
        if on_progress is not None:
            on_progress("building the stand-in")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "built").write_text("yes")
        self._status = RuntimeStatus(installed=True, path=str(self.root), pins=PINS, detail="built")

    def uninstall(self) -> None:
        self.calls.append(("uninstall",))
        shutil.rmtree(self.root, ignore_errors=True)


class RuntimeFactory(FakeFactory):
    """A planner with a runtime to install, pinned to PINS."""

    def __init__(self, name: str, runtime) -> None:
        super().__init__(name)
        self.info = replace(self.info, summary=f"The {name} planner.", sources=PINS)
        self._runtime = runtime

    def runtime(self, settings=None):
        if isinstance(self._runtime, Exception):
            raise self._runtime
        return self._runtime


@pytest.fixture(autouse=True)
def clean_registries(monkeypatch):
    isolate_registry(monkeypatch)
    monkeypatch.setattr(executors, "_registered", {})
    monkeypatch.setattr(executors, "metadata", SimpleNamespace(entry_points=lambda group: []))
    monkeypatch.setattr(executors, "_discovered", None)


@pytest.fixture
def active(profile):
    """The `test` profile, made the active one."""
    cfg = settings_mod.load()
    cfg.active_profile = profile.name
    settings_mod.save(cfg)
    return profile


def _declare_planners(monkeypatch, *targets: tuple[str, str]) -> None:
    """Pretend installed packages declare these (name, "module:attribute") planner entry points."""
    declared = [importlib.metadata.EntryPoint(name, value, registry.GROUP) for name, value in targets]
    real = importlib.metadata.entry_points

    def entry_points(**params):
        if params.get("group") == registry.GROUP:
            return list(declared)
        return real(**params)

    monkeypatch.setattr(importlib.metadata, "entry_points", entry_points)


def _run(*args: str, input: str | None = None):
    return CliRunner().invoke(app, list(args), input=input)


def _json(*args: str) -> dict:
    result = _run(*args)
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def _stubs(tmp_path) -> dict[str, StubRuntime]:
    """Four planners, one in each state a machine can be in, plus one that needs nothing."""
    runtimes = {
        "solver": StubRuntime(tmp_path / "rt-solver"),
        "ready": StubRuntime(tmp_path / "rt-ready", installed=True, pins=PINS),
        "old": StubRuntime(tmp_path / "rt-old", pins=OLD, problems=("solver is at bbbbbbb",)),
    }
    for name, runtime in runtimes.items():
        registry.register_backend(name, RuntimeFactory(name, runtime))
    registry.register_backend("pure", FakeFactory("pure"))
    return runtimes


# --- the listing ------------------------------------------------------------------------------------


def test_every_planner_is_listed_with_whether_this_machine_has_it_and_which_is_in_use(
    active, tmp_path, monkeypatch
):
    _stubs(tmp_path)
    _declare_planners(monkeypatch, ("broken", "tandem_no_such_planner:FACTORY"))

    payload = _json("planners", "list", "--json")
    rows = {row["name"]: row for row in payload["planners"]}
    assert set(rows) == {"tiptop", "solver", "ready", "old", "pure", "broken"}
    assert {name: row["status"] for name, row in rows.items()} == {
        "tiptop": "not installed",
        "solver": "not installed",
        "ready": "installed",
        "old": "outdated",
        "pure": "no runtime needed",
        "broken": "broken",
    }
    # In use and installed are two facts: the profile's planner here is not installed.
    assert [name for name, row in rows.items() if row["active"]] == ["tiptop"]
    assert [name for name, row in rows.items() if row["default"]] == ["tiptop"]
    assert payload["profile"] == active.name and payload["profile_planner"] == "tiptop"
    assert payload["default_planner"] == "tiptop"
    # Only what can be installed has the command that installs it.
    assert {name: row["install_command"] for name, row in rows.items() if row["install_command"]} == {
        "tiptop": "tandem planners install tiptop",
        "solver": "tandem planners install solver",
        "old": "tandem planners install old",
    }
    assert "tandem_no_such_planner" in rows["broken"]["error"] and not rows["broken"]["ok"]
    assert rows["broken"]["origin"].startswith("entry point")
    assert rows["solver"]["summary"] == "The solver planner." and rows["solver"]["runtime"]["path"]

    # The same, for a person: every planner, the broken one with its reason, what to run next.
    shown = _run("planners", "list")
    assert shown.exit_code == 0, shown.output
    for name in rows:
        assert name in shown.output
    assert "tandem_no_such_planner" in shown.output
    assert "●" in shown.output
    assert "tandem planners install tiptop" in shown.output


def test_outdated_is_a_runtime_built_from_other_commits_than_the_planner_pins_now(tmp_path):
    info = SimpleNamespace(sources=PINS)

    def state(runtime):
        registry.register_backend("x", RuntimeFactory("x", runtime), replace=True)
        return planners_cli.runtime_state("x", info, settings_mod.load())

    stale = state(StubRuntime(tmp_path, pins=OLD))
    assert stale["status"] == "outdated" and stale["mismatched"] == ["solver"]
    assert "solver bbbbbbb → aaaaaaa" in stale["detail"]
    # A runtime that runs, but at the old commits, is still outdated: what it would plan with is not
    # the planner tandem's sidecar is written against.
    assert state(StubRuntime(tmp_path, installed=True, pins=OLD))["status"] == "outdated"
    # Half-built at the right commits is not outdated: the same command finishes it, from where it is.
    assert state(StubRuntime(tmp_path, pins=PINS))["status"] == "not installed"
    assert state(StubRuntime(tmp_path, installed=True, pins=PINS))["status"] == "installed"

    # A runtime that cannot even be looked at is that planner's problem, reported, never raised.
    broken = state(RuntimeError("the disk is gone"))
    assert broken["status"] == "broken" and "the disk is gone" in broken["detail"]

    class Unreadable(StubRuntime):
        def status(self):
            raise OSError("permission denied")

    assert state(Unreadable(tmp_path))["status"] == "broken"


def test_a_missing_profile_is_said_next_to_the_listing_not_instead_of_it():
    payload = _json("planners", "list", "--json")
    assert payload["profile_planner"] is None and "no profile named 'default'" in payload["profile_problem"]
    assert [row["name"] for row in payload["planners"]] == ["tiptop"]
    assert not any(row["active"] for row in payload["planners"])


# --- one planner in full ----------------------------------------------------------------------------


def test_info_says_what_a_planner_is_needs_and_can_be_asked_for(active):
    from tandem.planners.tiptop.recipe import RECIPE

    payload = _json("planners", "info", "tiptop", "--json")
    assert payload["name"] == "tiptop" and payload["display_name"] == "TiPToP"
    assert payload["origin"] == "built-in" and payload["active"] and payload["default"]
    assert payload["requires"] and any("GPU" in line for line in payload["requires"])
    assert [pin["commit"] for pin in payload["sources"]] == [pin.commit for pin in RECIPE.pins]
    assert payload["status"] == "not installed"
    assert payload["install_command"] == "tandem planners install tiptop"
    assert payload["runtime"]["needed"] is True and payload["runtime"]["mismatched"] == [
        "tiptop",
        "cuTAMP",
        "curobo",
    ]

    caps = payload["capabilities"]
    signatures = {p["name"]: p["signature"] for p in caps["goal_predicates"]}
    assert signatures["On"] == "On(?obj: movable, ?surface: surface)"
    wire = {p["name"]: p["wire_name"] for p in caps["goal_predicates"]}
    assert wire == {"On": "on", "Holding": "holding", "HandEmpty": None}, "HandEmpty is the planner's own"
    assert caps["robot_operators"] == ["Pick(?obj: movable)", "Place(?obj: movable, ?surface: surface)"]
    assert caps["supports"] == {
        "movable_restriction": True,
        "return_home": True,
        "cooperative_stop": False,
        "skeleton_reuse": False,
    }
    # The planner.options TiPToP reads, each with a line: what `tandem planners info` lists.
    assert set(payload["options"]) == {"robot", "perception", "tamp"}

    shown = _run("planners", "info", "tiptop")
    assert shown.exit_code == 0, shown.output
    assert "On(?obj: movable, ?surface: surface)" in shown.output
    assert "tandem planners install tiptop" in shown.output


def test_a_planner_written_with_the_sdk_says_which_options_it_reads():
    from toy_planner import ToyPlanner

    registry.register_backend("toy", ToyPlanner)
    payload = _json("planners", "info", "toy", "--json")
    assert payload["options"] == {"items": "the items on the floor when the session starts"}
    assert payload["runtime"] == {"needed": False, "mismatched": []}
    assert payload["status"] == "no runtime needed" and payload["install_command"] is None
    assert [p["signature"] for p in payload["capabilities"]["goal_predicates"]] == [
        "InBin(?obj: item, ?bin: container)"
    ]


@pytest.mark.parametrize("command", ["info", "install", "use", "remove"])
def test_an_unknown_planner_is_named_with_its_nearest_and_the_listing(command):
    result = _run("planners", command, "tiptopp")
    assert result.exit_code == 1
    error = result.exception
    assert isinstance(error, TandemError) and error.message == "Unknown planner backend 'tiptopp'."
    assert error.hint.startswith("Did you mean 'tiptop'?")
    assert "`tandem planners list`" in error.hint


def test_a_planner_that_will_not_load_is_refused_by_name(active, monkeypatch):
    _declare_planners(monkeypatch, ("broken", "tandem_no_such_planner:FACTORY"))
    for command in ("info", "use", "install"):
        result = _run("planners", command, "broken")
        assert result.exit_code == 1, result.output
        assert "could not be loaded" in result.exception.message
    assert profiles.load(active.name).planner.backend == "tiptop"


# --- choosing one -----------------------------------------------------------------------------------


def test_use_switches_the_profile_and_says_when_the_planner_is_not_installed(active, tmp_path):
    _stubs(tmp_path)
    active.planner = profiles.PlannerSpec(backend="tiptop", options={"tamp": {"num_particles": 64}})
    profiles.save(active)

    result = _run("planners", "use", "solver")
    assert result.exit_code == 0, result.output
    switched = profiles.load(active.name)
    assert switched.planner.backend == "solver"
    # The old planner's own settings would be refused by the new one, so they go -- and it is said.
    assert switched.planner.options == {}
    assert "Removed planner.options perception, robot, tamp" in result.output
    assert "not installed" in result.output and "tandem planners install solver" in result.output
    assert settings_mod.load(force=True).default_planner == "tiptop", "--default was not asked for"

    again = _json("planners", "use", "solver", "--json")
    assert again["changed"] is False and again["previous"] == "solver"
    assert again["status"] == "not installed" and again["install_command"] == "tandem planners install solver"

    made_default = _json("planners", "use", "ready", "--default", "--json")
    assert made_default["changed"] is True and made_default["default_planner"] == "ready"
    assert made_default["status"] == "installed" and made_default["install_command"] is None
    assert settings_mod.load(force=True).default_planner == "ready"
    assert profiles.load(active.name).planner.backend == "ready"


def test_use_on_another_profile_leaves_the_active_one_alone(active, tmp_path):
    _stubs(tmp_path)
    other = active.model_copy(deep=True)
    other.name = "other"
    profiles.save(other)
    assert _run("planners", "use", "pure", "--profile", "other").exit_code == 0
    assert profiles.load("other").planner.backend == "pure"
    assert profiles.load(active.name).planner.backend == "tiptop"

    missing = _run("planners", "use", "pure", "--profile", "nope")
    assert missing.exit_code == 1 and "Profile 'nope' not found" in missing.exception.message


def test_the_default_is_what_every_new_profile_plans_with(tmp_path):
    from tandem.cli import runtime as runtime_cli

    registry.register_backend("pure", FakeFactory("pure"))
    # No profile exists yet: --default alone is what a fresh machine can be told.
    result = _run("planners", "use", "pure", "--default")
    assert result.exit_code == 0, result.output
    assert settings_mod.load(force=True).default_planner == "pure"
    assert runtime_cli.active_planner("not-made-yet") == "pure", "init builds the default's runtime"

    assert _run("profile", "create", "fresh").exit_code == 0
    assert profiles.load("fresh").planner.backend == "pure"

    # A default naming a planner this machine no longer has stops the next profile before it is
    # written, rather than leaving one every later command refuses.
    registry.unregister_backend("pure")
    refused = _run("profile", "create", "later")
    assert refused.exit_code == 1
    assert "default_planner" in refused.exception.message
    assert "tandem planners default NAME" in refused.exception.hint
    assert not profiles.exists("later")


def test_a_default_planner_must_be_a_planner_name_and_config_set_says_so_plainly():
    with pytest.raises(ValueError, match="default_planner"):
        settings_mod.Settings(default_planner="TiPToP")
    result = _run("config", "set", "default_planner", "TiPToP")
    assert result.exit_code == 1
    assert isinstance(result.exception, TandemError), "a pydantic traceback is not an answer"
    assert "not a valid default_planner" in result.exception.message
    # Refused, and left as it was -- also in this process's cached settings.
    assert settings_mod.load().default_planner == "tiptop"


# --- installing and removing ---------------------------------------------------------------------------


def test_install_builds_only_what_is_missing(active, tmp_path):
    runtimes = _stubs(tmp_path)
    sources = tmp_path / "sources"
    sources.mkdir()

    result = _run("planners", "install", "solver", "--yes", "--sources", str(sources))
    assert result.exit_code == 0, result.output
    assert runtimes["solver"].calls == [("install", sources, False)]
    assert "tandem planners use solver" in result.output, "the profile plans with another planner"

    again = _run("planners", "install", "solver")
    assert again.exit_code == 0 and "already installed" in again.output
    assert len(runtimes["solver"].calls) == 1, "an installed runtime is not rebuilt"

    forced = _run("planners", "install", "solver", "--force", "--yes")
    assert forced.exit_code == 0, forced.output
    assert runtimes["solver"].calls[-1] == ("install", None, True)

    # Outdated is rebuilt without --force: what is on disk is not what the planner pins.
    assert _run("planners", "install", "old", "--yes").exit_code == 0
    assert runtimes["old"].calls == [("install", None, False)]
    assert _run("planners", "install", "ready").exit_code == 0 and runtimes["ready"].calls == []

    pure = _run("planners", "install", "pure")
    assert pure.exit_code == 0 and "pure Python" in pure.output


def test_pixi_is_never_installed_without_a_yes(active, tmp_path, monkeypatch):
    from tandem.planners.runtime import PixiEnvironment, RecipeRuntime, RuntimeRecipe, Source

    recipe = RuntimeRecipe(
        planner="heavy",
        sources=(Source(PINS[0]),),
        environment=PixiEnvironment(manifest="solver/pixi.toml"),
    )
    registry.register_backend("heavy", RuntimeFactory("heavy", RecipeRuntime(recipe, tmp_path / "heavy")))
    pixi = {"installed": False}
    monkeypatch.setattr(
        "tandem.core.probe.find_pixi", lambda: tmp_path / "pixi" if pixi["installed"] else None
    )
    monkeypatch.setattr("tandem.cli.runtime.install_pixi", lambda log=None: pixi.update(installed=True))
    builds: list = []
    monkeypatch.setattr("tandem.cli.runtime.run_build", lambda rt, **kw: builds.append(kw))

    refused = _run("planners", "install", "heavy")
    assert refused.exit_code == 1
    assert "pixi is not installed" in refused.exception.message and "--yes" in refused.exception.hint
    assert not pixi["installed"] and builds == []

    consented = _run("planners", "install", "heavy", "--yes")
    assert consented.exit_code == 0, consented.output
    assert pixi["installed"] and builds == [{"force": False, "sources_dir": None, "planner": "heavy"}]


def test_remove_deletes_the_runtime_only_when_told_to(active, tmp_path):
    runtimes = _stubs(tmp_path)
    root = tmp_path / "rt-ready"
    root.mkdir()
    (root / "weights.bin").write_bytes(b"\0" * 1024)
    active.planner = profiles.PlannerSpec(backend="ready")
    profiles.save(active)

    declined = _run("planners", "remove", "ready", input="n\n")
    assert declined.exit_code == 1 and root.is_dir() and runtimes["ready"].calls == []
    assert f"Profile {active.name!r} plans with" in declined.output, "removing what is in use is said"

    removed = _run("planners", "remove", "ready", "--yes")
    assert removed.exit_code == 0, removed.output
    assert runtimes["ready"].calls == [("uninstall",)] and not root.exists()
    assert "tandem planners install ready" in removed.output

    nothing = _run("planners", "remove", "solver", "--yes")
    assert nothing.exit_code == 0 and "nothing to remove" in nothing.output and runtimes["solver"].calls == []
    assert "no runtime to remove" in _run("planners", "remove", "pure").output


# --- executors ----------------------------------------------------------------------------------------


class ReadyExecutor:
    """A policy that needs nothing on this machine."""

    segment_source = "policy"
    display_name = "Ready policy"
    summary = "Always ready."

    def __init__(self, ctx) -> None:
        self.ctx = ctx


def test_executors_are_listed_with_what_each_still_needs(active, monkeypatch):
    executors.register_human_executor("ready", ReadyExecutor)
    points = [
        SimpleNamespace(
            name="broken",
            value="tandem_no_such_executor:FACTORY",
            group=executors.ENTRY_POINT_GROUP,
            dist=None,
        )
    ]
    monkeypatch.setattr(executors, "metadata", SimpleNamespace(entry_points=lambda group: points))

    payload = _json("executors", "list", "--json")
    rows = {row["name"]: row for row in payload["executors"]}
    assert set(rows) == {"teleop", "ready", "broken"}
    assert rows["teleop"]["status"] == "needs setup" and rows["teleop"]["unmet"]
    assert rows["teleop"]["active"] and not rows["ready"]["active"]
    assert rows["ready"]["status"] == "ready" and rows["ready"]["display_name"] == "Ready policy"
    assert rows["broken"]["status"] == "broken" and "tandem_no_such_executor" in rows["broken"]["error"]
    assert payload["profile_executor"] == "teleop" and payload["phase_planning"] is False

    shown = _run("executors", "list")
    assert shown.exit_code == 0, shown.output
    assert "teleop is not enabled" in shown.output and "tandem_no_such_executor" in shown.output
    assert "phase planning is off" in shown.output


def test_executors_use_switches_the_profile_and_refuses_an_unknown_name(active):
    executors.register_human_executor("ready", ReadyExecutor)
    result = _json("executors", "use", "ready", "--json")
    assert result == {
        "executor": "ready",
        "display_name": "Ready policy",
        "profile": active.name,
        "previous": "teleop",
        "changed": True,
        "status": "ready",
        "unmet": [],
        "phase_planning": False,
    }
    assert profiles.load(active.name).hitl.human_executor == "ready"

    back = _run("executors", "use", "teleop")
    assert back.exit_code == 0 and "teleop is not enabled" in back.output, "chosen, and what it needs is said"
    assert profiles.load(active.name).hitl.human_executor == "teleop"

    unknown = _run("executors", "use", "teleopp")
    assert unknown.exit_code == 1
    assert unknown.exception.hint.startswith("Did you mean 'teleop'?")
    assert "`tandem executors list`" in unknown.exception.hint


# --- one name rule --------------------------------------------------------------------------------------


def test_planners_and_executors_accept_the_same_names():
    from tandem.planning import config

    assert registry._NAME is names.NAME and config.HUMAN_EXECUTOR_NAME is names.NAME
    for bad in ("ACT", "Toy", "toy\n", "1toy", "to y", "pkg.mod:Cls", ""):
        with pytest.raises(TandemError, match="not a usable planner name"):
            registry.register_backend(bad, FakeFactory("toy"))
        with pytest.raises(TandemError, match="cannot be the name of a human executor") as caught:
            executors.register_human_executor(bad, ReadyExecutor)
        assert names.RULE in caught.value.hint
    for good in ("toy", "act_v2", "my-planner"):
        registry.register_backend(good, FakeFactory(good))
        executors.register_human_executor(good, ReadyExecutor)
        registry.unregister_backend(good)
        executors.unregister_human_executor(good)


# --- where else the catalog shows ---------------------------------------------------------------------


def test_doctor_reports_every_planners_runtime_not_only_the_one_in_use(active, tmp_path, monkeypatch):
    from tandem.cli.doctor import collect_checks

    _stubs(tmp_path)
    _declare_planners(monkeypatch, ("broken", "tandem_no_such_planner:FACTORY"))
    checks = {c.name: c for c in collect_checks(profile_name=active.name, probe_hardware=False)}
    assert checks["planner runtime"].state == "warn" and checks["planner runtime"].detail.startswith("tiptop:")
    assert "tandem planners install tiptop" in checks["planner runtime"].hint
    assert "planner tiptop" not in checks, "the planner in use has its own row already"
    assert checks["planner ready"].state == "ok"
    assert checks["planner old"].state == "warn"
    assert "tandem planners install old" in checks["planner old"].hint
    assert "tandem planners remove old" in checks["planner old"].hint
    assert checks["planner solver"].state == "skip" and checks["planner pure"].state == "skip"
    assert (
        checks["planner broken"].state == "warn"
        and "tandem_no_such_planner" in checks["planner broken"].detail
    )

    # The planner in use, outdated: the row a session depends on says so, with the command.
    active.planner = profiles.PlannerSpec(backend="old")
    profiles.save(active)
    checks = {c.name: c for c in collect_checks(profile_name=active.name, probe_hardware=False)}
    assert (
        checks["planner runtime"].state == "warn"
        and "tandem planners install old" in checks["planner runtime"].hint
    )
    assert "planner old" not in checks and checks["planner tiptop"].state == "skip"


def test_runtime_status_points_at_the_planners_commands():
    result = _run("runtime", "status")
    assert result.exit_code == 0, result.output
    assert "tandem planners install tiptop" in result.output


# --- tandem init ---------------------------------------------------------------------------------------


def test_init_sets_the_machine_up_for_the_planner_asked_for(tmp_path, monkeypatch):
    from tandem.cli import init as init_cli

    runtimes = _stubs(tmp_path)
    registry.register_backend("other", RuntimeFactory("other", StubRuntime(tmp_path / "rt-other")))
    monkeypatch.setattr(init_cli, "_preflight", lambda viz_only: [])

    result = _run("init", "--yes", "--planner", "solver")
    assert result.exit_code == 0, result.output
    assert "Planner: fake solver" in result.output and "tiptop" in result.output, "the catalog is shown"
    assert runtimes["solver"].calls == [("install", None, False)], "the chosen planner's runtime is built"
    assert profiles.load("default").planner.backend == "solver"
    assert settings_mod.load(force=True).default_planner == "tiptop", "init chose for this profile only"

    # Re-run with no planner named: the profile keeps its own, whose runtime is already built.
    again = _run("init", "--yes")
    assert again.exit_code == 0, again.output
    assert "Runtime is already built" in again.output and len(runtimes["solver"].calls) == 1
    assert profiles.load("default").planner.backend == "solver"

    # Re-run naming another: the existing profile is switched, and that runtime built.
    switched = _run("init", "--yes", "--planner", "other")
    assert switched.exit_code == 0, switched.output
    assert profiles.load("default").planner.backend == "other"
    assert "now plans with fake other" in switched.output

    unknown = _run("init", "--yes", "--planner", "solverr")
    assert unknown.exit_code == 1 and unknown.exception.hint.startswith("Did you mean 'solver'?")


def test_a_laptop_init_names_the_planner_without_building_it(tmp_path):
    runtimes = _stubs(tmp_path)
    result = _run("init", "--viz-only", "--yes", "--planner", "solver")
    assert result.exit_code == 0, result.output
    assert profiles.load("default").planner.backend == "solver"
    assert runtimes["solver"].calls == []


def test_init_asks_which_planner_only_when_there_is_a_choice(tmp_path, monkeypatch):
    from tandem.cli import init as init_cli

    asked: list[str] = []

    def prompt(text, default=None):
        asked.append(default)
        return "solver"

    monkeypatch.setattr(init_cli.typer, "prompt", prompt)
    # TiPToP alone is no choice: it is shown, and taken.
    assert init_cli._choose_planner("default", None, interactive=True) == "tiptop"
    assert asked == []

    _stubs(tmp_path)
    assert init_cli._choose_planner("default", None, interactive=True) == "solver"
    assert asked == ["tiptop"], "the machine's default is what is offered"
    # Without a terminal the default is taken, as `--yes` promises.
    assert init_cli._choose_planner("default", None, interactive=False) == "tiptop"
    assert len(asked) == 1
