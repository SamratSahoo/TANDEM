"""The planner registry: how a planner tandem has never heard of becomes one a profile can name.

Three promises are tested here. A planner registered by name -- in code, or by an installed package's
``tandem.planners`` entry point -- is accepted by a profile and built by a session through exactly the
path TiPToP takes, with nothing TiPToP-specific on it. A plugin that is broken is reported against its
own name and breaks nothing else. And TiPToP itself is now just one factory: everything the session
used to set up for it inline, it sets up for itself.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from fake_backend import FakeBackend
from helpers import FakeFactory, isolate_registry, wait_for

from tandem.core import paths, profiles, secrets
from tandem.core import settings as settings_mod
from tandem.core.errors import RuntimeNotReady, TandemError
from tandem.core.session import Session, State
from tandem.planners import registry
from tandem.planners.base import (
    BackendContext,
    BackendFactory,
    BackendRuntime,
    RuntimeStatus,
    SourcePin,
)
from tandem.planners.tiptop import FACTORY as TIPTOP
from tandem.planners.tiptop import runtime as runtime_mod
from tandem.planners.tiptop.backend import TiptopBackend
from tandem.planners.tiptop.capabilities import CAPABILITIES
from tandem.planners.tiptop.factory import SOURCES

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    isolate_registry(monkeypatch)


def _declare(monkeypatch, *targets: tuple[str, str]) -> None:
    """Pretend the installed packages declare these (name, "module:attribute") entry points."""
    declared = [importlib.metadata.EntryPoint(name, value, registry.GROUP) for name, value in targets]
    real = importlib.metadata.entry_points

    def entry_points(**params):
        if params.get("group") == registry.GROUP:
            return list(declared)
        return real(**params)

    monkeypatch.setattr(importlib.metadata, "entry_points", entry_points)


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    """Write an importable plugin module, and forget it again afterwards."""
    monkeypatch.syspath_prepend(str(tmp_path))
    written: list[str] = []

    def write(module: str, body: str) -> str:
        (tmp_path / f"{module}.py").write_text(textwrap.dedent(body))
        written.append(module)
        return module

    yield write
    for module in written:
        sys.modules.pop(module, None)


def _entries(name: str) -> list[registry.CatalogEntry]:
    return [entry for entry in registry.catalog() if entry.name == name]


# --- a planner registered in code -----------------------------------------------------------------


def test_a_registered_planner_is_one_a_profile_can_name():
    registry.register_backend("toy", FakeFactory("toy"))

    spec = profiles.PlannerSpec(backend="toy", options={"bins": ["red", "blue"]})
    assert spec.backend == "toy" and spec.options == {"bins": ["red", "blue"]}

    # A typo is still an error with the nearest name in it, never a fallback to the default planner.
    with pytest.raises(ValueError, match="did you mean 'toy'"):
        profiles.PlannerSpec(backend="toyy")

    registry.unregister_backend("toy")
    # Not a typo but a planner this machine does not have: said as that, not as "must be one of".
    with pytest.raises(ValueError, match="no planner named 'toy' is installed on this machine"):
        profiles.PlannerSpec(backend="toy")


def test_options_survive_a_profile_round_trip(profile):
    registry.register_backend("toy", FakeFactory("toy"))
    profile.planner = profiles.PlannerSpec(backend="toy", options={"bins": ["red", "blue"], "speed": 0.5})
    profiles.save(profile)
    loaded = profiles.load(profile.name)
    assert loaded.planner.backend == "toy"
    assert loaded.planner.options == {"bins": ["red", "blue"], "speed": 0.5}
    # A profile that says nothing about options has the planner's defaults, rather than failing to load.
    assert profiles.PlannerSpec().options == TIPTOP.validate_options({})


def test_the_session_builds_a_registered_planner_from_one_context(profile, tmp_path, monkeypatch):
    """The whole point of the factory seam: the session has no idea which planner it is driving.

    Nor which runtime: it holds none. TiPToP's is never built on this machine, and a session driving a
    different planner does not look at it.
    """
    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
    factory = FakeFactory("toy")
    registry.register_backend("toy", factory)
    profile.planner = profiles.PlannerSpec(backend="toy", options={"bins": ["red"]})

    session = Session(profile, task="put the red block in the red bin")
    session.start()
    try:
        assert wait_for(lambda: session.state is State.AWAITING_TASK), f"stuck in {session.state}"
        (ctx,) = factory.contexts
        # The session's own profile: the one it was given, pinned to where it started (`Profile.pinned`).
        assert ctx.profile is session.profile and ctx.profile.model_dump() == profile.model_dump()
        assert ctx.options == {"bins": ["red"]}
        assert ctx.output_dir == profile.trajectories_dir()
        assert ctx.session_id == session.id
        assert ctx.task == "put the red block in the red bin"
        assert ctx.execute is True and ctx.record == session.record
        assert ctx.runtime_dir is None, "the factory finds its own runtime, as every command does"
        assert ctx.settings is not None
        assert ctx.session_dir.is_dir()
        assert ctx.events_file == ctx.session_dir / "events.jsonl" and ctx.events_file.is_file()

        # The backend's log is the session's log, which is what an operator reads.
        ctx.log("toy planner says hello")
        assert any(line["text"] == "toy planner says hello" for line in session.logs())

        # Nothing of TiPToP's was set up for a planner that is not TiPToP.
        assert not (paths.session_scratch_dir() / profile.name / "tiptop.yml").exists()
        assert not (ctx.session_dir / "curobo-overrides.json").exists()

        (backend,) = factory.built
        assert backend.warmed
        session.next_task()
        assert wait_for(lambda: session.state is State.AWAITING_LABEL), f"stuck in {session.state}"
        session.label(True)
        assert wait_for(lambda: session.labeled_count == 1)
    finally:
        session.stop(park=False)
        session.wait(timeout=5)
    assert factory.built[0].closed


def test_a_planner_that_is_not_ready_is_closed_and_the_session_never_starts(profile, tmp_path, monkeypatch):
    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")

    class NotBuilt(FakeBackend):
        def require_ready(self) -> None:
            raise RuntimeNotReady("the toy planner's runtime is missing", hint="install it")

    factory = FakeFactory("toy", backend_type=NotBuilt)
    registry.register_backend("toy", factory)
    profile.planner = profiles.PlannerSpec(backend="toy")

    session = Session(profile, task="x")
    with pytest.raises(RuntimeNotReady, match="toy planner's runtime is missing"):
        session.start()
    assert session.state is State.SPAWNING
    assert factory.built[0].closed, "a backend that was built must be given back even if it never ran"


def test_a_factory_that_builds_something_that_is_not_a_backend_is_refused(tmp_path):
    class Hollow(FakeFactory):
        def create(self, ctx):
            return object()

    registry.register_backend("toy", Hollow("toy"))
    ctx = BackendContext(profile=None, session_dir=tmp_path, output_dir=tmp_path)
    with pytest.raises(TandemError, match="not a planner backend: it has no name, capabilities"):
        registry.create("toy", ctx)


# --- registering, loudly ---------------------------------------------------------------------------


def test_a_name_that_is_taken_is_not_quietly_replaced():
    with pytest.raises(TandemError, match="already registered"):
        registry.register_backend("tiptop", FakeFactory("tiptop"))
    assert registry.factory("tiptop") is TIPTOP

    stand_in = FakeFactory("tiptop")
    registry.register_backend("tiptop", stand_in, replace=True)
    assert registry.factory("tiptop") is stand_in
    registry.unregister_backend("tiptop")
    assert registry.factory("tiptop") is TIPTOP, "unregistering a replacement brings the built-in back"

    # The same factory twice is not a conflict: a plugin module imported twice registers twice.
    toy = FakeFactory("toy")
    registry.register_backend("toy", toy)
    registry.register_backend("toy", toy)


@pytest.mark.parametrize("name", ["", "Toy", "toy planner", "2toy", "toy/planner"])
def test_a_name_a_profile_could_not_hold_is_refused(name):
    with pytest.raises(TandemError, match="not a usable planner name"):
        registry.register_backend(name, FakeFactory("toy"))


def test_an_import_path_is_resolved_only_when_the_planner_is_used(plugin):
    module = plugin(
        "toy_lazy_plugin",
        """
        from helpers import FakeFactory
        FACTORY = FakeFactory("toy")
        """,
    )
    registry.register_backend("toy", f"{module}:FACTORY")
    assert "toy" in registry.available()
    assert module not in sys.modules
    assert registry.info("toy").name == "toy"
    assert module in sys.modules

    with pytest.raises(TandemError, match="'module:attribute'"):
        registry.register_backend("other", "toy_lazy_plugin.FACTORY")


def test_a_factory_must_call_itself_by_the_name_it_is_registered_under():
    registry.register_backend("toy", FakeFactory("other"))
    with pytest.raises(TandemError, match="describes itself as 'other'"):
        registry.factory("toy")


def test_an_unknown_planner_is_named_with_its_nearest_neighbour():
    registry.register_backend("toy", FakeFactory("toy"))
    with pytest.raises(TandemError) as excinfo:
        registry.factory("toyy")
    assert excinfo.value.message == "Unknown planner backend 'toyy'."
    # The nearest name first, then the listing that shows every planner, broken ones included.
    assert excinfo.value.hint.startswith("Did you mean 'toy'?")
    assert "`tandem planners list`" in excinfo.value.hint


def test_a_pure_python_planner_has_no_runtime_to_install():
    registry.register_backend("toy", FakeFactory("toy"))
    assert registry.runtime("toy") is None


# --- a planner installed as a package ---------------------------------------------------------------


def test_an_installed_plugin_is_listed_by_name_without_being_imported(plugin, monkeypatch):
    module = plugin(
        "toy_planner_plugin",
        """
        from helpers import FakeFactory
        FACTORY = FakeFactory("toy")
        """,
    )
    other = plugin(
        "other_planner_plugin",
        """
        from helpers import FakeFactory
        FACTORY = FakeFactory("other")
        """,
    )
    _declare(monkeypatch, ("toy", f"{module}:FACTORY"), ("other", f"{other}:FACTORY"))

    # Listing planners imports none of them.
    assert registry.available() == ["other", "tiptop", "toy"]
    assert module not in sys.modules and other not in sys.modules
    # A profile naming one imports that one -- its options are its to check -- and never the rest:
    # loading a profile must not cost an import of every plugin installed.
    assert profiles.PlannerSpec(backend="toy").backend == "toy"
    assert module in sys.modules and other not in sys.modules

    loaded = registry.factory("toy")
    assert registry.factory("toy") is loaded
    (entry,) = _entries("toy")
    assert entry.ok and entry.origin.startswith("entry point") and entry.info.name == "toy"


def test_a_broken_plugin_is_reported_in_the_listing_and_breaks_nothing_else(plugin, monkeypatch):
    good = plugin(
        "toy_good_plugin",
        """
        from helpers import FakeFactory
        FACTORY = FakeFactory("toy")
        """,
    )
    _declare(monkeypatch, ("broken", "tandem_no_such_plugin:FACTORY"), ("toy", f"{good}:FACTORY"))

    entries = registry.catalog()  # must not raise
    by_name = {entry.name: entry for entry in entries}
    assert set(by_name) == {"broken", "tiptop", "toy"}
    assert not by_name["broken"].ok
    assert "tandem_no_such_plugin" in by_name["broken"].error
    assert by_name["broken"].origin.startswith("entry point")
    assert by_name["toy"].ok
    assert by_name["tiptop"].ok and by_name["tiptop"].info.title == "TiPToP"
    json.dumps([entry.to_dict() for entry in entries])

    # Everything else keeps working, including a profile that names the broken one -- so it can be
    # loaded and fixed -- while a session that asks for it is told why it cannot have it.
    assert registry.capabilities("tiptop") is CAPABILITIES
    assert profiles.PlannerSpec(backend="broken").backend == "broken"
    with pytest.raises(TandemError, match="could not be loaded: ModuleNotFoundError") as excinfo:
        registry.factory("broken")
    assert "entry point" in excinfo.value.hint


def test_a_plugin_that_is_not_a_factory_is_reported_with_what_it_lacks(plugin, monkeypatch):
    module = plugin(
        "toy_half_plugin",
        """
        from helpers import FakeFactory

        class Half:
            info = FakeFactory("toy").info

            def capabilities(self):
                return None

        FACTORY = Half()
        """,
    )
    _declare(monkeypatch, ("toy", f"{module}:FACTORY"))
    (entry,) = _entries("toy")
    assert not entry.ok
    assert "not a planner factory: it has no create, runtime" in entry.error


def test_a_plugin_may_register_its_factory_class(plugin, monkeypatch):
    module = plugin(
        "toy_class_plugin",
        """
        from helpers import FakeFactory

        class ToyFactory(FakeFactory):
            def __init__(self):
                super().__init__("toy")
        """,
    )
    _declare(monkeypatch, ("toy", f"{module}:ToyFactory"))
    first = registry.factory("toy")
    assert isinstance(first, BackendFactory)
    assert registry.factory("toy") is first, "one instance, so a factory may keep state"


def test_tandems_own_entry_point_is_not_a_conflict_but_a_rival_plugin_is(plugin, monkeypatch):
    _declare(monkeypatch, ("tiptop", "tandem.planners.tiptop:FACTORY"))
    (entry,) = _entries("tiptop")
    assert entry.ok and entry.origin == "built-in"

    rival = plugin(
        "toy_rival_plugin",
        """
        from helpers import FakeFactory
        FACTORY = FakeFactory("tiptop")
        """,
    )
    _declare(monkeypatch, ("tiptop", f"{rival}:FACTORY"))
    assert registry.factory("tiptop") is TIPTOP, "an installed package does not get to swap a built-in"
    builtin, shadowed = _entries("tiptop")
    assert builtin.ok and builtin.origin == "built-in"
    assert not shadowed.ok and "takes precedence" in shadowed.error


def test_two_packages_claiming_one_name_is_an_error_not_a_guess(monkeypatch):
    _declare(monkeypatch, ("toy", "toy_one:FACTORY"), ("toy", "toy_two:FACTORY"))
    with pytest.raises(TandemError, match="More than one installed package registers a planner named 'toy'"):
        registry.factory("toy")
    (entry,) = _entries("toy")
    assert not entry.ok and "toy_one" in entry.error and "toy_two" in entry.error


def test_a_plugin_name_no_profile_could_use_is_reported_not_offered(monkeypatch):
    _declare(monkeypatch, ("Toy Planner", "toy_mod:FACTORY"))
    assert "Toy Planner" not in registry.available()
    (entry,) = _entries("Toy Planner")
    assert not entry.ok and "not a usable planner name" in entry.error


def test_unreadable_package_metadata_does_not_hide_the_built_in_planners(monkeypatch):
    def entry_points(**params):
        raise OSError("a dist-info directory is unreadable")

    monkeypatch.setattr(importlib.metadata, "entry_points", entry_points)
    assert registry.available() == ["tiptop"]
    assert registry.capabilities("tiptop") is CAPABILITIES
    by_name = {entry.name: entry for entry in registry.catalog()}
    assert by_name["tiptop"].ok
    assert "unreadable" in by_name[registry.GROUP].error


def test_listing_the_catalog_imports_no_planner():
    """The catalog runs on a laptop. Reading what TiPToP is must not import what TiPToP runs on."""
    script = textwrap.dedent(
        """
        import builtins
        forbidden = ("torch", "cv2", "pyzed", "open3d", "curobo", "cutamp", "tiptop", "warp")
        real_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name.split(".")[0] in forbidden:
                raise AssertionError(f"listing planners imported {name}")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = guard
        from tandem.planners import registry
        (tiptop,) = [entry for entry in registry.catalog() if entry.name == "tiptop"]
        assert tiptop.ok, tiptop.error
        registry.info("tiptop"); registry.capabilities("tiptop")
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout


# --- TiPToP, as one factory among any ---------------------------------------------------------------


def test_tiptop_describes_itself_for_a_catalog():
    assert isinstance(TIPTOP, BackendFactory)
    info = registry.info("tiptop")
    assert info is TIPTOP.info and info.name == "tiptop" and info.title == "TiPToP"
    assert info.summary and info.homepage.startswith("https://") and info.requires
    assert [pin.name for pin in info.sources] == ["tiptop", "cuTAMP", "curobo"]
    assert all(len(pin.commit) == 40 for pin in info.sources)
    assert registry.capabilities("tiptop") is CAPABILITIES
    assert json.loads(json.dumps(info.to_dict()))["sources"][0]["name"] == "tiptop"


def test_tiptops_catalog_pins_are_the_commits_its_install_delivers():
    """An install fetches what TiPToP's runtime recipe pins, so that is what the catalog must say."""
    from tandem.planners.tiptop.recipe import RECIPE

    assert SOURCES == RECIPE.pins
    assert TIPTOP.runtime().recipe is RECIPE


@pytest.fixture
def gemini_key(monkeypatch):
    """TiPToP's perception calls Gemini, so its factory will not build a backend without a key."""
    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")


def _context(profile, tmp_path, **overrides) -> BackendContext:
    session_dir = paths.session_scratch_dir() / profile.name / "s1"
    session_dir.mkdir(parents=True, exist_ok=True)
    events = session_dir / "events.jsonl"
    events.touch()
    fields = dict(
        profile=profile,
        session_dir=session_dir,
        output_dir=profile.trajectories_dir(),
        execute=False,
        record=True,
        session_id="s1",
        task="stack the cups",
        events_file=events,
        runtime_dir=tmp_path / "runtime",
        options=dict(profile.planner.options),
    )
    fields.update(overrides)
    return BackendContext(**fields)


def test_the_tiptop_factory_builds_what_the_session_used_to_build_inline(profile, tmp_path, gemini_key):
    # The session's events file, wherever the session keeps it: the sidecar appends to the same one.
    # TiPToP's settings are what the session hands the factory: the profile's planner.options.
    ctx = _context(
        profile,
        tmp_path,
        events_file=tmp_path / "the-sessions-events.jsonl",
        options={**profile.planner.options, "tamp": {"num_particles": 256}},
    )

    backend = registry.create("tiptop", ctx)

    assert isinstance(backend, TiptopBackend)
    assert backend._channel is None, "built, not warmed: nothing is spawned until the session warms it"
    assert backend._runtime.root == tmp_path / "runtime"
    assert backend._output_dir == profile.trajectories_dir()
    assert backend._execute is False and backend._record is True
    assert backend._env["TIPTOP_EVENTS_FILE"] == str(ctx.events_file)
    assert backend._env["TIPTOP_TASK"] == "stack the cups"
    assert backend._env["TIPTOP_CALIBRATION"] == str(profile.calibration_file())
    assert Path(backend._env["TIPTOP_CONFIG"]).is_file(), "tiptop.yml is written where $TIPTOP_CONFIG points"
    assert backend._cost_overrides_file == ctx.session_dir / "curobo-overrides.json"
    assert json.loads(backend._cost_overrides_file.read_text()) == {"num_particles": 256}


def test_tiptop_is_handed_no_overrides_when_the_profile_sets_none(profile, tmp_path, gemini_key):
    backend = TIPTOP.create(_context(profile, tmp_path, options={**profile.planner.options, "tamp": {}}))
    assert backend._cost_overrides_file is None


def test_tiptop_refuses_options_it_does_not_read(profile, tmp_path):
    with pytest.raises(TandemError, match=r"planner.options.speed: Extra inputs are not permitted"):
        TIPTOP.create(_context(profile, tmp_path, options={"speed": 2}))


def test_tiptop_refuses_a_camera_with_no_extrinsics_before_writing_anything(profile, tmp_path, gemini_key):
    profile.calibration_file().write_text("{}\n")
    with pytest.raises(TandemError, match="no camera extrinsics"):
        TIPTOP.create(_context(profile, tmp_path))
    assert not (paths.session_scratch_dir() / profile.name / "tiptop.yml").exists()


def test_tiptops_asset_warnings_reach_the_operator(profile, tmp_path, gemini_key):
    logs: list[tuple[str, str]] = []
    TIPTOP.create(
        _context(
            profile,
            tmp_path,
            options={**profile.planner.options, "tamp": {"blend_ops": ["Pick"]}},
            on_log=lambda stream, text: logs.append((stream, text)),
        )
    )
    assert any(
        stream == "tandem" and text.startswith("warning: blend_ops is set but blend_trajectory is not true")
        for stream, text in logs
    )


def test_tiptop_finds_its_runtime_in_the_settings_when_the_caller_names_none(
    profile, tmp_path, monkeypatch, gemini_key
):
    monkeypatch.delenv("TANDEM_RUNTIME_DIR", raising=False)
    settings = settings_mod.Settings(runtime_dir=str(tmp_path / "elsewhere"))
    backend = TIPTOP.create(_context(profile, tmp_path, runtime_dir=None, settings=settings))
    assert backend._runtime.root == (tmp_path / "elsewhere").resolve()
    assert TIPTOP.runtime(settings).status().path == str((tmp_path / "elsewhere").resolve())


# --- TiPToP's runtime, as a catalog installs it -----------------------------------------------------


def _tiptop_runtime(tmp_path, monkeypatch):
    monkeypatch.delenv("TANDEM_RUNTIME_DIR", raising=False)
    return TIPTOP.runtime(settings_mod.Settings(runtime_dir=str(tmp_path / "rt")))


def test_an_absent_tiptop_runtime_says_so(tmp_path, monkeypatch):
    rt = _tiptop_runtime(tmp_path, monkeypatch)
    assert isinstance(rt, BackendRuntime)
    status = rt.status()
    assert not status.installed
    assert status.detail == "not created"
    assert any("has not been created" in problem for problem in status.problems)
    assert status.mismatched(SOURCES) == ("tiptop", "cuTAMP", "curobo")
    json.dumps(status.to_dict())


def test_a_half_built_tiptop_runtime_reports_its_pins_and_what_is_missing(tmp_path, monkeypatch):
    rt = _tiptop_runtime(tmp_path, monkeypatch)
    root = rt.root
    (root / "tiptop").mkdir(parents=True)
    (root / "tiptop" / "pixi.toml").write_text("")
    (root / "cuTAMP" / "cutamp").mkdir(parents=True)
    (root / "cuTAMP" / "cutamp" / "__init__.py").write_text("")
    (root / "curobo" / "src" / "curobo").mkdir(parents=True)
    vendor = {pin.name: {"url": pin.url, "commit": pin.commit} for pin in SOURCES}
    vendor["checkpoints"] = {"files": ["vae/checkpoints/vae_full_v2.pt"]}
    (root / runtime_mod.STAMP_FILE).write_text(json.dumps({"vendor": vendor, "built_at": None}))

    status = rt.status()
    assert not status.installed
    assert status.detail == "sources present · pixi env not built · cuRobo kernels not compiled"
    assert status.pins == SOURCES
    assert status.mismatched(SOURCES) == ()

    vendor["tiptop"]["commit"] = "1c6daf3" + "0" * 33
    (root / runtime_mod.STAMP_FILE).write_text(json.dumps({"vendor": vendor, "built_at": None}))
    assert rt.status().mismatched(SOURCES) == ("tiptop",)


def test_installing_tiptop_fetches_then_builds_in_order(tmp_path, monkeypatch):
    from tandem.planners import runtime as recipe_runtime

    calls: list[tuple] = []
    ready = {"value": True}
    cls = recipe_runtime.RecipeRuntime

    def fetch(self, *, sources_dir=None, force=False, log=None):
        calls.append(("fetch", sources_dir, force))
        log("tiptop: fetching")
        return []

    monkeypatch.setattr(recipe_runtime, "_find_pixi", lambda: Path("/opt/pixi"))
    monkeypatch.setattr(cls, "fetch", fetch)
    monkeypatch.setattr(cls, "place_assets", lambda self, *, log=None: calls.append(("assets",)))
    monkeypatch.setattr(cls, "build_environment", lambda self, *, log=None, extra_env=None: calls.append(("env",)))
    monkeypatch.setattr(cls, "run_step", lambda self, step, *, log=None, extra_env=None: calls.append(("step", step.task)))
    monkeypatch.setattr(cls, "record_built", lambda self: calls.append(("built",)))
    monkeypatch.setattr(
        cls,
        "inspect",
        lambda self: recipe_runtime.RecipeStatus(
            root=self.root, exists=True, problems=() if ready["value"] else ("x is missing",)
        ),
    )

    rt = _tiptop_runtime(tmp_path, monkeypatch)
    lines: list[str] = []
    rt.install(on_progress=lines.append, sources_dir=tmp_path / "bundle", force=True)
    assert calls == [
        ("fetch", tmp_path / "bundle", True),
        ("assets",),
        ("env",),
        ("step", "setup-planners"),
        ("built",),
    ]
    assert "tiptop: fetching" in lines

    calls.clear()
    rt.install()
    assert calls[0] == ("fetch", None, False), "fetched from the pins, unless a sources directory is named"

    ready["value"] = False
    with pytest.raises(TandemError, match="still looks incomplete: x is missing"):
        rt.install()


def test_uninstalling_tiptop_refuses_a_directory_that_is_not_its_runtime(tmp_path, monkeypatch):
    rt = _tiptop_runtime(tmp_path, monkeypatch)
    rt.uninstall()  # nothing there: nothing to do

    rt.root.mkdir(parents=True)
    precious = rt.root / "thesis.tex"
    precious.write_text("four years of work")
    with pytest.raises(TandemError, match="does not look like a TiPToP runtime"):
        rt.uninstall()
    assert precious.is_file()

    precious.unlink()
    (rt.root / runtime_mod.STAMP_FILE).write_text("{}")
    (rt.root / "tiptop").mkdir()
    rt.uninstall()
    assert not rt.root.exists()


# --- the status type ---------------------------------------------------------------------------------


def test_a_runtime_that_does_not_say_what_it_was_built_from_matches_nothing():
    wanted = (SourcePin("a", "u", "1" * 40), SourcePin("b", "u", "2" * 40))
    assert RuntimeStatus().mismatched(wanted) == ("a", "b")
    assert RuntimeStatus(pins=wanted).mismatched(wanted) == ()
    assert RuntimeStatus(pins=wanted[:1]).mismatched(wanted) == ("b",)
    assert wanted[0].short() == "1111111"
