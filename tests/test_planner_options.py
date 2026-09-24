"""A planner's settings are the planner's: stored under planner.options, checked by the planner, shown by it.

Until profile version 2 TiPToP's settings were top-level sections of every profile (robot:,
perception:, tamp:) and core validated them against TiPToP's schema, so every profile -- whichever
planner it named -- was a TiPToP profile. These tests hold the other side of that move:

- a profile written in the older layout still loads, says so once, and is written in the new one;
- a planner that is not TiPToP has its options checked by itself, never by TiPToP's schema;
- nothing outside planners/tiptop imports TiPToP, so every one of those answers has to come
  through the registry;
- `tandem doctor` and `tandem profile show` ask the planner, and show a toy planner's rows as
  readily as TiPToP's.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path

import pytest
from helpers import isolate_registry
from pydantic import ValidationError
from ruamel.yaml import YAML
from toy_planner import ToyPlanner
from typer.testing import CliRunner

from tandem.core import probe, profiles, secrets
from tandem.core.errors import ProfileError, TandemError
from tandem.core.profiles import Profile
from tandem.planners import registry
from tandem.planners.base import OptionsSection, OptionsView
from tandem.planners.tiptop import FACTORY as TIPTOP
from tandem.planners.tiptop.options import TiptopOptions, options_of

FIXTURES = Path(__file__).parent / "fixtures" / "profiles"
SRC = Path(__file__).resolve().parents[1] / "src" / "tandem"
_yaml = YAML(typ="safe")


@pytest.fixture(autouse=True)
def _registry(monkeypatch):
    isolate_registry(monkeypatch)
    # Every notice is said once per process; each test starts from a process that has said none.
    monkeypatch.setattr(profiles, "_noticed", set())


def _write_profile(name: str, text: str) -> Path:
    directory = profiles.profiles_root() / name
    (directory / "trajectories").mkdir(parents=True, exist_ok=True)
    path = directory / "profile.yml"
    path.write_text(text)
    (directory / "calibration.json").write_text("{}\n")
    return path


def _notices(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "tandem.core.profiles"]


# --- (a) the older layout ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["v1_ef1411f.yml", "v1_68076df.yml"])
def test_a_profile_in_the_older_layout_loads_says_so_once_and_saves_in_the_new_one(
    isolated_env, caplog, fixture
):
    """Two real version-1 profiles: the template as it shipped at ef1411f (hitl and planner blocks) and
    at the first commit (neither). Each loads with its settings under planner.options, unchanged."""
    old_text = (FIXTURES / fixture).read_text()
    old = _yaml.load(old_text)
    path = _write_profile("legacy", old_text)

    with caplog.at_level(logging.WARNING, logger="tandem.core.profiles"):
        loaded = profiles.load("legacy")
        again = profiles.load("legacy")

    # One line, naming the file, what moved, and the command that rewrites it -- and only once.
    (notice,) = _notices(caplog)
    assert str(path) in notice and "tandem profile migrate legacy" in notice
    assert "robot under planner.options" in notice and "tamp under planner.options" in notice
    assert "\n" not in notice

    assert loaded.version == profiles.LAYOUT_VERSION
    assert loaded.planner.backend == "tiptop"
    options = options_of(loaded)
    assert options.robot.host == old["robot"]["host"] and options.robot.q_home == old["robot"]["q_home"]
    assert options.perception.gemini.model == old["perception"]["gemini"]["model"]
    assert options.perception.m2t2.url == old["perception"]["m2t2"]["url"]
    assert options.tamp["num_particles"] == old["tamp"]["num_particles"]
    assert options.tamp["traj_length_norm"] == "inf", "normalised by TiPToP's own rules, as before"
    # tandem's own sections stay where they were.
    assert loaded.cameras.external.serial == str(old["cameras"]["external"]["serial"])
    assert loaded.task.prompt == old["task"]["prompt"]
    assert again.model_dump() == loaded.model_dump()

    profiles.save(loaded)
    written = _yaml.load(path.read_text())
    assert not {"robot", "perception", "tamp"} & set(written), "saved in the new layout"
    assert set(written["planner"]["options"]) == {"robot", "perception", "tamp"}
    assert written["version"] == profiles.LAYOUT_VERSION

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="tandem.core.profiles"):
        profiles._noticed.clear()
        round_tripped = profiles.load("legacy")
    assert _notices(caplog) == [], "a profile in the new layout is read without a notice"
    assert round_tripped.model_dump() == loaded.model_dump()


def test_an_old_profiles_detector_setting_still_loads_and_is_said_to_change_nothing(
    isolated_env, monkeypatch
):
    """perception.gemini is a statement, not a choice: the pinned tiptop reads neither key. Every
    version-1 profile names er-1.6, so refusing it would stop them loading; it is kept, and flagged."""
    from tandem.cli.doctor import collect_checks
    from tandem.planners.tiptop import render
    from tandem.planners.tiptop.options import DETECTOR_MODEL

    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
    _write_profile("legacy", (FIXTURES / "v1_ef1411f.yml").read_text())
    loaded = profiles.load("legacy")
    assert options_of(loaded).perception.gemini.model == "gemini-robotics-er-1.6-preview"

    problems = render.check_assets(loaded)
    assert any(
        p.startswith("perception.gemini.model is 'gemini-robotics-er-1.6-preview'") and DETECTOR_MODEL in p
        for p in problems
    ), problems
    checks = collect_checks(profile_name="legacy", probe_hardware=False)
    (row,) = [c for c in checks if c.name == "perception settings"]
    assert row.state == probe.WARN and "changes nothing" in row.detail
    assert [c.state for c in checks if c.name == "tamp settings"] == [probe.OK], "not a TAMP finding"

    # The template states what runs, so it says nothing; a temperature is as dead as the model.
    template = Profile.model_validate({"name": "fresh"})
    assert not [p for p in render.check_assets(template) if p.startswith("perception.")]
    template.planner.options["perception"]["gemini"]["temperature"] = 0.2
    assert any(p.startswith("perception.gemini.temperature is set") for p in render.check_assets(template))


def test_a_bad_tamp_key_in_an_old_profile_is_still_refused_with_the_nearest_real_one(isolated_env):
    text = (FIXTURES / "v1_ef1411f.yml").read_text().replace("num_particles: 256", "num_particle: 256")
    _write_profile("typo", text)
    with pytest.raises(ProfileError) as excinfo:
        profiles.load("typo")
    message = excinfo.value.message
    assert "options.tamp: unknown TAMP setting 'num_particle'" in message
    assert "Did you mean: num_particles" in message


def test_a_setting_in_both_places_is_refused_rather_than_one_silently_winning():
    with pytest.raises(ValidationError, match="tamp is set both at the top level"):
        Profile.model_validate(
            {"name": "x", "tamp": {}, "planner": {"backend": "tiptop", "options": {"tamp": {}}}}
        )


def test_old_sections_of_a_profile_that_plans_with_another_planner_are_dropped_and_said_to_be(caplog):
    ToyPlanner_ = _toy()
    registry.register_backend("toy", ToyPlanner_)
    with caplog.at_level(logging.WARNING, logger="tandem.core.profiles"):
        profile = Profile.model_validate(
            {
                "name": "x",
                "robot": {"host": "10.0.0.9"},
                "tamp": {"num_particles": 8},
                "planner": {"backend": "toy"},
            }
        )
    assert profile.planner.options == {}, "TiPToP's settings configure nothing the toy planner reads"
    (notice,) = _notices(caplog)
    assert "robot dropped (tiptop's setting; this profile plans with toy)" in notice
    assert "tamp dropped" in notice


def test_tandem_profile_migrate_rewrites_old_profiles_and_leaves_current_ones(isolated_env):
    from tandem.cli.app import app

    _write_profile("legacy", (FIXTURES / "v1_ef1411f.yml").read_text())
    profiles.save(
        profiles.load_file(
            Path(__file__).parents[1] / "src/tandem/resources/profile_template.yml", name="fresh"
        )
    )

    result = CliRunner().invoke(app, ["profile", "migrate"])
    assert result.exit_code == 0, result.output
    assert "legacy: rewritten in the current layout" in result.output
    assert "fresh: already current" in result.output
    assert "robot" not in _yaml.load((profiles.profiles_root() / "legacy" / "profile.yml").read_text())


def test_the_shipped_template_is_in_the_current_layout():
    data = _yaml.load((SRC / "resources" / "profile_template.yml").read_text())
    assert data["version"] == profiles.LAYOUT_VERSION
    assert not {"robot", "perception", "tamp"} & set(data)
    assert data["planner"]["backend"] == "tiptop"
    TiptopOptions.model_validate(data["planner"]["options"])  # TiPToP's own schema accepts it as written


# --- (b) a planner's options are checked by that planner -------------------------------------------------


def _toy(**attributes):
    """A ToyPlanner subclass registered as "toy", with ``attributes`` overriding the class's own."""
    return type("Toy", (ToyPlanner,), {"__module__": __name__, **attributes})


def test_a_toy_planners_options_are_checked_by_the_toy_planner_not_by_tiptops_schema():
    registry.register_backend("toy", _toy())

    # "items" is nothing TiPToP reads -- its schema forbids unknown keys -- and the toy planner reads it.
    spec = profiles.PlannerSpec(backend="toy", options={"items": ["duck", "ball"]})
    assert spec.options == {"items": ["duck", "ball"]}
    with pytest.raises(ValidationError):
        TiptopOptions.model_validate({"items": ["duck"]})

    # And what TiPToP reads is refused by the toy planner, in the toy planner's words.
    with pytest.raises(ValidationError) as excinfo:
        profiles.PlannerSpec(backend="toy", options={"robot": {"host": "10.0.0.1"}})
    message = str(excinfo.value)
    assert "options: The Toy planner does not read planner.options robot" in message
    assert "Extra inputs are not permitted" not in message, "not TiPToP's pydantic schema"


def test_a_planner_that_normalises_its_options_has_them_stored_normalised(isolated_env):
    class Toy(ToyPlanner):
        @classmethod
        def validate_options(cls, options):
            checked = super().validate_options(options)
            items = checked.get("items", ["duck"])
            if not isinstance(items, list) or not all(isinstance(i, str) for i in items):
                raise ValueError("items must be a list of names")
            return {"items": sorted(items)}

    registry.register_backend("toy", Toy)
    profile = Profile.model_validate(
        {"name": "bins", "planner": {"backend": "toy", "options": {"items": ["b", "a"]}}}
    )
    assert profile.planner.options == {"items": ["a", "b"]}
    assert Profile.model_validate({"name": "bins", "planner": {"backend": "toy"}}).planner.options == {
        "items": ["duck"]
    }

    profiles.save(profile)
    assert profiles.load("bins").planner.options == {"items": ["a", "b"]}

    with pytest.raises(ValidationError, match="options: items must be a list of names"):
        Profile.model_validate({"name": "bins", "planner": {"backend": "toy", "options": {"items": [3]}}})
    # A session is built from them as the planner left them.
    from tandem.planners.base import BackendContext

    built = Toy.create(
        BackendContext(
            profile=profile, session_dir=Path("."), output_dir=Path("."), options={"items": ["z", "y"]}
        )
    )
    assert built.options == {"items": ["y", "z"]}


def test_a_factory_without_the_hook_takes_its_options_as_written():
    from helpers import FakeFactory

    registry.register_backend("plain", FakeFactory("plain"))
    assert profiles.PlannerSpec(backend="plain", options={"anything": 1}).options == {"anything": 1}
    assert registry.validate_options("plain", {"anything": 1}) == {"anything": 1}


def test_tiptops_options_are_tiptops_schema_with_its_defaults_filled_in():
    options = registry.validate_options("tiptop", {"tamp": {"traj_length_norm": float("inf")}})
    assert set(options) == {"robot", "perception", "tamp"}
    assert options["robot"]["type"] == "fr3_robotiq" and options["tamp"] == {"traj_length_norm": "inf"}
    assert registry.validate_options("tiptop", options) == options, "its own output is accepted unchanged"
    with pytest.raises(ValidationError):
        registry.validate_options("tiptop", {"speed": 2})


# --- (c) nothing outside planners/tiptop reaches into TiPToP ------------------------------------------------

# The registry's table of built-in planners names TiPToP's factory by import path, and the deprecated
# tandem.core.runtime re-exports TiPToP's runtime for scripts written before there was a registry.
ALLOWED = {"planners/registry.py", "core/runtime.py"}


def _tiptop_references(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.startswith("tandem.planners.tiptop")]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("tandem.planners.tiptop"):
                found.append(module)
            elif module == "tandem.planners":
                found += [f"tandem.planners.{a.name}" for a in node.names if a.name == "tiptop"]
        elif isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) in {
            "import_module",
            "__import__",
        }:
            found += [
                arg.value
                for arg in node.args
                if isinstance(arg, ast.Constant) and str(arg.value).startswith("tandem.planners.tiptop")
            ]
    return [f"{path.relative_to(SRC)}:{name}" for name in found]


def test_no_module_outside_planners_tiptop_imports_tiptop():
    """Every question tandem asks TiPToP goes through the registry. An import from outside the package is
    a place where tandem knows it is talking to TiPToP -- which is the thing this layout exists to end."""
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC).as_posix()
        if relative.startswith("planners/tiptop/") or relative in ALLOWED:
            continue
        offenders += _tiptop_references(path)
    assert offenders == [], "import TiPToP only inside planners/tiptop:\n  " + "\n  ".join(offenders)


def test_the_guard_sees_every_way_of_importing_tiptop(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import tandem.planners.tiptop.render\n"
        "from tandem.planners.tiptop import FACTORY\n"
        "from tandem.planners import tiptop\n"
        "import importlib\nimportlib.import_module('tandem.planners.tiptop.options')\n"
        "from tandem.planners import registry\n"
    )
    global SRC
    real, SRC = SRC, tmp_path
    try:
        names = [ref.split(":", 1)[1] for ref in _tiptop_references(sample)]
    finally:
        SRC = real
    assert names == [
        "tandem.planners.tiptop.render",
        "tandem.planners.tiptop",
        "tandem.planners.tiptop",
        "tandem.planners.tiptop.options",
    ]


def test_the_allowed_exceptions_are_only_what_they_claim_to_be():
    registry_refs = _tiptop_references(SRC / "planners" / "registry.py")
    assert registry_refs == [], "the registry names TiPToP by string in its table, and imports nothing of it"
    shim = (SRC / "core" / "runtime.py").read_text()
    assert "deprecated" in shim.lower() and "def " not in shim, "the shim only re-exports"


# --- (d) doctor and profile show ask the planner ---------------------------------------------------------


def _toy_profile(name: str = "bins", **options):
    registry.register_backend("toy", _toy(doctor_checks=classmethod(_toy_rows)))
    profile = Profile.model_validate({"name": name, "planner": {"backend": "toy", "options": options}})
    profiles.save(profile)
    return profile


def _toy_rows(cls, profile, *, settings=None, probe_hardware=True):
    rows = [
        probe.Check("bin sensor", probe.OK, "3 bins seen", group="hardware" if probe_hardware else "profile")
    ]
    if profile is not None:
        rows.append(
            probe.Check("toy items", probe.WARN, f"{len(profile.planner.options.get('items') or [])} item(s)")
        )
    return rows


TIPTOP_ROWS = {
    "perception settings",
    "nvidia driver",
    "cuda runtime",
    "nvcc",
    "zed sdk",
    "gemini for perception",
    "camera calibration",
    "tamp settings",
    "robot control",
    "robot state port",
    "m2t2 grasp server",
}


def test_doctor_shows_a_toy_planners_own_checks_and_none_of_tiptops(isolated_env, monkeypatch):
    from tandem.cli.doctor import collect_checks

    # A machine with no pixi: nothing the toy planner builds needs one, so that is no failure.
    monkeypatch.setattr(probe, "find_pixi", lambda: None)
    _toy_profile(items=["duck"])
    checks = {c.name: c for c in collect_checks(profile_name="bins", probe_hardware=True)}
    assert checks["bin sensor"].state == probe.OK and checks["bin sensor"].detail == "3 bins seen"
    assert checks["toy items"].detail == "1 item(s)"
    assert not TIPTOP_ROWS & set(checks), (
        "a profile that does not plan with TiPToP is asked nothing of TiPToP's"
    )
    assert checks["planner runtime"].detail == "toy is pure Python"
    assert "gpu runtime" not in checks
    assert checks["pixi"].state == probe.SKIP and "disk space" not in checks, "a GPU runtime's needs"
    assert not [c for c in checks.values() if c.state == probe.FAIL]
    # tandem's own rows are there whichever planner runs.
    assert {"python", "profile", "cameras", "phase planning", "gemini api key"} <= set(checks)
    assert checks["gemini api key"].state == probe.SKIP, (
        "phase planning is off, and the toy planner needs no key"
    )


def test_doctor_asks_tiptop_for_tiptops_rows(profile, monkeypatch):
    from tandem.cli.doctor import collect_checks

    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
    checks = {c.name: c for c in collect_checks(profile_name=profile.name, probe_hardware=False)}
    assert {"nvidia driver", "gemini for perception", "camera calibration", "tamp settings"} <= set(checks)
    assert checks["camera calibration"].state == probe.OK
    assert not {"robot control", "m2t2 grasp server", "zed sdk"} & set(checks), (
        "--no-hardware touches nothing"
    )


def test_a_planner_whose_doctor_checks_break_is_one_failed_row_and_doctor_carries_on(isolated_env):
    from tandem.cli.doctor import collect_checks

    def broken(cls, profile, *, settings=None, probe_hardware=True):
        raise RuntimeError("bin sensor driver missing")

    registry.register_backend("toy", _toy(doctor_checks=classmethod(broken)))
    profiles.save(Profile.model_validate({"name": "bins", "planner": {"backend": "toy"}}))
    checks = {c.name: c for c in collect_checks(profile_name="bins", probe_hardware=False)}
    assert checks["planner toy"].state == probe.FAIL
    assert "RuntimeError: bin sensor driver missing" in checks["planner toy"].detail
    assert "profile" in checks and "python" in checks


def test_init_preflight_asks_the_chosen_planner_what_it_needs_of_the_machine():
    registry.register_backend("toy", _toy(doctor_checks=classmethod(_toy_rows)))
    rows = registry.doctor_checks("toy", None, probe_hardware=True)
    assert [row.name for row in rows] == ["bin sensor"], "no profile yet: the machine only"
    tiptop = {row.name for row in registry.doctor_checks("tiptop", None, probe_hardware=False)}
    assert tiptop == {"nvidia driver", "cuda runtime", "nvcc"}, (
        "the key is init's later question, not a blocker"
    )


def test_profile_show_is_the_planners_description_of_its_options(isolated_env):
    from tandem.cli.app import app

    def describe(cls, profile, *, settings=None):
        items = profile.planner.options.get("items") or []
        return OptionsView(
            summary=f"{len(items)} item(s) on the floor",
            sections=(OptionsSection("floor", tuple((item, "on the floor") for item in items)),),
            receives={"items": list(items)},
            receives_note="handed to the toy world as it starts",
            warnings=("no bin is labelled",),
        )

    registry.register_backend("toy", _toy(describe_options=classmethod(describe)))
    profiles.save(
        Profile.model_validate(
            {"name": "bins", "planner": {"backend": "toy", "options": {"items": ["duck"]}}}
        )
    )

    runner = CliRunner()
    shown = runner.invoke(app, ["profile", "show", "bins"])
    assert shown.exit_code == 0, shown.output
    assert "floor" in shown.output and "duck" in shown.output and "no bin is labelled" in shown.output
    assert "fr3_robotiq" not in shown.output and "tamp" not in shown.output

    receives = runner.invoke(app, ["profile", "show", "bins", "--planner", "--json"])
    assert json.loads(receives.output) == {"items": ["duck"]}
    payload = json.loads(runner.invoke(app, ["profile", "show", "bins", "--json"]).output)
    assert payload["_resolved"]["planner"]["summary"] == "1 item(s) on the floor"
    assert payload["_resolved"]["warnings"] == ["no bin is labelled"]


def test_profile_show_still_shows_a_profile_whose_planner_will_not_load(isolated_env, monkeypatch):
    """The profile is where a broken planner gets noticed, so it must still be shown -- options as
    written, the planner's error among the warnings. What that planner receives is not guessed at."""
    import importlib.metadata

    from tandem.cli.app import app

    declared = [importlib.metadata.EntryPoint("gone", "tandem_no_such_planner:FACTORY", registry.GROUP)]
    real = importlib.metadata.entry_points
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda **params: list(declared) if params.get("group") == registry.GROUP else real(**params),
    )
    profiles.save(
        Profile.model_validate(
            {"name": "bins", "planner": {"backend": "gone", "options": {"items": ["duck"]}}}
        )
    )

    runner = CliRunner()
    shown = runner.invoke(app, ["profile", "show", "bins"])
    assert shown.exit_code == 0, shown.output
    assert "duck" in shown.output and "tandem_no_such_planner" in shown.output
    receives = runner.invoke(app, ["profile", "show", "bins", "--planner"])
    assert receives.exit_code != 0


def test_a_planner_that_describes_nothing_is_shown_its_options_as_they_are():
    registry.register_backend("toy", _toy())
    profile = Profile.model_validate(
        {"name": "bins", "planner": {"backend": "toy", "options": {"items": ["duck"]}}}
    )
    view = registry.describe_options("toy", profile)
    assert view.receives == {"items": ["duck"]}
    assert view.sections[0].rows == (("items", "duck"),)
    json.dumps(view.to_dict())


def test_tiptop_describes_its_robot_perception_and_overrides(profile):
    view = registry.describe_options("tiptop", profile)
    assert view.summary == "fr3_robotiq at 172.16.0.2  ·  20% speed"
    assert [section.title for section in view.sections] == ["robot", "perception", "tamp"]
    assert view.receives["num_particles"] == 256 and view.receives["traj_length_norm"] == "inf"
    assert "--curobo-overrides" in view.receives_note
    assert view.warnings == ()
    json.dumps(view.to_dict())


def test_tandem_profile_show_tamp_still_prints_what_tiptop_receives(profile):
    from tandem.cli.app import app

    result = CliRunner().invoke(app, ["profile", "show", profile.name, "--tamp", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["num_particles"] == 256


# --- each asks for what it needs: the Gemini key --------------------------------------------------------


def test_a_script_that_still_hands_the_session_a_runtime_is_told_and_has_it_used(
    profile, monkeypatch, tmp_path
):
    """``Session(profile, Runtime(...))`` predates the registry. It still works -- the runtime's root is
    the one the planner's factory is told to use -- and it says it is deprecated."""
    from helpers import FakeFactory

    from tandem.core.session import Session

    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
    factory = FakeFactory("tiptop")
    registry.register_backend("tiptop", factory, replace=True)

    class OldRuntime:  # what a script built with tandem.core.runtime.Runtime(...)
        root = tmp_path / "runtime"

    with pytest.warns(DeprecationWarning, match=r"Session\(profile, runtime\) is deprecated"):
        session = Session(profile, OldRuntime(), task="x")
    session.start()
    try:
        assert [ctx.runtime_dir for ctx in factory.contexts] == [OldRuntime.root]
    finally:
        session.stop(park=False)
        session.wait(timeout=5)


def test_a_session_needs_a_gemini_key_only_for_phase_planning_or_a_planner_that_asks(profile, monkeypatch):
    from helpers import use_fake_backend

    from tandem.core.session import Session

    monkeypatch.setattr(secrets, "gemini_api_key", lambda: None)
    built = use_fake_backend(monkeypatch)
    session = Session(profile, task="x")
    session.start()  # phase planning is off, and the stand-in planner asks for no key
    try:
        assert built, "the planner was built"
    finally:
        session.stop(park=False)
        session.wait(timeout=5)

    profile.hitl.enabled = True
    with pytest.raises(TandemError, match="phase planning \\(hitl.enabled\\) asks Gemini"):
        Session(profile, task="x").start()


def test_tiptop_asks_for_the_key_its_perception_calls(profile, monkeypatch, tmp_path):
    from tandem.core import paths
    from tandem.planners.base import BackendContext

    monkeypatch.setattr(secrets, "gemini_api_key", lambda: None)
    session_dir = paths.session_scratch_dir() / profile.name / "s1"
    ctx = BackendContext(
        profile=profile, session_dir=session_dir, output_dir=tmp_path, options=dict(profile.planner.options)
    )
    with pytest.raises(TandemError, match="TiPToP's perception calls Gemini every rollout"):
        TIPTOP.create(ctx)


# --- the rest of what tandem asks a planner by name ------------------------------------------------------


def test_replaying_a_trajectory_asks_the_profiles_planner(tmp_path):
    registry.register_backend("toy", _toy())
    with pytest.raises(TandemError, match="The Toy planner has no viewer"):
        registry.replay("toy", tmp_path)
    with pytest.raises(TandemError, match="has no tiptop_plan.json, so there is no plan to replay"):
        registry.replay("tiptop", tmp_path)


def test_there_is_no_importer_any_more(isolated_env, tmp_path):
    """A profile no longer holds a rig to import: the hitl-tamp-vla importer, its hook and its flags are gone."""
    import tandem.planners as sdk
    from tandem.cli.app import app

    assert not hasattr(registry, "importer")
    assert not hasattr(TIPTOP, "importer")
    for name in ("ProfileImporter", "WARNING_NOTE"):
        with pytest.raises(AttributeError):
            getattr(sdk, name)
    with pytest.raises(ImportError):
        __import__("tandem.planners.tiptop.importers")
    for flags in (["--import-from", str(tmp_path)], ["--tamp-config", str(tmp_path / "x.yml")]):
        result = CliRunner().invoke(app, ["profile", "create", "x", *flags])
        assert result.exit_code == 2, result.output
        assert "No such option" in result.output
    result = CliRunner().invoke(app, ["init", "-y", "--import-from", str(tmp_path)])
    assert result.exit_code == 2 and "No such option" in result.output


def test_a_merge_prefers_the_planner_runtimes_own_ffmpeg(tmp_path, monkeypatch):
    from tandem.core import merge
    from tandem.planners.runtime import RecipeRuntime

    bin_dir = tmp_path / "env" / "envs" / "default" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").write_text("")
    (bin_dir / "ffmpeg").write_text("")
    monkeypatch.setattr(RecipeRuntime, "bin_dir", property(lambda self: bin_dir))
    assert registry.tools_dir("tiptop") == bin_dir
    assert merge._tool("ffmpeg", bin_dir) == str(bin_dir / "ffmpeg")

    registry.register_backend("toy", _toy())
    assert registry.tools_dir("toy") is None, "a pure-Python planner has no tools of its own"
