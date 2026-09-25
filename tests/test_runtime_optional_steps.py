"""An optional build step: part of a runtime only some machines can have, done when they can.

TiPToP's is the ZED Python API (pyzed), which comes from the ZED SDK's own installer rather than from
PyPI. Without the SDK the runtime still builds and works -- planning, replay, the viewer -- and every place
a person looks (the install's output, `planners info`, `planners list`, `tandem doctor`) says the ZED cameras
will not open until the SDK is installed, and how to fix it. With it, the install runs the step; once run,
it is skipped. Here the step belongs to the toy recipe of tests/test_runtime_recipe.py, and its
requirement is a file the test creates or does not.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from helpers import isolate_registry
from test_runtime_recipe import (  # noqa: F401 - fixtures
    _ToyFactory,
    never_install_pixi,
    no_sources_override,
    shipped,
    toy,
    upstream,
)
from typer.testing import CliRunner

import tandem.planners.runtime as rt_mod
from tandem.cli.app import app
from tandem.core.errors import TandemError
from tandem.planners import registry
from tandem.planners.runtime import BuildStep, RecipeRuntime

LABEL = "camera API"


@pytest.fixture
def sdk(tmp_path) -> Path:
    """Where the camera SDK's installer would be. Not there until a test creates it."""
    return tmp_path / "opt" / "camera-sdk" / "get_python_api.py"


def _missing(sdk: Path) -> str:
    return f"the camera SDK is not installed (no {sdk}), so the cameras will not open. Install it, then run " \
        "`tandem planners install toy`."


def with_camera(recipe, sdk: Path, **overrides):
    step = BuildStep(
        "camera",
        task="install-cam",
        optional=True,
        requires=(str(sdk),),
        produces=("env/envs/default/lib/python3*/site-packages/cam",),
        label=LABEL,
        done="installed",
        todo="not installed",
        missing=_missing(sdk),
    )
    return replace(recipe, steps=(*recipe.steps, replace(step, **overrides)))


@pytest.fixture
def fake_pixi(tmp_path, monkeypatch):
    """pixi, as far as a build goes: `install` makes the environment, `run compile` a kernel, `run install-cam`
    the camera bindings in the environment's site-packages (or fails, with TOY_CAM_FAIL=1). Logs each call."""
    calls = tmp_path / "pixi-calls.jsonl"
    script = tmp_path / "bin" / "pixi"
    script.parent.mkdir()
    script.write_text(
        f"""#!{sys.executable}
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
manifest = Path(args[args.index("--manifest-path") + 1])
root = manifest.parent.parent
with open({str(calls)!r}, "a") as fh:
    fh.write(json.dumps({{"args": args}}) + "\\n")
if args[0] == "install":
    python = manifest.parent / ".pixi" / "envs" / "default" / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("")
elif args[-1] == "compile":
    (root / "build").mkdir(exist_ok=True)
    (root / "build" / "kernel.so").write_text("")
elif args[-1] == "install-cam":
    if os.environ.get("TOY_CAM_FAIL") == "1":
        print("the camera installer fell over")
        sys.exit(3)
    cam = root / "env" / "envs" / "default" / "lib" / "python3.12" / "site-packages" / "cam"
    cam.mkdir(parents=True, exist_ok=True)
    (cam / "__init__.py").write_text("")
print("pixi: done")
"""
    )
    script.chmod(0o755)
    monkeypatch.setattr(rt_mod, "_find_pixi", lambda: script)
    monkeypatch.setattr("tandem.core.probe.find_pixi", lambda: script)

    def tasks() -> list[str]:
        lines = calls.read_text().splitlines() if calls.is_file() else []
        return [json.loads(line)["args"][-1] for line in lines if json.loads(line)["args"][0] == "run"]

    return tasks


@pytest.fixture
def runtime(upstream, shipped, tmp_path, sdk) -> RecipeRuntime:  # noqa: F811 - the fixtures above
    return RecipeRuntime(with_camera(toy(upstream, shipped), sdk), tmp_path / "runtime")


def _install(rt: RecipeRuntime) -> list[str]:
    lines: list[str] = []
    rt.install(on_progress=lines.append)
    return lines


# --------------------------------------------------------------------------- the recipe runtime


def test_without_its_requirement_the_install_succeeds_and_says_what_is_missing(runtime, sdk, fake_pixi):
    lines = _install(runtime)
    assert f"{LABEL}: skipped: {_missing(sdk)}" in lines
    assert fake_pixi() == ["compile"], "the step never ran"

    status = runtime.status()
    assert status.installed and status.problems == ()
    assert status.notes == (f"{LABEL} not installed: {_missing(sdk)}",)
    assert status.to_dict()["notes"] == list(status.notes)
    st = runtime.inspect()
    assert st.steps_done and st.optional == ("camera",), "an optional step never decides whether it is built"
    assert runtime.optional_to_run(st) == []


def test_with_its_requirement_the_step_runs_once(runtime, sdk, fake_pixi):
    sdk.parent.mkdir(parents=True)
    sdk.write_text("# the SDK's installer")
    _install(runtime)
    assert fake_pixi() == ["compile", "install-cam"]
    assert runtime.status().notes == () and dict(runtime.inspect().steps)["camera"]

    lines = _install(runtime)
    assert f"{LABEL}: already installed" in lines
    assert fake_pixi() == ["compile", "install-cam", "compile"], "done, so skipped"


def test_installing_the_sdk_later_makes_the_next_install_run_the_step(runtime, sdk, fake_pixi):
    _install(runtime)
    assert runtime.optional_to_run() == []
    sdk.parent.mkdir(parents=True)
    sdk.write_text("")
    assert runtime.optional_to_run() == [LABEL]
    assert runtime.status().notes == (f"{LABEL} not installed: `tandem planners install toy` installs it",)
    _install(runtime)
    assert fake_pixi()[-1] == "install-cam" and runtime.optional_to_run() == []


def test_a_step_that_fails_is_said_and_does_not_fail_the_install(runtime, sdk, fake_pixi, monkeypatch):
    sdk.parent.mkdir(parents=True)
    sdk.write_text("")
    monkeypatch.setenv("TOY_CAM_FAIL", "1")
    lines = _install(runtime)
    assert any(line.startswith(f"{LABEL} failed: pixi run install-cam failed (exit 3).") for line in lines)
    assert any("The runtime works without it." in line for line in lines)
    status = runtime.status()
    assert status.installed and runtime.record()["last_build"] is None, "not a build that did not finish"
    assert status.notes == (f"{LABEL} not installed: `tandem planners install toy` installs it",)


def test_a_listing_shows_the_step(runtime, fake_pixi):
    _install(runtime)
    assert f"{LABEL} not installed" in runtime.describe()
    assert (LABEL, "not installed") in runtime.rows()


def test_a_recipe_that_could_not_say_whether_its_optional_step_ran_is_refused(upstream, shipped, sdk):  # noqa: F811
    recipe = toy(upstream, shipped)
    with pytest.raises(TandemError, match="declares no produces"):
        with_camera(recipe, sdk, produces=())
    with pytest.raises(TandemError, match="not an absolute path"):
        with_camera(recipe, sdk, requires=("opt/camera-sdk/get_python_api.py",))
    with pytest.raises(TandemError, match="must be optional and say what is `missing`"):
        with_camera(recipe, sdk, missing="")
    with pytest.raises(TandemError, match="must be optional"):
        with_camera(recipe, sdk, optional=False)


# --------------------------------------------------------------------------- where a person reads it


@pytest.fixture
def installed_toy(runtime, fake_pixi, monkeypatch):
    isolate_registry(monkeypatch)
    factory = _ToyFactory(runtime.recipe, runtime.root)
    registry.register_backend("toy", factory)
    return runtime


def _said(result) -> str:
    return " ".join(result.output.split())


def test_the_install_output_says_what_will_not_work_and_the_fix(installed_toy, sdk):
    result = CliRunner().invoke(app, ["planners", "install", "toy", "--yes"])
    assert result.exit_code == 0, result.output
    assert "Runtime built" in result.output
    # (The path itself may be wrapped across lines.)
    assert f"{LABEL} not installed: the camera SDK is not installed (no " in _said(result)
    assert "so the cameras will not open. Install it, then run `tandem planners install toy`" in _said(result)

    # Nothing to do on a second run -- and it still says so.
    again = CliRunner().invoke(app, ["planners", "install", "toy", "--yes"])
    assert "already installed" in again.output and f"{LABEL} not installed" in _said(again)


def test_an_installed_runtime_is_built_again_once_the_step_can_run(installed_toy, sdk, fake_pixi):
    assert CliRunner().invoke(app, ["planners", "install", "toy", "--yes"]).exit_code == 0
    sdk.parent.mkdir(parents=True)
    sdk.write_text("")
    result = CliRunner().invoke(app, ["planners", "install", "toy", "--yes"])
    assert result.exit_code == 0, result.output
    assert f"now installing {LABEL}" in _said(result)
    assert fake_pixi()[-1] == "install-cam"


def test_info_list_and_doctor_say_it_too(installed_toy, sdk, profile, monkeypatch):
    from tandem.cli.doctor import collect_checks
    from tandem.core import profiles
    from tandem.core import settings as settings_mod

    installed_toy.install()
    info = CliRunner().invoke(app, ["planners", "info", "toy"])
    assert info.exit_code == 0 and f"{LABEL} not installed: the camera SDK is not installed" in _said(info)
    listed = json.loads(CliRunner().invoke(app, ["planners", "list", "--json"]).output)
    (row,) = [row for row in listed["planners"] if row["name"] == "toy"]
    assert row["status"] == "installed" and f"{LABEL} not installed" in row["detail"]

    profile.planner = profiles.PlannerSpec(backend="toy")
    profiles.save(profile)
    cfg = settings_mod.load()
    cfg.active_profile = profile.name
    settings_mod.save(cfg)
    checks = {c.name: c for c in collect_checks(profile_name=profile.name, probe_hardware=False)}
    row = checks[LABEL.lower()]
    assert row.state == "warn" and row.detail == "not installed" and row.hint == _missing(sdk)
    assert checks["planner runtime"].state == "ok", "the runtime is ready without it"

    sdk.parent.mkdir(parents=True)
    sdk.write_text("")
    checks = {c.name: c for c in collect_checks(profile_name=profile.name, probe_hardware=False)}
    assert checks[LABEL.lower()].hint == "`tandem planners install toy` installs it."
    installed_toy.install()
    checks = {c.name: c for c in collect_checks(profile_name=profile.name, probe_hardware=False)}
    assert checks[LABEL.lower()].state == "ok"


def test_init_builds_again_for_a_step_it_can_now_take(installed_toy, sdk, fake_pixi):
    from tandem.cli import init as init_cli

    init_cli._build_runtime("toy", interactive=False, repair=False)
    runs = len(fake_pixi())
    init_cli._build_runtime("toy", interactive=False, repair=False)
    assert len(fake_pixi()) == runs, "built, and nothing it can do now: left alone"
    sdk.parent.mkdir(parents=True)
    sdk.write_text("")
    init_cli._build_runtime("toy", interactive=False, repair=False)
    assert fake_pixi()[-1] == "install-cam"


# --------------------------------------------------------------------------- TiPToP's


def test_tiptops_recipe_installs_the_zed_python_api_when_the_sdk_is_there():
    from tandem.planners.tiptop.recipe import RECIPE, ZED_PYTHON_API

    (zed,) = [step for step in RECIPE.steps if step.optional]
    assert zed.task == "install-zed" and zed.requires == (ZED_PYTHON_API,)
    assert ZED_PYTHON_API == "/usr/local/zed/get_python_api.py"
    assert zed.produces == ("env/envs/default/lib/python3*/site-packages/pyzed",)
    assert zed.title == "ZED Python API"
    assert "ZED cameras will not open" in zed.missing and "`tandem planners install tiptop`" in zed.missing
    assert "stereolabs.com" in zed.missing
    # The kernels are what the runtime cannot run without; the ZED step comes after them. That the pinned
    # tiptop has the task, and what it runs: tests/test_tiptop_bump.py.
    assert [step.name for step in RECIPE.steps] == ["planners", "zed"]
