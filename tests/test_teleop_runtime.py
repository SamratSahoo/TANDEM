"""The teleop runtime tandem builds, and a hand-off launched from it.

With neither ``teleop.python`` nor ``teleop.droid_dir`` set, the driver runs in the runtime ``tandem
executors install teleop`` builds (``tandem.teleop.recipe``): its interpreter, both of its trees on
PYTHONPATH, and the rig's robot address and camera serials in the environment DROID reads them from.
Nothing is fetched or built here; the runtime is a stand-in with the same few members.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from tandem.cli.app import app
from tandem.core.errors import TandemError
from tandem.core.settings import Settings
from tandem.planners.runtime import RecipeRuntime, RuntimeNotReady
from tandem.teleop import child as child_mod
from tandem.teleop import recipe


class _Session:
    """Just enough Session for TeleopChild.start, with a rig the way the executor's profile view has one."""

    def __init__(self, tmp_path: Path) -> None:
        cams = {"hand": SimpleNamespace(serial="111"), "external": SimpleNamespace(serial="222")}
        self._trajectory_id = "traj-abc123"
        self.instruction = "open the box"
        self._files = {"session_dir": tmp_path}
        self.current = None
        self.profile = SimpleNamespace(
            trajectories_dir=lambda: tmp_path / "trajectories",
            cameras=SimpleNamespace(configured=lambda: cams),
            rig=SimpleNamespace(robot=SimpleNamespace(host="10.1.2.3")),
        )

    def _log(self, stream: str, text: str) -> None:
        pass

    def _pump(self, stream, name: str) -> None:
        pass


class _Runtime:
    """The members of a RecipeRuntime the launch reads."""

    def __init__(self, root: Path, *, ready: bool = True) -> None:
        self.root = root
        self.ready = ready

    def require_ready(self) -> None:
        if not self.ready:
            raise RuntimeNotReady("not built", hint=f"Run `{recipe.INSTALL_COMMAND}`.")

    def python(self) -> Path:
        return Path(sys.executable)

    def source_dir(self, name: str) -> Path:
        return self.root / name


def _launch(tmp_path, monkeypatch, cfg: Settings) -> tuple[list[str], dict]:
    launched: list[tuple[list[str], dict]] = []

    def popen(args, **kwargs):
        launched.append((list(args), kwargs))
        return SimpleNamespace(stdout=None)

    monkeypatch.setattr(child_mod.subprocess, "Popen", popen)
    monkeypatch.setattr(child_mod.events_mod, "EventTailer", lambda *a, **k: SimpleNamespace(start=lambda: None))
    child_mod.TeleopChild(_Session(tmp_path), cfg).start()
    assert launched, "the driver was never launched"
    return launched[0]


def test_the_recipe_pins_droids_tandem_branch_and_names_its_own_install_command():
    droid = recipe.RECIPE.source("droid")
    assert droid.pin.ref == "TANDEM" and len(droid.pin.commit) == 40
    assert recipe.RECIPE.environment.manifest == "droid/pixi.toml"
    assert recipe.RECIPE.build_command == "tandem executors install teleop"
    # Every hint the runtime gives says how to build THIS runtime, not a planner's.
    rt = RecipeRuntime(recipe.RECIPE, Path("/nonexistent/teleop"))
    with pytest.raises(RuntimeNotReady) as caught:
        rt.require_ready()
    assert "tandem executors install teleop" in caught.value.hint


def test_with_no_override_the_driver_runs_in_the_runtime_with_the_rigs_robot_and_cameras(tmp_path, monkeypatch):
    monkeypatch.setattr(recipe, "runtime", lambda settings=None: _Runtime(tmp_path / "rt"))
    cfg = Settings()
    cfg.teleop.enabled = True
    args, kwargs = _launch(tmp_path, monkeypatch, cfg)

    assert args[0] == sys.executable
    assert kwargs["cwd"] == str(tmp_path / "rt" / "droid")
    env = kwargs["env"]
    path = env["PYTHONPATH"].split(":")
    assert path[:2] == [str(tmp_path / "rt" / "droid"), str(tmp_path / "rt" / "oculus_reader")]
    # What droid.misc.parameters reads, so nobody edits that file and it never disagrees with the rig.
    assert env["DROID_NUC_IP"] == "10.1.2.3"
    assert env["TIPTOP_HAND_CAMERA_ID"] == "111"
    assert env["TIPTOP_EXTERNAL_CAMERA_ID"] == "222"


def test_a_runtime_that_is_not_built_says_how_to_build_it(tmp_path, monkeypatch):
    monkeypatch.setattr(recipe, "runtime", lambda settings=None: _Runtime(tmp_path / "rt", ready=False))
    cfg = Settings()
    cfg.teleop.enabled = True
    with pytest.raises(TandemError) as caught:
        _launch(tmp_path, monkeypatch, cfg)
    assert "tandem executors install teleop" in (caught.value.hint or "")


def test_a_checkout_of_ones_own_still_wins_over_the_runtime(tmp_path, monkeypatch):
    def no_runtime(settings=None):
        raise AssertionError("the runtime was consulted although both overrides are set")

    monkeypatch.setattr(recipe, "runtime", no_runtime)
    cfg = Settings()
    cfg.teleop.enabled = True
    cfg.teleop.python = sys.executable
    cfg.teleop.droid_dir = str(tmp_path)
    args, kwargs = _launch(tmp_path, monkeypatch, cfg)
    assert args[0] == sys.executable
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["env"]["PYTHONPATH"].split(":")[0] == str(tmp_path)


def test_only_teleop_has_a_runtime_to_install(tmp_path, monkeypatch):
    monkeypatch.setenv("TANDEM_CONFIG_DIR", str(tmp_path / "config"))
    result = CliRunner().invoke(app, ["executors", "install", "teleopp"])
    assert isinstance(result.exception, TandemError), result.output
    assert "Did you mean 'teleop'?" in result.exception.hint


def test_installing_the_runtime_turns_teleop_on(tmp_path, monkeypatch):
    from tandem.cli import runtime as runtime_cli
    from tandem.core import settings as settings_mod

    monkeypatch.setenv("TANDEM_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("TANDEM_RUNTIMES_DIR", str(tmp_path / "runtimes"))
    built: list[Path] = []
    monkeypatch.setattr(runtime_cli, "needs_pixi", lambda rt: False)
    monkeypatch.setattr(runtime_cli, "run_build", lambda rt, **kw: built.append(rt.root))

    result = CliRunner().invoke(app, ["executors", "install", "teleop", "--yes"])
    assert result.exit_code == 0, result.output
    assert built == [tmp_path / "runtimes" / "teleop"]
    assert settings_mod.load().teleop.enabled
