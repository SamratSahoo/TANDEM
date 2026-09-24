"""The commands, where the review found them saying or doing the wrong thing.

- `tandem init` stopped on pixi and on TiPToP's disk budget before the planner was even chosen, so
  `init --yes` failed on every machine without pixi and a pure-Python planner could not be set up.
- `init --repair` rebuilt an existing profile from the template, silently.
- Presets could only be applied by `tandem profile create`; not by the web, not by `init`.
- A profile for another planner was warned about TiPToP's camera extrinsics.
- `profile migrate` crashed on an unknown name, rewrote current profiles (dropping comments, freezing
  ``${oc.env}``) and stopped at the first broken one.
- `profile edit` crashed on ``EDITOR='code --wait'`` and left a backup behind.
- `tandem ui --profile NAME` and `tandem collect NAME --web` opened on the active profile instead.
- A policy executor's legs were shown as the planner's, and not counted as hand-offs.
- `traj open` replayed with the profile's current planner rather than the one that recorded it.
- `tandem plan` alone said --backend, and names the docs tell planner authors to import were not there.
- pixi's consent text said it touched nothing else, while its installer edited the shell's rc file.
- `planners new` told a pipx user to `pip install` into the wrong environment.
- A planner's preset could override tandem's phase planning under the paper's name.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_merge_optional_keys
import test_public_api
from fastapi.testclient import TestClient
from helpers import isolate_registry
from test_catalog_cli import PINS, RuntimeFactory
from toy_planner import ToyPlanner
from typer.testing import CliRunner

from tandem.cli.app import app
from tandem.core import probe, profiles
from tandem.core import settings as settings_mod
from tandem.core.errors import ProfileError, TandemError
from tandem.executors import base as executors
from tandem.planners import registry
from tandem.server.app import create_app

FIXTURES = Path(__file__).parent / "fixtures" / "profiles"

# Shared, not copied: the merge tests' stand-in for ffmpeg, and the public API's scripted model.
fake_video = test_merge_optional_keys.fake_video
model = test_public_api.model
photo = test_public_api.photo


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    isolate_registry(monkeypatch)
    monkeypatch.setattr(executors, "_registered", dict(executors._registered))
    monkeypatch.setattr(executors, "_discovered", None)


def _run(*args: str, input: str | None = None):
    return CliRunner().invoke(app, list(args), input=input)


def _activate(name: str) -> None:
    cfg = settings_mod.load()
    cfg.active_profile = name
    settings_mod.save(cfg)


# --- init: the runtime's needs are asked of the chosen planner, after it is chosen ------------------------


@pytest.fixture
def no_pixi(monkeypatch):
    """No pixi on this machine until something installs it; installing it is recorded, never run."""
    state = {"installed": 0, "builds": []}
    monkeypatch.setattr(probe, "find_pixi", lambda: Path("/opt/pixi") if state["installed"] else None)
    monkeypatch.setattr(probe, "check_ffmpeg", lambda: probe.Check("ffmpeg", probe.OK, "stand-in"))
    monkeypatch.setattr("tandem.cli.runtime.install_pixi", lambda log=None: state.update(installed=state["installed"] + 1))
    monkeypatch.setattr("tandem.cli.runtime.run_build", lambda rt, **kw: state["builds"].append(kw))
    return state


def test_init_yes_sets_up_a_pure_python_planner_with_no_pixi_and_little_disk(no_pixi, monkeypatch):
    registry.register_backend("toy", ToyPlanner)
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: SimpleNamespace(total=6e9, used=1e9, free=5e9))
    result = _run("init", "--yes", "--planner", "toy")
    assert result.exit_code == 0, result.output
    assert settings_mod.load(force=True).default_planner == "toy"
    assert no_pixi["installed"] == 0, "a pure-Python planner needs no pixi"
    assert "blocking problems" not in result.output and "pure Python" in result.output


def test_init_yes_installs_pixi_for_a_planner_that_needs_it_instead_of_stopping(no_pixi, tmp_path):
    from tandem.planners.runtime import PixiEnvironment, RecipeRuntime, RuntimeRecipe, Source

    recipe = RuntimeRecipe(
        planner="heavy", sources=(Source(PINS[0]),), environment=PixiEnvironment(manifest="solver/pixi.toml")
    )
    registry.register_backend("heavy", RuntimeFactory("heavy", RecipeRuntime(recipe, tmp_path / "heavy")))
    result = _run("init", "--yes", "--planner", "heavy")
    assert result.exit_code == 0, result.output
    assert no_pixi["installed"] == 1 and len(no_pixi["builds"]) == 1
    assert "init installs it" in result.output


def test_a_planner_whose_runtime_does_not_fit_still_stops_init(no_pixi, tmp_path, monkeypatch):
    from tandem.planners.runtime import PixiEnvironment, RecipeRuntime, RuntimeRecipe, Source

    recipe = RuntimeRecipe(
        planner="heavy", sources=(Source(PINS[0]),), environment=PixiEnvironment(manifest="solver/pixi.toml")
    )
    registry.register_backend("heavy", RuntimeFactory("heavy", RecipeRuntime(recipe, tmp_path / "heavy")))
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: SimpleNamespace(total=6e9, used=1e9, free=5e9))
    result = _run("init", "--yes", "--planner", "heavy")
    assert result.exit_code == 1
    assert "disk space" in result.exception.message and no_pixi["builds"] == []


def test_init_says_nothing_about_extrinsics_to_a_planner_that_never_reads_them(no_pixi):
    registry.register_backend("toy", ToyPlanner)
    result = _run("init", "--yes", "--planner", "toy")
    assert result.exit_code == 0, result.output
    assert "extrinsics" not in result.output and "refuse to start" not in result.output


# --- init never rebuilds an existing profile ---------------------------------------------------------------


def test_init_repair_leaves_an_existing_profile_as_it_was(isolated_env):
    assert _run("profile", "create", "p1", "--prompt", "fold it").exit_code == 0
    before = profiles.load("p1")
    result = _run("init", "--viz-only", "--yes", "--repair", "--profile", "p1")
    assert result.exit_code == 0, result.output
    after = profiles.load("p1")
    assert after.hitl.enabled and after.task.prompt == "fold it"
    assert after.planner == before.planner
    assert "Created profile" not in result.output


# --- presets, in the web (until its create takes a prompt) ---------------------------------------------


def test_the_web_creates_a_profile_with_a_preset_and_lists_them(profile):
    _activate(profile.name)
    client = TestClient(create_app())
    assert "paper" in [p["name"] for p in client.get("/api/presets").json()["presets"]]

    made = client.post(
        "/api/profiles", json={"name": "bread", "from": profile.name, "preset": "paper", "prompt": "bread in the box"}
    )
    assert made.status_code == 200, made.text
    bread = profiles.load("bread")
    assert bread.hitl.enabled and bread.planner.options["tamp"]["blend_mode"] == "vae"
    assert bread.task.prompt == "bread in the box", "the prompt is the profile's own and wins"
    assert made.json()["preset"]["name"] == "paper" and made.json()["preset"]["changed"]

    typo = client.post("/api/profiles", json={"name": "bun", "preset": "papr"})
    assert typo.status_code == 400 and "paper" in typo.json()["error"]
    assert not (profiles.profiles_root() / "bun").exists()

    unknown = client.post("/api/profiles", json={"name": "bun", "presett": "paper"})
    assert unknown.status_code == 400 and "presett" in unknown.json()["error"]


# --- the planner says what is wrong with a new profile, not TiPToP -------------------------------------------


def test_profile_create_warns_about_extrinsics_only_for_a_planner_that_reads_them(isolated_env):
    from tandem.core import rig as rig_mod

    # This machine's cameras, not yet calibrated.
    rig_mod.update({"cameras.hand": {"serial": "111"}, "cameras.external": {"serial": "222"}})
    registry.register_backend("toy", ToyPlanner)
    assert _run("planners", "default", "toy").exit_code == 0
    toy = _run("profile", "create", "shelf1", "--prompt", "stack the blocks")
    assert toy.exit_code == 0, toy.output
    assert "extrinsics" not in toy.output

    assert _run("planners", "default", "tiptop").exit_code == 0
    tiptop = _run("profile", "create", "x", "--prompt", "put the cup on the plate")
    assert tiptop.exit_code == 0, tiptop.output
    assert "no extrinsics" in tiptop.output


# --- profile migrate ------------------------------------------------------------------------------------------


def _write_old(name: str, text: str) -> Path:
    """A profile in the layout before version 3: a directory, with its profile.yml."""
    path = profiles.profiles_root() / name / "profile.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_migrate_with_nothing_to_move_says_so(profile):
    result = _run("profile", "migrate")
    assert result.exit_code == 0, result.output
    assert "No profiles in the old layout" in result.output
    # It moves every old profile or none: there is no one profile to name.
    assert _run("profile", "migrate", "test").exit_code == 2


def test_migrate_leaves_a_current_profile_byte_for_byte(profile, monkeypatch):
    text = profile.file().read_text()
    text = text.replace("target_episodes: 20", "target_episodes: 20   # the lab's weekly quota")
    assert "the lab's weekly quota" in text
    text += "# a note at the end\n"
    path = profiles.path_of("ctrl")
    path.write_text(text)
    _write_old("legacy", (FIXTURES / "v1_ef1411f.yml").read_text())
    before = path.read_bytes()
    monkeypatch.setenv("X_PORT", "9999")
    result = _run("profile", "migrate")
    assert result.exit_code == 0, result.output
    assert "legacy: moved" in " ".join(result.output.split())
    assert path.read_bytes() == before


def test_migrate_goes_on_past_a_broken_profile_and_keeps_the_originals(isolated_env):
    legacy = (FIXTURES / "v1_ef1411f.yml").read_text()
    _write_old("legacy_a", legacy), _write_old("legacy_z", legacy)
    broken = _write_old("broken", "version: 2\ntask:\n  bogus_key: 1\n")
    result = _run("profile", "migrate")
    assert result.exit_code != 0
    assert "broken" in result.output
    for name in ("legacy_a", "legacy_z"):
        written = profiles.path_of(name).read_text()
        assert "robot:" not in written and "cameras:" not in written, "migrated"
        archived = profiles.profiles_root() / ".migrated" / name / "profile.yml"
        assert archived.read_text() == legacy, "the original is kept whole"
    assert broken.read_text() == "version: 2\ntask:\n  bogus_key: 1\n", "left exactly as it was"


# --- $EDITOR with arguments -------------------------------------------------------------------------------------


@pytest.mark.parametrize("command", [["profile", "edit", "test"], ["config", "edit"]])
def test_an_editor_with_arguments_is_run_and_a_missing_one_is_an_error(profile, monkeypatch, command):
    backup = profile.file().with_suffix(".yml.bak")
    monkeypatch.setenv("EDITOR", "true --wait")
    ok = _run(*command)
    assert ok.exit_code == 0, ok.output
    assert not backup.exists()

    monkeypatch.setenv("EDITOR", "/nonexistent/ed --wait")
    missing = _run(*command)
    assert missing.exit_code == 1
    assert isinstance(missing.exception, TandemError) and "/nonexistent/ed" in missing.exception.message
    assert not backup.exists(), "the editor never ran, so there is nothing to restore"


# --- the web opens on the profile it was asked for ------------------------------------------------------------


@pytest.fixture
def served(monkeypatch):
    apps: list = []
    import uvicorn

    def run(self, sockets=None):
        # Serves nothing: the app it was handed is what the test reads.
        apps.append(self.config.app)
        self.started = True

    monkeypatch.setattr(uvicorn.Server, "run", run)
    monkeypatch.setattr("webbrowser.open", lambda url: None)
    return apps


@pytest.mark.parametrize("command", [["collect", "bread", "--web"], ["ui", "--no-open", "--profile", "bread"]])
def test_the_web_ui_opens_and_collects_under_the_named_profile(profile, served, command):
    bread = profile.model_copy(deep=True)
    bread.name = "bread"
    profiles.save(bread)
    _activate(profile.name)
    result = _run(*command)
    assert result.exit_code == 0, result.output
    assert TestClient(served[-1]).get("/api/profiles").json()["active"] == "bread"


def test_ui_refuses_a_profile_that_does_not_exist(profile, served):
    result = _run("ui", "--no-open", "--profile", "nope")
    assert result.exit_code != 0 and isinstance(result.exception, ProfileError)
    assert not served


# --- a policy executor's legs are human legs --------------------------------------------------------------------


def test_a_policy_executors_leg_is_shown_as_a_human_hand_off(profile, fake_video):
    from tandem.core import merge as merge_mod

    test_merge_optional_keys._hand_off(
        profile, "pol", ("tamp", "captured"), ("policy", "teleop"), ("tamp", "captured")
    )
    merged = merge_mod.merge(profile, "pol", status="success", tools_dir=None)
    assert merged.get("merged"), merged
    traj_id = Path(merged["dir"]).name

    listed = _run("traj", "list", profile.name)
    assert listed.exit_code == 0, listed.output
    assert "hand-off ×1" in listed.output

    shown = _run("traj", "show", traj_id, "-p", profile.name)
    assert shown.exit_code == 0, shown.output
    table = shown.output.split("hand-off legs", 1)[1].split("motion", 1)[0]
    assert "human (policy)" in table and table.count("TAMP") == 2


def test_the_web_counts_any_non_planner_leg_as_human():
    root = Path(__file__).resolve().parents[1] / "src/tandem/server/static"
    for page in (root / "review.js", root / "pages/trajectories.js"):
        text = page.read_text()
        assert '=== "teleop").length' not in text and 's.source === "teleop"' not in text, page.name


# --- traj open replays with the planner that recorded it ---------------------------------------------------------


def test_traj_open_uses_the_recording_planner_not_the_profiles_current_one(profile, make_trajectory, monkeypatch):
    from tandem.cli import planners as planners_cli
    from tandem.core import trajectories

    directory = make_trajectory(profile, "2026-02-02_00-00-00")
    (directory / trajectories.HITL_FILE).write_text(json.dumps({"planner": "tiptop"}))
    registry.register_backend("toy", ToyPlanner)
    _activate(profile.name)
    planners_cli.use_planner("toy")
    asked: list[str] = []
    monkeypatch.setattr(registry, "replay", lambda name, path, settings=None: asked.append(name))

    result = _run("traj", "open", "2026-02-02_00-00-00")
    assert result.exit_code == 0, result.output
    assert asked == ["tiptop"] and "Recorded by tiptop" in result.output


def test_traj_open_falls_back_to_the_profiles_planner_and_show_offers_only_a_real_viewer(
    profile, make_trajectory, monkeypatch
):
    from tandem.cli import planners as planners_cli

    directory = make_trajectory(profile, "2026-02-03_00-00-00")
    meta = json.loads((directory / "_meta.json").read_text())
    meta.pop("source")
    (directory / "_meta.json").write_text(json.dumps(meta))
    registry.register_backend("toy", ToyPlanner)
    _activate(profile.name)
    planners_cli.use_planner("toy")
    asked: list[str] = []
    monkeypatch.setattr(registry, "replay", lambda name, path, settings=None: asked.append(name))
    assert _run("traj", "open", "2026-02-03_00-00-00").exit_code == 0
    assert asked == ["toy"]

    shown = _run("traj", "show", "2026-02-03_00-00-00")
    assert shown.exit_code == 0, shown.output
    assert "tandem traj open" not in shown.output, "the toy planner has no viewer of its own"


# --- one word for a planner --------------------------------------------------------------------------------------


def test_tandem_plan_takes_planner_and_still_backend(model, photo):
    outputs = []
    for flag in ("--planner", "--backend"):
        model()
        result = _run("plan", "do the thing", "--image", str(photo), "--json", flag, "tiptop", "-o", "blue_toy", "-o", "white_box")
        assert result.exit_code == 0, result.output
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1]


def test_what_a_planner_author_is_told_to_import_is_there():
    import tandem
    from tandem.planners import (  # noqa: F401
        OptionsView,
        RuntimeNotReady,
        TandemError,
        register_planner,
    )

    assert tandem.register_planner is tandem.register_backend
    assert register_planner is registry.register_backend
    assert RuntimeNotReady is sys.modules["tandem.core.errors"].RuntimeNotReady


# --- pixi does what its consent says ------------------------------------------------------------------------------


def test_pixi_is_installed_without_editing_the_shells_rc_file(monkeypatch):
    from tandem.cli import runtime as runtime_cli

    monkeypatch.delenv("PIXI_NO_PATH_UPDATE", raising=False)
    installed = {"yes": False}
    monkeypatch.setattr(probe, "find_pixi", lambda: Path("/opt/pixi") if installed["yes"] else None)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    seen: dict = {}

    class _Proc:
        stdout: list = []

        def wait(self):
            installed["yes"] = True
            return 0

    def popen(args, **kwargs):
        seen.update(kwargs)
        return _Proc()

    monkeypatch.setattr(runtime_cli.subprocess, "Popen", popen)
    runtime_cli.install_pixi()
    assert seen["env"]["PIXI_NO_PATH_UPDATE"] == "1"


def test_the_consent_text_says_what_the_installer_does(monkeypatch, capsys):
    from tandem.cli import runtime as runtime_cli

    monkeypatch.setattr(probe, "find_pixi", lambda: None)
    monkeypatch.delenv("PIXI_NO_PATH_UPDATE", raising=False)
    def said() -> str:
        return " ".join(capsys.readouterr().out.split())  # as one line, whatever the console wrapped

    with pytest.raises(TandemError):
        runtime_cli.ensure_pixi("Heavy", ask=False, allowed=False)
    assert "rc file is left alone" in said()
    monkeypatch.setenv("PIXI_NO_PATH_UPDATE", "")
    with pytest.raises(TandemError):
        runtime_cli.ensure_pixi("Heavy", ask=False, allowed=False)
    assert "adds ~/.pixi/bin to your shell's rc file" in said()


# --- a new plugin is installed where tandem runs ------------------------------------------------------------------


def test_planners_new_names_the_install_for_how_tandem_is_installed(tmp_path, monkeypatch):
    from tandem.cli import planners as planners_cli

    venv = tmp_path / "venvs" / "tandem-tamp"
    venv.mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(venv))
    (venv / "pipx_metadata.json").write_text("{}")
    install, test = planners_cli.install_into_tandem()
    assert install.startswith("pipx inject tandem-tamp --editable .") and test.endswith("-m pytest")
    (venv / "pipx_metadata.json").unlink()
    (venv / "uv-receipt.toml").write_text("")
    assert "--with-editable ." in planners_cli.install_into_tandem()[0]
    (venv / "uv-receipt.toml").unlink()
    install, test = planners_cli.install_into_tandem()
    assert install == f'{sys.executable} -m pip install -e ".[test]"' and test == f"{sys.executable} -m pytest"

    readme = (Path(planners_cli.__file__).parents[1] / "resources/scaffold/README.md.tmpl").read_text()
    assert "pipx inject tandem-tamp --editable ." in readme and "--with-editable ." in readme


# --- a planner's preset states planner.options and nothing else -----------------------------------------------------


def test_a_planner_preset_may_not_override_tandems_phase_planning_or_state_cameras(tmp_path):
    from tandem.core import presets
    from tandem.planners.testing import ConformanceError, check_presets

    directory = tmp_path / "presets"
    directory.mkdir()
    path = directory / "paper.yml"
    path.write_text(
        "title: Mine\nextends: paper\nprofile:\n  hitl: {verify_final_phase: false}\n"
        "  cameras: {perception: hand}\n  planner: {options: {items: [duck]}}\n"
    )
    registry.register_backend("toy", type("Toy", (ToyPlanner,), {"__module__": __name__, "presets_dir": directory}))
    with pytest.raises(TandemError) as caught:
        presets.load(path, origin="toy")
    # Cameras are no profile's at all now: the rig's.
    assert "'hitl'" in caught.value.message and "profile.cameras is not a profile setting" in caught.value.message
    with pytest.raises(TandemError):
        presets.available("toy")
    with pytest.raises(ConformanceError):
        check_presets("toy")
