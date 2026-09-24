"""Offline bundles, build failures and the planner-source checks, where the review found them wrong.

- Every `--sources` hint pointed at ``python tools/bundle.py``, which a pip or pipx install does not have;
  and a bundle has to be made by the same tandem that installs from it. It is `tandem planners bundle`.
- An offline install was described as needing no network, while `pixi install` still solves the
  environment from conda-forge and PyPI.
- A failed build said "re-run `tandem runtime build`" -- which builds the ACTIVE profile's planner.
- The planner-source tests used an installed runtime whatever commits it was built at, and the CI job
  that exists to run them could pass with them skipped.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import test_runtime_recipe
from helpers import isolate_registry
from test_catalog_cli import RuntimeFactory, StubRuntime
from test_runtime_recipe import _ToyFactory, toy
from typer.testing import CliRunner

from tandem.cli.app import app
from tandem.core.errors import TandemError
from tandem.planners import registry
from tandem.planners import runtime as rt_mod
from tandem.planners.runtime import RecipeRuntime

# The recipe tests' own fixtures: a git upstream with two commits, a shipped patch, a stand-in pixi.
upstream = test_runtime_recipe.upstream
shipped = test_runtime_recipe.shipped
fake_pixi = test_runtime_recipe.fake_pixi


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    isolate_registry(monkeypatch)
    monkeypatch.delenv("TANDEM_PLANNER_SOURCES", raising=False)
    monkeypatch.delenv("TANDEM_VENDOR_DIR", raising=False)


def _run(*args: str):
    return CliRunner().invoke(app, list(args))


def _refuse_network(monkeypatch) -> None:
    def refuse(*_args, **_kwargs):
        raise AssertionError("nothing may be fetched here")

    monkeypatch.setattr(rt_mod, "export_pinned", refuse)


# --- the bundler ships with tandem ----------------------------------------------------------------------


def test_planners_bundle_makes_a_directory_an_offline_install_verifies(upstream, shipped, tmp_path, monkeypatch):
    recipe = toy(upstream, shipped)
    registry.register_backend("toy", _ToyFactory(recipe, tmp_path / "toy-runtime"))
    out = tmp_path / "bundle"

    made = _run("planners", "bundle", "toy", "--out", str(out), "--from", f"toy={upstream.work}", "--archive")
    assert made.exit_code == 0, made.output
    marker = json.loads((out / "toy" / rt_mod.SOURCE_MARKER).read_text())
    assert marker["commit"] == upstream.first and marker["planner"] == "toy"
    assert (tmp_path / "bundle.tar.gz").is_file()
    assert "tandem planners install toy --sources" in " ".join(made.output.split())

    _refuse_network(monkeypatch)
    rt = RecipeRuntime(recipe, tmp_path / "toy-runtime")
    rt.fetch(sources_dir=out)
    assert rt.record()["sources"]["toy"]["verified"] is True


def test_a_partial_bundle_says_so_instead_of_offering_the_install(tmp_path, monkeypatch):
    from tandem.planners.base import SourcePin
    from tandem.planners.runtime import RuntimeRecipe, Source

    recipe = RuntimeRecipe(
        planner="two",
        sources=(Source(SourcePin("a", "https://example.com/a.git", "a" * 40)), Source(SourcePin("b", "https://example.com/b.git", "b" * 40))),
    )
    registry.register_backend("two", _ToyFactory(recipe, tmp_path / "rt"))
    registry._registered["two"].info = registry._registered["two"].info.__class__(name="two", display_name="Two", sources=recipe.pins)

    def fake_export(pin, dest, *, scratch, log=None):
        dest.mkdir(parents=True)
        (dest / "x").write_text(pin.name)
        return {"origin": "stand-in", "verified": True}

    monkeypatch.setattr(rt_mod, "export_pinned", fake_export)
    made = _run("planners", "bundle", "two", "--out", str(tmp_path / "b"), "--only", "a")
    assert made.exit_code == 0, made.output
    said = " ".join(made.output.split())
    assert "does not hold b yet" in said and "tandem planners install two --sources" not in said


def test_no_hint_sends_anyone_to_a_file_the_wheel_does_not_ship():
    for command in (["planners", "install", "--help"], ["runtime", "build", "--help"]):
        # The help is drawn in a box: its borders out, and its wrapped lines joined back up.
        shown = " ".join("".join(c for c in _run(*command).output if c not in "│╭╮╰╯─").split())
        assert "tools/bundle.py" not in shown and "planners bundle" in shown, command
        assert "conda-forge" in shown, "that the environment is still downloaded is said"


def test_a_bundle_for_another_version_says_how_to_make_the_right_one(upstream, shipped, tmp_path, monkeypatch):
    _refuse_network(monkeypatch)
    sources = tmp_path / "sources"
    rt_mod._export_with_git(str(upstream.bare), upstream.second, sources / "toy", scratch=tmp_path / "s", fetch=True)
    (sources / "toy" / rt_mod.SOURCE_MARKER).write_text(json.dumps({"name": "toy", "commit": upstream.second}))
    with pytest.raises(TandemError) as caught:
        RecipeRuntime(toy(upstream, shipped), tmp_path / "rt").fetch(sources_dir=sources)
    assert "tandem planners bundle" in caught.value.hint
    assert upstream.first in caught.value.hint and str(upstream.bare) in caught.value.hint


# --- an offline install still downloads the environment, and says so ---------------------------------------------


def test_installing_from_a_sources_directory_says_the_environment_is_still_downloaded(
    upstream, shipped, tmp_path, fake_pixi
):
    sources = tmp_path / "sources"
    sources.mkdir()
    import shutil

    shutil.copytree(upstream.work, sources / "toy", symlinks=True)
    lines: list[str] = []
    RecipeRuntime(toy(upstream, shipped), tmp_path / "rt").install(on_progress=lines.append, sources_dir=sources)
    assert any("still solved and downloaded" in line for line in lines), lines


# --- a failed build names the command that re-runs THAT build, and its log ----------------------------------------


class _Failing(StubRuntime):
    def install(self, *, on_progress=None, sources_dir=None, force=False) -> None:
        raise TandemError("pixi install failed (exit 3).", hint="The full build log is in the log directory.")


def test_a_failed_install_points_at_planners_install_and_the_exact_log(profile, tmp_path):
    registry.register_backend("broke", RuntimeFactory("broke", _Failing(tmp_path / "rt")))
    sources = tmp_path / "sources"
    sources.mkdir()
    result = _run("planners", "install", "broke", "--yes", "--sources", str(sources))
    assert result.exit_code == 1
    hint = result.exception.hint
    assert f"tandem planners install broke --sources {sources}" in hint
    assert "runtime build" not in hint and "runtime-build-" in hint


def test_a_failing_build_step_no_longer_names_the_active_profiles_rebuild(tmp_path):
    with pytest.raises(TandemError) as caught:
        rt_mod._stream(["false"], cwd=tmp_path, env={}, log=None, what="pixi install")
    assert "runtime build" not in (caught.value.hint or "")


# --- the planner-source checks: an outdated runtime is skipped, and the CI job never skips ---------------------------


def _runtime(root: Path, commits: dict[str, str]) -> Path:
    (root / "tiptop" / "tiptop").mkdir(parents=True)
    (root / "tiptop" / "tiptop" / "tiptop_run.py").write_text("")
    (root / "cuTAMP" / "cutamp").mkdir(parents=True)
    record = {"format": 2, "planner": "tiptop", "sources": {n: {"url": "u", "commit": c} for n, c in commits.items()}}
    (root / rt_mod.MANIFEST_FILE).write_text(json.dumps(record))
    return root


@pytest.fixture
def reload_sources(monkeypatch):
    import planner_sources

    monkeypatch.delenv(planner_sources.ENV, raising=False)

    def reload(root: Path):
        monkeypatch.setenv("TANDEM_RUNTIME_DIR", str(root))
        return importlib.reload(planner_sources)

    yield reload
    monkeypatch.delenv("TANDEM_RUNTIME_DIR", raising=False)
    importlib.reload(planner_sources)


def test_an_outdated_installed_runtime_is_skipped_and_a_current_one_used(tmp_path, reload_sources):
    from tandem.planners.tiptop.recipe import RECIPE

    old = reload_sources(_runtime(tmp_path / "old", {"tiptop": "4db8f92671b4" + "0" * 28, "cuTAMP": "7b0aeaea452f" + "0" * 28}))
    with pytest.raises(pytest.skip.Exception, match="outdated"):
        old.planner_sources("tiptop/tiptop/tiptop_run.py", "cuTAMP/cutamp")

    current = reload_sources(_runtime(tmp_path / "new", {pin.name: pin.commit for pin in RECIPE.pins}))
    assert current.planner_sources("tiptop/tiptop/tiptop_run.py", "cuTAMP/cutamp") == tmp_path / "new"


def test_with_the_sources_variable_set_a_check_that_would_skip_fails_instead(tmp_path, monkeypatch):
    import planner_sources
    import test_planners

    root = tmp_path / "bundle"
    (root / "tiptop" / "tiptop" / "hitl").mkdir(parents=True)
    (root / "tiptop" / "tiptop" / "tiptop_run.py").write_text("")
    (root / "cuTAMP" / "cutamp").mkdir(parents=True)
    (root / "cuTAMP" / "cutamp" / "tamp_domain.py").write_text("import torch_not_installed_xyz\n")

    monkeypatch.setenv(planner_sources.ENV, str(root))
    for check in (
        test_planners.test_every_symbol_the_sidecar_imports_exists_in_the_pinned_planner,
        test_planners.test_the_sidecar_calls_the_planner_with_the_right_arguments,
        test_planners.test_the_declaration_matches_the_real_cutamp_domain,
    ):
        with pytest.raises(pytest.fail.Exception):
            check()

    monkeypatch.delenv(planner_sources.ENV)
    with pytest.raises(pytest.skip.Exception):
        planner_sources.skip_or_fail("a fork")
