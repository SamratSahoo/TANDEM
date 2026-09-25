"""A pin names the branch its commit was taken from, and everything that shows or records a pin says it.

TiPToP's tiptop and cuTAMP are pinned to the TANDEM branches of SamratSahoo's forks, not to their mains,
so "6820474" alone no longer tells anyone where to look for the next commit. ``SourcePin.ref`` carries the
branch: shown by ``tandem planners info`` and ``tandem runtime status``, written into the runtime's record
(``.tandem-runtime.json``) and into every bundle's marker, and used by a fetch whose server will not hand
out a bare commit. The commit stays the only thing installed and compared.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from tandem.core.errors import TandemError
from tandem.planners import runtime as rt_mod
from tandem.planners.base import RuntimeStatus, SourcePin
from tandem.planners.runtime import RecipeRuntime, RuntimeRecipe, Source

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="these fixtures build git repositories")

A, B = "a" * 40, "b" * 40


@pytest.fixture(autouse=True)
def no_sources_override(monkeypatch):
    # CI sets this for the static checks; here it would turn every fetch into an offline install.
    monkeypatch.delenv("TANDEM_PLANNER_SOURCES", raising=False)


# --- the pin itself ----------------------------------------------------------------------------------


def test_a_pin_shows_its_branch_beside_its_commit():
    pin = SourcePin("tiptop", "https://github.com/SamratSahoo/tiptop.git", A, ref="TANDEM")
    assert pin.label() == "aaaaaaa (TANDEM)"
    assert pin.to_dict() == {"name": "tiptop", "url": pin.url, "commit": A, "ref": "TANDEM"}
    bare = SourcePin("curobo", "u", B)
    assert bare.label() == "bbbbbbb" and bare.to_dict()["ref"] is None


def test_the_branch_takes_no_part_in_comparing_pins():
    # A runtime recorded before pins named a branch is at the same pin as ever: only the commit is installed.
    assert SourcePin("t", "u", A, ref="TANDEM") == SourcePin("t", "u", A)
    assert SourcePin("t", "u", A, ref="TANDEM") != SourcePin("t", "u", B, ref="TANDEM")
    status = RuntimeStatus(installed=True, pins=(SourcePin("t", "u", A),))
    assert status.mismatched((SourcePin("t", "u", A, ref="TANDEM"),)) == ()


@pytest.mark.parametrize("ref", ["TANDEM", "main", "feat/placement", "release-0.2", "v1.x"])
def test_a_recipe_takes_a_branch_name(ref):
    RuntimeRecipe(planner="p", sources=(Source(SourcePin("s", "u", A, ref=ref)),))


@pytest.mark.parametrize(
    "ref",
    [
        "--upload-pack=touch x",  # an option, handed to git fetch
        "-b",
        "a..b",
        "x y",
        "main.lock",
        "feat/.hidden",
        "feat//x",
        "trailing/",
        "refs:heads",
        "c" * 40,  # a commit: that is what `commit` is for
    ],
)
def test_a_recipe_refuses_what_is_not_a_branch_name(ref):
    with pytest.raises(TandemError, match="is not a branch name"):
        RuntimeRecipe(planner="p", sources=(Source(SourcePin("s", "u", A, ref=ref)),))


# --- TiPToP's pins -----------------------------------------------------------------------------------


def test_every_tiptop_pin_names_its_branch():
    from tandem.planners.tiptop.recipe import RECIPE

    assert {pin.name: pin.ref for pin in RECIPE.pins} == {
        "tiptop": "TANDEM",
        "cuTAMP": "TANDEM",
        "curobo": "main",
    }
    # tiptop and cuTAMP move together (recipe.py says why), so they follow the same branch.
    assert RECIPE.source("tiptop").pin.ref == RECIPE.source("cuTAMP").pin.ref


def test_planners_info_shows_each_pins_branch(monkeypatch):
    from tandem.cli.app import app
    from tandem.planners.tiptop.recipe import RECIPE

    monkeypatch.setenv("COLUMNS", "200")
    result = CliRunner().invoke(app, ["planners", "info", "tiptop", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert {pin["name"]: pin["ref"] for pin in payload["sources"]} == {p.name: p.ref for p in RECIPE.pins}

    result = CliRunner().invoke(app, ["planners", "info", "tiptop"])
    assert result.exit_code == 0, result.output
    assert "branch" in result.output and "TANDEM" in result.output


def test_runtime_status_shows_each_pins_branch(monkeypatch):
    from tandem.cli.app import app
    from tandem.planners.tiptop.recipe import RECIPE

    monkeypatch.setenv("COLUMNS", "200")
    result = CliRunner().invoke(app, ["runtime", "status", "--json", "--planner", "tiptop"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert {s["name"]: s["ref"] for s in payload["sources"]} == {p.name: p.ref for p in RECIPE.pins}
    assert {pin["name"]: pin["ref"] for pin in payload["wanted"]} == {p.name: p.ref for p in RECIPE.pins}

    result = CliRunner().invoke(app, ["runtime", "status", "--planner", "tiptop"])
    assert result.exit_code == 0, result.output
    assert "branch" in result.output and "TANDEM" in result.output


# --- fetched, recorded, bundled ------------------------------------------------------------------------


def _git(*args: str, cwd: Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    base = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    return subprocess.run(
        ["git", *args], cwd=cwd, env={**base, **(env or {})}, check=False, capture_output=True, text=True
    )


@pytest.fixture
def fork(tmp_path):
    """A fork with a TANDEM branch two commits long, pinned at the first: a pin behind its branch's tip."""
    work = tmp_path / "fork-work"
    work.mkdir()
    _git("init", "-q", "-b", "TANDEM", cwd=work)
    commits = []
    for n in (1, 2):
        (work / "pkg").mkdir(exist_ok=True)
        (work / "pkg" / "__init__.py").write_text(f"VERSION = {n}\n")
        _git("add", "-A", cwd=work)
        _git("-c", "commit.gpgsign=false", "commit", "-q", "-m", f"commit {n}", cwd=work)
        commits.append(_git("rev-parse", "HEAD", cwd=work).stdout.strip())
    bare = tmp_path / "fork.git"
    _git("clone", "-q", "--bare", str(work), str(bare))
    return SimpleNamespace(url=bare.as_uri(), first=commits[0], second=commits[1])


def _recipe(fork, ref: str = "TANDEM") -> RuntimeRecipe:
    return RuntimeRecipe(
        planner="toy",
        sources=(Source(SourcePin("toy", fork.url, fork.first, ref=ref), marker="pkg/__init__.py"),),
    )


@needs_git
def test_the_runtime_records_the_branch_it_installed_from(fork, tmp_path):
    rt = RecipeRuntime(_recipe(fork), tmp_path / "runtime")
    lines: list[str] = []
    rt.fetch(log=lines.append)
    entry = rt.record()["sources"]["toy"]
    assert entry["commit"] == fork.first and entry["ref"] == "TANDEM" and entry["verified"] is True
    assert (rt.root / "toy" / "pkg" / "__init__.py").read_text() == "VERSION = 1\n", "the pin, not the tip"
    assert any(f"installed at {fork.first[:7]} (TANDEM)" in line for line in lines)
    (pin,) = rt.status().pins
    assert pin.ref == "TANDEM" and pin.commit == fork.first
    (state,) = rt.inspect().sources
    assert state.ref == "TANDEM"


def _refuses_bare_commits(fork, tmp_path) -> dict:
    """An environment in which git's own protocol v0 is spoken: it refuses a want for a commit that is not
    a branch tip, as a server with uploadpack.allowReachableSHA1InWant off does. Skips where that cannot
    be arranged, rather than pass without exercising the fallback."""
    env = {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "protocol.version", "GIT_CONFIG_VALUE_0": "0"}
    probe = tmp_path / "probe.git"
    _git("init", "-q", "--bare", str(probe))
    fetched = _git("-C", str(probe), "fetch", "--depth", "1", "-q", fork.url, fork.first, env=env)
    if fetched.returncode == 0:
        pytest.skip("this git hands out a bare commit even over protocol v0 (or ignores GIT_CONFIG_COUNT)")
    return env


@needs_git
def test_a_server_that_will_not_hand_out_a_bare_commit_is_asked_for_the_branch(fork, tmp_path, monkeypatch):
    for key, value in _refuses_bare_commits(fork, tmp_path).items():
        monkeypatch.setenv(key, value)
    rt = RecipeRuntime(_recipe(fork), tmp_path / "runtime")
    lines: list[str] = []
    rt.fetch(log=lines.append)
    assert any("fetching its TANDEM branch to find it there" in line for line in lines), lines
    entry = rt.record()["sources"]["toy"]
    assert entry["commit"] == fork.first and entry["origin"] == "git" and entry["verified"] is True
    assert (rt.root / "toy" / "pkg" / "__init__.py").read_text() == "VERSION = 1\n"


@needs_git
def test_a_pin_its_branch_does_not_hold_is_named_as_such(fork, tmp_path, monkeypatch):
    for key, value in _refuses_bare_commits(fork, tmp_path).items():
        monkeypatch.setenv(key, value)
    recipe = RuntimeRecipe(
        planner="toy",
        sources=(Source(SourcePin("toy", fork.url, "d" * 40, ref="TANDEM"), marker="pkg/__init__.py"),),
    )
    with pytest.raises(TandemError, match="does not have commit d{40} on its TANDEM branch") as caught:
        RecipeRuntime(recipe, tmp_path / "runtime").fetch()
    assert "pushed to the TANDEM branch" in (caught.value.hint or "")


@needs_git
def test_a_branch_that_is_not_there_is_said_to_be_missing(fork, tmp_path, monkeypatch):
    for key, value in _refuses_bare_commits(fork, tmp_path).items():
        monkeypatch.setenv(key, value)
    with pytest.raises(TandemError, match="nor the unpushed branch") as caught:
        RecipeRuntime(_recipe(fork, ref="unpushed"), tmp_path / "runtime").fetch()
    assert "unpushed is pushed there" in (caught.value.hint or "")


@needs_git
def test_without_a_branch_a_refused_commit_is_simply_an_error(fork, tmp_path, monkeypatch):
    for key, value in _refuses_bare_commits(fork, tmp_path).items():
        monkeypatch.setenv(key, value)
    recipe = RuntimeRecipe(planner="toy", sources=(Source(SourcePin("toy", fork.url, fork.first)),))
    with pytest.raises(TandemError, match="git failed fetching"):
        RecipeRuntime(recipe, tmp_path / "runtime").fetch()


@needs_git
def test_a_bundle_marker_names_the_branch(fork, tmp_path, monkeypatch):
    from tandem.planners import bundle as bundle_mod

    recipe = _recipe(fork)
    monkeypatch.setattr(bundle_mod, "recipe_of", lambda planner: recipe)
    lines: list[str] = []
    markers = bundle_mod.bundle_sources("toy", tmp_path / "bundle", log=lines.append)
    assert markers["toy"]["ref"] == "TANDEM" and markers["toy"]["commit"] == fork.first
    on_disk = json.loads((tmp_path / "bundle" / "toy" / rt_mod.SOURCE_MARKER).read_text())
    assert on_disk["ref"] == "TANDEM"
    assert any(f"{fork.first[:7]} (TANDEM) ->" in line for line in lines)

    # And an install from that bundle records the branch the recipe names, verified by commit.
    rt = RecipeRuntime(recipe, tmp_path / "runtime")
    rt.fetch(sources_dir=tmp_path / "bundle")
    entry = rt.record()["sources"]["toy"]
    assert entry["ref"] == "TANDEM" and entry["verified"] is True
