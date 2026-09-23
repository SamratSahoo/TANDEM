"""A planner's runtime from a recipe: fetched at pinned commits, patched loudly, recorded, and rebuilt.

tandem no longer ships any planner's sources; an install fetches them. Everything that can go wrong
with that is tested here against a real git repository on disk -- no network, no pixi, no GPU:

- the tree that lands is exactly the pinned commit, trimmed and patched, and the record says so;
- a patch that does not apply, or that applies to nothing, stops the install;
- moving a pin replaces the tree and keeps the environment, which lives outside it;
- an offline machine installs from a directory of checkouts or exports, and never reaches for the
  network while it does;
- status compares what is installed against what the recipe pins, and says what to do.

The recipe used is a toy with one source, a pixi environment and one build step; pixi is a stand-in
script that records how it was called. TiPToP's own recipe is checked at the end.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from helpers import isolate_registry

from tandem.core.errors import RuntimeNotReady, TandemError
from tandem.planners import runtime as rt_mod
from tandem.planners.base import PlannerInfo, SourcePin
from tandem.planners.runtime import (
    Asset,
    BuildStep,
    PixiEnvironment,
    RecipeRuntime,
    RuntimeRecipe,
    Source,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="these fixtures build git repositories")

CONFIG_V1 = "NAME = 'toy'\nCALIBRATION = 'assets/calibration.json'\n"

# A plain unified diff, the kind tandem ships: no `diff --git` header for `git apply` to filter on.
PATCH = """Let the caller choose the calibration file.

--- a/pkg/config.py
+++ b/pkg/config.py
@@ -1,2 +1,4 @@
+import os
+
 NAME = 'toy'
-CALIBRATION = 'assets/calibration.json'
+CALIBRATION = os.environ.get('TOY_CALIBRATION') or 'assets/calibration.json'
"""


def _git(*args: str, cwd: Path | None = None) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(work: Path, files: dict[str, str], message: str) -> str:
    for rel, text in files.items():
        path = work / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    _git("add", "-A", cwd=work)
    # Signing is a developer's global setting, not the fixture's business.
    _git("-c", "commit.gpgsign=false", "commit", "-q", "-m", message, cwd=work)
    return _git("rev-parse", "HEAD", cwd=work)


@pytest.fixture
def upstream(tmp_path):
    """A planner's upstream repository: two commits, served from a bare clone by path."""
    work = tmp_path / "upstream-work"
    work.mkdir()
    _git("init", "-q", "-b", "main", cwd=work)
    first = _commit(
        work,
        {
            "pixi.toml": "[workspace]\nname = 'toy'\n",
            "pkg/__init__.py": "",
            "pkg/config.py": CONFIG_V1,
            "docs/recording.txt": "x" * 4096,
            "README.md": "toy planner v1\n",
        },
        "one",
    )
    second = _commit(work, {"README.md": "toy planner v2\n", "pkg/new.py": "NEW = True\n"}, "two")
    bare = tmp_path / "upstream.git"
    _git("clone", "-q", "--bare", str(work), str(bare))
    return SimpleNamespace(work=work, bare=bare, first=first, second=second)


@pytest.fixture
def shipped(tmp_path):
    """The files a planner package ships: its patch and a checkpoint."""
    root = tmp_path / "package"
    root.mkdir()
    (root / "0001-toy-calibration.patch").write_text(PATCH)
    (root / "model.pt").write_bytes(b"\x00weights v1")
    return SimpleNamespace(patch=root / "0001-toy-calibration.patch", asset=root / "model.pt")


def toy(
    upstream, shipped, commit: str | None = None, *, patches=None, url: str | None = None
) -> RuntimeRecipe:
    return RuntimeRecipe(
        planner="toy",
        title="Toy",
        sources=(
            Source(
                SourcePin("toy", url or str(upstream.bare), commit or upstream.first),
                trim=("docs",),
                patches=(shipped.patch,) if patches is None else tuple(patches),
                marker="pkg/__init__.py",
            ),
        ),
        environment=PixiEnvironment(manifest="toy/pixi.toml", env={"TOY_VERSION": "{version:toy}"}),
        steps=(
            BuildStep(
                "compile",
                task="compile",
                env={"TOY_SRC": "{source:toy}", "TOY_COMMIT": "{commit:toy}"},
                # Outside the tree, so it outlives a bump and only the record can say it is stale.
                produces=("build/*.so",),
                label="toy kernels",
                done="compiled",
                todo="not compiled",
            ),
        ),
        assets=(Asset(shipped.asset, "weights/model.pt"),),
    )


@pytest.fixture
def fake_pixi(tmp_path, monkeypatch):
    """A stand-in for pixi that does what a successful one leaves behind, and logs how it was called."""
    calls = tmp_path / "pixi-calls.jsonl"
    script = tmp_path / "bin" / "pixi"
    script.parent.mkdir()
    script.write_text(
        f"""#!{sys.executable}
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
manifest = Path(args[args.index("--manifest-path") + 1])
with open({str(calls)!r}, "a") as fh:
    fh.write(json.dumps({{"args": args, "cwd": os.getcwd(), "env": {{k: v for k, v in os.environ.items() if k.startswith("TOY_")}}}}) + "\\n")
if args[0] == "install":
    python = manifest.parent / ".pixi" / "envs" / "default" / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("")
elif args[0] == "run" and os.environ.get("TOY_FAIL_STEP") != "1":
    out = manifest.parent.parent / "build"
    out.mkdir(exist_ok=True)
    (out / "kernel.so").write_text("")
print("pixi: done")
"""
    )
    script.chmod(0o755)
    monkeypatch.setattr(rt_mod, "_find_pixi", lambda: script)
    # `tandem init` asks the probe, and installs pixi into the home directory when it finds none.
    monkeypatch.setattr("tandem.core.probe.find_pixi", lambda: script)

    def read() -> list[dict]:
        return [json.loads(line) for line in calls.read_text().splitlines()] if calls.is_file() else []

    return read


@pytest.fixture(autouse=True)
def no_sources_override(monkeypatch):
    # CI sets this for the static checks; here it would turn every fetch into an offline install.
    monkeypatch.delenv("TANDEM_PLANNER_SOURCES", raising=False)
    monkeypatch.delenv("TANDEM_VENDOR_DIR", raising=False)


@pytest.fixture(autouse=True)
def never_install_pixi(monkeypatch):
    """The real installer is `curl | bash` into ~/.pixi, editing the shell's rc file on the way. A test
    that reached it would do that to the machine running the suite."""

    def refuse(*_args, **_kwargs):
        raise AssertionError("a test reached the real pixi installer")

    monkeypatch.setattr("tandem.cli.runtime.install_pixi", refuse)


# --- fetching ------------------------------------------------------------------------------------------


def test_a_pinned_source_is_fetched_trimmed_patched_and_recorded(upstream, shipped, tmp_path):
    rt = RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime")
    lines: list[str] = []
    assert rt.fetch(log=lines.append) == ["toy"]

    tree = rt.root / "toy"
    assert (tree / "README.md").read_text() == "toy planner v1\n", "the pinned commit, not the branch tip"
    assert not (tree / "pkg" / "new.py").exists()
    assert not (tree / "docs").exists(), "trimmed"
    assert "TOY_CALIBRATION" in (tree / "pkg" / "config.py").read_text(), "patched"
    assert not (tree / ".git").exists(), "exported, so no VCS state rides along"
    assert not (rt.root / rt_mod.STAGING).exists(), "no scratch left behind"

    entry = rt.record()["sources"]["toy"]
    assert entry["commit"] == upstream.first and entry["url"] == str(upstream.bare)
    assert entry["origin"] == "git" and entry["verified"] is True
    assert entry["trimmed"] == ["docs"]
    assert entry["patches"] == [{"name": shipped.patch.name, "sha256": rt_mod._sha256(shipped.patch)}]
    assert any("applied 0001-toy-calibration.patch" in line for line in lines)


def test_an_installed_pin_is_left_alone_unless_forced(upstream, shipped, tmp_path):
    rt = RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime")
    rt.fetch()
    scribble = rt.root / "toy" / "scribble"
    scribble.write_text("left by a build")

    lines: list[str] = []
    assert rt.fetch(log=lines.append) == []
    assert scribble.is_file() and any("already at" in line for line in lines)

    assert rt.fetch(force=True) == ["toy"]
    assert not scribble.exists()


def test_moving_a_pin_replaces_the_tree_and_keeps_the_environment(upstream, shipped, tmp_path, fake_pixi):
    """The environment is ~20 GB of torch and CUDA. A bump must not delete it with the tree it sat in."""
    root = tmp_path / "runtime"
    RecipeRuntime(toy(upstream, shipped), root).install()
    home = root / "env"
    link = root / "toy" / ".pixi"
    assert link.is_symlink() and os.readlink(link) == os.path.join("..", "env")
    assert (home / "envs" / "default" / "bin" / "python").is_file(), "the environment lives outside the tree"
    (home / "envs" / "default" / "precious").write_text("a solved environment")

    bumped = RecipeRuntime(toy(upstream, shipped, upstream.second), root)
    before = bumped.status()
    assert not before.installed and before.mismatched(bumped.recipe.pins) == ("toy",)
    assert any(
        f"toy is at {upstream.first[:7]}, but Toy now pins {upstream.second[:7]}" in p
        for p in before.problems
    )

    lines: list[str] = []
    bumped.fetch(log=lines.append)
    assert (root / "toy" / "pkg" / "new.py").is_file(), "the new commit's tree"
    assert (home / "envs" / "default" / "precious").is_file(), "the environment survived the swap"
    assert link.is_symlink(), "and the new tree points at it again"
    assert any(f"replacing {upstream.first[:7]} with {upstream.second[:7]}" in line for line in lines)

    # Everything is on disk, but it was built for the old commit: rebuild, don't run.
    after = bumped.inspect()
    assert after.problems == ("the runtime has not been built since toy changed",)
    bumped.install()
    assert bumped.status().installed


def test_a_patch_that_does_not_apply_stops_the_install_and_leaves_the_runtime_as_it_was(
    upstream, shipped, tmp_path
):
    root = tmp_path / "runtime"
    RecipeRuntime(toy(upstream, shipped), root).fetch()
    stale = tmp_path / "0002-stale.patch"
    stale.write_text(PATCH.replace("NAME = 'toy'", "NAME = 'something upstream renamed'"))

    broken = RecipeRuntime(toy(upstream, shipped, upstream.second, patches=[stale]), root)
    with pytest.raises(TandemError, match="0002-stale.patch does not apply to toy") as caught:
        broken.fetch()
    assert "Rewrite it against the new source" in caught.value.hint
    # The old tree is still there, whole, and still recorded as what it is.
    assert (root / "toy" / "README.md").read_text() == "toy planner v1\n"
    assert broken.record()["sources"]["toy"]["commit"] == upstream.first
    assert not (root / rt_mod.STAGING).exists()


def test_a_git_style_patch_applies_even_inside_someone_elses_repository(upstream, shipped, tmp_path):
    """Run from inside a repository, `git apply` drops a `diff --git` patch's hunks outside the current
    prefix and reports success. A runtime under a checkout must not be mistaken for part of it."""
    host = tmp_path / "a-checkout"
    host.mkdir()
    _git("init", "-q", cwd=host)
    git_style = tmp_path / "0001-git-style.patch"
    git_style.write_text("diff --git a/pkg/config.py b/pkg/config.py\n" + PATCH[PATCH.index("--- a/") :])
    rt = RecipeRuntime(toy(upstream, shipped, patches=[git_style]), host / "runtime")
    rt.fetch()
    assert "TOY_CALIBRATION" in (rt.root / "toy" / "pkg" / "config.py").read_text()


def test_git_reporting_a_skipped_patch_or_nothing_applied_is_an_error():
    with pytest.raises(TandemError, match="was skipped rather than applied"):
        rt_mod._check_applied("Skipped patch 'pkg/config.py'.\n", "0001.patch", "toy")
    with pytest.raises(TandemError, match="applied to nothing"):
        rt_mod._check_applied("", "0001.patch", "toy")
    rt_mod._check_applied(
        "Checking patch pkg/config.py...\nApplied patch pkg/config.py cleanly.\n", "0001.patch", "toy"
    )


def test_a_commit_the_repository_does_not_have_is_named(upstream, shipped, tmp_path):
    rt = RecipeRuntime(toy(upstream, shipped, "0" * 40), tmp_path / "runtime")
    with pytest.raises(TandemError, match="0000000"):
        rt.fetch()
    assert not (rt.root / "toy").exists()


def _github_style_archive(upstream, commit: str, tmp_path: Path, *, stamp: bool = True) -> str:
    """A tarball shaped like GitHub's archive of a commit: `git archive` with a <repo>-<sha>/ prefix."""
    out = tmp_path / f"toy-{commit}.tar.gz"
    with open(out, "wb") as fh:
        subprocess.run(
            [
                "git",
                "-C",
                str(upstream.bare),
                "archive",
                "--format=tar.gz",
                f"--prefix=toy-{commit}/",
                commit,
            ],
            stdout=fh,
            check=True,
        )
    if not stamp:
        # Rewrite it without the pax header naming the commit, as a proxy that re-packs archives might.
        plain = tmp_path / "unstamped.tar.gz"
        with tarfile.open(out) as src, tarfile.open(plain, "w:gz", format=tarfile.GNU_FORMAT) as dst:
            for member in src.getmembers():
                dst.addfile(member, src.extractfile(member) if member.isfile() else None)
        out = plain
    return out.as_uri()


def test_without_git_the_github_archive_is_fetched_and_its_commit_checked(
    upstream, shipped, tmp_path, monkeypatch
):
    if shutil.which("patch") is None:
        pytest.skip("patch(1) is not installed, and without git it is what applies the recipe's patches")
    archive = _github_style_archive(upstream, upstream.first, tmp_path)
    real_which = rt_mod._which
    monkeypatch.setattr(rt_mod, "_which", lambda tool: None if tool == "git" else real_which(tool))
    monkeypatch.setattr(rt_mod, "_github_archive_url", lambda url, commit: archive)

    rt = RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime")
    rt.fetch()
    tree = rt.root / "toy"
    assert (tree / "README.md").read_text() == "toy planner v1\n"
    assert not (tree / f"toy-{upstream.first}").exists(), "the archive's wrapper directory is stripped"
    assert "TOY_CALIBRATION" in (tree / "pkg" / "config.py").read_text(), "patched with patch(1)"
    assert rt.record()["sources"]["toy"]["origin"] == "archive"


def test_an_archive_of_some_other_commit_is_refused(upstream, shipped, tmp_path, monkeypatch):
    real_which = rt_mod._which
    monkeypatch.setattr(rt_mod, "_which", lambda tool: None if tool == "git" else real_which(tool))
    wrong = _github_style_archive(upstream, upstream.second, tmp_path)
    monkeypatch.setattr(rt_mod, "_github_archive_url", lambda url, commit: wrong)
    with pytest.raises(TandemError, match=f"archive is commit {upstream.second[:7]}, not the pinned"):
        RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime").fetch()

    # With the header stripped, the wrapper directory's name is all that says which commit it is.
    unstamped = _github_style_archive(upstream, upstream.second, tmp_path, stamp=False)
    monkeypatch.setattr(rt_mod, "_github_archive_url", lambda url, commit: unstamped)
    with pytest.raises(TandemError, match="Cannot confirm the archive is commit"):
        RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime2").fetch()


def test_when_git_cannot_fetch_from_github_the_archive_is_tried(upstream, shipped, tmp_path, monkeypatch):
    """Some networks pass HTTPS downloads and break git's protocol. The same commit comes either way."""
    archive = _github_style_archive(upstream, upstream.first, tmp_path)
    real = rt_mod._export_with_git

    def git_blocked(repository, commit, dest, *, scratch, fetch):
        if fetch:
            (dest / "half-extracted").mkdir(parents=True)
            raise TandemError("git failed fetching: the remote end hung up unexpectedly")
        return real(repository, commit, dest, scratch=scratch, fetch=fetch)

    monkeypatch.setattr(rt_mod, "_export_with_git", git_blocked)
    monkeypatch.setattr(rt_mod, "_github_archive_url", lambda url, commit: archive)
    rt = RecipeRuntime(toy(upstream, shipped, url="https://github.com/example/toy.git"), tmp_path / "runtime")
    lines: list[str] = []
    rt.fetch(log=lines.append)
    assert any("trying GitHub's archive" in line for line in lines)
    assert rt.record()["sources"]["toy"]["origin"] == "archive"
    assert not (rt.root / "toy" / "half-extracted").exists()


def test_without_git_a_repository_off_github_cannot_be_fetched(upstream, shipped, tmp_path, monkeypatch):
    monkeypatch.setattr(rt_mod, "_which", lambda tool: None)
    with pytest.raises(TandemError, match="git is not installed, and .* is not on GitHub"):
        RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime").fetch()


def test_github_urls_map_to_the_archive_of_the_commit():
    sha = "a" * 40
    for url in (
        "https://github.com/SamratSahoo/tiptop.git",
        "https://github.com/SamratSahoo/tiptop",
        "git@github.com:SamratSahoo/tiptop.git",
    ):
        assert (
            rt_mod._github_archive_url(url, sha)
            == f"https://github.com/SamratSahoo/tiptop/archive/{sha}.tar.gz"
        )
    assert rt_mod._github_archive_url("https://gitlab.com/x/y.git", sha) is None


def test_an_archive_entry_outside_its_own_tree_is_refused(tmp_path):
    evil = tmp_path / "evil.tar"
    with tarfile.open(evil, "w") as tar:
        data = b"gotcha"
        info = tarfile.TarInfo("../outside.txt")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    with pytest.raises(TandemError, match="outside its own tree"):
        rt_mod.extract(evil, tmp_path / "dest")
    assert not (tmp_path / "outside.txt").exists()

    link = tmp_path / "link.tar"
    with tarfile.open(link, "w") as tar:
        info = tarfile.TarInfo("escape")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../etc"
        tar.addfile(info)
    with pytest.raises(TandemError, match="link out of its own tree"):
        rt_mod.extract(link, tmp_path / "dest2")


# --- installing from a directory instead of the network -----------------------------------------------


def _no_network(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("a sources directory is in force; nothing may be fetched")

    monkeypatch.setattr(rt_mod, "export_pinned", refuse)


def test_a_checkout_in_the_sources_directory_gives_the_pinned_commit_whatever_it_has_checked_out(
    upstream, shipped, tmp_path, monkeypatch
):
    _no_network(monkeypatch)
    sources = tmp_path / "sources"
    sources.mkdir()
    shutil.copytree(upstream.work, sources / "toy", symlinks=True)  # checked out at the SECOND commit
    (sources / "toy" / "README.md").write_text("an uncommitted edit\n")

    rt = RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime")
    rt.fetch(sources_dir=sources)
    assert (rt.root / "toy" / "README.md").read_text() == "toy planner v1\n"
    entry = rt.record()["sources"]["toy"]
    assert entry["verified"] is True and entry["origin"].startswith("checkout ")


def test_an_export_from_a_bundle_must_be_the_pinned_commit(upstream, shipped, tmp_path, monkeypatch):
    _no_network(monkeypatch)
    sources = tmp_path / "sources"
    # What tools/bundle.py leaves: an export of a commit, with a marker naming it.
    rt_mod._export_with_git(
        str(upstream.bare), upstream.second, sources / "toy", scratch=tmp_path / "s", fetch=True
    )
    marker = sources / "toy" / rt_mod.SOURCE_MARKER
    marker.write_text(json.dumps({"name": "toy", "commit": upstream.second}))

    with pytest.raises(
        TandemError, match=f"is toy at {upstream.second[:7]}, but the recipe pins {upstream.first[:7]}"
    ):
        RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime").fetch(sources_dir=sources)

    rt = RecipeRuntime(toy(upstream, shipped, upstream.second), tmp_path / "runtime")
    rt.fetch(sources_dir=sources)
    assert rt.record()["sources"]["toy"]["verified"] is True
    assert not (rt.root / "toy" / rt_mod.SOURCE_MARKER).exists(), (
        "the bundle's marker is not part of the tree"
    )


def test_an_unmarked_directory_is_taken_on_trust_and_says_so(upstream, shipped, tmp_path, monkeypatch):
    _no_network(monkeypatch)
    sources = tmp_path / "sources"
    rt_mod._export_with_git(
        str(upstream.bare), upstream.first, sources / "toy", scratch=tmp_path / "s", fetch=True
    )
    (sources / "toy" / "pkg" / "__pycache__").mkdir()
    (sources / "toy" / "pkg" / "__pycache__" / "junk.pyc").write_text("")

    rt = RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime")
    lines: list[str] = []
    rt.fetch(sources_dir=sources, log=lines.append)
    assert any("on trust" in line for line in lines)
    assert rt.record()["sources"]["toy"]["verified"] is False
    assert not (rt.root / "toy" / "pkg" / "__pycache__").exists(), "a working tree's build junk is not copied"
    assert any("was not verified" in note for note in rt.inspect().notes)


def test_a_source_missing_from_the_sources_directory_is_an_error_not_a_download(
    upstream, shipped, tmp_path, monkeypatch
):
    _no_network(monkeypatch)
    (tmp_path / "sources").mkdir()
    with pytest.raises(TandemError, match="toy is not in the planner sources directory"):
        RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime").fetch(sources_dir=tmp_path / "sources")
    with pytest.raises(TandemError, match="does not exist"):
        RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime").fetch(sources_dir=tmp_path / "nowhere")


def test_the_sources_directory_can_come_from_the_environment(upstream, shipped, tmp_path, monkeypatch):
    _no_network(monkeypatch)
    sources = tmp_path / "sources"
    sources.mkdir()
    shutil.copytree(upstream.work, sources / "toy", symlinks=True)
    monkeypatch.setenv("TANDEM_PLANNER_SOURCES", str(sources))
    rt = RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime")
    rt.fetch()
    assert rt.record()["sources"]["toy"]["origin"] == f"checkout {sources / 'toy'}"


# --- building, status, and what the record says -------------------------------------------------------


def test_status_from_nothing_to_ready(upstream, shipped, tmp_path, fake_pixi):
    rt = RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime")

    absent = rt.status()
    assert not absent.installed and absent.detail == "not created"
    assert absent.problems == ("the runtime has not been created yet",)
    assert absent.mismatched(rt.recipe.pins) == ("toy",)

    rt.fetch()
    rt.place_assets()
    fetched = rt.status()
    assert fetched.detail == "sources present · pixi env not built · toy kernels not compiled"
    assert fetched.problems == ("the pixi environment has not been created", "toy kernels not compiled")
    assert fetched.pins == rt.recipe.pins and fetched.mismatched(rt.recipe.pins) == ()
    with pytest.raises(RuntimeNotReady, match="Toy runtime .* is not ready"):
        rt.require_ready()

    rt.install()
    ready = rt.status()
    assert ready.installed and ready.problems == ()
    assert ready.detail.startswith("sources present · pixi env built · toy kernels compiled · built ")
    rt.require_ready()
    assert rt.python() == rt.root / "env" / "envs" / "default" / "bin" / "python"


def test_install_solves_the_environment_then_runs_each_step_with_the_recipes_variables(
    upstream, shipped, tmp_path, fake_pixi
):
    rt = RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime")
    steps: list[tuple[str, str]] = []
    rt.install(on_step=lambda key, text: steps.append((key, text)))

    assert [key for key, _ in steps] == ["sources", "environment", "compile"] == [k for k, _ in rt.plan()]
    install, run = fake_pixi()
    manifest = str(rt.root / "toy" / "pixi.toml")
    assert install["args"] == ["install", "--manifest-path", manifest]
    assert run["args"] == ["run", "--manifest-path", manifest, "compile"]
    assert Path(run["cwd"]).resolve() == (rt.root / "toy").resolve()
    assert run["env"] == {
        "TOY_VERSION": f"0.0.0+g{upstream.first[:7]}",
        "TOY_SRC": str(rt.root / "toy"),
        "TOY_COMMIT": upstream.first,
    }
    assert rt.record()["built"]["sources"] == {"toy": upstream.first}

    # Solving again is pixi's business to make cheap; tandem just asks. The sources are not refetched.
    rt.install()
    assert len(fake_pixi()) == 4


def test_env_only_stops_once_the_environment_is_solved(upstream, shipped, tmp_path, fake_pixi):
    rt = RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime")
    rt.install(env_only=True)
    assert [call["args"][0] for call in fake_pixi()] == ["install"]
    assert rt.inspect().environment_built and not rt.status().installed


def test_an_install_with_no_pixi_stops_before_fetching_anything(upstream, shipped, tmp_path, monkeypatch):
    monkeypatch.setattr(rt_mod, "_find_pixi", lambda: None)
    rt = RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime")
    with pytest.raises(TandemError, match="pixi is not installed, and the Toy runtime is built in a pixi"):
        rt.install()
    assert not rt.root.exists()


def test_an_install_that_leaves_something_missing_says_what(
    upstream, shipped, tmp_path, fake_pixi, monkeypatch
):
    monkeypatch.setenv("TOY_FAIL_STEP", "1")
    with pytest.raises(TandemError, match="still looks incomplete: toy kernels not compiled"):
        RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime").install()


def test_a_command_is_run_inside_the_environment(upstream, shipped, tmp_path, fake_pixi):
    rt = RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime")
    argv = rt.command(["python", "sidecar.py"])
    assert argv[1:] == ["run", "--manifest-path", str(rt.root / "toy" / "pixi.toml"), "python", "sidecar.py"]
    assert rt.workdir == rt.root / "toy"


def test_an_environment_left_inside_an_old_tree_is_moved_out_not_deleted(upstream, shipped, tmp_path):
    """Runtimes built before the environment had a home of its own kept it at <tree>/.pixi."""
    root = tmp_path / "runtime"
    RecipeRuntime(toy(upstream, shipped, patches=[]), root).fetch()
    (root / "toy" / ".pixi").unlink()
    legacy = root / "toy" / ".pixi" / "envs" / "default"
    (legacy / "bin").mkdir(parents=True)
    (legacy / "bin" / "python").write_text("")
    (legacy / "precious").write_text("twenty minutes of solving")
    shutil.rmtree(root / "env")
    # It works where it is, so it counts as built: an upgrade of tandem alone forces no rebuild.
    assert RecipeRuntime(toy(upstream, shipped, patches=[]), root).inspect().environment_built

    lines: list[str] = []
    RecipeRuntime(toy(upstream, shipped, upstream.second, patches=[]), root).fetch(log=lines.append)
    assert (root / "env" / "envs" / "default" / "precious").is_file()
    assert (root / "toy" / ".pixi").is_symlink()
    assert any("moved out of the source tree" in line for line in lines)


def test_assets_are_placed_and_replaced_when_the_package_ships_new_ones(upstream, shipped, tmp_path):
    rt = RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime")
    rt.place_assets()
    placed = rt.root / "weights" / "model.pt"
    assert placed.read_bytes() == b"\x00weights v1"

    shipped.asset.write_bytes(b"\x00weights v2")
    rt.place_assets()
    assert placed.read_bytes() == b"\x00weights v2"
    assert rt.record()["assets"]["weights/model.pt"]["sha256"] == rt_mod._sha256(shipped.asset)


def test_a_file_the_package_should_ship_but_does_not_is_named(upstream, shipped, tmp_path):
    shipped.patch.unlink()
    with pytest.raises(TandemError, match="missing files its runtime needs") as caught:
        RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime").fetch()
    assert shipped.patch.name in caught.value.message


def test_an_edited_patch_counts_as_a_different_patch(upstream, shipped, tmp_path):
    rt = RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime")
    rt.fetch()
    shipped.patch.write_text(PATCH.replace("TOY_CALIBRATION", "TOY_CALIBRATION_FILE"))
    assert "toy was patched differently from what the recipe applies now" in rt.inspect().problems
    assert rt.fetch() == ["toy"], "and an install re-exports the tree to apply it"
    assert "TOY_CALIBRATION_FILE" in (rt.root / "toy" / "pkg" / "config.py").read_text()


def test_a_runtime_recorded_before_sources_were_fetched_is_still_read(upstream, shipped, tmp_path, fake_pixi):
    """The stamp a runtime built from the wheel's own copy of the sources wrote. Its trees came out of
    `git archive` of exactly those commits, so they are current -- nothing is fetched again."""
    rt = RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime")
    rt.install()
    stamp = {
        "vendor": {
            "toy": {
                "url": str(upstream.bare),
                "commit": upstream.first,
                "version": upstream.first[:7],
                "trimmed": ["docs"],
                "patches": [shipped.patch.name],
            },
            "checkpoints": {"files": ["weights/model.pt"]},
        },
        "built_at": "2026-01-01T00:00:00+00:00",
    }
    rt.manifest_file.write_text(json.dumps(stamp))

    st = rt.inspect()
    assert st.ready and st.built_at == "2026-01-01T00:00:00+00:00"
    assert rt.status().pins == rt.recipe.pins
    assert rt.fetch() == [], "a current runtime is not fetched again"


def test_uninstall_deletes_a_runtime_and_nothing_else(upstream, shipped, tmp_path):
    rt = RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime")
    rt.uninstall()  # nothing there: nothing to do

    rt.root.mkdir()
    precious = rt.root / "thesis.tex"
    precious.write_text("four years of work")
    with pytest.raises(TandemError, match="does not look like a Toy runtime"):
        rt.uninstall()
    assert precious.is_file()

    precious.unlink()
    rt.fetch()
    rt.uninstall()
    assert not rt.root.exists()


def test_each_planner_gets_a_runtime_directory_of_its_own(tmp_path, monkeypatch):
    from tandem.core import paths

    assert rt_mod.default_root("toy") == paths.share_dir() / "runtimes" / "toy"
    monkeypatch.setenv("TANDEM_RUNTIMES_DIR", str(tmp_path / "big-disk"))
    assert rt_mod.default_root("toy") == (tmp_path / "big-disk" / "toy").resolve()
    # TiPToP keeps the directory it has always had, and the setting that names it.
    from tandem.core import settings as settings_mod
    from tandem.planners.tiptop.factory import FACTORY

    assert FACTORY.runtime().root == settings_mod.load().resolved_runtime_dir()


@pytest.mark.parametrize(
    "change, problem",
    [
        (dict(commit="4db8f92"), "not a full 40-character commit"),
        (dict(home="toy/env"), "must be its own directory, outside every source"),
        (dict(manifest="elsewhere/pixi.toml"), "is not inside one of the sources"),
        (dict(step_env={"X": "{source:nope}"}), "refers to a source the recipe does not have"),
        (dict(trim=("../outside",)), "is not a path inside the tree"),
        (dict(name=".staging"), "cannot be a directory name"),
        (dict(asset_dest="env/model.pt"), "a place the runtime uses for itself"),
    ],
)
def test_a_recipe_that_could_not_be_installed_is_refused_when_it_is_declared(change, problem):
    pin = SourcePin(
        change.get("name", "toy"), "https://example.invalid/toy.git", change.get("commit", "a" * 40)
    )
    with pytest.raises(TandemError, match=problem):
        RuntimeRecipe(
            planner="toy",
            sources=(Source(pin, trim=change.get("trim", ())),),
            environment=PixiEnvironment(
                manifest=change.get("manifest", f"{pin.name}/pixi.toml"), home=change.get("home", "env")
            ),
            steps=(BuildStep("compile", task="compile", env=change.get("step_env", {})),),
            assets=(Asset(Path("model.pt"), change.get("asset_dest", "weights/model.pt")),),
        )


# --- the command line ----------------------------------------------------------------------------------


class _ToyFactory:
    """A planner whose runtime is a recipe: what `tandem runtime build` installs for a profile using it."""

    def __init__(self, recipe: RuntimeRecipe, root: Path) -> None:
        self.recipe, self.root = recipe, root
        self.info = PlannerInfo(name="toy", display_name="Toy", sources=recipe.pins)

    def capabilities(self):
        from tandem.planners.tiptop.capabilities import CAPABILITIES

        return CAPABILITIES

    def create(self, ctx):  # pragma: no cover - never built into a session here
        raise AssertionError("not in this test")

    def runtime(self, settings=None):
        return RecipeRuntime(self.recipe, self.root)


def test_runtime_build_installs_the_active_profiles_planner(
    upstream, shipped, tmp_path, fake_pixi, monkeypatch, profile
):
    from typer.testing import CliRunner

    from tandem.cli.app import app
    from tandem.core import profiles
    from tandem.core import settings as settings_mod
    from tandem.planners import registry

    isolate_registry(monkeypatch)
    registry.register_backend("toy", _ToyFactory(toy(upstream, shipped), tmp_path / "toy-runtime"))
    profile.planner.backend = "toy"
    profiles.save(profile)
    cfg = settings_mod.load()
    cfg.active_profile = profile.name
    settings_mod.save(cfg)
    tiptop_root = settings_mod.load().resolved_runtime_dir()

    sources = tmp_path / "sources"
    sources.mkdir()
    shutil.copytree(upstream.work, sources / "toy", symlinks=True)
    result = CliRunner().invoke(app, ["runtime", "build", "--sources", str(sources)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "toy-runtime" / "toy" / "pkg" / "config.py").is_file()
    assert not tiptop_root.exists(), "TiPToP's runtime is not the active profile's, so it is left alone"

    status = CliRunner().invoke(app, ["runtime", "status", "--json"])
    assert status.exit_code == 0, status.output
    payload = json.loads(status.output)
    assert payload["planner"] == "toy" and payload["installed"] is True and payload["ready"] is True
    assert payload["mismatched"] == [] and payload["sources"][0]["installed"] == upstream.first
    assert payload["rows"][2] == ["toy kernels", "compiled"]

    assert CliRunner().invoke(app, ["runtime", "path"]).output.strip() == str(tmp_path / "toy-runtime")
    cleaned = CliRunner().invoke(app, ["runtime", "clean", "--yes"])
    assert cleaned.exit_code == 0, cleaned.output
    assert not (tmp_path / "toy-runtime").exists()

    # `tandem init` builds the same runtime, through the same path, from $TANDEM_PLANNER_SOURCES.
    from tandem.cli import init as init_cli

    monkeypatch.setenv("TANDEM_PLANNER_SOURCES", str(sources))
    init_cli._build_runtime(profile.name, interactive=False, repair=False)
    assert RecipeRuntime(toy(upstream, shipped), tmp_path / "toy-runtime").status().installed
    runs = len(fake_pixi())
    init_cli._build_runtime(profile.name, interactive=False, repair=False)
    assert len(fake_pixi()) == runs, "an installed runtime is not rebuilt by a second `tandem init`"


def test_runtime_status_on_a_fresh_machine_describes_the_planner_a_new_profile_gets():
    from typer.testing import CliRunner

    from tandem.cli.app import app
    from tandem.planners.tiptop.recipe import RECIPE

    result = CliRunner().invoke(app, ["runtime", "status", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["planner"] == "tiptop" and payload["ready"] is False
    assert payload["problems"] == ["the runtime has not been created yet"]
    assert [s["commit"] for s in payload["sources"]] == [pin.commit for pin in RECIPE.pins]
    assert payload["mismatched"] == ["tiptop", "cuTAMP", "curobo"]


def test_the_web_ui_and_doctor_read_the_same_runtime_the_terminal_does(profile):
    from fastapi.testclient import TestClient

    from tandem.cli.doctor import collect_checks
    from tandem.core import settings as settings_mod
    from tandem.server.app import create_app

    cfg = settings_mod.load()
    cfg.active_profile = profile.name
    settings_mod.save(cfg)

    payload = TestClient(create_app()).get("/api/runtime").json()
    assert payload["planner"] == "tiptop" and payload["ready"] is False
    # The card draws these rows as they come, whatever the planner calls its parts.
    assert payload["rows"] == [
        ["sources", "missing"],
        ["pixi env", "not built"],
        ["cuRobo kernels", "not compiled"],
        ["built", "—"],
    ]
    assert [s["name"] for s in payload["sources"]] == ["tiptop", "cuTAMP", "curobo"]
    # And the keys the payload always had, for anything reading them.
    assert {"sources_present", "env_built", "kernels_built", "vendor", "built_at", "root"} <= set(payload)

    (check,) = [
        c for c in collect_checks(profile_name=profile.name, probe_hardware=False) if c.name == "gpu runtime"
    ]
    assert check.state == "warn" and "has not been created" in check.detail


# --- TiPToP's recipe ---------------------------------------------------------------------------------


def test_tiptop_pins_the_user_s_main_branches():
    """tiptop 1c6daf3 and cuTAMP 3a2e4d0 are the heads of SamratSahoo's main branches, and move
    together (the recipe says why); cuRobo's main has not moved from 3a90ff4. Full hashes. Moving a
    pin is a deliberate act that moves the sidecar checks with it (tests/test_tiptop_bump.py)."""
    from tandem.planners.tiptop.factory import INFO, SOURCES
    from tandem.planners.tiptop.recipe import RECIPE

    assert {pin.name: pin.commit for pin in RECIPE.pins} == {
        "tiptop": "1c6daf3f5d1ab822a0787c40ec0ed6b6caa472de",
        "cuTAMP": "3a2e4d000339f7460f1989bde84d32b328ac92f9",
        "curobo": "3a90ff49eee169d9636b2a679d98457a2592fb52",
    }
    assert SOURCES == RECIPE.pins == INFO.sources, "the catalog names exactly what an install fetches"
    assert all(pin.url.startswith("https://github.com/SamratSahoo/") for pin in RECIPE.pins)
    # A bundle of these is a redistribution; the NVIDIA License wants its copy kept with each tree.
    assert not any("LICENSE" in path for source in RECIPE.sources for path in source.trim)


def test_tiptops_environment_lives_outside_its_source_trees():
    from tandem.planners.tiptop.recipe import RECIPE

    env = RECIPE.environment
    assert env is not None and env.manifest == "tiptop/pixi.toml"
    assert env.home not in {s.name for s in RECIPE.sources}
    # The checkpoints go where the cuRobo fork's costs look for them: parents[5] of their module.
    assert {a.dest for a in RECIPE.assets} == {
        "vae/checkpoints/vae_full_v2.pt",
        "rnd/checkpoints/rnd_droid.pt",
    }


def test_the_planner_build_pins_curobos_version(tmp_path, monkeypatch):
    """cuRobo takes its version from setuptools_scm, and an exported tree has no SCM metadata.

    Without a pinned version the editable install fails with "unable to detect version" before the
    5-20 minute CUDA kernel build even starts -- and `tandem init` ends with "the build finished but the
    runtime still looks incomplete", which names neither the cause nor the fix.
    """
    from tandem.planners.tiptop.factory import TiptopRuntime
    from tandem.planners.tiptop.recipe import RECIPE

    runs: list[tuple[list[str], dict]] = []
    monkeypatch.setattr(rt_mod, "_find_pixi", lambda: Path("/opt/pixi"))
    monkeypatch.setattr(rt_mod, "_stream", lambda cmd, *, cwd, env, log, what: runs.append((cmd, env)))
    monkeypatch.delenv("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TIPTOP", raising=False)

    rt = TiptopRuntime(tmp_path / "runtime")
    rt.build_environment()
    rt.run_step(RECIPE.steps[0])

    (install, install_env), (build, build_env) = runs
    manifest = str(rt.root / "tiptop" / "pixi.toml")
    assert install == ["/opt/pixi", "install", "--manifest-path", manifest]
    assert build == ["/opt/pixi", "run", "--manifest-path", manifest, "setup-planners"]
    assert install_env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TIPTOP"] == "0.1.0"
    assert build_env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO"] == "0.0.0+g3a90ff4"
    assert build_env["CUROBO_DIR"] == str(rt.root / "curobo")
    assert build_env["CUTAMP_DIR"] == str(rt.root / "cuTAMP")


def test_the_old_runtime_interface_reads_the_new_record(tmp_path, monkeypatch):
    """core.runtime.Runtime is what the backend, `traj open` and the merge still hold. Same answers."""
    from tandem.core.runtime import Runtime
    from tandem.planners.tiptop.recipe import RECIPE

    runtime = Runtime(tmp_path / "runtime")
    assert runtime.status().problems == ["the runtime has not been created yet"]
    assert {n: m["commit"] for n, m in runtime.pending_vendor().items()} == {
        p.name: p.commit for p in RECIPE.pins
    }

    root = runtime.root
    for source in RECIPE.sources:
        marker = root / source.name / source.marker
        marker.parent.mkdir(parents=True, exist_ok=True)
        if marker.suffix:
            marker.write_text("")
        else:
            marker.mkdir()
    rt_mod._write_manifest(
        runtime.stamp_file,
        {
            "sources": {
                s.name: {
                    "url": s.pin.url,
                    "commit": s.pin.commit,
                    "origin": "archive",
                    "verified": True,
                    "patches": [{"name": p.name, "sha256": rt_mod._sha256(p)} for p in s.patches],
                }
                for s in RECIPE.sources
            },
            "assets": {},
            "built": None,
        },
    )
    runtime.recipe_runtime.place_assets()
    half = runtime.status()
    assert half.exists and half.sources_present and not half.env_built and not half.kernels_built
    assert half.vendor["curobo"]["commit"] == RECIPE.source("curobo").pin.commit

    _no_network(monkeypatch)
    runtime.recipe_runtime.fetch()  # current: nothing is fetched, but tiptop/.pixi is linked to env/
    (runtime.pixi_env / "bin").mkdir(parents=True)
    (runtime.pixi_env / "bin" / "python").write_text("")
    assert (root / "env" / "envs" / "default" / "bin" / "python").is_file(), "through the link, into env/"
    kernels = root / "curobo" / "src" / "curobo" / "curobolib"
    kernels.mkdir(parents=True)
    (kernels / "geom_cu.so").write_text("")
    runtime.recipe_runtime.record_built()

    ready = runtime.status()
    assert ready.ready and ready.env_built and ready.kernels_built and ready.built_at
    assert ready.to_dict()["problems"] == []
    runtime.require_ready()
