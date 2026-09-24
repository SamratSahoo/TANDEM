"""A planner's runtime, held to what it promises where the review found it did not.

- An install replaces only what tandem installed. Pointed at a workspace -- the monorepo this layout
  mirrors -- it refuses, and `remove` / `runtime clean` refuse to delete it, instead of taking a
  person's checkout and its uncommitted work with them.
- Two installs of one runtime do not run at once: the second is refused, naming the first.
- An archive cannot write outside the tree it is extracted into, whatever chain of links it carries.
- A directory a planner downloads into its own tree (TiPToP's SAM-2 checkpoint) outlives the tree.
- An export from a bundle is recorded as verified only if its files are the ones bundled.
- A build that failed says so, instead of blaming sources nobody changed.

Built on the fixtures of tests/test_runtime_recipe.py: a toy planner's upstream repository on disk, and
a stand-in for pixi. TiPToP's side of the same review is tests/test_review_tiptop.py.
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
from pathlib import Path

import pytest
import test_runtime_recipe as recipe_tests

from tandem.core.errors import TandemError
from tandem.planners import runtime as rt_mod
from tandem.planners.runtime import RecipeRuntime, RuntimeRecipe

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="these fixtures build git repositories")

# tests/test_runtime_recipe.py's fixtures and helpers, under the names its tests use them by. The two
# autouse ones come along: no $TANDEM_PLANNER_SOURCES turning a fetch into an offline install, and no
# test reaching the real pixi installer.
upstream, shipped, fake_pixi = recipe_tests.upstream, recipe_tests.shipped, recipe_tests.fake_pixi
no_sources_override, never_install_pixi = recipe_tests.no_sources_override, recipe_tests.never_install_pixi
toy, _git = recipe_tests.toy, recipe_tests._git


def _checkout(upstream, where: Path) -> Path:
    """A person's clone of the planner, with work in it that exists nowhere else."""
    _git("clone", "-q", str(upstream.bare), str(where))
    (where / "MY_NOTES.txt").write_text("three days of uncommitted work")
    return where


# --- only what tandem installed is ever replaced ------------------------------------------------------


@needs_git
def test_an_unrecorded_checkout_where_a_source_belongs_is_not_deleted(upstream, shipped, tmp_path):
    root = tmp_path / "hitl-tamp-vla"  # the workspace a mistyped TANDEM_RUNTIME_DIR points at
    notes = _checkout(upstream, root / "toy") / "MY_NOTES.txt"
    rt = RecipeRuntime(toy(upstream, shipped), root)

    for force in (False, True):  # --force means "fetch again", not "delete a checkout"
        with pytest.raises(TandemError, match="is not a tree tandem installed .*git checkout") as caught:
            rt.fetch(force=force)
        assert "--sources" in caught.value.hint and "tandem runtime path" in caught.value.hint
    assert notes.is_file() and (root / "toy" / ".git").is_dir()
    assert sorted(p.name for p in root.iterdir()) == ["toy"], "nothing was written into the workspace"

    assert not rt.looks_like_a_runtime()
    with pytest.raises(TandemError, match="does not look like a Toy runtime"):
        rt.uninstall()
    assert notes.is_file()


@needs_git
def test_a_checkout_is_refused_even_in_a_root_that_has_a_record(upstream, shipped, tmp_path, fake_pixi):
    """The record says what tandem installed, and a checkout is never that: nothing tandem installs has a
    .git. So a checkout moved in beside a runtime's record is refused too -- by install as by fetch."""
    root = tmp_path / "runtime"
    root.mkdir()
    rt = RecipeRuntime(toy(upstream, shipped), root)
    rt_mod._write_manifest(rt.manifest_file, {"planner": "toy", "sources": {}})
    notes = _checkout(upstream, root / "toy") / "MY_NOTES.txt"
    with pytest.raises(TandemError, match="git checkout"):
        rt.install()
    assert notes.is_file()
    assert not rt.looks_like_a_runtime(), "a runtime with someone's checkout in it is not deleted whole"


@needs_git
def test_a_directory_with_no_record_is_not_replaced(upstream, shipped, tmp_path):
    root = tmp_path / "somewhere"
    (root / "toy").mkdir(parents=True)
    (root / "toy" / "thesis.tex").write_text("four years of work")
    rt = RecipeRuntime(toy(upstream, shipped), root)
    with pytest.raises(TandemError, match=f"has no {re.escape(rt_mod.MANIFEST_FILE)}"):
        rt.fetch()
    assert (root / "toy" / "thesis.tex").is_file()
    with pytest.raises(TandemError, match="does not look like a Toy runtime"):
        rt.uninstall()


@needs_git
def test_an_install_interrupted_before_its_record_resumes(upstream, shipped, tmp_path, monkeypatch):
    """The record is written before the first tree lands, so a tree left without an entry in it -- an
    install stopped between the swap and the write -- is still replaced when the install is run again."""
    root = tmp_path / "runtime"
    rt = RecipeRuntime(toy(upstream, shipped), root)

    def interrupted(*_args, **_kwargs):
        raise TandemError("the network went away")

    real_trim = rt_mod.trim
    monkeypatch.setattr(rt_mod, "trim", interrupted)
    with pytest.raises(TandemError, match="network went away"):
        rt.fetch()
    assert rt.manifest_file.is_file(), "written before anything was installed"
    assert rt.looks_like_a_runtime()

    # As if the tree had landed and the process died before recording it.
    (root / "toy" / "pkg").mkdir(parents=True)
    (root / "toy" / "pkg" / "half.py").write_text("")
    monkeypatch.setattr(rt_mod, "trim", real_trim)
    assert rt.fetch() == ["toy"]
    assert (root / "toy" / "README.md").read_text() == "toy planner v1\n"
    assert not (root / "toy" / "pkg" / "half.py").exists()


@needs_git
def test_the_sources_directory_cannot_be_the_runtime(upstream, shipped, tmp_path):
    root = tmp_path / "runtime"
    rt = RecipeRuntime(toy(upstream, shipped), root)
    rt.fetch()
    with pytest.raises(TandemError, match="is the runtime itself"):
        rt.fetch(sources_dir=root, force=True)
    assert (root / "toy" / "README.md").is_file()


def test_only_a_record_or_nothing_makes_a_directory_a_runtime(tmp_path):
    recipe = RuntimeRecipe(
        planner="toy",
        sources=(rt_mod.Source(rt_mod.SourcePin("toy", "https://example.invalid/toy.git", "a" * 40)),),
    )
    rt = RecipeRuntime(recipe, tmp_path / "rt")
    rt.root.mkdir()
    assert rt.looks_like_a_runtime(), "an empty directory"
    (rt.root / rt_mod.LOCK_FILE).write_text("123\n")
    assert rt.looks_like_a_runtime(), "the lock an install takes first is not someone else's file"
    (rt.root / "toy").mkdir()
    assert not rt.looks_like_a_runtime(), "a directory named like a source is not a record"
    rt.manifest_file.write_text("{}")
    assert rt.looks_like_a_runtime()
    (rt.root / "toy" / ".git").write_text("gitdir: elsewhere")  # a submodule's .git is a file
    assert not rt.looks_like_a_runtime()


# --- one install at a time --------------------------------------------------------------------------------


_HOLDER = """
import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o644)
fcntl.flock(fd, fcntl.LOCK_EX)
os.ftruncate(fd, 0)
os.write(fd, f"{os.getpid()}\\n".encode())
print("locked", flush=True)
sys.stdin.read()
"""


@needs_git
@pytest.mark.skipif(sys.platform == "win32", reason="flock")
def test_a_second_install_is_refused_while_one_is_running(upstream, shipped, tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    rt = RecipeRuntime(toy(upstream, shipped), root)
    rt_mod._write_manifest(rt.manifest_file, {"planner": "toy", "sources": {}})  # the first install's
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(root / rt_mod.LOCK_FILE)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "locked"
        # What the running install has staged, and the old tree it keeps to roll back to.
        staged = root / rt_mod.STAGING / "toy.old" / "README.md"
        staged.parent.mkdir(parents=True)
        staged.write_text("the tree the first install would put back")

        with pytest.raises(TandemError, match=rf"Another install of the Toy runtime is running \(pid {holder.pid}\)"):
            rt.fetch()
        with pytest.raises(TandemError, match="Another install"):
            rt.uninstall()
        assert staged.is_file(), "the running install's staging area is not touched"
    finally:
        holder.stdin.close()
        holder.wait(timeout=30)

    assert rt.fetch() == ["toy"], "and once it is done, the next one runs"


@needs_git
def test_the_lock_is_reentrant_in_a_thread_and_refused_across_threads(upstream, shipped, tmp_path):
    rt = RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime")
    with rt._install_lock():
        assert rt.fetch() == ["toy"], "install holds the lock and calls fetch, which takes it again"

    entered, release = threading.Event(), threading.Event()

    def hold() -> None:
        with RecipeRuntime(rt.recipe, rt.root)._install_lock():
            entered.set()
            release.wait(10)

    other = threading.Thread(target=hold)
    other.start()
    try:
        assert entered.wait(10)
        with pytest.raises(TandemError, match=f"pid {os.getpid()}"):
            rt.fetch(force=True)
    finally:
        release.set()
        other.join(10)
    assert rt.fetch(force=True) == ["toy"]


def test_the_runtime_s_own_names_are_reserved():
    pin = rt_mod.SourcePin("cache", "https://example.invalid/x.git", "a" * 40)
    with pytest.raises(TandemError, match="a name the runtime uses for itself"):
        RuntimeRecipe(planner="toy", sources=(rt_mod.Source(pin),))
    ok = rt_mod.Source(rt_mod.SourcePin("toy", "https://example.invalid/x.git", "a" * 40))
    for dest in (f"{rt_mod.LOCK_FILE}/x", "cache/model.pt"):
        with pytest.raises(TandemError, match="a place the runtime uses for itself"):
            RuntimeRecipe(planner="toy", sources=(ok,), assets=(rt_mod.Asset(Path("m.pt"), dest),))
    with pytest.raises(TandemError, match="is not a path inside the tree"):
        RuntimeRecipe(planner="toy", sources=(dataclasses.replace(ok, persistent=("../out",)),))


# --- extracting an archive --------------------------------------------------------------------------------


def _tar(path: Path, entries: list[tuple]) -> Path:
    """entries: ("dir", name) | ("file", name, bytes) | ("sym", name, target) | ("hard", name, target)."""
    with tarfile.open(path, "w") as tar:
        for kind, name, *rest in entries:
            info = tarfile.TarInfo(name)
            if kind == "dir":
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                tar.addfile(info)
            elif kind == "file":
                info.size = len(rest[0])
                tar.addfile(info, io.BytesIO(rest[0]))
            else:
                info.type = tarfile.SYMTYPE if kind == "sym" else tarfile.LNKTYPE
                info.linkname = rest[0]
                tar.addfile(info)
    return path


@pytest.mark.parametrize(
    "entries",
    [
        # Every name passes a check of the name: c/d is the tree's root, and e is spelled c.
        [("dir", "c"), ("sym", "c/d", ".."), ("sym", "e", "c/d/.."), ("file", "e/ESCAPED.txt", b"x")],
        # The same, with the links in the other order.
        [("dir", "c"), ("sym", "e", "c/d/.."), ("sym", "c/d", ".."), ("file", "e/ESCAPED.txt", b"x")],
        # A hard link written through the chain instead of a file.
        [
            ("file", "real.txt", b"x"),
            ("dir", "c"),
            ("sym", "c/d", ".."),
            ("sym", "e", "c/d/.."),
            ("hard", "e/ESCAPED.txt", "real.txt"),
        ],
    ],
    ids=["file", "links-reversed", "hardlink"],
)
def test_a_chain_of_links_cannot_carry_a_write_out_of_the_tree(tmp_path, entries):
    archive = _tar(tmp_path / "evil.tar", entries)
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(TandemError, match="archive has"):
        rt_mod.extract(archive, work / "dest")
    assert not (work / "ESCAPED.txt").exists() and not (tmp_path / "ESCAPED.txt").exists()
    assert not (work / "dest").exists(), "a refused tree is deleted, not left for a trim or patch to follow"


def test_links_alone_that_lead_out_of_the_tree_are_refused(tmp_path):
    for order in ([("sym", "c/d", ".."), ("sym", "e", "c/d/..")], [("sym", "e", "c/d/.."), ("sym", "c/d", "..")]):
        archive = _tar(tmp_path / "links.tar", [("dir", "c"), *order])
        with pytest.raises(TandemError, match="link out of its own tree"):
            rt_mod.extract(archive, tmp_path / "dest")
        assert not (tmp_path / "dest").exists()


def test_links_inside_the_tree_still_extract(tmp_path):
    archive = _tar(
        tmp_path / "fine.tar",
        [
            ("dir", "pkg"),
            ("file", "pkg/a.txt", b"hello"),
            ("sym", "pkg/b.txt", "a.txt"),
            ("sym", "alias", "pkg"),
            ("sym", "dangling", "pkg/trimmed.txt"),
            ("hard", "copy.txt", "pkg/a.txt"),
        ],
    )
    dest = tmp_path / "dest"
    rt_mod.extract(archive, dest)
    assert (dest / "alias" / "b.txt").read_text() == "hello"
    assert (dest / "copy.txt").read_text() == "hello" and os.readlink(dest / "dangling") == "pkg/trimmed.txt"


# --- what a tree downloads into itself outlives it ----------------------------------------------------------


def _persistent(recipe: RuntimeRecipe) -> RuntimeRecipe:
    (source,) = recipe.sources
    return dataclasses.replace(recipe, sources=(dataclasses.replace(source, persistent=("pkg/.cache",)),))


@needs_git
def test_a_persistent_directory_survives_a_bump_and_a_forced_fetch(upstream, shipped, tmp_path):
    root = tmp_path / "runtime"
    RecipeRuntime(_persistent(toy(upstream, shipped)), root).fetch()
    cache = root / "toy" / "pkg" / ".cache"
    assert cache.is_symlink() and cache.is_dir(), "a link to a directory that exists: mkdir(exist_ok) passes"
    (cache / "sam2.1_hiera_large.pt").write_bytes(b"\0" * 1024)

    bumped = RecipeRuntime(_persistent(toy(upstream, shipped, upstream.second)), root)
    assert bumped.fetch() == ["toy"]
    assert bumped.fetch(force=True) == ["toy"]
    assert (root / "toy" / "pkg" / "new.py").is_file(), "the tree really was replaced"
    assert cache.is_symlink() and (cache / "sam2.1_hiera_large.pt").stat().st_size == 1024
    assert (root / rt_mod.PERSISTENT / "toy" / "pkg" / ".cache" / "sam2.1_hiera_large.pt").is_file()


@needs_git
def test_a_runtime_from_before_keeps_the_checkpoint_already_in_its_tree(upstream, shipped, tmp_path):
    root = tmp_path / "runtime"
    RecipeRuntime(toy(upstream, shipped), root).fetch()
    inside = root / "toy" / "pkg" / ".cache"
    inside.mkdir()
    (inside / "ckpt.pt").write_text("downloaded at the last warm-up")

    lines: list[str] = []
    RecipeRuntime(_persistent(toy(upstream, shipped, upstream.second)), root).fetch(log=lines.append)
    assert inside.is_symlink() and (inside / "ckpt.pt").read_text() == "downloaded at the last warm-up"
    assert any("moved pkg/.cache out of the tree" in line for line in lines)


# --- a bundle's export is verified by its contents ----------------------------------------------------------


def _bundle(upstream, shipped, tmp_path: Path, monkeypatch) -> Path:
    """What `tandem planners bundle` makes (and `python tools/bundle.py`, a wrapper of it), made by it."""
    from tandem.planners import bundle as bundle_mod

    monkeypatch.setattr(bundle_mod, "recipe_of", lambda planner: toy(upstream, shipped))
    out = tmp_path / "usb"
    bundle_mod.bundle_sources("toy", out, only=[], local={"toy": upstream.work}, log=lambda _line: None)
    return out


def _refuse_network(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("a sources directory is in force; nothing may be fetched")

    monkeypatch.setattr(rt_mod, "export_pinned", refuse)


@needs_git
def test_an_untouched_bundle_installs_verified(upstream, shipped, tmp_path, monkeypatch):
    bundle = _bundle(upstream, shipped, tmp_path, monkeypatch)
    assert json.loads((bundle / "toy" / rt_mod.SOURCE_MARKER).read_text())["sha256"]
    _refuse_network(monkeypatch)
    rt = RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime")
    rt.fetch(sources_dir=bundle)
    assert rt.record()["sources"]["toy"]["verified"] is True
    assert not any("was not verified" in note for note in rt.inspect().notes)


@needs_git
@pytest.mark.parametrize("change", ["edited", "added", "removed", "relinked"])
def test_a_bundle_changed_after_it_was_made_is_refused(upstream, shipped, tmp_path, monkeypatch, change):
    bundle = _bundle(upstream, shipped, tmp_path, monkeypatch)
    tree = bundle / "toy"
    if change == "edited":
        (tree / "pkg" / "config.py").write_text("NAME = 'someone else'\n")
    elif change == "added":
        (tree / "pkg" / "backdoor.py").write_text("")
    elif change == "removed":
        (tree / "README.md").unlink()
    else:
        (tree / "README.md").unlink()
        (tree / "README.md").symlink_to("pkg/config.py")
    _refuse_network(monkeypatch)
    rt = RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime")
    with pytest.raises(TandemError, match="its files have changed since") as caught:
        rt.fetch(sources_dir=bundle)
    assert rt_mod.SOURCE_MARKER in caught.value.hint
    assert "toy" not in rt.record()["sources"]


@needs_git
def test_a_marker_copied_onto_another_tree_is_refused(upstream, shipped, tmp_path, monkeypatch):
    bundle = _bundle(upstream, shipped, tmp_path, monkeypatch)
    other = tmp_path / "other"
    rt_mod._export_with_git(str(upstream.bare), upstream.second, other / "toy", scratch=tmp_path / "s", fetch=True)
    shutil.copy2(bundle / "toy" / rt_mod.SOURCE_MARKER, other / "toy" / rt_mod.SOURCE_MARKER)
    _refuse_network(monkeypatch)
    with pytest.raises(TandemError, match="its files have changed since"):
        RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime").fetch(sources_dir=other)


@needs_git
def test_a_marker_with_no_digest_is_taken_on_trust_and_says_so(upstream, shipped, tmp_path, monkeypatch):
    bundle = _bundle(upstream, shipped, tmp_path, monkeypatch)
    marker = bundle / "toy" / rt_mod.SOURCE_MARKER
    stated = json.loads(marker.read_text())
    del stated["sha256"]  # a bundle an older tandem made
    marker.write_text(json.dumps(stated))
    _refuse_network(monkeypatch)
    rt = RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime")
    lines: list[str] = []
    rt.fetch(sources_dir=bundle, log=lines.append)
    assert rt.record()["sources"]["toy"]["verified"] is False
    assert any("records no digest" in line for line in lines)
    assert any("was not verified" in note for note in rt.inspect().notes)


def test_a_tree_digest_is_of_contents_and_names_not_of_junk_or_modes(tmp_path):
    tree = tmp_path / "tree"
    (tree / "pkg").mkdir(parents=True)
    (tree / "pkg" / "a.py").write_text("A = 1\n")
    (tree / "link").symlink_to("pkg/a.py")
    before = rt_mod.tree_digest(tree)

    (tree / "pkg" / "__pycache__").mkdir()
    (tree / "pkg" / "__pycache__" / "a.pyc").write_bytes(b"junk")
    (tree / rt_mod.SOURCE_MARKER).write_text("{}")
    os.chmod(tree / "pkg" / "a.py", 0o755)
    assert rt_mod.tree_digest(tree) == before

    (tree / "link").unlink()
    (tree / "link").symlink_to("pkg")
    assert rt_mod.tree_digest(tree) != before
    # The same bytes under another name are another tree.
    (tree / "link").unlink()
    (tree / "link").symlink_to("pkg/a.py")
    (tree / "pkg" / "a.py").rename(tree / "pkg" / "b.py")
    (tree / "link").unlink()
    (tree / "link").symlink_to("pkg/b.py")
    assert rt_mod.tree_digest(tree) != before


# --- a build that failed says so -----------------------------------------------------------------------


@pytest.fixture
def failing_pixi(tmp_path, monkeypatch):
    script = _failing_pixi(tmp_path)
    monkeypatch.setattr(rt_mod, "_find_pixi", lambda: script)
    return script


def _failing_pixi(tmp_path: Path) -> Path:
    """pixi whose build step compiles one extension and then fails: what `produces` looks for is there."""
    script = tmp_path / "failing-bin" / "pixi"
    script.parent.mkdir()
    script.write_text(
        f"""#!{sys.executable}
import sys
from pathlib import Path
args = sys.argv[1:]
manifest = Path(args[args.index("--manifest-path") + 1])
if args[0] == "install":
    python = manifest.parent / ".pixi" / "envs" / "default" / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("")
elif args[0] == "run":
    out = manifest.parent.parent / "build"
    out.mkdir(exist_ok=True)
    (out / "kernel.so").write_text("")
    print("nvcc: error compiling the second extension")
    sys.exit(1)
"""
    )
    script.chmod(0o755)
    return script


@needs_git
def test_a_failed_build_step_is_named_not_blamed_on_the_sources(upstream, shipped, tmp_path, failing_pixi):
    rt = RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime")
    with pytest.raises(TandemError, match="pixi run compile failed"):
        rt.install()
    problems = rt.inspect().problems
    assert not any("changed" in p for p in problems), problems
    assert any(p.startswith("pixi run compile did not finish") and "build log" in p for p in problems), problems
    assert not rt.status().installed


@needs_git
def test_a_build_that_finishes_clears_the_failure(upstream, shipped, tmp_path, fake_pixi, monkeypatch):
    working, failing = rt_mod._find_pixi(), _failing_pixi(tmp_path)
    rt = RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime")
    monkeypatch.setattr(rt_mod, "_find_pixi", lambda: failing)
    with pytest.raises(TandemError, match="pixi run compile failed"):
        rt.install()
    assert rt.record()["last_build"]["step"] == "compile"

    monkeypatch.setattr(rt_mod, "_find_pixi", lambda: working)
    rt.install()
    assert rt.status().installed and rt.record()["last_build"] is None


@needs_git
def test_an_unbuilt_runtime_says_its_sources_were_installed_not_changed(upstream, shipped, tmp_path, fake_pixi):
    rt = RecipeRuntime(toy(upstream, shipped), tmp_path / "runtime")
    rt.fetch()
    rt.place_assets()
    rt.build_environment()
    rt.run_step(rt.recipe.steps[0])  # everything on disk, but no install ever finished
    assert rt.inspect().problems == ("the runtime has not been built since its sources were installed",)

