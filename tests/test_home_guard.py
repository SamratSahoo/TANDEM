"""The test run never resolves the developer's real home, even from a thread that outlived its test.

`isolated_env` points every tandem root at a per-test directory and puts the environment back when the
test ends. A background merge that ran on after that resolved the real ~/tandem-data, and wrote a
trajectory into it. conftest now gives the whole run a home and roots of its own underneath
(`home_guard.apply`), and fails the run if anything under the real ones changed while it ran.
"""

from __future__ import annotations

import os
from pathlib import Path

import home_guard
import platformdirs

from tandem.core import paths
from tandem.core import settings as settings_mod


def _under(path: Path, root: Path) -> bool:
    return Path(path).resolve().is_relative_to(root.resolve())


def _real_home() -> Path | None:
    original = home_guard.ORIGINAL["HOME"]
    return Path(original) if original else None


def test_a_thread_that_outlives_its_test_still_resolves_the_runs_own_roots(monkeypatch):
    # What a merge still running after its test sees: the environment the test started from.
    monkeypatch.undo()
    settings_mod._cache = None
    run_root = Path(os.environ["HOME"]).parent
    resolved = {
        "home": Path.home(),
        "data root": settings_mod.load().resolved_data_root(),
        "config": paths.config_dir(),
        "state": paths.state_dir(),
        "share": paths.share_dir(),
        "runtime": settings_mod.load().resolved_runtime_dir(),
        "runtimes": paths.runtimes_dir(),
        "platformdirs": Path(platformdirs.user_data_dir("tandem")),
    }
    outside = {name: path for name, path in resolved.items() if not _under(path, run_root)}
    assert not outside, f"resolved outside the run's own directory {run_root}: {outside}"
    real = _real_home()
    if real is not None:
        assert not any(_under(path, real) for path in resolved.values()), resolved


def test_the_guard_sees_what_was_written(tmp_path):
    watched = tmp_path / "tandem-data"
    (watched / "profiles").mkdir(parents=True)
    (watched / "profiles" / "profile.yml").write_text("name: a\n")
    before = home_guard.snapshot([watched, tmp_path / "never-made"])

    (watched / "profiles" / "b").mkdir()
    (watched / "profiles" / "profile.yml").write_text("name: a, and more\n")
    changed = home_guard.changes(before, home_guard.snapshot([watched, tmp_path / "never-made"]))

    assert f"new      {watched / 'profiles' / 'b'}" in changed
    assert f"changed  {watched / 'profiles' / 'profile.yml'}" in changed
    assert home_guard.changes(before, before) == []


def test_the_developers_own_environment_is_there_for_what_asks_for_it(monkeypatch):
    """planner_sources finds a developer's built runtime this way; a key the test set stays the test's."""
    run_home = os.environ["HOME"]
    monkeypatch.setenv("TANDEM_RUNTIME_DIR", "/the/test/says")
    with home_guard.as_the_developer():
        assert os.environ.get("HOME") == home_guard.ORIGINAL["HOME"]
        assert os.environ["TANDEM_RUNTIME_DIR"] == "/the/test/says"
    assert os.environ["HOME"] == run_home and os.environ["TANDEM_RUNTIME_DIR"] == "/the/test/says"


def test_a_planner_imported_from_a_runtime_leaves_no_bytecode_in_it(tmp_path):
    """The cuTAMP domain check imported from the developer's runtime, which the guard watches, and wrote
    `__pycache__` into it: the run failed after every runtime build, blaming a thread that outlived its test."""
    import sys

    from planner_sources import importable_from

    tree = tmp_path / "cuTAMP"
    (tree / "toyplanner" / "domain").mkdir(parents=True)
    (tree / "toyplanner" / "__init__.py").write_text("")
    (tree / "toyplanner" / "domain" / "__init__.py").write_text("from toyplanner.domain.facts import FLUENTS\n")
    (tree / "toyplanner" / "domain" / "facts.py").write_text("FLUENTS = ('On', 'Holding')\n")
    writes_bytecode = sys.dont_write_bytecode
    before = home_guard.snapshot([tree])

    with importable_from(tree, "toyplanner"):
        from toyplanner.domain import FLUENTS

    assert FLUENTS == ("On", "Holding")
    assert home_guard.changes(before, home_guard.snapshot([tree])) == [], "the import wrote into the tree"
    assert not any(name.split(".")[0] == "toyplanner" for name in sys.modules)
    assert str(tree) not in sys.path and sys.dont_write_bytecode is writes_bytecode
