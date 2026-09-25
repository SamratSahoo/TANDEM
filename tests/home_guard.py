"""The developer's real home, kept out of the test run, and checked afterwards to have been.

``isolated_env`` points every tandem root at a per-test temporary directory, and puts them back when
the test ends. A thread that outlives its test then resolves the real ones: a background merge once
wrote a trajectory into the real ~/tandem-data that way. So for the whole run conftest also points
HOME, the XDG directories and the tandem roots at a directory of the run's own (``apply``), and at the
end compares everything tandem keeps under the real ones with how it was before (``snapshot``). Any
difference fails the run, naming what was written.

Imported by conftest before it changes anything, so ``ORIGINAL`` is the environment the run was
started in. ``as_the_developer`` restores it for the few checks that deliberately look at the
developer's own install (tests/planner_sources.py finds a built planner runtime that way).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

#: Set for the whole run. The four tandem roots are the ones `isolated_env` sets for each test.
SET = (
    "HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
    "XDG_CACHE_HOME",
    "TANDEM_CONFIG_DIR",
    "TANDEM_STATE_DIR",
    "TANDEM_SHARE_DIR",
    "TANDEM_DATA_ROOT",
)
#: Unset for the whole run. A developer who exports one points it at their real runtime, which a test
#: that builds or cleans a runtime would then build or clean. Unset, each follows the share directory of
#: the test that reads it, which is what every such test expects.
UNSET = ("TANDEM_RUNTIME_DIR", "TANDEM_RUNTIMES_DIR")

ORIGINAL: dict[str, str | None] = {key: os.environ.get(key) for key in (*SET, *UNSET)}
# What `apply` set each key to (None: unset), so `as_the_developer` can tell a key a test has since set
# for itself, and leave it alone.
_applied: dict[str, str | None] = {}

# Deep enough for a trajectory's files under a data root (profiles/<p>/trajectories/<status>/<id>/<file>),
# and shallow enough not to walk every file of a built planner environment.
DEPTH = 6


def _real_paths() -> list[Path]:
    """Every directory tandem keeps anything in under the real home, as the environment says now."""
    import platformdirs

    from tandem.core import settings as settings_mod

    home = Path.home()
    found = [
        home / "tandem-data",
        home / ".config" / "tandem",
        home / ".local" / "state" / "tandem",
        home / ".local" / "share" / "tandem",
        home / ".cache" / "tandem",
        Path(platformdirs.user_config_dir("tandem")),
        Path(platformdirs.user_state_dir("tandem")),
        Path(platformdirs.user_data_dir("tandem")),
        Path(platformdirs.user_cache_dir("tandem")),
        Path(platformdirs.user_log_dir("tandem")),
    ]
    found += [Path(value) for key, value in ORIGINAL.items() if key.startswith("TANDEM_") and value]
    try:
        # The data root and the runtime the developer's own settings name, wherever those are.
        cfg = settings_mod.load(force=True)
        found += [cfg.resolved_data_root(), cfg.resolved_runtime_dir()]
    except Exception:  # an unreadable config is the developer's to fix, not a reason to stop the run
        pass
    finally:
        settings_mod._cache = None
        settings_mod._stamp = None
    unique: list[Path] = []
    for path in found:
        path = Path(os.path.expanduser(str(path))).absolute()
        if path not in unique:
            unique.append(path)
    return unique


#: Read before anything is changed: this module is imported first.
WATCHED = _real_paths()


def snapshot(paths: list[Path] = WATCHED) -> dict[str, tuple[bool, int, int]]:
    """Every entry under ``paths`` to ``DEPTH`` levels: whether it is a directory, its mtime and size."""
    seen: dict[str, tuple[bool, int, int]] = {}

    def walk(directory: Path, depth: int) -> None:
        try:
            entries = list(os.scandir(directory))
        except OSError:
            return
        for entry in entries:
            try:
                stat = entry.stat(follow_symlinks=False)
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            seen[entry.path] = (is_dir, stat.st_mtime_ns, 0 if is_dir else stat.st_size)
            if is_dir and depth < DEPTH:
                walk(Path(entry.path), depth + 1)

    for root in paths:
        try:
            stat = root.stat()
        except OSError:
            continue
        seen[str(root)] = (root.is_dir(), stat.st_mtime_ns, 0)
        if root.is_dir():
            walk(root, 1)
    return seen


def changes(before: dict, after: dict) -> list[str]:
    """What differs between two snapshots, one line per path."""
    lines = [f"new      {path}" for path in sorted(after.keys() - before.keys())]
    lines += [f"removed  {path}" for path in sorted(before.keys() - after.keys())]
    lines += [f"changed  {path}" for path in sorted(before.keys() & after.keys()) if before[path] != after[path]]
    return lines


def apply(root: Path) -> None:
    """Point HOME, the XDG directories and the tandem roots under ``root`` for the rest of the run."""
    home = root / "home"
    values = {
        "HOME": home,
        "XDG_CONFIG_HOME": home / ".config",
        "XDG_DATA_HOME": home / ".local" / "share",
        "XDG_STATE_HOME": home / ".local" / "state",
        "XDG_CACHE_HOME": home / ".cache",
        "TANDEM_CONFIG_DIR": root / "config",
        "TANDEM_STATE_DIR": root / "state",
        "TANDEM_SHARE_DIR": root / "share",
        "TANDEM_DATA_ROOT": root / "data",
    }
    home.mkdir(parents=True, exist_ok=True)
    for key in SET:
        os.environ[key] = _applied[key] = str(values[key])
    for key in UNSET:
        os.environ.pop(key, None)
        _applied[key] = None


def restore() -> None:
    """Put the environment the run started in back."""
    for key, value in ORIGINAL.items():
        _put(key, value)
    _applied.clear()


@contextmanager
def as_the_developer() -> Iterator[None]:
    """The environment the run started in, for the keys no test has since set for itself."""
    ours = {key: value for key, value in _applied.items() if os.environ.get(key) == value}
    for key in ours:
        _put(key, ORIGINAL[key])
    try:
        yield
    finally:
        for key, value in ours.items():
            _put(key, value)


def _put(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
