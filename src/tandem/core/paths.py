"""Where tandem keeps things on disk.

Three roots, each overridable by an environment variable so a whole install can be
relocated (CI, a shared workstation account, a scratch disk):

    config    ~/.config/tandem                $TANDEM_CONFIG_DIR    config.toml, credentials.toml, and the
                                                                    rig: rig.yml, calibration.json
    state     ~/.local/state/tandem           $TANDEM_STATE_DIR     logs, session scratch
    data      ~/tandem-data                   $TANDEM_DATA_ROOT     profiles/<name>.yml, trajectories/<name>/
    runtime   ~/.local/share/tandem/runtime   $TANDEM_RUNTIME_DIR   TiPToP's runtime: its sources + pixi env
    runtimes  ~/.local/share/tandem/runtimes  $TANDEM_RUNTIMES_DIR  every other planner's, one directory each

The data root is also settable in config.toml (the env var wins) because it is the one a
user actually wants somewhere else -- trajectories are large. So is the runtime, for the same
reason: it is ~25 GB.

TiPToP's runtime keeps the name and the setting it had before tandem drove more than one planner.
A built pixi environment has its own absolute path baked into it, so moving an existing one would
break it; every workstation that has built one keeps it where it is.
"""

from __future__ import annotations

import os
from pathlib import Path

import platformdirs

_APP = "tandem"


def _env_path(var: str) -> Path | None:
    raw = os.environ.get(var, "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def config_dir() -> Path:
    return _env_path("TANDEM_CONFIG_DIR") or Path(platformdirs.user_config_dir(_APP))


def state_dir() -> Path:
    return _env_path("TANDEM_STATE_DIR") or Path(platformdirs.user_state_dir(_APP))


def share_dir() -> Path:
    return _env_path("TANDEM_SHARE_DIR") or Path(platformdirs.user_data_dir(_APP))


def config_file() -> Path:
    return config_dir() / "config.toml"


def credentials_file() -> Path:
    return config_dir() / "credentials.toml"


def rig_file() -> Path:
    """This machine's rig: its robot, its cameras and their calibration, shared by every profile."""
    return config_dir() / "rig.yml"


def log_dir() -> Path:
    return state_dir() / "logs"


def session_scratch_dir() -> Path:
    """Per-session temp files (events JSONL, rendered configs). Not in /tmp: these are
    useful post-mortem, and /tmp is wiped under the user's feet on some systems."""
    return state_dir() / "sessions"


def default_data_root() -> Path:
    return Path.home() / "tandem-data"


def default_runtime_dir() -> Path:
    return share_dir() / "runtime"


def runtimes_dir() -> Path:
    """Where a planner's runtime lives by default: one directory per planner, named for it."""
    return _env_path("TANDEM_RUNTIMES_DIR") or share_dir() / "runtimes"


def planner_sources_override() -> Path | None:
    """A directory of planner sources to install from instead of fetching them, or None.

    $TANDEM_PLANNER_SOURCES holds one checkout or export per source, named as the planner's recipe
    names them (``tiptop/``, ``cuTAMP/``, ``curobo/``). It is how a workstation that cannot reach GitHub
    gets a planner's sources, from a bundle ``tandem planners bundle`` made elsewhere (the environment is
    still downloaded from conda-forge and PyPI). $TANDEM_VENDOR_DIR is its old name,
    from when the sources shipped inside the wheel; a directory set up for that has the same shape.
    """
    return _env_path("TANDEM_PLANNER_SOURCES") or _env_path("TANDEM_VENDOR_DIR")


def ensure_dir(path: Path, *, mode: int | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if mode is not None:
        os.chmod(path, mode)
    return path


def write_atomic(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` whole or not at all: written beside it, then renamed over it.

    Truncating a settings file before a write that could fail -- a value that cannot be serialised, a
    Ctrl-C, a full disk -- once left an empty profile that then loaded, silently, as one of defaults.
    """
    partial = path.with_name(f".{path.name}.partial")
    try:
        with partial.open("w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
