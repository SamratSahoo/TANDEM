"""Where tandem keeps things on disk.

Three roots, each overridable by an environment variable so a whole install can be
relocated (CI, a shared workstation account, a scratch disk):

    config   ~/.config/tandem              $TANDEM_CONFIG_DIR    config.toml, credentials.toml
    state    ~/.local/state/tandem         $TANDEM_STATE_DIR     logs, session scratch
    data     ~/tandem-data                 $TANDEM_DATA_ROOT     profiles/ and their trajectories
    runtime  ~/.local/share/tandem/runtime $TANDEM_RUNTIME_DIR   the pixi env + vendored sources

The data root is also settable in config.toml (the env var wins) because it is the one a
user actually wants somewhere else -- trajectories are large.
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


def vendor_dir() -> Path:
    """The vendored tiptop / cuTAMP / cuRobo sources shipped inside the wheel.

    $TANDEM_VENDOR_DIR points this at a live checkout during development.
    """
    override = _env_path("TANDEM_VENDOR_DIR")
    if override:
        return override
    return Path(__file__).resolve().parent.parent / "_vendor"


def ensure_dir(path: Path, *, mode: int | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if mode is not None:
        os.chmod(path, mode)
    return path
