"""Global settings — everything that is NOT per-profile.

Lives in ``~/.config/tandem/config.toml``. Read on demand and cached; ``save()`` rewrites it
with tomlkit so hand-written comments survive a programmatic edit.
"""

from __future__ import annotations

from pathlib import Path

import tomlkit
from pydantic import BaseModel, Field

from tandem.core import paths
from tandem.core.errors import TandemError


class UiSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8787
    open_browser: bool = True


class TeleopSettings(BaseModel):
    """The TAMP⇄teleop hand-off needs a DROID checkout and its interpreter.

    Both empty means the feature is simply unavailable; the UI disables its button with a
    reason rather than failing mid-session.
    """

    enabled: bool = False
    droid_dir: str = ""
    python: str = ""
    device: str = "vr"  # vr | spacemouse
    controller: str = "right"  # right | left, VR only


class Settings(BaseModel):
    model_config = {"extra": "forbid"}

    active_profile: str = "default"
    data_root: str = ""  # blank -> paths.default_data_root()
    runtime_dir: str = ""  # blank -> paths.default_runtime_dir()
    hf_org: str = ""
    ui: UiSettings = Field(default_factory=UiSettings)
    teleop: TeleopSettings = Field(default_factory=TeleopSettings)

    # ---- resolved accessors -------------------------------------------------

    def resolved_data_root(self) -> Path:
        import os

        env = os.environ.get("TANDEM_DATA_ROOT", "").strip()
        if env:
            return Path(env).expanduser().resolve()
        if self.data_root:
            return Path(self.data_root).expanduser().resolve()
        return paths.default_data_root()

    def resolved_runtime_dir(self) -> Path:
        import os

        env = os.environ.get("TANDEM_RUNTIME_DIR", "").strip()
        if env:
            return Path(env).expanduser().resolve()
        if self.runtime_dir:
            return Path(self.runtime_dir).expanduser().resolve()
        return paths.default_runtime_dir()

    def profiles_root(self) -> Path:
        return self.resolved_data_root() / "profiles"


_cache: Settings | None = None


def load(*, force: bool = False) -> Settings:
    global _cache
    if _cache is not None and not force:
        return _cache
    path = paths.config_file()
    if not path.is_file():
        _cache = Settings()
        return _cache
    try:
        raw = tomlkit.parse(path.read_text())
    except Exception as exc:
        raise TandemError(
            f"{path} is not valid TOML: {exc}",
            hint="Fix it by hand, or delete it to start from defaults.",
        ) from exc
    try:
        _cache = Settings.model_validate(_plain(raw))
    except Exception as exc:
        raise TandemError(
            f"{path} has settings tandem does not understand.\n{exc}",
            hint="Run `tandem config list` to see the valid keys.",
        ) from exc
    return _cache


def save(settings: Settings) -> Path:
    """Write settings back, preserving comments and key order where possible."""
    global _cache
    path = paths.config_file()
    paths.ensure_dir(path.parent)
    doc = tomlkit.parse(path.read_text()) if path.is_file() else tomlkit.document()
    for key, value in settings.model_dump(mode="python").items():
        doc[key] = value
    path.write_text(tomlkit.dumps(doc))
    _cache = settings
    return path


def _plain(obj):
    """tomlkit returns proxy types; pydantic wants plain dict/list/scalars."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    if isinstance(obj, str):
        return str(obj)
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    return obj


# --- dotted get/set for `tandem config get|set` -----------------------------

def get_dotted(settings: Settings, key: str):
    node = settings
    for part in key.split("."):
        if isinstance(node, BaseModel):
            if part not in type(node).model_fields:
                raise TandemError(
                    f"Unknown setting {key!r} (no field {part!r}).",
                    hint="Run `tandem config list` to see every key.",
                )
            node = getattr(node, part)
        else:
            raise TandemError(f"Unknown setting {key!r}.", hint="Run `tandem config list`.")
    return node


def set_dotted(settings: Settings, key: str, value: str) -> Settings:
    parts = key.split(".")
    node = settings
    for part in parts[:-1]:
        if not isinstance(node, BaseModel) or part not in type(node).model_fields:
            raise TandemError(f"Unknown setting {key!r}.", hint="Run `tandem config list`.")
        node = getattr(node, part)
    leaf = parts[-1]
    if not isinstance(node, BaseModel) or leaf not in type(node).model_fields:
        raise TandemError(f"Unknown setting {key!r}.", hint="Run `tandem config list`.")

    field = type(node).model_fields[leaf]
    coerced = _coerce(value, field.annotation)
    setattr(node, leaf, coerced)
    # Re-validate the whole tree so a bad value is rejected here, not at next load.
    return Settings.model_validate(settings.model_dump())


def _coerce(value: str, annotation):
    if annotation is bool:
        low = value.strip().lower()
        if low in {"1", "true", "yes", "on"}:
            return True
        if low in {"0", "false", "no", "off"}:
            return False
        raise TandemError(f"{value!r} is not a boolean.", hint="Use true or false.")
    if annotation is int:
        try:
            return int(value)
        except ValueError as exc:
            raise TandemError(f"{value!r} is not an integer.") from exc
    return value


def flatten(settings: Settings, prefix: str = "") -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    for name in type(settings).model_fields:
        value = getattr(settings, name)
        key = f"{prefix}{name}"
        if isinstance(value, BaseModel):
            out.extend(flatten(value, prefix=f"{key}."))
        else:
            out.append((key, value))
    return out
