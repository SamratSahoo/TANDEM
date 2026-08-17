"""API keys and tokens.

Stored in ``~/.config/tandem/credentials.toml`` at mode 0600, separate from config.toml so
the settings file stays safe to paste into an issue.

Resolution order for the Gemini key is deliberately env-first: a shell that already exports
GEMINI_API_KEY (a shared workstation, a CI job) should win over a stale stored key.
"""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path

import tomlkit

from tandem.core import paths
from tandem.core.errors import TandemError

GEMINI_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
HF_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")

_KEY_FIELDS = ("gemini_api_key", "hf_token")


def _read() -> dict:
    path = paths.credentials_file()
    if not path.is_file():
        return {}
    try:
        return dict(tomlkit.parse(path.read_text()))
    except Exception as exc:
        raise TandemError(
            f"{path} is not valid TOML: {exc}",
            hint="Delete it and re-run `tandem config set-gemini-key`.",
        ) from exc


def _write(data: dict) -> Path:
    path = paths.credentials_file()
    paths.ensure_dir(path.parent)
    doc = tomlkit.document()
    doc.add(tomlkit.comment("tandem credentials — keep this file private (mode 0600)."))
    for key, value in data.items():
        if value:
            doc[key] = value
    # Create with 0600 from the start; writing then chmod'ing leaves a window where the key
    # is world-readable.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w") as fh:
        fh.write(tomlkit.dumps(doc))
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    return path


def _from_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def gemini_api_key() -> str | None:
    return _from_env(GEMINI_ENV_VARS) or (_read().get("gemini_api_key") or None)


def gemini_key_source() -> str:
    if _from_env(GEMINI_ENV_VARS):
        return "env"
    if _read().get("gemini_api_key"):
        return "file"
    return "none"


def set_gemini_api_key(key: str) -> Path:
    key = key.strip()
    if not key:
        raise TandemError("Refusing to store an empty Gemini API key.")
    data = _read()
    data["gemini_api_key"] = key
    return _write(data)


def clear_gemini_api_key() -> Path:
    data = _read()
    data.pop("gemini_api_key", None)
    return _write(data)


def hf_token() -> str | None:
    stored = _read().get("hf_token") or None
    if stored:
        return stored
    env = _from_env(HF_ENV_VARS)
    if env:
        return env
    cached = Path.home() / ".cache" / "huggingface" / "token"
    if cached.is_file():
        value = cached.read_text().strip()
        if value:
            return value
    return None


def hf_token_source() -> str:
    if _read().get("hf_token"):
        return "file"
    if _from_env(HF_ENV_VARS):
        return "env"
    if (Path.home() / ".cache" / "huggingface" / "token").is_file():
        return "cache"
    return "none"


def set_hf_token(token: str) -> Path:
    data = _read()
    data["hf_token"] = token.strip()
    return _write(data)


def mask(value: str | None) -> str:
    """`sk-abc…wxyz` — enough to tell two keys apart, not enough to use one."""
    if not value:
        return "—"
    if len(value) <= 8:
        return "•" * len(value)
    return f"{value[:4]}…{value[-4:]}"


# --- redaction ---------------------------------------------------------------

# Long opaque tokens: Google API keys (AIza...), HF tokens (hf_...), and anything that looks
# like KEY=<40+ chars> in a log line.
_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"hf_[0-9A-Za-z]{20,}"),
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\b(\s*[=:]\s*)(\S{8,})"),
]


def redact(text: str) -> str:
    """Scrub secrets out of anything that reaches a log, the API, or the terminal.

    Also scrubs the *live* key values, because a subprocess may echo them back in a form the
    generic patterns miss.
    """
    if not text:
        return text
    out = text
    for live in (gemini_api_key(), _read().get("hf_token")):
        if live and len(live) >= 8:
            out = out.replace(live, "«redacted»")
    out = _PATTERNS[0].sub("«redacted»", out)
    out = _PATTERNS[1].sub("«redacted»", out)
    out = _PATTERNS[2].sub(lambda m: f"{m.group(1)}{m.group(2)}«redacted»", out)
    return out
