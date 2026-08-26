"""Which planner backends tandem knows about, by name.

Resolution is loud. A profile naming a backend that does not exist is a typo, and the failure mode
that must never happen is quietly falling back to a default -- a session that plans with a different
planner from the one the config asked for produces a dataset nobody can interpret afterwards. Same
discipline as the ``tamp`` keys: unknown name, clear error, nearest suggestion.

The registry maps a name to a *factory* rather than an instance, because building a backend opens
cameras and a robot connection, and importing one may need the heavy environment. Listing the
available names must cost nothing.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from typing import Any

from tandem.core.errors import TandemError
from tandem.planners.base import Capabilities

# name -> (import path of the capability declaration, import path of the backend class)
_BACKENDS: dict[str, tuple[str, str]] = {
    "tiptop": (
        "tandem.planners.tiptop.capabilities:CAPABILITIES",
        "tandem.planners.tiptop.backend:TiptopBackend",
    ),
}


def available() -> list[str]:
    """Every backend name, for an error message or a `--help` line."""
    return sorted(_BACKENDS)


def _resolve(spec: str) -> Any:
    module_name, _, attribute = spec.partition(":")
    from importlib import import_module

    return getattr(import_module(module_name), attribute)


def _entry(name: str) -> tuple[str, str]:
    key = (name or "").strip()
    if key in _BACKENDS:
        return _BACKENDS[key]
    suggestion = difflib.get_close_matches(key, _BACKENDS, n=1, cutoff=0.6)
    hint = f"Did you mean {suggestion[0]!r}?" if suggestion else f"Known backends: {', '.join(available())}."
    raise TandemError(f"Unknown planner backend {name!r}.", hint=hint)


def capabilities(name: str) -> Capabilities:
    """What a backend can be asked for, without building it.

    This is what makes ``tandem plan`` work on a laptop: a decomposition can be proposed and checked
    against the real planner's goal language with no planner, no runtime and no robot in sight.
    """
    return _resolve(_entry(name)[0])


def backend_class(name: str) -> Callable[..., Any]:
    """The backend class, imported on demand."""
    return _resolve(_entry(name)[1])
