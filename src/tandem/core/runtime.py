"""Deprecated: TiPToP's runtime moved to ``tandem.planners.tiptop.runtime``.

This path is kept, and only re-exports, because it was the one way a script built a session before
tandem drove more than one planner (``Session(profile, Runtime(...))``) or found the GPU runtime
(``runtime.default()``). A session now builds its planner's runtime itself, through the registry,
and a script that wants one asks ``tandem.planners.registry.runtime(name, settings)``, which is
right whichever planner a profile names. Nothing in tandem imports this module.
"""

from __future__ import annotations

import warnings

from tandem.planners.tiptop.runtime import (  # noqa: F401  (re-exported)
    CUROBO,
    CUTAMP,
    STAMP_FILE,
    TIPTOP,
    Runtime,
    RuntimeStatus,
    TiptopRuntime,
    default,
)

warnings.warn(
    "tandem.core.runtime is deprecated: TiPToP's runtime is tandem.planners.tiptop.runtime, and any "
    "planner's is tandem.planners.registry.runtime(name, settings).",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["CUROBO", "CUTAMP", "STAMP_FILE", "TIPTOP", "Runtime", "RuntimeStatus", "TiptopRuntime", "default"]
