"""Task and motion planners tandem can drive, behind one narrow protocol -- and the kit to write one.

Everything a planner author needs is importable from here::

    from tandem.planners import Capabilities, Parameter, Planner, PlannerInfo, Predicate, register_backend

- ``Planner``: the base class. Declare ``info`` and ``CAPABILITIES``, implement ``perceive``,
  ``plan`` and ``execute``; every other verb has a default, and the class is its own factory.
- ``SidecarPlanner``: the same, for a planner that runs in an environment of its own; its work is
  done by a sidecar script written with ``tandem_sidecar`` (``planners/sidecar_kit``).
- ``RuntimeRecipe`` (with ``Source``, ``SourcePin``, ``PixiEnvironment``, ``BuildStep``, ``Asset``):
  the runtime such a planner needs built, declared rather than scripted.
- ``register_backend``: make a planner available under a name. A package does the same with a
  ``tandem.planners`` entry point.
- ``tandem.planners.testing``: a conformance kit a planner's own test suite subclasses.

Underneath: ``base`` is the protocol and the wire types, ``registry`` resolves a planner by name,
``rpc`` is the channel to a sidecar, ``runtime`` builds a recipe, and ``tiptop`` is the first
implementation.

The names above are resolved on first use, not when this package is imported. ``tandem.planners.base``
is imported by the phase planner, and the planner package's ``__init__`` runs first: importing the SDK
here would put the SDK, the registry and the runtime builder on the path of every module that only
wanted the protocol's types -- and would close an import cycle through ``tandem.planning``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# Public name -> the module that defines it.
_EXPORTS = {
    "Planner": "tandem.planners.sdk",
    "UnsupportedVerb": "tandem.planners.sdk",
    "SidecarPlanner": "tandem.planners.sidecar",
    "Capabilities": "tandem.planners.base",
    "PlannerInfo": "tandem.planners.base",
    "SourcePin": "tandem.planners.base",
    "GoalAtom": "tandem.planners.base",
    "SceneView": "tandem.planners.base",
    "PlanResult": "tandem.planners.base",
    "ExecuteResult": "tandem.planners.base",
    "LegSpec": "tandem.planners.base",
    "BackendContext": "tandem.planners.base",
    "BackendError": "tandem.planners.base",
    "TampBackend": "tandem.planners.base",
    "RuntimeRecipe": "tandem.planners.runtime",
    "Source": "tandem.planners.runtime",
    "PixiEnvironment": "tandem.planners.runtime",
    "BuildStep": "tandem.planners.runtime",
    "Asset": "tandem.planners.runtime",
    "Predicate": "tandem.planning.symbols",
    "Parameter": "tandem.planning.symbols",
    "register_backend": "tandem.planners.registry",
}

__all__ = [
    "Asset",
    "BackendContext",
    "BackendError",
    "BuildStep",
    "Capabilities",
    "ExecuteResult",
    "GoalAtom",
    "LegSpec",
    "Parameter",
    "PixiEnvironment",
    "PlanResult",
    "Planner",
    "PlannerInfo",
    "Predicate",
    "RuntimeRecipe",
    "SceneView",
    "SidecarPlanner",
    "Source",
    "SourcePin",
    "TampBackend",
    "UnsupportedVerb",
    "register_backend",
]

if TYPE_CHECKING:  # pragma: no cover - for editors and type checkers only
    from tandem.planners.base import (
        BackendContext,
        BackendError,
        Capabilities,
        ExecuteResult,
        GoalAtom,
        LegSpec,
        PlannerInfo,
        PlanResult,
        SceneView,
        SourcePin,
        TampBackend,
    )
    from tandem.planners.registry import register_backend
    from tandem.planners.runtime import Asset, BuildStep, PixiEnvironment, RuntimeRecipe, Source
    from tandem.planners.sdk import Planner, UnsupportedVerb
    from tandem.planners.sidecar import SidecarPlanner
    from tandem.planning.symbols import Parameter, Predicate


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        # AttributeError, and nothing else: `from tandem.planners import registry` asks for the
        # attribute first and falls back to importing the submodule only on exactly this error.
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
