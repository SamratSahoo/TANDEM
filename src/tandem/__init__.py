"""TANDEM: Task and Motion Planning with As-Needed Demonstrations.

A data-collection system for fine-tuning vision-language-action models on long-horizon manipulation
tasks. A vision-language model splits an instruction into an ordered list of phases. Each phase is
either a sub-goal for a task and motion planner or a step only a person can do, and for the second
kind the model invents the predicates and the "magic operator" that describe it. The planner runs the
robot's phases, a person teleoperates the rest, every human phase is checked from a photo, and a
trial's legs are merged into one demonstration. `tandem` is the command line; this module is the
library.

The library surface is small, and every name is resolved on first use: ``import tandem`` imports
nothing but this file, so it costs nothing on a laptop with no GPU, no robot and no planner.

Decompose a task from a photo (``tandem.api``; what `tandem plan` runs)::

    plan = tandem.plan_task("put the bread in the box", "workspace.png", planner="tiptop")

- ``plan_task`` / ``plan_task_async``: an instruction and a workspace photo in, the ``PhasePlan`` a
  session would walk out. Needs a Gemini key, and nothing else.
- ``PhasePlan``: that plan: its phases, invented predicates, magic operators, and the record
  ``hitl.json`` is written from (``tandem.planning.plan``).
- ``PlanningConfig``: the phase-planning settings, a profile's ``hitl:`` block as a plain dataclass.

Add a planner or a human executor. The full kits are ``tandem.planners`` and ``tandem.executors``;
``docs/ADDING_A_PLANNER.md`` and ``docs/ADDING_A_HUMAN_EXECUTOR.md`` walk through each::

    class MyPlanner(tandem.Planner):
        info = tandem.PlannerInfo(name="mine", display_name="Mine")
        CAPABILITIES = tandem.Capabilities(name="mine", goal_predicates=..., ...)

        def perceive(self, *, task_hint, save_dir, reset_arm=True, open_gripper=False): ...
        def plan(self, scene_id, goal, *, surfaces=frozenset(), save_dir, reuse_skeleton=None): ...
        def execute(self, plan_handle, leg, *, save_dir, should_stop=None): ...

    tandem.register_backend("mine", MyPlanner)   # a package uses the `tandem.planners` entry point

- ``Planner``, ``SidecarPlanner``: the base classes a planner is written against, in tandem's process
  or in an environment of its own.
- ``Capabilities``, ``PlannerInfo``, ``Predicate``, ``Parameter``, ``RuntimeRecipe``: what a planner
  declares -- its goal language, what a catalog says about it, the runtime it is built in.
- ``register_backend``: make a planner available by name (a package uses the ``tandem.planners``
  entry point instead).
- ``register_human_executor``: the same for whoever carries out a human phase (entry point:
  ``tandem.human_executors``).

``TandemError`` is the error tandem raises on purpose: a message, and a hint saying what to do about
it.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

# Public name -> the module that defines it. Resolved on first use (``__getattr__``), never here.
_EXPORTS = {
    "plan_task": "tandem.api",
    "plan_task_async": "tandem.api",
    "PhasePlan": "tandem.planning.plan",
    "PlanningConfig": "tandem.planning.config",
    "Planner": "tandem.planners.sdk",
    "SidecarPlanner": "tandem.planners.sidecar",
    "Capabilities": "tandem.planners.base",
    "PlannerInfo": "tandem.planners.base",
    "Predicate": "tandem.planning.symbols",
    "Parameter": "tandem.planning.symbols",
    "RuntimeRecipe": "tandem.planners.runtime",
    "register_backend": "tandem.planners.registry",
    "register_planner": "tandem.planners.registry",
    "register_human_executor": "tandem.executors.base",
    "TandemError": "tandem.core.errors",
}

__all__ = [
    "Capabilities",
    "Parameter",
    "PhasePlan",
    "Planner",
    "PlannerInfo",
    "PlanningConfig",
    "Predicate",
    "RuntimeRecipe",
    "SidecarPlanner",
    "TandemError",
    "__version__",
    "plan_task",
    "plan_task_async",
    "register_backend",
    "register_human_executor",
    "register_planner",
]

if TYPE_CHECKING:  # pragma: no cover - for editors and type checkers only
    from tandem.api import plan_task, plan_task_async
    from tandem.core.errors import TandemError
    from tandem.executors.base import register_human_executor
    from tandem.planners.base import Capabilities, PlannerInfo
    from tandem.planners.registry import register_backend, register_planner
    from tandem.planners.runtime import RuntimeRecipe
    from tandem.planners.sdk import Planner
    from tandem.planners.sidecar import SidecarPlanner
    from tandem.planning.config import PlanningConfig
    from tandem.planning.plan import PhasePlan
    from tandem.planning.symbols import Parameter, Predicate


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        # AttributeError and nothing else: `from tandem import planners` asks for the attribute first,
        # and falls back to importing the submodule only on exactly this error.
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
