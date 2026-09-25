"""Two planners tandem has never heard of, written with the planner SDK the way a plugin would be.

Both drive ``toy_world``: items on a floor, dropped into bins. The goal language is not cuTAMP's --
``InBin(?obj: item, ?bin: container)``, no table, no exclusivity, an item that can be re-binned --
so anything TipTop-shaped left in the generic code shows up as a failure here.

- ``ToyPlanner`` is a ``Planner``: the world lives in tandem's own process, and the class is all
  there is to it.
- ``ToySidecarPlanner`` is a ``SidecarPlanner``: the same world served by ``toy_sidecar.py`` in a
  child process, launched with this interpreter because it declares no runtime to launch in. It
  declares cooperative stop, so a stop crosses the process boundary the way it would for a real one.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from toy_world import ITEMS, ToyWorld

from tandem.planners import (
    Capabilities,
    ExecuteResult,
    Parameter,
    Planner,
    PlannerInfo,
    PlanResult,
    Predicate,
    SceneView,
    SidecarPlanner,
)

IN_BIN = Predicate("InBin", (Parameter("obj", "item"), Parameter("bin", "container")))

TOY_CAPABILITIES = Capabilities(
    name="toy",
    goal_predicates={"InBin": IN_BIN},
    robot_description="drop an item into a bin",
    goal_predicate_wire_names={"InBin": "in_bin"},
    achievable_predicates=frozenset({"InBin"}),
    reserved_predicate_names=frozenset({"InBin"}),
    movable_type="item",
    surface_type="container",
    predicate_descriptions={"InBin": "{0} is inside {1}"},
    checkable_predicates=frozenset({"InBin"}),
    # An item can go into one bin and then another within one plan, and nothing symbolic orders two
    # drops: neither of cuTAMP's assumptions holds here.
    one_pick_per_object=False,
    initial_state_is_clean=False,
    supports_cooperative_stop=True,
    moved_arguments={"InBin": 0},
    robot_operators=("Drop(?obj: item, ?bin: container)",),
    supports_movable_restriction=True,
    supports_return_home=True,
)


#: What the toys read of the machine: rig.yml's planners.<name>. Declared, so a key in a profile is refused.
TOY_RIG_OPTIONS = {"station": "which of this machine's bin stations the arm drops into"}
DEFAULT_STATION = "bench-1"


class ToyPlanner(Planner):
    """The toy world as an in-process planner."""

    info = PlannerInfo(name="toy", display_name="Toy", summary="Drops items into bins, in memory.")
    CAPABILITIES = TOY_CAPABILITIES
    OPTIONS = {"items": "the items on the floor when the session starts"}
    RIG_OPTIONS = TOY_RIG_OPTIONS

    def __init__(self, ctx=None) -> None:
        super().__init__(ctx)
        self.world = ToyWorld(items=tuple(self.options.get("items") or ITEMS))
        # A task's settings from the profile, the machine's from the rig: what a real planner reads of each.
        self.station = self.rig_options.get("station", DEFAULT_STATION)
        self.robot_host = self.rig.robot.host if self.rig is not None else None

    def perceive(
        self, *, task_hint: str, save_dir: Path, reset_arm: bool = True, open_gripper: bool = False
    ) -> SceneView:
        return SceneView.from_dict(
            self.world.perceive(
                task_hint=task_hint, save_dir=str(save_dir), reset_arm=reset_arm, open_gripper=open_gripper
            )
        )

    def plan(
        self,
        scene_id,
        goal,
        *,
        surfaces=frozenset(),
        movables=None,
        return_home=True,
        save_dir,
        reuse_skeleton=None,
    ) -> PlanResult:
        return PlanResult.from_dict(
            self.world.plan(
                scene_id=scene_id,
                goal=[atom.to_dict() for atom in goal],
                surfaces=sorted(surfaces),
                save_dir=str(save_dir),
                movables=sorted(movables) if movables is not None else None,
                return_home=return_home,
            )
        )

    def execute(self, plan_handle, leg, *, save_dir, should_stop=None) -> ExecuteResult:
        return ExecuteResult.from_dict(
            self.world.execute(
                plan_handle=plan_handle, leg=leg.to_dict(), save_dir=str(save_dir), should_stop=should_stop
            )
        )

    def capture_frame(self, *, camera: str = "external") -> str:
        return self.world.capture_frame(camera=camera)["path"]

    def release_hardware(self) -> None:
        self.world.release_hardware()

    def reacquire_hardware(self) -> None:
        self.world.reacquire_hardware()


class ToySidecarPlanner(SidecarPlanner):
    """The toy world behind a sidecar. ``flags`` go to ``toy_sidecar.py`` to make it misbehave."""

    info = PlannerInfo(
        name="toy-sidecar", display_name="Toy (sidecar)", summary="Drops items into bins, next door."
    )
    CAPABILITIES = replace(TOY_CAPABILITIES, name="toy-sidecar")
    SIDECAR = "toy_sidecar.py"
    TIMEOUTS = {"warm": 30.0, "perceive": 30.0, "plan": 30.0, "execute": 30.0}
    RIG_OPTIONS = TOY_RIG_OPTIONS

    def __init__(self, ctx=None, *, flags: tuple[str, ...] = (), **kwargs: Any) -> None:
        super().__init__(ctx, **kwargs)
        self.flags = tuple(flags)

    def warm_args(self) -> dict[str, Any]:
        # The sidecar has no rig of its own to read: the machine's settings go over with the warm-up.
        return {
            **super().warm_args(),
            "station": self.rig_options.get("station", DEFAULT_STATION),
            "robot_host": self.rig.robot.host if self.rig is not None else None,
        }

    def launch_command(self) -> list[str]:
        return [*super().launch_command(), *self.flags]
