"""The arms TiPToP drives, and how it reaches each one.

Two readers need this: the options schema (``options.py``), which refuses any other ``robot.type``, and
the catalog (``factory.py``), which says what a rig needs. The catalog said "a Franka FR3 (or UR5)"
while the schema also accepted panda and panda_robotiq, so they now read one table. It is a module of
its own, stdlib only, because the catalog is imported just to list planners, on a laptop, and the
schema brings pydantic with it.
"""

from __future__ import annotations

_BAMBOO = "over the bamboo-polymetis shim"
# tiptop lists ur_rtde as its `ur5` extra, and the runtime installs tiptop without extras (the pinned
# pixi.lock has no ur-rtde), so a UR5 client fails to import until somebody adds it.
_RTDE = "over ur_rtde (tiptop's ur5 extra; the runtime does not install it)"

#: ``robot.type`` -> (the arm and its gripper, how tiptop reaches it).
#:
#: The types both halves of the pinned planner know: tiptop's robot client and cuRobo solvers
#: (get_robot_client, and get_ik_solver / get_motion_gen under build_curobo_solvers), which raise
#: "Unknown robot type" at warm-up for anything else, and cuTAMP's validate_tamp_config, which tiptop
#: hands the same name as TAMPConfiguration.robot at every plan. Left out on purpose:
#:   - "fr3" (a Franka Hand on an FR3): tiptop calls it "fr3" and cuTAMP "fr3_franka", so neither name
#:     gets through both -- it warms and then fails every plan.
#:   - the bimanual YAM types: the recipe trims their meshes, and bimanual is out of scope.
#: tests/test_review_tiptop.py reads the set out of the pinned sources, so a bump that changes it fails
#: there.
ARMS: dict[str, tuple[str, str]] = {
    "fr3_robotiq": ("a Franka FR3 with a Robotiq 2F-85", _BAMBOO),
    "panda_robotiq": ("a Franka Panda with a Robotiq 2F-85", _BAMBOO),
    "panda": ("a Franka Panda with the Franka Hand", _BAMBOO),
    "ur5": ("a UR5 with a Robotiq gripper", _RTDE),
}

ROBOT_TYPES = frozenset(ARMS)


def requirement() -> str:
    """The arm a rig needs, as the catalog lists it: every type the options accept, by how it is reached."""
    by_link: dict[str, list[str]] = {}
    for robot_type, (arm, link) in ARMS.items():
        by_link.setdefault(link, []).append(f"{arm} ({robot_type})")
    ways = [f"{_one_of(arms)}, {link}" for link, arms in by_link.items()]
    return "one of these arms (robot.type): " + "; or ".join(ways)


def _one_of(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]
