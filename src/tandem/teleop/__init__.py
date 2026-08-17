"""The teleop half of the TAMP⇄teleop hand-off.

"Switch to teleop" lends the arm to a human mid-task and takes it back afterwards, without
ever returning to home. It is not a preempt: the planner finishes the current plan step, saves
the partial rollout, releases the robot **and closes its cameras**, and waits. A teleop process
then drives the arm from a VR controller or a SpaceMouse, capturing in exactly the same raw
format. When control comes back, the planner re-opens the cameras, reconnects, and replans the
same task from wherever the human left the arm.

Every leg is stamped with a shared ``trajectory_id`` and joined into one trajectory by
``tandem.core.merge`` as soon as the operator labels it.

**This runs under a DROID environment's interpreter, not tandem's.** The driver needs
``droid.controllers.oculus_controller`` and ``droid.stable_camera_env``, which are a separate
install with their own hardware bindings — so tandem ships the driver and the user points it
at their DROID checkout (``tandem config set teleop.droid_dir …`` / ``teleop.python``). All
``droid.*`` imports inside the driver are lazy, so this package stays importable anywhere.
"""

from __future__ import annotations

from pathlib import Path


def driver_path() -> Path:
    """The teleop driver script, to be executed by the DROID environment's interpreter."""
    return Path(__file__).resolve().parent / "driver.py"


def helper_paths() -> list[Path]:
    """Modules the driver imports from its own directory at runtime."""
    here = Path(__file__).resolve().parent
    return [here / "raw_episode.py", here / "spacemouse.py"]
