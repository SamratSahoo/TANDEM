"""The teleop driver's runtime, as a recipe: DROID's workstation side, oculus_reader, one small pixi environment.

``tandem executors install teleop`` builds it (``tandem init`` offers to), so nobody has to keep a DROID
checkout and conda environment of their own and point tandem at them. ``teleop.droid_dir`` and
``teleop.python`` still override it for anyone who does.

The driver drives the arm through the NUC's DROID server, which runs the inverse kinematics. What the
workstation imports -- ``droid.stable_camera_env``, ``droid.controllers.oculus_controller`` and what they
pull in -- is numpy, scipy, zerorpc, gym, OpenCV, the ZED bindings and, for VR, oculus_reader. The
environment is exactly that (``pixi.toml`` on DROID's TANDEM branch), not DROID's whole dependency list:
no mujoco, dm_control, polymetis or torch. The droid package is not installed into it; the driver runs
with both trees on ``PYTHONPATH`` (``pythonpath``).

Layout::

    <runtimes>/teleop/
        droid/            SamratSahoo/droid at the pinned TANDEM commit; droid/.pixi -> ../env
        oculus_reader/    rail-berkeley/oculus_reader at the commit DROID's submodule names
        env/              the pixi environment

Imported to list executors, so it declares and does nothing: no file is read until an install runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tandem.core import paths
from tandem.planners.base import SourcePin
from tandem.planners.runtime import BuildStep, PixiEnvironment, RecipeRuntime, RuntimeRecipe, Source

#: The command that builds this runtime, for hints and listings.
INSTALL_COMMAND = "tandem executors install teleop"
#: The ZED SDK's installer for its Python API, where the SDK puts it.
ZED_PYTHON_API = "/usr/local/zed/get_python_api.py"

# DROID's TANDEM branch is its main plus the two things a managed runtime needs: pixi.toml and pixi.lock
# for the workstation side, and droid.misc.parameters reading nuc_ip from $DROID_NUC_IP, so the rig's
# robot.host reaches it (the camera serials were already read from $TIPTOP_*_CAMERA_ID). Its install-zed
# task also puts NumPy < 2 back after the ZED installer pulls in NumPy 2, which OpenCV 4.6 cannot import.
DROID = Source(
    SourcePin(
        "droid",
        "https://github.com/SamratSahoo/droid.git",
        "a2ebef2e3729435e9f518eaced29c35d7e4070f5",
        ref="TANDEM",
    ),
    trim=(
        "docs",  # 27 MB of the hardware guide's photos
    ),
    marker="pixi.toml",
)

# DROID vendors oculus_reader as a git submodule, which an exported tree does not carry, so it is a source
# of its own, at the commit that submodule names. Its APK is a Git LFS file and arrives as a pointer: a
# headset that already has the teleop app (any headset DROID's VR teleop has run on) never needs it, and
# a new one gets the app once (https://prpl-group.com/tandem/docs/teleop/).
OCULUS_READER = Source(
    SourcePin(
        "oculus_reader",
        "https://github.com/rail-berkeley/oculus_reader.git",
        "de73f3d259b3c41c4564f70a64682e24aa3ac31c",
        ref="main",
    ),
    trim=(
        "app_source",  # the headset app's sources and 3D assets
    ),
    marker="oculus_reader/reader.py",
)

RECIPE = RuntimeRecipe(
    planner="teleop",
    title="DROID teleop",
    install_command=INSTALL_COMMAND,
    sources=(DROID, OCULUS_READER),
    environment=PixiEnvironment(manifest="droid/pixi.toml", home="env"),
    steps=(
        # The ZED cameras' Python bindings come with the ZED SDK, not from PyPI. Optional for the same reason
        # as TiPToP's step: the SDK is a system install tandem cannot make. Without it the driver cannot open
        # the cameras, so a hand-off has nothing to record from.
        BuildStep(
            name="zed",
            task="install-zed",
            optional=True,
            requires=(ZED_PYTHON_API,),
            produces=("env/envs/default/lib/python3*/site-packages/pyzed",),
            description="installing the ZED Python API (pyzed) from the ZED SDK",
            label="ZED Python API",
            done="installed",
            todo="not installed",
            missing=f"the ZED SDK is not installed (no {ZED_PYTHON_API}), so the teleop driver cannot open the "
            "cameras. Install it from https://www.stereolabs.com/developers/release, then run "
            f"`{INSTALL_COMMAND}`.",
        ),
    ),
    notes=(
        "Building the teleop driver's environment: DROID's workstation side (numpy, scipy, zerorpc, gym, "
        "OpenCV), oculus_reader for VR, and the ZED Python API when the ZED SDK is installed.",
        "About 100 MB of downloads; a few minutes.",
    ),
)


def root(settings: Any = None) -> Path:
    """Where the teleop runtime lives: beside the planners' runtimes (``$TANDEM_RUNTIMES_DIR``)."""
    return paths.runtimes_dir() / "teleop"


def runtime(settings: Any = None) -> RecipeRuntime:
    return RecipeRuntime(RECIPE, root(settings))


def pythonpath(rt: RecipeRuntime) -> list[Path]:
    """What the driver needs on ``PYTHONPATH``: DROID's tree (for ``droid``) and oculus_reader's."""
    return [rt.source_dir("droid"), rt.source_dir("oculus_reader")]
