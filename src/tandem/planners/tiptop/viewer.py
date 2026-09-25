#!/usr/bin/env python3
"""Replay a leg tandem recorded with TiPToP in Rerun: tiptop's own viewer, with one stale gate opened.

Run inside the planner's runtime, the way the sidecar is (``TiptopRuntime.replay_command``): the viewer
needs cuRobo and cuTAMP to load the robot model and the leg's saved TAMP environment, and those are
there and tandem is not. So, like the sidecar, it imports nothing from ``tandem``. Its arguments are
``viz-tiptop-run``'s own, passed through untouched (``--save-dir DIR``, ``--no-visualize-grasps``, ...).

Why not run ``viz-tiptop-run`` itself: the pinned tiptop's viewer refuses every plan the same tiptop
writes. Its gate compares a plan's version with "1.0.0", the only version there was when the gate was
written, and ``serialize_plan`` has since moved to "1.4.0". The schema is semver -- a minor version only
adds optional fields (``serialize_plan``'s docstring), and ``load_tiptop_plan`` already reads the one
field added since (``cost``, 1.1.0) -- so every 1.x plan is one the viewer can draw. The gate is opened
for 1.x here rather than patched in the recipe, because a recipe patch changes the installed tree, and
every workstation's runtime would be fetched and rebuilt for the sake of a viewer. A 2.x plan is still
refused, by the viewer itself.
"""

from __future__ import annotations

import sys


def accept_any_1x(load):
    """``load_tiptop_plan``, reporting any 1.x plan as the "1.0.0" the viewer's gate asks for."""

    def load_plan(path):
        plan = load(path)
        if str(plan.get("version", "")).split(".")[0] == "1":
            plan["version"] = "1.0.0"
        return plan

    return load_plan


def main(argv: list[str] | None = None) -> int:
    from tiptop.scripts import viz_tiptop_run as viewer

    # The viewer calls load_tiptop_plan through its own module's globals, so this is the one it uses.
    viewer.load_tiptop_plan = accept_any_1x(viewer.load_tiptop_plan)
    sys.argv = ["viz-tiptop-run", *(sys.argv[1:] if argv is None else argv)]
    viewer.viz_tiptop_run_entrypoint()
    return 0


if __name__ == "__main__":
    sys.exit(main())
