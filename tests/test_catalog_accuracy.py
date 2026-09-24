"""What the planner catalog says is what is true.

* `tandem planners info tiptop` said a rig needs "a Franka FR3 (or UR5)", while the options accept
  panda and panda_robotiq as well. The requirement is now written from the same table the options
  check robot.type against.
* `tandem planners list --json` listed every source of a planner that had never been installed as
  "mismatched", as if it had been built from the wrong commits. A runtime never built has nothing at
  the wrong commit; "not installed" says it. One built at other commits is still outdated, and says
  which sources.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from tandem.cli import planners as planners_cli
from tandem.cli.app import app
from tandem.core import settings as settings_mod
from tandem.planners.base import RuntimeStatus, SourcePin

# --- the arms ------------------------------------------------------------------------------------------


def test_the_catalog_names_every_arm_the_options_accept_and_no_other():
    from tandem.planners.tiptop import arms
    from tandem.planners.tiptop.factory import INFO
    from tandem.planners.tiptop.options import ROBOT_TYPES, TiptopOptions

    (line,) = [line for line in INFO.requires if "robot.type" in line]
    named = {robot_type for robot_type in arms.ARMS if f"({robot_type})" in line}
    assert named == set(ROBOT_TYPES) == {"fr3_robotiq", "panda_robotiq", "panda", "ur5"}
    for robot_type in named:
        assert TiptopOptions.model_validate({"robot": {"type": robot_type}}).robot.type == robot_type
    # Each named as the arm it is, and how tiptop reaches it: a UR5 is not behind the bamboo shim.
    assert "a Franka Panda with the Franka Hand (panda)" in line
    assert line.index("bamboo-polymetis") < line.index("(ur5)") < line.index("ur_rtde")


def test_planners_info_shows_the_arms():
    shown = CliRunner().invoke(app, ["planners", "info", "tiptop"])
    assert shown.exit_code == 0, shown.output
    flat = " ".join(shown.output.split())
    for robot_type in ("fr3_robotiq", "panda_robotiq", "panda", "ur5"):
        assert f"({robot_type})" in flat, robot_type
    assert "Franka FR3 (or UR5)" not in flat


# --- mismatched ----------------------------------------------------------------------------------------

PINS = (SourcePin("planner", "https://example.invalid/p.git", "a" * 40), SourcePin("solver", "u", "b" * 40))


@pytest.mark.parametrize(
    ("status", "outdated"),
    [
        (RuntimeStatus(), ()),
        (RuntimeStatus(installed=True, pins=PINS), ()),
        (RuntimeStatus(pins=(PINS[0], SourcePin("solver", "u", "c" * 40))), ("solver",)),
        (RuntimeStatus(installed=True, pins=PINS[:1]), ("solver",)),
    ],
    ids=["never built", "current", "half-built at an old commit", "built without a source"],
)
def test_only_a_runtime_that_was_built_has_sources_at_the_wrong_commit(status, outdated):
    assert status.outdated(PINS) == outdated


def test_what_decides_whether_a_runtime_is_current_is_unchanged():
    # A runtime that cannot say what it was built from matches nothing, so an install still builds it.
    assert RuntimeStatus(installed=True).mismatched(PINS) == ("planner", "solver")


def test_a_planner_never_installed_lists_nothing_as_mismatched():
    from tandem.planners.tiptop.factory import INFO

    state = planners_cli.runtime_state("tiptop", INFO, settings_mod.load())
    assert state["status"] == planners_cli.NOT_INSTALLED and state["mismatched"] == []

    shown = CliRunner().invoke(app, ["runtime", "status", "--planner", "tiptop", "--json"])
    assert shown.exit_code == 0, shown.output
    assert json.loads(shown.output)["mismatched"] == []
