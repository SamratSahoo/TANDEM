"""What `tandem profile show` and a web profile card say about a TiPToP profile, read by someone new.

- The speed: every paper profile, and every new one, sets tamp.time_dilation_factor_literal: 1.0, so the
  planned motions run at the encoder's pace. "20% speed" read as an arm that moves slowly.
- The warnings: on a machine with no runtime yet, "encoder_path does not exist: <data>/profiles/vae/..."
  for every profile sent people to put a checkpoint beside their profiles that the runtime installs; and
  on a laptop with no rig.yml, every profile warned about missing cameras.
"""

from __future__ import annotations

import pytest

from tandem.core import profiles, settings
from tandem.planners import registry
from tandem.planners.tiptop import render
from tandem.planners.tiptop.doctor import planned_speed


def test_a_paper_profile_says_its_planned_motions_run_at_the_encoders_pace(machine_rig):
    profile = profiles.create("cups", prompt="stack the cups")
    view = registry.describe_options("tiptop", profile, settings=settings.load())
    assert view.summary.endswith("planned motions at the encoder's pace; homing and capture at 20%")
    (speed,) = [value for key, value in view.sections[0].rows if key == "speed"]
    assert "the encoder's pace (time_dilation_factor_literal 1)" in speed and "homing and capture: 20%" in speed


@pytest.mark.parametrize(
    ("tamp", "expected"),
    [
        ({}, None),
        ({"time_dilation_factor": 1.0}, None),  # tiptop's "no extra scaling": the robot's own speed
        ({"time_dilation_factor": 0.5}, ("50%", "tamp.time_dilation_factor")),
        ({"time_dilation_factor_literal": 0.4}, ("40%", "time_dilation_factor_literal")),
        ({"time_dilation_factor_literal": 1.0}, ("full speed", "time_dilation_factor_literal 1")),
        (
            {"time_dilation_factor_literal": 1.0, "retime_trajectory": True},
            ("the encoder's pace", "time_dilation_factor_literal 1"),
        ),
    ],
)
def test_the_planned_speed_is_tiptops_own_reading_of_the_settings(tamp, expected):
    assert planned_speed(tamp) == expected


def test_a_checkpoint_the_runtime_installs_is_not_warned_about_before_there_is_a_runtime(machine_rig, tmp_path):
    profile = profiles.create("cups", prompt="stack the cups")
    from tandem.planners.tiptop.options import resolve_profile

    resolved = resolve_profile(profile, machine_rig)
    absent = tmp_path / "no-runtime-here"
    assert not any("encoder_path" in p for p in render.check_assets(profile, machine_rig, resolved, runtime_dir=absent))
    assert not any("encoder_path" in p for p in render.check_assets(profile, machine_rig, resolved))

    # A runtime that is there but lacks it: that is a real problem, and said.
    built = tmp_path / "runtime"
    built.mkdir()
    (problem,) = [p for p in render.check_assets(profile, machine_rig, resolved, runtime_dir=built) if "encoder_path" in p]
    assert "encoder_path does not exist" in problem

    # An absolute path is the person's own: always checked.
    elsewhere = profile.model_copy(deep=True)
    elsewhere.planner.options["tamp"]["encoder_path"] = str(tmp_path / "mine.pt")
    resolved = resolve_profile(elsewhere, machine_rig)
    assert any("mine.pt" in p for p in render.check_assets(elsewhere, machine_rig, resolved, runtime_dir=absent))


def test_a_machine_with_no_rig_is_not_told_about_cameras_for_every_profile(isolated_env):
    profile = profiles.create("cups", prompt="stack the cups")
    view = registry.describe_options("tiptop", profile, settings=settings.load())
    assert not any(w.startswith(render.MISSING_CAMERA) for w in view.warnings)
    assert view.warnings == ()
