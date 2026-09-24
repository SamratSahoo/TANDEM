"""What `tandem doctor` checks for TiPToP, and how its settings for a profile are shown to a person.

Both used to be written into tandem's own commands -- doctor probed a ZED SDK, a bamboo shim's ports
and an M2T2 server for every profile, and `tandem profile show` printed a robot's address and the cuRobo
overrides -- which is to say every command knew it was talking to TiPToP. They are TiPToP's answers
now, given through its factory's ``doctor_checks`` and ``describe_options``, and a profile that plans
with another planner is asked none of these questions.

Imported only when one of those hooks is called, so listing planners does not pay for it.
"""

from __future__ import annotations

from typing import Any

from tandem.core import probe, secrets
from tandem.core import rig as rig_mod
from tandem.core.errors import RigInvalid, TandemError, one_line
from tandem.planners.base import OptionsSection, OptionsView
from tandem.planners.tiptop import probe as tiptop_probe
from tandem.planners.tiptop import render
from tandem.planners.tiptop.options import TiptopOptions, check_robot_type, resolve

# check_assets findings that have rows of their own (camera calibration, tiptop cameras), and so are not
# repeated among the TAMP ones.
_OWN_ROWS = (render.MISSING_EXTRINSICS, render.MISSING_CAMERA)


def doctor_checks(
    profile: Any, *, settings: Any, runtime_ready: bool, probe_hardware: bool
) -> list[probe.Check]:
    """TiPToP's rows: the GPU its runtime compiles for, the key its perception calls, and -- with a
    profile -- what it needs of the rig (both cameras, their extrinsics, an arm it drives), the profile's
    TAMP settings, and the hardware and servers the rig names."""
    checks = [probe.check_nvidia_driver(), probe.check_cuda_runtime(), probe.check_nvcc()]
    if probe_hardware:
        checks.append(tiptop_probe.check_zed_sdk())
    if profile is None:
        # The machine only (`tandem init`'s preflight). The key is a question about collecting, which
        # init asks later, in its own step -- a missing one here would stop init before that step.
        return checks
    checks.append(_gemini_for_perception(runtime_ready))

    try:
        rig = rig_mod.load()
        rig_options = rig_mod.planner_options(rig, "tiptop")
    except RigInvalid as exc:
        # tandem's own "rig" row says what is wrong with the file; this says TiPToP cannot be checked.
        checks.append(
            probe.Check("tiptop rig settings", probe.FAIL, one_line(exc.message), exc.hint or "", group="rig")
        )
        return checks
    checks.append(_robot_type(rig))
    try:
        options = resolve(rig, rig_options, profile.planner.options, check_type=False)
    except (TandemError, ValueError) as exc:
        # The profile loader and the rig validate these, so this is a profile built by hand; say so, and stop.
        message = exc.message if isinstance(exc, TandemError) else str(exc)
        checks.append(probe.Check("tiptop options", probe.FAIL, message.splitlines()[0], group="profile"))
        return checks

    checks.append(_cameras(rig, runtime_ready))
    checks.append(_calibration(rig))
    runtime_dir = settings.resolved_runtime_dir() if settings is not None else None
    warnings = [
        w
        for w in render.check_assets(profile, rig, options, runtime_dir=runtime_dir)
        if not w.startswith(_OWN_ROWS)
    ]
    # A setting of the perception block that does nothing is the rig's, not a TAMP finding: a row of its own.
    for warning in (w for w in warnings if w.startswith("perception.")):
        checks.append(probe.Check("perception settings", probe.WARN, warning, group="rig"))
    warnings = [w for w in warnings if not w.startswith("perception.")]
    for warning in warnings:
        checks.append(probe.Check("tamp settings", probe.WARN, warning, group="profile"))
    if not warnings:
        n = len(options.tamp)
        checks.append(
            probe.Check(
                "tamp settings", probe.OK, f"{n} override(s)" if n else "stock settings", group="profile"
            )
        )

    if probe_hardware:
        robot = options.robot
        checks.append(tiptop_probe.check_robot(robot.host, robot.port))
        checks.append(tiptop_probe.check_robot_state_port(robot.host, robot.state_port))
        checks.append(tiptop_probe.check_m2t2(options.perception.m2t2.url))
        checks.append(tiptop_probe.check_foundation_stereo(options.perception.foundation_stereo.url))
    return checks


def _gemini_for_perception(runtime_ready: bool) -> probe.Check:
    """TiPToP's own need for the key, beside phase planning's: its detector is a Gemini model."""
    name = "gemini for perception"
    if secrets.gemini_api_key():
        return probe.Check(name, probe.OK, f"set (from {secrets.gemini_key_source()})", group="credentials")
    hint = (
        "Run `tandem config set-gemini-key`. TiPToP's perception calls Gemini once per rollout to turn the "
        "task into objects and goal predicates, so a session with TiPToP will not start without it."
    )
    if not runtime_ready:
        # A machine that cannot collect with TiPToP anyway: worth a note, not a failure it cannot act on.
        return probe.Check(name, probe.WARN, "not set (only needed to collect)", hint, group="credentials")
    return probe.Check(name, probe.FAIL, "not set", hint, group="credentials")


def _robot_type(rig: Any) -> probe.Check:
    """Whether TiPToP drives the arm the rig names. tandem checks only that it is a name."""
    try:
        check_robot_type(rig.robot.type)
    except ValueError as exc:
        return probe.Check(
            "robot type",
            probe.FAIL,
            str(exc),
            "`tandem rig set robot.type fr3_robotiq` (or another arm TiPToP drives); `tandem planners info "
            "tiptop` lists them.",
            group="rig",
        )
    return probe.Check("robot type", probe.OK, rig.robot.type, group="rig")


def _cameras(rig: Any, runtime_ready: bool) -> probe.Check:
    """The two cameras the pinned tiptop opens at every warm-up, whichever one perception reads.

    Graded like the Gemini key: a FAIL where TiPToP could collect, a WARN on a machine that cannot
    anyway -- a laptop keeping a profile's trajectories often has no cameras in it at all.
    """
    name = "tiptop cameras"
    missing = [slot for slot in ("hand", "external") if slot not in rig.cameras.configured()]
    if not missing:
        return probe.Check(name, probe.OK, "hand and external configured", group="rig")
    detail = "no " + " or ".join(f"cameras.{slot}" for slot in missing)
    hint = (
        "The pinned tiptop opens cameras.hand and cameras.external at every warm-up, whichever one "
        f"perception reads, so a session fails to start without both. `tandem rig set cameras.{missing[0]}.serial "
        "SERIAL` adds one."
    )
    if not runtime_ready:
        return probe.Check(name, probe.WARN, detail + " (only needed to collect)", hint, group="rig")
    return probe.Check(name, probe.FAIL, detail, hint, group="rig")


def _calibration(rig: Any) -> probe.Check:
    """Every configured camera's extrinsics: tiptop raises at warm-up, arm moving, for one it lacks."""
    configured = rig.cameras.configured()
    if not configured:
        return probe.Check("camera calibration", probe.SKIP, "no cameras configured", group="rig")
    try:
        missing = rig.missing_calibration()
    except RigInvalid as exc:
        return probe.Check("camera calibration", probe.FAIL, one_line(exc.message), exc.hint or "", group="rig")
    if missing:
        return probe.Check(
            "camera calibration",
            probe.FAIL,
            f"no extrinsics for {', '.join(missing)}",
            f"Extrinsics are keyed by serial. Add them to {rig.calibration_file()} (`tandem runtime run "
            "calibrate-wrist-cam` writes the wrist camera's).",
            group="rig",
        )
    return probe.Check("camera calibration", probe.OK, f"{len(configured)} camera(s) calibrated", group="rig")


# --------------------------------------------------------------------------- showing the options


def describe(profile: Any, *, settings: Any) -> OptionsView:
    """TiPToP's settings for a profile: this machine's robot and perception, the profile's TAMP overrides,
    and exactly what the planner receives."""
    rig = rig_mod.load()
    options: TiptopOptions = resolve(
        rig, rig_mod.planner_options(rig, "tiptop"), profile.planner.options, check_type=False
    )
    runtime_dir = settings.resolved_runtime_dir() if settings is not None else None
    overrides = render.render_tamp_overrides(profile, options, runtime_dir=runtime_dir)
    robot, perception = options.robot, options.perception
    return OptionsView(
        summary=f"{robot.type} at {robot.host}  ·  {robot.time_dilation_factor:.0%} speed",
        sections=(
            OptionsSection(
                "robot",
                (
                    ("type", robot.type),
                    ("address", f"{robot.host}:{robot.port}"),
                    ("gripper / state", f"{robot.gripper_port} / {robot.state_port}"),
                    ("speed", f"{robot.time_dilation_factor:.0%} (time_dilation_factor)"),
                ),
                "this machine's (rig.yml)",
            ),
            OptionsSection(
                "perception",
                (
                    ("detector", perception.gemini.model),
                    ("grasps", perception.m2t2.url),
                    ("depth", perception.foundation_stereo.url),
                    (
                        "segmentation",
                        f"SAM-2, {perception.sam_mode}"
                        + (f" at {perception.sam_url}" if perception.sam_mode == "remote" else ""),
                    ),
                ),
                f"this machine's (rig.yml) · from the {rig.cameras.perception} camera",
            ),
            OptionsSection(
                "tamp",
                tuple((key, _shown(overrides[key])) for key in sorted(overrides)),
                f"{len(overrides)} override(s)" if overrides else "stock settings",
            ),
        ),
        receives=overrides,
        receives_note="the cuRobo cost overrides, passed as --curobo-overrides (paths made absolute)",
        warnings=tuple(render.check_assets(profile, rig, options, runtime_dir=runtime_dir)),
    )


def _shown(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        import json

        return json.dumps(value)
    return str(value)
