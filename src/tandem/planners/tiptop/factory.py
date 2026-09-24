"""How TiPToP is built for a session, installed on a machine, and described in a catalog.

Everything TiPToP-specific about STARTING a session lives here, and nowhere in tandem's core: which
runtime it runs in, the ``TIPTOP_*`` environment its sidecar reads, the ``tiptop.yml`` and cuRobo
cost overrides rendered from the rig and the profile, and the checks that catch a broken asset before
a twenty-second warm-up does. The session only hands over a ``BackendContext`` and gets a backend back.

So is everything else tandem asks of TiPToP by name: its settings checked and shown -- the task's
``planner.options`` (``OPTIONS``: tamp) and this machine's ``planners.tiptop`` in rig.yml
(``RIG_OPTIONS``: robot, perception; ``options.py``, ``doctor.py``) -- its rows in `tandem doctor`, the
rig's tiptop.yml for a command run in its runtime (`tandem runtime run calibrate-wrist-cam`), and a leg
replayed in tiptop's own viewer.

This module is imported to list planners and to read TiPToP's capabilities, both of which happen on a
laptop. So it imports the declarations -- capabilities, runtime recipe -- and the protocol types and
nothing else at module level; the renderer and the backend itself are imported inside the calls that
need them.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tandem.core.errors import TandemError
from tandem.planners.base import (
    BackendContext,
    Capabilities,
    PlannerInfo,
    SourcePin,
)
from tandem.planners.tiptop import arms
from tandem.planners.tiptop.capabilities import CAPABILITIES
from tandem.planners.tiptop.recipe import RECIPE
from tandem.planners.tiptop.runtime import TiptopRuntime

#: The serialized plan tiptop writes into a leg it planned, and what its viewer replays.
PLAN_FILE = "tiptop_plan.json"

#: Everything the pinned tiptop's viewer (scripts/viz_tiptop_run.py) opens to replay a plan, relative to
#: the leg. The sidecar leaves each one: perception writes the images, the depth and the point cloud
#: (save_perception_outputs), planning the plan and metadata.json (save_run_metadata), execution the
#: rest (save_run_outputs). A merged trajectory has its first planned leg's at the top (core/merge.py).
#: tests/test_review_tiptop.py checks the list against the viewer's source and the recorder's.
REPLAY_FILES = (
    "metadata.json",
    "tiptop.yml",
    "rgb.png",
    "bboxes_viz.png",
    "masks_viz.png",
    "perception/intrinsics.json",
    "perception/depth.png",
    "perception/pointcloud.ply",
    "perception/grasps.pt",
    # Read inside a try by the viewer, which then draws everything but the plan -- so for a replay of
    # the plan, required.
    "perception/cutamp_env.pkl",
    PLAN_FILE,
)

# What an install builds the runtime from: the commits the recipe fetches. Read from the recipe rather
# than restated, because a catalog that names a commit the install does not deliver is a false
# statement about every dataset collected with it.
SOURCES: tuple[SourcePin, ...] = RECIPE.pins

INFO = PlannerInfo(
    name="tiptop",
    display_name="TiPToP",
    summary=(
        "GPU task and motion planning with cuTAMP and cuRobo, perceiving with Gemini, SAM-2 and M2T2 "
        "grasps, and executing pick-and-place on a real arm."
    ),
    homepage="https://github.com/SamratSahoo/tiptop",
    requires=(
        "Linux with an NVIDIA GPU, CUDA 12 or newer and a recent driver",
        "pixi, and about 25 GB of free disk for the runtime",
        # Written from the table the options schema checks robot.type against, so it names every arm
        # the schema accepts and no other.
        arms.requirement(),
        "2-3 ZED cameras and the ZED SDK",
        "an M2T2 grasp server (rig: planners.tiptop.perception.m2t2.url)",
        # tiptop estimates the ZEDs' depth with it, every rollout.
        "a FoundationStereo depth server (rig: planners.tiptop.perception.foundation_stereo.url, default "
        "http://localhost:1234)",
        "a Gemini API key",
    ),
    sources=SOURCES,
)


class TiptopFactory:
    """The registry's handle on TiPToP. Stateless: one instance serves every session."""

    info = INFO
    #: What a profile's planner.options may set for TiPToP: the task's. Their schema is ``options.py``.
    OPTIONS = {
        "tamp": "cuTAMP / cuRobo overrides, by tiptop's own key names; unknown keys are refused",
    }
    #: What rig.yml's planners.tiptop may set: this machine's, which every profile shares.
    RIG_OPTIONS = {
        "robot": "the arm's shim ports, speed, joint count, home and capture poses (its address and type are "
        "the rig's robot.host and robot.type)",
        "perception": "the M2T2 grasp server, the FoundationStereo depth server, SAM-2 and the depth pipeline",
    }

    def capabilities(self) -> Capabilities:
        return CAPABILITIES

    def runtime(self, settings: Any = None) -> TiptopRuntime:
        return TiptopRuntime(_settings(settings).resolved_runtime_dir())

    def validate_options(self, options: Mapping[str, Any] | None) -> dict[str, Any]:
        """A profile's ``planner.options`` checked against TiPToP's task schema, with every default filled in.

        Raises the pydantic ValidationError itself, whose locations (``tamp``) the profile loader reads as
        paths under ``planner.options``.
        """
        from tandem.planners.tiptop.options import TiptopTaskOptions

        return TiptopTaskOptions.model_validate(dict(options or {})).to_options()

    def validate_rig_options(self, options: Mapping[str, Any] | None) -> dict[str, Any]:
        """rig.yml's ``planners.tiptop`` checked against TiPToP's machine schema, with every default filled in.

        Raises the pydantic ValidationError itself, located (``robot.q_home``) under ``planners.tiptop``.
        """
        from tandem.planners.tiptop.options import TiptopRigOptions

        return TiptopRigOptions.model_validate(dict(options or {})).to_options()

    def describe_options(self, profile: Any, *, settings: Any = None):
        from tandem.planners.tiptop import doctor

        return doctor.describe(profile, settings=_settings(settings))

    def doctor_checks(self, profile: Any, *, settings: Any = None, probe_hardware: bool = True) -> list:
        from tandem.planners.tiptop import doctor

        settings = _settings(settings)
        ready = self.runtime(settings).status().installed
        return doctor.doctor_checks(profile, settings=settings, runtime_ready=ready, probe_hardware=probe_hardware)

    def runtime_env(self, *, rig: Any, settings: Any = None) -> dict[str, str]:
        """This machine's rig, where tiptop's own scripts read it (`tandem runtime run`, `runtime shell`).

        tiptop's scripts -- calibrate-wrist-cam, viz-calibration, anything reading ``tiptop_cfg()`` -- take
        the robot's address, the arm and the camera serials from $TIPTOP_CONFIG, and read and write the
        extrinsics at $TIPTOP_CALIBRATION (patch 0001). Pointed at a tiptop.yml rendered from the rig and at
        the rig's calibration.json, `tandem runtime run calibrate-wrist-cam` reaches the NUC at rig.yml's
        robot.host and writes the wrist camera's extrinsics where a session reads them. Unset, they would
        use tiptop's stock tiptop.yml (172.16.0.2, its authors' serials) and the runtime's own calibration
        file, which is what ``--raw`` gives.
        """
        from tandem.core import paths
        from tandem.core import rig as rig_mod
        from tandem.planners.tiptop import render

        options = _resolve(rig, rig_mod.planner_options(rig, "tiptop"), {})
        config = render.write_tiptop_config(rig, paths.state_dir() / "rig" / "tiptop.yml", options)
        env = {
            "TIPTOP_CONFIG": str(config),
            # Created when missing: tiptop's calibration scripts write into it.
            "TIPTOP_CALIBRATION": str(rig_mod.ensure_calibration_file(rig)),
            "TIPTOP_STATE_PORT": str(options.robot.state_port),
        }
        cameras = rig.cameras.configured()
        for role, var in (
            ("hand", "TIPTOP_HAND_CAMERA_ID"),
            ("external", "TIPTOP_EXTERNAL_CAMERA_ID"),
            ("external_2", "TIPTOP_EXTERNAL_2_CAMERA_ID"),
        ):
            if role in cameras:
                env[var] = cameras[role].serial
        return env

    def replay(self, rollout_dir: Path, *, settings: Any = None) -> None:
        """Replay a leg's saved plan in Rerun, with tiptop's viz-tiptop-run: it needs cuRobo and cuTAMP to
        load the robot model and the saved TAMP environment, so it runs inside the runtime."""
        import subprocess

        rollout_dir = Path(rollout_dir)
        if not (rollout_dir / PLAN_FILE).is_file():
            raise TandemError(
                f"{rollout_dir.name} has no {PLAN_FILE}, so there is no plan to replay.",
                hint="Only rollouts whose planning succeeded record one.",
            )
        # Checked here, not left to the viewer: it would start Rerun, then die of the first one missing
        # with a traceback that says nothing about why the leg lacks it.
        missing = [name for name in REPLAY_FILES if not (rollout_dir / name).is_file()]
        if missing:
            why = (
                "A leg recorded before tandem wrote tiptop's metadata.json into each one cannot be replayed."
                if "metadata.json" in missing
                else "Its recording did not finish: the sidecar's log for that session says why."
            )
            raise TandemError(
                f"{rollout_dir.name} has no {', '.join(missing)}, which tiptop's viewer needs to replay it.",
                hint=f"{why} `tandem ui` shows any trajectory's cameras and robot state.",
            )
        runtime = self.runtime(settings)
        runtime.require_ready()
        code = subprocess.call(runtime.replay_command(rollout_dir), cwd=str(runtime.tiptop_dir))
        if code != 0:
            raise TandemError(
                f"tiptop's viewer exited with status {code} replaying {rollout_dir.name}.",
                hint="Its own output, above, says why.",
            )

    def create(self, ctx: BackendContext):
        """A TiptopBackend for this session, with its config rendered and its assets checked.

        What ``Session._build_backend`` and part of ``Session.start`` used to do inline, moved here
        unchanged so the session no longer knows any of it is TiPToP's.
        """
        from tandem.core import rig as rig_mod
        from tandem.core import secrets
        from tandem.planners.tiptop import render
        from tandem.planners.tiptop.backend import TiptopBackend

        # The session hands over the rig and its planners.tiptop block, checked; a context built by hand
        # without a rig gets this machine's.
        rig = ctx.rig if ctx.rig is not None else rig_mod.load()
        rig_options = ctx.rig_options if ctx.rig is not None else rig_mod.planner_options(rig, "tiptop")
        options = _resolve(rig, rig_options, ctx.options)

        # TiPToP's perception is a Gemini call every rollout: the detector turns the instruction into
        # objects and goal atoms. Asked for here, by the planner that needs it, rather than by the
        # session -- a planner with its own detector needs no key at all.
        if not secrets.gemini_api_key():
            raise TandemError(
                "No Gemini API key is set, and TiPToP's perception calls Gemini every rollout.",
                hint="Run `tandem config set-gemini-key`.",
            )

        # The runtime the caller already resolved wins, so a session and its planner can never be
        # looking at two different runtimes.
        root = ctx.runtime_dir if ctx.runtime_dir is not None else _settings(ctx.settings).resolved_runtime_dir()
        runtime = TiptopRuntime(Path(root))
        profile = ctx.profile

        # Problems that would otherwise surface minutes into a warmed session: a checkpoint the VAE
        # cost loads lazily, a blending key that does nothing. Two are fatal, because tiptop raises
        # for them at warm-up, the first with the arm already moving to its capture pose: missing
        # extrinsics, and a camera tiptop opens that the profile does not configure.
        problems = render.check_assets(profile, rig, options, runtime_dir=runtime.root)
        cameras = [p for p in problems if p.startswith(render.MISSING_CAMERA)]
        if cameras:
            raise TandemError(
                "\n".join(cameras),
                hint="Add the missing camera to this machine's rig (`tandem rig set cameras.ROLE.serial SERIAL`). "
                "TiPToP needs both a hand and an external camera, even though perception reads only one of them.",
            )
        fatal = [p for p in problems if p.startswith(render.MISSING_EXTRINSICS)]
        if fatal:
            raise TandemError(
                "\n".join(fatal),
                hint="Extrinsics are keyed by camera serial; add them before collecting "
                "(`tandem rig path --calibration` is the file).",
            )
        for problem in problems:
            ctx.log(f"warning: {problem}")

        # tiptop.yml (per profile, where $TIPTOP_CONFIG points) and the cuRobo cost overrides (in the
        # session's own directory).
        files = render.prepare_session_files(profile, rig, ctx.session_dir, options, runtime_dir=runtime.root)
        env = render.render_env(
            profile,
            rig,
            options,
            events_file=ctx.events_file if ctx.events_file is not None else files["events_file"],
            task=ctx.task or None,
            runtime_dir=runtime.root,
        )
        return TiptopBackend(
            runtime,
            env=env,
            output_dir=ctx.output_dir,
            execute=ctx.execute,
            record=ctx.record,
            cost_overrides_file=files.get("overrides_file"),
            on_log=ctx.on_log,
        )


def _resolve(rig: Any, rig_options: Mapping[str, Any] | None, options: Mapping[str, Any] | None):
    """TiPToP's whole configuration, or a TandemError saying which setting is wrong -- for a context built by
    hand, whose options nothing has checked yet."""
    from pydantic import ValidationError

    from tandem.planners.tiptop.options import TiptopRigOptions, TiptopTaskOptions, resolve

    for model, block, where in (
        (TiptopRigOptions, rig_options, "planners.tiptop"),
        (TiptopTaskOptions, options, "planner.options"),
    ):
        try:
            model.model_validate(dict(block or {}))
        except ValidationError as exc:
            lines = [
                f"  {'.'.join((where, *map(str, err['loc'])))}: {str(err['msg']).removeprefix('Value error, ')}"
                for err in exc.errors()
            ]
            owner = "this machine's rig.yml" if where == "planners.tiptop" else "the profile"
            raise TandemError(
                f"TiPToP's settings in {owner} are not valid:\n" + "\n".join(lines),
                hint="`tandem planners info tiptop` lists what TiPToP reads, and where.",
            ) from None
    return resolve(rig, rig_options, options)


def _settings(settings: Any):
    if settings is not None:
        return settings
    from tandem.core import settings as settings_mod

    return settings_mod.load()


FACTORY = TiptopFactory()
