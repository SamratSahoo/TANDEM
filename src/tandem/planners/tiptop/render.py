"""The rig, a profile and TiPToP's options → the three things a tiptop subprocess actually consumes.

    render_tiptop_config(rig, options)            -> the YAML tiptop_cfg() loads   ($TIPTOP_CONFIG)
    render_tamp_overrides(profile, options)       -> the JSON --curobo-overrides reads
    render_env(profile, rig, options, ...)        -> the environment the child runs in

Keeping this in one small, unit-testable module is deliberate: it is the seam where tandem's settings
stop being tandem's concern and become tiptop's, and it is where "did my override actually apply?" is
decided.

The rig is this machine's (``tandem.core.rig``): the arm's type and address, the cameras and their
calibration. The profile is the task: its prompt, and where a relative path in it is read from.
``options`` is TiPToP's whole configuration, both halves resolved (``options.resolve``): the rig's
``planners.tiptop`` and the profile's tamp overrides.

The extrinsics file tiptop reads ($TIPTOP_CALIBRATION) is the rig's ``calibration.json``, beside
rig.yml. Patch 0001's prose still says a profile owns its extrinsics; it is left as it is, because the
runtime records every patch's digest and an edit would rebuild every installed tree for a comment.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from tandem.core import paths, profiles, secrets
from tandem.core.errors import TandemError
from tandem.core.profiles import Profile
from tandem.core.rig import Rig
from tandem.planners.tiptop import tamp_keys
from tandem.planners.tiptop.options import DETECTOR_MODEL, TiptopOptions


def _new_yaml() -> YAML:
    """A fresh YAML instance, one per dump -- the rule ``profiles._new_yaml`` states.

    ruamel keeps the half-written state of a dump that raised, and the next dump through the same
    instance then writes nothing at all: a shared instance let one bad render leave every later
    tiptop config empty, for the life of the process (a `tandem ui` serving many sessions).
    """
    yaml = YAML()
    yaml.default_flow_style = False
    return yaml


# How check_assets starts the two problems that stop a session before it starts, rather than warn:
# a camera with no extrinsics, and a camera the pinned tiptop opens that the rig does not have.
# The factory refuses on either; doctor shows each as a FAIL row of its own.
MISSING_EXTRINSICS = "no camera extrinsics"
MISSING_CAMERA = "no cameras."


# --------------------------------------------------------------------------- tiptop.yml


def _options(options: Any) -> TiptopOptions:
    return options if isinstance(options, TiptopOptions) else TiptopOptions.model_validate(dict(options or {}))


def render_tiptop_config(rig: Rig, options: Any) -> dict:
    """The dict tiptop's ``tiptop_cfg()`` expects (robot / cameras / perception).

    The perception knobs a ``tamp:`` block may set (``tamp_keys.PERCEPTION_KEYS``: M2T2's grasp
    threshold and pass count, the voxel size, the contact threshold) are written where tiptop reads
    them -- ``perception.m2t2.num_runs`` and ``perception.m2t2.grasp_threshold`` for the M2T2 pair
    (perception_wrapper.predict_depth_and_grasps) -- overriding the rig's ``perception:`` value where
    both are set, as tiptop's own override does. A task's number wins over the machine's. Unset, they
    are left out, so tiptop's own defaults (5 passes, 0.035) apply rather than a copy of them that could
    go stale.
    """
    o = _options(options)
    cameras: dict[str, Any] = {"perception": rig.cameras.perception}
    for key, cam in rig.cameras.configured().items():
        cameras[key] = {
            "serial": cam.serial,
            "type": cam.type,
            "resolution": cam.resolution,
            "fps": cam.fps,
        }

    rendered = {
        "robot": {
            "type": o.robot.type,
            "dof": o.robot.dof,
            "host": o.robot.host,
            "port": o.robot.port,
            "gripper_port": o.robot.gripper_port,
            "time_dilation_factor": o.robot.time_dilation_factor,
            "q_home": list(o.robot.q_home),
            "q_capture": list(o.robot.q_capture),
        },
        "cameras": cameras,
        "perception": {
            # Used: tiptop estimates a ZED's depth by sending its stereo pair to FoundationStereo
            # (perception/cameras get_depth_estimator -> zed_infer_depth_async), and the sidecar passes
            # that estimator to run_perception. Not a setting, so the server has to be here.
            "foundation_stereo": {"url": "http://localhost:1234"},
            "m2t2": {
                "url": o.perception.m2t2.url,
                "apply_bounds": o.perception.m2t2.apply_bounds,
            },
            # tiptop reads sam.url whenever the mode is not "local"; the options refuse a remote
            # mode without one.
            "sam": {
                "mode": o.perception.sam_mode,
                **({"url": o.perception.sam_url} if o.perception.sam_url else {}),
            },
            "robot_mask_margin_m": o.perception.robot_mask_margin_m,
            "depth_trunc_m": o.perception.depth_trunc_m,
            "voxel_downsample_size": o.perception.voxel_downsample_size,
            "contact_threshold_m": o.perception.contact_threshold_m,
            "mask_erosion_pixels": o.perception.mask_erosion_pixels,
            "depth_smoothing": {"num_frames": o.perception.depth_smoothing_frames},
        },
    }
    for key, path in tamp_keys.PERCEPTION_KEYS.items():
        if key in o.tamp:
            node = rendered
            for part in path[:-1]:
                node = node.setdefault(part, {})
            node[path[-1]] = o.tamp[key]
    return rendered


def write_tiptop_config(rig: Rig, dest: Path, options: Any) -> Path:
    # Rendered in full before the file is opened, so a render or a dump that raises leaves the last
    # good config in place rather than a truncated one.
    buf = io.StringIO()
    _new_yaml().dump(render_tiptop_config(rig, options), buf)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        "# Generated by tandem from this machine's rig and the profile — edits here are overwritten.\n"
        + buf.getvalue()
    )
    return dest


# --------------------------------------------------------------------------- overrides


def render_tamp_overrides(
    profile: Profile | None, options: Any, *, runtime_dir: Path | None = None
) -> dict:
    """The flat dict of solver-cost knobs the planner backend is built with.

    Path-valued knobs are resolved to absolute paths here rather than left for tiptop to
    interpret against a repo root, so "relative" means "relative to the profile's file" — the only
    reading a profile author can reasonably expect. With no profile, relative to the runtime.
    """
    out: dict[str, Any] = dict(_options(options).tamp)
    for key in tamp_keys.PATH_KEYS:
        if key in out:
            out[key] = str(_resolve_asset(profile, str(out[key]), key, runtime_dir))
    return out


def _resolve_asset(profile: Profile | None, value: str, key: str, runtime_dir: Path | None) -> Path:
    """Resolve a checkpoint path: absolute wins, then beside the profile's file, then the runtime dir.

    The runtime dir is included because TiPToP's runtime recipe puts the DATAFARM checkpoints there
    under the same relative layout the source monorepo used (vae/checkpoints/...,
    rnd/checkpoints/...), so the paper's settings name them as its configs did.
    """
    candidate = Path(os.path.expanduser(value))
    if candidate.is_absolute():
        return candidate
    beside = profile.file().parent if profile is not None else None
    for base in filter(None, (beside, runtime_dir)):
        resolved = base / candidate
        if resolved.exists():
            return resolved.resolve()
    # Nothing on disk yet. Return the profile-relative interpretation so the error message
    # names the place the user most likely meant.
    if profile is not None:
        return profiles.resolve_path(profile, value)
    return (runtime_dir / candidate) if runtime_dir is not None else candidate.resolve()


def write_tamp_overrides(
    profile: Profile | None, dest: Path, options: Any, *, runtime_dir: Path | None = None
) -> Path | None:
    """Write the overrides JSON, or return None when the profile sets nothing.

    Returning None matters: passing an empty --curobo-overrides is not the same as passing
    none, and we want stock behaviour to be exactly stock.
    """
    overrides = render_tamp_overrides(profile, options, runtime_dir=runtime_dir)
    if not overrides:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(overrides, indent=2, sort_keys=True) + "\n")
    return dest


def check_assets(
    profile: Profile | None, rig: Rig, options: Any, *, runtime_dir: Path | None = None
) -> list[str]:
    """Problems that would only surface minutes into a warmed session. Cheap to check now.

    The VAE manifold cost torch.loads its checkpoint lazily, so a missing file does not fail
    until the first plan — after cuRobo, SAM2 and the cameras have all warmed up.
    """
    problems: list[str] = []
    o = _options(options)
    tamp = o.tamp

    if tamp.get("vae_manifold_weight") and tamp.get("vae_path"):
        path = _resolve_asset(profile, str(tamp["vae_path"]), "vae_path", runtime_dir)
        if not path.is_file():
            problems.append(f"vae_manifold_weight is set but vae_path does not exist: {path}")

    if str(tamp.get("blend_mode", "")).lower() == "flow":
        raw = tamp.get("blend_model_path")
        if raw:
            path = _resolve_asset(profile, str(raw), "blend_model_path", runtime_dir)
            if not path.is_file():
                problems.append(f"blend_mode is 'flow' but blend_model_path does not exist: {path}")

    if tamp.get("blend_ops") and not tamp.get("blend_trajectory"):
        problems.append("blend_ops is set but blend_trajectory is not true, so no blending happens")

    for key in ("blend_pace", "blend_boundary_mode", "blend_flow_steps", "blend_flow_retime_only"):
        if key in tamp and str(tamp.get("blend_mode", "spline")).lower() != "flow":
            problems.append(f"{key} only applies when blend_mode is 'flow'; it is ignored here")
    if tamp.get("blend_vae_sample_target") and str(tamp.get("blend_mode", "spline")).lower() != "vae":
        problems.append("blend_vae_sample_target only applies when blend_mode is 'vae'; it is ignored here")

    # The knobs below are each read only behind another one, by the same resolve_* function that
    # reads the gate -- so set without it, they are accepted, passed on, and change nothing.
    retiming = bool(tamp.get("vae_retiming")) and bool(tamp.get("vae_manifold_weight"))
    if tamp.get("vae_retiming") and not retiming:
        problems.append(
            "vae_retiming is set but vae_manifold_weight is 0 or unset, so nothing would optimize the "
            "trajectory clock and the planner ignores vae_retiming"
        )
    for key in ("retime_scale", "retime_smooth_weight", "retime_limit_weight"):
        if key in tamp and not retiming:
            problems.append(f"{key} only applies when vae_retiming is on; it is ignored here")
    if retiming and tamp.get("blend_trajectory"):
        # Not a mistake -- the planner means it -- but it turns a whole blend_* block off, and the
        # planner's own warning about it lands in the sidecar log, not in front of anyone.
        problems.append(
            "vae_retiming gives the VAE cost the trajectory clock, so trajectory blending "
            "(blend_trajectory and every blend_* key) is switched off for every plan"
        )

    seeds = int(tamp.get("posture_selection_seeds") or 0)
    for key in ("posture_grasp_roll", "posture_ref", "posture_pos_tol", "posture_rot_tol"):
        if key in tamp and seeds <= 1:
            problems.append(f"{key} only applies when posture_selection_seeds is above 1; it is ignored here")
    if seeds > 1 and tamp.get("posture_ref"):
        # cuTAMP loads the prior at its first plan, well after warm-up.
        path = _resolve_asset(profile, str(tamp["posture_ref"]), "posture_ref", runtime_dir)
        if not path.is_file():
            problems.append(f"posture_selection_seeds is set but posture_ref does not exist: {path}")

    if "transit_apex_min_dist" in tamp and not tamp.get("transit_apex_height"):
        problems.append(
            "transit_apex_min_dist only applies when transit_apex_height is above 0; it is ignored here"
        )

    # resolve_placement_support returns nothing at all unless placement_support is on, so a tuned
    # margin or flatness beside `placement_support: false` places exactly as the bounding box does.
    if not tamp.get("placement_support"):
        for key in tamp_keys.PLACEMENT_GATED:
            if key in tamp:
                problems.append(f"{key} only applies when placement_support is true; it is ignored here")
    # Both of its effects are inside trajectory blending (resolve_blend_config's BlendConfig).
    if "blend_stretch_to_caps" in tamp and not tamp.get("blend_trajectory"):
        problems.append(
            "blend_stretch_to_caps only applies when blend_trajectory is true; it is ignored here"
        )

    # Settings tiptop never reads (see options.GeminiSpec): kept so a profile can say which detector
    # labelled its data, and warned about when what they say is not what runs.
    gemini = o.perception.gemini
    if gemini.model != DETECTOR_MODEL:
        problems.append(
            f"perception.gemini.model is {gemini.model!r}, but the pinned tiptop always runs {DETECTOR_MODEL!r} "
            "(it takes no model from its config), so this changes nothing; set it to that"
        )
    if gemini.temperature is not None:
        problems.append(
            "perception.gemini.temperature is set, but the pinned tiptop calls its detector with its own "
            "default, so this changes nothing; set it to null"
        )

    # tandem's rig asks only for the camera perception reads, which is right for teleop and for another
    # planner. The pinned tiptop opens BOTH of these at every warm-up (get_demo_container:
    # get_hand_camera(), get_external_camera()), whichever one perception reads, and the tiptop.yml
    # rendered here names only the cameras the rig has -- so a missing one is an OmegaConf missing-key
    # error tens of seconds into the warm-up.
    configured = rig.cameras.configured()
    for slot in ("hand", "external"):
        if slot not in configured:
            problems.append(
                f"{MISSING_CAMERA}{slot}: the pinned tiptop opens cameras.hand and cameras.external at every "
                "warm-up, whichever one perception reads"
            )

    missing = rig.missing_calibration()
    if missing:
        problems.append(
            f"{MISSING_EXTRINSICS} for serial(s) " + ", ".join(missing) + f" in {rig.calibration_file()}"
        )
    return problems


# --------------------------------------------------------------------------- environment


def render_env(
    profile: Profile,
    rig: Rig,
    options: Any,
    *,
    events_file: Path,
    task: str | None = None,
    runtime_dir: Path | None = None,
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """The environment the planner backend's process runs in.

    Every name here is read by code in the planner's pinned tree; nothing is aspirational. tandem no longer
    spawns the planner's own CLI — it runs its own sidecar inside the same environment (see
    ``tandem.planners.tiptop``) — but that sidecar calls the same functions, which read the same
    variables.
    """
    env = dict(base if base is not None else os.environ)
    o = _options(options)

    # First task; later ones are typed at the child's stdin prompt.
    env["TIPTOP_TASK"] = task or profile.goal_or_prompt()
    # Only set when the goal genuinely differs from the language label, so tiptop otherwise
    # records whatever task actually ran (which is what a session where the operator retypes
    # the task needs).
    goal = profile.task.goal
    if goal and profile.task.prompt and goal != profile.task.prompt:
        env["TIPTOP_INSTRUCTION"] = profile.task.prompt
    else:
        env.pop("TIPTOP_INSTRUCTION", None)

    env["TIPTOP_EVENTS_FILE"] = str(events_file)
    env["TIPTOP_STATE_PORT"] = str(o.robot.state_port)
    env["TIPTOP_CONFIG"] = str(_session_config_path(profile))
    # The rig's: extrinsics belong to the mounted cameras, which every profile on this machine shares.
    env["TIPTOP_CALIBRATION"] = str(rig.calibration_file())
    # The source scoped per-robot data with DC_WORKSPACE and keyed extrinsics off it; a
    # tandem profile plays that role, and setting it keeps the planner's own
    # workspace-aware paths pointing somewhere sane.
    env["DC_WORKSPACE"] = profile.name
    env["TANDEM_PROFILE"] = profile.name

    # Camera serial overrides tiptop and droid both read.
    cams = rig.cameras.configured()
    for key, var in (
        ("hand", "TIPTOP_HAND_CAMERA_ID"),
        ("external", "TIPTOP_EXTERNAL_CAMERA_ID"),
        ("external_2", "TIPTOP_EXTERNAL_2_CAMERA_ID"),
    ):
        if key in cams:
            env[var] = cams[key].serial

    # google-genai's bare Client() reads either name; set both so it cannot miss. TiPToP's perception
    # needs it (its detector is Gemini), so a session with TiPToP refuses to start without one.
    key = secrets.gemini_api_key()
    if key:
        env["GEMINI_API_KEY"] = key
        env["GOOGLE_API_KEY"] = key

    # The cuRobo fork's VAE/RND costs default their checkpoints relative to what they assume
    # is a monorepo root. The runtime dir mirrors that layout, but set the env vars they also
    # honour so a relocated or profile-local checkpoint wins regardless.
    if runtime_dir is not None:
        vae = runtime_dir / "vae" / "checkpoints" / "vae_full_v2.pt"
        rnd = runtime_dir / "rnd" / "checkpoints" / "rnd_droid.pt"
        if vae.is_file():
            env.setdefault("VAE_MANIFOLD_CKPT", str(vae))
        if rnd.is_file():
            env.setdefault("RND_NOVELTY_CKPT", str(rnd))
    if o.tamp.get("vae_path"):
        env["VAE_MANIFOLD_CKPT"] = str(_resolve_asset(profile, str(o.tamp["vae_path"]), "vae_path", runtime_dir))

    # opencv's LAPACK and torch both drive one shared libmkl_core; the threaded path returns a
    # corrupt pivot array and cuRobo's get_stomp_cov() dies inside torch.inverse. Planning runs
    # on the GPU, so serialising MKL costs nothing that matters.
    env.setdefault("MKL_NUM_THREADS", "1")
    return env


def _session_config_path(profile: Profile) -> Path:
    return paths.session_scratch_dir() / profile.name / "tiptop.yml"


def prepare_session_files(
    profile: Profile, rig: Rig, session_dir: Path, options: Any, *, runtime_dir: Path | None = None
) -> dict:
    """Materialise everything a session needs before the child is spawned, in the session's own directory.

    ``session_dir`` is the session's (``BackendContext.session_dir``): the session made it and owns
    it, so where it is is not this module's to work out. Returns ``{events_file, config_file,
    overrides_file, session_dir}``.
    """
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    config_file = write_tiptop_config(rig, _session_config_path(profile), options)
    overrides_file = write_tamp_overrides(
        profile, session_dir / "curobo-overrides.json", options, runtime_dir=runtime_dir
    )

    events_file = session_dir / "events.jsonl"
    # Pre-create so the tailer can attach before the child writes its first line.
    events_file.touch()

    if not rig.calibration_file().is_file():
        raise TandemError(
            f"This machine's rig has no calibration file at {rig.calibration_file()}.",
            hint="`tandem init` creates it (so does any `tandem rig set`); or create it holding `{}`, and "
            "calibrate.",
        )

    return {
        "session_dir": session_dir,
        "events_file": events_file,
        "config_file": config_file,
        "overrides_file": overrides_file,
    }
