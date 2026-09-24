"""Import an existing hitl-tamp-vla setup into a tandem profile that plans with TiPToP.

The monorepo split one setup across two files — ``tiptop/tiptop/config/tiptop.yml`` (robot,
cameras, perception) and ``data-collection/cfg/tamp/<name>.yml`` (task, TAMP overrides,
episode target, HF slug) — plus a calibration JSON keyed by camera serial. This reassembles
them into a single profile so an existing rig is one command away from working.

TiPToP's, because everything it reads is TiPToP's configuration: the robot, the perception and the
TAMP overrides land in the profile's ``planner.options`` (``options.py``), the cameras and the task
in the profile itself. It is offered to tandem through the factory's ``importer`` (``IMPORTER``
below), so `tandem profile create --import-from` and `tandem init` reach it without naming TiPToP.

A task config's ``hitl:`` block is imported too, as the profile's own ``hitl:``: phase planning is
tandem's, whichever planner runs (``HITL_KEYS`` below says what each key becomes). What the block
chose that tandem cannot do -- a learned policy for the human phases, a robot planner other than
cuTAMP -- is refused rather than swapped for something it can, because the profile would then collect
something other than what the config it names collected.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from tandem.core.errors import TandemError
from tandem.core.profiles import Profile, format_errors, resolve_interpolation
from tandem.planners.base import WARNING_NOTE
from tandem.planners.tiptop import tamp_keys
from tandem.planners.tiptop.options import validate_tamp

# The same dereferencing a profile does when it is read, so an imported value and a stored
# one can never disagree about what ${oc.env:...} means.
_deref = resolve_interpolation

_yaml = YAML(typ="safe")

def _load_yaml(path: Path) -> dict:
    try:
        with path.open() as fh:
            return _yaml.load(fh) or {}
    except Exception as exc:
        raise TandemError(f"{path} is not valid YAML: {exc}") from exc


def find_sources(root: Path) -> dict[str, Path | None]:
    """Locate the pieces inside a monorepo checkout (or a bare tiptop checkout)."""
    root = root.expanduser().resolve()
    candidates = {
        "tiptop_config": [
            root / "tiptop" / "tiptop" / "config" / "tiptop.yml",
            root / "tiptop" / "config" / "tiptop.yml",
            root / "config" / "tiptop.yml",
        ],
        "calibration": [
            root / "tiptop" / "tiptop" / "config" / "assets" / "calibration_info.json",
            root / "tiptop" / "config" / "assets" / "calibration_info.json",
            root / "config" / "assets" / "calibration_info.json",
        ],
    }
    found: dict[str, Path | None] = {}
    for key, options in candidates.items():
        found[key] = next((p for p in options if p.is_file()), None)
    tamp_dir = root / "data-collection" / "cfg" / "tamp"
    found["tamp_dir"] = tamp_dir if tamp_dir.is_dir() else None
    return found


def find_calibration_layers(base: Path) -> list[Path]:
    """The per-workspace extrinsics files that sit beside the defaults.

    Upstream scopes each robot's cameras with ``calibration_info_<workspace>.json`` layered
    over ``calibration_info.json``. A tandem profile is that scoping, so an import wants all
    the layers — extrinsics are keyed by camera serial, so two robots' entries coexist.
    """
    if not base.is_file():
        return []
    return sorted(base.parent.glob("calibration_info_*.json"))


def list_tamp_configs(root: Path) -> list[Path]:
    tamp_dir = find_sources(root)["tamp_dir"]
    if tamp_dir is None:
        return []
    return sorted(tamp_dir.glob("*.yml"))


def build_profile(
    name: str,
    *,
    root: Path | None = None,
    tiptop_config: Path | None = None,
    tamp_config: Path | None = None,
) -> tuple[Profile, dict, list[str]]:
    """Assemble a Profile, its calibration dict, and any notes worth showing the user.

    Starts from the shipped template and layers the imported files on top, so a checkout
    that is missing a piece (an uninitialised ``tiptop`` submodule is the common case)
    produces a working profile with sensible defaults plus a note, rather than a validation
    error about a field the user never touched.
    """
    from tandem import resources
    from tandem.core.profiles import load_file

    notes: list[str] = []

    sources: dict[str, Path | None] = {"tiptop_config": tiptop_config, "calibration": None, "tamp_dir": None}
    if root is not None:
        discovered = find_sources(root)
        sources = {**discovered, **{k: v for k, v in sources.items() if v is not None}}

    template = load_file(resources.path("profile_template.yml"), name=name)
    data: dict[str, Any] = template.model_dump(mode="python", exclude_none=True)
    data["name"] = name
    data["description"] = ""
    # Whatever the template plans with, an import is TiPToP's settings: they are TiPToP's options.
    planner = data.setdefault("planner", {})
    if planner.get("backend") != "tiptop":
        from tandem.planners.tiptop.options import TiptopOptions

        planner.update(backend="tiptop", options=TiptopOptions().to_options())
    options = planner.setdefault("options", {})

    cfg_path = sources.get("tiptop_config")
    if cfg_path and cfg_path.is_file():
        _merge_tiptop_config(data, options, _load_yaml(cfg_path))
        notes.append(f"robot and cameras from {cfg_path}")
    elif root is not None:
        notes.append(
            f"no tiptop.yml under {root} — using template defaults for the robot and cameras "
            "(is the tiptop submodule checked out?)"
        )

    if tamp_config is not None:
        if not tamp_config.is_file():
            raise TandemError(f"No such TAMP config: {tamp_config}")
        raw = _load_yaml(tamp_config)
        about: list[str] = []
        _merge_tamp_config(data, options, raw, about)
        _merge_hitl(data, raw, tamp_config, about)
        data["description"] = f"imported from {tamp_config.name}"
        notes.append(f"task and TAMP settings from {tamp_config}")
        notes.extend(about)

    calibration: dict = {}
    calib_path = sources.get("calibration")
    if calib_path and calib_path.is_file():
        layers = [calib_path, *find_calibration_layers(calib_path)]
        for layer in layers:
            try:
                calibration.update(json.loads(layer.read_text()))
            except json.JSONDecodeError as exc:
                raise TandemError(f"{layer} is not valid JSON: {exc}") from exc
        extra = f" (+{len(layers) - 1} workspace layer(s))" if len(layers) > 1 else ""
        notes.append(f"{len(calibration)} camera extrinsics from {calib_path.name}{extra}")
    elif root is not None:
        notes.append(f"no calibration_info.json under {root} — extrinsics must be added by hand")

    try:
        profile = Profile.model_validate(data)
    except Exception as exc:
        raise TandemError(
            f"The imported settings do not form a valid profile:\n{format_errors(exc)}",
            hint="Import what works, then fix the rest with `tandem profile edit`.",
        ) from exc
    return profile, calibration, notes


def _merge_tiptop_config(data: dict, options: dict, raw: dict) -> None:
    # Merged into the template's defaults, not substituted for them: an upstream tiptop.yml
    # may omit a field (q_capture, gripper_port) that the profile schema still needs.
    robot = raw.get("robot") or {}
    if robot:
        options.setdefault("robot", {}).update(
            {
                k: _deref(v)
                for k, v in robot.items()
                if k
                in {"type", "dof", "host", "port", "gripper_port", "time_dilation_factor", "q_home", "q_capture"}
            }
        )

    # Cameras ARE substituted: the imported rig's camera set is the truth, and keeping a
    # template camera the rig does not have would abort collection at warmup.
    cameras = raw.get("cameras") or {}
    if cameras:
        out: dict[str, Any] = {}
        if "perception" in cameras:
            out["perception"] = str(_deref(cameras["perception"]))
        for key in ("hand", "external", "external_2"):
            cam = cameras.get(key)
            if isinstance(cam, dict) and cam.get("serial") is not None:
                out[key] = {
                    "serial": str(_deref(cam["serial"])),
                    "type": str(_deref(cam.get("type", "zed"))),
                    "resolution": str(_deref(cam.get("resolution", "HD720"))),
                    "fps": int(_deref(cam.get("fps", 15))),
                }
        if out.get("hand") or out.get("external"):
            data["cameras"] = out

    perc = raw.get("perception") or {}
    if perc:
        target = options.setdefault("perception", {})
        m2t2 = perc.get("m2t2") or {}
        if m2t2:
            # The upstream URL embeds ${oc.env:TIPTOP_M2T2_PORT,8123} mid-string, which is
            # exactly the case _deref has to handle.
            target["m2t2"] = {
                "url": str(_deref(m2t2.get("url", "http://localhost:8123"))),
                "apply_bounds": bool(m2t2.get("apply_bounds", True)),
            }
        sam = perc.get("sam") or {}
        if sam.get("mode"):
            target["sam_mode"] = str(_deref(sam["mode"]))
        # A rig with a SAM-2 server names it here, and tiptop reads it for any mode but "local":
        # dropped, the imported profile would warm with no perception.sam.url at all.
        if sam.get("url"):
            target["sam_url"] = str(_deref(sam["url"]))
        smoothing = perc.get("depth_smoothing") or {}
        if smoothing.get("num_frames") is not None:
            target["depth_smoothing_frames"] = int(smoothing["num_frames"])
        for key in (
            "robot_mask_margin_m",
            "depth_trunc_m",
            "voxel_downsample_size",
            "contact_threshold_m",
            "mask_erosion_pixels",
        ):
            if perc.get(key) is not None:
                target[key] = perc[key]


def _merge_tamp_config(data: dict, options: dict, raw: dict, notes: list[str]) -> None:
    task: dict[str, Any] = {}
    if raw.get("prompt"):
        task["prompt"] = str(raw["prompt"])
    if raw.get("tamp_prompt") and raw.get("tamp_prompt") != raw.get("prompt"):
        task["goal"] = str(raw["tamp_prompt"])
    if raw.get("num_episodes"):
        task["target_episodes"] = int(raw["num_episodes"])
    if task:
        data["task"] = task

    overrides = dict(raw.get("tamp_overrides") or {})
    # Keys tiptop has but tandem refuses, because on the path tandem runs they would do nothing
    # (tamp_keys.REFUSED): tiptop's own loop's auto_mode, reset_placement_region, clear_goal_surfaces.
    # A profile stating one is refused when it loads, so that leaving one out is a decision and never a
    # surprise. Here the import is the decision: each is left out with a warning naming it and why, at
    # the one moment a person is looking at this config. Refusing instead would make such a config
    # unimportable, though everything else in it -- the task, the rest of the TAMP settings, the hitl
    # block -- carries over, and the file is not tandem's to fix. (The placement_* keys of
    # 1_toy_puzzle_v3 and every 4_bread_box* used to be among them; the pinned TiPToP reads them now,
    # and they import as they are.)
    refused: dict[str, list[str]] = {}
    for key in [k for k in overrides if k in tamp_keys.REFUSED]:
        overrides.pop(key)
        refused.setdefault(tamp_keys.REFUSED[key], []).append(key)
    for reason, keys in refused.items():
        one = len(keys) == 1
        what = f"TAMP setting {keys[0]}" if one else f"TAMP settings {', '.join(keys)}"
        # The reason's first sentence says why; the rest is advice for a profile, not an import.
        why = reason.split(". ")[0]
        notes.append(
            f"{WARNING_NOTE}{what} not imported: {'it' if one else 'each'} {why}. The runs this config made had "
            f"{'it' if one else 'them'}; check the task still works without {'it' if one else 'them'}."
        )
    if raw.get("tamp_overrides"):
        _note_lj_behaviours(overrides, notes)
        # Substituted, not merged: an upstream cfg/tamp file is a complete TAMP specification,
        # and folding the template's defaults into it would change the trajectories it
        # produces. Unknown keys fail loudly rather than import a knob that does nothing --
        # upstream configs really do contain such typos, and they cost whole datasets.
        try:
            options["tamp"] = validate_tamp(overrides)
        except ValueError as exc:
            raise TandemError(
                f"The TAMP settings in this config are not valid:\n  {exc}",
                hint=(
                    "This is usually a typo that the original system ignored silently. "
                    "Fix it in the source file, or import without --tamp-config and add the "
                    "settings with `tandem profile edit`."
                ),
            ) from exc

    # The upstream files misspell it; accept both so a real config imports.
    slug = raw.get("hugginface_slug") or raw.get("huggingface_slug")
    if slug:
        data["export"] = {"hf_repo": str(slug), "private": False}


#: What LJ1356's tiptop did unconditionally, from the commits the monorepo's placement configs were
#: tuned on (LJ1356/tiptop@37b9678 and @ffe370a), and the switch that does it in the pinned TiPToP, off
#: by default there. The keys are ported under the same names (tamp_keys.py).
LJ_BEHAVIOURS: dict[str, str] = {
    "table_plane_support_vote": "chose the table plane by the objects resting on it",
    "disjoint_object_masks": "built object meshes and point clouds from disjoint masks",
    "blend_stretch_to_caps": "slowed a stroke it could not re-time into the velocity and acceleration caps",
}


def _note_lj_behaviours(overrides: dict, notes: list[str]) -> None:
    """Say what a placement config's runs had that its keys do not ask for, rather than turn it on.

    A config that sets ``placement_support`` came from LJ1356's tiptop, where the three behaviours in
    ``LJ_BEHAVIOURS`` were not settings but the code, so the config never names them. The pinned TiPToP
    has each behind a switch, off by default, so the placement keys import exactly (they are the same
    keys) and still do not plan as those runs did. Adding the switches here would be the translation
    layer this importer exists not to have -- a key in the profile the file never said -- so the
    import names them instead, at the one moment someone is reading this config.
    """
    if overrides.get("placement_support") is not True:
        return
    missing = [key for key in LJ_BEHAVIOURS if key not in overrides]
    if not overrides.get("blend_trajectory") and "blend_stretch_to_caps" in missing:
        missing.remove("blend_stretch_to_caps")  # nothing to stretch with blending off
    if not missing:
        return
    did = "; ".join(LJ_BEHAVIOURS[key] for key in missing)
    keys = missing[0] if len(missing) == 1 else f"{', '.join(missing[:-1])} and {missing[-1]}"
    notes.append(
        f"{WARNING_NOTE}this config's placement settings were tuned on LJ1356's tiptop, which always {did}. "
        f"The pinned TiPToP does {'that' if len(missing) == 1 else 'each'} only when asked: to plan as those "
        f"runs did, also set {keys} to true (`tandem profile edit`)."
    )


# --------------------------------------------------------------------------- the hitl block
#
# A cfg/tamp file's `hitl:` block is read, upstream, by tiptop/hitl/config.py (HITLConfig) at
# LJ1356/tiptop cf75a68, the tiptop hitl-tamp-vla pins. In tandem the same settings are the profile's
# own `hitl:` (core/profiles.py HitlSpec): phase planning is the part of that system tandem
# re-implemented rather than wrapped, so most keys carry over as they are. HITL_KEYS is every key
# cf75a68 accepts and what each becomes here, and anything else is refused. cf75a68 refuses unknown
# hitl keys too (resolve_hitl_config), so a config carrying one never ran there either.

#: What a learned policy's own settings become: nothing. See ``policy_type``.
_POLICY_ONLY = (
    "a setting of the learned-policy executor that policy_type selects. tandem has no such executor, "
    "so there is nothing for it to configure"
)

#: monorepo ``hitl`` key -> (the tandem ``hitl`` key it becomes, or None for none; why).
HITL_KEYS: dict[str, tuple[str | None, str]] = {
    # Same name, same meaning: tandem's phase planning ports what each of these switched.
    "enabled": ("enabled", "phase planning on or off"),
    "proposal_model": ("proposal_model", "the model that splits the task into phases and invents operators"),
    "vlm_model": ("vlm_model", "the model that answers each per-atom check"),
    "max_attempts": ("max_attempts", "proposals tried in all, each rejection fed back to the next"),
    "classify_initial": ("classify_initial", "classify the invented predicates on the first image"),
    "verify_retries": ("verify_retries", "extra goes at a human phase whose check failed"),
    "verify_enforced": ("verify_enforced", "a check that still fails ends the trial as a failure"),
    "check_plan_effects": ("check_plan_effects", "the symbolic contract check, before the arm moves"),
    "check_human_preconditions": ("check_human_preconditions", "a human phase's preconditions, on camera"),
    "check_human_effects": ("check_human_effects", "a human phase's add and delete effects, on camera"),
    "check_tamp_preconditions": ("check_tamp_preconditions", "what a robot leg starts from, on camera"),
    "check_tamp_effects": ("check_tamp_effects", "what a robot leg achieved, on camera"),
    "precondition_enforced": ("precondition_enforced", "an unmet precondition stops the trial"),
    "save_vlm_io": ("save_vlm_io", "every image sent to a model, and its reply, kept beside the rollout"),
    # A relative path meant the tiptop process's working directory there; here it means beside the
    # profile (profiles.resolve_cache_path). Harmless: it is a cache, and a miss only asks again.
    "cache_path": ("cache_path", "the SQLite cache of proposals"),
    # Renamed, and the value translated. Only "human" has a translation: see HUMAN_EXECUTORS.
    "policy_type": ("human_executor", "who carries out a human phase: 'human' is the teleop hand-off"),
    # Checked, not carried. Which planner does a profile's robot phases is its planner.backend, and
    # this importer builds TiPToP profiles, whose robot phases are cuTAMP's.
    "robot_planner": (None, "which planner carries out the robot phases: cutamp, which is what TiPToP runs"),
    # The learned-policy executor's own settings (hitl-baseline): see policy_type.
    "policy_checkpoint": (None, _POLICY_ONLY),
    "open_loop_horizon": (None, _POLICY_ONLY),
    "policy_num_inference_steps": (None, _POLICY_ONLY),
    "policy_max_steps": (None, _POLICY_ONLY),
    "policy_velocity_scale": (None, _POLICY_ONLY),
    "policy_start_joint_angle": (None, _POLICY_ONLY),
    "policy_python": (None, _POLICY_ONLY),
}

#: The keys that configure a learned policy and nothing else.
POLICY_KEYS = tuple(key for key, (_, why) in HITL_KEYS.items() if why is _POLICY_ONLY)

#: The names cf75a68 registers (tiptop/hitl/planners.py): its one robot planner, and who can carry out
#: a human phase. "human" is the teleoperator. "diffusion" and "act" are LeRobot policies trained on
#: the teleop legs of earlier runs of the same task: the paper's HITL-TAMP baseline.
ROBOT_PLANNERS = ("cutamp",)
POLICY_TYPES = ("human", "diffusion", "act")

#: What a policy_type becomes in tandem (``hitl.human_executor``), where tandem has a counterpart.
HUMAN_EXECUTORS = {"human": "teleop"}


def _merge_hitl(data: dict, raw: dict, source: Path, notes: list[str]) -> None:
    """The config's ``hitl:`` block, as the profile's ``hitl:``. Refuses what it cannot carry over faithfully.

    Laid over the template's ``hitl:``, so a key the block leaves out keeps tandem's default. For the
    keys hitl-tamp-vla has, the two defaults are the same (tests/test_import_hitl.py).
    """
    block = raw.get("hitl")
    if block is None:
        return
    if not isinstance(block, dict):
        raise TandemError(f"The hitl block in {source.name} is a {type(block).__name__}, not a mapping.")

    unknown = sorted(str(key) for key in block if key not in HITL_KEYS)
    if unknown:
        lines = []
        for key in unknown:
            close = difflib.get_close_matches(key, list(HITL_KEYS), n=1, cutoff=0.6)
            lines.append(f"  hitl.{key}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise TandemError(
            f"{source.name} has hitl setting(s) hitl-tamp-vla does not have:\n" + "\n".join(lines),
            hint="hitl-tamp-vla refuses these too, so this config never ran as written. Fix it in the source "
            "file. The settings it can have are HITL_KEYS in tandem/planners/tiptop/importers.py.",
        )

    # Truthiness, as cf75a68 reads it (`if not cfg.enabled`). What matters below is whether that system
    # ever ran a phase under these settings.
    enabled = bool(block.get("enabled", False))
    # Who planned the robot phases and who did the human ones. With phase planning off there were no
    # phases, so neither choice ever took effect, and nothing is lost by leaving it behind. With it on,
    # a choice tandem cannot honour is refused: the profile would collect something else under this
    # config's name.
    robot_planner = block.get("robot_planner", "cutamp")
    if robot_planner not in ROBOT_PLANNERS:
        if enabled:
            raise TandemError(
                f"{source.name} has its robot phases planned by {robot_planner!r} (hitl.robot_planner). A "
                "profile imported from it plans with TiPToP, whose robot phases are cuTAMP's -- the only "
                "robot planner hitl-tamp-vla ships.",
                hint="In tandem a profile's robot phases are its planner.backend's. If cuTAMP is what the "
                "config meant, set hitl.robot_planner: cutamp in the source file (or leave it out).",
            )
        notes.append(
            f"{WARNING_NOTE}hitl.robot_planner {robot_planner!r} not imported: phase planning is off in this "
            "config, so it never planned a phase. With phase planning on, this profile's robot phases are "
            "cuTAMP's."
        )

    policy_type = block.get("policy_type", "human")
    policy_set = [key for key in POLICY_KEYS if key in block]
    if policy_type not in POLICY_TYPES:
        close = difflib.get_close_matches(str(policy_type), POLICY_TYPES, n=1, cutoff=0.6)
        raise TandemError(
            f"{source.name} has hitl.policy_type {policy_type!r}, which hitl-tamp-vla does not have"
            + (f" (did you mean {close[0]!r}?)" if close else "")
            + ".",
            hint=f"It has {', '.join(POLICY_TYPES)}. Fix it in the source file.",
        )
    if policy_type not in HUMAN_EXECUTORS:
        if enabled:
            raise TandemError(*_refuse_policy(source, raw, str(policy_type)))
        settings = f" and its settings ({', '.join(policy_set)})" if policy_set else ""
        notes.append(
            f"{WARNING_NOTE}hitl.policy_type {policy_type!r}{settings} not imported: phase planning is off "
            "in this config, so no phase ever ran with it. With phase planning on, a person does this "
            "profile's human phases (hitl.human_executor: teleop)."
        )
        policy_type = "human"
    elif policy_set:
        # Not a warning: with policy_type human, cf75a68 never read them either.
        notes.append(
            f"hitl {', '.join(policy_set)} not imported: {'it is' if len(policy_set) == 1 else 'each is'} "
            "read only when policy_type names a learned policy, and this config's is 'human'"
        )

    hitl: dict[str, Any] = dict(data.get("hitl") or {})
    for key, value in block.items():
        target, _why = HITL_KEYS[key]
        if target is None:
            continue
        hitl[target] = HUMAN_EXECUTORS[policy_type] if key == "policy_type" else value
    data["hitl"] = hitl

    cache = block.get("cache_path")
    if cache and not Path(str(cache)).expanduser().is_absolute():
        notes.append(
            f"hitl.cache_path {cache!r} is relative: it now means beside the profile, not the directory "
            "tiptop ran in"
        )
    if enabled:
        # What the block could not say, because hitl-tamp-vla had no setting for it, keeps tandem's
        # default. Two of those defaults do what the paper says rather than what that code did. A person
        # reproducing its runs has to know which, and which setting restores the old behaviour.
        notes.append(
            "phase planning settings from its hitl block. Two of tandem's own differ from what hitl-tamp-vla "
            "did: a trial whose human phase still fails its check is excluded without asking for a label "
            "(hitl.on_verification_failure: exclude; `label` asks, as it did), and the last phase is checked "
            "too (hitl.verify_final_phase: true; `false` leaves it to the label, as it did)"
        )


def _refuse_policy(source: Path, raw: dict, policy_type: str) -> tuple[str, str]:
    """The refusal of a config whose human phases a learned policy carries out, and what to import instead.

    Refused, not mapped to teleop with a warning. Such a config exists to collect the HITL-TAMP
    baseline, where the policy's legs are the data: each has its own dataset slug
    (``3_pen_open_book_diffusion``). Imported as teleop, it would collect a person's legs under that
    name, and the dataset would say it is something it is not -- a mistake a warning scrolled past
    at import cannot undo once episodes are pushed. A learned-policy executor is not part of
    tandem. And every such config in hitl-tamp-vla has a twin a person runs, with the same task and TAMP
    settings, so the right import is one file away. The refusal names it.
    """
    slug = raw.get("hugginface_slug") or raw.get("huggingface_slug")
    dataset = f" (dataset {slug})" if slug else ""
    message = (
        f"{source.name} hands its human phases to a learned {policy_type} policy (hitl.policy_type: "
        f"{policy_type}), and tandem has no executor for that: only a person, through teleop. Imported "
        f"as teleop, it would collect a person's demonstrations under a config{dataset} meant for the "
        "policy's."
    )
    twins = _human_twins(source, raw, policy_type)
    if twins:
        hint = (
            "The same task, with a person doing its human phases: "
            + ", ".join(t.name for t in twins)
            + ". Import that one instead (--tamp-config)."
        )
    else:
        hint = (
            "Import a config of the same task whose human phases a person does (policy_type: human, or "
            "left out), or copy this one and remove policy_type and the policy_* settings."
        )
    return message, hint


def _human_twins(source: Path, raw: dict, policy_type: str) -> list[Path]:
    """Configs beside ``source`` for the same task, with phase planning on and a person doing the human phases.

    hitl-tamp-vla names a task's policy variant ``<task>_<policy>.yml``, beside ``<task>.yml`` and
    ``<task>_v3.yml``. A twin must also state the same prompt: a file that only shares the prefix is
    another task.
    """
    suffix = f"_{policy_type}"
    if not source.stem.endswith(suffix):
        return []
    base = source.stem[: -len(suffix)]
    twins = []
    for path in sorted(source.parent.glob(f"{base}*.yml")):
        if path == source:
            continue
        try:
            other = _load_yaml(path)
        except TandemError:
            continue
        hitl = other.get("hitl") if isinstance(other.get("hitl"), dict) else {}
        if (
            other.get("prompt") == raw.get("prompt")
            and hitl.get("enabled")
            and hitl.get("policy_type", "human") in HUMAN_EXECUTORS
        ):
            twins.append(path)
    return twins


def resolve_hf_repo(slug: str, org: str | None) -> str:
    """`toys20` + org -> `org/toys20`; an already-qualified slug is left alone."""
    if "/" in slug or not org:
        return slug
    return f"{org}/{slug}"


# --------------------------------------------------------------------------- as the factory offers it


class HitlTampVlaImporter:
    """The monorepo importer, as TiPToP's factory offers it (``tandem.planners.base.ProfileImporter``)."""

    source = "a hitl-tamp-vla checkout"

    def find(self, near: Path) -> Path | None:
        """A hitl-tamp-vla checkout at or above ``near``. Only a suggestion: it is confirmed before any read."""
        here = Path(near).resolve()
        for base in (here, *here.parents):
            for name in ("hitl-tamp-vla", "tamp-vla"):
                candidate = base / name
                if candidate.is_dir() and find_sources(candidate)["tiptop_config"]:
                    return candidate
        return None

    def configs(self, source: Path) -> list[Path]:
        return list_tamp_configs(source)

    def build(
        self, name: str, *, source: Path | None = None, config: Path | None = None
    ) -> tuple[Profile, dict, list[str]]:
        return build_profile(name, root=source, tamp_config=config)


IMPORTER = HitlTampVlaImporter()
