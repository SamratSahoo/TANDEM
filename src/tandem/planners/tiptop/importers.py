"""Import an existing hitl-tamp-vla setup into a tandem profile.

The monorepo split one setup across two files — ``tiptop/tiptop/config/tiptop.yml`` (robot,
cameras, perception) and ``data-collection/cfg/tamp/<name>.yml`` (task, TAMP overrides,
episode target, HF slug) — plus a calibration JSON keyed by camera serial. This reassembles
them into a single profile so an existing rig is one command away from working.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from tandem.core.errors import TandemError
from tandem.core.profiles import Profile, resolve_interpolation, validate_tamp

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

    cfg_path = sources.get("tiptop_config")
    if cfg_path and cfg_path.is_file():
        _merge_tiptop_config(data, _load_yaml(cfg_path))
        notes.append(f"robot and cameras from {cfg_path}")
    elif root is not None:
        notes.append(
            f"no tiptop.yml under {root} — using template defaults for the robot and cameras "
            "(is the tiptop submodule checked out?)"
        )

    if tamp_config is not None:
        if not tamp_config.is_file():
            raise TandemError(f"No such TAMP config: {tamp_config}")
        _merge_tamp_config(data, _load_yaml(tamp_config))
        data["description"] = f"imported from {tamp_config.name}"
        notes.append(f"task and TAMP settings from {tamp_config}")

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
            f"The imported settings do not form a valid profile:\n{exc}",
            hint="Import what works, then fix the rest with `tandem profile edit`.",
        ) from exc
    return profile, calibration, notes


def _merge_tiptop_config(data: dict, raw: dict) -> None:
    # Merged into the template's defaults, not substituted for them: an upstream tiptop.yml
    # may omit a field (q_capture, gripper_port) that the profile schema still needs.
    robot = raw.get("robot") or {}
    if robot:
        data.setdefault("robot", {}).update(
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
        target = data.setdefault("perception", {})
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
            target["sam_mode"] = str(sam["mode"])
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


def _merge_tamp_config(data: dict, raw: dict) -> None:
    task: dict[str, Any] = {}
    if raw.get("prompt"):
        task["prompt"] = str(raw["prompt"])
    if raw.get("tamp_prompt") and raw.get("tamp_prompt") != raw.get("prompt"):
        task["goal"] = str(raw["tamp_prompt"])
    if raw.get("num_episodes"):
        task["target_episodes"] = int(raw["num_episodes"])
    if task:
        data["task"] = task

    overrides = raw.get("tamp_overrides") or {}
    if overrides:
        # Substituted, not merged: an upstream cfg/tamp file is a complete TAMP specification,
        # and folding the template's defaults into it would change the trajectories it
        # produces. Unknown keys fail loudly rather than import a knob that does nothing --
        # upstream configs really do contain such typos, and they cost whole datasets.
        try:
            data["tamp"] = validate_tamp(dict(overrides))
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


def resolve_hf_repo(slug: str, org: str | None) -> str:
    """`toys20` + org -> `org/toys20`; an already-qualified slug is left alone."""
    if "/" in slug or not org:
        return slug
    return f"{org}/{slug}"
