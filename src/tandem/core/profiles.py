"""Profiles — a named collection setup and the trajectories it produced.

One profile replaces three overlapping ideas from the source monorepo:

  * ``cfg/tamp/<name>.yml``  — task prompt, TAMP overrides, episode target, HF slug
  * ``DC_WORKSPACE``         — per-robot data scoping and camera extrinsics
  * ``settings.json``        — robot host/ports, camera serials

On disk::

    <data_root>/profiles/<name>/
        profile.yml
        calibration.json          # camera extrinsics keyed by serial
        trajectories/
            eval/<ts>/            # collected, not yet labeled
            success/<ts>/
            failure/<ts>/
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator
from ruamel.yaml import YAML

from tandem.core import settings as settings_mod
from tandem.core import tamp_keys
from tandem.core.errors import ProfileError

# Same rule the source used for DC_WORKSPACE: a safe single path segment (no traversal) that
# is also a valid HuggingFace repo-name fragment.
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

STATUSES = ("eval", "success", "failure")

_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.width = 100


# --------------------------------------------------------------------------- models


class TaskSpec(BaseModel):
    model_config = {"extra": "forbid"}

    prompt: str = "Place the toys on the plate with no collisions"
    # The goal instruction handed to the planner, when it must differ from the language label
    # stored with the episode. Absent means "they are the same thing".
    goal: str | None = None
    target_episodes: int = 20

    @field_validator("target_episodes")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("target_episodes must be > 0")
        return v


class RobotSpec(BaseModel):
    model_config = {"extra": "forbid"}

    type: str = "fr3_robotiq"
    dof: int = 7
    host: str = "172.16.0.2"
    port: int = 5555
    gripper_port: int = 5559
    # The bamboo shim's --state-port. JointSampler reads encoders here while the control
    # thread is parked inside a blocking move; without it, capture aborts.
    state_port: int = 5557
    time_dilation_factor: float = 0.2
    q_home: list[float] = Field(default_factory=lambda: [0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.0])
    q_capture: list[float] = Field(
        default_factory=lambda: [-0.034, 0.090, 0.080, -1.319, -0.003, 1.253, 0.030]
    )

    @field_validator("type")
    @classmethod
    def _known_robot(cls, v: str) -> str:
        if v not in {"fr3_robotiq", "ur5", "franka", "panda_robotiq"}:
            raise ValueError(f"unsupported robot type {v!r} (fr3_robotiq | ur5 | franka | panda_robotiq)")
        return v

    @field_validator("time_dilation_factor")
    @classmethod
    def _sane_tdf(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("time_dilation_factor must be in (0, 1]; start at 0.2 (20% speed)")
        return v

    @model_validator(mode="after")
    def _joint_counts(self):
        for field in ("q_home", "q_capture"):
            vals = getattr(self, field)
            if len(vals) != self.dof:
                raise ValueError(f"{field} has {len(vals)} values but robot.dof is {self.dof}")
        return self


class CameraSpec(BaseModel):
    model_config = {"extra": "forbid"}

    serial: str
    type: str = "zed"
    resolution: str = "HD720"
    # 15, not 30: three ZEDs at HD720@30 exceed the USB bandwidth budget and fail to open.
    # The LeRobot export resamples to 15 Hz anyway, so nothing is lost.
    fps: int = 15


class CamerasSpec(BaseModel):
    model_config = {"extra": "forbid"}

    # Which camera the perception pipeline reads. "external" leaves the arm at q_home (a
    # static third-person pose comes straight from calibration); "hand" drives the arm to
    # q_capture and recomputes the pose by forward kinematics each time.
    perception: str = "external"
    hand: CameraSpec | None = None
    external: CameraSpec | None = None
    # Recorded as DROID exterior_2. Omit for a deliberate two-camera rig: when it IS
    # configured and fails to open, tiptop aborts before any rollout rather than collect an
    # episode with a missing camera.
    external_2: CameraSpec | None = None

    @field_validator("perception")
    @classmethod
    def _perception_choice(cls, v: str) -> str:
        if v not in {"hand", "external"}:
            raise ValueError("cameras.perception must be 'hand' or 'external'")
        return v

    @model_validator(mode="after")
    def _perception_camera_present(self):
        # A profile with no cameras at all is legitimate: on a laptop it is just a folder of
        # trajectories collected elsewhere. But a profile that HAS cameras and points
        # perception at one it does not have would fail at warmup, minutes in, with a message
        # about the wrong thing.
        if not self.configured():
            return self
        if getattr(self, self.perception) is None:
            raise ValueError(
                f"cameras.perception is {self.perception!r} but cameras.{self.perception} is not configured"
            )
        return self

    def configured(self) -> dict[str, CameraSpec]:
        return {
            key: cam
            for key in ("hand", "external", "external_2")
            if (cam := getattr(self, key)) is not None
        }


class GeminiSpec(BaseModel):
    model_config = {"extra": "forbid"}

    model: str = "gemini-robotics-er-1.6-preview"
    temperature: float | None = None


class M2T2Spec(BaseModel):
    model_config = {"extra": "forbid"}

    url: str = "http://localhost:8123"
    apply_bounds: bool = True


class PerceptionSpec(BaseModel):
    model_config = {"extra": "forbid"}

    gemini: GeminiSpec = Field(default_factory=GeminiSpec)
    m2t2: M2T2Spec = Field(default_factory=M2T2Spec)
    sam_mode: str = "local"
    # Temporal depth smoothing: N stereo frames grabbed back-to-back at the static capture
    # pose and per-pixel median-fused. 1 disables it.
    depth_smoothing_frames: int = 5
    # Padding on the robot's collision spheres before they are projected out of a
    # third-person point cloud. Raise it if a rim of arm survives.
    robot_mask_margin_m: float = 0.02
    depth_trunc_m: float = 5.0
    voxel_downsample_size: float = 0.0075
    contact_threshold_m: float = 0.01
    mask_erosion_pixels: int = 3


class RecordingSpec(BaseModel):
    model_config = {"extra": "forbid"}

    enabled: bool = True
    fps: int = 15


class ExportSpec(BaseModel):
    model_config = {"extra": "forbid"}

    hf_repo: str = ""
    private: bool = False


class Profile(BaseModel):
    """A validated profile. `raw_dir` is where it came from; not part of the file."""

    model_config = {"extra": "forbid"}

    version: int = 1
    name: str = "default"
    description: str = ""
    task: TaskSpec = Field(default_factory=TaskSpec)
    robot: RobotSpec = Field(default_factory=RobotSpec)
    cameras: CamerasSpec = Field(default_factory=CamerasSpec)
    perception: PerceptionSpec = Field(default_factory=PerceptionSpec)
    # Flat, using tiptop's own key names -- see tandem/core/tamp_keys.py for why.
    tamp: dict[str, Any] = Field(default_factory=dict)
    recording: RecordingSpec = Field(default_factory=RecordingSpec)
    export: ExportSpec = Field(default_factory=ExportSpec)

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        if not NAME_RE.match(v):
            raise ValueError(
                f"profile name {v!r} is invalid; use lowercase letters, digits, '-' and '_' "
                f"(must match {NAME_RE.pattern})"
            )
        return v

    @field_validator("tamp")
    @classmethod
    def _valid_tamp(cls, v: dict) -> dict:
        return validate_tamp(v)

    # ---- paths -------------------------------------------------------------

    def dir(self) -> Path:
        return profiles_root() / self.name

    def profile_file(self) -> Path:
        return self.dir() / "profile.yml"

    def calibration_file(self) -> Path:
        return self.dir() / "calibration.json"

    def trajectories_dir(self) -> Path:
        return self.dir() / "trajectories"

    def status_dir(self, status: str) -> Path:
        if status not in STATUSES:
            raise ProfileError(f"unknown status {status!r}", hint=f"one of: {', '.join(STATUSES)}")
        return self.trajectories_dir() / status

    def goal_or_prompt(self) -> str:
        return self.task.goal or self.task.prompt


# --------------------------------------------------------------------------- tamp validation


def validate_tamp(raw: dict | None) -> dict:
    """Type-check and normalise a ``tamp:`` block against tiptop's real key set.

    Unknown keys are a hard error with a suggestion. Silently-ignored overrides are the
    single worst failure mode here: a run looks fine, produces data, and the knob you were
    studying never applied.
    """
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("tamp must be a mapping")

    out: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None:
            continue
        if key not in tamp_keys.ALL_KEYS:
            hint = tamp_keys.suggest(key)
            extra = f" Did you mean: {', '.join(hint)}?" if hint else ""
            raise ValueError(f"unknown TAMP setting {key!r}.{extra}")

        if key == "traj_length_norm":
            out[key] = _normalise_traj_norm(value)
        elif key in tamp_keys.PATH_KEYS:
            out[key] = str(value)
        elif key in tamp_keys.LIST_KEYS:
            out[key] = _validate_blend_ops(key, value)
        elif key in tamp_keys.INDEX_MAP_KEYS:
            out[key] = _validate_index_map(key, value)
        else:
            out[key] = _coerce_scalar(key, value)

    _check_enums(out)
    _check_positives(out)
    return out


def _normalise_traj_norm(value: Any) -> str | float:
    """Force the infinity norm to the STRING "inf".

    YAML `inf` parses as a string but `.inf` parses as float infinity, and the overrides dict
    round-trips through JSON, which cannot represent Infinity. Both spellings land here as
    the one form tiptop's resolve_traj_length_norm accepts.
    """
    if isinstance(value, str):
        if value.strip().lower() in tamp_keys.INFINITY_ALIASES:
            return "inf"
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(
                f"traj_length_norm must be a number or 'inf' (got {value!r})"
            ) from exc
    if isinstance(value, (int, float)):
        if math.isinf(float(value)):
            return "inf"
        return float(value)
    raise ValueError(f"traj_length_norm must be a number or 'inf' (got {value!r})")


def _validate_blend_ops(key: str, value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key} must be a list of operation names")
    ops = [str(x).strip() for x in value]
    unknown = [o for o in ops if o not in tamp_keys.BLEND_OPS]
    if unknown:
        # The source shipped a config with `MoveFree. MoveHolding` -- a typo'd '.' that YAML
        # folded into one token and the planner silently ignored. Catch that class here.
        raise ValueError(
            f"{key} names unknown operations {unknown}; valid: {', '.join(tamp_keys.BLEND_OPS)}"
        )
    return ops


def _validate_index_map(key: str, value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping of index -> value, e.g. {{0: 1.0}}")
    out: dict[str, float] = {}
    for idx, val in value.items():
        try:
            out[str(int(idx))] = float(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key}[{idx!r}] = {val!r} is not an index -> number pair") from exc
    return out


def _coerce_scalar(key: str, value: Any) -> Any:
    expected = tamp_keys.SCALAR_KEYS[key]
    try:
        if expected is bool:
            if isinstance(value, bool):
                return value
            raise ValueError
        if expected is int:
            if isinstance(value, bool):
                raise ValueError
            return int(value)
        if expected is float:
            if isinstance(value, bool):
                raise ValueError
            return float(value)
        return str(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a {expected.__name__} (got {value!r})") from exc


def _check_enums(cfg: dict) -> None:
    for key, allowed in tamp_keys.ENUMS.items():
        if key in cfg and str(cfg[key]).lower() not in allowed:
            raise ValueError(f"{key} must be one of {sorted(allowed)} (got {cfg[key]!r})")


def _check_positives(cfg: dict) -> None:
    for key in ("num_particles", "opt_steps_per_skeleton", "blend_speed_scale", "blend_pace_scale"):
        if key in cfg and cfg[key] <= 0:
            raise ValueError(f"{key} must be > 0 (got {cfg[key]})")
    for key in ("blend_boundary_window", "blend_boundary_window_sec", "blend_profile_end_sec"):
        if key in cfg and cfg[key] < 0:
            raise ValueError(f"{key} must be >= 0 (got {cfg[key]})")
    tdf = cfg.get("time_dilation_factor_literal")
    if tdf is not None and not 0.0 < tdf <= 1.0:
        raise ValueError(f"time_dilation_factor_literal must be in (0, 1] (got {tdf})")


# --------------------------------------------------------------------------- store


def profiles_root() -> Path:
    return settings_mod.load().profiles_root()


def list_names() -> list[str]:
    root = profiles_root()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "profile.yml").is_file())


def exists(name: str) -> bool:
    return (profiles_root() / name / "profile.yml").is_file()


def load(name: str | None = None) -> Profile:
    """Load a profile by name, or the active one."""
    cfg = settings_mod.load()
    name = name or cfg.active_profile
    path = profiles_root() / name / "profile.yml"
    if not path.is_file():
        known = list_names()
        hint = (
            f"Known profiles: {', '.join(known)}."
            if known
            else "No profiles exist yet — run `tandem init` or `tandem profile create <name>`."
        )
        raise ProfileError(f"Profile {name!r} not found at {path}.", hint=hint)
    return load_file(path, name=name)


def load_file(path: Path, *, name: str | None = None) -> Profile:
    try:
        with path.open() as fh:
            data = _yaml.load(fh) or {}
    except ProfileError:
        raise
    except Exception as exc:
        raise ProfileError(f"{path} is not valid YAML: {exc}") from exc
    data = _plain(data)
    if name is not None:
        data.setdefault("name", name)
        if data.get("name") != name:
            # The directory is the identity; a stale `name:` inside the file would make
            # run dirs and HF slugs disagree with where the data actually lives.
            data["name"] = name
    try:
        return Profile.model_validate(data)
    except Exception as exc:
        raise ProfileError(f"{path} is not a valid profile:\n{format_errors(exc)}") from exc


def save(profile: Profile) -> Path:
    """Write a profile, creating its directory tree. Round-trips through validation first."""
    Profile.model_validate(profile.model_dump())
    pdir = profile.dir()
    pdir.mkdir(parents=True, exist_ok=True)
    for status in STATUSES:
        (pdir / "trajectories" / status).mkdir(parents=True, exist_ok=True)
    if not profile.calibration_file().is_file():
        profile.calibration_file().write_text("{}\n")
    path = profile.profile_file()
    with path.open("w") as fh:
        _yaml.dump(_dump_dict(profile), fh)
    return path


def _dump_dict(profile: Profile) -> dict:
    """model_dump with None-valued optionals dropped, so the file stays readable."""
    data = profile.model_dump(mode="python", exclude_none=True)
    cams = data.get("cameras", {})
    for key in ("hand", "external", "external_2"):
        if cams.get(key) is None:
            cams.pop(key, None)
    if not data.get("tamp"):
        data["tamp"] = {}
    return data


def delete(name: str, *, keep_data: bool = True) -> Path:
    """Remove a profile. By default the trajectories survive, mirroring the source's
    non-destructive workspace delete — re-creating the name brings the data back."""
    import shutil

    pdir = profiles_root() / name
    if not pdir.is_dir():
        raise ProfileError(f"Profile {name!r} does not exist.")
    if keep_data:
        pdir.joinpath("profile.yml").unlink(missing_ok=True)
    else:
        shutil.rmtree(pdir)
    return pdir


def calibration(profile: Profile) -> dict:
    path = profile.calibration_file()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ProfileError(f"{path} is not valid JSON: {exc}") from exc


def missing_calibration(profile: Profile) -> list[str]:
    """Configured camera serials with no extrinsics entry.

    Extrinsics are keyed by serial, and tiptop raises at warmup for a serial it cannot find —
    better to say so before a session starts.
    """
    known = set(calibration(profile))
    return [cam.serial for cam in profile.cameras.configured().values() if cam.serial not in known]


# --------------------------------------------------------------------------- helpers


def _plain(obj):
    """ruamel returns CommentedMap/CommentedSeq; pydantic wants plain containers."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def format_errors(exc: Exception) -> str:
    """Turn a pydantic ValidationError into something a human wants to read."""
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return str(exc)
    lines = []
    for err in errors():
        loc = ".".join(str(p) for p in err.get("loc", ())) or "(root)"
        msg = err.get("msg", "")
        msg = msg.removeprefix("Value error, ")
        lines.append(f"  {loc}: {msg}")
    return "\n".join(lines)
