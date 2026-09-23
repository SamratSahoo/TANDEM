"""TiPToP's ``planner.options``: the arm it drives, how it perceives, and its solver overrides.

    planner:
      backend: tiptop
      options:
        robot:       the arm and the bamboo-polymetis shim it is reached through
        perception:  Gemini detection, the M2T2 grasp server, SAM-2 and the depth pipeline
        tamp:        cuTAMP / cuRobo overrides, by tiptop's own key names (tamp_keys.py)

These three were top-level sections of every profile while TiPToP was the only planner. Nothing in
tandem reads them but TiPToP -- the teleop executor, the merge and the export do not -- so they are
TiPToP's to define and to validate, and a planner that is not TiPToP never sees them. The profile's
``cameras`` block stays tandem's: the teleop executor records from those cameras, and the dataset's
camera layout (hand, external, external_2) is tandem's recording format, not the planner's.

Validated when a profile naming TiPToP loads (``FACTORY.validate_options``), with the same loud
errors these sections always had -- an unknown TAMP key names the nearest real one -- and again when
the factory builds a session's backend from them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from tandem.planners.tiptop import tamp_keys

# --------------------------------------------------------------------------- the arm


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


# --------------------------------------------------------------------------- perception


#: The detector the pinned tiptop runs: perception/gemini.py's default model_id. Keep it equal on a bump.
DETECTOR_MODEL = "gemini-robotics-er-2-preview"


class GeminiSpec(BaseModel):
    """TiPToP's Gemini detector, as a statement rather than a choice.

    The pinned tiptop reads neither of these: it calls its detector with its own default model and
    temperature, whatever tiptop.yml says. So the block states what runs, and ``render.check_assets``
    warns when it states something else -- refusing would stop every profile written before the
    1c6daf3 bump from loading (they name er-1.6), and dropping the key would leave a profile unable to
    say which detector labelled its data.
    """

    model_config = {"extra": "forbid"}

    model: str = DETECTOR_MODEL
    temperature: float | None = None


class M2T2Spec(BaseModel):
    model_config = {"extra": "forbid"}

    url: str = "http://localhost:8123"
    apply_bounds: bool = True

    @field_validator("url")
    @classmethod
    def _parseable(cls, v: str) -> str:
        """Reject a URL that cannot be parsed, where the field name is still in hand.

        An import used to leave OmegaConf's ``${oc.env:TIPTOP_M2T2_PORT,8123}`` in here, and
        the first thing to notice was urlparse raising several layers away, inside the
        diagnostic command you run *because* something is wrong.
        """
        from urllib.parse import urlparse

        try:
            parsed = urlparse(v)
            parsed.port  # noqa: B018 — raises when the port is not an integer
        except ValueError as exc:
            raise ValueError(f"{v!r} is not a usable URL: {exc}") from exc
        if not parsed.scheme or not parsed.hostname:
            raise ValueError(f"{v!r} needs a scheme and a host, e.g. http://localhost:8123")
        return v


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


# --------------------------------------------------------------------------- the whole block


class TiptopOptions(BaseModel):
    """Everything TiPToP is configured with beyond the profile's task and cameras."""

    model_config = {"extra": "forbid"}

    robot: RobotSpec = Field(default_factory=RobotSpec)
    perception: PerceptionSpec = Field(default_factory=PerceptionSpec)
    # Flat, using tiptop's own key names -- see tamp_keys.py for why. Deliberately NOT where phase
    # planning is configured: that is the profile's `hitl:` block, tandem's own, because it changes
    # what a dataset CONTAINS rather than how the arm moves.
    tamp: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tamp", mode="before")
    @classmethod
    def _valid_tamp(cls, v: Any) -> dict:
        return validate_tamp(v)

    def to_options(self) -> dict[str, Any]:
        """As a profile stores it: plain containers, unset optionals left out so the file stays readable."""
        return self.model_dump(mode="python", exclude_none=True)


def parse(options: Mapping[str, Any] | None) -> TiptopOptions:
    """``planner.options`` as TiPToP reads them. A pydantic ``ValidationError`` names what is wrong."""
    if isinstance(options, TiptopOptions):
        return options
    return TiptopOptions.model_validate(dict(options or {}))


def options_of(profile: Any) -> TiptopOptions:
    """The options of a profile that plans with TiPToP."""
    return parse(profile.planner.options)


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
            if key in tamp_keys.NULL_REFUSED:
                raise ValueError(tamp_keys.NULL_REFUSED[key])
            continue
        if key in tamp_keys.REFUSED:
            raise ValueError(f"TAMP setting {key!r} {tamp_keys.REFUSED[key]}.")
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
            raise ValueError(f"traj_length_norm must be a number or 'inf' (got {value!r})") from exc
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
        raise ValueError(f"{key} names unknown operations {unknown}; valid: {', '.join(tamp_keys.BLEND_OPS)}")
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
    for key in tamp_keys.POSITIVE_KEYS:
        if key in cfg and cfg[key] <= 0:
            raise ValueError(f"{key} must be > 0 (got {cfg[key]})")
    for key in tamp_keys.NON_NEGATIVE_KEYS:
        if key in cfg and cfg[key] < 0:
            raise ValueError(f"{key} must be >= 0 (got {cfg[key]})")
    tdf = cfg.get("time_dilation_factor_literal")
    if tdf is not None and not 0.0 < tdf <= 1.0:
        raise ValueError(f"time_dilation_factor_literal must be in (0, 1] (got {tdf})")
