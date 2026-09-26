"""TiPToP's settings: the arm it drives and how it perceives (this machine's), and its solver overrides (a task's).

    rig.yml                                 this machine's, every profile shares them (RIG_OPTIONS)
      robot: {type, host}                   tandem's own: the arm, and the NUC it is reached through
      planners:
        tiptop:
          robot:       the shim's ports, the speed, the joint count, the home and capture poses
          perception:  the M2T2 grasp server, the FoundationStereo depth server, SAM-2, the depth pipeline

    <profile>.yml                           the task's (OPTIONS)
      planner:
        backend: tiptop
        options:
          tamp:        cuTAMP / cuRobo overrides, by tiptop's own key names (tamp_keys.py)

All three were one ``planner.options`` block of every profile, which put a robot's address and a grasp
server's URL into every task and left the other profiles stale whenever one was edited. The robot and
perception settings describe the machine, so they are the rig's; the TAMP overrides are what a task was
collected with, so they are the profile's. A task that needs its own perception numbers still has
them: the tamp PERCEPTION_KEYS (voxel size, contact threshold, grasp threshold, M2T2 passes) override
the rig's for that task.

``resolve`` puts the two halves back together, with the rig's robot type and host, into the one
``TiptopOptions`` everything that renders tiptop's config reads. Each half is validated where it is
stored -- the task's when a profile naming TiPToP loads (``FACTORY.validate_options``), the machine's
when rig.yml is read (``validate_rig_options``) -- with the same loud errors these sections always had.
"""

from __future__ import annotations

import difflib
import math
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from tandem.planners.tiptop import tamp_keys
from tandem.planners.tiptop.arms import ROBOT_TYPES

# --------------------------------------------------------------------------- the arm

# ROBOT_TYPES, the arms both halves of the pinned planner know, is arms.py's. The catalog lists the same
# table (arms.requirement), so the arms it names are exactly the ones this schema accepts.

# Names people reach for that are not the planner's, and what they mean by them.
_ROBOT_ALIASES = {
    "franka": "panda (a Panda with the Franka Hand) or fr3_robotiq (an FR3 with a Robotiq 2F-85)",
    "fr3": "fr3_robotiq (an FR3 with a Robotiq 2F-85); an FR3 with the Franka Hand is not supported",
    "fr3_franka": "fr3_robotiq (an FR3 with a Robotiq 2F-85); an FR3 with the Franka Hand is not supported",
    "ur5e": "ur5",
}


class _RobotSettings(BaseModel):
    """The arm's settings that are TiPToP's own: how its shim is reached, how fast it moves, where it parks."""

    model_config = {"extra": "forbid"}

    dof: int = 7
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


class RobotSpec(_RobotSettings):
    """rig.yml's ``planners.tiptop.robot``. The arm's type and address are not here: they are the rig's own
    ``robot.type`` and ``robot.host``, which every planner reads."""

    @model_validator(mode="before")
    @classmethod
    def _address_is_the_rigs(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            for key in ("host", "type"):
                if key in data:
                    raise ValueError(
                        f"robot.{key} is the rig's own robot.{key} (at the top of rig.yml), shared by every "
                        f"planner: `tandem rig set robot.{key} VALUE`"
                    )
        return data


def check_robot_type(v: str) -> str:
    """``v`` if TiPToP drives that arm; a ValueError naming the arms it does, and the one meant, if not."""
    if v not in ROBOT_TYPES:
        known = " | ".join(sorted(ROBOT_TYPES))
        meant = _ROBOT_ALIASES.get(v) or next(iter(difflib.get_close_matches(v, sorted(ROBOT_TYPES), n=1)), "")
        hint = f"; did you mean {meant}?" if meant else ""
        raise ValueError(f"unsupported robot type {v!r} ({known}){hint}")
    return v


class ResolvedRobot(_RobotSettings):
    """The arm as tiptop's config states it: TiPToP's robot settings, with the rig's type and address."""

    type: str = "fr3_robotiq"
    host: str = "172.16.0.2"


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


def _usable_url(v: str, example: str) -> str:
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
        raise ValueError(f"{v!r} needs a scheme and a host, e.g. {example}")
    return v


class M2T2Spec(BaseModel):
    model_config = {"extra": "forbid"}

    url: str = "http://localhost:8123"
    apply_bounds: bool = True

    @field_validator("url")
    @classmethod
    def _parseable(cls, v: str) -> str:
        return _usable_url(v, "http://localhost:8123")


class FoundationStereoSpec(BaseModel):
    """The FoundationStereo server tiptop sends every ZED stereo pair to, for its depth."""

    model_config = {"extra": "forbid"}

    url: str = "http://localhost:1234"

    @field_validator("url")
    @classmethod
    def _parseable(cls, v: str) -> str:
        return _usable_url(v, "http://localhost:1234")


class PerceptionSpec(BaseModel):
    model_config = {"extra": "forbid"}

    gemini: GeminiSpec = Field(default_factory=GeminiSpec)
    m2t2: M2T2Spec = Field(default_factory=M2T2Spec)
    # Used: tiptop estimates a ZED's depth by sending its stereo pair to FoundationStereo (perception/
    # cameras get_depth_estimator -> zed_infer_depth_async), every rollout.
    foundation_stereo: FoundationStereoSpec = Field(default_factory=FoundationStereoSpec)
    # Where SAM-2 runs: in the runtime ("local"), or on a SAM-2 server (tiptop's scripts/sam_server.py)
    # at sam_url. tiptop reads perception.sam.url for any mode that is not "local" -- a typo'd "Local"
    # included -- so the mode is one of the two, and a remote one has to say where.
    sam_mode: Literal["local", "remote"] = "local"
    sam_url: str | None = None
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

    @field_validator("sam_url")
    @classmethod
    def _sam_url_parseable(cls, v: str | None) -> str | None:
        return None if v is None else _usable_url(v, "http://localhost:8000")

    @model_validator(mode="after")
    def _remote_sam_says_where(self):
        # Otherwise the rendered tiptop.yml has no perception.sam.url, and the first thing to read it is
        # the warm-up's sam2_client(), with an OmegaConf missing-key error that names no setting of ours.
        if self.sam_mode == "remote" and not self.sam_url:
            raise ValueError(
                "sam_mode is 'remote' but perception.sam_url is not set; name the SAM-2 server, "
                "e.g. sam_url: http://localhost:8000"
            )
        return self


# --------------------------------------------------------------------------- the two halves, and the whole


class TiptopRigOptions(BaseModel):
    """rig.yml's ``planners.tiptop``: what TiPToP needs of this machine."""

    model_config = {"extra": "forbid"}

    robot: RobotSpec = Field(default_factory=RobotSpec)
    perception: PerceptionSpec = Field(default_factory=PerceptionSpec)

    @model_validator(mode="before")
    @classmethod
    def _not_the_tasks(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and "tamp" in data:
            raise ValueError(
                "tamp is a task setting, a profile's planner.options.tamp, not this machine's: "
                "`tandem profile edit NAME`"
            )
        return data

    def to_options(self) -> dict[str, Any]:
        """As rig.yml stores it: plain containers, unset optionals left out so the file stays readable."""
        return self.model_dump(mode="python", exclude_none=True)


class TiptopTaskOptions(BaseModel):
    """A profile's ``planner.options`` for TiPToP: what a task is collected with."""

    model_config = {"extra": "forbid"}

    # Flat, using tiptop's own key names -- see tamp_keys.py for why. Deliberately NOT where phase
    # planning is configured: that is the profile's `hitl:` block, tandem's own, because it changes
    # what a dataset CONTAINS rather than how the arm moves.
    tamp: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _not_the_machines(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            for key in ("robot", "perception"):
                if key in data:
                    raise ValueError(
                        f"{key} is a machine setting, rig.yml's planners.tiptop.{key}, which every profile "
                        f"shares: `tandem rig set planners.tiptop.{key}.KEY VALUE`"
                    )
        return data

    @field_validator("tamp", mode="before")
    @classmethod
    def _valid_tamp(cls, v: Any) -> dict:
        return validate_tamp(v)

    def to_options(self) -> dict[str, Any]:
        """As a profile stores it: plain containers, unset optionals left out so the file stays readable."""
        return self.model_dump(mode="python", exclude_none=True)


class TiptopOptions(BaseModel):
    """Everything TiPToP is configured with, both halves together: what tiptop.yml and the overrides are
    rendered from. Built by ``resolve``, never stored."""

    model_config = {"extra": "forbid"}

    robot: ResolvedRobot = Field(default_factory=ResolvedRobot)
    perception: PerceptionSpec = Field(default_factory=PerceptionSpec)
    tamp: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tamp", mode="before")
    @classmethod
    def _valid_tamp(cls, v: Any) -> dict:
        return validate_tamp(v)


def resolve(
    rig: Any,
    rig_options: Mapping[str, Any] | None = None,
    task_options: Mapping[str, Any] | None = None,
    *,
    check_type: bool = True,
) -> TiptopOptions:
    """TiPToP's whole configuration: the rig's arm and address, its ``planners.tiptop`` block, a task's tamp.

    A pydantic ``ValidationError`` names a bad setting in either block. An arm TiPToP does not drive is a
    ``TandemError`` naming rig.yml's ``robot.type``, which is where it is fixed; ``check_type=False`` lets
    doctor say so in a row of its own and still check everything else.
    """
    from tandem.core.errors import TandemError

    machine = TiptopRigOptions.model_validate(dict(rig_options or {}))
    task = TiptopTaskOptions.model_validate(dict(task_options or {}))
    robot_type = rig.robot.type
    if check_type:
        try:
            check_robot_type(robot_type)
        except ValueError as exc:
            raise TandemError(
                f"rig.yml's robot.type: {exc}",
                hint="`tandem rig set robot.type fr3_robotiq` (or another arm TiPToP drives).",
            ) from None
    return TiptopOptions.model_validate(
        {
            "robot": {**machine.robot.model_dump(mode="python"), "type": robot_type, "host": rig.robot.host},
            "perception": machine.perception.model_dump(mode="python"),
            "tamp": task.tamp,
        }
    )


def resolve_profile(profile: Any, rig: Any = None, *, check_type: bool = True) -> TiptopOptions:
    """TiPToP's configuration for a profile that plans with it, on this machine's rig (loaded when not given)."""
    from tandem.core import rig as rig_mod

    rig = rig if rig is not None else rig_mod.load()
    return resolve(
        rig,
        rig_mod.planner_options(rig, "tiptop"),
        getattr(getattr(profile, "planner", None), "options", None),
        check_type=check_type,
    )


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
    for key, value in _upgrade_renamed(raw).items():
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
            out[key] = _validate_retime_ops(key, value)
        elif key in tamp_keys.INDEX_MAP_KEYS:
            out[key] = _validate_index_map(key, value)
        else:
            out[key] = _coerce_scalar(key, value)

    _check_enums(out)
    _check_positives(out)
    if out.get("retime_trajectory") and not out.get("encoder_path"):
        # tiptop's own check (override_keys.check_override_keys), made when the profile loads rather than
        # when a session starts with the arm about to move.
        raise ValueError("retime_trajectory needs encoder_path, the trajectory-encoder checkpoint that times each stroke")
    return out


def _upgrade_renamed(raw: dict) -> dict:
    """A ``tamp:`` block written before tiptop's re-timing rename, read under the new names.

    A renamed key (``tamp_keys.RENAMED``) is the same setting, so it is read as its new name. A key of a
    removed mode (``tamp_keys.REMOVED``) is dropped when its value describes what tiptop still does
    (``vae_retiming: false``, ``blend_mode: vae``) and refused otherwise, because that behavior is gone.
    Setting a key under both names with different values is refused rather than guessed.
    """
    out: dict = {}
    for key, value in raw.items():
        if key in tamp_keys.REMOVED:
            kept = tamp_keys.REMOVED[key]
            if value is None or (kept is not None and value == kept):
                continue
            raise ValueError(f"TAMP setting {key!r}: {tamp_keys.REMOVED_BECAUSE}. Remove it from the profile.")
        new = tamp_keys.RENAMED.get(key, key)
        if new in out and out[new] != value:
            raise ValueError(
                f"TAMP setting {key!r} is the old name of {new!r}, and the profile sets both, differently. "
                f"Keep {new!r} only."
            )
        out[new] = value
    return out


def _normalise_traj_norm(value: Any) -> str | float:
    """Force the infinity norm to the STRING "inf", and refuse a finite norm below 1.

    YAML `inf` parses as a string but `.inf` parses as float infinity, and the overrides dict
    round-trips through JSON, which cannot represent Infinity. Both spellings land here as
    the one form tiptop's resolve_traj_length_norm accepts.

    Below 1 is cuTAMP's own refusal (validate_tamp_config, run at every plan), in its own words:
    accepted here, it would load, warm and perceive, and then fail every robot leg's plan.
    """
    if isinstance(value, str):
        if value.strip().lower() in tamp_keys.INFINITY_ALIASES:
            return "inf"
        try:
            norm = float(value)
        except ValueError as exc:
            raise ValueError(f"traj_length_norm must be a number or 'inf' (got {value!r})") from exc
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        norm = float(value)
        if math.isinf(norm) and norm > 0:  # -.inf is refused below, not read as the infinity norm
            return "inf"
    else:
        raise ValueError(f"traj_length_norm must be a number or 'inf' (got {value!r})")
    # `not >=` rather than `<`, so NaN -- which float() accepts and every comparison fails -- is refused.
    if not norm >= 1:
        raise ValueError(f"traj_length_norm must be >= 1 (or inf), not {norm}")
    return norm


def _validate_retime_ops(key: str, value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key} must be a list of operation names")
    ops = [str(x).strip() for x in value]
    unknown = [o for o in ops if o not in tamp_keys.RETIME_OPS]
    if unknown:
        # The source shipped a config with `MoveFree. MoveHolding` -- a typo'd '.' that YAML
        # folded into one token and the planner silently ignored. Catch that class here.
        raise ValueError(f"{key} names unknown operations {unknown}; valid: {', '.join(tamp_keys.RETIME_OPS)}")
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
        if key in cfg:
            if str(cfg[key]).lower() not in allowed:
                raise ValueError(f"{key} must be one of {sorted(allowed)} (got {cfg[key]!r})")
            # tiptop compares it as written (override_keys: retime_mode != "encoder"), so pass it lowercase.
            cfg[key] = str(cfg[key]).lower()


def _check_positives(cfg: dict) -> None:
    for key in tamp_keys.POSITIVE_KEYS:
        if key in cfg and cfg[key] <= 0:
            raise ValueError(f"{key} must be > 0 (got {cfg[key]})")
    for key in tamp_keys.NON_NEGATIVE_KEYS:
        if key in cfg and cfg[key] < 0:
            raise ValueError(f"{key} must be >= 0 (got {cfg[key]})")
    for key in tamp_keys.UNIT_INTERVAL_KEYS:
        # `not (0 <= v <= 1)`, so NaN is refused too. cuTAMP refuses the same range, but only once the
        # first plan builds its config -- with the arm already at the capture pose.
        if key in cfg and not 0.0 <= cfg[key] <= 1.0:
            raise ValueError(f"{key} must be in [0, 1] (got {cfg[key]})")
    # Both replace robot.time_dilation_factor -- the speed the robot block guards to (0, 1] -- for
    # every plan (resolve_time_dilation_factor), and neither is checked again before the arm moves:
    # cuTAMP's validate_tamp_config does not look, and cuRobo refuses > 1 only at the first plan. 1.0
    # stays legal: for time_dilation_factor it is tiptop's "no extra scaling" sentinel, which falls
    # back to robot.time_dilation_factor.
    for key in ("time_dilation_factor", "time_dilation_factor_literal"):
        tdf = cfg.get(key)
        # `not (0 < tdf <= 1)`, so NaN is refused too.
        if tdf is not None and not 0.0 < tdf <= 1.0:
            raise ValueError(
                f"{key} must be in (0, 1] (got {tdf}); it replaces robot.time_dilation_factor, so start at "
                "0.2 (20% speed)"
            )
