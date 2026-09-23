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

# OmegaConf env interpolations: "${oc.env:VAR}" or "${oc.env:VAR,default}".
#
# Not anchored, because they are not always the whole value — the upstream M2T2 URL embeds one
# mid-string ("http://localhost:${oc.env:TIPTOP_M2T2_PORT,8123}").
_OC_ENV = re.compile(r"\$\{oc\.env:\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:,\s*([^}]*))?\}")


def resolve_interpolation(value: Any) -> Any:
    """Resolve OmegaConf env interpolations so a profile holds concrete values.

    A profile is meant to be read and edited by a person; a ``${...}`` left in one is a string
    that looks like configuration and behaves like a crash. Resolution follows OmegaConf's own
    rule — the environment variable when it is set, otherwise the literal default.

    An interpolation with no default and no variable set has nothing to resolve to, and is left
    intact so validation can point at it by name.
    """
    if not isinstance(value, str):
        return value

    def replace(match: re.Match) -> str:
        import os

        name, default = match.group(1), match.group(2)
        env = os.environ.get(name)
        if env is not None and env.strip():
            return env.strip()
        if default is not None:
            return default.strip()
        return match.group(0)

    return _OC_ENV.sub(replace, value.strip())


def _resolve_all(obj):
    """Every string in a nested structure, dereferenced.

    Applied when a profile is READ, not only when one is imported: profiles written before
    the importer learned about embedded interpolations still have them on disk, and a stored
    `${...}` should not stop the tool from opening the file that contains it.
    """
    if isinstance(obj, dict):
        return {k: _resolve_all(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_all(v) for v in obj]
    return resolve_interpolation(obj)

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


class HitlSpec(BaseModel):
    """Phase planning — the deep human-in-the-loop mode.

    With this on, a VLM breaks the instruction into an ORDERED list of phases, each one either
    a sub-goal for the planner or something only a person can do. It invents the predicate it
    needs to describe the human's part, hands that phase over with written instructions, and
    checks from a photo that it happened.

    That ordering runs both ways, which is why phases rather than one final-state goal: "put
    the toy on the cloth, then fold it" needs the human last, "open the box, then put the toy
    in" needs the robot last, and some tasks need an intermediate state no final-state goal
    can express.

    Off by default. Disabled, the package is never imported and a session behaves exactly as
    it always has — the operator can still press "hand to human" whenever they like.

    Read by ``tandem.planning``, which is tandem's own code -- so this model is the definition
    of these settings rather than a mirror of one inside a planner, and ``extra: forbid`` below is
    the only validation layer there is.
    """

    model_config = {"extra": "forbid"}

    enabled: bool = False
    # Splitting a task into phases and inventing a predicate is reasoning, not localisation, so
    # it does NOT reuse the detection model — that one runs with thinking disabled.
    proposal_model: str = "gemini-2.5-pro"
    # Grounding ("is the cloth folded?") is one visual judgement over one image, and it is the
    # query a run pays for repeatedly. Flash is enough.
    vlm_model: str = "gemini-2.5-flash"
    # Reprompts allowed when a proposal comes back unparseable. The error is fed back to the
    # model, which is what makes a second attempt worth making.
    max_attempts: int = 3
    # Classify the invented predicates on the first image, before anything runs. Off by
    # default: a human is asked precisely because the predicate is false. Worth turning on for
    # a scene that may start already solved.
    classify_initial: bool = False
    # Extra chances at a human phase the VLM says did not happen.
    verify_retries: int = 1
    # Treat a failed verification as a rollout failure. False records the verdict and carries
    # on, which is what you want while calibrating the classifier prompts.
    verify_enforced: bool = True
    # What becomes of a trial whose human phase still fails its check once the retries are spent.
    # `exclude` is the paper's rule: filed under failure/ with `excluded: true`, verdicts and raw
    # legs kept, and no label prompt -- a check the operator can overrule by answering "success" is
    # not a filter on the dataset. `label` asks the operator anyway, for calibrating the classifier.
    on_verification_failure: str = "exclude"
    # Check the last phase too when it is a human one. False leaves it to the operator's label, as
    # the reference implementation did; the paper checks every human phase.
    verify_final_phase: bool = True
    # Which halves of each operator's contract a camera is asked about. Each costs a model call per
    # atom with the arm parked, so only the one nothing else can answer is on: a human phase's
    # effects are the only evidence it happened. Its preconditions are already proved symbolically
    # by check_plan_effects (and the paper ran with them off); a robot leg's are a guard against a
    # stale belief; a robot leg's effects are something the arm reports better than a camera sees.
    check_human_effects: bool = True
    check_human_preconditions: bool = False
    check_tamp_preconditions: bool = False
    check_tamp_effects: bool = False
    # Stop at an unmet precondition. Off: one classifier call must not cost a demonstration, so the
    # verdict is recorded and the phase goes ahead.
    precondition_enforced: bool = False
    # Check the declared operators against each other before the arm moves, and send a plan that
    # deletes what a later phase needs back to the model to repair. Symbolic, so it costs nothing.
    check_plan_effects: bool = True
    # Write every image sent to the VLM and a rendered PNG of the reply into vlm/ beside the
    # rollout. When a run goes wrong the question is always "what did the model see, and what
    # did it say", and that is unanswerable afterwards without this.
    save_vlm_io: bool = True
    # SQLite cache for PROPOSAL responses only. Worth setting while iterating on prompts;
    # never applied to grounding or verification.
    cache_path: str | None = None
    # What happens when the planner cannot plan a robot phase. `abort` ends the trial as a failure;
    # `teleop` describes the sub-goal to the operator and lets them do it by hand, checked exactly as
    # any other human phase; `replan` feeds the failure back to the proposer. `abort` is the default
    # because the paper counts a TAMP failure as a trial failure: a teleop fallback turns a robot
    # phase into a human one, which inflates the human effort a dataset cost and credits the method
    # with trials it did not complete. A leg that was planned but failed to EXECUTE always ends the
    # trial; there is no setting for that.
    on_robot_phase_failure: str = "abort"
    # Plan consecutive robot phases as one goal where that is sound: one continuous motion, and no
    # re-perception in the middle for labels to drift across. False re-perceives before every one.
    conjoin_robot_phases: bool = True
    # Who carries out a human phase, by registered name (tandem.executors). "teleop" -- a person driving
    # the arm -- is the only one that ships; a package adds another through the
    # `tandem.human_executors` entry point. The name must be one the registry knows on this machine,
    # like planner.backend: a misspelling found at load time costs nothing, and found at the first
    # human phase it costs the trial.
    human_executor: str = "teleop"
    # Accept "done" for a human phase that was never teleoperated, while recording (with recording
    # off it is always accepted). The episode then lacks the one demonstration the trial exists to
    # capture, while looking complete.
    allow_unrecorded_human_phase: bool = False
    # Which camera the verification frame comes from. Third-person by default: after a hand-off the
    # arm is wherever the operator left it, so a wrist view points nowhere useful.
    verification_camera: str = "external"

    @field_validator("verify_retries")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be >= 0")
        return v

    @field_validator("max_attempts")
    @classmethod
    def _at_least_one_attempt(cls, v: int) -> int:
        # Not merely non-negative. Zero attempts means never asking, which is not a configuration of
        # the feature but a way of turning it off, and it used to pass validation here and then raise
        # out of the planner package with the raw traceback the CLI exists to suppress.
        if v < 1:
            raise ValueError("must be >= 1 (0 would mean never asking the model at all)")
        return v

    @field_validator("on_robot_phase_failure")
    @classmethod
    def _known_failure_policy(cls, v: str) -> str:
        from tandem.planning.config import ON_FAILURE_CHOICES

        if v not in ON_FAILURE_CHOICES:
            raise ValueError(f"must be one of {', '.join(ON_FAILURE_CHOICES)}")
        return v

    @field_validator("on_verification_failure")
    @classmethod
    def _known_verification_policy(cls, v: str) -> str:
        from tandem.planning.config import ON_VERIFICATION_FAILURE_CHOICES

        if v not in ON_VERIFICATION_FAILURE_CHOICES:
            raise ValueError(f"must be one of {', '.join(ON_VERIFICATION_FAILURE_CHOICES)}")
        return v

    @field_validator("human_executor")
    @classmethod
    def _executor_name(cls, v: str) -> str:
        # The shape first, here rather than left to PlanningConfig, whose ValueError would escape as a
        # raw traceback -- the same trap max_attempts fell into. Then the registry, by name only: it
        # reads installed packages' metadata and imports none of them. The price is the one
        # planner.backend already pays: a profile naming an executor this machine has not installed
        # does not load here until the package providing it is installed.
        from tandem.core.errors import TandemError
        from tandem.executors import base as executors
        from tandem.planning.config import HUMAN_EXECUTOR_NAME

        if not HUMAN_EXECUTOR_NAME.fullmatch(v):
            raise ValueError("must be the name of a human executor, such as teleop")
        try:
            executors.check_name(v)
        except TandemError as exc:
            raise ValueError(f"{exc.message} {exc.hint}" if exc.hint else exc.message) from None
        return v

    @field_validator("verification_camera")
    @classmethod
    def _known_camera(cls, v: str) -> str:
        if v not in ("external", "hand", "perception"):
            raise ValueError("must be one of external, hand, perception")
        return v

    def to_planning_config(self, *, cache_path: str | None = None):
        """The same settings as ``tandem.planning`` takes them.

        A plain dataclass on the other side, so the phase planner can be used -- and tested -- with
        no profile, no runtime and no robot. ``cache_path`` overrides the stored one when a caller
        has already resolved it against the profile directory.
        """
        from tandem.planning.config import PlanningConfig

        data = self.model_dump(mode="python")
        data["cache_path"] = cache_path if cache_path is not None else data.get("cache_path")
        return PlanningConfig(**data)


class PlannerSpec(BaseModel):
    """Which task and motion planner tandem drives.

    tandem plans the task itself and calls a planner for the robot's phases, so which planner that
    is, is a setting. `backend` names one of ``tandem.planners.registry.available()``; an unknown
    name is an error rather than a fallback, because a session that silently planned with a
    different planner from the one asked for produces a dataset nobody can interpret afterwards.

    `options` is the named planner's own settings block, passed to its factory verbatim. Its keys are
    the planner's to define and to check, so they are not validated here -- only when the planner is
    built for a session, by the planner, which refuses a key it does not read.
    """

    model_config = {"extra": "forbid"}

    backend: str = "tiptop"
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("backend")
    @classmethod
    def _known_backend(cls, v: str) -> str:
        import difflib

        from tandem.planners import registry

        # By name only: a planner installed as a package is known from its entry point without being
        # imported. One that is installed but broken therefore still validates, so the profile can be
        # loaded and edited -- and the session that tries to build it says what is wrong with it.
        known = registry.available()
        if v not in known:
            close = difflib.get_close_matches(v, known, n=1, cutoff=0.6)
            suffix = f" (did you mean {close[0]!r}?)" if close else ""
            raise ValueError(f"must be one of {', '.join(known)}{suffix}")
        return v


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
    # Deliberately NOT part of `tamp`: that dict is a solver-cost funnel read by a hand-written
    # if-ladder, and this changes what a dataset CONTAINS rather than how the arm moves.
    hitl: HitlSpec = Field(default_factory=HitlSpec)
    planner: PlannerSpec = Field(default_factory=PlannerSpec)
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
    data = _resolve_all(_plain(data))
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
