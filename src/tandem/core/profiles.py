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
import logging
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
from ruamel.yaml import YAML

from tandem.core import settings as settings_mod
from tandem.core.errors import ProfileError

# Same rule the source used for DC_WORKSPACE: a safe single path segment (no traversal) that
# is also a valid HuggingFace repo-name fragment.
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

STATUSES = ("eval", "success", "failure")

#: What each profile.yml is written as. 2: a planner's own settings are under planner.options.
LAYOUT_VERSION = 2

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
    # model, which is what makes a second attempt worth making. Also the most re-plans one trial
    # gets under `on_robot_phase_failure: replan`.
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
            raise ValueError("must be the name of a human executor (lowercase), such as teleop")
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

    `options` is the named planner's own settings block: TiPToP's robot, perception and TAMP
    overrides, a toy planner's list of items. Its keys are the planner's to define and to check, so
    the planner checks them -- here, when the profile loads, through its ``validate_options`` -- and
    what is stored is what the planner returned: normalised, defaults filled in. A mistake is then
    found when the profile is edited, not when a session starts with the arm about to move. A planner
    that cannot be loaded at all cannot check them; its options are kept as written, and the session
    that tries to build it says why.
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
            raise ValueError(
                f"must be one of {', '.join(known)}{suffix}; `{registry.LIST_COMMAND}` shows every planner"
            )
        return v

    @model_validator(mode="after")
    def _options_the_planner_accepts(self):
        from tandem.core.errors import TandemError
        from tandem.planners import registry

        try:
            factory = registry.factory(self.backend)
        except TandemError:
            # Installed but broken: nothing can check these, and refusing the profile over it would
            # stop it being opened to fix -- the same reason the name above is checked by name only.
            return self
        try:
            self.options = registry.options_for(factory, self.options)
        except TandemError as exc:
            raise ValueError(_options_problem(exc.message, exc.hint)) from None
        except ValueError as exc:  # a pydantic ValidationError is one
            raise ValueError(_options_problem(exc)) from None
        return self


def _options_problem(exc: Any, hint: str | None = None) -> str:
    """One planner's complaint about its options, located under ``options.`` for the reader.

    A planner that validates with pydantic raises a ValidationError whose locations are relative to
    its own block (``tamp``, ``robot.q_home``); prefixed, each reads as the path in the profile.
    """
    errors = getattr(exc, "errors", None)
    if callable(errors):
        lines = []
        for err in errors():
            loc = ".".join(str(part) for part in err.get("loc", ()))
            msg = str(err.get("msg", "")).removeprefix("Value error, ")
            lines.append(f"options.{loc}: {msg}" if loc else f"options: {msg}")
        return "\n    ".join(lines) or str(exc)
    text = str(exc)
    return f"options: {text} {hint}".rstrip() if hint else f"options: {text}"


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

    version: int = LAYOUT_VERSION
    name: str = "default"
    description: str = ""
    task: TaskSpec = Field(default_factory=TaskSpec)
    # The rig's cameras, by role. tandem's, not the planner's: the teleop executor records from
    # them, and their roles are the dataset's camera layout. A planner reads them from here too.
    cameras: CamerasSpec = Field(default_factory=CamerasSpec)
    # Phase planning: tandem's own method, so its settings are tandem's whichever planner runs.
    hitl: HitlSpec = Field(default_factory=HitlSpec)
    # Which planner, and everything that is only that planner's business (planner.options).
    planner: PlannerSpec = Field(default_factory=PlannerSpec)
    recording: RecordingSpec = Field(default_factory=RecordingSpec)
    export: ExportSpec = Field(default_factory=ExportSpec)

    @model_validator(mode="before")
    @classmethod
    def _current_layout(cls, data: Any, info: ValidationInfo) -> Any:
        if not isinstance(data, dict):
            return data
        data, moved = migrate(data)
        if moved:
            context = info.context if isinstance(info.context, dict) else {}
            _notice_migrated(str(data.get("name") or "?"), moved, context.get("source"))
        return data

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        if not NAME_RE.match(v):
            raise ValueError(
                f"profile name {v!r} is invalid; use lowercase letters, digits, '-' and '_' "
                f"(must match {NAME_RE.pattern})"
            )
        return v

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


# --------------------------------------------------------------------------- the older layout
#
# Until profile version 2 a planner's settings sat at the top level of every profile -- `robot:`,
# `perception:` and `tamp:`, beside `task:` and `cameras:`. They were TiPToP's, the only planner there
# was, and they are TiPToP's `planner.options` now. A profile in the older layout still loads: it is
# migrated as it is read, saying so once, and written in the current layout the next time it is saved.

#: The top-level sections of the older layout that belong to a planner, and the planner they belong to.
LEGACY_PLANNER_SECTIONS = ("robot", "perception", "tamp")
LEGACY_PLANNER = "tiptop"

_log = logging.getLogger(__name__)
# Profiles already said to be in the older layout, so a command that loads one ten times says it once.
_noticed: set[str] = set()


def migrate(data: dict) -> tuple[dict, list[str]]:
    """``data`` in the current layout, and which of its top-level sections had to move to get there.

    A planner's sections go under ``planner.options`` when the profile plans with the planner they
    belong to. When it plans with another, nothing ever read them -- a planner is built from its own
    options alone -- so they are dropped, and said to be: the one change that loses a setting is the
    one that must not be quiet. A section set in both places is refused rather than one of them
    silently winning.
    """
    moved = [key for key in LEGACY_PLANNER_SECTIONS if key in data]
    if not moved:
        if int(data.get("version") or 0) < LAYOUT_VERSION:
            data = {**data, "version": LAYOUT_VERSION}
        return data, []
    data = dict(data)
    sections = {key: data.pop(key) for key in moved}
    planner = dict(data.get("planner") or {})
    backend = planner.get("backend") or LEGACY_PLANNER
    if backend == LEGACY_PLANNER:
        options = dict(planner.get("options") or {})
        both = [key for key in moved if key in options]
        if both:
            raise ValueError(
                f"{', '.join(both)} is set both at the top level (the layout before profile version "
                f"{LAYOUT_VERSION}) and under planner.options; keep only the planner.options one"
            )
        planner["options"] = {**sections, **options}
        planner["backend"] = backend
        data["planner"] = planner
        moved = [f"{key} under planner.options" for key in moved]
    else:
        moved = [f"{key} dropped ({LEGACY_PLANNER}'s setting; this profile plans with {backend})" for key in moved]
    data["version"] = LAYOUT_VERSION
    return data, moved


def _notice_migrated(name: str, moved: list[str], source: str | None) -> None:
    key = source or name
    if key in _noticed:
        return
    _noticed.add(key)
    where = source or f"profile {name!r}"
    _log.warning(
        "%s is in the layout before profile version %s, and was read with %s; "
        "`tandem profile migrate %s` rewrites it",
        where,
        LAYOUT_VERSION,
        ", ".join(moved),
        name,
    )


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
        return Profile.model_validate(data, context={"source": str(path)})
    except Exception as exc:
        raise ProfileError(f"{path} is not a valid profile:\n{format_errors(exc)}") from exc


def save(profile: Profile) -> Path:
    """Write a profile, creating its directory tree. Round-trips through validation first.

    What is written is the validated copy, so a planner's options reach the file as the planner
    normalised them, and a profile read in the older layout is written in the current one.
    """
    profile = Profile.model_validate(profile.model_dump())
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


def resolve_path(profile: Profile, value: str) -> Path:
    """A path a profile names, as the profile's author meant it: absolute as written, else beside the profile.

    One reading for every path a profile holds, so a relative path means "next to profile.yml" and not
    "wherever the command happened to be started from".
    """
    candidate = Path(os.path.expanduser(str(value)))
    if candidate.is_absolute():
        return candidate
    return (profile.dir() / candidate).resolve()


def resolve_cache_path(profile: Profile) -> str | None:
    """The proposal cache's absolute path, or None when the profile sets none.

    A relative cache path means "beside the profile" (``resolve_path``). One reading, because three
    different commands read this key: a collection session, `tandem plan --profile`, and `tandem
    doctor`. Resolving it differently in any of them means they open DIFFERENT SQLite files, so the
    cache never hits across them -- and since opening one creates its parent directories, the odd one
    out silently litters a second cache wherever it was run from.
    """
    if not profile.hitl.cache_path:
        return None
    return str(resolve_path(profile, str(profile.hitl.cache_path)))


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

    Extrinsics are keyed by serial, and a planner that localises from them (TiPToP raises at warm-up
    for a serial it cannot find) is better told before a session starts.
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
