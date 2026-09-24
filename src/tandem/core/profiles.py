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

import io
import json
import logging
import os
import re
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr, ValidationInfo, field_validator, model_validator
from ruamel.yaml import YAML

from tandem.core import settings as settings_mod
from tandem.core.errors import ProfileError, ProfileInvalid, TandemError

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

    Applied whenever a profile is READ: profiles written from the source monorepo's configs still have
    them on disk, and a stored `${...}` should not stop the tool from opening the file that contains it.
    """
    if isinstance(obj, dict):
        return {k: _resolve_all(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_all(v) for v in obj]
    return resolve_interpolation(obj)


def _new_yaml() -> YAML:
    """A fresh round-trip YAML instance.

    One per dump, never a shared one: ruamel keeps the half-written state of a dump that raised (a
    RepresenterError over a value it cannot write), and the next dump through the same instance then
    writes nothing at all -- which is how one bad save could leave every later save empty.
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 100
    return yaml


# --------------------------------------------------------------------------- names on this machine
#
# A profile names a planner (planner.backend) and a human executor (hitl.human_executor), and both are
# checked against what this machine has installed when the profile loads: a misspelling found there
# costs nothing, found at the first human phase it costs a trial. But a profile is also a folder of
# trajectories that is browsed, exported, repaired and edited on machines that do not have the plugin
# that collected it -- a laptop, a cluster, a workstation after an uninstall. So a validation context
# can say which names are allowed to be absent here:
#
#   ABSENT_OK: True        every well-formed name (read-only use: listing, browsing, export)
#   ABSENT_OK: {names}     these names only: the ones a profile ALREADY had on disk, when it is rewritten
#                          for some other reason. A name being newly written is always checked.
#
# A name that is not well-formed is refused either way; so is a planner that is installed but whose
# options it refuses.
ABSENT_OK = "absent_ok"


def _absent_ok(info: ValidationInfo, name: str) -> bool:
    context = info.context if isinstance(info.context, dict) else {}
    allowed = context.get(ABSENT_OK, False)
    if allowed is True:
        return True
    return isinstance(allowed, (set, frozenset, tuple, list)) and name in allowed


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
    # Each executor's own settings, keyed by the executor's name: a policy executor's checkpoint, a
    # policy server's address. Keyed by name rather than one block for whichever executor is chosen, so
    # `tandem executors use` switching to another executor and back never throws a configured one's
    # settings away. An installed executor with a ``validate_options`` hook checks its own block when
    # the profile loads (as a planner checks planner.options); one this machine does not have keeps its
    # block as written. The executor receives its block as ``ExecutorContext.options``.
    human_executor_options: dict[str, dict[str, Any]] = Field(default_factory=dict)
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
    def _executor_name(cls, v: str, info: ValidationInfo) -> str:
        # The shape first, here rather than left to PlanningConfig, whose ValueError would escape as a
        # raw traceback -- the same trap max_attempts fell into. Then the registry, by name only: it
        # reads installed packages' metadata and imports none of them. The price is the one
        # planner.backend already pays: a profile naming an executor this machine has not installed
        # does not load for collection here until the package providing it is installed. Reading it,
        # and rewriting it for another reason, are allowed (ABSENT_OK above).
        from tandem.executors import base as executors
        from tandem.planning.config import HUMAN_EXECUTOR_NAME

        if not HUMAN_EXECUTOR_NAME.fullmatch(v):
            raise ValueError("must be the name of a human executor (lowercase), such as teleop")
        try:
            executors.check_name(v)
        except TandemError as exc:
            if _absent_ok(info, v):
                return v
            raise ValueError(f"{exc.message} {exc.hint}" if exc.hint else exc.message) from None
        return v

    @field_validator("human_executor_options")
    @classmethod
    def _executor_options(cls, v: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        from tandem.executors import base as executors
        from tandem.planning.config import HUMAN_EXECUTOR_NAME

        checked: dict[str, dict[str, Any]] = {}
        for name, options in v.items():
            if not HUMAN_EXECUTOR_NAME.fullmatch(name):
                raise ValueError(f"{name!r} is not the name of a human executor (lowercase), such as teleop")
            try:
                checked[name] = executors.options_for(name, options)
            except TandemError as exc:
                raise ValueError(f"{name}: {exc.message} {exc.hint or ''}".rstrip()) from None
            except ValueError as exc:  # a pydantic ValidationError is one
                raise ValueError(f"{name}: {_options_problem(exc)}") from None
        return checked

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
    def _known_backend(cls, v: str, info: ValidationInfo) -> str:
        import difflib

        from tandem.core import names
        from tandem.planners import registry

        # By name only: a planner installed as a package is known from its entry point without being
        # imported. One that is installed but broken therefore still validates, so the profile can be
        # loaded and edited -- and the session that tries to build it says what is wrong with it.
        known = registry.available()
        if v not in known:
            if names.is_valid(v) and _absent_ok(info, v):
                # Not installed here, and this is a read (or a rewrite of what was already on disk). The
                # options are kept as written below, since nothing here can check them.
                return v
            close = difflib.get_close_matches(v, known, n=1, cutoff=0.6)
            if close:
                raise ValueError(
                    f"no planner named {v!r} (did you mean {close[0]!r}?); "
                    f"`{registry.LIST_COMMAND}` shows every planner"
                )
            raise ValueError(
                f"no planner named {v!r} is installed on this machine (it has {', '.join(known)}); install "
                f"the package that provides it, or `tandem planners use NAME` to switch this profile to "
                f"another. `{registry.LIST_COMMAND}` shows every planner"
            )
        return v

    @model_validator(mode="after")
    def _options_the_planner_accepts(self):
        from tandem.planners import registry

        try:
            factory = registry.factory(self.backend)
        except TandemError:
            # Installed but broken, or (read only) not installed at all: nothing can check these, and
            # refusing the profile over it would stop it being opened to fix -- the same reason the
            # name above is checked by name only.
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


def planner_spec(backend: str, options: Mapping[str, Any] | None = None, *, profile: str | None = None) -> PlannerSpec:
    """A ``PlannerSpec`` for ``backend`` with ``options``, or a ``ProfileError`` that says what it lacks.

    Every command that points a profile at a planner builds one of these -- `planners use`, `profile
    create --planner`, `init`, the web's create and use buttons -- usually with no options at all. A
    planner may require a setting that has no sensible default (a robot's address), and its
    ``validate_options`` then refuses the empty block; built bare, that refusal is a pydantic
    ValidationError, which the CLI shows as a traceback and the server as a 500. Here it is the planner's
    own complaint, located under planner.options, with the way to supply what it asked for.
    """
    from pydantic import ValidationError

    try:
        return PlannerSpec(backend=backend, options=dict(options or {}))
    except ValidationError as exc:
        where = f"`tandem profile edit {profile}`" if profile else "`tandem profile edit NAME`"
        raise ProfileError(
            f"The {backend} planner does not accept the planner.options it would be given:\n{format_errors(exc)}",
            hint=f"Give it what it asks for: `tandem planners use {backend} --option KEY=VALUE` (repeatable), or "
            f"{where} and set them under planner.options. `tandem planners info {backend}` lists its options.",
        ) from None


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
    # The profiles root this profile's directory is under, fixed by `pinned`; None resolves it afresh
    # on every call. Not a setting: it is never written into profile.yml.
    _root: Path | None = PrivateAttr(default=None)

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
        # fullmatch: NAME_RE ends in `$`, which also matches before a trailing newline.
        if not NAME_RE.fullmatch(v):
            raise ValueError(
                f"profile name {v!r} is invalid; use lowercase letters, digits, '-' and '_' "
                f"(must match {NAME_RE.pattern})"
            )
        return v

    # ---- paths -------------------------------------------------------------

    def dir(self) -> Path:
        return (self._root if self._root is not None else profiles_root()) / self.name

    def pinned(self) -> Profile:
        """This profile, with its directory fixed where it resolves now.

        ``dir()`` otherwise resolves the data root afresh on every call, from $TANDEM_DATA_ROOT, the
        settings and the home directory, whichever thread asks. A session's merge runs on a thread of
        its own, and can outlive the environment it was started in: one that outlived its test put a
        trajectory into the real ~/tandem-data, the test's own data root having been unset by then.
        A session works from a pinned copy, so every leg, record and merge of it lands in the one
        place it started in. A shallow copy: the settings it holds are the same objects. It compares
        unequal to an unpinned profile (pydantic compares private attributes too); compare
        ``model_dump()`` to ask whether two say the same thing.
        """
        copy = self.model_copy()
        copy._root = self.dir().parent
        return copy

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


def is_older_layout(data: Mapping[str, Any]) -> bool:
    """Whether a profile's raw settings are in a layout before the current one, and so need rewriting."""
    try:
        version = int(data.get("version") or 0)
    except (TypeError, ValueError):
        version = 0
    return version < LAYOUT_VERSION or any(key in data for key in LEGACY_PLANNER_SECTIONS)


def migrate(data: dict) -> tuple[dict, list[str]]:
    """``data`` in the current layout, and which of its top-level sections had to move to get there.

    A planner's sections go under ``planner.options`` when the profile plans with the planner they
    belong to. When it plans with another, nothing ever read them -- a planner is built from its own
    options alone -- so they are dropped, and said to be: the one change that loses a setting is the
    one that must not be quiet. A section set in both places is refused rather than one of them
    silently winning.
    """
    older = int(data.get("version") or 0) < LAYOUT_VERSION
    kept = _kept_from_version_1(data) if older else []
    moved = [key for key in LEGACY_PLANNER_SECTIONS if key in data]
    if not moved:
        if older:
            data = {**data, "version": LAYOUT_VERSION}
        return data, kept
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
    return data, moved + kept


#: What version 1 wrote into every profile for on_robot_phase_failure: its default, and the template's.
_VERSION_1_ROBOT_FAILURE = "teleop"


def _kept_from_version_1(data: dict) -> list[str]:
    """Settings a version-1 profile carries over unchanged that no longer mean what they meant.

    Version 1 defaulted ``hitl.on_robot_phase_failure`` to teleop, stated it in the template, and wrote
    every field on save -- so every version-1 profile on disk says teleop whether or not anyone chose
    it. The default is now abort, the paper's rule (a planning failure is a trial failure; a teleop
    fallback credits the method with trials it did not complete and understates the human effort).
    The value is KEPT -- a migration cannot tell a deliberate choice from the old default, and silently
    changing what a collection run does is worse -- but it is said, alongside the sections that moved.
    """
    hitl = data.get("hitl")
    if isinstance(hitl, dict) and hitl.get("on_robot_phase_failure") == _VERSION_1_ROBOT_FAILURE:
        return [
            "hitl.on_robot_phase_failure kept at teleop (version 1's default; the default is now abort, "
            "the paper's rule: set it to abort unless teleop was chosen on purpose)"
        ]
    return []


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


def is_name(name: object) -> bool:
    """Whether ``name`` can be a profile's name: one safe path segment, never ``.`` or ``..``."""
    return isinstance(name, str) and NAME_RE.fullmatch(name) is not None


def _checked(name: object) -> str:
    """``name``, if it can be a profile's name; otherwise a ProfileError.

    Every path under profiles/ is built from a name, and names arrive from URLs as well as from the
    command line -- where ``%2E%2E`` decodes to ``..`` before any route sees it. A name that is not a
    profile name never becomes a path.
    """
    if not is_name(name):
        known = list_names()
        raise ProfileError(
            f"{name!r} is not a profile name.",
            hint=f"A profile's name matches {NAME_RE.pattern}. "
            + (f"Known profiles: {', '.join(known)}." if known else "No profiles exist yet."),
        )
    return str(name)


def _not_found(name: str, path: Path) -> ProfileError:
    known = list_names()
    hint = (
        f"Known profiles: {', '.join(known)}."
        if known
        else "No profiles exist yet — run `tandem init` or `tandem profile create <name>`."
    )
    return ProfileError(f"Profile {name!r} not found at {path}.", hint=hint)


def exists(name: str) -> bool:
    return is_name(name) and (profiles_root() / name / "profile.yml").is_file()


def _existing_file(name: str) -> Path:
    path = profiles_root() / _checked(name) / "profile.yml"
    if not path.is_file():
        raise _not_found(name, path)
    return path


def load(name: str | None = None, *, require_installed: bool = True) -> Profile:
    """Load a profile by name, or the active one.

    ``require_installed=False`` is for reading only -- listing, browsing and exporting trajectories,
    showing the profile: a planner or human executor this machine has not installed is then accepted by
    name, its settings kept as written, so a profile collected elsewhere can still be looked at. Anything
    that builds the planner or runs a session loads with the default, and is refused.
    """
    cfg = settings_mod.load()
    name = name or cfg.active_profile
    return load_file(_existing_file(name), name=name, require_installed=require_installed)


def read_data(path: Path, *, name: str | None = None) -> dict:
    """``path``'s settings as a plain mapping, interpolations resolved, not validated.

    What `load_file` validates, and what a command that must repair an invalid profile edits (see
    ``switch_planner``). An empty file, or one that is not a mapping, is refused here: read as "no
    settings" it would load as a profile of pure defaults -- another planner, another robot address,
    the template's task -- without a word.
    """
    try:
        with path.open() as fh:
            data = _new_yaml().load(fh)
    except Exception as exc:
        raise ProfileInvalid(f"{path} is not valid YAML: {exc}") from exc
    if data is None:
        raise ProfileInvalid(
            f"{path} is empty.",
            hint="Restore it (from a backup beside it, if one was left), or recreate the profile with "
            "`tandem profile create NAME --force`; its trajectories are untouched.",
        )
    data = _resolve_all(_plain(data))
    if not isinstance(data, dict):
        raise ProfileInvalid(
            f"{path} must be a mapping of profile settings, not a {type(data).__name__}.",
            hint="`tandem profile edit NAME` opens it.",
        )
    if name is not None:
        data.setdefault("name", name)
        if data.get("name") != name:
            # The directory is the identity; a stale `name:` inside the file would make
            # run dirs and HF slugs disagree with where the data actually lives.
            data["name"] = name
    return data


def load_file(
    path: Path,
    *,
    name: str | None = None,
    require_installed: bool = True,
    keep_absent: Collection[str] = (),
) -> Profile:
    """The profile in ``path``. ``keep_absent``: planner or executor names accepted though not installed
    here -- the ones the file named before an edit, which an edit of something else must not be refused over."""
    absent_ok: bool | frozenset[str] = True if not require_installed else frozenset(keep_absent)
    return _validate(read_data(path, name=name), source=path, absent_ok=absent_ok)


def names_in_file(path: Path) -> frozenset[str]:
    """The planner and human executor ``path`` names, read as written. Empty when it cannot be read."""
    try:
        return _names_in(migrate(read_data(path))[0])
    except (ProfileError, ValueError, OSError):
        return frozenset()


def _validate(data: dict, *, source: Path | str, absent_ok: bool | Collection[str] = ()) -> Profile:
    try:
        return Profile.model_validate(data, context={"source": str(source), ABSENT_OK: absent_ok})
    except Exception as exc:
        raise ProfileInvalid(f"{source} is not a valid profile:\n{format_errors(exc)}") from exc


def _names_in(data: Mapping[str, Any]) -> frozenset[str]:
    """The planner and human executor a raw profile names, defaulted as a profile would default them."""
    planner = data.get("planner") if isinstance(data.get("planner"), Mapping) else {}
    hitl = data.get("hitl") if isinstance(data.get("hitl"), Mapping) else {}
    named = (planner.get("backend") or LEGACY_PLANNER, hitl.get("human_executor") or "teleop")
    return frozenset(n for n in named if isinstance(n, str))


def _names_on_disk(name: str) -> frozenset[str]:
    """What the profile ``name`` names on disk now: the names a rewrite of it may keep though absent here."""
    if not exists(name):
        return frozenset()
    return names_in_file(profiles_root() / name / "profile.yml")


def save(profile: Profile) -> Path:
    """Write a profile, creating its directory tree. Round-trips through validation first.

    What is written is the validated copy, so a planner's options reach the file as the planner
    normalised them, and a profile read in the older layout is written in the current one.

    A planner or executor the file ALREADY names may be absent from this machine: a profile collected on
    a workstation is rewritten on a laptop for reasons that have nothing to do with it (its prompt, a
    switch of the other one). A name being newly written must be installed.

    The file is replaced whole or not at all: serialised first, written beside it, and renamed over it.
    Truncating profile.yml before a dump that could fail -- over a value YAML cannot represent, a Ctrl-C,
    a full disk -- once left an empty file that then loaded, silently, as a profile of defaults.
    """
    try:
        profile = Profile.model_validate(profile.model_dump(), context={ABSENT_OK: _names_on_disk(profile.name)})
    except Exception as exc:
        raise ProfileInvalid(f"Profile {profile.name!r} is not valid:\n{format_errors(exc)}") from exc
    text = _yaml_text(_dump_dict(profile), what=f"Profile {profile.name!r}")
    pdir = profile.dir()
    pdir.mkdir(parents=True, exist_ok=True)
    for status in STATUSES:
        (pdir / "trajectories" / status).mkdir(parents=True, exist_ok=True)
    if not profile.calibration_file().is_file():
        profile.calibration_file().write_text("{}\n")
    path = profile.profile_file()
    _write_atomic(path, text)
    return path


def _yaml_text(data: Any, *, what: str) -> str:
    from ruamel.yaml.representer import RepresenterError

    buf = io.StringIO()
    try:
        _new_yaml().dump(data, buf)
    except RepresenterError as exc:
        raise ProfileError(
            f"{what} holds a value that cannot be written as YAML: {exc}",
            hint="planner.options (and an executor's options) must be plain data: strings, numbers, "
            "booleans, lists and mappings. A validate_options that normalises through pydantic returns "
            "model_dump(mode='json').",
        ) from None
    text = buf.getvalue()
    if not text.strip():
        raise ProfileError(f"{what} serialised to nothing, so nothing was written.")
    return text


def _write_atomic(path: Path, text: str) -> None:
    partial = path.with_name(f".{path.name}.partial")
    try:
        with partial.open("w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


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
    non-destructive workspace delete — re-creating the name brings the data back.

    ``name`` is checked before it becomes a path, and a purge removes only a directory directly inside
    profiles/: this is reached from a URL, and ``DELETE /api/profiles/%2E%2E?purge=true`` once meant
    ``rmtree(<data root>)``. A soft-deleted profile (profile.yml gone, data kept) can still be purged.
    """
    import shutil

    root = profiles_root()
    pdir = root / _checked(name)
    if not pdir.is_dir():
        raise ProfileError(f"Profile {name!r} does not exist.")
    if keep_data:
        pdir.joinpath("profile.yml").unlink(missing_ok=True)
        return pdir
    if pdir.is_symlink() or pdir.resolve().parent != root.resolve():
        raise ProfileError(
            f"Refusing to purge {pdir}: it is not a profile directory inside {root}.",
            hint="`tandem profile delete NAME` without --purge removes the profile and keeps its data.",
        )
    shutil.rmtree(pdir)
    return pdir


# --------------------------------------------------------------------------- switching what a profile uses
#
# `tandem planners use` and `tandem executors use` (and the web's buttons, and `tandem init --planner`)
# change one name in a profile. They work from the file as written rather than from a loaded profile,
# because they are also how a profile is REPAIRED: one naming a planner or executor this machine no
# longer has does not load, and the command every error points at must not refuse for that reason.
# Everything else in the file is still validated, and a problem anywhere else still refuses.


def stash_file(profile_dir: Path, backend: str) -> Path:
    """Where a planner's options are set aside when its profile switches to another planner."""
    return profile_dir / f"planner-options.{backend}.yml"


def switch_planner(name: str, backend: str, *, options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Point the profile ``name`` at the planner ``backend``, and say what happened to planner.options.

    ``backend`` must already be known to be installed (the caller asks the registry first). ``options``
    are laid over what the planner starts with -- for a planner that needs a setting no default can give.

    A planner's options are its own -- another planner refuses them rather than ignore them -- so they
    cannot stay in profile.yml once it plans with another. They are not thrown away either: they are the
    rig's robot address and home pose, a preset's TAMP settings, which a switch back to bare defaults
    would lose without a word. They are set aside in ``planner-options.<planner>.yml`` beside the
    profile, and switching back restores them, checked again by the planner. If it no longer accepts
    them, the switch goes ahead with its defaults, the file is kept, and the result says why.

    Returns ``profile`` (as saved), ``previous``, ``changed``, ``dropped_options`` (what left
    profile.yml), ``saved_to`` (where they went), ``restored_options`` and ``restore_problem``.
    """
    from tandem.core.errors import one_line

    path = _existing_file(name)
    data, _ = migrate(read_data(path, name=name))
    planner = data.get("planner") if isinstance(data.get("planner"), dict) else {}
    previous = str(planner.get("backend") or LEGACY_PLANNER)
    current = dict(planner.get("options") or {}) if isinstance(planner.get("options"), dict) else {}
    given = dict(options or {})
    changed = previous != backend
    restored: dict[str, Any] = {}
    problem: str | None = None
    stash = stash_file(path.parent, backend)

    spec: PlannerSpec | None = None
    if not changed:
        if given:
            spec = planner_spec(backend, {**current, **given}, profile=name)
    else:
        if stash.is_file():
            try:
                kept = _read_mapping(stash)
                spec = planner_spec(backend, {**kept, **given}, profile=name)
                restored = kept
            except ProfileError as exc:
                problem = f"{stash.name} was not restored: {one_line(exc.message)}"
        if spec is None:
            spec = planner_spec(backend, given, profile=name)
    if spec is not None:
        data["planner"] = spec.model_dump(mode="python")
    profile = _validate(data, source=path, absent_ok=_names_in(data) | {previous})

    saved_to: str | None = None
    if changed and current:
        target = stash_file(path.parent, previous)
        _write_atomic(target, _yaml_text(current, what=f"{previous}'s planner.options"))
        saved_to = str(target)
    if spec is not None:
        save(profile)
    if restored:
        stash.unlink(missing_ok=True)
    return {
        "profile": profile,
        "previous": previous,
        "changed": changed,
        "dropped_options": current if changed else {},
        "saved_to": saved_to,
        "restored_options": restored,
        "restore_problem": problem,
    }


def set_human_executor(name: str, executor: str) -> dict[str, Any]:
    """Make the profile ``name`` hand its human phases to ``executor``, working from the file as written.

    ``executor`` must already be known (the caller asks the executor registry first). Every executor's
    settings under hitl.human_executor_options stay where they are.
    """
    path = _existing_file(name)
    data, _ = migrate(read_data(path, name=name))
    hitl = dict(data["hitl"]) if isinstance(data.get("hitl"), dict) else {}
    previous = str(hitl.get("human_executor") or "teleop")
    hitl["human_executor"] = executor
    data["hitl"] = hitl
    profile = _validate(data, source=path, absent_ok=_names_in(data) | {previous})
    if previous != executor:
        save(profile)
    return {"profile": profile, "previous": previous, "changed": previous != executor}


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        with path.open() as fh:
            data = _new_yaml().load(fh)
    except Exception as exc:
        raise ProfileError(f"{path} is not valid YAML: {exc}") from exc
    data = _plain(data)
    if not isinstance(data, dict):
        raise ProfileError(f"{path} is not a mapping of settings.")
    return data


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
