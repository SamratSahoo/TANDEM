"""Profiles — a task, and the trajectories collected for it.

A profile is what a task is collected with: the prompt, phase planning, the planner and its TAMP
settings, recording and export. It is one YAML file, named for the profile. What the machine it is
collected on has -- the robot's address, the cameras, their extrinsics, a grasp server's URL -- is not a
task's: it is the rig's (``tandem.core.rig``, rig.yml beside config.toml), which every profile shares.

On disk::

    <data_root>/profiles/<name>.yml           the task's settings (the file's name is the profile's)
    <data_root>/profiles/.planner-options/    a previous planner's options, set aside by `planners use`
    <data_root>/trajectories/<name>/
        eval/<ts>/            # collected, not yet labeled
        success/<ts>/
        failure/<ts>/

The TANDEM paper's five tasks ship as five ordinary profile files (``BUILTIN``, in
``tandem/resources/profiles/``), with the settings the paper collected each one with. `tandem init`
copies them into profiles/; after that they are the user's own, to edit or delete like any other. A new
task starts from ``resources/profile_template.yml``: what those five share, with its own prompt
(``create``).

Profiles written before version 3 were directories (``profiles/<name>/profile.yml``, with the cameras,
the robot and a calibration.json of their own). ``tandem.core.layout`` moves them into this layout and
their machine settings into the rig; until it has, they are said to be there and are not loaded.
"""

from __future__ import annotations

import io
import logging
import os
import re
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr, ValidationInfo, field_validator, model_validator
from ruamel.yaml import YAML
from ruamel.yaml.representer import RoundTripRepresenter

from tandem.core import settings as settings_mod
from tandem.core.errors import ProfileError, ProfileInvalid, TandemError

# Same rule the source used for DC_WORKSPACE: a safe single path segment (no traversal) that
# is also a valid HuggingFace repo-name fragment.
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

STATUSES = ("eval", "success", "failure")

#: What each profile file is written as. 3: one file per profile, holding the task's settings only; the
#: robot, cameras and calibration are the rig's. (2: a planner's settings under planner.options; 1: at the
#: top level.)
LAYOUT_VERSION = 3

#: The top-level sections a profile had before version 3 that are not a task's. Refused in a profile file.
OLD_SECTIONS = ("cameras", "robot", "perception", "tamp")

#: Where a profile's previous planner's options are set aside (``stash_file``), inside profiles/.
STASH_DIR = ".planner-options"

#: The TANDEM paper's five tasks, in the paper's order (Fig. 3): profiles shipped in resources/profiles/,
#: which `tandem init` copies into profiles/ (``seed_builtins``).
BUILTIN = (
    "cover-bread-rolls",
    "solve-constrained-puzzle",
    "sort-and-cover-snacks",
    "open-obstructed-book",
    "store-bread-in-closed-box",
)

#: What every new profile starts from (``create``): the paper's settings, with no task yet.
TEMPLATE = "profile_template.yml"

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

    `options` is the named planner's own settings for this TASK: TiPToP's TAMP overrides, a toy
    planner's list of items. What a planner needs of the MACHINE -- a robot shim's ports, a grasp
    server -- is the rig's (``planners.<name>`` in rig.yml), and refused here with where it lives. Its
    keys are the planner's to define and to check, so the planner checks them -- here, when the profile
    loads, through its ``validate_options`` -- and what is stored is what the planner returned:
    normalised, defaults filled in. A mistake is then found when the profile is edited, not when a
    session starts with the arm about to move. A planner that cannot be loaded at all cannot check them;
    its options are kept as written, and the session that tries to build it says why.
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
    planner may require a task setting that has no sensible default (a scene file), and its
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

    @model_validator(mode="before")
    @classmethod
    def _no_rate(cls, data: Any) -> Any:
        if isinstance(data, dict) and "fps" in data:
            raise ValueError(
                "recording.fps is gone: nothing ever read it. Each camera records at the rig's "
                "cameras.<role>.fps; remove it from the profile"
            )
        return data


class ExportSpec(BaseModel):
    model_config = {"extra": "forbid"}

    hf_repo: str = ""
    private: bool = False


class Profile(BaseModel):
    """A validated profile. Its name is its file's; it is never written into the file."""

    model_config = {"extra": "forbid"}

    version: int = LAYOUT_VERSION
    name: str = "default"
    description: str = ""
    task: TaskSpec = Field(default_factory=TaskSpec)
    # Phase planning: tandem's own method, so its settings are tandem's whichever planner runs.
    hitl: HitlSpec = Field(default_factory=HitlSpec)
    # Which planner, and its settings for this task (planner.options).
    planner: PlannerSpec = Field(default_factory=PlannerSpec)
    recording: RecordingSpec = Field(default_factory=RecordingSpec)
    export: ExportSpec = Field(default_factory=ExportSpec)
    # The data root this profile's file and trajectories are under, fixed by `pinned`; None resolves it
    # afresh on every call. Not a setting: it is never written into the file.
    _root: Path | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _current_layout(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        try:
            version = int(data.get("version", LAYOUT_VERSION))
        except (TypeError, ValueError):
            return data  # the version field says what is wrong with it
        older = [key for key in OLD_SECTIONS if key in data]
        if version < LAYOUT_VERSION or older:
            found = f" (it has {', '.join(older)} at the top)" if older else f" (version {version})"
            raise ValueError(
                f"this profile is in the layout before version {LAYOUT_VERSION}{found}: its robot and cameras "
                "are this machine's rig now. `tandem profile migrate` converts old profile directories; for a "
                "file, move those sections into rig.yml (`tandem rig edit`) and set version: "
                f"{LAYOUT_VERSION}"
            )
        return data

    @field_validator("version")
    @classmethod
    def _readable(cls, v: int) -> int:
        if v > LAYOUT_VERSION:
            raise ValueError(
                f"version {v} was written by a newer tandem (this one reads version {LAYOUT_VERSION}); "
                "upgrade tandem"
            )
        return v

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

    def _data_root(self) -> Path:
        return self._root if self._root is not None else settings_mod.load().resolved_data_root()

    def pinned(self) -> Profile:
        """This profile, with its data root fixed where it resolves now.

        The paths below otherwise resolve the data root afresh on every call, from $TANDEM_DATA_ROOT, the
        settings and the home directory, whichever thread asks. A session's merge runs on a thread of
        its own, and can outlive the environment it was started in: one that outlived its test put a
        trajectory into the real ~/tandem-data, the test's own data root having been unset by then.
        A session works from a pinned copy, so every leg, record and merge of it lands in the one
        place it started in. A shallow copy: the settings it holds are the same objects. It compares
        unequal to an unpinned profile (pydantic compares private attributes too); compare
        ``model_dump()`` to ask whether two say the same thing.
        """
        copy = self.model_copy()
        copy._root = self._data_root()
        return copy

    def file(self) -> Path:
        return self._data_root() / "profiles" / f"{self.name}.yml"

    def trajectories_dir(self) -> Path:
        return self._data_root() / "trajectories" / self.name

    def status_dir(self, status: str) -> Path:
        if status not in STATUSES:
            raise ProfileError(f"unknown status {status!r}", hint=f"one of: {', '.join(STATUSES)}")
        return self.trajectories_dir() / status

    def goal_or_prompt(self) -> str:
        return self.task.goal or self.task.prompt


_log = logging.getLogger(__name__)
# Whether the old-layout notice was given in this process: a command that lists profiles ten times says it once.
_noticed_old_layout = False


# --------------------------------------------------------------------------- store


def profiles_root() -> Path:
    return settings_mod.load().profiles_root()


def trajectories_root() -> Path:
    return settings_mod.load().trajectories_root()


def list_names() -> list[str]:
    """Every profile here: the names of the ``<name>.yml`` files in profiles/, sorted."""
    root = profiles_root()
    _notice_old_layout(root)
    if not root.is_dir():
        return []
    return sorted(p.stem for p in root.iterdir() if p.suffix == ".yml" and is_name(p.stem) and p.is_file())


def _notice_old_layout(root: Path) -> None:
    """Say, once per process, that profiles in the layout before version 3 are here and not listed."""
    global _noticed_old_layout
    if _noticed_old_layout:
        return
    from tandem.core import layout

    pending = layout.pending(root)
    if not pending:
        return
    _noticed_old_layout = True
    _log.warning(
        "%d profile(s) are in the old layout (%s): `tandem init` moves them, and sets up the rig from them "
        "(or `tandem profile migrate`)",
        len(pending),
        ", ".join(pending),
    )


def is_name(name: object) -> bool:
    """Whether ``name`` can be a profile's name: one safe path segment, never ``.`` or ``..``."""
    return isinstance(name, str) and NAME_RE.fullmatch(name) is not None


def _checked(name: object) -> str:
    """``name``, if it can be a profile's name; otherwise a ProfileError.

    Every path under profiles/ and trajectories/ is built from a name, and names arrive from URLs as well
    as from the command line -- where ``%2E%2E`` decodes to ``..`` before any route sees it. A name that
    is not a profile name never becomes a path.
    """
    if not is_name(name):
        known = list_names()
        raise ProfileError(
            f"{name!r} is not a profile name.",
            hint=f"A profile's name matches {NAME_RE.pattern}. "
            + (f"Known profiles: {', '.join(known)}." if known else "No profiles exist yet."),
        )
    return str(name)


def path_of(name: str) -> Path:
    """The profile ``name``'s file, whether or not it exists. ``name`` is checked before it becomes a path."""
    return profiles_root() / f"{_checked(name)}.yml"


def _not_found(name: str, path: Path) -> ProfileError:
    if (profiles_root() / name / "profile.yml").is_file():
        return ProfileError(
            f"Profile {name!r} is in the layout before version {LAYOUT_VERSION} "
            f"({profiles_root() / name / 'profile.yml'}), and has not been moved yet.",
            hint="`tandem init` moves it, and sets up this machine's rig from it (or `tandem profile migrate`).",
        )
    known = list_names()
    hint = (
        f"Known profiles: {', '.join(known)}."
        if known
        else "No profiles here yet: `tandem init` adds the paper's five tasks, and `tandem profile create NAME "
        '--prompt "..."` makes your own.'
    )
    return ProfileError(f"Profile {name!r} not found at {path}.", hint=hint)


def exists(name: str) -> bool:
    return is_name(name) and path_of(name).is_file()


def _existing_file(name: str) -> Path:
    path = path_of(name)
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
    settings" it would load as a profile of pure defaults -- another planner, the template's task --
    without a word.
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
        # The file's name is the identity; a stale `name:` inside it would make run dirs and HF slugs
        # disagree with where the data actually lives.
        data["name"] = name
    return data


def load_file(
    path: Path,
    *,
    name: str | None = None,
    require_installed: bool = True,
    keep_absent: Collection[str] = (),
) -> Profile:
    """The profile in ``path``, named ``name`` (default: the file's stem). ``keep_absent``: planner or
    executor names accepted though not installed here -- the ones the file named before an edit, which an
    edit of something else must not be refused over."""
    absent_ok: bool | frozenset[str] = True if not require_installed else frozenset(keep_absent)
    name = name if name is not None else (Path(path).stem if is_name(Path(path).stem) else None)
    return _validate(read_data(path, name=name), source=path, absent_ok=absent_ok)


def names_in_file(path: Path) -> frozenset[str]:
    """The planner and human executor ``path`` names, read as written. Empty when it cannot be read."""
    try:
        return _names_in(read_data(path))
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
    named = (planner.get("backend") or PlannerSpec().backend, hitl.get("human_executor") or "teleop")
    return frozenset(n for n in named if isinstance(n, str))


def _names_on_disk(name: str) -> frozenset[str]:
    """What the profile ``name`` names on disk now: the names a rewrite of it may keep though absent here."""
    if not exists(name):
        return frozenset()
    return names_in_file(path_of(name))


def save(profile: Profile) -> Path:
    """Write a profile's file, and make its trajectories' directories. Round-trips through validation first.

    What is written is the validated copy, so a planner's options reach the file as the planner
    normalised them.

    A planner or executor the file ALREADY names may be absent from this machine: a profile collected on
    a workstation is rewritten on a laptop for reasons that have nothing to do with it (its prompt, a
    switch of the other one). A name being newly written must be installed.

    The file is replaced whole or not at all (``paths.write_atomic``): truncating it before a dump that
    could fail -- over a value YAML cannot represent, a Ctrl-C, a full disk -- once left an empty file
    that then loaded, silently, as a profile of defaults.
    """
    # Where it goes, from the profile given: a pinned one keeps its data root through the copy below.
    path, trajectories = profile.file(), profile.trajectories_dir()
    try:
        profile = Profile.model_validate(profile.model_dump(), context={ABSENT_OK: _names_on_disk(profile.name)})
    except Exception as exc:
        raise ProfileInvalid(f"Profile {profile.name!r} is not valid:\n{format_errors(exc)}") from exc
    text = _yaml_text(_dump_dict(profile), what=f"Profile {profile.name!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    _make_trajectory_dirs(trajectories)
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
    from tandem.core import paths

    paths.write_atomic(path, text)


def _dump_dict(profile: Profile) -> dict:
    """model_dump with None-valued optionals dropped, so the file stays readable, and without the name:
    the file's own name is the profile's."""
    data = profile.model_dump(mode="python", exclude_none=True)
    data.pop("name", None)
    return data


def delete(name: str, *, keep_data: bool = True) -> Path:
    """Remove a profile's file. By default its trajectories survive, mirroring the source's
    non-destructive workspace delete — re-creating the name brings the data back.

    ``name`` is checked before it becomes a path, and a purge removes only a directory directly inside
    trajectories/: this is reached from a URL, and ``DELETE /api/profiles/%2E%2E?purge=true`` once meant
    ``rmtree(<data root>)``. A soft-deleted profile (its file gone, its data kept) can still be purged.
    """
    import shutil

    name = _checked(name)
    path = path_of(name)
    root = trajectories_root()
    data = root / name
    if not path.is_file() and not data.is_dir():
        raise ProfileError(f"Profile {name!r} does not exist.")
    if not keep_data and data.exists():
        if data.is_symlink() or data.resolve().parent != root.resolve():
            raise ProfileError(
                f"Refusing to purge {data}: it is not a profile's trajectories directory inside {root}.",
                hint="`tandem profile delete NAME` without --purge removes the profile and keeps its data.",
            )
    path.unlink(missing_ok=True)
    if not keep_data and data.exists():
        shutil.rmtree(data)
    return path


# --------------------------------------------------------------------------- the paper's five, and new profiles


def builtin_path(name: str) -> Path:
    """The packaged copy of one of the paper's five (``BUILTIN``)."""
    from tandem import resources

    return resources.path(f"profiles/{name}.yml")


def builtin_text(name: str) -> str:
    """The packaged copy of one of the paper's five, as written. Read through the package, so a zipped
    install reads it as well as an unpacked one."""
    from tandem import resources

    return resources.read(f"profiles/{name}.yml")


def seed_builtins() -> list[str]:
    """Copy each of the paper's five that profiles/ does not have yet, word for word. Returns the names copied.

    Never over one that is there: once copied it is the user's, and their edits are the point. One they
    deleted comes back the next time `tandem init` runs.
    """
    written = []
    for name in BUILTIN:
        path = path_of(name)
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(path, builtin_text(name))
        _make_trajectory_dirs(trajectories_root() / name)
        written.append(name)
    return written


def source_text(name: str) -> str:
    """The profile ``name`` as written, to copy: this machine's file, else the paper's packaged one.

    The packaged copy is what lets `tandem profile create X --from cover-bread-rolls` work before `tandem
    init` has copied the five in.
    """
    import difflib

    name = _checked(name)
    path = path_of(name)
    if path.is_file():
        return path.read_text()
    if name in BUILTIN:
        return builtin_text(name)
    if (profiles_root() / name / "profile.yml").is_file():
        raise _not_found(name, path)  # in the old layout: it says how to move it
    close = difflib.get_close_matches(name, sorted({*list_names(), *BUILTIN}), n=1, cutoff=0.6)
    raise ProfileError(
        f"There is no profile {name!r} to copy.",
        hint=(f"Did you mean {close[0]!r}? " if close else "")
        + f"It copies a profile here (`tandem profile list`) or one of the paper's: {', '.join(BUILTIN)}.",
    )


def create(
    name: str,
    *,
    source: str | None = None,
    prompt: str | None = None,
    planner: str | None = None,
    force: bool = False,
) -> Profile:
    """Write a new profile ``name``: the paper's settings (the template) with ``prompt``, or a copy of ``source``.

    ``source`` is any profile here or one of the paper's five, copied as written -- comments included --
    with ``prompt`` as its task when one is given. Without a source there is no task yet, so ``prompt`` is
    required: a profile that would collect "describe the task here" is not one. ``planner`` is the planner
    a new task plans with (default: the machine's ``default_planner``); the template's tamp settings are
    TiPToP's, so a new task on another planner starts from that planner's own defaults instead.

    Validated before anything is written, as ``load`` would read it: the planner and human executor it
    names must be installed here, since a profile is created to collect with.
    """
    from datetime import date

    from tandem import resources

    name = _checked(name)
    path = path_of(name)
    if path.exists() and not force:
        raise ProfileError(
            f"Profile {name!r} already exists ({path}).",
            hint="Pass --force to replace it (its trajectories are kept), or choose another name.",
        )
    prompt = (prompt or "").strip() or None
    if source is None and prompt is None:
        raise ProfileError(
            "A new profile needs its task.",
            hint=f'`tandem profile create {name} --prompt "put the cup on the plate"` starts it from the paper\'s '
            f"settings; `--from PROFILE` copies another profile instead (the paper's: {', '.join(BUILTIN)}).",
        )

    text = source_text(source) if source is not None else resources.read(TEMPLATE)
    doc = _new_yaml().load(_without_header(text))
    if not isinstance(doc, dict):
        raise ProfileInvalid(f"{source or TEMPLATE} is not a mapping of profile settings, so it cannot be copied.")
    doc.pop("name", None)
    doc["version"] = LAYOUT_VERSION
    doc["description"] = f"copied from {source}" if source is not None else ""
    if prompt is not None:
        if not isinstance(doc.get("task"), dict):
            from ruamel.yaml.comments import CommentedMap

            doc["task"] = CommentedMap()
        doc["task"]["prompt"] = prompt
    if source is None:
        chosen = planner or settings_mod.load().default_planner
        written = doc.get("planner") if isinstance(doc.get("planner"), dict) else {}
        if written.get("backend") != chosen:
            doc["planner"] = planner_spec(chosen, profile=name).model_dump(mode="python")

    origin = f"as a copy of {source}" if source is not None else "with the TANDEM paper's settings"
    header = (
        f"# {name}: made by `tandem profile create` {origin} on {date.today():%Y-%m-%d}.\n"
        "# Only the task: this machine's robot, cameras and calibration are its rig (`tandem rig show`).\n"
    )
    body = io.StringIO()
    yaml = _new_yaml()
    yaml.width = 4096  # lines as the source wrote them, not folded at 100 columns
    yaml.Representer = _NullAsWritten
    yaml.dump(doc, body)
    data = _resolve_all(_plain(doc))
    data["name"] = name
    profile = _validate(data, source=f"{name} ({'copied from ' + source if source else 'new'})")
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, header + body.getvalue())
    _make_trajectory_dirs(profile.trajectories_dir())
    return profile


class _NullAsWritten(RoundTripRepresenter):
    """Writes None as ``null``, as the template does, not as the bare ``goal:`` ruamel writes by default, which
    reads as a key left unfinished. A subclass of its own: ``add_representer`` changes the class it is called
    on, and on ruamel's own it would change every dump in the process."""


_NullAsWritten.add_representer(
    type(None), lambda representer, _: representer.represent_scalar("tag:yaml.org,2002:null", "null")
)


def _without_header(text: str) -> str:
    """``text`` from its first setting on: the comment block a file opens with says what THAT file is."""
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.strip() and not line.lstrip().startswith("#"):
            return "".join(lines[index:])
    return ""


def _make_trajectory_dirs(root: Path) -> None:
    for status in STATUSES:
        (root / status).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- switching what a profile uses
#
# `tandem planners use` and `tandem executors use` (and the web's buttons) change one name in a profile.
# They work from the file as written rather than from a loaded profile, because they are also how a
# profile is REPAIRED: one naming a planner or executor this machine no longer has does not load, and the
# command every error points at must not refuse for that reason. Everything else in the file is still
# validated, and a problem anywhere else still refuses.


def stash_file(name: str, backend: str) -> Path:
    """Where the profile ``name``'s options for ``backend`` are set aside when it switches to another planner."""
    return profiles_root() / STASH_DIR / f"{_checked(name)}.{backend}.yml"


def switch_planner(name: str, backend: str, *, options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Point the profile ``name`` at the planner ``backend``, and say what happened to planner.options.

    ``backend`` must already be known to be installed (the caller asks the registry first). ``options``
    are laid over what the planner starts with -- for a planner that needs a setting no default can give.

    A planner's options are its own -- another planner refuses them rather than ignore them -- so they
    cannot stay in the profile once it plans with another. They are not thrown away either: they are a
    task's TAMP settings, which a switch back to bare defaults would lose without a word. They are set
    aside in ``profiles/.planner-options/<name>.<planner>.yml``, and switching back restores them,
    checked again by the planner. If it no longer accepts them, the switch goes ahead with its defaults,
    the file is kept, and the result says why. (A planner's machine settings are the rig's, and never
    move: every profile shares them.)

    Returns ``profile`` (as saved), ``previous``, ``changed``, ``dropped_options`` (what left the
    profile), ``saved_to`` (where they went), ``restored_options`` and ``restore_problem``.
    """
    from tandem.core.errors import one_line

    path = _existing_file(name)
    data = read_data(path, name=name)
    planner = data.get("planner") if isinstance(data.get("planner"), dict) else {}
    previous = str(planner.get("backend") or PlannerSpec().backend)
    current = dict(planner.get("options") or {}) if isinstance(planner.get("options"), dict) else {}
    given = dict(options or {})
    changed = previous != backend
    restored: dict[str, Any] = {}
    problem: str | None = None
    stash = stash_file(name, backend)

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
        target = stash_file(name, previous)
        target.parent.mkdir(parents=True, exist_ok=True)
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
    data = read_data(path, name=name)
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
    """A path a profile names, as its author meant it: absolute as written, else beside the profile's file.

    One reading for every path a profile holds, so a relative path means "in profiles/, next to the
    file" and not "wherever the command happened to be started from".
    """
    candidate = Path(os.path.expanduser(str(value)))
    if candidate.is_absolute():
        return candidate
    return (profile.file().parent / candidate).resolve()


def resolve_cache_path(profile: Profile) -> str | None:
    """The proposal cache's absolute path, or None when the profile sets none.

    A relative cache path means "beside the profile's file" (``resolve_path``). One reading, because three
    different commands read this key: a collection session, `tandem plan --profile`, and `tandem
    doctor`. Resolving it differently in any of them means they open DIFFERENT SQLite files, so the
    cache never hits across them -- and since opening one creates its parent directories, the odd one
    out silently litters a second cache wherever it was run from.
    """
    if not profile.hitl.cache_path:
        return None
    return str(resolve_path(profile, str(profile.hitl.cache_path)))


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
