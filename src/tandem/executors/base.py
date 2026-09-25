"""Who carries out a human phase: the executor protocol, and the registry that finds one by name.

The paper gives every magic operator an executor, pi_omega_Delta: whatever brings about the effects the
proposer invented for a human phase. TANDEM's own is a person driving the arm through teleoperation, and
that is the only one that ships. It is not the only thing that could do the job. A phase is a person's
because the planner cannot express what it asks for, not because a person is the only thing able to do
it, and the HITL-TAMP baseline hands the same phase to a policy trained on the teleop legs of earlier
runs. So the phase loop asks for an executor by name (``hitl.human_executor``), and this module decides
what that name means.

What an executor decides, and what it does not:

* It decides HOW a phase is carried out, and it records that as a leg of the trajectory, stamped with
  the ``LegSpec`` it is given, so the merge can join it to the robot's legs.
* It does not decide the phase, its operator, or whether the phase happened. The proposal fixed the
  first two before the arm moved, and the loop's camera check decides the third, against the same
  effects whoever did the work. Swapping a person for a policy changes who drives the arm for one leg;
  the plan does not know the difference.
* It does not own the hardware either. The planner holds the robot and every camera exclusively, so the
  loop releases them before `HumanExecutor.run` and takes them back after it. An executor starts from an
  arm that is free and must hand it back free, which is why a leg that cannot be ended is a
  `CustodyError` and not an ordinary failure.

The registry maps a name to a factory rather than an instance, for the reason the planner registry
does: building an executor may start a process or open a device, and importing one may need an
environment the machine does not have. Listing executors, and saying what each one needs, must cost
nothing and must work on a laptop, because that is where a profile is edited. Three sources feed it:

* the executors that ship with tandem (`teleop`);
* ``register_human_executor``, for code that runs in this process (a test, a script, a plugin's
  import side effect);
* the ``tandem.human_executors`` entry point group, for a package installed beside tandem. The entry
  point names an `ExecutorFactory`, or a class with the metadata below as class attributes, as
  ``"package.module:NAME"``. The module it names must import nothing heavy, because naming the
  executor in a listing imports it. Heavy imports belong inside ``create`` or ``run``.

Resolution is loud. An unknown name is an error with the nearest known name as the hint, and two
installed packages claiming one name is an error rather than a guess, because a dataset recorded by one
executor and attributed to another cannot be interpreted afterwards.
"""

from __future__ import annotations

import difflib
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module, metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from tandem.core import names
from tandem.core.errors import TandemError
from tandem.planners.base import LegSpec

if TYPE_CHECKING:
    from tandem.core.phase_loop import HumanPhase

ENTRY_POINT_GROUP = "tandem.human_executors"

#: What an unknown name's hint points at: the listing of every executor, broken ones included.
LIST_COMMAND = "tandem executors list"

# What a leg an executor records is, in its _meta.json. The merge and the export treat "tamp" legs as
# the planner's (the DROID action identity is derived only for those), so an executor may not claim it:
# its legs would be read as robot legs.
SEGMENT_SOURCES = ("teleop", "policy")

STATUSES = ("done", "aborted", "ended_by_operator")


# --------------------------------------------------------------------------- what is asked, and answered


@dataclass(frozen=True)
class HumanPhaseRequest:
    """One human phase, as whoever carries it out is asked to do it.

    Everything a person is shown and a policy might condition on: which step of how many (0-based, like
    every leg stamp), the phase's words, its magic operator in the form ``HumanOperator.to_json`` writes
    (None for a phase that has none), and what the camera will be asked afterwards. `attempt` counts
    from 1. On a retry, `missing` is what the last check said is still not true, already phrased for a
    person, so a second attempt is aimed at what the first one missed.

    `n_phases` is 0 for a phase with no plan behind it. Nothing is then stamped on the leg, because
    "phase 0 of 0" reads as knowledge nobody has.
    """

    phase_index: int
    n_phases: int
    description: str
    instructions: str
    operator: dict | None = None
    expected: list[str] = field(default_factory=list)
    attempt: int = 1
    missing: list[str] = field(default_factory=list)

    @classmethod
    def from_view(cls, view: HumanPhase, *, operator: Any = None) -> HumanPhaseRequest:
        """The request for the step the loop is showing the operator.

        Built from the loop's own view (``phase_loop.HumanPhase``) so the executor is asked for exactly
        what the person is looking at. `operator` is the phase's ``HumanOperator``, or its JSON form.
        """
        if operator is not None and not isinstance(operator, dict):
            operator = operator.to_json()
        return cls(
            phase_index=int(view.index),
            n_phases=int(view.total),
            description=view.description,
            instructions=view.instructions,
            operator=operator,
            expected=list(view.expected),
            attempt=int(view.attempt),
            missing=list(view.missing),
        )

    @property
    def stamped(self) -> bool:
        """Whether a leg of this phase says which phase it is."""
        return self.n_phases > 0

    def leg_spec(
        self, *, trajectory_id: str, instruction: str, segment_source: str, record: bool = True
    ) -> LegSpec:
        """The leg this phase is recorded as, stamped with the phase.

        Built from the request, so the leg and the prompt cannot disagree about which phase this is. The
        merge copies a leg's stamp into segments[], and that is the only place a merged demonstration
        says which stretch of it was which phase. `instruction` is the WHOLE task, as for a robot leg:
        it is the dataset's language label, and the phase's own words travel as `phase_description`.
        """
        return LegSpec(
            trajectory_id=trajectory_id,
            instruction=instruction,
            segment_source=segment_source,
            phase_index=self.phase_index if self.stamped else None,
            n_phases=self.n_phases if self.stamped else None,
            phase_description=self.description if self.stamped else "",
            record=record,
        )


@dataclass(frozen=True)
class HumanPhaseResult:
    """How one leg of a human phase ended.

    ``status``:

    * ``done``: the executor carried the phase through and handed the arm back. The loop checks the
      phase's effects next. It is still ``done`` when nothing was recorded (`n_frames` 0), for example
      when a person moved the arm with no driver running. Whether an unrecorded phase may stand is the
      loop's decision (``hitl.allow_unrecorded_human_phase``), not the executor's.
    * ``ended_by_operator``: the operator stopped the executor on purpose, to move on to the next phase.
      This is the HITL-TAMP baseline's "continue" signal to a policy that runs until a step limit. The leg
      is kept, and the reference implementation skips the check for it.
    * ``aborted``: the leg was cut off, or the executor could not carry out the phase at all (a policy
      server that never came up). Nothing the arm did may be read as the phase being done. A person
      giving up is not this. That is the "abort" answer at the loop's prompt, and it is given before
      any executor runs.

    `n_frames` counts every frame the leg wrote, over every recording the hand-off made. `leg_dir` is
    the last directory a recording went into, or None when nothing was recorded. `leg_dirs` is every
    one of them, in order, when the executor knows them (a teleop driver starts a new recording
    whenever one ends short of quitting); empty means `leg_dir` is the only one. The loop notes each
    against the plan the phase belongs to, so a merged episode can say which plan's phase it was.
    """

    status: Literal["done", "aborted", "ended_by_operator"]
    n_frames: int = 0
    leg_dir: Path | None = None
    leg_dirs: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"a human phase ends as one of {', '.join(STATUSES)}, not {self.status!r}")
        if int(self.n_frames) < 0:
            raise ValueError(f"n_frames cannot be negative, got {self.n_frames}")
        object.__setattr__(self, "n_frames", int(self.n_frames))
        if self.leg_dir is not None:
            object.__setattr__(self, "leg_dir", Path(self.leg_dir))
        object.__setattr__(self, "leg_dirs", tuple(Path(d) for d in self.leg_dirs))

    @property
    def recorded(self) -> bool:
        return self.n_frames > 0


class CustodyError(TandemError):
    """The executor cannot give the robot or the cameras back.

    Raised from `HumanExecutor.run` when whatever drove the arm will not let go of it. Nothing may reach
    for the hardware afterwards: it would open a camera another process still holds, and the failure
    would look like broken hardware rather than a process that will not die. The session must end, and
    this error says what is actually true.
    """


# --------------------------------------------------------------------------- what an executor is


def _ignore(*_args: Any) -> None:
    return None


@dataclass(frozen=True)
class ExecutorContext:
    """What an executor is built with: where it may write, and how it reaches the person watching.

    Given once per session, like a planner backend's construction arguments, and never per leg: a leg's
    particulars travel in `HumanPhaseRequest` and `LegSpec`.

    * `profile` is the validated ``Profile``: the task.
    * `rig` is this machine's rig (``tandem.core.rig.Rig``): the cameras a leg records from, the robot's
      address. None for a context built by hand; an executor that needs it then reads
      ``tandem.core.rig.load()``.
    * `session_dir` is the session's scratch directory, for anything that is not a recording: event
      files, logs, a policy server's socket.
    * `settings` is the machine's ``tandem.core.settings.Settings``. None re-reads them at every leg, so
      a ``tandem config set`` between two hand-offs takes effect at the next one.
    * `on_log(stream, text)` adds a line to the session's log. `on_emit(payload)` sends a message to
      every subscriber (the web UI).
    * `on_problem(message)` reports a problem the operator has to see now and can do something about,
      such as a driver that would not start while the arm is theirs. The session shows it until the leg
      ends.
    * `options` are this executor's own settings: the profile's
      ``hitl.human_executor_options.<name>`` block (a policy executor's checkpoint, say), as its
      ``validate_options`` hook returned it when the profile loaded. Empty when the profile sets none.
    """

    profile: Any
    session_dir: Path
    settings: Any = None
    on_log: Callable[[str, str], None] = _ignore
    on_emit: Callable[[dict], None] = _ignore
    on_problem: Callable[[str], None] = _ignore
    options: Mapping[str, Any] = field(default_factory=dict)
    # Last, so a context built positionally before it existed still means what it meant.
    rig: Any = None


@runtime_checkable
class HumanExecutor(Protocol):
    """Carries out one human phase at a time, as one leg of the trajectory.

    `name` is the registered name. `segment_source` is what its legs are (one of `SEGMENT_SOURCES`), and
    it must be written into each leg's ``_meta.json``. `display_name` and `summary` are what a person
    choosing an executor is shown.

    An executor may also define ``close() -> None``: release whatever building it started (a policy
    server, a device it opened). It is optional, and not in this protocol's members, so an executor
    that holds nothing between legs need not write one. When defined, it is called exactly once, when
    the session ends -- on every way it ends, a failure or a forced stop included -- after the arm is
    parked and the planner closed (``PhaseLoop.close``). It must not raise; one that does is logged.
    """

    name: str
    segment_source: str
    display_name: str
    summary: str

    def run(
        self,
        request: HumanPhaseRequest | None,
        leg: LegSpec,
        *,
        save_root: Path,
        should_stop: Callable[[], bool],
    ) -> HumanPhaseResult:
        """Carry out one phase, record it, and return once the arm is free again.

        Called with the robot and the cameras already released, and must not return until whatever drove
        the arm has let go of both. Raise `CustodyError` when that cannot be done.

        `request` is None for a hand-off the operator asked for at a phase boundary: the arm is lent to
        a person with no phase attached. Only an executor a person drives can honour that.

        `leg` is the recording contract, exactly as a robot leg gets it. The leg's ``_meta.json`` carries
        its `trajectory_id` and `segment_source`, and its phase stamp when `phase_index` is set.
        `save_root` is the profile's trajectories directory. A leg is written under
        ``save_root/eval/<stamp>/``, which is where ``merge.find_legs`` looks for it.

        `should_stop` is polled. It turns true when the leg must end: the operator handed the arm back,
        the session is stopping, or the caller's deadline passed. It is the only way a leg ends besides
        the executor finishing on its own, so an executor must never wait for anything without polling it.
        """
        ...

    def kill(self) -> None:
        """End the leg in flight at once. Called from another thread, by a forced stop.

        `run` then returns as soon as it can, with status ``aborted``. A no-op when nothing is running.
        """
        ...


# The members `create` checks a built executor for, so a plugin that misses one fails when it is built
# and names what is missing, not minutes later in the middle of a hand-off.
_EXECUTOR_MEMBERS = ("name", "segment_source", "display_name", "summary", "run", "kill")


@dataclass(frozen=True)
class ExecutorFactory:
    """How to build one kind of executor, and what to say about it before building it.

    The metadata lives here, beside the constructor, because it is read where the executor cannot be
    built: in a listing, on a laptop with no robot. `requirements` says in plain words what the executor
    needs on this machine. `check(settings)` returns the ones that are not met yet. It must be cheap and
    must not start anything, and an empty list means the executor is ready.
    """

    create: Callable[[ExecutorContext], HumanExecutor]
    display_name: str
    summary: str
    segment_source: str
    requirements: tuple[str, ...] = ()
    check: Callable[[Any], Sequence[str]] | None = None
    # Checks and normalises the executor's own settings (hitl.human_executor_options.<name>) when a
    # profile loads, the way a planner's validate_options checks planner.options: returns them as the
    # executor will read them, or raises TandemError / ValueError naming the key. None takes them as
    # written. What it returns is written back to the profile's file, so it must be plain YAML data.
    validate_options: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.segment_source not in SEGMENT_SOURCES:
            raise ValueError(
                f"an executor's legs are one of {', '.join(SEGMENT_SOURCES)}, not {self.segment_source!r}"
            )
        object.__setattr__(self, "requirements", tuple(str(r) for r in self.requirements))


@dataclass(frozen=True)
class ExecutorInfo:
    """One executor, described for a person choosing one. Building it is not needed.

    `origin` says where the name comes from. `unmet` lists the requirements this machine does not meet
    yet. `error` is set, and the rest is mostly empty, when the executor could not even be described:
    a plugin that fails to import, or a name two packages claim.
    """

    name: str
    display_name: str
    summary: str
    segment_source: str
    requirements: tuple[str, ...]
    origin: str
    unmet: tuple[str, ...] = ()
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.error is None and not self.unmet

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "summary": self.summary,
            "segment_source": self.segment_source,
            "requirements": list(self.requirements),
            "origin": self.origin,
            "unmet": list(self.unmet),
            "error": self.error,
            "ready": self.ready,
        }


# --------------------------------------------------------------------------- the registry

# The executors that ship with tandem, by import path, so listing them imports nothing. tandem's own
# pyproject declares the same paths under the entry point group, and `_candidates` counts an entry
# point that repeats a built-in's path as that built-in rather than as a second claim to the name.
_BUILTIN: dict[str, str] = {
    "teleop": "tandem.executors.teleop:FACTORY",
}

BUILTIN_ORIGIN = "built in"
REGISTERED_ORIGIN = "registered in this process"

_registered: dict[str, Any] = {}
# name -> [(import path, origin)], one entry per distinct path. Filled on first use and kept, because
# scanning every installed distribution's metadata is not free. `refresh` forgets it.
_discovered: dict[str, list[tuple[str, str]]] | None = None
_lock = threading.RLock()


def register_human_executor(name: str, factory: Any, *, replace: bool = False) -> None:
    """Make `factory` available as the human executor `name`, in this process.

    `factory` is an `ExecutorFactory`, a class (or other callable) that takes an `ExecutorContext` and
    has the metadata as class attributes, or the ``"module:attribute"`` import path of either. An
    import path is resolved when the executor is first described or built, not here.

    A name that is already taken, built in or installed, is refused unless `replace` is set. Shadowing
    an executor has to be done on purpose, because every leg it records is attributed to that name.
    """
    if not names.is_valid(name):
        raise TandemError(
            f"{name!r} cannot be the name of a human executor.",
            hint=f"Use {names.RULE}, as it is written in hitl.human_executor -- the same rule as a planner's.",
        )
    if factory is None:
        raise TandemError(f"No factory was given for the human executor {name!r}.")
    with _lock:
        if not replace and name in available():
            raise TandemError(
                f"A human executor named {name!r} already exists ({', '.join(_origins(name))}).",
                hint="Register it under another name, or pass replace=True to shadow it on purpose.",
            )
        _registered[name] = factory


def unregister_human_executor(name: str) -> None:
    """Forget an executor registered in this process. Built-in and installed ones are not removable here."""
    with _lock:
        if name not in _registered:
            raise TandemError(
                f"No human executor named {name!r} was registered in this process.",
                hint="Only register_human_executor's own registrations can be removed; uninstall a package "
                "to remove the executors it provides.",
            )
        del _registered[name]


def refresh() -> None:
    """Scan the installed packages' entry points again, for an install made while this process ran."""
    global _discovered
    with _lock:
        _discovered = None


def available() -> list[str]:
    """Every executor name, sorted. Imports nothing but the registry's own metadata scan."""
    with _lock:
        return sorted(set(_BUILTIN) | set(_registered) | set(_entry_points()))


def check_name(name: str) -> None:
    """Raise the registry's own error if nothing is called `name`. Loads nothing.

    For a caller that only has to know the name is real, such as profile validation, which must not
    import a plugin just to accept the name it asks for.
    """
    if name not in available():
        raise _unknown(name)


def info(name: str, *, settings: Any = None) -> ExecutorInfo:
    """Describe the executor `name`, including what this machine still lacks for it.

    `settings` defaults to the machine's own. Raises `TandemError` for an unknown name, a name two
    packages claim, or a plugin that cannot be imported. ``catalog`` reports those three instead.
    """
    factory, origin = _resolve(name)
    return ExecutorInfo(
        name=name,
        display_name=factory.display_name,
        summary=factory.summary,
        segment_source=factory.segment_source,
        requirements=factory.requirements,
        origin=origin,
        unmet=_unmet(factory, settings),
    )


def catalog(*, settings: Any = None) -> list[ExecutorInfo]:
    """Every executor, described. One broken plugin does not hide the others.

    This is what a listing shows. An executor that cannot be described still appears, with `error`
    saying why, because an installed package that silently vanished from the list would leave a person
    wondering what they did wrong.
    """
    if settings is None:
        settings = _machine_settings()
    out = []
    for name in available():
        try:
            out.append(info(name, settings=settings))
        except TandemError as exc:
            out.append(
                ExecutorInfo(
                    name=name,
                    display_name=name,
                    summary="",
                    segment_source="",
                    requirements=(),
                    origin=", ".join(_origins(name)),
                    error=exc.message if not exc.hint else f"{exc.message} {exc.hint}",
                )
            )
    return out


def create(name: str, ctx: ExecutorContext) -> HumanExecutor:
    """Build the executor `name` for one session.

    Its requirements are not checked here. An executor decides for itself what an unmet one means when
    it runs: teleop with no driver configured still lends the arm, so a person can do the step by hand.
    A caller that wants to refuse early asks `info(name).unmet` first.
    """
    factory, origin = _resolve(name)
    try:
        executor = factory.create(ctx)
    except TandemError:
        raise
    except (Exception, SystemExit) as exc:
        raise TandemError(
            f"Could not build the human executor {name!r} ({origin}): {type(exc).__name__}: {exc}"
        ) from exc

    missing = [member for member in _EXECUTOR_MEMBERS if not hasattr(executor, member)]
    missing += [m for m in ("run", "kill") if m not in missing and not callable(getattr(executor, m))]
    if missing:
        _discard(executor)
        raise TandemError(
            f"The human executor {name!r} ({origin}) built a {type(executor).__name__}, which has no "
            f"{', '.join(missing)}.",
            hint="An executor implements tandem.executors.HumanExecutor.",
        )
    if executor.segment_source != factory.segment_source:
        _discard(executor)
        raise TandemError(
            f"The human executor {name!r} ({origin}) declares its legs as {factory.segment_source!r} but "
            f"built one that records them as {executor.segment_source!r}.",
            hint="The two must agree: a listing and the recorded legs would otherwise disagree about what "
            "the legs are.",
        )
    return executor


def _discard(executor: Any) -> None:
    """Close an executor that was built and then refused, before the refusal is raised.

    Building it may already have started a process or opened a device, and nothing will hold it once
    `create` raises: the loop never keeps it, so its own ``close`` would never be reached.
    """
    close = getattr(executor, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            # Not raised in place of the refusal that follows: that says what is actually wrong with
            # the executor, and a second error from tidying up after it would only hide it.
            pass


# ---- resolution ---------------------------------------------------------------


def _entry_points() -> dict[str, list[tuple[str, str]]]:
    """Every executor installed packages declare, by name. Imports none of them."""
    global _discovered
    with _lock:
        if _discovered is None:
            found: dict[str, list[tuple[str, str]]] = {}
            # The selectable API, which Python 3.10 has. The dict-style result of 3.9 is not supported.
            for entry in metadata.entry_points(group=ENTRY_POINT_GROUP):
                target = entry.value.split("[", 1)[0].strip()
                claims = found.setdefault(entry.name, [])
                if all(target != known for known, _ in claims):
                    claims.append((target, _installed_by(entry)))
            _discovered = found
        return _discovered


def _installed_by(entry: Any) -> str:
    dist = getattr(entry, "dist", None)
    name = getattr(dist, "name", None) if dist is not None else None
    if name is None and dist is not None:
        try:
            name = dist.metadata["Name"]
        except Exception:
            name = None
    return f"installed by {name}" if name else "installed entry point"


def _candidates(name: str) -> list[tuple[Any, str]]:
    """Every distinct thing that claims `name`, with where each claim comes from."""
    with _lock:
        if name in _registered:
            # Registered in this process: either the only claim, or one that shadowed the others on
            # purpose (register_human_executor refuses a taken name without replace=True).
            return [(_registered[name], REGISTERED_ORIGIN)]
        claims: list[tuple[Any, str]] = []
        if name in _BUILTIN:
            claims.append((_BUILTIN[name], BUILTIN_ORIGIN))
        for target, origin in _entry_points().get(name, ()):
            if name in _BUILTIN and target == _BUILTIN[name]:
                continue  # tandem's own pyproject declaring its built-in
            claims.append((target, origin))
        return claims


def _origins(name: str) -> list[str]:
    return [origin for _, origin in _candidates(name)] or ["unknown"]


def _resolve(name: str) -> tuple[ExecutorFactory, str]:
    claims = _candidates(name)
    if not claims:
        raise _unknown(name)
    if len(claims) > 1:
        described = "; ".join(f"{target} ({origin})" for target, origin in claims)
        raise TandemError(
            f"More than one package provides a human executor named {name!r}: {described}.",
            hint="Uninstall one of them. tandem will not guess which one a dataset should be recorded "
            "with.",
        )
    target, origin = claims[0]
    return _as_factory(name, _load(name, target, origin), origin), origin


def _unknown(name: str) -> TandemError:
    known = available()
    suggestion = difflib.get_close_matches(str(name), known, n=1, cutoff=0.6)
    if suggestion:
        hint = f"Did you mean {suggestion[0]!r}?"
    else:
        hint = (
            f"Known human executors: {', '.join(known)}. A package adds one through the "
            f"{ENTRY_POINT_GROUP!r} entry point, so install the package that provides it, or name one "
            "of these."
        )
    # The listing also shows an installed executor that will not load, and why.
    hint += f" `{LIST_COMMAND}` shows every human executor."
    return TandemError(f"Unknown human executor {name!r}.", hint=hint)


def _load(name: str, target: Any, origin: str) -> Any:
    if not isinstance(target, str):
        return target
    module_name, sep, attribute = target.partition(":")
    if not sep or not module_name or not attribute:
        raise TandemError(
            f"The human executor {name!r} ({origin}) points at {target!r}, which is not "
            "'package.module:attribute'.",
        )
    try:
        obj: Any = import_module(module_name)
        for part in attribute.split("."):
            obj = getattr(obj, part)
    # SystemExit by name: a module that parses argv at import exits, and caught as nothing it took the
    # whole `tandem executors list` down with it. Not BaseException, so Ctrl-C still stops the command.
    except (Exception, SystemExit) as exc:
        raise TandemError(
            f"The human executor {name!r} ({origin}) could not be loaded from {target}: "
            f"{type(exc).__name__}: {exc}",
            hint="Reinstall the package that provides it, or uninstall it if it is no longer wanted.",
        ) from exc
    return obj


def _as_factory(name: str, obj: Any, origin: str) -> ExecutorFactory:
    """An `ExecutorFactory` for whatever was registered: one already, or a class carrying its metadata."""
    if isinstance(obj, ExecutorFactory):
        return obj
    if not callable(obj):
        raise TandemError(
            f"The human executor {name!r} ({origin}) is {obj!r}, which is neither an ExecutorFactory nor "
            "something that builds an executor.",
        )
    source = getattr(obj, "segment_source", None)
    if source not in SEGMENT_SOURCES:
        raise TandemError(
            f"The human executor {name!r} ({origin}) does not say what its legs are: segment_source is "
            f"{source!r}.",
            hint=f"Give it a segment_source of {' or '.join(map(repr, SEGMENT_SOURCES))}, or register an "
            "ExecutorFactory.",
        )
    doc = (getattr(obj, "__doc__", None) or "").strip()
    return ExecutorFactory(
        create=obj,
        display_name=str(getattr(obj, "display_name", "") or name),
        summary=str(getattr(obj, "summary", "") or (doc.splitlines()[0] if doc else "")),
        segment_source=source,
        requirements=tuple(getattr(obj, "requirements", ()) or ()),
        validate_options=getattr(obj, "validate_options", None),
    )


def options_for(name: str, options: Mapping[str, Any] | None) -> dict[str, Any]:
    """``options`` as the executor ``name`` reads them: its ``hitl.human_executor_options`` block.

    An executor with a ``validate_options`` hook is asked, and what it returns is kept. One without the
    hook, or one this machine does not have or cannot load, keeps its block as written: the profile is
    still edited and browsed where the executor is not installed, and the block is checked again on the
    machine that collects.
    """
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TandemError(f"must be a mapping of settings, not a {type(options).__name__}.")
    raw = dict(options)
    try:
        factory, _ = _resolve(name)
    except TandemError:
        return raw
    if factory.validate_options is None:
        return raw
    checked = factory.validate_options(raw)
    if not isinstance(checked, Mapping):
        raise TandemError(
            f"The human executor {name!r}'s validate_options returned a {type(checked).__name__}, not "
            "the options.",
            hint="validate_options(options) returns the options as the executor will read them, or raises.",
        )
    from tandem.planners.registry import not_plain_data

    problems = not_plain_data(checked, "options")
    if problems:
        # Written into the profile's file as it stands, like a planner's (registry.options_for says why).
        raise TandemError(
            f"The human executor {name!r}'s validate_options returned what a profile cannot store: "
            + "; ".join(problems[:5]),
            hint="Return plain data: mappings, lists, strings, numbers, booleans, None.",
        )
    return dict(checked)


def _machine_settings() -> Any:
    from tandem.core import settings as settings_mod

    return settings_mod.load()


def _unmet(factory: ExecutorFactory, settings: Any) -> tuple[str, ...]:
    if factory.check is None:
        return ()
    try:
        unmet = factory.check(settings if settings is not None else _machine_settings())
        return tuple(str(item) for item in unmet)
    except Exception as exc:
        # A readiness check that cannot run is itself something unmet, not a reason to hide the executor.
        return (f"could not check what it needs: {type(exc).__name__}: {exc}",)
