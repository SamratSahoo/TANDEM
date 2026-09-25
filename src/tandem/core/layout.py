"""Moving profiles written before version 3 into the current layout, and their machine settings into the rig.

Until version 3 a profile was a directory, and every one held a copy of the machine it was collected on::

    profiles/<name>/profile.yml                 the task, the cameras, and planner.options (TiPToP's
                                                robot, perception and tamp; at the top level in version 1)
    profiles/<name>/calibration.json            the cameras' extrinsics
    profiles/<name>/planner-options.<p>.yml     a previous planner's options
    profiles/<name>/trajectories/{eval,success,failure}/

Now a profile is the task alone, ``profiles/<name>.yml``; its trajectories are ``trajectories/<name>/``;
and the robot, the cameras and their extrinsics are said once, in the rig (rig.yml and calibration.json
beside config.toml). ``migrate_all`` gets from the one to the other:

1. Each old profile is split (``split_legacy``): the task's settings stay, the cameras and the planner's
   machine settings (its declared ``RIG_OPTIONS``; for TiPToP, robot and perception) go to the rig, and
   TiPToP's robot address and arm become the rig's own ``robot.host`` and ``robot.type``.
2. Before anything moves, the rig: when this machine has no rig.yml yet, one is written from the active
   profile (or the first one that has cameras). Then, on every run and whether or not rig.yml was there,
   every old profile's extrinsics go into the rig's calibration file for the cameras it has none for --
   never over an entry that is there. An existing rig.yml is never touched: a profile whose settings
   differ from it (extrinsics included) is said to, and its own are kept in the archive. When the rig
   cannot be written, or the extrinsics cannot be kept, NOTHING is moved, and what to fix is said: once a
   directory is archived no run looks in it again.
3. Per profile, in an order a re-run resumes from: its trajectories are renamed into
   ``trajectories/<name>``, its file is written, a previous planner's options go where `tandem planners
   use` finds them, and its old directory is renamed into ``profiles/.migrated/<name>/`` with a
   ``migration.json`` saying what was done. A symlinked directory is linked again at its new place, to
   where it pointed.

Nothing is deleted. Every move is an ``os.rename`` inside the data root -- atomic, and nothing copied. A
profile that cannot be moved at all (it does not validate, or both trajectory directories hold runs) is
reported and left exactly as it was; the others still move. One that failed part way (its trajectories
moved, its directory not archived) is reported as partly moved, with what was done, and a re-run
finishes it.

It runs when asked: `tandem init` runs it, and so does `tandem profile migrate`. Everything else only
says old profiles are there (``pending``), and does not load them.
"""

from __future__ import annotations

import datetime as _dt
import errno
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tandem.core import paths
from tandem.core import profiles as profiles_mod
from tandem.core.errors import ProfileError, TandemError, one_line

#: Where old profile directories are archived, inside profiles/.
ARCHIVE_DIR = ".migrated"
#: What every archived directory gains: a record of what was done with it.
RECORD = "migration.json"

# What version 1 had at the top level that belonged to the only planner there was, and whose it was.
_VERSION_1_SECTIONS = ("robot", "perception", "tamp")
_VERSION_1_PLANNER = "tiptop"
# What version 1 wrote into every profile for on_robot_phase_failure: its default, and the template's.
_VERSION_1_ROBOT_FAILURE = "teleop"


def pending(root: Path | None = None) -> list[str]:
    """The profiles under ``root`` (profiles/) still in the layout before version 3: directories holding a
    profile.yml, or only trajectories (a profile deleted with its data kept). One listdir."""
    root = root if root is not None else profiles_mod.profiles_root()
    try:
        entries = list(root.iterdir())
    except OSError:
        return []
    return sorted(
        entry.name
        for entry in entries
        if profiles_mod.is_name(entry.name)
        and entry.is_dir()
        and ((entry / "profile.yml").is_file() or (entry / "trajectories").is_dir())
    )


def refuse_rig_change() -> None:
    """Refuse to write rig.yml while old profiles that hold this machine's robot, cameras and calibration have
    not set it up yet: a rig.yml written first (`tandem rig set`, `rig edit`, the web's rig card) is one the
    migration never touches, so their robot and cameras would go no further than the archive."""
    from tandem.core import rig as rig_mod

    if rig_mod.exists():
        return
    waiting = pending()
    if not waiting:
        return
    raise TandemError(
        f"{len(waiting)} profile(s) in the old layout ({', '.join(waiting)}) hold this machine's robot, cameras "
        "and calibration: `tandem profile migrate` first.",
        hint="It writes rig.yml from them and keeps their extrinsics; change the rig after. A profile that cannot "
        "be moved says why: fix it, or move its directory out of profiles/.",
    )


def refuse_name(name: str) -> None:
    """Refuse a new profile named as one still in the old layout: the migration would keep the new file, and
    give it the old task's trajectories."""
    if name in pending():
        raise TandemError(
            f"Profile {name!r} is in the old layout ({profiles_mod.profiles_root() / name}): `tandem profile "
            "migrate` first.",
            hint="It moves that profile into the current layout under its name; or choose another name.",
        )


# --------------------------------------------------------------------------- one profile, split


def split_legacy(raw: Mapping[str, Any], *, name: str | None = None) -> tuple[dict, dict, list[str]]:
    """An old profile's settings as ``(profile, rig parts, notes)``: the task's settings in the current
    layout, what of it was the machine's (``robot``, ``cameras``, ``planners.<name>``), and one line for
    each thing a person should know about what moved or was dropped. Pure: reads nothing, writes nothing.
    """
    data, notes = _to_version_2(dict(raw))
    rig: dict[str, Any] = {}

    cameras = data.pop("cameras", None)
    if isinstance(cameras, Mapping):
        rig["cameras"] = dict(cameras)

    planner = dict(data.get("planner") or {}) if isinstance(data.get("planner"), Mapping) else {}
    backend = str(planner.get("backend") or profiles_mod.PlannerSpec().backend)
    options = dict(planner.get("options") or {}) if isinstance(planner.get("options"), Mapping) else {}
    declared = _rig_options_declared(backend)
    if declared is None:
        if options:
            notes.append(
                f"the {backend} planner is not installed here, so which of its planner.options are this "
                "machine's could not be told: all of them were kept in the profile"
            )
    else:
        machine = {key: options.pop(key) for key in list(options) if key in declared}
        # TiPToP's robot block used the generic names for the arm and its address. They are the rig's own
        # now, which every planner reads; the rest of the block is the planner's machine settings. TiPToP's
        # alone: that is the historical fact this encodes, and another planner's own `robot` block may have
        # a `host` that is not the arm's at all.
        robot = machine.get("robot") if backend == _VERSION_1_PLANNER else None
        if isinstance(robot, Mapping):
            robot = dict(robot)
            generic = {key: robot.pop(key) for key in ("type", "host") if key in robot}
            if generic:
                rig["robot"] = generic
            machine["robot"] = robot
        if machine:
            rig["planners"] = {backend: machine}
    planner["options"] = options
    planner.setdefault("backend", backend)
    data["planner"] = planner

    recording = data.get("recording")
    if isinstance(recording, Mapping) and "fps" in recording:
        data["recording"] = {key: value for key, value in recording.items() if key != "fps"}
        notes.append("recording.fps dropped: nothing ever read it (each camera records at the rig's cameras.<role>.fps)")
    stated = data.pop("name", None)
    if name is not None and stated not in (None, name):
        notes.append(f"its name: {stated!r} was dropped; the profile is {name!r}, its file's name")

    hitl = data.get("hitl")
    cache = hitl.get("cache_path") if isinstance(hitl, Mapping) else None
    if cache and not Path(os.path.expanduser(str(cache))).is_absolute():
        notes.append(
            f"hitl.cache_path {cache!r} is now relative to profiles/; the old cache is in the archive, and a "
            "cache miss only asks the model again"
        )
    data["version"] = profiles_mod.LAYOUT_VERSION
    return data, rig, notes


def _to_version_2(data: dict) -> tuple[dict, list[str]]:
    """A version-1 profile's planner sections put under planner.options, as version 2 had them.

    They went to TiPToP's options when the profile planned with TiPToP. When it planned with another
    planner nothing had ever read them -- a planner is built from its own options alone -- so they are
    dropped, and said to be: the one change that loses a setting is the one that must not be quiet.
    """
    try:
        older = int(data.get("version") or 0) < 2
    except (TypeError, ValueError):
        older = True
    notes = _kept_from_version_1(data) if older else []
    moved = [key for key in _VERSION_1_SECTIONS if key in data]
    if not moved:
        return data, notes
    sections = {key: data.pop(key) for key in moved}
    planner = dict(data.get("planner") or {})
    backend = planner.get("backend") or _VERSION_1_PLANNER
    if backend != _VERSION_1_PLANNER:
        notes.append(
            f"{', '.join(moved)} dropped: {_VERSION_1_PLANNER}'s settings, and this profile plans with {backend}"
        )
        return data, notes
    options = dict(planner.get("options") or {})
    both = [key for key in moved if key in options]
    if both:
        raise ValueError(
            f"{', '.join(both)} is set both at the top level (version 1) and under planner.options; keep only "
            "the planner.options one, then migrate again"
        )
    planner["options"] = {**sections, **options}
    planner["backend"] = backend
    data["planner"] = planner
    return data, notes


def _kept_from_version_1(data: dict) -> list[str]:
    """Settings a version-1 profile carries over unchanged that no longer mean what they meant.

    Version 1 defaulted ``hitl.on_robot_phase_failure`` to teleop, stated it in the template, and wrote
    every field on save -- so every version-1 profile on disk says teleop whether or not anyone chose
    it. The default is now abort, the paper's rule (a planning failure is a trial failure; a teleop
    fallback credits the method with trials it did not complete and understates the human effort).
    The value is KEPT -- a migration cannot tell a deliberate choice from the old default, and silently
    changing what a collection run does is worse -- but it is said.
    """
    hitl = data.get("hitl")
    if isinstance(hitl, dict) and hitl.get("on_robot_phase_failure") == _VERSION_1_ROBOT_FAILURE:
        return [
            "hitl.on_robot_phase_failure kept at teleop (version 1's default; the default is now abort, "
            "the paper's rule: set it to abort unless teleop was chosen on purpose)"
        ]
    return []


def _rig_options_declared(backend: str) -> Mapping[str, str] | None:
    """The machine settings ``backend`` declares, ``{}`` when it declares none; None when it cannot be asked."""
    from tandem.planners import registry

    if backend not in registry.available():
        return None
    try:
        declared = registry.rig_options_declared(backend)
    except TandemError:
        return None
    return declared if declared is not None else {}


# --------------------------------------------------------------------------- every profile, moved


@dataclass
class Moved:
    """What happened to one old profile."""

    name: str
    ok: bool = True
    # The profile's new file, when one was written or kept; None for a profile deleted with its data kept.
    file: str | None = None
    trajectories: str | None = None
    archive: str | None = None
    notes: list[str] = field(default_factory=list)
    error: str | None = None
    # What was already done when a later step failed: the profile is partly moved, and a re-run finishes it.
    done: list[str] = field(default_factory=list)
    # Where its old directory still is, when it was not archived.
    left_at: str | None = None


@dataclass
class Report:
    profiles: list[Moved] = field(default_factory=list)
    # "rig.yml written from NAME: ...", or None when rig.yml was there already (or nothing was pending).
    rig: str | None = None
    # "N camera(s)' extrinsics added to ...", or None when the old profiles had none the rig lacked.
    calibration: str | None = None
    notes: list[str] = field(default_factory=list)
    # Why nothing was moved: the rig could not be set up from the old profiles, or their extrinsics could
    # not be kept in it. Every profile is then where it was.
    aborted: ProfileError | None = None

    @property
    def failed(self) -> list[Moved]:
        return [moved for moved in self.profiles if not moved.ok]


@dataclass
class _Plan:
    name: str
    directory: Path
    archive: Path
    version: Any = None
    profile: dict | None = None  # None: a profile deleted with its data kept
    backend: str = ""
    rig: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    # Why it cannot be moved (its profile.yml does not validate). Its extrinsics are still kept.
    error: str | None = None


def migrate_all(*, active: str | None = None, log: Callable[[str], None] | None = None) -> Report:
    """Move every old-layout profile into the current layout (see the module docstring). Never deletes.

    The rig first, and every old profile's extrinsics into it, before anything moves: a profile's
    directory is archived once it is moved, and a re-run never looks in the archive again, so what the
    rig needs of the old profiles is taken from them while they are still where they were. When that
    cannot be done, nothing is moved (``Report.aborted``).
    """
    from tandem.core import rig as rig_mod

    say = log or (lambda _line: None)
    report = Report()
    root = profiles_mod.profiles_root()
    names = pending(root)
    if not names:
        return report

    plans = [_plan(root, name) for name in names]
    usable = [plan for plan in plans if plan.error is None]
    for plan in plans:
        if plan.error is not None:
            report.profiles.append(Moved(plan.name, ok=False, error=plan.error, left_at=str(plan.directory)))
            say(f"{plan.name}: not moved: {one_line(plan.error)}")

    source = _source_of(usable, active)
    seeded = False
    try:
        if not rig_mod.exists() and source is not None:
            report.rig = _seed_rig(source)
            seeded = True
            say(report.rig)
        rig = _loaded_rig()
        # Every old profile's, those that cannot move included: theirs are this machine's cameras too.
        order = ([source] if source is not None else []) + [plan for plan in plans if plan is not source]
        report.calibration, extrinsics = _merge_calibration(rig, order)
    except ProfileError as exc:
        report.aborted = exc
        say(f"nothing was moved: {one_line(exc.message)}")
        report.profiles.sort(key=lambda moved: moved.name)
        return report
    if report.calibration:
        say(report.calibration)

    rig_state: dict[str, str] = {}
    for plan in usable:
        if plan.profile is None:
            continue
        differs, problem = _differences(plan, rig, extrinsics)
        if problem:
            plan.notes.append(
                f"its robot and camera settings are not valid ({problem}): the rig keeps its own, and this "
                f"profile's are in {plan.archive}"
            )
        if differs:
            written = f" (written from {source.name})" if seeded and source is not None else ""
            plan.notes.append(
                f"its machine settings differ from this machine's rig{written}: {', '.join(differs)}. The rig "
                f"keeps its own; this profile's are in {plan.archive}"
            )
        rig_state[plan.name] = f"differs: {', '.join(differs)}" if differs else "same"
    if seeded and source is not None:
        rig_state[source.name] = "seeded rig.yml"

    for plan in usable:
        moved = _move(root, plan, rig_state.get(plan.name))
        report.profiles.append(moved)
        if moved.ok:
            what = f"{moved.file}" if moved.file else "its trajectories only (the profile was deleted)"
            say(f"{plan.name}: moved: {what}; the original is in {moved.archive}")
        else:
            say(f"{plan.name}: {'partly moved' if moved.done else 'not moved'}: {one_line(moved.error or '')}")
    report.profiles.sort(key=lambda moved: moved.name)
    return report


def _plan(root: Path, name: str) -> _Plan:
    """What moving the old profile ``name`` involves. Never raises: a profile that cannot move says why in
    ``error``, and its extrinsics are read all the same."""
    directory = root / name
    plan = _Plan(name, directory, archive=_free(root / ARCHIVE_DIR / name))
    calibration = directory / "calibration.json"
    if calibration.is_file():
        try:
            data = json.loads(calibration.read_text())
            plan.calibration = data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            plan.notes.append(f"its calibration.json could not be read ({exc}); it is kept in the archive")
    source = directory / "profile.yml"
    if not source.is_file():
        return plan  # deleted, with its data kept
    try:
        raw = profiles_mod.read_data(source, name=name)
        plan.version = raw.get("version")
        profile, rig, notes = split_legacy(raw, name=name)
        # Checked as a read would check it: a planner or executor this machine lacks is kept by name.
        profiles_mod._validate({**profile, "name": name}, source=source, absent_ok=True)
    except (TandemError, ValueError, OSError) as exc:
        plan.error = exc.message if isinstance(exc, TandemError) else str(exc)
        return plan
    plan.profile, plan.rig = profile, rig
    plan.backend = str(profile["planner"].get("backend") or "")
    plan.notes.extend(notes)
    return plan


def _source_of(plans: list[_Plan], active: str | None) -> _Plan | None:
    """The profile the rig is written from: the active one, else the first with cameras, else the first.
    None when no old profile has settings to write it from."""
    usable = [plan for plan in plans if plan.profile is not None]
    if not usable:
        return None
    with_cameras = [plan for plan in usable if (plan.rig.get("cameras") or {})]
    for plan in with_cameras:
        if plan.name == active:
            return plan
    return (with_cameras or usable)[0]


def _seed_rig(source: _Plan) -> str:
    """Write rig.yml from one old profile's robot, cameras and planner machine settings. ``ProfileError``,
    naming the setting and the file it came from, when they do not make a valid rig: nothing is written."""
    from tandem.core import rig as rig_mod

    where = source.directory / "profile.yml"
    try:
        rig = rig_mod.update(_rig_changes(source.rig))
    except TandemError as exc:
        located = exc.message.splitlines()[1:] or [exc.message]
        raise ProfileError(
            f"This machine's rig could not be set up from {source.name}'s robot and cameras ({where}):\n"
            + "\n".join(_as_in_profile(line, source) for line in located),
            hint=f"Fix that in {where}, then `tandem profile migrate` again. Nothing was moved or written: every "
            "old profile is where it was.",
        ) from None
    except OSError as exc:
        raise ProfileError(
            f"rig.yml could not be written from {source.name}: {exc}",
            hint="Nothing was moved: every old profile is where it was. `tandem profile migrate` again once "
            f"{rig_mod.paths.rig_file().parent} is writable.",
        ) from None
    cameras = ", ".join(f"{role} {cam.serial}" for role, cam in rig.cameras.configured().items()) or "none"
    return f"rig.yml written from {source.name}: {rig.summary()}; cameras {cameras}"


def _as_in_profile(text: str, plan: _Plan) -> str:
    """A rig setting named in ``text`` as the old profile.yml spelled it: the rig's planners.tiptop.robot.port
    and robot.host were planner.options.robot.port and planner.options.robot.host (version 1: robot.port and
    robot.host at the top). The cameras were where they are."""
    try:
        older = int(plan.version or 0) < 2
    except (TypeError, ValueError):
        older = True
    prefix = "" if older else "planner.options."
    if plan.backend:
        text = re.sub(rf"(?<![\w.]){re.escape(f'planners.{plan.backend}.')}", prefix, text)
    return re.sub(r"(?<![\w.])robot\.(host|type)\b", rf"{prefix}robot.\1", text)


def _loaded_rig():
    """This machine's rig as it is now. A rig.yml that does not load stops the migration: the old profiles'
    extrinsics could not be kept in it, and once moved nothing looks for them again."""
    from tandem.core import rig as rig_mod

    try:
        return rig_mod.load(force=True)
    except TandemError as exc:
        raise ProfileError(
            f"This machine's rig.yml does not load, so the old profiles' cameras and extrinsics cannot be kept "
            f"in it: {exc.message}",
            hint="Fix it (`tandem rig edit`), then `tandem profile migrate` again. Nothing was moved.",
        ) from None


def _merge_calibration(rig: Any, plans: list[_Plan]) -> tuple[str | None, dict]:
    """Every old profile's extrinsics into the rig's calibration file, for the cameras it has none for.

    Never over an entry that is there: the file is this machine's as it is now (put there by hand, by a
    calibration script, or merged by an earlier run). Among the old profiles the first in ``plans`` wins --
    the one the rig was written from -- and an old profile whose own extrinsics differ is said to, with
    where its own are kept (``_differences``). Returns what to say, and the file's extrinsics after.
    """
    path = rig.calibration_file()
    try:
        present = rig.extrinsics()
    except TandemError as exc:
        raise ProfileError(
            f"{path} does not read, so the old profiles' extrinsics cannot be kept in it: {one_line(exc.message)}",
            hint="Fix it (it holds each camera's extrinsics, keyed by serial), then `tandem profile migrate` "
            "again. Nothing was moved.",
        ) from None
    merged = dict(present)
    for plan in plans:
        for serial, extrinsics in plan.calibration.items():
            merged.setdefault(serial, extrinsics)
    added = [serial for serial in merged if serial not in present]
    if not added:
        return None, merged
    try:
        paths.ensure_dir(path.parent)
        paths.write_atomic(path, json.dumps(merged, indent=2) + "\n")
    except OSError as exc:
        raise ProfileError(
            f"The old profiles' extrinsics could not be written into {path}: {exc}",
            hint="Nothing was moved: every old profile is where it was. `tandem profile migrate` again once "
            "it is writable.",
        ) from None
    return f"{len(added)} camera(s)' extrinsics kept in {path} ({', '.join(added)})", merged


def _rig_changes(parts: Mapping[str, Any]) -> dict[str, Any]:
    """An old profile's rig parts as ``rig.update`` takes them: leaf by leaf, so the template's comments stay."""
    changes: dict[str, Any] = {}
    for key, value in (parts.get("robot") or {}).items():
        changes[f"robot.{key}"] = value
    cameras = parts.get("cameras") or {}
    for key, value in cameras.items():
        changes[f"cameras.{key}"] = value
    for backend, block in (parts.get("planners") or {}).items():
        changes[f"planners.{backend}"] = block
    return changes


def _differences(plan: _Plan, rig: Any, extrinsics: Mapping[str, Any]) -> tuple[list[str], str | None]:
    """The rig settings an old profile had that this machine's rig says otherwise, extrinsics included; and
    what is wrong with the old profile's own, when they are not valid on their own."""
    from tandem.core import rig as rig_mod

    differs = [
        f"extrinsics of camera {serial}"
        for serial, own in sorted(plan.calibration.items())
        if serial in extrinsics and extrinsics[serial] != own
    ]
    if not plan.rig:
        return differs, None
    try:
        candidate = rig_mod.Rig.model_validate(
            {key: value for key, value in plan.rig.items() if key in ("robot", "cameras", "planners")}
        )
    except ValueError as exc:
        return differs, one_line(str(exc))
    ours: dict[str, Any] = {}
    ours.update(_flat("robot", candidate.robot.model_dump(mode="json")) if "robot" in plan.rig else {})
    ours.update(_flat("cameras", candidate.cameras.model_dump(mode="json")) if "cameras" in plan.rig else {})
    theirs = {
        **_flat("robot", rig.robot.model_dump(mode="json")),
        **_flat("cameras", rig.cameras.model_dump(mode="json")),
    }
    for backend in (plan.rig.get("planners") or {}):
        try:
            ours.update(_flat(f"planners.{backend}", rig_mod.planner_options(candidate, backend)))
            theirs.update(_flat(f"planners.{backend}", rig_mod.planner_options(rig, backend)))
        except TandemError:
            ours[f"planners.{backend}"] = "?"
    return sorted(key for key, value in ours.items() if theirs.get(key) != value) + differs, None


def _flat(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out.update(_flat(f"{prefix}.{key}", item))
        return out
    return {prefix: value}


#: What an old profile directory held that the migration itself deals with; anything else is said.
_KNOWN = ("profile.yml", "calibration.json", "trajectories", RECORD)


def _move(root: Path, plan: _Plan, rig_state: str | None) -> Moved:
    moved = Moved(plan.name, notes=list(plan.notes))
    old_trajectories = plan.directory / "trajectories"
    new_trajectories = profiles_mod.trajectories_root() / plan.name
    archive = plan.archive
    target = root / f"{plan.name}.yml"
    try:
        # a. The trajectories first: a re-run after a failure further on finds them moved and goes on.
        if old_trajectories.is_dir():
            if new_trajectories.is_dir() and _holds_no_files(new_trajectories):
                # A profile of the same name made before the move (`tandem profile create`, the web's create
                # button) left only its empty eval/success/failure: nothing to merge, so no reason to refuse.
                _remove_empty(new_trajectories)
            if new_trajectories.exists() or new_trajectories.is_symlink():
                if not _holds_no_files(old_trajectories):
                    moved.ok = False
                    moved.error = (
                        f"both {old_trajectories} and {new_trajectories} hold trajectories. Move each run from "
                        f"{old_trajectories}/<status>/ into {new_trajectories}/<status>/ (e.g. `mv "
                        f"{old_trajectories}/success/* {new_trajectories}/success/`), then `tandem profile migrate`"
                    )
                    moved.left_at = str(plan.directory)
                    return moved
                # Every run was moved over by hand, and only the empty status directories are left.
                _remove_empty(old_trajectories)
            else:
                new_trajectories.parent.mkdir(parents=True, exist_ok=True)
                _move_path(old_trajectories, new_trajectories)
                moved.done.append(f"its trajectories are now in {new_trajectories}")
        if new_trajectories.is_dir():
            moved.trajectories = str(new_trajectories)
        # b. The profile's file, unless one of that name was made since.
        if plan.profile is not None:
            if target.exists():
                if _written_by_migration(target, plan.name):
                    moved.notes.append(f"{target.name} was written by an earlier run of the migration")
                else:
                    moved.notes.append(
                        f"kept the existing {target.name} (made since, under the same name); the old "
                        "profile.yml is in the archive"
                    )
            else:
                date = _dt.date.today().isoformat()
                header = (
                    f"# {plan.name}: migrated from profiles/{plan.name}/profile.yml (version {plan.version}) on "
                    f"{date}; the original is in profiles/{ARCHIVE_DIR}/{archive.name}/.\n"
                    "# Task settings only: this machine's robot, cameras and calibration are the rig "
                    "(`tandem rig show`).\n"
                )
                body = profiles_mod._yaml_text(plan.profile, what=f"Profile {plan.name!r}")
                profiles_mod._write_atomic(target, header + body)
                moved.done.append(f"its file is {target}")
                for status in profiles_mod.STATUSES:
                    (new_trajectories / status).mkdir(parents=True, exist_ok=True)
                moved.trajectories = str(new_trajectories)
            moved.file = str(target)
            moved.notes.extend(_restash(plan))
        moved.notes.extend(_others(plan, archive))
        # c. The old directory, archived whole: profile.yml, calibration.json, backups, anything else.
        archive.parent.mkdir(parents=True, exist_ok=True)
        _move_path(plan.directory, archive)
        moved.archive = str(archive)
        # d. What was done, where the original now is.
        record = {
            "migrated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "tandem_version": _version(),
            "from": f"profiles/{plan.name}/",
            "from_version": plan.version,
            "profile": moved.file,
            "trajectories": moved.trajectories,
            "rig": rig_state,
            "notes": moved.notes,
        }
        (archive / RECORD).write_text(json.dumps(record, indent=2) + "\n")
    except _CrossDevice as exc:
        moved.ok = False
        moved.error = f"{exc}: they are on different filesystems, so it was not moved; move it by hand"
    except (OSError, ProfileError) as exc:
        moved.ok = False
        moved.error = exc.message if isinstance(exc, TandemError) else str(exc)
    if not moved.ok and moved.archive is None:
        moved.left_at = str(plan.directory)
    return moved


def _written_by_migration(path: Path, name: str) -> bool:
    """Whether ``path`` is the file an earlier run of the migration wrote for ``name`` (its first line says so)."""
    try:
        with path.open() as fh:
            first = fh.readline()
    except OSError:
        return False
    return first.startswith(f"# {name}: migrated from profiles/{name}/profile.yml")


def _restash(plan: _Plan) -> list[str]:
    """An old profile's set-aside options of another planner (``planner-options.<p>.yml``), put where `tandem
    planners use` restores them from now (``profiles.stash_file``): the task's settings only, since a planner's
    machine settings are the rig's. The original stays in the archive either way."""
    notes = []
    for stash in sorted(plan.directory.glob("planner-options.*.yml")):
        backend = stash.name[len("planner-options.") : -len(".yml")]
        if not profiles_mod.is_name(backend):
            notes.append(f"{stash.name} stays in the archive")
            continue
        target = profiles_mod.stash_file(plan.name, backend)
        if target.exists():
            notes.append(f"{stash.name} stays in the archive: {target} is there already")
            continue
        try:
            options = profiles_mod._read_mapping(stash)
        except ProfileError as exc:
            notes.append(f"{stash.name} stays in the archive: {one_line(exc.message)}")
            continue
        declared = _rig_options_declared(backend) or {}
        machine = sorted(key for key in options if key in declared)
        task = {key: value for key, value in options.items() if key not in declared}
        target.parent.mkdir(parents=True, exist_ok=True)
        text = profiles_mod._yaml_text(task, what=f"{backend}'s planner.options") if task else "{}\n"
        profiles_mod._write_atomic(target, text)
        dropped = ""
        if machine:
            dropped = f" (without {', '.join(machine)}: {'it is' if len(machine) == 1 else 'they are'} the rig's)"
        notes.append(f"{stash.name} is now {target}{dropped}: `tandem planners use {backend}` restores it")
    return notes


def _others(plan: _Plan, archive: Path) -> list[str]:
    """Anything else the old directory holds, which goes to the archive with it: said, because a relative path
    in the profile that pointed at it (a checkpoint, a proposal cache) now resolves beside profiles/."""
    try:
        entries = sorted(entry.name for entry in plan.directory.iterdir())
    except OSError:
        return []
    others = [
        name
        for name in entries
        if name not in _KNOWN
        and not name.startswith(("profile.yml", ".profile.yml", "planner-options."))
    ]
    if not others:
        return []
    return [
        f"{', '.join(others)} {'goes' if len(others) == 1 else 'go'} to the archive ({archive}): a relative path "
        f"to {'it' if len(others) == 1 else 'them'} in the profile now resolves beside profiles/"
    ]


class _CrossDevice(OSError):
    pass


def _rename(source: Path, target: Path) -> None:
    try:
        os.rename(source, target)
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            raise _CrossDevice(f"{source} -> {target}") from exc
        raise


def _move_path(source: Path, target: Path) -> None:
    """``source`` renamed to ``target``; a symlink is made again at ``target``, pointing where it pointed.

    Renamed as it is, a relative link -- trajectories on a bigger disk, ``../../../bigdisk/cloth-traj`` --
    would point somewhere else from its new place, or nowhere, and the data would be found in neither.
    """
    if source.is_symlink():
        os.symlink(os.path.realpath(source), target, target_is_directory=True)
        os.unlink(source)
        return
    _rename(source, target)


def _holds_no_files(path: Path) -> bool:
    """Whether ``path`` is a real directory with nothing under it but real, empty directories."""
    if path.is_symlink():
        return False
    for directory, subdirectories, filenames in os.walk(path):
        if filenames or any(os.path.islink(os.path.join(directory, sub)) for sub in subdirectories):
            return False
    return True


def _remove_empty(path: Path) -> None:
    """Remove a tree of empty directories. ``os.rmdir`` only: a file that appeared since cannot be lost."""
    for directory, _subdirectories, _filenames in os.walk(path, topdown=False):
        os.rmdir(directory)


def _free(path: Path) -> Path:
    """``path``, or ``path-2``, ``path-3``... whichever is not taken."""
    if not path.exists():
        return path
    n = 2
    while path.with_name(f"{path.name}-{n}").exists():
        n += 1
    return path.with_name(f"{path.name}-{n}")


def _version() -> str:
    try:
        from tandem import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - a broken install still migrates
        return "unknown"
