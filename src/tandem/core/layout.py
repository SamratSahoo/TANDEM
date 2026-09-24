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
   the robot's address and type become the rig's own ``robot.host`` and ``robot.type``.
2. When this machine has no rig.yml yet, one is written from the active profile (or the first one that
   has cameras), with every profile's extrinsics merged by serial. An existing rig.yml is never touched:
   a profile whose settings differ from it is said to, and its own are kept in the archive.
3. Per profile, in an order a re-run resumes from: its trajectories are renamed into
   ``trajectories/<name>``, its file is written, and its old directory is renamed into
   ``profiles/.migrated/<name>/`` with a ``migration.json`` saying what was done.

Nothing is deleted. Every move is an ``os.rename`` inside the data root -- atomic, and nothing copied. A
profile that cannot be moved (it does not validate, or both trajectory directories exist) is reported
and left exactly as it was; the others still move.

It runs when asked: `tandem init` runs it, and so does `tandem profile migrate`. Everything else only
says old profiles are there (``pending``), and does not load them.
"""

from __future__ import annotations

import datetime as _dt
import errno
import json
import os
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
        # now, which every planner reads; the rest of the block is the planner's machine settings.
        robot = machine.get("robot")
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


@dataclass
class Report:
    profiles: list[Moved] = field(default_factory=list)
    # "rig.yml written from NAME: ...", or None when rig.yml was there already (or nothing was pending).
    rig: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def failed(self) -> list[Moved]:
        return [moved for moved in self.profiles if not moved.ok]


@dataclass
class _Plan:
    name: str
    directory: Path
    version: Any = None
    profile: dict | None = None  # None: a profile deleted with its data kept
    backend: str = ""
    rig: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def migrate_all(*, active: str | None = None, log: Callable[[str], None] | None = None) -> Report:
    """Move every old-layout profile into the current layout (see the module docstring). Never deletes."""
    from tandem.core import rig as rig_mod

    say = log or (lambda _line: None)
    report = Report()
    root = profiles_mod.profiles_root()
    names = pending(root)
    if not names:
        return report

    plans: list[_Plan] = []
    for name in names:
        try:
            plans.append(_plan(root, name))
        except (TandemError, ValueError, OSError) as exc:
            message = exc.message if isinstance(exc, TandemError) else str(exc)
            report.profiles.append(Moved(name, ok=False, error=message))
            say(f"{name}: not moved: {one_line(message)}")

    rig_state: dict[str, str] = {}
    if not rig_mod.exists():
        report.rig = _seed_rig(plans, active, report)
        if report.rig:
            say(report.rig)
    current = _current_rig(report)
    for plan in plans:
        if plan.profile is not None and plan.rig and current is not None:
            differs = _differences(plan, current)
            rig_state[plan.name] = f"differs: {differs}" if differs else "same"
            if differs:
                plan.notes.append(
                    f"its machine settings differ from this machine's rig.yml ({', '.join(differs)}); the rig was "
                    "left as it is, and the originals are in the archive"
                )
    if report.rig is not None:
        rig_state[_source_of(plans, active).name] = "seeded rig.yml"

    for plan in plans:
        moved = _move(root, plan, rig_state.get(plan.name))
        report.profiles.append(moved)
        if moved.ok:
            what = f"{moved.file}" if moved.file else "its trajectories only (the profile was deleted)"
            say(f"{plan.name}: moved: {what}; the original is in {moved.archive}")
        else:
            say(f"{plan.name}: not moved: {one_line(moved.error or '')}")
    report.profiles.sort(key=lambda moved: moved.name)
    return report


def _plan(root: Path, name: str) -> _Plan:
    directory = root / name
    plan = _Plan(name, directory)
    source = directory / "profile.yml"
    calibration = directory / "calibration.json"
    if calibration.is_file():
        try:
            data = json.loads(calibration.read_text())
            plan.calibration = data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            plan.notes.append(f"its calibration.json could not be read ({exc}); it is kept in the archive")
    for stash in sorted(directory.glob("planner-options.*.yml")):
        plan.notes.append(f"{stash.name} stays in the archive: `tandem planners use` no longer restores it")
    if not source.is_file():
        return plan  # deleted, with its data kept
    raw = profiles_mod.read_data(source, name=name)
    plan.version = raw.get("version")
    profile, rig, notes = split_legacy(raw, name=name)
    plan.profile, plan.rig = profile, rig
    plan.backend = str(profile["planner"].get("backend") or "")
    plan.notes.extend(notes)
    # Checked as a read would check it: a planner or executor this machine lacks is kept by name.
    profiles_mod._validate({**profile, "name": name}, source=source, absent_ok=True)
    return plan


def _source_of(plans: list[_Plan], active: str | None) -> _Plan:
    """The profile the rig is written from: the active one, else the first with cameras, else the first."""
    usable = [plan for plan in plans if plan.profile is not None]
    with_cameras = [plan for plan in usable if (plan.rig.get("cameras") or {})]
    for plan in with_cameras:
        if plan.name == active:
            return plan
    return (with_cameras or usable)[0]


def _seed_rig(plans: list[_Plan], active: str | None, report: Report) -> str | None:
    """Write rig.yml from one old profile, with every profile's extrinsics. None when nothing could be."""
    from tandem.core import rig as rig_mod

    if not any(plan.profile is not None for plan in plans):
        return None
    source = _source_of(plans, active)
    changes = _rig_changes(source.rig)
    calibration = dict(source.calibration)
    for plan in plans:
        for serial, extrinsics in plan.calibration.items():
            if serial not in calibration:
                calibration[serial] = extrinsics
            elif calibration[serial] != extrinsics:
                report.notes.append(
                    f"camera {serial}: {plan.name}'s extrinsics differ from {source.name}'s; {source.name}'s are "
                    f"in the rig, and {plan.name}'s are in its archive"
                )
    try:
        rig = rig_mod.update(changes)
    except TandemError as exc:
        report.notes.append(
            f"rig.yml could not be written from {source.name}: {one_line(exc.message)}. Set it up with "
            "`tandem rig set` (or `tandem init`); every profile's cameras are in its archive"
        )
        return None
    # A calibration.json can be there before rig.yml is (put there by hand, or by a calibration script
    # pointed at it): its extrinsics are this machine's as it is now, so they are kept, and merged into.
    try:
        present = rig.extrinsics()
    except TandemError as exc:
        report.notes.append(
            f"{rig.calibration_file()} does not read ({one_line(exc.message)}), so the old profiles' extrinsics "
            "were not merged into it; they are in their archives"
        )
        calibration, present = {}, {}
    for serial, extrinsics in present.items():
        if serial in calibration and calibration[serial] != extrinsics:
            report.notes.append(
                f"camera {serial}: {rig.calibration_file().name} already had extrinsics for it, and they were kept; "
                "the old profiles' are in their archives"
            )
        calibration[serial] = extrinsics
    if calibration and calibration != present:
        paths.write_atomic(rig.calibration_file(), json.dumps(calibration, indent=2) + "\n")
    cameras = ", ".join(f"{role} {cam.serial}" for role, cam in rig.cameras.configured().items()) or "none"
    return (
        f"rig.yml written from {source.name}: {rig.summary()}; cameras {cameras}; {len(calibration)} "
        f"extrinsics, in {rig.calibration_file()}"
    )


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


def _current_rig(report: Report):
    from tandem.core import rig as rig_mod

    try:
        return rig_mod.load(force=True)
    except TandemError as exc:
        report.notes.append(f"this machine's rig.yml does not load, so no profile was compared with it: {exc.message}")
        return None


def _differences(plan: _Plan, rig: Any) -> list[str]:
    """The rig settings an old profile had that this machine's rig.yml says otherwise."""
    from tandem.core import rig as rig_mod

    ours: dict[str, Any] = {}
    try:
        candidate = rig_mod.Rig.model_validate(
            {key: value for key, value in plan.rig.items() if key in ("robot", "cameras", "planners")}
        )
    except ValueError as exc:
        plan.notes.append(f"its cameras and robot settings are not valid on their own: {one_line(str(exc))}")
        return ["(invalid)"]
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
    return sorted(key for key, value in ours.items() if theirs.get(key) != value)


def _flat(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out.update(_flat(f"{prefix}.{key}", item))
        return out
    return {prefix: value}


def _move(root: Path, plan: _Plan, rig_state: str | None) -> Moved:
    moved = Moved(plan.name, notes=list(plan.notes))
    old_trajectories = plan.directory / "trajectories"
    new_trajectories = profiles_mod.trajectories_root() / plan.name
    archive = _free(root / ARCHIVE_DIR / plan.name)
    target = root / f"{plan.name}.yml"
    try:
        # a. The trajectories first: a re-run after a failure further on finds them moved and goes on.
        if old_trajectories.is_dir():
            if new_trajectories.is_dir() and _holds_no_files(new_trajectories):
                # A profile of the same name made before the move (`tandem profile create`, the web's create
                # button) left only its empty eval/success/failure: nothing to merge, so no reason to refuse.
                _remove_empty(new_trajectories)
            if new_trajectories.exists():
                moved.ok = False
                moved.error = (
                    f"both {old_trajectories} and {new_trajectories} exist; merge them by hand, then migrate again"
                )
                return moved
            new_trajectories.parent.mkdir(parents=True, exist_ok=True)
            _rename(old_trajectories, new_trajectories)
        if new_trajectories.is_dir():
            moved.trajectories = str(new_trajectories)
        # b. The profile's file, unless one of that name was made since.
        if plan.profile is not None:
            if target.exists():
                moved.notes.append(f"kept the existing {target.name}; the old profile.yml is in the archive")
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
                for status in profiles_mod.STATUSES:
                    (new_trajectories / status).mkdir(parents=True, exist_ok=True)
                moved.trajectories = str(new_trajectories)
            moved.file = str(target)
        # c. The old directory, archived whole: profile.yml, calibration.json, backups, stashes.
        archive.parent.mkdir(parents=True, exist_ok=True)
        _rename(plan.directory, archive)
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
    return moved


class _CrossDevice(OSError):
    pass


def _rename(source: Path, target: Path) -> None:
    try:
        os.rename(source, target)
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            raise _CrossDevice(f"{source} -> {target}") from exc
        raise


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
