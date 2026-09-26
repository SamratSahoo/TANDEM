"""`tandem rig` — this machine's robot, cameras and calibration, which every profile shares.

    tandem rig show [--json]          the robot, the cameras (and which are calibrated), each planner's settings
    tandem rig set KEY VALUE          robot.host NUC_ADDRESS · cameras.hand.serial SERIAL ·
                                      cameras.external_2 null · planners.tiptop.perception.m2t2.url http://HOST:8123
    tandem rig edit                   rig.yml in $EDITOR, validated on save
    tandem rig path [--calibration]   where rig.yml is, or its calibration file

A profile is a task, and says nothing about the machine; `tandem profile` is for those. tandem's own
preferences (the active profile, the data root, teleop) are `tandem config`.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from typing import Any

import typer
from rich.markup import escape

from tandem.cli import theme
from tandem.core import paths, profiles
from tandem.core import rig as rig_mod
from tandem.core import settings as settings_mod
from tandem.core.errors import RigInvalid, TandemError

app = typer.Typer(
    no_args_is_help=True,
    help="This machine's robot, cameras and calibration, shared by every profile.",
)


# --------------------------------------------------------------------------- what is there


def show_payload() -> dict:
    """The rig as `tandem rig show --json` and the web's rig card read it. Raises RigInvalid for a bad file."""
    from tandem.planners import registry

    rig = rig_mod.load()
    written = _written()
    configured = rig.cameras.configured()
    try:
        missing = rig.missing_calibration()
        calibration_problem = None
    except RigInvalid as exc:
        missing, calibration_problem = [cam.serial for cam in configured.values()], exc.message
    planners: dict[str, Any] = {}
    for name in registry.available():
        try:
            declared = registry.rig_options_declared(name)
        except TandemError as exc:
            if name in rig.planners:
                planners[name] = {
                    "declared": None,
                    "options": rig.planners[name],
                    "installed": True,
                    "problem": exc.message,
                    "set": sorted(_keys(written, name)),
                }
            continue
        if not declared and name not in rig.planners:
            continue
        try:
            options, problem = rig_mod.planner_options(rig, name), None
        except TandemError as exc:
            options, problem = dict(rig.planners.get(name) or {}), exc.message
        planners[name] = {
            "declared": dict(declared or {}),
            "options": options,
            "installed": True,
            "problem": problem,
            "set": sorted(_keys(written, name)),
        }
    for name, block in rig.planners.items():
        if name not in planners:
            planners[name] = {
                "declared": None,
                "options": block,
                "installed": False,
                "problem": None,
                "set": sorted(_keys(written, name)),
            }
    return {
        "file": str(rig.file()),
        "exists": rig_mod.exists(),
        "rig": rig.model_dump(mode="json"),
        "calibration_file": str(rig.calibration_file()),
        "calibrated": [cam.serial for cam in configured.values() if cam.serial not in missing],
        "missing_calibration": missing,
        "calibration_problem": calibration_problem,
        "perception_missing": rig.cameras.perception_missing() if configured else None,
        "planners": planners,
    }


def _written() -> Mapping[str, Any]:
    """rig.yml as written, before any default is filled in: what `show` marks as set rather than default."""
    from ruamel.yaml import YAML

    if not rig_mod.exists():
        return {}
    try:
        data = YAML(typ="safe").load(paths.rig_file().read_text())
    except Exception:
        return {}
    return data if isinstance(data, Mapping) else {}


def _keys(written: Mapping[str, Any], name: str) -> set[str]:
    """The settings of ``planners.<name>`` the file states, as dotted paths to each value."""
    block = (written.get("planners") or {}).get(name) if isinstance(written.get("planners"), Mapping) else None
    return set(_flat(block)) if isinstance(block, Mapping) else set()


def _flat(value: Any, prefix: str = "") -> dict[str, Any]:
    """A nested block as ``{"perception.m2t2.url": ...}``: one row per setting, as `tandem rig set` names it."""
    if isinstance(value, Mapping) and value:
        out: dict[str, Any] = {}
        for key, item in value.items():
            out.update(_flat(item, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: value} if prefix else {}


@app.command("show", help="This machine's robot, cameras and calibration, and each planner's machine settings.")
def show(as_json: bool = typer.Option(False, "--json", help="Machine-readable output.")) -> None:
    payload = show_payload()
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return
    rig = payload["rig"]
    theme.blank()
    theme.heading("rig", payload["file"] if payload["exists"] else f"{payload['file']} (not written yet: defaults)")
    theme.kv([("robot", rig["robot"]["type"]), ("address", rig["robot"]["host"])])

    theme.blank()
    theme.heading("cameras", f"perception reads the {rig['cameras']['perception']} camera")
    cameras = {role: rig["cameras"].get(role) for role in rig_mod.ROLES}
    if any(cameras.values()):
        table = theme.table("role", "serial", "type", "resolution", "fps", "calibrated")
        for role, cam in cameras.items():
            if not cam:
                continue
            label = f"[accent]{role}[/accent]" if role == rig["cameras"]["perception"] else role
            calibrated = "[ok]yes[/ok]" if cam["serial"] in payload["calibrated"] else "[err]no[/err]"
            table.add_row(label, cam["serial"], cam["type"], cam["resolution"], str(cam["fps"]), calibrated)
        theme.console().print(table)
    else:
        theme.info("none: this machine can browse trajectories but not collect", "`tandem rig set cameras.ROLE.serial SERIAL`")
    if payload["perception_missing"]:
        theme.warn(payload["perception_missing"])

    total = len([cam for cam in cameras.values() if cam])
    theme.blank()
    theme.kv(
        [
            ("calibration", payload["calibration_file"]),
            ("calibrated", f"{len(payload['calibrated'])} of {total} camera(s)"),
        ]
    )
    if payload["calibration_problem"]:
        theme.warn(payload["calibration_problem"])

    for name, planner in payload["planners"].items():
        theme.blank()
        subtitle = "not installed here: kept as written" if not planner["installed"] else "this planner's machine settings"
        theme.heading(f"planners.{escape(name)}", subtitle)
        rows = []
        for key, value in _flat(planner["options"]).items():
            shown = _shown(value)
            if planner["installed"] and key not in planner["set"]:
                shown += "  (default)"
            rows.append((escape(key), shown))
        if rows:
            theme.kv(rows)
        if planner["problem"]:
            theme.warn(planner["problem"])
    theme.blank()
    theme.info("`tandem rig set KEY VALUE` changes one setting; `tandem rig edit` opens the file.")


def _shown(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, list):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)


# --------------------------------------------------------------------------- changing it


@app.command("set", help="Change one rig setting, e.g. `tandem rig set robot.host NUC_ADDRESS`. `null` removes one.")
def set_(
    key: str = typer.Argument(..., help="Dotted key: robot.host, cameras.hand.serial, planners.tiptop.robot.port, ..."),
    value: str = typer.Argument(..., help="The new value, as YAML (a serial stays text); null removes the key."),
) -> None:
    from tandem.core import layout

    coerced = rig_mod.coerce(key, value)
    layout.refuse_rig_change()
    rig = rig_mod.update({key: coerced})
    shown = "removed" if coerced is None else json.dumps(coerced) if not isinstance(coerced, str) else coerced
    theme.ok(f"{key} = {shown}", str(rig.file()))
    warn_planner_rig_checks()


@app.command("edit", help="Open rig.yml in $EDITOR and validate it on save (the previous file is restored if not).")
def edit() -> None:
    from tandem.cli.editor import open_in_editor
    from tandem.core import layout

    layout.refuse_rig_change()
    path = paths.rig_file()
    created = not path.is_file()
    if created:
        paths.ensure_dir(path.parent)
        path.write_text(rig_mod.template())
    backup = path.with_name(f"{path.name}.bak")
    shutil.copy2(path, backup)
    try:
        open_in_editor(path)
    except TandemError:
        _restore(path, backup, created)
        raise
    try:
        rig_mod.parse_text(path.read_text(), source=path)
    except RigInvalid as exc:
        # The person's text is kept: restored over, it was gone, and the edit had to be made again from nothing.
        rejected = path.with_name(f"{path.name}.rejected")
        shutil.copy2(path, rejected)
        _restore(path, backup, created)
        restored = "rig.yml was not written" if created else "the previous rig restored"
        raise RigInvalid(
            f"Your edit was rejected and {restored}; your text is kept in {rejected}.\n\n{exc.message}",
            hint="Re-run `tandem rig edit` and fix the reported line (your rejected text is in "
            f"{rejected.name}, beside it).",
        ) from exc
    backup.unlink(missing_ok=True)
    rig = rig_mod.load(force=True)
    rig_mod.ensure_calibration_file(rig)
    theme.ok("The rig is valid", str(path))
    warn_planner_rig_checks()


def _restore(path, backup, created: bool) -> None:
    if created:
        path.unlink(missing_ok=True)
        backup.unlink(missing_ok=True)
    else:
        shutil.move(str(backup), str(path))


@app.command("path", help="Print where rig.yml is (or, with --calibration, its calibration file).")
def path_(
    calibration: bool = typer.Option(False, "--calibration", help="The calibration file instead."),
) -> None:
    if calibration:
        typer.echo(str(rig_mod.load().calibration_file()))
    else:
        typer.echo(str(paths.rig_file()))


def rig_failures() -> list:
    """What stops collection on this rig: a camera perception reads that is not configured, and the FAILs the
    active profile's planner has about the rig (an arm TiPToP does not drive, a camera it opens that is not
    there, missing extrinsics). Asked as `tandem doctor` asks, touching no hardware; the softer findings are
    doctor's to list."""
    from tandem.cli import doctor
    from tandem.core import probe
    from tandem.planners import registry

    checks = [check for check in doctor.rig_checks() if check.name != "rig"]
    profile = _profile_to_ask()
    if profile is not None:
        checks += registry.doctor_checks(
            profile.planner.backend, profile, settings=settings_mod.load(), probe_hardware=False
        )
    return [check for check in checks if check.group == "rig" and check.state == probe.FAIL]


def warn_planner_rig_checks() -> None:
    """``rig_failures``, said the moment the rig is changed."""
    for check in rig_failures():
        theme.warn(f"{check.name}: {check.detail}", check.hint or None)


def _profile_to_ask():
    """The active profile, or -- with none that loads -- a bare one on the machine's default planner."""
    try:
        return profiles.load()
    except TandemError:
        pass
    try:
        return profiles.Profile(planner=profiles.planner_spec(settings_mod.load().default_planner))
    except (TandemError, ValueError):
        return None
