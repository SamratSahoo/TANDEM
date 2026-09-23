"""`tandem profile` — create and manage collection profiles."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import typer
from rich.syntax import Syntax

from tandem import resources
from tandem.cli import theme
from tandem.core import importers, profiles, render
from tandem.core import settings as settings_mod
from tandem.core.errors import ProfileError

app = typer.Typer(no_args_is_help=True, help="Create and manage collection profiles.")


@app.command("list", help="List every profile.")
def list_profiles(as_json: bool = typer.Option(False, "--json", help="Machine-readable output.")) -> None:
    cfg = settings_mod.load()
    names = profiles.list_names()

    if not names:
        if as_json:
            typer.echo(json.dumps([]))
            return
        theme.info("No profiles yet.")
        theme.next_steps([("tandem init", "set tandem up and create the default profile")])
        return

    rows = []
    for name in names:
        try:
            profile = profiles.load(name)
            counts = _counts(profile)
            rows.append(
                {
                    "name": name,
                    "active": name == cfg.active_profile,
                    "description": profile.description,
                    "prompt": profile.task.prompt,
                    "robot": profile.robot.type,
                    "collected": counts["success"],
                    "target": profile.task.target_episodes,
                    "eval": counts["eval"],
                    "failure": counts["failure"],
                    "path": str(profile.dir()),
                    "valid": True,
                }
            )
        except ProfileError as exc:
            rows.append({"name": name, "active": name == cfg.active_profile, "valid": False, "error": exc.message})

    if as_json:
        typer.echo(json.dumps(rows, indent=2))
        return

    table = theme.table("", "profile", "task", "robot", "collected", "")
    for row in rows:
        marker = "[accent]●[/accent]" if row["active"] else " "
        if not row.get("valid"):
            table.add_row(marker, f"[err]{row['name']}[/err]", "[err]invalid[/err]", "", "", "")
            continue
        progress = _progress_bar(row["collected"], row["target"])
        extra = []
        if row["failure"]:
            extra.append(f"[faint]{row['failure']} failed[/faint]")
        if row["eval"]:
            extra.append(f"[warn]{row['eval']} unlabeled[/warn]")
        table.add_row(
            marker,
            row["name"],
            _truncate(row["prompt"], 42),
            row["robot"],
            progress,
            "  ".join(extra),
        )
    theme.console().print(table)
    theme.info(f"active profile: {cfg.active_profile}", str(cfg.profiles_root()))


@app.command("show", help="Show a profile in full.")
def show(
    name: str = typer.Argument(None, help="Profile name (default: the active one)."),
    tamp_only: bool = typer.Option(False, "--tamp", help="Print only what the planner will receive."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    profile = profiles.load(name)
    cfg = settings_mod.load()
    overrides = render.render_tamp_overrides(profile, runtime_dir=cfg.resolved_runtime_dir())

    if tamp_only:
        text = json.dumps(overrides, indent=2, sort_keys=True)
        if as_json:
            typer.echo(text)
        else:
            theme.heading("planner overrides", "passed as --curobo-overrides")
            if overrides:
                theme.console().print(Syntax(text, "json", theme="ansi_dark", background_color="default"))
            else:
                theme.info("none — the planner runs with stock settings")
        return

    counts = _counts(profile)
    if as_json:
        payload = profile.model_dump(mode="json")
        payload["_resolved"] = {
            "dir": str(profile.dir()),
            "trajectories": counts,
            "tamp_overrides": overrides,
            "warnings": render.check_assets(profile, runtime_dir=cfg.resolved_runtime_dir()),
        }
        typer.echo(json.dumps(payload, indent=2))
        return

    theme.blank()
    theme.heading(profile.name, profile.description)
    theme.kv(
        [
            ("task", profile.task.prompt),
            ("goal", profile.task.goal),
            ("target", f"{counts['success']} / {profile.task.target_episodes} collected"),
            ("directory", profile.dir()),
        ]
    )

    theme.blank()
    theme.heading("robot")
    theme.kv(
        [
            ("type", profile.robot.type),
            ("address", f"{profile.robot.host}:{profile.robot.port}"),
            ("gripper / state", f"{profile.robot.gripper_port} / {profile.robot.state_port}"),
            ("speed", f"{profile.robot.time_dilation_factor:.0%} (time_dilation_factor)"),
        ]
    )

    theme.blank()
    theme.heading("cameras", f"perception reads the {profile.cameras.perception} camera")
    cam_table = theme.table("role", "serial", "type", "resolution", "fps")
    for role, cam in profile.cameras.configured().items():
        label = f"[accent]{role}[/accent]" if role == profile.cameras.perception else role
        cam_table.add_row(label, cam.serial, cam.type, cam.resolution, str(cam.fps))
    theme.console().print(cam_table)

    theme.blank()
    theme.heading("tamp", f"{len(overrides)} override(s)" if overrides else "stock settings")
    if overrides:
        tamp_table = theme.table("setting", "value")
        for key in sorted(overrides):
            tamp_table.add_row(f"[key]{key}[/key]", _render(overrides[key]))
        theme.console().print(tamp_table)

    theme.blank()
    if profile.hitl.enabled:
        theme.heading("phase planning", "on — the task is split into robot and human steps")
        theme.kv(
            [
                ("proposal model", profile.hitl.proposal_model),
                ("verification model", profile.hitl.vlm_model),
                ("retries", f"{profile.hitl.verify_retries} extra attempt(s) at a step that does not verify"),
                (
                    "on failure",
                    "the rollout fails" if profile.hitl.verify_enforced else "recorded, and the run carries on",
                ),
                ("vlm audit trail", profile.hitl.save_vlm_io),
            ]
        )
    else:
        theme.heading("phase planning", "off")
        theme.info(
            "the task is one planner goal; hand the arm over yourself when you need to",
            "set hitl.enabled to let a model split it into steps",
        )

    warnings = render.check_assets(profile, runtime_dir=cfg.resolved_runtime_dir())
    if warnings:
        theme.blank()
        theme.heading("warnings")
        for warning in warnings:
            theme.warn(warning)


@app.command("create", help="Create a profile.")
def create(
    name: str = typer.Argument(..., help="Profile name (lowercase, digits, - and _)."),
    from_profile: str = typer.Option(None, "--from", help="Clone an existing profile."),
    import_from: Path = typer.Option(
        None, "--import-from", help="Import from a hitl-tamp-vla checkout.", exists=True, file_okay=False
    ),
    tamp_config: Path = typer.Option(
        None, "--tamp-config", help="A cfg/tamp/*.yml to import task + TAMP settings from.", exists=True, dir_okay=False
    ),
    prompt: str = typer.Option(None, "--prompt", help="The task prompt."),
    activate: bool = typer.Option(False, "--use", help="Make this the active profile."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing profile.yml."),
) -> None:
    if profiles.exists(name) and not force:
        raise ProfileError(f"Profile {name!r} already exists.", hint="Pass --force to overwrite it.")

    calibration: dict = {}
    notes: list[str] = []
    if from_profile:
        source = profiles.load(from_profile)
        profile = source.model_copy(deep=True)
        profile.name = name
        profile.description = f"copied from {from_profile}"
        calibration = profiles.calibration(source)
        origin = f"cloned from {from_profile}"
    elif import_from or tamp_config:
        profile, calibration, notes = importers.build_profile(
            name, root=import_from, tamp_config=tamp_config
        )
        origin = f"imported from {import_from or tamp_config}"
    else:
        from tandem.cli import planners as planners_cli

        profile = profiles.load_file(resources.path("profile_template.yml"), name=name)
        profile.description = ""
        # The machine's default planner (`tandem planners use NAME --default`), not the template's.
        profile.planner = profiles.PlannerSpec(backend=planners_cli.planner_for_new_profile())
        origin = "from the built-in template"

    if prompt:
        profile.task.prompt = prompt

    path = profiles.save(profile)
    if calibration:
        profile.calibration_file().write_text(json.dumps(calibration, indent=2) + "\n")

    theme.ok(f"Created profile {name!r}", origin)
    theme.info(str(path))
    for note in notes:
        theme.info(note)

    missing = profiles.missing_calibration(profile)
    if missing:
        theme.warn(
            f"No extrinsics for camera serial(s) {', '.join(missing)}",
            f"add them to {profile.calibration_file().name}",
        )

    if activate:
        use(name)


@app.command("use", help="Make a profile the active one.")
def use(name: str = typer.Argument(..., help="Profile name.")) -> None:
    if not profiles.exists(name):
        known = profiles.list_names()
        raise ProfileError(
            f"Profile {name!r} does not exist.",
            hint=f"Known profiles: {', '.join(known)}." if known else "Run `tandem init` first.",
        )
    cfg = settings_mod.load()
    cfg.active_profile = name
    settings_mod.save(cfg)
    theme.ok(f"Active profile is now {name!r}")


@app.command("edit", help="Open a profile in $EDITOR and validate it on save.")
def edit(name: str = typer.Argument(None, help="Profile name (default: the active one).")) -> None:
    cfg = settings_mod.load()
    name = name or cfg.active_profile
    path = profiles.profiles_root() / name / "profile.yml"
    if not path.is_file():
        raise ProfileError(f"Profile {name!r} not found at {path}.")

    backup = path.with_suffix(".yml.bak")
    shutil.copy2(path, backup)
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
    subprocess.call([editor, str(path)])

    try:
        profiles.load(name)
    except ProfileError as exc:
        # Never leave a broken profile in place: a session would fail at warmup, minutes
        # later, with a message about the wrong thing.
        shutil.move(str(backup), str(path))
        raise ProfileError(
            f"Your edit was rejected and the previous profile restored.\n\n{exc.message}",
            hint="Re-run `tandem profile edit` and fix the reported line.",
        ) from exc
    backup.unlink(missing_ok=True)
    theme.ok(f"Profile {name!r} is valid", str(path))


@app.command("delete", help="Delete a profile (its trajectories are kept by default).")
def delete(
    name: str = typer.Argument(..., help="Profile name."),
    purge: bool = typer.Option(False, "--purge", help="Also delete every collected trajectory. Irreversible."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
) -> None:
    profile = profiles.load(name)
    counts = _counts(profile)
    total = sum(counts.values())

    if purge:
        theme.warn(f"This deletes {total} trajector{'y' if total == 1 else 'ies'} from disk. There is no undo.")
    else:
        theme.info(f"{total} trajector{'y' if total == 1 else 'ies'} will be kept at {profile.trajectories_dir()}")

    if not yes and not typer.confirm(f"Delete profile {name!r}?", default=False):
        raise typer.Abort()

    cfg = settings_mod.load()
    profiles.delete(name, keep_data=not purge)
    theme.ok(f"Deleted profile {name!r}", "data purged" if purge else "data kept")

    if cfg.active_profile == name:
        remaining = profiles.list_names()
        cfg.active_profile = remaining[0] if remaining else "default"
        settings_mod.save(cfg)
        theme.info(f"Active profile is now {cfg.active_profile!r}")


@app.command("path", help="Print a profile's directory.")
def path_(name: str = typer.Argument(None, help="Profile name (default: the active one).")) -> None:
    typer.echo(str(profiles.load(name).dir()))


# --------------------------------------------------------------------------- helpers


def _counts(profile: profiles.Profile) -> dict[str, int]:
    from tandem.core import trajectories

    return trajectories.counts(profile)


def _progress_bar(done: int, target: int, width: int = 14) -> str:
    if target <= 0:
        return f"{done}"
    filled = min(width, round(width * done / target))
    colour = "ok" if done >= target else "accent"
    bar = f"[{colour}]{'█' * filled}[/{colour}][faint]{'░' * (width - filled)}[/faint]"
    return f"{bar} [faint]{done}/{target}[/faint]"


def _truncate(text: str, width: int) -> str:
    text = text or ""
    return text if len(text) <= width else text[: width - 1] + "…"


def _render(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)
