"""`tandem profile` — create and manage collection profiles.

A profile is a task: one YAML file in profiles/, and its trajectories in trajectories/<name>/. This
machine's robot, cameras and calibration are the rig's (`tandem rig`), which every profile shares.
"""

from __future__ import annotations

import json
import shutil

import typer
from rich.syntax import Syntax

from tandem.cli import theme
from tandem.core import profiles
from tandem.core import settings as settings_mod
from tandem.core.errors import ProfileError, TandemError
from tandem.planners import registry

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
        theme.next_steps(
            [
                ("tandem init", "set tandem up, and add the paper's five tasks"),
                ('tandem profile create NAME --prompt "..."', "make one of your own"),
            ]
        )
        return

    rows = []
    for name in names:
        try:
            # Read-only: a profile naming a planner or executor this machine lacks is still listed (and
            # its trajectories counted), with what is missing said beside it.
            profile = profiles.load(name, require_installed=False)
            counts = _counts(profile)
            rows.append(
                {
                    "name": name,
                    "active": name == cfg.active_profile,
                    "description": profile.description,
                    "prompt": profile.task.prompt,
                    "planner": profile.planner.backend,
                    "collected": counts["success"],
                    "target": profile.task.target_episodes,
                    "eval": counts["eval"],
                    "failure": counts["failure"],
                    "path": str(profile.file()),
                    "file": str(profile.file()),
                    "trajectories": str(profile.trajectories_dir()),
                    "valid": True,
                    "missing": missing_here(profile),
                }
            )
        except ProfileError as exc:
            rows.append({"name": name, "active": name == cfg.active_profile, "valid": False, "error": exc.message})

    if as_json:
        typer.echo(json.dumps(rows, indent=2))
        return

    table = theme.table("", "profile", "task", "planner", "collected", "")
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
        if row["missing"]:
            extra.append(f"[warn]{', '.join(row['missing'])} not installed here[/warn]")
        table.add_row(
            marker,
            row["name"],
            _truncate(row["prompt"], 42),
            row["planner"],
            progress,
            "  ".join(extra),
        )
    theme.console().print(table)
    theme.info(f"active profile: {cfg.active_profile}", str(cfg.profiles_root()))
    for row in rows:
        if not row.get("valid"):
            from tandem.core.errors import one_line

            theme.fail(f"{row['name']}: {one_line(row['error'])}")


def missing_here(profile: profiles.Profile) -> list[str]:
    """What a profile names that this machine does not have: "planner shelfbot", "executor policybot".

    A profile read for browsing may name either (``profiles.load(require_installed=False)``); it
    collects only where both are installed, so a listing says which is missing rather than nothing.
    """
    from tandem.executors import base as executors

    missing = []
    if profile.planner.backend not in registry.available():
        missing.append(f"planner {profile.planner.backend}")
    if profile.hitl.human_executor not in executors.available():
        missing.append(f"executor {profile.hitl.human_executor}")
    return missing


@app.command("show", help="Show a profile in full.")
def show(
    name: str = typer.Argument(None, help="Profile name (default: the active one)."),
    receives_only: bool = typer.Option(
        False,
        "--planner",
        "--tamp",
        help="Print only what the planner will receive from its options (--tamp is the older name).",
    ),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    # Read-only: a profile collected with a plugin this machine does not have is still shown, whole.
    profile = profiles.load(name, require_installed=False)
    cfg = settings_mod.load()
    backend = profile.planner.backend
    # The planner's own account of its options: this command knows no planner's schema.
    try:
        view = registry.describe_options(backend, profile, settings=cfg)
    except TandemError as exc:
        if receives_only:
            raise  # what a planner that will not load receives is not something to guess at
        # The rest of the profile is still worth showing -- this is where a broken planner gets
        # noticed -- so its options are shown as written, with the planner's own error among the warnings.
        from dataclasses import replace

        from tandem.planners.base import OptionsView

        view = replace(OptionsView.generic(profile.planner.options), warnings=(exc.message,))

    if receives_only:
        text = json.dumps(dict(view.receives), indent=2, sort_keys=True, default=str)
        if as_json:
            typer.echo(text)
        else:
            theme.heading(f"what {backend} receives", view.receives_note)
            if view.receives:
                theme.console().print(Syntax(text, "json", theme="ansi_dark", background_color="default"))
            else:
                theme.info("nothing — the planner runs with its stock settings")
        return

    counts = _counts(profile)
    if as_json:
        payload = profile.model_dump(mode="json")
        payload["_resolved"] = {
            "file": str(profile.file()),
            "trajectories_dir": str(profile.trajectories_dir()),
            "trajectories": counts,
            "planner": view.to_dict(),
            "warnings": list(view.warnings),
        }
        typer.echo(json.dumps(payload, indent=2, default=str))
        return

    theme.blank()
    theme.heading(profile.name, profile.description)
    theme.kv(
        [
            ("task", profile.task.prompt),
            ("goal", profile.task.goal),
            ("target", f"{counts['success']} / {profile.task.target_episodes} collected"),
            ("file", profile.file()),
            ("trajectories", profile.trajectories_dir()),
        ]
    )

    theme.blank()
    theme.heading("planner", _planner_title(backend))
    for section in view.sections:
        theme.blank()
        theme.heading(f"  {section.title}", section.subtitle)
        if section.rows:
            rows = theme.table("setting", "value")
            for key, value in section.rows:
                rows.add_row(f"[key]{key}[/key]", value)
            theme.console().print(rows)

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

    missing = missing_here(profile)
    if view.warnings or missing:
        theme.blank()
        theme.heading("warnings")
        for warning in view.warnings:
            theme.warn(warning)
        for what in missing:
            theme.warn(
                f"{what} is not installed on this machine",
                "this profile can be browsed and exported here, and collects where it is installed",
            )


@app.command(
    "create",
    help='Create a profile: the paper\'s settings with your task (--prompt "..."), or a copy of another (--from).',
)
def create(
    name: str = typer.Argument(..., help="Profile name (lowercase, digits, - and _)."),
    prompt: str = typer.Option(None, "--prompt", help="The task, in words: what the robot and you are to do."),
    from_profile: str = typer.Option(
        None,
        "--from",
        help="Copy this profile instead: one of yours, or one of the paper's five, such as "
        "store-bread-in-closed-box.",
    ),
    activate: bool = typer.Option(False, "--use", help="Make it the active profile."),
    force: bool = typer.Option(False, "--force", help="Replace a profile of that name (its trajectories are kept)."),
) -> None:
    from tandem.cli import planners as planners_cli

    if prompt is None and from_profile is None and theme.is_tty():
        prompt = typer.prompt("  Task prompt").strip()
    # The machine's default planner (`tandem planners default NAME`), checked by name before anything is
    # written: a default naming a planner this machine no longer has is said here, not by every command after.
    planner = planners_cli.planner_for_new_profile() if from_profile is None else None
    profile = profiles.create(name, source=from_profile, prompt=prompt, planner=planner, force=force)

    origin = f"a copy of {from_profile}" if from_profile else "the paper's settings"
    theme.ok(f"Created profile {name!r}", origin)
    theme.info(str(profile.file()))
    warn_planner_profile_checks(profile)
    if activate:
        use(name)
    else:
        theme.next_steps(
            [
                (f"tandem profile edit {name}", "change any of its settings"),
                (f"tandem collect {name}", "collect with it"),
            ]
        )


def warn_planner_profile_checks(profile: profiles.Profile) -> None:
    """What the profile's planner says will stop it collecting with the profile it was just given: missing
    extrinsics for TiPToP, say. Asked of the planner, as `tandem doctor` asks it, because only the planner
    knows what it reads -- camera extrinsics mean nothing to a planner that never localises from them, and
    telling its author to calibrate cameras it ignores sends them off to do exactly that. Only the
    planner's failures, about the profile and the rig it will collect on: its softer findings are
    `tandem doctor`'s to list."""
    from tandem.core import probe

    checks = registry.doctor_checks(
        profile.planner.backend, profile, settings=settings_mod.load(), probe_hardware=False
    )
    for check in checks:
        if check.group in ("profile", "rig") and check.state == probe.FAIL:
            theme.warn(f"{check.name}: {check.detail}", check.hint or None)


@app.command(
    "migrate",
    help="Move profiles written before version 3 (a directory each) into the current layout, and their "
    "robot and cameras into this machine's rig. `tandem init` does it too.",
    # A one-time move that `tandem init` makes anyway: kept out of the listing, and in the docs.
    hidden=True,
)
def migrate() -> None:
    """Every old-layout profile, moved: see ``tandem.core.layout``. Nothing is deleted; one profile that
    cannot be moved does not stop the others, and the command fails at the end if any could not."""
    report = run_migration()
    if report is None:
        theme.info("No profiles in the old layout.", str(profiles.profiles_root()))
        return
    raise_if_incomplete(report)


def raise_if_incomplete(report) -> None:
    """The migration's failure, if it had one: nothing moved (the rig could not be set up), or some profiles
    not moved or only partly."""
    from tandem.core import layout

    if report.aborted is not None:
        raise report.aborted
    if report.failed:
        partly = [moved.name for moved in report.failed if moved.done]
        raise ProfileError(
            f"{len(report.failed)} profile(s) could not be moved: {', '.join(m.name for m in report.failed)}.",
            hint="What is wrong with each is said above. "
            + (
                f"{', '.join(partly)} {'is' if len(partly) == 1 else 'are'} partly moved (what was done is said "
                "above), and the rest left exactly as they were. "
                if partly
                else "Each one is left exactly as it was. "
            )
            + "`tandem profile migrate` again once it is fixed finishes the move. (The moved ones are archived "
            f"in {profiles.profiles_root() / layout.ARCHIVE_DIR}.)",
        )


def run_migration():
    """Move the old-layout profiles, saying what happened to each. None when there were none. For `init` too.

    Settings written before version 3 left the active profile unset when it was ``default``, the default then;
    it is still the active one after the move when nothing else was chosen.
    """
    from tandem.core import layout

    if not layout.pending():
        return None
    cfg = settings_mod.load()
    report = layout.migrate_all(active=cfg.active_profile or "default")
    if report.rig:
        theme.ok("Rig written from the old profiles", report.rig)
    if report.calibration:
        theme.ok("Extrinsics kept", report.calibration)
    for note in report.notes:
        theme.warn(note)
    for moved in report.profiles:
        if not moved.ok:
            _say_not_moved(moved)
            continue
        if moved.file:
            theme.ok(f"{moved.name}: moved", moved.file)
        else:
            theme.ok(f"{moved.name}: its trajectories moved", moved.trajectories or "")
            theme.info(
                "  the profile itself had been deleted: "
                f'`tandem profile create {moved.name} --prompt "..."` makes them a profile\'s again'
            )
        theme.info(f"  the original is in {moved.archive}")
        for note in moved.notes:
            theme.info(f"  {note}")
    if report.aborted is not None:
        theme.fail("Nothing was moved", _one_line(report.aborted.message))
    elif not cfg.active_profile and profiles.exists("default"):
        cfg.active_profile = "default"
        settings_mod.save(cfg)
    return report


def _say_not_moved(moved) -> None:
    if moved.done:
        theme.fail(f"{moved.name}: partly moved", "; ".join(moved.done))
        theme.info(
            f"  its old directory is still at {moved.left_at}: {_one_line(moved.error or '')}. "
            "`tandem profile migrate` again finishes it."
        )
    else:
        theme.fail(f"{moved.name}: not moved", _one_line(moved.error or ""))


def _one_line(message: str) -> str:
    from tandem.core.errors import one_line

    return one_line(message)


def _planner_title(backend: str) -> str:
    try:
        return registry.info(backend).title
    except TandemError as exc:
        return f"{backend} — {exc.message}"


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
    path = profiles._existing_file(name)

    from tandem.cli.editor import open_in_editor

    backup = path.with_suffix(".yml.bak")
    # The names the profile had before the edit may be absent from this machine and stay accepted: an
    # edit of its prompt on a laptop must not be refused over the planner it was collected with.
    before = profiles.names_in_file(path)
    shutil.copy2(path, backup)
    try:
        open_in_editor(path)
    except TandemError:
        backup.unlink(missing_ok=True)  # the editor never ran: the file is as it was
        raise

    try:
        profiles.load_file(path, name=name, keep_absent=before)
    except ProfileError as exc:
        # Never leave a broken profile in place: a session would fail at warmup, minutes
        # later, with a message about the wrong thing. But keep what the person wrote: restored over, it
        # was gone, and the edit had to be made again from nothing.
        rejected = path.with_name(f"{path.name}.rejected")
        shutil.copy2(path, rejected)
        shutil.move(str(backup), str(path))
        raise ProfileError(
            f"Your edit was rejected and the previous profile restored; your text is kept in {rejected}."
            f"\n\n{exc.message}",
            hint=f"Re-run `tandem profile edit {name}` and fix the reported line (your rejected text is in "
            f"{rejected.name}, beside it).",
        ) from exc
    backup.unlink(missing_ok=True)
    theme.ok(f"Profile {name!r} is valid", str(path))


@app.command("delete", help="Delete a profile (its trajectories are kept by default).")
def delete(
    name: str = typer.Argument(..., help="Profile name."),
    purge: bool = typer.Option(False, "--purge", help="Also delete every collected trajectory. Irreversible."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
) -> None:
    # Read-only: deleting a profile must not need the plugin it was collected with.
    profile = profiles.load(name, require_installed=False)
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


@app.command("path", help="Print a profile's file.")
def path_(name: str = typer.Argument(None, help="Profile name (default: the active one).")) -> None:
    typer.echo(str(profiles.load(name, require_installed=False).file()))


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
