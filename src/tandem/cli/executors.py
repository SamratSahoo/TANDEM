"""`tandem executors` — who carries out a human phase: list them, choose one for a profile.

A human phase is carried out by an executor (pi_omega_Delta in the paper): a person driving the arm
through teleop is the one that ships, and a package adds another through the
``tandem.human_executors`` entry point. A profile names the one it uses in ``hitl.human_executor``::

    tandem executors list             every executor, whether it is ready here, which one is in use
    tandem executors use NAME         make a profile's human phases run with it
    tandem executors install teleop   build the teleop driver's runtime, and turn teleop on

An executor is a Python package, and what it needs on this machine (a policy server, say) is its own
requirements list, which the listing shows with whatever is still unmet -- the same way ``tandem
planners`` shows whether a planner's runtime is built. Teleop is the one with a runtime tandem builds:
DROID's workstation side (``tandem.teleop.recipe``). As with planners, choosing an executor that is not
ready yet is allowed and warned about, never refused: a profile is edited wherever it is edited, and
collected with on the machine that has the hardware.

The payloads here are also what the web UI reads (``server/routes/planners.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.markup import escape
from rich.text import Text

from tandem.cli import theme
from tandem.core import settings as settings_mod
from tandem.core.errors import ProfileError, one_line

app = typer.Typer(no_args_is_help=True, help="Who carries out a human phase.")

_PROFILE = typer.Option(None, "--profile", "-p", help="A profile other than the active one.")
_AS_JSON = typer.Option(False, "--json", help="Machine-readable output.")

READY = "ready"
NEEDS_SETUP = "needs setup"
BROKEN = "broken"
STATUSES = (READY, NEEDS_SETUP, BROKEN)

_STYLE = {READY: "ok", NEEDS_SETUP: "warn", BROKEN: "err"}


def profile_executor(profile_name: str | None = None) -> dict:
    """The profile's name, the executor it names, whether its phase planning is on, and why that could
    not be read. Never raises, for the reason ``planners.profile_planner`` does not."""
    from tandem.core import profiles

    name = profile_name or settings_mod.load().active_profile
    out: dict[str, Any] = {"profile": name, "executor": None, "phase_planning": None, "problem": None}
    if not profiles.exists(name):
        out["problem"] = f"there is no profile named {name!r} yet" if name else "no profile is active"
        return out
    try:
        profile = profiles.load(name)
    except ProfileError as exc:
        out["problem"] = one_line(exc.message)
        return out
    out.update(executor=profile.hitl.human_executor, phase_planning=profile.hitl.enabled)
    return out


def executor_row(info: Any, *, active: bool) -> dict:
    """One executor as a listing shows it: ``ExecutorInfo`` plus a status and whether it is in use."""
    if info.error:
        status, detail = BROKEN, info.error
    elif info.unmet:
        status, detail = NEEDS_SETUP, "; ".join(info.unmet)
    else:
        status, detail = READY, "ready"
    return {
        "kind": "executor",
        **info.to_dict(),
        "ok": info.error is None,
        "status": status,
        "detail": detail,
        "active": active,
    }


def catalog_payload(*, profile_name: str | None = None) -> dict:
    """Every human executor, as `tandem executors list --json` and the web UI show it. JSON-safe."""
    from tandem.core import profiles
    from tandem.executors import base as executors

    # Scanned afresh: in the web server's long-lived process, a package installed since the last
    # listing would otherwise stay invisible until a restart.
    executors.refresh()
    cfg = settings_mod.load()
    current = profile_executor(profile_name)
    rows = [
        executor_row(info, active=info.name == current["executor"])
        for info in executors.catalog(settings=cfg)
    ]
    return {
        "profile": current["profile"],
        "profile_executor": current["executor"],
        "phase_planning": current["phase_planning"],
        "profile_problem": current["problem"],
        "profile_exists": profiles.exists(current["profile"]),
        "executors": rows,
    }


def use_executor(name: str, *, profile_name: str | None = None) -> dict:
    """Make a profile's human phases run with ``name``. What changed, and whether it is ready here.

    An unknown name, or an installed executor that will not load, is refused with the registry's own
    error. One whose requirements are unmet on this machine is not: the result lists them.

    Works from the profile's file as written (``profiles.set_human_executor``), so it also repairs a
    profile naming an executor this machine no longer has -- the command that profile's error points at.
    """
    from tandem.core import profiles
    from tandem.executors import base as executors

    executors.refresh()
    info = executors.info(name)
    target = profile_name or settings_mod.load().active_profile
    switched = profiles.set_human_executor(target, name)
    return {
        "executor": name,
        "display_name": info.display_name,
        "profile": target,
        "previous": switched["previous"],
        "changed": switched["changed"],
        "status": READY if not info.unmet else NEEDS_SETUP,
        "unmet": list(info.unmet),
        "phase_planning": switched["profile"].hitl.enabled,
    }


@app.command("list", help="Every human executor, whether it is ready on this machine, and which is in use.")
def list_executors(as_json: bool = _AS_JSON, profile_name: str = _PROFILE) -> None:
    payload = catalog_payload(profile_name=profile_name)
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    table = theme.table("", "executor", "status", "summary")
    for row in payload["executors"]:
        marker = Text("●", style="accent") if row["active"] else Text(" ")
        name = Text(row["name"], style="bold")
        if row["display_name"] and row["display_name"] != row["name"]:
            name.append(f"  {row['display_name']}", style="faint")
        summary = row["summary"] if row["ok"] else "will not load — see below"
        table.add_row(marker, name, Text(row["status"], style=_STYLE[row["status"]]), Text(summary or "—"))
    theme.console().print(table)
    theme.blank()

    for row in payload["executors"]:
        if row["status"] == BROKEN:
            theme.fail(f"{row['name']}: {row['detail']}")
        elif row["unmet"]:
            # Each unmet requirement is already a sentence saying what is missing and how to fix it.
            for unmet in row["unmet"]:
                theme.warn(f"{row['name']}: {unmet}")
    profile = payload["profile"]
    if payload["profile_problem"]:
        theme.info(f"profile {profile!r}: {payload['profile_problem']}")
    else:
        theme.info(f"profile {profile!r} hands human phases to {payload['profile_executor']}")
        if payload["phase_planning"] is False:
            theme.info(
                "phase planning is off in that profile, so it has no human phases until it is on",
                "hitl.enabled",
            )
    if payload["profile_exists"]:
        # Offered for a profile that does not load too: `use` works from its file, and repairs one that
        # names an executor this machine no longer has.
        theme.next_steps([("tandem executors use NAME", f"hand the human phases of profile {profile!r} to it")])


@app.command("use", help="Make a profile's human phases run with an executor.")
def use(
    name: str = typer.Argument(..., help="The executor's name (see `tandem executors list`)."),
    profile_name: str = _PROFILE,
    as_json: bool = _AS_JSON,
) -> None:
    result = use_executor(name, profile_name=profile_name)
    if as_json:
        typer.echo(json.dumps(result, indent=2))
        return

    title = result["display_name"] or name
    if result["changed"]:
        theme.ok(f"Profile {result['profile']!r} hands human phases to {title}", f"was {result['previous']}")
    else:
        theme.ok(f"Profile {result['profile']!r} already hands human phases to {title}")
    for unmet in result["unmet"]:
        theme.warn(f"{name}: {unmet}", "needed on the machine that collects")
    if not result["phase_planning"]:
        theme.info(
            "Phase planning is off in this profile, so nothing runs a human phase until it is on",
            "hitl.enabled",
        )


# --------------------------------------------------------------------------- the teleop runtime


def _runtime(name: str, cfg: Any):
    """The runtime ``name`` is driven through. Only teleop has one tandem builds."""
    if name != "teleop":
        from tandem.executors import base

        base.check_name(name)
        raise typer.BadParameter(f"{name} has no runtime for tandem to install", param_hint="NAME")
    from tandem.teleop import recipe

    return recipe.runtime(cfg)


def install_teleop(*, force: bool = False, sources: Path | None = None, yes: bool = False) -> bool:
    """Build the teleop runtime (skipped when it is built at its pins), then turn teleop on.

    Shared by `tandem executors install teleop` and `tandem init`. Returns whether teleop is ready to
    drive with this runtime: False when the person declined the build.
    """
    from tandem.cli import runtime as runtime_cli

    cfg = settings_mod.load()
    rt = _runtime("teleop", cfg)
    title = rt.recipe.display_name
    status = rt.status()
    pending = runtime_cli.optional_steps_to_run(rt)
    if status.installed and not status.mismatched(rt.recipe.pins) and not force and not pending:
        theme.ok(f"The {title} runtime is already installed", str(status.path or ""))
        runtime_cli.say_notes(status)
    else:
        interactive = theme.is_tty() and not yes
        if runtime_cli.needs_pixi(rt):
            runtime_cli.ensure_pixi(title, ask=interactive, allowed=yes)
        theme.heading(f"installing {title}", escape(str(status.path or "")))
        if status.installed and not status.mismatched(rt.recipe.pins) and not force:
            theme.info(f"It is built; now installing {', '.join(pending)}.")
        else:
            for note in rt.recipe.notes:
                theme.info(note)
        if interactive and not typer.confirm(f"  Install {title} now?", default=True):
            return False
        runtime_cli.run_build(rt, force=force, sources_dir=sources)

    cfg = settings_mod.load()
    if not cfg.teleop.enabled:
        cfg.teleop.enabled = True
        settings_mod.save(cfg)
        theme.ok("Teleop enabled", f"device: {cfg.teleop.device}")
    if cfg.teleop.python or cfg.teleop.droid_dir:
        theme.warn(
            "teleop.python and teleop.droid_dir are set, so the driver still runs your own DROID checkout",
            "unset both to use this runtime: tandem config set teleop.python '' && "
            "tandem config set teleop.droid_dir ''",
        )
    return True


@app.command("install", help="Build an executor's runtime. teleop: DROID's workstation side, then teleop is on.")
def install(
    name: str = typer.Argument(..., help="The executor whose runtime to build (only teleop has one)."),
    sources: Path = typer.Option(
        None,
        "--sources",
        help="Take the sources from this directory of checkouts or exports instead of fetching them "
        "(default: $TANDEM_PLANNER_SOURCES). The environment is still downloaded (conda-forge).",
        file_okay=False,
    ),
    force: bool = typer.Option(False, "--force", help="Fetch every source again and rebuild, even if installed."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Ask nothing: install pixi if it is missing, and build."),
) -> None:
    _runtime(name, settings_mod.load())
    if not install_teleop(force=force, sources=sources, yes=yes):
        raise typer.Abort()
    theme.next_steps(
        [
            ("tandem config set teleop.device spacemouse", "drive with a SpaceMouse instead of VR (the default)"),
            ("tandem executors list", "teleop should say ready"),
        ]
    )


@app.command("remove", help="Delete an executor's runtime. It can be installed again.")
def remove(
    name: str = typer.Argument(..., help="The executor whose runtime to delete (only teleop has one)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
) -> None:
    rt = _runtime(name, settings_mod.load())
    title = rt.recipe.display_name
    root = rt.root
    if not (root.exists() or root.is_symlink()):
        theme.info(f"The {title} runtime is not installed: there is nothing to remove.")
        return
    size_gb = sum(f.stat().st_size for f in root.rglob("*") if f.is_file() and not f.is_symlink()) / 1e9
    theme.warn(f"This deletes {root} ({size_gb:.1f} GB), the {title} runtime.")
    if not yes and not typer.confirm("Delete the runtime?", default=False):
        raise typer.Abort()
    rt.uninstall()
    theme.ok(f"Removed the {title} runtime", str(root))
    theme.info(f"`{rt.recipe.build_command}` builds it again.")
