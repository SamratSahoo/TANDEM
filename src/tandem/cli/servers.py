"""`tandem servers` — the helper servers a planner runs beside itself (TiPToP: M2T2 grasps, FoundationStereo depth).

    tandem servers install            build their runtimes (`tandem init` does this)
    tandem servers status             installed? answering? started by tandem?
    tandem servers start [NAME]       start the ones that are down, and leave them running
    tandem servers stop [NAME]        stop the ones tandem started

A collection session does not need any of these: it starts a server that is down before it warms, and
stops the ones it started when it ends. These are for starting them ahead of time, checking on them, and
cleaning up after a session that crashed. The servers are the planner's (``registry.services``): the
active profile's planner, else the one new profiles get.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from tandem.cli import theme
from tandem.core import settings as settings_mod
from tandem.core.errors import TandemError

app = typer.Typer(no_args_is_help=True, help="The helper servers a planner runs: build, start, stop, check.")

_NAME = typer.Argument(None, help="One server by name (see `tandem servers status`); default: all of them.")


def _planner() -> str:
    from tandem.cli import runtime as runtime_cli

    try:
        return runtime_cli.active_planner()
    except TandemError:
        return settings_mod.load().default_planner


def planner_services(planner: str | None = None, name: str | None = None) -> list:
    from tandem.planners import registry

    planner = planner or _planner()
    found = registry.services(planner, settings_mod.load())
    if name is None:
        return found
    chosen = [service for service in found if service.name == name]
    if not chosen:
        known = ", ".join(service.name for service in found) or "none"
        raise TandemError(f"The {planner} planner runs no server called {name!r}.", hint=f"It runs: {known}.")
    return chosen


def install_servers(
    planner: str | None = None,
    *,
    force: bool = False,
    sources: Path | None = None,
    yes: bool = False,
    repair: bool = False,
) -> bool:
    """Build every server's runtime, each skipped when it is built at its pins. Shared with `tandem init`.

    Returns False when the person declined a build.
    """
    from tandem.cli import runtime as runtime_cli

    cfg = settings_mod.load()
    for service in planner_services(planner):
        rt = service.runtime(cfg)
        if not runtime_cli.install_recipe_runtime(rt, force=force, sources_dir=sources, yes=yes, repair=repair):
            return False
    return True


@app.command("install", help="Build the planner's servers. Safe to re-run.")
def install(
    sources: Path = typer.Option(
        None,
        "--sources",
        help="Take the sources from this directory of checkouts or exports instead of fetching them "
        "(default: $TANDEM_PLANNER_SOURCES). The environments are still downloaded.",
        file_okay=False,
    ),
    force: bool = typer.Option(False, "--force", help="Fetch every source again and rebuild, even if installed."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Ask nothing: install pixi if it is missing, and build."),
) -> None:
    if not planner_services():
        theme.ok(f"The {_planner()} planner runs no servers", "there is nothing to install")
        return
    if not install_servers(force=force, sources=sources, yes=yes):
        raise typer.Abort()
    theme.next_steps([("tandem servers start", "start them now (a session also starts them when it needs them)")])


def status_rows() -> list[dict]:
    cfg = settings_mod.load()
    return [
        {
            "name": service.name,
            "title": service.title,
            "url": service.url(),
            "local": service.local(),
            "installed": service.runtime(cfg).is_ready(),
            "healthy": service.healthy(),
            "started_by_tandem": service.started_pid(),
            "log": str(service.log_path),
        }
        for service in planner_services()
    ]


@app.command("status", help="Whether each server is installed and answering.")
def status(as_json: bool = typer.Option(False, "--json", help="Machine-readable output.")) -> None:
    rows = status_rows()
    if as_json:
        typer.echo(json.dumps(rows, indent=2))
        return
    if not rows:
        theme.ok(f"The {_planner()} planner runs no servers")
        return
    for row in rows:
        where = "" if row["local"] else " · on another machine"
        detail = f"{row['url']}{where} · {'installed' if row['installed'] else 'not installed'}"
        if row["started_by_tandem"]:
            detail += f" · started by tandem (pid {row['started_by_tandem']})"
        if row["healthy"]:
            theme.ok(f"{row['title']}: answering", detail)
        else:
            theme.warn(f"{row['title']}: not answering", detail)


@app.command("start", help="Start the servers that are down, and leave them running.")
def start(name: str = _NAME) -> None:
    for service in planner_services(name=name):
        if not service.healthy() and service.local():
            theme.info(f"starting the {service.title}", str(service.log_path))
        service.start()
        if service.healthy():
            theme.ok(f"The {service.title} is answering", service.url())
        elif not service.local():
            theme.warn(f"The {service.title} is not answering", f"{service.url()} is another machine's to start")
        else:
            theme.warn(f"The {service.title} is not answering", "`tandem servers install` builds it")


@app.command("stop", help="Stop the servers tandem started.")
def stop(name: str = _NAME) -> None:
    for service in planner_services(name=name):
        if service.stop():
            theme.ok(f"Stopped the {service.title}")
        else:
            theme.info(f"The {service.title} was not started by tandem, or has already stopped.")
