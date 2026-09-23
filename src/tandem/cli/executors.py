"""`tandem executors` — who carries out a human phase: list them, choose one for a profile.

A human phase is carried out by an executor (pi_omega_Delta in the paper): a person driving the arm
through teleop is the one that ships, and a package adds another through the
``tandem.human_executors`` entry point. A profile names the one it uses in ``hitl.human_executor``::

    tandem executors list             every executor, whether it is ready here, which one is in use
    tandem executors use NAME         make a profile's human phases run with it

There is no install step. An executor is a Python package, and what it needs on this machine (a DROID
checkout, a policy server) is its own requirements list, which the listing shows with whatever is
still unmet -- the same way ``tandem planners`` shows whether a planner's runtime is built. As there,
choosing one that is not ready yet is allowed and warned about, never refused: a profile is edited
wherever it is edited, and collected with on the machine that has the hardware.

The payloads here are also what the web UI reads (``server/routes/planners.py``).
"""

from __future__ import annotations

import json
from typing import Any

import typer
from rich.text import Text

from tandem.cli import theme
from tandem.core import settings as settings_mod
from tandem.core.errors import ProfileError

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
        out["problem"] = f"there is no profile named {name!r} yet"
        return out
    try:
        profile = profiles.load(name)
    except ProfileError as exc:
        out["problem"] = exc.message.splitlines()[0]
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
        "executors": rows,
    }


def use_executor(name: str, *, profile_name: str | None = None) -> dict:
    """Make a profile's human phases run with ``name``. What changed, and whether it is ready here.

    An unknown name, or an installed executor that will not load, is refused with the registry's own
    error. One whose requirements are unmet on this machine is not: the result lists them.
    """
    from tandem.core import profiles
    from tandem.executors import base as executors

    executors.refresh()
    info = executors.info(name)
    target = profile_name or settings_mod.load().active_profile
    profile = profiles.load(target)
    previous = profile.hitl.human_executor
    if previous != name:
        profile.hitl.human_executor = name
        profiles.save(profile)
    return {
        "executor": name,
        "display_name": info.display_name,
        "profile": target,
        "previous": previous,
        "changed": previous != name,
        "status": READY if not info.unmet else NEEDS_SETUP,
        "unmet": list(info.unmet),
        "phase_planning": profile.hitl.enabled,
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
