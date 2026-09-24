"""`tandem planners` — the task and motion planners tandem can drive: list them, install one, choose one.

tandem plans the task and asks a planner only for the robot's phases, so which planner that is, is a
choice: TiPToP today, and anything that registers a factory tomorrow. This is where the choice is
made::

    tandem planners list              every planner, whether this machine has it, which one is in use
    tandem planners info NAME         what it is, what it needs, what a robot phase may ask it for
    tandem planners install NAME      fetch its pinned sources and build its runtime
    tandem planners use NAME          make a profile plan with it (--default: every new profile too)
    tandem planners remove NAME       delete its runtime
    tandem planners new NAME          scaffold a package for a planner of your own

Every planner comes from the registry (``tandem.planners.registry``): the ones that ship inside
tandem, the ones installed packages declare under the ``tandem.planners`` entry point, and any
registered in this process. One that will not load is listed with its reason, never dropped: the
listing is what somebody runs to find out why their plugin is not being picked up.

Whether a profile USES a planner and whether this machine HAS it are two facts, shown side by side,
and `use` never refuses over the second: a profile is edited on a laptop and collected with on a
workstation, and the laptop has no business building a GPU runtime to be allowed to name one.

The payloads here are also what the web UI's Planners card reads (``server/routes/planners.py``),
so the page and the terminal cannot disagree about what is installed or in use.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import typer
from rich.markup import escape
from rich.text import Text

from tandem.cli import theme
from tandem.core import names
from tandem.core import settings as settings_mod
from tandem.core.errors import ProfileError, TandemError

app = typer.Typer(no_args_is_help=True, help="The task and motion planners tandem can drive.")

_PROFILE = typer.Option(None, "--profile", "-p", help="A profile other than the active one.")
_AS_JSON = typer.Option(False, "--json", help="Machine-readable output.")

# Whether this machine has a planner, in one word a listing can colour. "Active" is deliberately not
# one of them: a profile using a planner is a separate fact from the machine having it, and a planner
# can be both active and not installed -- which is exactly the state worth seeing at a glance.
INSTALLED = "installed"
NOT_INSTALLED = "not installed"
OUTDATED = "outdated"
NO_RUNTIME = "no runtime needed"
BROKEN = "broken"
STATUSES = (INSTALLED, NOT_INSTALLED, OUTDATED, NO_RUNTIME, BROKEN)

_STYLE = {INSTALLED: "ok", NO_RUNTIME: "ok", OUTDATED: "warn", NOT_INSTALLED: "faint", BROKEN: "err"}


def install_command(name: str) -> str:
    return f"tandem planners install {name}"


# --------------------------------------------------------------------------- what is where


def profile_planner(profile_name: str | None = None) -> tuple[str, str | None, str | None]:
    """(the profile's name, the planner it names or None, why that could not be read or None).

    Never raises: a listing marks which planner is active, and a profile that is missing or does not
    load is something to show next to the listing, not a reason to withhold it.
    """
    from tandem.core import profiles

    name = profile_name or settings_mod.load().active_profile
    if not profiles.exists(name):
        return name, None, f"there is no profile named {name!r} yet"
    try:
        return name, profiles.load(name).planner.backend, None
    except ProfileError as exc:
        return name, None, exc.message.splitlines()[0]


def runtime_state(name: str, info: Any, settings: Any) -> dict:
    """Whether ``name``'s runtime is on this machine, and built at the commits it pins now.

    ``status`` is one of ``STATUSES``; ``detail`` is one line saying why; ``runtime`` is the runtime's
    own status (JSON-safe) and ``mismatched`` the pinned sources it was not built at. Never raises: a
    runtime that cannot be inspected is that planner's problem and its row says so, while every other
    row is still worth reading.

    Outdated means built, or half-built, from other commits than the planner pins now -- what a tandem
    upgrade that moved a pin leaves behind. It is told apart from "not installed" because the fix is
    the same command but the cost is not: an update keeps the environment, a first install solves it.
    """
    from tandem.planners import registry

    try:
        rt = registry.runtime(name, settings)
    except Exception as exc:  # a plugin's runtime() is its own code, and its bug is its own row's
        return _state(BROKEN, f"its runtime could not be located: {_why(exc)}")
    if rt is None:
        return _state(NO_RUNTIME, "pure Python: nothing to install")
    try:
        status = rt.status()
    except Exception as exc:
        return _state(BROKEN, f"its runtime could not be inspected: {_why(exc)}")

    wanted = tuple(getattr(info, "sources", ()) or ())
    mismatched = list(status.mismatched(wanted))
    if status.installed and not mismatched:
        kind, detail = INSTALLED, status.detail or f"ready at {status.path}"
    elif status.pins and mismatched:
        have = {pin.name: pin.commit for pin in status.pins}
        moved = ", ".join(
            f"{pin.name} {str(have.get(pin.name) or '—')[:7]} → {pin.short()}"
            for pin in wanted
            if pin.name in mismatched
        )
        kind, detail = OUTDATED, f"built from other commits than it pins now: {moved}"
    else:
        kind, detail = NOT_INSTALLED, "; ".join(status.problems) or status.detail or "not built"
    return {"status": kind, "detail": detail, "runtime": status.to_dict(), "mismatched": mismatched}


def _state(kind: str, detail: str) -> dict:
    return {"status": kind, "detail": detail, "runtime": None, "mismatched": []}


def _why(exc: BaseException) -> str:
    if isinstance(exc, TandemError):
        return exc.message.splitlines()[0]
    return f"{type(exc).__name__}: {exc}"


def catalog_payload(*, profile_name: str | None = None) -> dict:
    """Every planner, as `tandem planners list --json` and the web UI show it. JSON-safe."""
    from tandem.planners import registry

    cfg = settings_mod.load()
    profile, in_use, problem = profile_planner(profile_name)
    entries = registry.catalog()
    # A plugin shadowed by a built-in of the same name is listed too, with why it is not used. Only
    # the one that IS used is marked active or default; a broken planner a profile names still is,
    # because "the planner this profile uses will not load" is the thing that row must say.
    working = {entry.name for entry in entries if entry.ok}

    rows = []
    for entry in entries:
        counts = entry.ok or entry.name not in working
        info = entry.info
        row = {
            "kind": "planner",
            "name": entry.name,
            "display_name": info.title if info is not None else entry.name,
            "summary": info.summary if info is not None else "",
            "homepage": info.homepage if info is not None else "",
            "origin": entry.origin,
            "ok": entry.ok,
            "error": entry.error,
            "active": counts and entry.name == in_use,
            "default": counts and entry.name == cfg.default_planner,
        }
        if entry.ok:
            row.update(runtime_state(entry.name, info, cfg))
        else:
            row.update(_state(BROKEN, entry.error or "it could not be loaded"))
        row["install_command"] = (
            install_command(entry.name) if row["status"] in (NOT_INSTALLED, OUTDATED) else None
        )
        rows.append(row)
    return {
        "profile": profile,
        "profile_planner": in_use,
        "profile_problem": problem,
        "default_planner": cfg.default_planner,
        "planners": rows,
    }


def capabilities_summary(caps: Any) -> dict:
    """What a robot phase may ask the planner for, and what it promises, JSON-safe."""
    predicates = []
    for name, predicate in caps.goal_predicates.items():
        args = ", ".join(f"?{p.name}: {p.type}" for p in predicate.parameters)
        predicates.append(
            {
                "name": name,
                "signature": f"{name}({args})",
                "description": caps.predicate_descriptions.get(name, ""),
                # None: the planner supplies it itself, and a goal that states it has it dropped.
                "wire_name": caps.goal_predicate_wire_names.get(name),
                "checkable": name in caps.checkable_predicates,
            }
        )
    return {
        "robot_description": caps.robot_description,
        "goal_predicates": predicates,
        "robot_operators": list(caps.robot_operators),
        "movable_type": caps.movable_type,
        "surface_type": caps.surface_type,
        "supports": {
            "movable_restriction": caps.supports_movable_restriction,
            "return_home": caps.supports_return_home,
            "cooperative_stop": caps.supports_cooperative_stop,
            "skeleton_reuse": caps.supports_skeleton_reuse,
        },
        "one_pick_per_object": caps.one_pick_per_object,
        "initial_state_is_clean": caps.initial_state_is_clean,
        "prompt_fragments": sorted(caps.prompt_fragments),
    }


def info_payload(name: str, *, profile_name: str | None = None) -> dict:
    """One planner in full, as `tandem planners info --json` and the web UI show it.

    Raises the registry's own error for a name nothing provides (with the nearest one) and for a
    plugin that will not load (with why): a description of a planner that cannot be described would
    be a guess.
    """
    from tandem.planners import registry

    factory = registry.factory(name)
    info = factory.info
    caps = factory.capabilities()
    cfg = settings_mod.load()
    profile, in_use, problem = profile_planner(profile_name)
    state = runtime_state(name, info, cfg)
    # A tandem.planners.Planner declares the planner.options it reads; any other factory does not
    # say, which is not the same as reading none.
    options = getattr(factory, "OPTIONS", None)
    return {
        "kind": "planner",
        **info.to_dict(),
        "origin": registry.origin(name),
        "status": state["status"],
        "detail": state["detail"],
        "active": in_use == name,
        "default": cfg.default_planner == name,
        "profile": profile,
        "profile_problem": problem,
        "install_command": install_command(name) if state["status"] in (NOT_INSTALLED, OUTDATED) else None,
        "runtime": {
            "needed": state["status"] != NO_RUNTIME,
            **(state["runtime"] or {}),
            "mismatched": state["mismatched"],
        },
        "options": dict(options) if isinstance(options, Mapping) else None,
        **_presets_of(name),
        "capabilities": capabilities_summary(caps),
    }


def _presets_of(name: str) -> dict:
    """The presets the planner ``name`` ships, for `tandem profile create --preset`. A broken one is said, not raised:
    a description of the planner is what somebody reads to find out what is wrong with it."""
    from tandem.core import presets

    try:
        found = presets.planner_presets(name)
    except TandemError as exc:
        return {"presets": [], "presets_problem": exc.message}
    listed = [
        {"name": p.name, "title": p.title, "summary": p.summary, "extends": p.extends} for p in found.values()
    ]
    return {"presets": sorted(listed, key=lambda p: p["name"]), "presets_problem": None}


# --------------------------------------------------------------------------- choosing one


def use_planner(name: str, *, profile_name: str | None = None, make_default: bool = False) -> dict:
    """Make a profile plan with ``name``; with ``make_default``, every new profile too. What changed.

    The profile is the active one unless ``profile_name`` says otherwise. With ``make_default`` and no
    profile named, a machine that has no profile yet only gets its default set -- the planner the
    first profile will be created with.

    ``planner.options`` are the old planner's own settings, which the new one would refuse (a planner
    refuses an option it does not read, rather than ignore it), so they are removed, and the result
    says which. Not installing the planner is not a reason to refuse: see the module docstring.
    """
    from tandem.core import profiles
    from tandem.planners import registry

    factory = registry.factory(name)  # an unknown or broken planner is refused here, loudly
    cfg = settings_mod.load()
    target = profile_name or cfg.active_profile
    result: dict[str, Any] = {
        "planner": name,
        "display_name": factory.info.title,
        "profile": None,
        "previous": None,
        "changed": False,
        "dropped_options": {},
    }
    if profile_name is not None or not make_default or profiles.exists(target):
        profile = profiles.load(target)
        previous = profile.planner.backend
        result.update(profile=target, previous=previous)
        if previous != name:
            dropped = dict(profile.planner.options)
            profile.planner = profiles.PlannerSpec(backend=name)
            profiles.save(profile)
            result.update(changed=True, dropped_options=dropped)
    if make_default:
        set_default_planner(name)
    result["default_planner"] = settings_mod.load().default_planner
    state = runtime_state(name, factory.info, settings_mod.load())
    result.update(
        status=state["status"],
        detail=state["detail"],
        install_command=install_command(name) if state["status"] in (NOT_INSTALLED, OUTDATED) else None,
    )
    return result


def set_default_planner(name: str) -> str:
    """Make ``name`` the planner every new profile gets. Checked against the registry first."""
    from tandem.planners import registry

    registry.factory(name)
    cfg = settings_mod.load()
    if cfg.default_planner != name:
        cfg.default_planner = name
        settings_mod.save(cfg)
    return name


def planner_for_new_profile(requested: str | None = None) -> str:
    """The planner a new profile is created with: ``requested`` when given, else the machine's default.

    Checked here, by name, so a default naming a planner this machine no longer has is an error now,
    rather than a profile that is written and then refused by every command after it. A planner that
    is installed but broken passes, as it does in a profile: that is the session's to report.
    """
    from tandem.planners import registry

    name = requested or settings_mod.load().default_planner
    if name in registry.available():
        return name
    try:
        registry.factory(name)
    except TandemError as exc:
        if requested:
            raise
        raise TandemError(
            f"New profiles plan with {name!r} (default_planner), which this machine does not have. {exc.message}",
            hint=f"{exc.hint or ''} `tandem planners use NAME --default` chooses another default.".strip(),
        ) from exc
    return name


# --------------------------------------------------------------------------- commands


@app.command("list", help="Every planner tandem can drive, whether this machine has it, and which is in use.")
def list_planners(as_json: bool = _AS_JSON, profile_name: str = _PROFILE) -> None:
    payload = catalog_payload(profile_name=profile_name)
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    rows = payload["planners"]
    render_catalog(rows)
    theme.blank()
    for row in rows:
        if row["status"] == BROKEN:
            theme.fail(f"{row['name']}: {row['detail']}")
    profile, in_use = payload["profile"], payload["profile_planner"]
    if payload["profile_problem"]:
        theme.info(f"profile {profile!r}: {payload['profile_problem']}")
    else:
        theme.info(f"profile {profile!r} plans with {in_use}")
    theme.info(f"new profiles plan with {payload['default_planner']}", "`tandem planners use NAME --default`")

    steps = []
    active = next((r for r in rows if r["active"]), None)
    if active is not None and active["install_command"]:
        steps.append((active["install_command"], f"build the runtime profile {profile!r} needs"))
    steps.append(("tandem planners info NAME", "what a planner needs, and what it can be asked for"))
    if payload["profile_problem"] is None:
        steps.append(("tandem planners use NAME", f"plan with it in profile {profile!r}"))
    else:
        steps.append(
            ("tandem planners use NAME --default", "the planner new profiles, and `tandem init`, start with")
        )
    theme.next_steps(steps)


def render_catalog(rows: list[dict]) -> None:
    """The planners as a table: shared by `list` and `tandem init`'s planner step."""
    table = theme.table("", "planner", "status", "summary")
    for row in rows:
        marker = Text("●", style="accent") if row["active"] else Text(" ")
        name = Text(row["name"], style="bold")
        if row["display_name"] != row["name"]:
            name.append(f"  {row['display_name']}", style="faint")
        if row["default"]:
            name.append("  default", style="violet")
        status = Text(row["status"], style=_STYLE.get(row["status"], "faint"))
        summary = row["summary"] if row["ok"] else "will not load — see below"
        table.add_row(
            marker, name, status, Text(_truncate(summary, 64), style="default" if row["ok"] else "err")
        )
    theme.console().print(table)


@app.command("info", help="What a planner is, what it needs, and what a robot phase may ask it for.")
def info(
    name: str = typer.Argument(..., help="The planner's name (see `tandem planners list`)."),
    as_json: bool = _AS_JSON,
    profile_name: str = _PROFILE,
) -> None:
    payload = info_payload(name, profile_name=profile_name)
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    caps = payload["capabilities"]
    theme.blank()
    theme.heading(payload["display_name"], f"{escape(name)} · {escape(payload['origin'])}")
    if payload["profile_problem"]:
        in_use = f"unknown: {payload['profile_problem']}"
    else:
        in_use = (
            f"yes, by {payload['profile']!r}"
            if payload["active"]
            else f"no — {payload['profile']!r} uses another"
        )
    theme.kv(
        [
            ("summary", payload["summary"]),
            ("homepage", payload["homepage"]),
            ("status", f"{payload['status']} — {payload['detail']}"),
            ("in use", in_use),
            ("default", payload["default"]),
        ]
    )

    if payload["requires"]:
        theme.blank()
        theme.heading("needs", "on the machine that runs it")
        for line in payload["requires"]:
            theme.info(line)

    if payload["sources"]:
        theme.blank()
        theme.heading("sources", "what an install fetches, pinned")
        installed = {pin["name"]: pin["commit"] for pin in payload["runtime"].get("pins") or []}
        table = theme.table("source", "pinned", "installed", "upstream")
        for pin in payload["sources"]:
            have = installed.get(pin["name"])
            shown = (
                Text("—", style="faint")
                if have is None
                else Text(have[:12], style="default" if have == pin["commit"] else "warn")
            )
            table.add_row(Text(pin["name"]), pin["commit"][:12], shown, Text(pin["url"], style="faint"))
        theme.console().print(table)

    theme.blank()
    theme.heading("goal language", "what a robot phase may ask it for")
    for predicate in caps["goal_predicates"]:
        line = Text(f"  {predicate['signature']}", style="code")
        if predicate["description"]:
            line.append(f"  {predicate['description']}", style="faint")
        if predicate["wire_name"] is None:
            line.append("  (supplied by the planner)", style="faint")
        theme.console().print(line)
    supports = caps["supports"]
    theme.kv(
        [
            ("the robot can", caps["robot_description"]),
            ("operators", caps["robot_operators"]),
            ("objects", f"{caps['movable_type']} (moved) · {caps['surface_type']} (put things on)"),
            ("restricts picks", supports["movable_restriction"]),
            ("ends away from home", supports["return_home"]),
            ("stops mid-leg", supports["cooperative_stop"]),
            ("reuses a skeleton", supports["skeleton_reuse"]),
        ]
    )
    if payload["options"] is not None:
        theme.blank()
        theme.heading("planner.options", "what a profile may set for it")
        if payload["options"]:
            theme.kv([(escape(str(key)), text) for key, text in payload["options"].items()])
        else:
            theme.info("none: it reads no planner.options")

    if payload["presets"] or payload["presets_problem"]:
        theme.blank()
        theme.heading("presets", "`tandem profile create NAME --preset NAME`")
        if payload["presets"]:
            theme.kv([(escape(p["name"]), p["title"]) for p in payload["presets"]])
        if payload["presets_problem"]:
            theme.warn(payload["presets_problem"])

    runtime = payload["runtime"]
    theme.blank()
    theme.heading("runtime")
    if not runtime["needed"]:
        theme.ok("pure Python", "there is nothing to install beyond its package")
    else:
        theme.kv([("path", runtime.get("path")), ("state", runtime.get("detail"))])
        for problem in runtime.get("problems") or []:
            theme.warn(problem)

    steps = []
    if payload["install_command"]:
        steps.append((payload["install_command"], "fetch its sources and build its runtime"))
    if payload["profile_problem"] is None and not payload["active"]:
        steps.append((f"tandem planners use {name}", f"plan with it in profile {payload['profile']!r}"))
    elif payload["profile_problem"] is not None and not payload["default"]:
        steps.append((f"tandem planners use {name} --default", "start new profiles with it"))
    if steps:
        theme.next_steps(steps)


@app.command("install", help="Fetch a planner's pinned sources and build its runtime. Safe to re-run.")
def install(
    name: str = typer.Argument(..., help="The planner's name (see `tandem planners list`)."),
    sources: Path = typer.Option(
        None,
        "--sources",
        help="Install from this directory of checkouts or exports instead of fetching "
        "(default: $TANDEM_PLANNER_SOURCES). `python tools/bundle.py` makes one.",
        file_okay=False,
    ),
    force: bool = typer.Option(
        False, "--force", help="Fetch every source again and rebuild, even if installed."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Ask nothing: install pixi if it is missing, and build."
    ),
) -> None:
    from tandem.cli import runtime as runtime_cli
    from tandem.planners import registry

    factory = registry.factory(name)
    title = factory.info.title
    cfg = settings_mod.load()
    rt = registry.runtime(name, cfg)
    if rt is None:
        theme.ok(f"{title} is pure Python", "there is no runtime to install")
        _suggest_use(name)
        return

    status = rt.status()
    if status.installed and not status.mismatched(factory.info.sources) and not force:
        # Idempotent without spending anything: no fetch, no pixi solve, no kernel build.
        theme.ok(f"{title} is already installed", str(status.path or ""))
        _suggest_use(name)
        return

    interactive = theme.is_tty() and not yes
    if runtime_cli.needs_pixi(rt):
        runtime_cli.ensure_pixi(title, ask=interactive, allowed=yes)
    theme.heading(f"installing {title}", escape(str(status.path or "")))
    recipe = getattr(rt, "recipe", None)
    for note in getattr(recipe, "notes", ()) or ():
        theme.info(note)
    if interactive and not typer.confirm(f"  Install {title} now?", default=True):
        raise typer.Abort()
    runtime_cli.run_build(rt, force=force, sources_dir=sources)
    _suggest_use(name)


def _suggest_use(name: str) -> None:
    profile, in_use, problem = profile_planner()
    if problem is None and in_use != name:
        theme.next_steps([(f"tandem planners use {name}", f"plan with it in profile {profile!r}")])


@app.command("use", help="Make a profile plan with a planner (--default: every new profile too).")
def use(
    name: str = typer.Argument(..., help="The planner's name (see `tandem planners list`)."),
    profile_name: str = _PROFILE,
    default: bool = typer.Option(False, "--default", help="Also make it the planner every new profile gets."),
    as_json: bool = _AS_JSON,
) -> None:
    result = use_planner(name, profile_name=profile_name, make_default=default)
    if as_json:
        typer.echo(json.dumps(result, indent=2))
        return

    title = result["display_name"]
    if result["profile"] is not None:
        if result["changed"]:
            theme.ok(f"Profile {result['profile']!r} now plans with {title}", f"was {result['previous']}")
        else:
            theme.ok(f"Profile {result['profile']!r} already plans with {title}")
    if result["dropped_options"]:
        theme.warn(
            f"Removed planner.options {', '.join(sorted(map(str, result['dropped_options'])))}",
            f"they were {result['previous']}'s own settings, which {name} would refuse",
        )
    if default:
        theme.ok(f"New profiles plan with {title}")

    if result["status"] in (NOT_INSTALLED, OUTDATED):
        verb = "is not installed" if result["status"] == NOT_INSTALLED else "is outdated"
        theme.warn(f"{title} {verb} on this machine", result["detail"])
        theme.next_steps([(result["install_command"], "build its runtime, to collect with it")])
    elif result["status"] == BROKEN:
        theme.warn(f"{title}'s runtime could not be checked", result["detail"])


@app.command(
    "remove", help="Delete a planner's runtime. The planner stays listed and can be installed again."
)
def remove(
    name: str = typer.Argument(..., help="The planner's name (see `tandem planners list`)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
) -> None:
    from tandem.planners import registry

    factory = registry.factory(name)
    title = factory.info.title
    rt = registry.runtime(name, settings_mod.load())
    if rt is None:
        theme.info(f"{title} is pure Python: it has no runtime to remove.")
        origin = registry.origin(name)
        if origin.startswith("entry point"):
            theme.info("To remove the planner itself, uninstall the package that provides it", origin)
        return

    status = rt.status()
    root = Path(status.path) if status.path else None
    if root is None or not (root.exists() or root.is_symlink()):
        theme.info(f"{title} is not installed: there is nothing to remove.")
        return
    size_gb = sum(f.stat().st_size for f in root.rglob("*") if f.is_file() and not f.is_symlink()) / 1e9
    theme.warn(f"This deletes {root} ({size_gb:.1f} GB), the {title} runtime.")
    profile, in_use, _ = profile_planner()
    if in_use == name:
        theme.warn(
            f"Profile {profile!r} plans with {title}",
            "it cannot collect until the runtime is installed again",
        )
    if not yes and not typer.confirm("Delete the runtime?", default=False):
        raise typer.Abort()
    rt.uninstall()
    theme.ok(f"Removed the {title} runtime", str(root))
    theme.info(f"`{install_command(name)}` builds it again.")


# --------------------------------------------------------------------------- a planner of your own

#: Where the templates live, inside ``tandem.resources``.
SCAFFOLD = "scaffold"
_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")

# (template, where it is written, which kind of planner it is for: None both, True a sidecar one, False
# an in-process one). Templates are `.tmpl`, never `.py`: they are not tandem's modules, and ruff,
# setuptools and pytest must never take them for some.
_FILES: tuple[tuple[str, str, bool | None], ...] = (
    ("pyproject.toml.tmpl", "pyproject.toml", None),
    ("README.md.tmpl", "README.md", None),
    ("gitignore.tmpl", ".gitignore", None),
    ("package/__init__.py.tmpl", "src/{module}/__init__.py", None),
    ("package/planner.py.tmpl", "src/{module}/planner.py", False),
    ("package/sidecar_planner.py.tmpl", "src/{module}/planner.py", True),
    ("package/sidecar.py.tmpl", "src/{module}/sidecar.py", True),
    ("tests/test_conformance.py.tmpl", "tests/test_conformance.py", None),
)


def scaffold_values(name: str, *, sidecar: bool = False) -> dict[str, str]:
    """What the templates are filled with, derived from the planner's name alone."""
    from tandem import __version__

    words = [word for word in re.split(r"[-_]+", name) if word]
    class_name = "".join(word[:1].upper() + word[1:] for word in words)
    if not class_name.endswith("Planner"):
        class_name += "Planner"
    module = "tandem_" + name.replace("-", "_")
    return {
        "name": name,
        "module": module,
        "dist": f"tandem-{name}",
        "class_name": class_name,
        "title": " ".join(word[:1].upper() + word[1:] for word in words),
        "tandem_version": __version__,
        "kind": "SidecarPlanner" if sidecar else "Planner",
        "layout": _layout(module, sidecar),
    }


def _layout(module: str, sidecar: bool) -> str:
    lines = [
        "pyproject.toml                  the package, and its tandem.planners entry point",
        f"src/{module}/planner.py        the planner: its declarations and its verbs",
    ]
    if sidecar:
        lines.append(
            f"src/{module}/sidecar.py        the script tandem runs in the planner's own environment"
        )
    lines.append("tests/test_conformance.py       tandem's conformance kit, run against it")
    return "\n".join(lines)


def render_template(text: str, values: Mapping[str, str], *, template: str = "") -> str:
    """Fill ``{{name}}`` placeholders. One the values do not have is an error, never left in the output."""
    unknown = sorted({m.group(1) for m in _PLACEHOLDER.finditer(text)} - set(values))
    if unknown:
        raise TandemError(
            f"The scaffold template {template or '?'} uses {', '.join(unknown)}, which nothing fills in.",
            hint="This is a bug in tandem's scaffold templates, not in what you asked for.",
        )
    return _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), text)


def read_template(template: str) -> str:
    from tandem import resources

    return resources.read(f"{SCAFFOLD}/{template}")


def scaffold(name: str, dest: Path, *, sidecar: bool = False) -> list[Path]:
    """Write a planner package for ``name`` into ``dest``. Returns the files written.

    What is written runs as it stands -- a stand-in world, planned and "executed" in memory -- and
    passes tandem's conformance kit, so its author starts from green and keeps it green while each
    TODO is replaced with the real planner. Nothing is overwritten: ``dest`` must be new or empty.
    """
    from tandem.planners import registry

    if not names.is_valid(name):
        raise TandemError(
            f"{name!r} cannot be a planner's name.",
            hint=f"Use {names.RULE}: it is typed into profiles and onto command lines.",
        )
    if name in registry.available():
        raise TandemError(
            f"There is already a planner named {name!r} ({registry.origin(name)}).",
            hint="Choose another name. A plugin whose name is taken is listed as shadowed and never used.",
        )
    dest = Path(dest).expanduser()
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise TandemError(
            f"{dest} already exists and is not empty, so nothing was written.",
            hint="Choose another --dir. The scaffold never overwrites a file.",
        )

    values = scaffold_values(name, sidecar=sidecar)
    written: list[Path] = []
    for template, target, variant in _FILES:
        if variant is not None and variant != sidecar:
            continue
        path = dest / target.format(**values)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_template(read_template(template), values, template=template))
        written.append(path)
    return written


@app.command(
    "new", help="Scaffold a package for a planner of your own. It passes tandem's conformance kit as written."
)
def new(
    name: str = typer.Argument(..., help=f"The planner's name ({names.RULE})."),
    directory: Path = typer.Option(
        None, "--dir", help="Where to create the package (default: ./tandem-NAME)."
    ),
    sidecar: bool = typer.Option(
        False,
        "--sidecar",
        help="A SidecarPlanner: the planner runs as a script in an environment of its own (torch, CUDA, a "
        "robot SDK), talking to tandem over JSON lines.",
    ),
) -> None:
    dest = directory if directory is not None else Path.cwd() / f"tandem-{name}"
    written = scaffold(name, dest, sidecar=sidecar)
    values = scaffold_values(name, sidecar=sidecar)
    theme.ok(f"Created the {values['title']} planner package", str(dest))
    for path in written:
        theme.info(str(path.relative_to(dest)))
    # Escaped: next_steps renders markup, and `.[test]` would otherwise vanish as a style tag.
    theme.next_steps(
        [
            (escape('pip install -e ".[test]"'), "in that directory: install it, and tandem sees it"),
            ("pytest", "run tandem's conformance kit against it: green as written"),
            (f"tandem planners info {name}", "see it as tandem does"),
            (f"tandem planners use {name}", "plan with it in a profile"),
        ]
    )


def _truncate(text: str, width: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= width else text[: width - 1] + "…"
