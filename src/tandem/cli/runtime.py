"""`tandem runtime` — the runtime of the planner the active profile uses.

Every command here acts on one planner's runtime: the planner the active profile names (or
``--profile``'s, or ``--planner``), found through the planner registry. For TiPToP that is the GPU
runtime `tandem init` builds: its pinned sources fetched, its pixi environment solved and cuRobo's
kernels compiled. A planner that is pure Python has no runtime, and says so.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import typer

from tandem.cli import theme
from tandem.core import paths
from tandem.core import settings as settings_mod
from tandem.core.errors import TandemError

app = typer.Typer(no_args_is_help=True, help="The runtime of the planner the active profile uses.")

_PLANNER = typer.Option(None, "--planner", help="A planner's name, instead of the active profile's.")
_PROFILE = typer.Option(
    None, "--profile", "-p", help="Use this profile's planner instead of the active one's."
)


# --------------------------------------------------------------------------- which runtime


def active_planner(profile_name: str | None = None) -> str:
    """The planner a profile names -- the active profile unless one is named.

    A profile that does not exist yet has the planner a new one would get. That is the one case with
    no profile to ask, and it is the first thing `tandem init` meets: it builds the runtime before it
    creates the first profile. A profile that exists but does not load is an error, not a reason to
    guess: installing the wrong planner's runtime is twenty minutes and 25 GB spent on nothing.
    """
    from tandem.core import profiles

    name = profile_name or settings_mod.load().active_profile
    if not profiles.exists(name):
        return profiles.PlannerSpec.model_fields["backend"].default
    return profiles.load(name).planner.backend


def planner_runtime(
    *, planner: str | None = None, profile_name: str | None = None, settings: Any = None
) -> tuple[str, Any]:
    """(planner name, its runtime or None for a pure-Python planner)."""
    from tandem.planners import registry

    name = planner or active_planner(profile_name)
    return name, registry.runtime(name, settings if settings is not None else settings_mod.load())


def _recipe_runtime(planner: str | None, profile_name: str | None):
    """The runtime, for the commands that enter it -- which only a recipe's runtime can be."""
    from tandem.planners.runtime import RecipeRuntime

    name, rt = planner_runtime(planner=planner, profile_name=profile_name)
    if rt is None:
        raise TandemError(f"The {name!r} planner is pure Python, so it has no runtime to enter.")
    if not isinstance(rt, RecipeRuntime) or rt.recipe.environment is None:
        raise TandemError(f"The {name!r} planner's runtime has no environment tandem knows how to enter.")
    return name, rt


def runtime_payload(*, planner: str | None = None, profile_name: str | None = None) -> dict:
    """What `runtime status --json` and the web UI's runtime card show. JSON-safe.

    The generic facts first -- installed, pins, what is missing -- then, for a runtime built from a
    recipe, the per-part rows and the keys this payload has always had (``sources_present``,
    ``env_built``, ``kernels_built``, ``vendor``), so a script written against it keeps working.
    """
    from tandem.planners import registry
    from tandem.planners.runtime import RecipeRuntime

    name, rt = planner_runtime(planner=planner, profile_name=profile_name)
    info = registry.info(name)
    wanted = [pin.to_dict() for pin in info.sources]
    if rt is None:
        return {
            "planner": name,
            "title": info.title,
            "installed": True,
            "ready": True,
            "root": None,
            "detail": "pure Python: nothing to build",
            "problems": [],
            "pins": [],
            "wanted": wanted,
            "mismatched": [],
            "rows": [],
            "sources": [],
        }

    status = rt.status()
    payload = {
        "planner": name,
        "title": info.title,
        **status.to_dict(),
        "ready": status.installed,
        "root": status.path,
        "wanted": wanted,
        "mismatched": list(status.mismatched(info.sources)),
    }
    if isinstance(rt, RecipeRuntime):
        st = rt.inspect()
        installed = rt.record()["sources"]
        payload.update(
            exists=st.exists,
            sources_present=st.sources_present,
            env_built=st.environment_built,
            # The one build step TiPToP has compiles cuRobo's kernels; for any recipe it means "every
            # step has run", which is what the key always meant to a reader.
            kernels_built=st.exists and st.sources_present and st.steps_done,
            built_at=st.built_at,
            notes=list(st.notes),
            rows=[list(row) for row in rt.rows(st)],
            sources=[
                {
                    "name": s.name,
                    "url": s.wanted.url,
                    "commit": s.wanted.commit,
                    "installed": s.commit,
                    "present": s.present,
                    "origin": s.origin,
                    "verified": s.verified,
                    "current": s.current,
                }
                for s in st.sources
            ]
            or [
                {
                    "name": p.name,
                    "url": p.url,
                    "commit": p.commit,
                    "installed": None,
                    "present": False,
                    "origin": None,
                    "verified": None,
                    "current": False,
                }
                for p in info.sources
            ],
            vendor={
                n: {"url": e.get("url") or "", "commit": e.get("commit"), "version": str(e.get("commit"))[:7]}
                for n, e in installed.items()
                if isinstance(e, dict) and e.get("commit")
            }
            or None,
            vendor_installed=bool(installed),
        )
    return payload


# --------------------------------------------------------------------------- commands


@app.command("status", help="What is built, from which sources.")
def status(
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
    planner: str = _PLANNER,
    profile_name: str = _PROFILE,
) -> None:
    payload = runtime_payload(planner=planner, profile_name=profile_name)
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    theme.blank()
    theme.heading(f"runtime · {payload['title']}", str(payload["root"] or ""))
    if payload["root"] is None:
        theme.ok(f"{payload['title']} is pure Python", "there is no runtime to build")
        return
    theme.kv(
        [(label, value) for label, value in payload.get("rows") or []] or [("detail", payload["detail"])]
    )

    sources = payload.get("sources") or []
    if sources:
        theme.blank()
        theme.heading("sources", "pinned by this version of tandem")
        table = theme.table("source", "pinned", "installed", "upstream")
        for source in sources:
            installed = source["installed"]
            if installed is None:
                shown = "[faint]—[/faint]"
            elif installed == source["commit"]:
                shown = installed[:12] + (
                    "" if source.get("verified") is not False else " [warn](unverified)[/warn]"
                )
            else:
                shown = f"[warn]{installed[:12]}[/warn]"
            table.add_row(source["name"], source["commit"][:12], shown, f"[faint]{source['url']}[/faint]")
        theme.console().print(table)

    theme.blank()
    for note in payload.get("notes") or []:
        theme.info(note)
    if payload["ready"]:
        theme.ok("Runtime is ready")
    else:
        for problem in payload["problems"]:
            theme.warn(problem)
        theme.next_steps([("tandem runtime build", "build or repair it")])


@app.command("build", help="Build (or repair) the runtime of the active profile's planner.")
def build(
    force: bool = typer.Option(False, "--force", help="Fetch every source again before building."),
    env_only: bool = typer.Option(False, "--env-only", help="Stop once the environment is solved."),
    sources: Path = typer.Option(
        None,
        "--sources",
        help="Install from this directory of checkouts or exports instead of fetching "
        "(default: $TANDEM_PLANNER_SOURCES). `python tools/bundle.py` makes one.",
        file_okay=False,
    ),
    planner: str = _PLANNER,
    profile_name: str = _PROFILE,
) -> None:
    name, rt = planner_runtime(planner=planner, profile_name=profile_name)
    if rt is None:
        theme.ok(f"The {name!r} planner is pure Python", "there is nothing to build")
        return
    run_build(rt, force=force, env_only=env_only, sources_dir=sources)


def run_build(rt, *, force: bool = False, env_only: bool = False, sources_dir: Path | None = None) -> None:
    """Shared by `runtime build` and `init`, so they cannot drift apart."""
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    from tandem.planners.runtime import RecipeRuntime

    recipe = isinstance(rt, RecipeRuntime)
    if env_only and not recipe:
        raise TandemError("--env-only needs a runtime built from a recipe; this planner's is not.")

    paths.ensure_dir(paths.log_dir())
    log_path = paths.log_dir() / f"runtime-build-{datetime.now():%Y%m%d-%H%M%S}.log"
    log_file = log_path.open("w")

    theme.info(f"build log: {log_path}")
    stages = rt.plan(env_only=env_only) if recipe else [("install", "installing")]

    try:
        with Progress(
            SpinnerColumn(style="accent"),
            TextColumn("[bold]{task.fields[step]}"),
            BarColumn(bar_width=18, complete_style="accent", finished_style="ok"),
            TimeElapsedColumn(),
            TextColumn("[faint]{task.description}"),
            console=theme.console(),
            transient=False,
        ) as progress:
            task = progress.add_task("", total=len(stages), step=stages[0][0])
            started: list[str] = []

            def on_step(key: str, text: str) -> None:
                if started:
                    progress.advance(task)
                started.append(key)
                progress.update(task, step=key, description=text)
                log_file.write(f"== {key}: {text}\n")

            def on_progress(line: str) -> None:
                log_file.write(line + "\n")
                log_file.flush()
                # One live line of context, trimmed: the build prints thousands.
                progress.update(task, description=_trim(line))

            if recipe:
                rt.install(
                    on_progress=on_progress,
                    sources_dir=sources_dir,
                    force=force,
                    env_only=env_only,
                    on_step=on_step,
                )
            else:
                on_step(*stages[0])
                rt.install(on_progress=on_progress, sources_dir=sources_dir, force=force)
            progress.advance(task)
            progress.update(task, step="done", description="environment only" if env_only else "done")
    finally:
        log_file.close()

    st = rt.status()
    if st.installed or env_only:
        theme.ok("Runtime built", str(st.path or ""))
    else:
        raise TandemError(
            "The build finished but the runtime still looks incomplete: " + "; ".join(st.problems),
            hint=f"The full log is at {log_path}.",
        )


@app.command("shell", help="Open a shell inside the runtime environment.")
def shell(planner: str = _PLANNER, profile_name: str = _PROFILE) -> None:
    _, rt = _recipe_runtime(planner, profile_name)
    rt.require_ready()
    theme.info(f"Entering the runtime at {rt.root}. Type `exit` to leave.")
    subprocess.call(rt.shell_command(), cwd=str(rt.workdir))


@app.command("python", help="Print the runtime's Python interpreter path.")
def python_(planner: str = _PLANNER, profile_name: str = _PROFILE) -> None:
    _, rt = _recipe_runtime(planner, profile_name)
    typer.echo(str(rt.python()))


@app.command("run", help="Run a command inside the runtime environment.")
def run(
    args: list[str] = typer.Argument(..., help="Command and arguments, e.g. cutamp-demo --motion_plan"),
    planner: str = _PLANNER,
    profile_name: str = _PROFILE,
) -> None:
    _, rt = _recipe_runtime(planner, profile_name)
    rt.require_ready()
    raise typer.Exit(subprocess.call(rt.command(list(args)), cwd=str(rt.workdir)))


@app.command("clean", help="Delete the runtime directory.")
def clean(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
    planner: str = _PLANNER,
    profile_name: str = _PROFILE,
) -> None:
    name, rt = planner_runtime(planner=planner, profile_name=profile_name)
    root = Path(rt.status().path) if rt is not None and rt.status().path else None
    if root is None or not root.is_dir():
        theme.info("Nothing to clean.")
        return
    size_gb = sum(f.stat().st_size for f in root.rglob("*") if f.is_file() and not f.is_symlink()) / 1e9
    theme.warn(f"This deletes {root} ({size_gb:.1f} GB), the {name} runtime. Rebuilding takes 5–20 minutes.")
    if not yes and not typer.confirm("Delete the runtime?", default=False):
        raise typer.Abort()
    rt.uninstall()
    theme.ok("Runtime deleted")


@app.command("path", help="Print the runtime directory.")
def path_(planner: str = _PLANNER, profile_name: str = _PROFILE) -> None:
    name, rt = planner_runtime(planner=planner, profile_name=profile_name)
    if rt is None:
        raise TandemError(f"The {name!r} planner is pure Python, so it has no runtime directory.")
    typer.echo(str(rt.status().path))


def _trim(line: str, width: int = 64) -> str:
    line = line.strip()
    if len(line) <= width:
        return line
    return "…" + line[-(width - 1) :]


def _pixi_installed() -> bool:
    from tandem.core.probe import find_pixi

    return find_pixi() is not None


def install_pixi(log=None) -> None:
    """Install pixi with the official script. Only ever called after explicit consent."""
    import shutil

    if _pixi_installed():
        return
    if not shutil.which("curl"):
        raise TandemError(
            "curl is not installed, so pixi cannot be fetched.",
            hint="Install curl, or install pixi yourself: https://pixi.sh",
        )
    proc = subprocess.Popen(
        ["bash", "-c", "curl -fsSL https://pixi.sh/install.sh | bash"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PIXI_NO_PATH_UPDATE": os.environ.get("PIXI_NO_PATH_UPDATE", "")},
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if log:
            log(line.rstrip("\n"))
    if proc.wait() != 0:
        raise TandemError(
            "The pixi installer failed.",
            hint="Install it yourself from https://pixi.sh and re-run `tandem init`.",
        )
    if not _pixi_installed():
        raise TandemError(
            "pixi installed but is not on PATH.",
            hint="Add ~/.pixi/bin to your PATH and re-run `tandem init`.",
        )
