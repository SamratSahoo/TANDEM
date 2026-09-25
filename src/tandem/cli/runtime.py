"""`tandem runtime` — the runtime of the planner the active profile uses.

Every command here acts on one planner's runtime: the planner the active profile names (or
``--profile``'s, or ``--planner``), found through the planner registry. For TiPToP that is the GPU
runtime `tandem init` builds: its pinned sources fetched, its pixi environment solved and cuRobo's
kernels compiled. A planner that is pure Python has no runtime, and says so.

Choosing a planner, and installing or removing any planner's runtime by name, is `tandem planners`
(``cli/planners.py``). This group stays for what is specific to the one in use -- entering its
environment, running a command in it -- and for the scripts already written against it. The pixi
consent flow and the build with progress live here and are shared by both, and by `tandem init`.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich.markup import escape

from tandem.cli import theme
from tandem.core import paths
from tandem.core import settings as settings_mod
from tandem.core.errors import TandemError

app = typer.Typer(no_args_is_help=True, help="The runtime of the planner the active profile uses.")

_PLANNER = typer.Option(None, "--planner", help="A planner's name, instead of the active profile's.")
_PROFILE = typer.Option(
    None, "--profile", "-p", help="Use this profile's planner instead of the active one's."
)
_RAW = typer.Option(
    False,
    "--raw",
    help="Leave the planner's own config as it ships (TiPToP: its stock tiptop.yml and calibration file) "
    "instead of pointing it at this machine's rig.",
)


# --------------------------------------------------------------------------- which runtime


def active_planner(profile_name: str | None = None) -> str:
    """The planner a profile names -- the active profile unless one is named.

    A profile that does not exist yet has the planner a new one would get: the machine's default
    (``default_planner``, which `tandem planners default NAME` sets). That is the one case with
    no profile to ask, and it is the first thing `tandem init` meets: it builds the runtime before it
    creates the first profile. A profile that exists but does not load is an error, not a reason to
    guess: installing the wrong planner's runtime is twenty minutes and 25 GB spent on nothing.
    """
    from tandem.core import profiles

    cfg = settings_mod.load()
    name = profile_name or cfg.active_profile
    if not profiles.exists(name):
        return cfg.default_planner
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
        "mismatched": list(status.outdated(info.sources)),
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
                    "ref": s.wanted.ref or None,
                    "installed": s.commit,
                    "installed_ref": s.ref,
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
                    "ref": p.ref or None,
                    "installed": None,
                    "installed_ref": None,
                    "present": False,
                    "origin": None,
                    "verified": None,
                    "current": False,
                }
                for p in info.sources
            ],
            vendor={
                n: {
                    "url": e.get("url") or "",
                    "commit": e.get("commit"),
                    "ref": e.get("ref"),
                    "version": str(e.get("commit"))[:7],
                }
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
        table = theme.table("source", "pinned", "branch", "installed", "upstream")
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
            branch = escape(source.get("ref") or "") or "[faint]—[/faint]"
            table.add_row(
                source["name"], source["commit"][:12], branch, shown, f"[faint]{source['url']}[/faint]"
            )
        theme.console().print(table)

    theme.blank()
    for note in payload.get("notes") or []:
        theme.info(note)
    if payload["ready"]:
        theme.ok("Runtime is ready")
    else:
        for problem in payload["problems"]:
            theme.warn(problem)
        theme.next_steps(
            [
                (f"tandem planners install {payload['planner']}", "build or repair it"),
                ("tandem planners list", "every planner, and which are installed"),
            ]
        )


@app.command(
    "build",
    help="Build (or repair) the runtime of the active profile's planner. `tandem planners install NAME` "
    "does the same for any planner, by name.",
)
def build(
    force: bool = typer.Option(False, "--force", help="Fetch every source again before building."),
    env_only: bool = typer.Option(False, "--env-only", help="Stop once the environment is solved."),
    sources: Path = typer.Option(
        None,
        "--sources",
        help="Take the planner's sources from this directory of checkouts or exports instead of fetching "
        "them from GitHub (default: $TANDEM_PLANNER_SOURCES). `tandem planners bundle NAME --out DIR` makes "
        "one, on a machine with network. The environment is still downloaded (conda-forge, PyPI).",
        file_okay=False,
    ),
    planner: str = _PLANNER,
    profile_name: str = _PROFILE,
) -> None:
    name, rt = planner_runtime(planner=planner, profile_name=profile_name)
    if rt is None:
        theme.ok(f"The {name!r} planner is pure Python", "there is nothing to build")
        return
    run_build(rt, force=force, env_only=env_only, sources_dir=sources, planner=name)


def run_build(
    rt,
    *,
    force: bool = False,
    env_only: bool = False,
    sources_dir: Path | None = None,
    planner: str | None = None,
) -> None:
    """Shared by `runtime build`, `planners install` and `init`, so they cannot drift apart.

    ``planner`` is whose runtime this is, for the error: a failed build says which command re-runs
    THIS build and which log has it, not `tandem runtime build`, which builds the active profile's.
    """
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
    except TandemError as exc:
        name = planner or getattr(getattr(rt, "recipe", None), "planner", None)
        if name:
            command = f"tandem planners install {name}" + (f" --sources {sources_dir}" if sources_dir else "")
            again = f"`{command}`"
        else:
            again = "the same command"
        raise TandemError(
            exc.message,
            hint=f"The full log is {log_path}. Re-run {again} once the cause is fixed"
            + (f"; {exc.hint}" if exc.hint else "."),
        ) from exc
    finally:
        log_file.close()

    st = rt.status()
    if st.installed or env_only:
        theme.ok("Runtime built", str(st.path or ""))
        say_notes(st)
    else:
        raise TandemError(
            "The build finished but the runtime still looks incomplete: " + "; ".join(st.problems),
            hint=f"The full log is at {log_path}.",
        )


def say_notes(status) -> None:
    """What a runtime that works would still like done -- an optional part this machine cannot have yet, with
    the fix -- said where the person who installs it reads: the install's own output."""
    for note in getattr(status, "notes", ()) or ():
        theme.warn(note)


def optional_steps_to_run(rt) -> list[str]:
    """The optional build steps an install of ``rt`` would run now (a camera SDK installed since, say).

    A runtime that is otherwise built is built again for these, which runs them and skips the rest.
    """
    from tandem.planners.runtime import RecipeRuntime

    return rt.optional_to_run() if isinstance(rt, RecipeRuntime) else []


@app.command("shell", help="Open a shell inside the runtime environment, pointed at this machine's rig.")
def shell(planner: str = _PLANNER, profile_name: str = _PROFILE, raw: bool = _RAW) -> None:
    name, rt = _recipe_runtime(planner, profile_name)
    rt.require_ready()
    env = command_env(name, raw=raw)
    theme.info(f"Entering the runtime at {rt.root}. Type `exit` to leave.")
    subprocess.call(rt.shell_command(), cwd=str(rt.workdir), env=env)


@app.command("python", help="Print the runtime's Python interpreter path.")
def python_(planner: str = _PLANNER, profile_name: str = _PROFILE) -> None:
    _, rt = _recipe_runtime(planner, profile_name)
    typer.echo(str(rt.python()))


@app.command(
    "run",
    help="Run a command inside the runtime environment, e.g. `tandem runtime run cutamp-demo --motion_plan`. "
    "It reaches this machine's robot and cameras (the rig), unless --raw. tandem's own options go before "
    "the command; everything after it is the command's.",
    # Everything from the command on is the command's, options included: `cutamp-demo --motion_plan` is not
    # an option of tandem's. (`--` before the command still works, and is no longer needed.)
    context_settings={"allow_interspersed_args": False, "ignore_unknown_options": True},
)
def run(
    args: list[str] = typer.Argument(
        ..., metavar="COMMAND [ARGS]...", help="The command and its arguments, e.g. cutamp-demo --motion_plan."
    ),
    planner: str = _PLANNER,
    profile_name: str = _PROFILE,
    raw: bool = _RAW,
) -> None:
    name, rt = _recipe_runtime(planner, profile_name)
    rt.require_ready()
    env = command_env(name, raw=raw)
    raise typer.Exit(subprocess.call(rt.command(list(args)), cwd=str(rt.workdir), env=env))


def command_env(planner: str, *, raw: bool) -> dict[str, str]:
    """The environment of a command run in ``planner``'s runtime: this one, and what the planner needs to find
    this machine's rig (``registry.runtime_env``) -- nothing added with ``raw``.

    Said on stderr, in one line, so a person knows which robot the command will reach, and the command's
    own output stays clean for a pipe.

    A machine with no rig.yml yet runs it as ``raw`` does, and says so: a config rendered from the rig's
    defaults names a robot nobody set up and no cameras, so a calibration script would fail on a missing
    camera, having created an empty calibration file on the way. A command that needs no robot (a demo)
    runs as it would have.
    """
    env = dict(os.environ)
    if raw:
        return env
    from tandem.core import rig as rig_mod
    from tandem.planners import registry

    if not rig_mod.exists():
        theme.err_console().print(
            "[faint]· no rig on this machine yet, so this runs with the planner's own config (as --raw); "
            "`tandem init` sets the rig up, or `tandem rig set robot.host HOST` and `tandem rig set "
            "cameras.hand.serial SERIAL`[/faint]",
            highlight=False,
        )
        return env
    rig = rig_mod.load()
    added = registry.runtime_env(planner, rig=rig, settings=settings_mod.load())
    if added:
        theme.err_console().print(
            f"[faint]· with this machine's rig: {escape(rig.summary())} ({escape(str(rig.file()))}); "
            "--raw runs it with the planner's own config[/faint]",
            highlight=False,
        )
    env.update(added)
    return env


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


def needs_pixi(rt: Any) -> bool:
    """Whether building ``rt`` runs pixi: a recipe's runtime that declares an environment."""
    from tandem.planners.runtime import RecipeRuntime

    return isinstance(rt, RecipeRuntime) and rt.recipe.environment is not None


def ensure_pixi(title: str, *, ask: bool, allowed: bool) -> None:
    """The consent flow before pixi is installed into the home directory: `init`'s and `planners install`'s.

    pixi's installer is `curl | bash` into ~/.pixi. Left to itself it would also append ~/.pixi/bin to
    the shell's rc file; tandem tells it not to (``install_pixi``), since it finds pixi in ~/.pixi/bin
    without that, so what the question says is what happens. That is done on an explicit yes and
    nothing less: ``ask`` puts the question to the person at the
    terminal, and ``allowed`` is a yes given in advance (``--yes``, or `tandem init`'s own "accept
    every default"). With neither, the answer is an error saying how to give it -- never a silent
    install, and never a build that fails twenty seconds in because the tool it needs is missing.
    """
    if _pixi_installed():
        return
    theme.info(f"pixi is the environment manager {title}'s planner stack needs.")
    if _path_update_skipped():
        theme.info("It installs to ~/.pixi and touches nothing else: your shell's rc file is left alone.")
    else:
        theme.info("It installs to ~/.pixi, and (PIXI_NO_PATH_UPDATE is empty) adds ~/.pixi/bin to your shell's rc file.")
    if ask:
        if not typer.confirm("  Install pixi now?", default=True):
            raise TandemError(
                "pixi is required to build the runtime.",
                hint="Install it from https://pixi.sh, then run the command again.",
            )
    elif not allowed:
        raise TandemError(
            f"pixi is not installed, and the {title} runtime is built in a pixi environment.",
            hint="Run again with --yes to let tandem install it (into ~/.pixi), or install it yourself "
            "from https://pixi.sh.",
        )
    theme.busy("Installing pixi")
    install_pixi(log=lambda _line: None)
    if _path_update_skipped():
        theme.ok("pixi installed", "in ~/.pixi/bin, where tandem finds it; add that to PATH to run pixi yourself")
    else:
        theme.ok("pixi installed")


def _path_update_skipped() -> bool:
    """Whether the pixi installer will leave the shell's rc file alone (it does unless told otherwise).

    tandem asks it to, by default: tandem finds pixi in ~/.pixi/bin itself, and consent given to
    "installs to ~/.pixi" is not consent to an edited ~/.zshrc. PIXI_NO_PATH_UPDATE set to an empty
    string by the person is their choice to let it edit the file.
    """
    return os.environ.get("PIXI_NO_PATH_UPDATE", "1") != ""


def install_pixi(log=None) -> None:
    """Install pixi with the official script. Only ever called after explicit consent.

    With PIXI_NO_PATH_UPDATE=1 unless the person set it themselves: see ``_path_update_skipped``.
    """
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
        # A byte that is not UTF-8 (a progress bar, a localized curl error) must not end the install
        # with a UnicodeDecodeError halfway through; the planners' own build streams read the same way.
        encoding="utf-8",
        errors="backslashreplace",
        env={**os.environ, "PIXI_NO_PATH_UPDATE": os.environ.get("PIXI_NO_PATH_UPDATE", "1")},
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
