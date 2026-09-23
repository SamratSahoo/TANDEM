"""`tandem runtime` — the GPU environment that `tandem init` builds."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime

import typer

from tandem.cli import theme
from tandem.core import paths
from tandem.core import runtime as runtime_mod
from tandem.core import settings as settings_mod
from tandem.core.errors import TandemError

app = typer.Typer(no_args_is_help=True, help="The GPU runtime that `tandem init` builds.")


@app.command("status", help="What is built, from which sources.")
def status(as_json: bool = typer.Option(False, "--json", help="Machine-readable output.")) -> None:
    runtime = runtime_mod.default()
    st = runtime.status()

    vendor = st.vendor or runtime.pending_vendor()

    if as_json:
        payload = st.to_dict()
        payload["root"] = str(runtime.root)
        payload["vendor"] = vendor
        payload["vendor_installed"] = st.vendor is not None
        typer.echo(json.dumps(payload, indent=2))
        return

    theme.blank()
    theme.heading("runtime", str(runtime.root))
    theme.kv(
        [
            ("sources", "present" if st.sources_present else "missing"),
            ("pixi env", "built" if st.env_built else "not built"),
            ("curobo kernels", "compiled" if st.kernels_built else "not compiled"),
            ("built", st.built_at),
        ]
    )

    if vendor:
        theme.blank()
        theme.heading(
            "vendored sources" if st.vendor else "sources that will be installed",
            "" if st.vendor else "shipped inside this package",
        )
        table = theme.table("component", "version", "commit", "upstream")
        for name, meta in sorted(vendor.items()):
            if not isinstance(meta, dict) or "commit" not in meta:
                continue
            table.add_row(
                name,
                str(meta.get("version", "")),
                str(meta.get("commit", ""))[:12],
                f"[faint]{meta.get('url', '')}[/faint]",
            )
        theme.console().print(table)
        patched = [n for n, m in vendor.items() if isinstance(m, dict) and m.get("patches")]
        if patched:
            theme.info(
                "patched: " + ", ".join(patched),
                "see tools/patches/ for what and why",
            )

    theme.blank()
    if st.ready:
        theme.ok("Runtime is ready")
    else:
        for problem in st.problems or []:
            theme.warn(problem)
        theme.next_steps([("tandem runtime build", "build or repair it")])


@app.command("build", help="Build (or repair) the runtime.")
def build(
    force: bool = typer.Option(False, "--force", help="Re-copy the vendored sources before building."),
    env_only: bool = typer.Option(False, "--env-only", help="Only solve the pixi environment."),
) -> None:
    cfg = settings_mod.load()
    runtime = runtime_mod.Runtime(cfg.resolved_runtime_dir())
    run_build(runtime, force=force, env_only=env_only)


def run_build(runtime: runtime_mod.Runtime, *, force: bool = False, env_only: bool = False) -> None:
    """Shared by `runtime build` and `init`, so they cannot drift apart."""
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    paths.ensure_dir(paths.log_dir())
    log_path = paths.log_dir() / f"runtime-build-{datetime.now():%Y%m%d-%H%M%S}.log"
    log_file = log_path.open("w")

    theme.info(f"build log: {log_path}")

    def make_logger(task_id, progress):
        def log(line: str) -> None:
            log_file.write(line + "\n")
            log_file.flush()
            # One live line of context, trimmed: the build prints thousands.
            progress.update(task_id, description=_trim(line))

        return log

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
            task = progress.add_task("", total=3, step="sources")

            progress.update(task, step="sources", description="copying vendored trees")
            runtime.materialize(paths.vendor_dir(), force=force, log=make_logger(task, progress))
            progress.advance(task)

            progress.update(task, step="pixi env", description="solving")
            runtime.build_env(log=make_logger(task, progress))
            progress.advance(task)

            if env_only:
                progress.update(task, step="done", description="environment only")
                progress.advance(task)
            else:
                progress.update(
                    task,
                    step="planners",
                    description="compiling cuRobo CUDA kernels — this takes 5–20 minutes the first time",
                )
                runtime.build_planners(log=make_logger(task, progress))
                progress.advance(task)
                progress.update(task, description="done")
    finally:
        log_file.close()

    st = runtime.status()
    if st.ready or env_only:
        theme.ok("Runtime built", str(runtime.root))
    else:
        raise TandemError(
            "The build finished but the runtime still looks incomplete: "
            + "; ".join(st.problems or []),
            hint=f"The full log is at {log_path}.",
        )


@app.command("shell", help="Open a shell inside the runtime environment.")
def shell() -> None:
    runtime = runtime_mod.default()
    runtime.require_ready()
    from tandem.core.probe import find_pixi

    pixi = find_pixi()
    theme.info(f"Entering the runtime at {runtime.root}. Type `exit` to leave.")
    subprocess.call(
        [str(pixi), "shell", "--manifest-path", str(runtime.tiptop_dir / "pixi.toml")],
        cwd=str(runtime.tiptop_dir),
    )


@app.command("python", help="Print the runtime's Python interpreter path.")
def python_() -> None:
    typer.echo(str(runtime_mod.default().python()))


@app.command("run", help="Run a command inside the runtime environment.")
def run(
    args: list[str] = typer.Argument(..., help="Command and arguments, e.g. cutamp-demo --motion_plan"),
) -> None:
    runtime = runtime_mod.default()
    runtime.require_ready()
    raise typer.Exit(subprocess.call(runtime.command(list(args)), cwd=str(runtime.tiptop_dir)))


@app.command("clean", help="Delete the runtime directory.")
def clean(yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation.")) -> None:
    import shutil

    runtime = runtime_mod.default()
    if not runtime.root.is_dir():
        theme.info("Nothing to clean.")
        return
    size_gb = sum(f.stat().st_size for f in runtime.root.rglob("*") if f.is_file()) / 1e9
    theme.warn(f"This deletes {runtime.root} ({size_gb:.1f} GB). Rebuilding takes 5–20 minutes.")
    if not yes and not typer.confirm("Delete the runtime?", default=False):
        raise typer.Abort()
    shutil.rmtree(runtime.root)
    theme.ok("Runtime deleted")


@app.command("path", help="Print the runtime directory.")
def path_() -> None:
    typer.echo(str(runtime_mod.default().root))


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
