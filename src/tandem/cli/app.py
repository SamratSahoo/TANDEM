"""The `tandem` command tree."""

from __future__ import annotations

import logging
import os
import sys

import typer

from tandem import __version__
from tandem.cli import theme
from tandem.core.errors import TandemError

app = typer.Typer(
    name="tandem",
    help="Human-in-the-loop TAMP data collection for real robots.",
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
    # We render our own error panels; typer's syntax-highlighted traceback is for developers,
    # not for someone standing next to a robot.
    pretty_exceptions_enable=False,
)


def _version_callback(value: bool) -> None:
    if value:
        theme.console().print(f"tandem [accent]{__version__}[/accent]")
        raise typer.Exit()


@app.callback()
def root(
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True, help="Show the version and exit."
    ),
    no_color: bool = typer.Option(False, "--no-color", help="Disable colour (also honours $NO_COLOR)."),
    debug: bool = typer.Option(False, "--debug", help="Show full tracebacks instead of a friendly error."),
) -> None:
    if no_color:
        theme.set_no_color(True)
    if debug:
        os.environ["TANDEM_DEBUG"] = "1"
    _show_profile_notices()


class _NoticeHandler(logging.Handler):
    """A profile's notices (read in an older layout, and how to rewrite it) as a warning line on stderr.

    stderr, so a `--json` on stdout stays parseable; a handler of tandem's own, so the line looks like
    every other warning rather than a bare logging record.
    """

    def emit(self, record: logging.LogRecord) -> None:
        from rich.text import Text

        line = Text()
        line.append(f"  {theme.WARN} ", style="warn")
        line.append(record.getMessage())
        theme.err_console().print(line)


def _show_profile_notices() -> None:
    logger = logging.getLogger("tandem.core.profiles")
    if not any(isinstance(handler, _NoticeHandler) for handler in logger.handlers):
        logger.addHandler(_NoticeHandler(logging.WARNING))


# Subcommands are imported here (not at module top) so `tandem --help` stays fast and a
# broken optional dependency in one command cannot take down the whole CLI.
from tandem.cli import collect as _collect  # noqa: E402
from tandem.cli import config as _config  # noqa: E402
from tandem.cli import doctor as _doctor  # noqa: E402
from tandem.cli import executors as _executors  # noqa: E402
from tandem.cli import export as _export  # noqa: E402
from tandem.cli import init as _init  # noqa: E402
from tandem.cli import plan as _plan  # noqa: E402
from tandem.cli import planners as _planners  # noqa: E402
from tandem.cli import profile as _profile  # noqa: E402
from tandem.cli import runtime as _runtime  # noqa: E402
from tandem.cli import traj as _traj  # noqa: E402
from tandem.cli import ui as _ui  # noqa: E402

app.command("init", help="Set tandem up on this machine (onboarding wizard).")(_init.init)
app.command("doctor", help="Check that everything tandem needs is present and working.")(_doctor.doctor)
app.command("collect", help="Run a human-in-the-loop collection session.")(_collect.collect)
app.command("plan", help="Decompose a task from a photo, with no robot and no GPU.")(_plan.plan)
app.command("ui", help="Serve the web UI for collecting and visualizing trajectories.")(_ui.ui)
app.add_typer(_profile.app, name="profile", help="Create and manage collection profiles.")
app.add_typer(_config.app, name="config", help="Global settings and credentials.")
app.add_typer(_traj.app, name="traj", help="Inspect collected trajectories.")
app.add_typer(_export.app, name="export", help="Export trajectories to other dataset formats.")
app.add_typer(
    _planners.app,
    name="planners",
    help="The task and motion planners tandem can drive: list, install, choose, or scaffold your own.",
)
app.add_typer(
    _executors.app, name="executors", help="Who carries out a human phase: list them, choose one."
)
app.add_typer(
    _runtime.app, name="runtime", help="The runtime of the active profile's planner, which `tandem init` builds."
)


def main() -> None:
    """Entry point with the error boundary that keeps tracebacks off the screen."""
    try:
        app()
    except TandemError as exc:
        if os.environ.get("TANDEM_DEBUG"):
            raise
        theme.error_panel(exc.message, exc.hint)
        sys.exit(1)
    except KeyboardInterrupt:
        theme.err_console().print("\n[faint]interrupted[/faint]")
        sys.exit(130)
    except BrokenPipeError:
        # `tandem traj list | head` — not an error worth a panel.
        sys.exit(0)


if __name__ == "__main__":
    main()
