"""`tandem ui` — serve the web UI."""

from __future__ import annotations

import socket
import threading
import webbrowser

import typer

from tandem.cli import theme
from tandem.core import profiles
from tandem.core import settings as settings_mod
from tandem.core.errors import TandemError, one_line


def ui(
    port: int = typer.Option(None, "--port", "-p", help="Port to serve on."),
    host: str = typer.Option(None, "--host", help="Address to bind."),
    no_open: bool = typer.Option(False, "--no-open", help="Do not open a browser."),
    profile_name: str = typer.Option(None, "--profile", help="Profile to open on."),
) -> None:
    """Browse profiles, play the camera videos, inspect the plots, and run collection.

    Works with no GPU, no robot and no cameras — everything except starting a session is
    pure file reading, so this is the command to run on a laptop.
    """
    serve(port=port, host=host, open_browser=not no_open, profile_name=profile_name)


def serve(
    *,
    port: int | None = None,
    host: str | None = None,
    open_browser: bool = True,
    profile_name: str | None = None,
) -> None:
    """Serve the UI. Kept separate from the Typer command so other commands can call it
    without going through argument objects that only mean something to the CLI."""
    import uvicorn

    cfg = settings_mod.load()
    host = host or cfg.ui.host
    port = port or cfg.ui.port

    if not profiles.list_names():
        raise TandemError(
            "There are no profiles yet, so there is nothing to show.",
            hint="Run `tandem init` (or `tandem init --viz-only` on a laptop).",
        )
    if profile_name and profile_name != cfg.active_profile:
        # The page works on the active profile -- its switcher, the runtime chip, the collect page's
        # sessions all read it -- so the profile asked for is made the active one, and said to be. Only
        # passing it to the page left `tandem collect bread --web` collecting under whichever profile was
        # active, while saying it was collecting under bread.
        profiles.load(profile_name, require_installed=False)  # an unknown name: the not-found error, with the known ones
        cfg.active_profile = profile_name
        settings_mod.save(cfg)
        theme.info(f"Active profile is now {profile_name!r}")

    port = _free_port(host, port)
    url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"

    theme.banner("ui")
    active = profile_name or cfg.active_profile
    theme.kv(
        [
            ("profile", active),
            ("data root", cfg.resolved_data_root()),
            ("runtime", _runtime_note(cfg, active)),
        ]
    )
    theme.blank()
    theme.ok("Serving", url)
    theme.info("Ctrl-C to stop")
    theme.blank()

    if open_browser and cfg.ui.open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    from tandem.server.app import create_app

    config = uvicorn.Config(
        create_app(),
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
        # An open collect page holds an event stream until its session has ended, and uvicorn waits for
        # every connection before it lets the app shut down. Without a bound, Ctrl-C sat there saying
        # nothing -- and the second Ctrl-C people then pressed skipped the app's shutdown altogether.
        timeout_graceful_shutdown=CONNECTION_GRACE,
    )
    server = _server(config)
    try:
        server.run()
    except KeyboardInterrupt:
        # uvicorn re-raises the Ctrl-C it caught once it has shut down; it has done its job by then.
        pass
    if not server.started:
        raise typer.Exit(code=3)


# How long, after Ctrl-C, open connections get to finish before they are cut. The wait that matters
# comes after it: the app's own shutdown, which ends every session and waits for it (server/app.py).
CONNECTION_GRACE = 5


def _server(config):
    """uvicorn's server, with the sessions told to stop as soon as it starts to shut down.

    Not only at the app's shutdown, which comes after every connection has closed: a collect page's
    event stream closes when its session has ended, so the sessions are what have to go first. A
    second Ctrl-C while they park and merge gives up the wait, as it does in `tandem collect`.
    """
    import signal

    import uvicorn

    from tandem.core import session as session_mod

    class Server(uvicorn.Server):
        async def shutdown(self, sockets=None) -> None:
            stopping = session_mod.manager().stop_all()
            if stopping:
                theme.busy(
                    f"Stopping {len(stopping)} session(s): parking the arm, finishing the episode merges",
                    "Ctrl-C again to quit now, leaving the arm where it is",
                )
            await super().shutdown(sockets)

        def handle_exit(self, sig, frame) -> None:
            if self.should_exit and sig == signal.SIGINT:
                session_mod.manager().abandon()
            super().handle_exit(sig, frame)

    return Server(config)


def _runtime_note(cfg, profile_name: str) -> str:
    """Whether the runtime of the planner this profile uses is there: the difference between a
    workstation that can collect and a laptop that can only look."""
    from tandem.cli.runtime import planner_runtime

    try:
        planner, runtime = planner_runtime(profile_name=profile_name, settings=cfg)
    except TandemError as exc:
        return f"unknown — {one_line(exc.message)}"
    if runtime is None:
        return f"{planner} is pure Python — collection available"
    if runtime.status().installed:
        return f"{planner} ready — collection available"
    return f"{planner} not built — visualization only"


def _free_port(host: str, port: int, tries: int = 20) -> int:
    """Step to the next free port rather than dying on 'address already in use'."""
    bind_host = "127.0.0.1" if host == "0.0.0.0" else host
    for offset in range(tries):
        candidate = port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((bind_host, candidate))
            except OSError:
                continue
        if offset:
            theme.warn(f"Port {port} is busy — using {candidate} instead")
        return candidate
    raise TandemError(
        f"No free port between {port} and {port + tries}.",
        hint="Pass --port explicitly.",
    )
