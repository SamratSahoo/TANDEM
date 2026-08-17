"""`tandem ui` — serve the web UI."""

from __future__ import annotations

import socket
import threading
import webbrowser

import typer

from tandem.cli import theme
from tandem.core import profiles
from tandem.core import runtime as runtime_mod
from tandem.core import settings as settings_mod
from tandem.core.errors import TandemError


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

    port = _free_port(host, port)
    url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"

    theme.banner("ui")
    active = profile_name or cfg.active_profile
    theme.kv(
        [
            ("profile", active),
            ("data root", cfg.resolved_data_root()),
            ("runtime", _runtime_note(cfg)),
        ]
    )
    theme.blank()
    theme.ok("Serving", url)
    theme.info("Ctrl-C to stop")
    theme.blank()

    if open_browser and cfg.ui.open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    from tandem.server.app import create_app

    app = create_app(initial_profile=active)
    uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)


def _runtime_note(cfg) -> str:
    status = runtime_mod.Runtime(cfg.resolved_runtime_dir()).status()
    if status.ready:
        return "ready — collection available"
    return "not built — visualization only"


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
