"""The FastAPI application behind `tandem ui`.

Serves a single-page app from ``server/static`` with no build step — no Node is involved at
install time or at runtime — plus a small JSON API over the same core modules the CLI uses.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from tandem import __version__
from tandem.core.errors import ProfileError, ProfileInvalid, SessionConflict, TandemError
from tandem.server.routes import planners as planners_routes
from tandem.server.routes import profiles as profiles_routes
from tandem.server.routes import rig as rig_routes
from tandem.server.routes import sessions as sessions_routes
from tandem.server.routes import settings as settings_routes
from tandem.server.routes import trajectories as trajectories_routes

STATIC_DIR = Path(__file__).resolve().parent / "static"
#: The web app's own files: revalidated on every load (see ``spa``).
NO_CACHE = {"Cache-Control": "no-cache"}

log = logging.getLogger("tandem.server")


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """On the way out, end every session this server started, and wait (bounded) for them to finish.

    A session drives the robot on daemon threads, so a server that simply exits kills them where they
    stand: the arm is not parked, the planner never lets go of the robot and the cameras, and a merge
    in flight is cut off with the trial's legs half moved. Nothing called `SessionManager.shutdown`,
    and that is what exiting `tandem ui` did. Waited for off the event loop, which stays free to
    close the remaining connections.
    """
    from tandem.core import session as session_mod

    yield
    report_still_ending(await asyncio.to_thread(session_mod.manager().shutdown))


def report_still_ending(sessions: Iterable) -> None:
    """Say which sessions were still ending as the server exited, and what that leaves to be done."""
    for session in sessions:
        log.warning(
            "session %s (%s) was still ending when the server exited; its arm may not be parked, and "
            "`tandem traj merge` joins a trial whose merge did not finish",
            session.id,
            session.profile.name,
        )


def close_streams(app: FastAPI) -> None:
    """End every session event stream ``app`` is serving, and any it is asked for from now on.

    For the server to call as it starts shutting down. A collect page's stream otherwise stays open
    until its session has ended -- a park and a merge of gigabytes of video, far longer than the
    server gives open connections -- and the server then cancelled it, printing an error and a
    traceback on the terminal of every Ctrl-C that went exactly as it should.
    """
    app.state.closing.set()


def create_app() -> FastAPI:
    # No "initial profile": the page works on the active one, and `tandem ui --profile NAME` makes NAME
    # the active one before serving (cli/ui.py).
    app = FastAPI(
        title="tandem",
        version=__version__,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=_lifespan,
    )
    # Set by `close_streams`. A threading.Event, not an asyncio one: this app outlives any one event loop
    # (a test client runs it on a loop of its own each time), and the streams only ever poll it.
    app.state.closing = threading.Event()

    @app.exception_handler(SessionConflict)
    async def _session_conflict(_request: Request, exc: SessionConflict) -> JSONResponse:
        # 409, not 500: "you cannot do that right now" is a normal answer for a UI whose
        # buttons race the robot's state.
        return JSONResponse(status_code=409, content={"error": exc.message, "hint": exc.hint})

    @app.exception_handler(ProfileInvalid)
    async def _profile_invalid(_request: Request, exc: ProfileInvalid) -> JSONResponse:
        # 422, not 404: the profile is there, and the editor that gets this can offer to fix it.
        return JSONResponse(status_code=422, content={"error": exc.message, "hint": exc.hint})

    @app.exception_handler(ProfileError)
    async def _profile_error(_request: Request, exc: ProfileError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"error": exc.message, "hint": exc.hint})

    @app.exception_handler(TandemError)
    async def _tandem_error(_request: Request, exc: TandemError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": exc.message, "hint": exc.hint})

    app.include_router(profiles_routes.router, prefix="/api")
    app.include_router(rig_routes.router, prefix="/api")
    app.include_router(trajectories_routes.router, prefix="/api")
    app.include_router(sessions_routes.router, prefix="/api")
    app.include_router(settings_routes.router, prefix="/api")
    app.include_router(planners_routes.router, prefix="/api")

    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True, "version": __version__}

    if STATIC_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa(full_path: str):
            # An unmatched /api/ path is a client bug, not a page. Returning the HTML app
            # there would make a typo'd endpoint look like a successful request whose JSON
            # failed to parse.
            if full_path == "api" or full_path.startswith("api/"):
                return JSONResponse(status_code=404, content={"error": f"No such endpoint: /{full_path}"})
            # Serve real files directly; everything else falls through to index.html so the
            # client-side router owns the URL space.
            #
            # Never from a cache without asking: the pages are ES modules importing each other, and with
            # no Cache-Control a browser keeps each one fresh for a while by guesswork -- so after an
            # upgrade it ran a new page against an old module that lacked what the page imported, and
            # showed a blank page. Asking costs a 304 per file (the ETag), on a local server.
            candidate = (STATIC_DIR / full_path).resolve()
            if full_path and STATIC_DIR in candidate.parents and candidate.is_file():
                return FileResponse(candidate, headers=NO_CACHE)
            return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)

    return app
