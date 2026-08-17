"""The FastAPI application behind `tandem ui`.

Serves a single-page app from ``server/static`` with no build step — no Node is involved at
install time or at runtime — plus a small JSON API over the same core modules the CLI uses.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from tandem import __version__
from tandem.core.errors import ProfileError, SessionConflict, TandemError
from tandem.server.routes import profiles as profiles_routes
from tandem.server.routes import sessions as sessions_routes
from tandem.server.routes import settings as settings_routes
from tandem.server.routes import trajectories as trajectories_routes

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(*, initial_profile: str | None = None) -> FastAPI:
    app = FastAPI(
        title="tandem",
        version=__version__,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.initial_profile = initial_profile

    @app.exception_handler(SessionConflict)
    async def _session_conflict(_request: Request, exc: SessionConflict) -> JSONResponse:
        # 409, not 500: "you cannot do that right now" is a normal answer for a UI whose
        # buttons race the robot's state.
        return JSONResponse(status_code=409, content={"error": exc.message, "hint": exc.hint})

    @app.exception_handler(ProfileError)
    async def _profile_error(_request: Request, exc: ProfileError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"error": exc.message, "hint": exc.hint})

    @app.exception_handler(TandemError)
    async def _tandem_error(_request: Request, exc: TandemError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": exc.message, "hint": exc.hint})

    app.include_router(profiles_routes.router, prefix="/api")
    app.include_router(trajectories_routes.router, prefix="/api")
    app.include_router(sessions_routes.router, prefix="/api")
    app.include_router(settings_routes.router, prefix="/api")

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
            candidate = (STATIC_DIR / full_path).resolve()
            if full_path and STATIC_DIR in candidate.parents and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(STATIC_DIR / "index.html")

    return app
