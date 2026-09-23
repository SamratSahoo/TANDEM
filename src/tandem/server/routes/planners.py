"""The planner and human-executor catalogs, for the settings page.

The payloads are the ones `tandem planners list --json` and `tandem executors list --json` print,
built by the same functions (``cli/planners.py``, ``cli/executors.py``), so the page and the terminal
cannot disagree about what is installed or which one a profile uses. Choosing one is the same call
as `tandem planners use` / `tandem executors use`.

Installing a planner is deliberately not an endpoint. It fetches sources and builds a GPU environment
for up to twenty minutes, may first need pixi installed into the home directory -- which takes the
person's consent -- and prints thousands of lines worth reading when it fails: all things a terminal
does well and a request handler does not, and the server has no job runner to hand them to. So the
catalog carries, for every planner that is not installed or is outdated, the exact command that
installs it (``install_command``), and the page shows it.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["planners"])


class ProfileBody(BaseModel):
    # The profile to change; the active one when absent.
    profile: str | None = None


@router.get("/planners")
async def list_planners(profile: str | None = None) -> dict:
    from tandem.cli import planners as planners_cli

    return planners_cli.catalog_payload(profile_name=profile)


@router.get("/planners/{name}")
async def planner_info(name: str, profile: str | None = None) -> dict:
    from tandem.cli import planners as planners_cli

    return planners_cli.info_payload(name, profile_name=profile)


@router.post("/planners/{name}/use")
async def use_planner(name: str, body: ProfileBody | None = None) -> dict:
    """Make a profile plan with ``name``. A planner that is not installed is accepted and said so."""
    from tandem.cli import planners as planners_cli

    return planners_cli.use_planner(name, profile_name=(body or ProfileBody()).profile)


@router.post("/planners/{name}/default")
async def make_default_planner(name: str) -> dict:
    """Make ``name`` the planner every new profile gets. No profile is changed."""
    from tandem.cli import planners as planners_cli
    from tandem.core import settings as settings_mod

    planners_cli.set_default_planner(name)
    return {"default_planner": settings_mod.load().default_planner}


@router.get("/executors")
async def list_executors(profile: str | None = None) -> dict:
    from tandem.cli import executors as executors_cli

    return executors_cli.catalog_payload(profile_name=profile)


@router.post("/executors/{name}/use")
async def use_executor(name: str, body: ProfileBody | None = None) -> dict:
    """Make a profile's human phases run with ``name``. One not ready on this machine is accepted, and said so."""
    from tandem.cli import executors as executors_cli

    return executors_cli.use_executor(name, profile_name=(body or ProfileBody()).profile)
