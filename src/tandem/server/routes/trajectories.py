"""Trajectory routes: index, per-frame series, media, relabel, delete."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from tandem.core import profiles as profiles_mod
from tandem.core import series as series_mod
from tandem.core import trajectories as traj_mod
from tandem.server.media import range_response

router = APIRouter(tags=["trajectories"])


class RelabelBody(BaseModel):
    status: str


@router.get("/trajectories")
async def list_trajectories(profile: str | None = None, status: str | None = None) -> dict:
    prof = profiles_mod.load(profile, require_installed=False)
    items = traj_mod.list_all(prof, status=status)
    return {
        "profile": prof.name,
        "counts": traj_mod.counts(prof),
        "target": prof.task.target_episodes,
        "trajectories": [t.to_dict() for t in items],
    }


@router.get("/trajectories/{profile}/{traj_id}")
async def get_trajectory(profile: str, traj_id: str) -> dict:
    prof = profiles_mod.load(profile, require_installed=False)
    traj = traj_mod.find(prof, traj_id)
    payload = traj.to_dict()
    payload["meta"] = traj.meta
    payload["summary"] = series_mod.summary(traj)
    return payload


@router.get("/trajectories/{profile}/{traj_id}/series")
async def get_series(profile: str, traj_id: str) -> dict:
    prof = profiles_mod.load(profile, require_installed=False)
    traj = traj_mod.find(prof, traj_id)
    return series_mod.series_for(traj.path)


@router.get("/trajectories/{profile}/{traj_id}/plan")
async def get_plan(profile: str, traj_id: str) -> dict:
    prof = profiles_mod.load(profile, require_installed=False)
    traj = traj_mod.find(prof, traj_id)
    return {"plan": traj_mod.plan(traj)}


@router.get("/media/{profile}/{traj_id}/{filename}")
async def get_media(profile: str, traj_id: str, filename: str, request: Request):
    """Range-served so the browser can scrub. Path traversal is rejected in media_path."""
    prof = profiles_mod.load(profile, require_installed=False)
    traj = traj_mod.find(prof, traj_id)
    return range_response(request, traj_mod.media_path(traj, filename))


@router.post("/trajectories/{profile}/{traj_id}/relabel")
async def relabel(profile: str, traj_id: str, body: RelabelBody) -> dict:
    prof = profiles_mod.load(profile, require_installed=False)
    traj = traj_mod.find(prof, traj_id)
    updated = traj_mod.relabel(prof, traj, body.status)
    return {"trajectory": updated.to_dict(), "counts": traj_mod.counts(prof)}


@router.delete("/trajectories/{profile}/{traj_id}")
async def delete(profile: str, traj_id: str) -> dict:
    prof = profiles_mod.load(profile, require_installed=False)
    traj = traj_mod.find(prof, traj_id)
    traj_mod.delete(prof, traj)
    return {"deleted": traj.id, "counts": traj_mod.counts(prof)}
