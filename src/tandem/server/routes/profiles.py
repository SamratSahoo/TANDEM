"""Profile routes.

The page shows every planner's settings the same way -- a summary, titled sections, what the planner
will receive and what is wrong with it -- because each planner describes its own options
(``registry.describe_options``). Nothing here knows any planner's schema.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body
from pydantic import BaseModel

from tandem.core import profiles as profiles_mod
from tandem.core import settings as settings_mod
from tandem.core import trajectories
from tandem.core.errors import ProfileError, TandemError
from tandem.planners import registry

router = APIRouter(tags=["profiles"])


class ActivateBody(BaseModel):
    name: str


def _card(name: str, active: str) -> dict:
    try:
        profile = profiles_mod.load(name)
    except ProfileError as exc:
        return {"name": name, "active": name == active, "valid": False, "error": exc.message}
    counts = trajectories.counts(profile)
    return {
        "name": name,
        "active": name == active,
        "valid": True,
        "description": profile.description,
        "prompt": profile.task.prompt,
        "goal": profile.task.goal,
        "planner": profile.planner.backend,
        "planner_summary": _view(profile).summary,
        "target": profile.task.target_episodes,
        "counts": counts,
        "cameras": list(profile.cameras.configured()),
        "dir": str(profile.dir()),
    }


def _view(profile, cfg=None):
    """The profile's planner options as its planner describes them; a bare view for one that will not load.

    A card must render for a profile whose planner is broken: the card is where that is noticed.
    """
    from tandem.planners.base import OptionsView

    try:
        return registry.describe_options(profile.planner.backend, profile, settings=cfg)
    except TandemError as exc:
        return OptionsView(warnings=(exc.message,))


@router.get("/profiles")
async def list_profiles() -> dict:
    cfg = settings_mod.load(force=True)
    names = profiles_mod.list_names()
    return {
        "active": cfg.active_profile,
        "data_root": str(cfg.resolved_data_root()),
        "profiles": [_card(name, cfg.active_profile) for name in names],
    }


@router.get("/profiles/{name}")
async def get_profile(name: str) -> dict:
    cfg = settings_mod.load()
    profile = profiles_mod.load(name)
    view = _view(profile, cfg)
    return {
        **_card(name, cfg.active_profile),
        "profile": profile.model_dump(mode="json"),
        "planner_view": view.to_dict(),
        "warnings": list(view.warnings),
        "calibration": profiles_mod.calibration(profile),
        "missing_calibration": profiles_mod.missing_calibration(profile),
    }


@router.put("/profiles/{name}")
async def update_profile(name: str, body: dict[str, Any] = Body(...)) -> dict:
    """Replace a profile wholesale. Validation happens before anything is written, so a bad
    edit from the browser cannot leave a profile that fails at session start."""
    existing = profiles_mod.load(name)
    payload = dict(body)
    payload["name"] = name  # the directory is the identity
    try:
        updated = profiles_mod.Profile.model_validate(payload)
    except Exception as exc:
        # A raw pydantic error would surface as a 500 and a wall of text. The editor needs a
        # 400 with the message the user can act on — usually a mistyped planner option.
        raise TandemError(
            "That profile is not valid:\n" + (profiles_mod.format_errors(exc).strip() or str(exc)),
            hint="Nothing was written; the profile on disk is unchanged.",
        ) from exc
    profiles_mod.save(updated)
    cfg = settings_mod.load()
    updated = profiles_mod.load(name)  # as saved: the planner's options as it normalised them
    view = _view(updated, cfg)
    return {
        **_card(name, cfg.active_profile),
        "profile": updated.model_dump(mode="json"),
        "planner_view": view.to_dict(),
        "warnings": list(view.warnings),
        "previous_prompt": existing.task.prompt,
    }


@router.post("/profiles/active")
async def set_active(body: ActivateBody) -> dict:
    if not profiles_mod.exists(body.name):
        raise ProfileError(f"Profile {body.name!r} does not exist.")
    cfg = settings_mod.load()
    cfg.active_profile = body.name
    settings_mod.save(cfg)
    return {"active": body.name}


@router.post("/profiles")
async def create_profile(body: dict[str, Any] = Body(...)) -> dict:
    from tandem import resources

    name = str(body.get("name") or "").strip()
    if not name:
        raise ProfileError("A profile needs a name.")
    if profiles_mod.exists(name):
        raise ProfileError(f"Profile {name!r} already exists.")

    source = body.get("from")
    if source:
        base = profiles_mod.load(str(source))
        profile = base.model_copy(deep=True)
        profile.name = name
        profile.description = f"copied from {source}"
        calibration = profiles_mod.calibration(base)
    else:
        from tandem.cli import planners as planners_cli

        profile = profiles_mod.load_file(resources.path("profile_template.yml"), name=name)
        profile.description = ""
        # The machine's default planner, as `tandem profile create` gives it -- keeping the template's
        # options when the template already names that planner.
        chosen = planners_cli.planner_for_new_profile()
        if profile.planner.backend != chosen:
            profile.planner = profiles_mod.PlannerSpec(backend=chosen)
        calibration = {}

    if body.get("prompt"):
        profile.task.prompt = str(body["prompt"])

    profiles_mod.save(profile)
    if calibration:
        import json

        profile.calibration_file().write_text(json.dumps(calibration, indent=2) + "\n")

    cfg = settings_mod.load()
    return _card(name, cfg.active_profile)


@router.delete("/profiles/{name}")
async def delete_profile(name: str, purge: bool = False) -> dict:
    cfg = settings_mod.load()
    profiles_mod.delete(name, keep_data=not purge)
    if cfg.active_profile == name:
        remaining = profiles_mod.list_names()
        cfg.active_profile = remaining[0] if remaining else "default"
        settings_mod.save(cfg)
    return {"deleted": name, "purged": purge, "active": cfg.active_profile}
