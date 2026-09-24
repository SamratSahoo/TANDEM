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
    from tandem.cli.profile import missing_here

    try:
        # Read-only: a profile collected with a plugin this machine does not have is still a card, with
        # what is missing said on it -- a laptop is where such a profile is browsed.
        profile = profiles_mod.load(name, require_installed=False)
    except ProfileError as exc:
        return {"name": name, "active": name == active, "valid": False, "error": exc.message}
    counts = trajectories.counts(profile)
    return {
        "name": name,
        "active": name == active,
        "valid": True,
        "missing": missing_here(profile),
        "description": profile.description,
        "prompt": profile.task.prompt,
        "goal": profile.task.goal,
        "planner": profile.planner.backend,
        "planner_summary": _view(profile).summary,
        "target": profile.task.target_episodes,
        "counts": counts,
        "file": str(profile.file()),
        "trajectories": str(profile.trajectories_dir()),
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
    from tandem.core import layout

    cfg = settings_mod.load(force=True)
    names = profiles_mod.list_names()
    return {
        "active": cfg.active_profile,
        "data_root": str(cfg.resolved_data_root()),
        "profiles": [_card(name, cfg.active_profile) for name in names],
        # Profiles written before version 3, not loaded until `tandem init` (or `tandem profile migrate`)
        # moves them: the page says so rather than show fewer profiles than there are.
        "old_layout": layout.pending(),
    }


@router.get("/profiles/{name}")
async def get_profile(name: str) -> dict:
    cfg = settings_mod.load()
    profile = profiles_mod.load(name, require_installed=False)
    view = _view(profile, cfg)
    return {
        **_card(name, cfg.active_profile),
        "profile": profile.model_dump(mode="json"),
        "planner_view": view.to_dict(),
        "warnings": list(view.warnings),
    }


@router.put("/profiles/{name}")
async def update_profile(name: str, body: dict[str, Any] = Body(...)) -> dict:
    """Replace a profile wholesale. Validation happens before anything is written, so a bad
    edit from the browser cannot leave a profile that fails at session start.

    The profile on disk is read as written, not validated: the editor is how a profile that no longer
    loads gets fixed, and refusing the corrected body because the old one was broken left no way to.
    A planner or executor the file already named may be absent from this machine and stay; a name the
    edit introduces must be installed.
    """
    path = profiles_mod.path_of(name)
    if not profiles_mod.exists(name):
        raise ProfileError(f"Profile {name!r} does not exist.")
    try:
        previous_prompt = (profiles_mod.read_data(path, name=name).get("task") or {}).get("prompt")
    except ProfileError:
        previous_prompt = None
    payload = dict(body)
    payload["name"] = name  # the file's name is the identity
    try:
        updated = profiles_mod.Profile.model_validate(
            payload, context={profiles_mod.ABSENT_OK: profiles_mod.names_in_file(path)}
        )
    except Exception as exc:
        # A raw pydantic error would surface as a 500 and a wall of text. The editor needs a
        # 400 with the message the user can act on — usually a mistyped planner option.
        raise TandemError(
            "That profile is not valid:\n" + (profiles_mod.format_errors(exc).strip() or str(exc)),
            hint="Nothing was written; the profile on disk is unchanged.",
        ) from exc
    profiles_mod.save(updated)
    cfg = settings_mod.load()
    # As saved: the planner's options as it normalised them.
    updated = profiles_mod.load(name, require_installed=False)
    view = _view(updated, cfg)
    return {
        **_card(name, cfg.active_profile),
        "profile": updated.model_dump(mode="json"),
        "planner_view": view.to_dict(),
        "warnings": list(view.warnings),
        "previous_prompt": previous_prompt,
    }


@router.post("/profiles/active")
async def set_active(body: ActivateBody) -> dict:
    if not profiles_mod.exists(body.name):
        raise ProfileError(f"Profile {body.name!r} does not exist.")
    cfg = settings_mod.load()
    cfg.active_profile = body.name
    settings_mod.save(cfg)
    return {"active": body.name}


#: What POST /profiles reads. Anything else is refused rather than dropped: a field the page sends and
#: the server ignores is a setting the person chose and never got.
CREATE_KEYS = ("name", "from", "prompt")


@router.post("/profiles")
async def create_profile(body: dict[str, Any] = Body(...)) -> dict:
    from tandem import resources

    unknown = sorted(set(body) - set(CREATE_KEYS))
    if unknown:
        raise TandemError(
            f"A new profile does not take {', '.join(unknown)}.",
            hint=f"It takes {', '.join(CREATE_KEYS)}; edit the rest once it exists.",
        )
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
    else:
        from tandem.cli import planners as planners_cli

        profile = profiles_mod.load_file(resources.path("profile_template.yml"), name=name)
        profile.description = ""
        # The machine's default planner, as `tandem profile create` gives it -- keeping the template's
        # options when the template already names that planner.
        chosen = planners_cli.planner_for_new_profile()
        if profile.planner.backend != chosen:
            profile.planner = profiles_mod.planner_spec(chosen, profile=name)

    if body.get("prompt"):
        profile.task.prompt = str(body["prompt"])
    elif not source:
        # The template is the paper's settings with no task of its own: as `tandem profile create` does, a
        # new profile is never written with its placeholder as the task. (A 400, not a ProfileError's 404.)
        raise TandemError(
            "A new profile needs its task.",
            hint="Say what the robot and you are to do, or start from a copy of a profile (the paper's five "
            "included).",
        )

    profiles_mod.save(profile)

    cfg = settings_mod.load()
    return _card(name, cfg.active_profile)


@router.delete("/profiles/{name}")
async def delete_profile(name: str, purge: bool = False) -> dict:
    # profiles.delete checks the name before it becomes a path: `%2E%2E` reaches here as "..".
    cfg = settings_mod.load()
    profiles_mod.delete(name, keep_data=not purge)
    if cfg.active_profile == name:
        remaining = profiles_mod.list_names()
        cfg.active_profile = remaining[0] if remaining else "default"
        settings_mod.save(cfg)
    return {"deleted": name, "purged": purge, "active": cfg.active_profile}
