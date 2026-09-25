"""tandem as a library: decompose a task into phases from a photo, with no robot, no GPU and no planner.

The phase planner is tandem's own code, and the only thing it needs from a planner is the planner's
``Capabilities`` -- what a robot phase may ask for -- which every planner declares without being
built. So the decomposition a collection session would start from can be asked for anywhere Python
runs::

    import tandem

    plan = tandem.plan_task("put the bread in the box", "workspace.png", planner="tiptop")
    for i, phase in enumerate(plan.phases):
        print(i, phase.executor, phase.description, sorted(map(str, phase.atoms)))
        if phase.operator is not None:  # a human phase's magic operator
            print("   ", phase.operator.display, phase.operator.add_effects, phase.operator.delete_effects)
    print(plan.spec.unrepresented)       # clauses of the instruction no phase carries out

This is what `tandem plan` runs, and the two share every step below, so a script and the command
cannot disagree about what a plan is. The model calls are real, so a Gemini key must be set
(`tandem config set-gemini-key`, or ``GEMINI_API_KEY``): one call to name the objects in the photo
(unless they are given), one for the proposal, one more for each repair, and -- with
``classify_initial`` on -- one per invented atom to measure the starting scene.

Nothing here is imported until it is called: ``import tandem`` stays free, and this module imports
only the standard library at its top.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - for editors and type checkers only
    from tandem.planners.base import Capabilities
    from tandem.planning.config import PlanningConfig
    from tandem.planning.plan import PhasePlan


def plan_task(
    instruction: str,
    image: Any,
    *,
    planner: str | Capabilities | None = None,
    profile: str | None = None,
    objects: Sequence[str] | None = None,
    table: str = "table",
    config: PlanningConfig | None = None,
    save_vlm_io: str | os.PathLike | None = None,
) -> PhasePlan:
    """Decompose ``instruction`` into an ordered list of robot and human phases, from a photo.

    ``image`` is the workspace: a path to an image file, a PIL image, or an RGB ``uint8`` array
    (H x W x 3).

    ``planner`` is whose goal language the robot phases are stated in: a registered planner's name
    (`tandem planners list`), or a ``Capabilities`` declaration for one that is not registered. By
    default the planner of ``profile``, else the machine's default planner (``tiptop`` unless
    `tandem planners default NAME` changed it). Nothing of the planner is built or installed:
    only its declaration is read.

    ``profile`` takes the planning settings (the profile's ``hitl:`` block: models, repair attempts,
    ``check_plan_effects``, ``classify_initial``, the proposal cache) and, when ``planner`` is not
    given, the planner from a saved profile. ``config`` gives the settings directly instead and wins
    over ``profile``'s. Neither: tandem's defaults. Phase planning is on whatever ``enabled`` says --
    asking for a plan is asking for one.

    ``objects`` pins the object labels the plan may use, as a planner's perception would have named
    them (a list: ``["bread", "box"]``). None asks a vision model to name what is in the photo first.
    ``table`` is the support surface's label. ``save_vlm_io`` is a directory to write every image
    sent to the model and its reply into, rejected attempts included (``index.jsonl`` and PNGs).

    Returns the ``PhasePlan`` a session would walk (``tandem.planning.plan``): ``.phases``, each a
    ``Phase`` with ``executor``, ``description``, ``atoms``, ``instructions`` and, for a human phase,
    its ``operator``; ``.spec`` with the invented predicates, ``coverage`` and ``unrepresented``;
    ``.to_json()`` for the record ``hitl.json`` is written from.

    Raises ``TandemError`` when no plan can be had: no Gemini key, an unknown planner or profile, a
    photo that cannot be read, or a model whose every attempt (``max_attempts``, 3 by default) was
    refused -- the last refusal is the hint. Blocks until done; inside a running event loop (Jupyter,
    an async server) use ``plan_task_async``.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        from tandem.core.errors import TandemError

        raise TandemError(
            "plan_task was called inside a running event loop, which it cannot block.",
            hint="Use `await tandem.plan_task_async(...)` there instead; it takes the same arguments.",
        )
    return asyncio.run(
        plan_task_async(
            instruction,
            image,
            planner=planner,
            profile=profile,
            objects=objects,
            table=table,
            config=config,
            save_vlm_io=save_vlm_io,
        )
    )


async def plan_task_async(
    instruction: str,
    image: Any,
    *,
    planner: str | Capabilities | None = None,
    profile: str | None = None,
    objects: Sequence[str] | None = None,
    table: str = "table",
    config: PlanningConfig | None = None,
    save_vlm_io: str | os.PathLike | None = None,
) -> PhasePlan:
    """``plan_task``, for a caller already inside an event loop. Same arguments, same result."""
    from tandem.planning import objects as objects_mod
    from tandem.planning.record import recording_to

    if isinstance(objects, str):
        from tandem.core.errors import TandemError

        # Iterated, a string is one label per character -- a plan over objects "b", "r", "e", ...
        raise TandemError(
            f"objects must be a list of labels, not the string {objects!r}.",
            hint='Pass a list, e.g. objects=["bread", "box"].',
        )
    caps = _capabilities(planner, profile)
    cfg = _config(config, profile)
    picture = _picture(image)
    with recording_to(Path(save_vlm_io) if save_vlm_io is not None else None):
        names = objects_mod.sanitize_all(list(objects or []))
        if not names:
            names = await _name_objects(picture, instruction, cfg)
        return await _decompose(picture, instruction, names, table, cfg, caps)


# --------------------------------------------------------------------------- shared with `tandem plan`


def _planner_name(profile: str | None) -> str:
    """The planner a plan is proposed for when none is named: the profile's, or the machine's default."""
    from tandem.core import profiles
    from tandem.core import settings as settings_mod

    if profile:
        return profiles.load(profile).planner.backend
    return settings_mod.load().default_planner


def _capabilities(planner: Any, profile: str | None) -> Capabilities:
    """``planner`` as the declaration the phase planner reads: a name is looked up, never built."""
    from tandem.planners import registry
    from tandem.planners.base import Capabilities

    if isinstance(planner, Capabilities):
        return planner
    if planner is not None and not isinstance(planner, str):
        from tandem.core.errors import TandemError

        raise TandemError(
            f"planner must be a planner's name or a Capabilities, not a {type(planner).__name__}.",
            hint="`tandem planners list` shows the names.",
        )
    return registry.capabilities(planner or _planner_name(profile))


def _profile_config(profile: str | None) -> PlanningConfig:
    """Planning settings from a profile, or the defaults with planning turned on.

    ``enabled`` is forced on: asking for a plan IS asking for one, and refusing because a profile has
    the feature switched off for collection would be obtuse.
    """
    from tandem.planning.config import PlanningConfig

    if not profile:
        return PlanningConfig(enabled=True)

    from tandem.core import profiles

    loaded = profiles.load(profile)
    # Through the profile, so this reads the same cache file a collection session would, rather than
    # one relative to wherever the caller happens to be running.
    cache = profiles.resolve_cache_path(loaded)
    return dataclasses.replace(loaded.hitl.to_planning_config(cache_path=cache), enabled=True)


def _config(config: PlanningConfig | None, profile: str | None) -> PlanningConfig:
    if config is None:
        return _profile_config(profile)
    return dataclasses.replace(config, enabled=True)


def _picture(image: Any):
    """The workspace as the model is sent it: an RGB PIL image, downscaled if it is large."""
    from PIL import Image, UnidentifiedImageError

    from tandem.core.errors import TandemError
    from tandem.planning.grounding import to_pil

    if isinstance(image, (str, os.PathLike)):
        path = Path(image).expanduser()
        if not path.is_file():
            raise TandemError(f"There is no image at {path}.", hint="Pass a photo of the workspace.")
        try:
            image = Image.open(path)
            image.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise TandemError(
                f"{path} could not be read as an image: {exc}",
                hint="Pass a PNG or JPEG photo of the workspace.",
            ) from exc
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    return to_pil(image)


async def _name_objects(picture: Any, instruction: str, cfg: PlanningConfig) -> list[str]:
    """The objects a vision model names in the photo, or a ``TandemError`` saying it could not."""
    from tandem.core.errors import TandemError
    from tandem.planning import objects as objects_mod
    from tandem.planning.symbols import ProposalError

    try:
        return await objects_mod.detect_objects(picture, instruction, cfg)
    except ProposalError as exc:
        # The model's answer never parsed. A ProposalError is a message for the MODEL, and outside
        # the repair loop it would reach a person as a traceback.
        raise TandemError(
            "The model could not name the objects in the photo.",
            hint=f"{exc} Pin them instead (`--object` on the command line, objects= in Python).",
        ) from exc


async def _decompose(
    picture: Any,
    instruction: str,
    names: Sequence[str],
    table: str,
    cfg: PlanningConfig,
    caps: Capabilities,
) -> PhasePlan:
    """The proposal, validated and repaired (``build_plan``), or a ``TandemError`` saying why there is none."""
    from tandem.core.errors import TandemError
    from tandem.planning.plan import build_plan
    from tandem.planning.symbols import ProposalError

    try:
        return await build_plan(picture, instruction, list(names), table, cfg, caps, trajectory_id=None)
    except ProposalError as exc:
        raise TandemError(
            "The model could not produce a usable plan for that instruction.", hint=str(exc)
        ) from exc
