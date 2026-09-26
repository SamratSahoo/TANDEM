"""This machine's rig: the robot, the cameras and their calibration, which every profile shares.

The same payload `tandem rig show --json` prints, and changes made the way `tandem rig set` makes them, so
the page and the terminal can never disagree about the rig. Nothing here knows any planner's machine
settings: each planner declares and checks its own (``RIG_OPTIONS``, ``validate_rig_options``).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body

from tandem.core import rig as rig_mod
from tandem.core.errors import TandemError

router = APIRouter(tags=["rig"])


def _payload() -> dict:
    from tandem.cli import rig as rig_cli

    return {
        **rig_cli.show_payload(),
        # What stops collection on this rig, as `tandem rig set` warns of it: the card says it beside the
        # setting that causes it rather than leave it for a session to refuse.
        "problems": [
            {"name": check.name, "detail": check.detail, "hint": check.hint} for check in rig_cli.rig_failures()
        ],
    }


@router.get("/rig")
async def get_rig() -> dict:
    """The rig, its calibration and each installed planner's machine settings; a 400 for a rig.yml that does
    not validate, saying which line (the page shows it, and `tandem rig edit` fixes it)."""
    return _payload()


@router.patch("/rig")
async def change_rig(body: dict[str, Any] = Body(...)) -> dict:
    """Change settings as `tandem rig set` does, several at once: ``{"robot.host": "172.16.0.5",
    "cameras.external_2": null}``, where null removes one.

    Only what the page changed is sent, so a default stays a default rather than being written into the
    file. Every key is checked first; the rig is then validated whole before anything is written, so a bad
    value leaves rig.yml as it was; and it is a round trip, so the comments a person wrote in it survive.
    """
    if not body:
        raise TandemError("Nothing to change.", hint='Send the settings to change, such as {"robot.host": "NUC_ADDRESS"}.')
    from tandem.core import layout

    for key in body:
        rig_mod.check_key(key)
    layout.refuse_rig_change()
    rig_mod.update(body)
    return _payload()
