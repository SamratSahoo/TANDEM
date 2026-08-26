"""Naming the objects in a workspace photo, for planning without a robot.

During a session the object labels come from the planner's own perception pass, and they are the
only ones that matter: a goal must be stated in the names the planner will recognise. This is for
the other case -- ``tandem plan``, checking a decomposition from a photo before anyone goes near the
arm -- where there is no planner to ask.

The labels it produces will not match a real perception pass exactly, and that is fine for what this
is for: a decomposition that is wrong is wrong whatever the objects end up being called, and the
whole point of checking one beforehand is that the remedy (rewrite the instruction, put the missing
object on the table) is only available before the arm moves.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tandem.planning.config import PlanningConfig
from tandem.planning.symbols import ProposalError

_SCHEMA = {
    "type": "object",
    "properties": {"objects": {"type": "array", "items": {"type": "string"}}},
    "required": ["objects"],
}


def _prompt(instruction: str) -> str:
    return f"""\
List the objects on the table in this image that a robot could pick up or place things on.

The robot has been asked to: {instruction}

Name every object the instruction refers to, plus anything else on the table. Use short
lowercase_with_underscores names a person would recognise -- `red_toy`, `cardboard_box`,
`folded_cloth`. Do not list the table itself, the robot, or anything off the table."""


async def detect_objects(image: Any, instruction: str, cfg: PlanningConfig) -> list[str]:
    """The objects a model can see in this image, as planning-ready labels."""
    from tandem.planning.llm import query_json

    def parse(data: Any) -> list[str]:
        names = (data or {}).get("objects")
        if not isinstance(names, list) or not names:
            raise ProposalError("Respond with an 'objects' list naming at least one object.")
        return sorted({sanitize(str(n)) for n in names if str(n).strip()})

    return await query_json(
        _prompt(instruction),
        parse,
        model=cfg.vlm_model,
        schema=_SCHEMA,
        image=image,
        max_attempts=cfg.max_attempts,
        label="detect objects",
    )


def sanitize(label: str) -> str:
    """A detected label as a symbol: the same squeeze a perception pass applies before use."""
    return label.strip().lower().replace(" ", "_")


def sanitize_all(labels: Sequence[str]) -> list[str]:
    return sorted({sanitize(x) for x in labels if x.strip()})
