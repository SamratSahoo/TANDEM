"""Reading invented predicates off an image, and checking the human did what was asked.

An invented predicate has no code behind it -- its definition is the sentence the proposer wrote. So
it is evaluated the only way it can be: show a vision model the workspace and the sentence, and ask.
The same machinery verifies a human phase, since "did the box get opened?" is that question asked
about the atoms of the phase the human was handed.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tandem.planners.base import Capabilities
from tandem.planning.config import PlanningConfig
from tandem.planning.prompts import CLASSIFIER_SCHEMA, classifier_prompt
from tandem.planning.structs import Phase, VLMPredicate
from tandem.planning.symbols import Atom, ProposalError, describe

_log = logging.getLogger(__name__)

# The largest image sent to the classifier. Full camera frames are far bigger than the model needs
# for "is this open", and shrinking them is the difference between a snappy check and one the
# operator waits on with the arm parked.
_MAX_IMAGE_EDGE = 1024


def to_pil(rgb: Any):
    """A camera frame as a PIL image the model can take, downscaled if it is large."""
    import numpy as np
    from PIL import Image

    image = rgb if isinstance(rgb, Image.Image) else Image.fromarray(np.asarray(rgb).astype(np.uint8))
    if max(image.size) > _MAX_IMAGE_EDGE:
        scale = _MAX_IMAGE_EDGE / max(image.size)
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
    return image


def descriptions_for(invented: Sequence[VLMPredicate], caps: Capabilities) -> dict[str, str]:
    """Predicate name -> natural-language template, for everything that can be described.

    The planner's own predicates bring their phrasings from ``Capabilities``; an invented one brings
    its own, which is also its definition.
    """
    return {**dict(caps.predicate_descriptions), **{p.name: p.instructions for p in invented}}


@dataclass(frozen=True)
class Verdict:
    """One judgement about one atom in one image."""

    atom: Atom
    statement: str
    holds: bool
    reason: str

    def summary(self) -> dict:
        return {
            "atom": str(self.atom),
            "statement": self.statement,
            "holds": self.holds,
            "reason": self.reason,
        }


async def classify(image: Any, atom: Atom, descriptions: Mapping[str, str], cfg: PlanningConfig) -> Verdict:
    """Ask whether one atom holds in one image."""
    from tandem.planning.llm import query_json

    statement = describe(atom, descriptions)

    def parse(data):
        if not isinstance(data, dict) or "holds" not in data:
            raise ProposalError("Respond with an object containing 'holds' (a boolean) and 'reason'.")
        return bool(data["holds"]), str(data.get("reason", ""))

    holds, reason = await query_json(
        classifier_prompt(statement),
        parse,
        model=cfg.vlm_model,
        schema=CLASSIFIER_SCHEMA,
        image=image,
        max_attempts=cfg.max_attempts,
        label=f"classify {atom}",
    )
    _log.info(f"vlm: {atom} = {holds} ({reason})")
    return Verdict(atom, statement, holds, reason)


async def classify_all(
    image: Any, atoms: Iterable[Atom], descriptions: Mapping[str, str], cfg: PlanningConfig
) -> list[Verdict]:
    """Classify several atoms against one image, concurrently."""
    atoms = list(atoms)
    if not atoms:
        return []
    return list(await asyncio.gather(*(classify(image, atom, descriptions, cfg) for atom in atoms)))


async def classify_initial_state(
    image: Any, spec, cfg: PlanningConfig, caps: Capabilities
) -> frozenset[Atom]:
    """Which invented atoms named in the plan already hold before anything is done.

    Only the invented atoms the PLAN mentions are checked, rather than every grounding of every
    invented predicate over every object tuple: the full cross product is the thing that will not
    scale, and nothing reads the others.
    """
    invented_names = {p.name for p in spec.invented}
    candidates = {a for phase in spec.phases for a in phase.atoms if a.predicate in invented_names}
    if not candidates:
        return frozenset()
    verdicts = await classify_all(
        image, sorted(candidates, key=str), descriptions_for(spec.invented, caps), cfg
    )
    return frozenset(v.atom for v in verdicts if v.holds)


async def verify_phase(
    image: Any,
    phase: Phase,
    invented: Sequence[VLMPredicate],
    cfg: PlanningConfig,
    caps: Capabilities,
) -> tuple[bool, list[Verdict]]:
    """Did the human's phase actually leave the workspace as it should have?

    Only atoms a camera can settle are put to the model: the invented predicates, plus whichever of
    the planner's own the backend declares ``checkable``. cuTAMP's ``Holding``/``HandEmpty`` are
    excluded -- the frame used here is a third-person view (after a hand-off the arm is wherever the
    operator left it, so a wrist view points nowhere useful), the gripper is often out of shot, and
    the classifier is told to answer false when it cannot see the statement to be true. That would
    fail a phase over something the robot knows exactly. They are still SHOWN to the human, just not
    used to judge them.
    """
    checkable = {p.name for p in invented} | set(caps.checkable_predicates)
    atoms = [a for a in sorted(phase.atoms, key=str) if a.predicate in checkable]
    verdicts = await classify_all(image, atoms, descriptions_for(invented, caps), cfg)
    return all(v.holds for v in verdicts), verdicts


def missing_statements(verdicts: Sequence[Verdict]) -> list[str]:
    """The statements that should hold and do not, phrased for the operator."""
    return [v.statement for v in verdicts if not v.holds]


def describe_expectations(phase: Phase, descriptions: Mapping[str, str]) -> list[str]:
    """What should be true after the human acts, for the hand-off instructions."""
    return [describe(a, descriptions) for a in sorted(phase.atoms, key=str)]
