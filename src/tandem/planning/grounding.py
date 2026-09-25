"""Reading invented predicates off an image, and checking the human did what was asked.

An invented predicate has no code behind it -- its definition is the sentence the proposer wrote. So
it is evaluated the only way it can be: show a vision model the workspace and the sentence, and ask.
The same machinery checks a human phase, since "did the box get opened?" is that question asked
about the phase the human was handed.

What a phase is checked against is its contract (``Phase.preconditions``, ``add_effects`` and
``delete_effects``), not only the atoms it should leave true. Preconditions are read off a frame
taken before the hand-off. After it, the add effects must hold and the delete effects must NOT. That
last half is why a :class:`Verdict` keeps what the model saw (``holds``) apart from what the plan
expected (``expected``). For a delete effect a "yes, it holds" answer is the failure, so every
caller branches on ``satisfied``.
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
    """One judgement about one atom in one image, against what the plan expected of it.

    ``holds`` is what the classifier saw; ``expected`` is what the plan said should be there. They
    come apart for a DELETE effect, where the plan expects the atom to be false afterwards and a
    "holds: true" answer is the failure. ``satisfied`` is the one to branch on: reading ``holds`` as
    the verdict silently inverts every delete-effect check.
    """

    atom: Atom
    statement: str
    holds: bool
    reason: str
    expected: bool = True
    # What this atom was being checked AS, for the audit record: "precondition", "effect" or
    # "effect (deleted)". Free text rather than an enum because nothing branches on it.
    role: str = "effect"

    @property
    def satisfied(self) -> bool:
        return self.holds == self.expected

    def summary(self) -> dict:
        return {
            "atom": str(self.atom),
            "statement": self.statement,
            "holds": self.holds,
            "expected": self.expected,
            "satisfied": self.satisfied,
            "role": self.role,
            "reason": self.reason,
        }


async def classify(
    image: Any,
    atom: Atom,
    descriptions: Mapping[str, str],
    cfg: PlanningConfig,
    *,
    expected: bool = True,
    role: str = "effect",
) -> Verdict:
    """Ask whether one atom holds in one image.

    The classifier is always asked the same, positive question -- "is this true?" -- whatever the
    plan expected. Asking it to confirm a negative ("the box is NOT open") reads as a double negative
    and got worse answers, and the prompt it is asked with is the paper's Appendix B word for word.
    ``expected`` is applied to the answer instead.
    """
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
    if holds != expected:
        _log.info(f"vlm: {atom} = {holds}, expected {expected} [{role}] ({reason})")
    else:
        _log.info(f"vlm: {atom} = {holds} [{role}] ({reason})")
    return Verdict(atom, statement, holds, reason, expected=expected, role=role)


async def classify_all(
    image: Any,
    atoms: Iterable[Atom],
    descriptions: Mapping[str, str],
    cfg: PlanningConfig,
    *,
    expected: bool = True,
    role: str = "effect",
) -> list[Verdict]:
    """Classify several atoms against one image, concurrently."""
    atoms = list(atoms)
    if not atoms:
        return []
    return list(
        await asyncio.gather(
            *(classify(image, atom, descriptions, cfg, expected=expected, role=role) for atom in atoms)
        )
    )


async def classify_initial_state(
    image: Any, spec, cfg: PlanningConfig, caps: Capabilities
) -> frozenset[Atom]:
    """Which invented atoms named in the plan already hold before anything is done.

    Only the invented atoms the PLAN mentions are checked, rather than every grounding of every
    invented predicate over every object tuple: the full cross product is the thing that will not
    scale, and nothing reads the others.
    """
    invented_names = {p.name for p in spec.invented}
    # Every invented atom the plan mentions ANYWHERE, operators included. A precondition or a delete
    # effect may be the only place one is named ("the box starts open, and the human closes it"),
    # and those are exactly the ones whose starting value the plan cannot derive.
    mentioned = {
        atom
        for phase in spec.phases
        for atom in (*phase.atoms, *phase.preconditions, *phase.add_effects, *phase.delete_effects)
    }
    candidates = {a for a in mentioned if a.predicate in invented_names}
    if not candidates:
        return frozenset()
    verdicts = await classify_all(
        image, sorted(candidates, key=str), descriptions_for(spec.invented, caps), cfg
    )
    return frozenset(v.atom for v in verdicts if v.holds)


def checkable(atoms: Iterable[Atom], invented: Sequence[VLMPredicate], caps: Capabilities) -> list[Atom]:
    """The atoms a camera can settle, sorted.

    Every invented predicate, because each one is a sentence written to be looked for in an image.
    Of the planner's own predicates, only those the backend declares in ``checkable_predicates``.
    cuTAMP's ``Holding``/``HandEmpty`` are left out on purpose. The frame used here is a
    third-person view: after a hand-off the arm is wherever the operator left it, so a wrist view
    points nowhere useful. In that view the gripper is often out of shot, and the classifier is told
    to answer false when it cannot see the statement to be true. That would fail a phase over
    something the robot knows exactly. Such atoms are still SHOWN to the human
    (``describe_expectations``), just never used to judge them.
    """
    names = {p.name for p in invented} | set(caps.checkable_predicates)
    return [a for a in sorted(set(atoms), key=str) if a.predicate in names]


async def verify_atoms(
    image: Any,
    invented: Sequence[VLMPredicate],
    cfg: PlanningConfig,
    caps: Capabilities,
    *,
    expect_true: Iterable[Atom] = (),
    expect_false: Iterable[Atom] = (),
    role: str = "effect",
) -> tuple[bool, list[Verdict]]:
    """Check one image against atoms that should hold and atoms that should not.

    Returns ``(ok, verdicts)``. ``ok`` means every checkable atom came out the way the plan said it
    would. The atoms that should hold come first in ``verdicts``, then the ones that should not,
    which carry the role ``f"{role} (deleted)"``. Both sets are asked about the one frame at the
    same time, so a phase with delete effects costs the operator no extra wait.
    """
    yes = checkable(expect_true, invented, caps)
    no = checkable(expect_false, invented, caps)
    # A contradiction in the contract, not something a camera can settle: one of the two verdicts
    # would fail whatever the workspace looks like, and the operator would be sent back to redo a
    # step that cannot pass. The proposal parser refuses an operator that both adds and deletes an
    # atom, so reaching here is a bug upstream, and it is reported as one rather than as a failed
    # phase.
    both = sorted(str(a) for a in set(yes) & set(no))
    if both:
        raise ValueError(
            f"The same frame was to be checked for {', '.join(both)} both holding and not holding "
            f"(role {role!r}). An operator may not add and delete the same atom."
        )
    descriptions = descriptions_for(invented, caps)
    held, gone = await asyncio.gather(
        classify_all(image, yes, descriptions, cfg, expected=True, role=role),
        classify_all(image, no, descriptions, cfg, expected=False, role=f"{role} (deleted)"),
    )
    verdicts = [*held, *gone]
    return all(v.satisfied for v in verdicts), verdicts


async def verify_preconditions(
    image: Any,
    phase: Phase,
    invented: Sequence[VLMPredicate],
    cfg: PlanningConfig,
    caps: Capabilities,
) -> tuple[bool, list[Verdict]]:
    """Is the workspace in a state this phase can be carried out from?

    The preconditions of the phase's operator, read off a frame taken BEFORE the hand-off. A phase
    with no operator has no preconditions and trivially passes. There is nothing to check, which is
    exactly what every plan proposed before operators existed says.
    """
    return await verify_atoms(
        image, invented, cfg, caps, expect_true=phase.preconditions, role="precondition"
    )


async def verify_effects(
    image: Any,
    phase: Phase,
    invented: Sequence[VLMPredicate],
    cfg: PlanningConfig,
    caps: Capabilities,
) -> tuple[bool, list[Verdict]]:
    """Did this phase leave the workspace as its operator said it would?

    Add effects must now hold, and delete effects must not. A phase with no operator falls back to
    its own atoms as the add effects and nothing deleted, which is the check every phase got before
    operators existed. Only atoms a camera can settle are put to the model (see ``checkable``).
    """
    return await verify_atoms(
        image,
        invented,
        cfg,
        caps,
        expect_true=phase.add_effects,
        expect_false=phase.delete_effects,
        role="effect",
    )


def missing_statements(verdicts: Sequence[Verdict]) -> list[str]:
    """What is wrong with the workspace, phrased for the operator.

    Reads ``satisfied``, not ``holds``. A delete effect that is still true is as much a reason the
    phase did not happen as an add effect that never became true. It has to be said the other way
    round, though, or the operator is told to do what they already did.
    """
    out = []
    for v in verdicts:
        if v.satisfied:
            continue
        out.append(v.statement if v.expected else f"{v.statement} -- and it should no longer be")
    return out


def describe_expectations(phase: Phase, descriptions: Mapping[str, str]) -> list[str]:
    """What the workspace should look like after the human acts, for the hand-off instructions.

    The add effects, then the delete effects phrased the other way round. A step whose whole point
    is that something stops being the case (the lid is no longer on the jar) reads as a missing
    instruction if only the add effects are shown. Everything is listed, including what the camera
    will not be asked about, so the human knows all that is expected of them.
    """
    out = [describe(a, descriptions) for a in sorted(phase.add_effects, key=str)]
    out += [f"NO LONGER: {describe(a, descriptions)}" for a in sorted(phase.delete_effects, key=str)]
    return out
