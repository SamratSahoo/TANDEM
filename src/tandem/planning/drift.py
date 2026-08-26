"""Keeping a plan pointed at the right objects when perception renames them.

Detection names objects afresh on every pass and the names drift: one pass calls them ``toy`` and
``box``, the next ``blue_toy`` and ``cardboard_box``. Mid-task that is fatal -- the plan refers to
objects this pass did not produce, so it would be thrown away and the whole task re-planned from a
scene that has already been half-rearranged, asking the human to redo their part. Observed doing
exactly that.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

_log = logging.getLogger(__name__)


def match_drifted_names(missing: Sequence[str], detected: Sequence[str]) -> dict[str, str] | None:
    """Map names a plan uses onto this pass's labels, or None if it cannot be done unambiguously.

    The rule is deliberately conservative and needs no extra model call: a name matches when it is a
    whole-word subset of exactly one detected label (or the other way round). Anything ambiguous
    returns None, and the caller re-plans as before rather than guessing which object was meant.
    """
    available = list(detected)
    mapping: dict[str, str] = {}
    for name in missing:
        wanted = set(name.split("_"))
        candidates = [d for d in available if wanted <= set(d.split("_")) or set(d.split("_")) <= wanted]
        if len(candidates) != 1:
            _log.info(
                f"cannot re-bind '{name}' to this pass's labels "
                f"({', '.join(sorted(detected))}): {len(candidates)} candidate(s)"
            )
            return None
        mapping[name] = candidates[0]
        available.remove(candidates[0])
    return mapping
