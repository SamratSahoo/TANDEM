"""The one rule for a name typed to choose a component: a planner or a human executor.

``planner.backend`` and ``hitl.human_executor`` are both typed into a profile's YAML, onto a command
line (``tandem planners use``, ``tandem executors use``) and into a package's entry-point table, and
a planner's name also becomes a directory (``<runtimes>/<name>``). The two registries used to
disagree -- planners lowercase only, executors any case -- which let ``ACT`` name an executor on a
case-sensitive filesystem and ``act`` a different one, while the same spelling was refused as a
planner. One rule now, lowercase, for both.

Anchored with ``\\A`` and ``\\Z`` rather than ``^`` and ``$``, because ``$`` also matches before a
trailing newline: ``"toy\\n"`` would pass, and resolve to nothing at the first lookup.

Standard library only: both registries import it, and so does ``tandem.planning.config``.
"""

from __future__ import annotations

import re

#: The shape, for a caller composing a larger pattern.
PATTERN = r"[a-z][a-z0-9_-]*"

#: Matches a whole name and nothing else, with ``match`` or ``fullmatch`` alike.
NAME = re.compile(rf"\A{PATTERN}\Z")

#: The rule in words, for an error message.
RULE = "lowercase letters, digits, _ and -, starting with a letter"


def is_valid(name: object) -> bool:
    """Whether ``name`` could name a planner or a human executor. Says nothing about either existing."""
    return isinstance(name, str) and NAME.match(name) is not None
