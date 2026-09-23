"""Where the tests that read a planner's own source find it. One place, so they cannot disagree.

tandem no longer ships tiptop, cuTAMP or cuRobo: an install fetches them. So the static checks of the
sidecar against the planner it calls (tests/test_planners.py, tests/test_sidecar_legs.py) need the
sources from somewhere, in this order:

1. ``$TANDEM_PLANNER_SOURCES``: a directory holding one checkout or export per source name, the same
   directory an offline install reads. CI fetches the pinned commits into one with
   ``python tools/bundle.py --planner tiptop --out DIR`` and sets this. Set, it is REQUIRED to hold
   what a test asks for: a check that quietly skipped in the one job that exists to run it would
   pass for exactly the wrong reason.
2. An installed TiPToP runtime: on a workstation that has one, the trees the sidecar really runs
   against.
3. Otherwise the test is skipped, and says how to stop skipping it.

The runtime is located when this module is imported -- at collection, before ``conftest`` points
every tandem root at a temporary directory -- so a developer's own runtime is found.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ENV = "TANDEM_PLANNER_SOURCES"


def _installed_runtime() -> Path | None:
    try:
        from tandem.core import settings as settings_mod

        root = settings_mod.load(force=True).resolved_runtime_dir()
    except Exception:  # pragma: no cover - an unreadable config is a reason to skip, not to fail
        return None
    finally:
        try:
            from tandem.core import settings as settings_mod

            settings_mod._cache = None
        except Exception:  # pragma: no cover
            pass
    return root if root.is_dir() else None


_RUNTIME = _installed_runtime()


def planner_sources(*required: str) -> Path:
    """A directory holding the source trees ``required`` names (paths relative to it), or skip.

    ``planner_sources("tiptop/tiptop/tiptop_run.py", "cuTAMP/cutamp")`` returns the root they are in.
    """
    override = os.environ.get(ENV, "").strip()
    if override:
        root = Path(override).expanduser()
        missing = [rel for rel in required if not (root / rel).exists()]
        if missing:
            pytest.fail(
                f"${ENV} is {root}, which does not have {', '.join(missing)}. It should hold one checkout "
                "or export per source: python tools/bundle.py --planner tiptop --out DIR",
                pytrace=False,
            )
        return root

    if _RUNTIME is not None and all((_RUNTIME / rel).exists() for rel in required):
        return _RUNTIME

    pytest.skip(
        f"no planner sources to check against: set ${ENV} to a directory of the pinned sources "
        "(python tools/bundle.py --planner tiptop --out DIR), or build the runtime (tandem runtime build)"
    )
