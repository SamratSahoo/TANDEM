"""Where the tests that read a planner's own source find it. One place, so they cannot disagree.

tandem no longer ships tiptop, cuTAMP or cuRobo: an install fetches them. So the static checks of the
sidecar against the planner it calls (tests/test_planners.py, tests/test_sidecar_legs.py) need the
sources from somewhere, in this order:

1. ``$TANDEM_PLANNER_SOURCES``: a directory holding one checkout or export per source name, the same
   directory an offline install reads. CI fetches the pinned commits into one with
   ``python tools/bundle.py --planner tiptop --out DIR`` and sets this. Set, it is REQUIRED to hold
   what a test asks for: a check that quietly skipped in the one job that exists to run it would
   pass for exactly the wrong reason. The same goes for any other reason a check would skip while it
   is set (``skip_or_fail``): the job fails instead.
2. An installed TiPToP runtime, when the trees a test reads are at the commits the recipe pins now: on
   a workstation that has one, the trees the sidecar really runs against. A runtime built before a pin
   moved is outdated, and checking the sidecar against it would fail for a reason that has nothing to
   do with the change under test -- so the test is skipped, and says to reinstall.
3. Otherwise the test is skipped, and says how to stop skipping it.

The runtime is located when this module is imported -- at collection, before ``conftest`` points
every tandem root at a temporary directory -- so a developer's own runtime is found.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ENV = "TANDEM_PLANNER_SOURCES"


def _installed_runtime() -> tuple[Path | None, dict[str, str]]:
    """(the installed TiPToP runtime or None, and each source in it that is NOT at the recipe's pin -> why).

    Read from the runtime's own record (.tandem-runtime.json): stat calls and one small JSON file, pure
    Python, safe at collection.
    """
    try:
        from tandem.core import settings as settings_mod

        root = settings_mod.load(force=True).resolved_runtime_dir()
    except Exception:  # pragma: no cover - an unreadable config is a reason to skip, not to fail
        return None, {}
    finally:
        try:
            from tandem.core import settings as settings_mod

            settings_mod._cache = None
            settings_mod._stamp = None
        except Exception:  # pragma: no cover
            pass
    if not root.is_dir():
        return None, {}
    try:
        from tandem.planners.tiptop.recipe import RECIPE
        from tandem.planners.tiptop.runtime import TiptopRuntime

        status = TiptopRuntime(root).status()
    except Exception as exc:  # pragma: no cover - a runtime that cannot be read is not one to test against
        return None, {"*": f"its record could not be read ({type(exc).__name__}: {exc})"}
    have = {pin.name: pin.commit for pin in status.pins}
    stale = {
        pin.name: f"{pin.name} is at {str(have.get(pin.name) or 'nothing recorded')[:7]}, the recipe pins {pin.short()}"
        for pin in RECIPE.pins
        if pin.name in status.mismatched(RECIPE.pins)
    }
    return root, stale


_RUNTIME, _STALE = _installed_runtime()


def skip_or_fail(reason: str):
    """Skip with ``reason`` -- unless $TANDEM_PLANNER_SOURCES is set, when the check must run and fails.

    For the checks that have a reason to skip besides a missing tree (sources that are a fork, a symbolic
    layer that will not import). In the job that sets the variable, any of those means the job checked
    nothing while reporting green, which is the one outcome that job exists to rule out.
    """
    if os.environ.get(ENV, "").strip():
        pytest.fail(f"${ENV} is set, so this check has to run, but: {reason}", pytrace=False)
    pytest.skip(reason)


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
        used = {Path(rel).parts[0] for rel in required}
        stale = [why for name, why in _STALE.items() if name in used or name == "*"]
        if stale:
            pytest.skip(
                f"the installed runtime at {_RUNTIME} is outdated ({'; '.join(stale)}): "
                "`tandem planners install tiptop` updates it, or set $TANDEM_PLANNER_SOURCES"
            )
        return _RUNTIME

    pytest.skip(
        f"no planner sources to check against: set ${ENV} to a directory of the pinned sources "
        "(python tools/bundle.py --planner tiptop --out DIR), or build the runtime (tandem runtime build)"
    )
