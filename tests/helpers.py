"""Scaffolding shared by more than one test module.

It lives here rather than in whichever test module happened to define it first. A test
module that imports another has to name it — `from tests.test_session import ...` — and that
only resolves when the repository root is on `sys.path`, which `python -m pytest` arranges
and a bare `pytest` does not. The suite passed locally and failed in CI for exactly that
reason.

pytest puts this directory on `sys.path` for every test module it collects, so `from helpers
import ...` works under either invocation. It also stops collecting one module from importing
another's fixtures and module-level state as a side effect.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

FAKE_DRIVER = Path(__file__).parent / "fake_driver.py"


class FakeRuntime:
    """A runtime that runs the stand-in driver instead of `pixi run tiptop-run`."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.tiptop_dir = root
        root.mkdir(parents=True, exist_ok=True)

    def require_ready(self) -> None:
        return None

    def command(self, args: list[str]) -> list[str]:
        # Drop the console-script name; keep the flags so argument handling is exercised.
        return [sys.executable, str(FAKE_DRIVER), *args[1:]]


def wait_for(predicate, timeout: float = 8.0, interval: float = 0.02) -> bool:
    """Poll until a predicate holds. Returns False on timeout so the caller can assert.

    The session engine is driven by threads reading pipes and a file tailer, so state
    changes land asynchronously; sleeping a fixed amount instead would be both slower and
    flakier.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False
