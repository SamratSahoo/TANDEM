"""The ZED cameras plugged into this machine, as the ZED SDK lists them.

tandem's own environment has no ZED bindings: ``pyzed`` comes with the ZED SDK and is installed into a
runtime (TiPToP's, teleop's) by its optional ZED step. So the listing runs in a subprocess, under the
first interpreter here that can import ``pyzed``, and reads back one line of JSON. It opens no camera:
``sl.Camera.get_device_list()`` only enumerates them.

``tandem init`` uses it to offer each camera role a serial to confirm, instead of asking for serials
nobody remembers.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Run under an interpreter with pyzed: every connected ZED camera, one JSON line.
_LIST = """
import json
import pyzed.sl as sl
print(json.dumps([
    {"serial": str(d.serial_number), "model": str(d.camera_model), "state": str(d.camera_state)}
    for d in sl.Camera.get_device_list()
]))
"""

# The SDK enumerates over USB (and GMSL) in well under a second; this only guards a wedged driver.
_TIMEOUT = 30.0


@dataclass(frozen=True)
class ZedCamera:
    serial: str
    model: str
    state: str = ""

    @property
    def mini(self) -> bool:
        """A ZED Mini or ZED X Mini: the model a wrist mount takes (DROID's wrist camera is one)."""
        name = "".join(ch for ch in self.model.upper() if ch.isalnum())
        return name in {"ZEDM", "ZEDMINI", "ZEDXM", "ZEDXMINI"}

    @property
    def available(self) -> bool:
        """False when another process has it open. An unknown state counts as available."""
        return "NOT" not in self.state.upper()


def interpreters(planner: str | None = None, settings: Any = None) -> list[Path]:
    """Interpreters that may have pyzed, most likely first: the planner's runtime, teleop's, then this one
    and ``python3`` (a ZED SDK whose Python API was installed system-wide)."""
    found: list[Path] = []

    def add(path: Path | None) -> None:
        if path is not None and path.is_file() and path not in found:
            found.append(path)

    if planner:
        try:
            from tandem.planners import registry

            rt = registry.runtime(planner, settings)
            if rt is not None and hasattr(rt, "python"):
                add(Path(rt.python()))
        except Exception:
            pass  # not built yet, or a planner with no interpreter of its own
    try:
        from tandem.teleop import recipe

        add(recipe.runtime(settings).python())
    except Exception:
        pass
    add(Path(sys.executable))
    python3 = shutil.which("python3")
    add(Path(python3) if python3 else None)
    return found


def detect(pythons: Iterable[Path]) -> list[ZedCamera] | None:
    """The connected ZED cameras, from the first interpreter that can list them. None: none of them has pyzed
    (or the SDK could not enumerate), so nothing is known -- not the same as an empty list."""
    for python in pythons:
        try:
            done = subprocess.run(
                [str(python), "-c", _LIST], capture_output=True, text=True, timeout=_TIMEOUT, check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if done.returncode != 0:
            continue
        lines = [line for line in done.stdout.splitlines() if line.strip().startswith("[")]
        if not lines:
            continue
        try:
            rows = json.loads(lines[-1])
        except json.JSONDecodeError:
            continue
        return [ZedCamera(str(r.get("serial", "")), str(r.get("model", "")), str(r.get("state", ""))) for r in rows]
    return None


def suggest(found: list[ZedCamera], roles: Iterable[str]) -> dict[str, str]:
    """A serial for each role, to be confirmed: a Mini for the wrist (``hand``), the rest for the others in
    the order the SDK lists them. Roles left over get none."""
    minis = [cam.serial for cam in found if cam.mini]
    others = [cam.serial for cam in found if not cam.mini]
    suggestion: dict[str, str] = {}
    for role in roles:
        pool = (minis or others) if role == "hand" else (others or minis)
        if pool:
            suggestion[role] = pool.pop(0)
    return suggestion
