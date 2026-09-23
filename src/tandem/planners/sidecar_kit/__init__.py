"""Where ``tandem_sidecar`` lives: the module every planner's sidecar imports to speak tandem's protocol.

A sidecar runs in the planner's own environment, where tandem is not installed. So the helper it
imports is not a tandem module at all: ``tandem_sidecar.py`` is a standalone, standard-library-only
file that ships inside this directory, and ``tandem.planners.sidecar.SidecarPlanner`` puts the
directory on the sidecar's ``PYTHONPATH`` when it launches one. A sidecar script then begins::

    from tandem_sidecar import log, serve

This package itself only says where that is. It must never import ``tandem_sidecar``: importing it
takes over the process's stdout, which is right in a sidecar and wrong everywhere else. Nothing but
this file and ``tandem_sidecar.py`` belongs in the directory, since all of it is on every sidecar's
import path.
"""

from __future__ import annotations

from pathlib import Path

#: The directory to put on a sidecar's PYTHONPATH.
DIRECTORY = Path(__file__).resolve().parent
#: The module a sidecar imports.
MODULE = "tandem_sidecar"


def path() -> Path:
    """The helper module's file, for reading or copying it (a bundle, a container image)."""
    return DIRECTORY / f"{MODULE}.py"
