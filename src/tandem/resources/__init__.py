"""Package data: the profile template and config templates.

Kept a package (not a bare data dir) so importlib.resources can find it in a wheel, a zip,
or an editable install without path guessing.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path


def path(name: str) -> Path:
    with resources.as_file(resources.files(__package__) / name) as p:
        return Path(p)


def read(name: str) -> str:
    return (resources.files(__package__) / name).read_text()
