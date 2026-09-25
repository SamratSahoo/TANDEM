"""Every link between the docs lands somewhere: the file exists, and so does the heading it names.

The docs link each other by heading (``CONFIGURATION.md#the-rig``), and a heading renamed or removed -- as
profiles, presets and the rig were rewritten -- leaves links that open the right page at the top, which
nobody notices. Anchors are GitHub's: the heading lowercased, punctuation dropped, spaces as hyphens.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCS = [REPO / "README.md", *sorted((REPO / "docs").glob("*.md"))]
LINK = re.compile(r"\]\(([^)\s]*)\)")
FENCE = re.compile(r"^```.*?^```", re.M | re.S)


def _anchors(path: Path) -> set[str]:
    text = FENCE.sub("", path.read_text())
    out = set()
    for heading in re.findall(r"^#{1,6} +(.+?) *$", text, re.M):
        slug = re.sub(r"[^\w\- ]", "", heading.strip().lower().replace("`", ""))
        out.add(slug.replace(" ", "-"))
    return out


def _links(path: Path) -> list[str]:
    return [target for target in LINK.findall(FENCE.sub("", path.read_text())) if "://" not in target]


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.relative_to(REPO).as_posix())
def test_every_link_lands_on_a_file_and_a_heading_that_exist(doc):
    broken = []
    for target in _links(doc):
        if target.startswith("mailto:"):
            continue
        file, _, anchor = target.partition("#")
        dest = (doc.parent / file).resolve() if file else doc
        if not dest.exists():
            broken.append(f"{target}: no such file")
        elif anchor and dest.suffix == ".md" and anchor not in _anchors(dest):
            broken.append(f"{target}: no such heading in {dest.name}")
    assert broken == []


def test_the_check_knows_a_heading_from_a_missing_one():
    anchors = _anchors(REPO / "docs" / "CONFIGURATION.md")
    assert {"the-rig", "the-papers-five", "phase-planning-hitl", "tandem-settings-and-credentials"} <= anchors
    assert "presets" not in anchors and "cameras-and-calibration" not in anchors
