"""Presets are gone: the paper's settings are five profiles and the template a new one starts from.

A preset was a second way to say what a profile already says -- settings laid over another profile, in
two halves (tandem's and the planner's), with its own format, loader, conformance check, CLI flag and web
field. Nothing of it is left to half-work: not the module, not the files, not a planner hook, not a route,
and not a flag or a mention anywhere a person would read one (the hitl-tamp-vla importer's flags included,
which went the same way; tests/test_planner_options.py checks its hook and module are gone).
"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import tandem
from tandem.cli.app import app
from tandem.planners import registry, sdk, testing
from tandem.planners.tiptop.factory import FACTORY as TIPTOP
from tandem.server.app import create_app

SRC = Path(tandem.__file__).parent
REPO = SRC.parents[1]

#: What a person could still be told to type, or a planner author to write, that no longer exists.
GONE = re.compile(r"--preset\b|--import-from|--tamp-config|presets_dir|tandem profile presets|check_presets")


def test_the_module_and_its_files_are_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("tandem.core.presets")
    assert not (SRC / "resources" / "presets").exists()
    assert not list(SRC.glob("planners/*/presets"))


def test_no_planner_hook_or_conformance_check_is_left():
    assert not hasattr(sdk.Planner, "presets_dir")
    assert not hasattr(TIPTOP, "presets_dir")
    assert not hasattr(registry, "presets_dir")
    assert not hasattr(testing, "check_presets")
    assert not any("preset" in name for name in dir(testing.PlannerConformance))


def test_planners_info_says_nothing_of_presets():
    result = CliRunner().invoke(app, ["planners", "info", "tiptop", "--json"])
    assert result.exit_code == 0, result.output
    assert not [key for key in json.loads(result.output) if "preset" in key]
    shown = CliRunner().invoke(app, ["planners", "info", "tiptop"])
    assert shown.exit_code == 0 and "preset" not in shown.output


def test_the_web_neither_lists_nor_takes_them(profile):
    client = TestClient(create_app())
    listed = client.get("/api/presets")
    assert listed.status_code == 404 and "No such endpoint" in listed.json()["error"]
    refused = client.post("/api/profiles", json={"name": "bread", "prompt": "bread in the box", "preset": "paper"})
    assert refused.status_code == 400 and "does not take preset" in refused.json()["error"]


def _mentions(root: Path, suffixes: tuple[str, ...]) -> list[str]:
    found = []
    for path in sorted(root.rglob("*")):
        if path.suffix in suffixes and "__pycache__" not in path.parts:
            for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                if GONE.search(line):
                    found.append(f"{path.relative_to(REPO)}:{number}: {line.strip()}")
    return found


def test_nothing_the_package_says_or_ships_mentions_them():
    assert _mentions(SRC, (".py", ".js", ".yml", ".tmpl", ".md", ".toml", ".html")) == []


def test_nothing_the_docs_say_mentions_them():
    assert _mentions(REPO / "docs", (".md",)) == []
    readme = [
        f"README.md:{number}: {line.strip()}"
        for number, line in enumerate((REPO / "README.md").read_text().splitlines(), 1)
        if GONE.search(line)
    ]
    assert readme == []
