"""`tandem planners new`: a planner package that is green before its author writes a line of it.

The promise is the one a new plugin author needs most: what is scaffolded installs, is found by
tandem through its entry point like any other planner, and passes tandem's conformance kit as it
stands -- both kinds, a planner in tandem's process and one behind a sidecar. So each is generated
into a temporary directory and its OWN test suite is run there, in a fresh interpreter, exactly as its
author would run `pytest`. Everything its author then changes is checked by a kit that was passing.
"""

from __future__ import annotations

import ast
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path

import pytest
import tomlkit
from helpers import isolate_registry
from typer.testing import CliRunner

import tandem
from tandem.cli import planners as planners_cli
from tandem.cli.app import app
from tandem.core import profiles
from tandem.core.errors import TandemError
from tandem.planners import registry
from tandem.planners.testing import check_declarations, check_sidecar_script

SRC = Path(tandem.__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    isolate_registry(monkeypatch)


def _new(tmp_path: Path, name: str, *extra: str) -> Path:
    dest = tmp_path / f"tandem-{name}"
    result = CliRunner().invoke(app, ["planners", "new", name, "--dir", str(dest), *extra])
    assert result.exit_code == 0, result.output
    return dest


def _run_its_own_tests(package: Path) -> subprocess.CompletedProcess:
    """`pytest` in the package's directory, in a fresh interpreter that has tandem and nothing else of ours."""
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider"],
        cwd=package,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _count(output: str, what: str) -> int:
    found = re.search(rf"(\d+) {what}", output)
    return int(found.group(1)) if found else 0


@pytest.mark.parametrize(
    ("kind", "extra", "script"),
    [("in-process", (), False), ("sidecar", ("--sidecar",), True)],
)
def test_a_new_planner_package_passes_the_conformance_kit_as_written(tmp_path, kind, extra, script):
    package = _new(tmp_path, "bins", *extra)
    module = package / "src" / "tandem_bins"
    assert (module / "planner.py").is_file() and (module / "__init__.py").is_file()
    assert (module / "sidecar.py").is_file() is script
    assert (package / "tests" / "test_conformance.py").is_file()

    result = _run_its_own_tests(package)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert _count(output, "failed") == 0 and _count(output, "error") == 0, output
    # The kit's lifecycle, scene, plan, leg, stop and custody checks all ran and passed; what it skips
    # is only what the stub does not declare (a restriction, ending away from home). Never the camera
    # frame or the perception image: phase planning needs both, and the kit requires both by default.
    assert _count(output, "passed") >= 9, output
    assert "cannot capture a frame" not in output, output
    assert "phase_planning" not in output and "verifies_human_phases" not in output, output
    if script:
        assert "no sidecar script" not in output, "the sidecar script itself was checked"


def test_the_generated_package_is_the_planner_tandem_finds_through_its_entry_point(
    tmp_path, monkeypatch, profile
):
    package = _new(tmp_path, "my-arm", "--sidecar")
    pyproject = tomlkit.parse((package / "pyproject.toml").read_text())
    assert pyproject["project"]["name"] == "tandem-my-arm"
    target = pyproject["project"]["entry-points"]["tandem.planners"]["my-arm"]
    assert target == "tandem_my_arm.planner:MyArmPlanner"
    assert any(dep.startswith("tandem-tamp>=") for dep in pyproject["project"]["dependencies"])

    # Installed, as far as tandem can tell: importable, and declared under the entry point group.
    monkeypatch.syspath_prepend(str(package / "src"))
    declared = [importlib.metadata.EntryPoint("my-arm", str(target), registry.GROUP)]
    real = importlib.metadata.entry_points
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda **params: list(declared) if params.get("group") == registry.GROUP else real(**params),
    )
    try:
        assert "my-arm" in registry.available()
        planner = registry.factory("my-arm")
        assert planner.__name__ == "MyArmPlanner" and planner.info.name == "my-arm"
        check_declarations(planner)
        check_sidecar_script(planner)

        listing = CliRunner().invoke(app, ["planners", "list", "--json"])
        assert listing.exit_code == 0, listing.output
        (row,) = [r for r in json.loads(listing.output)["planners"] if r["name"] == "my-arm"]
        assert row["ok"] and row["status"] == "no runtime needed"
        assert row["origin"].startswith("entry point") and row["display_name"] == "My Arm"

        # And a profile can plan with it, the way it would with any planner.
        used = CliRunner().invoke(app, ["planners", "use", "my-arm", "--profile", profile.name])
        assert used.exit_code == 0, used.output
        assert profiles.load(profile.name).planner.backend == "my-arm"
    finally:
        for name in [m for m in sys.modules if m == "tandem_my_arm" or m.startswith("tandem_my_arm.")]:
            sys.modules.pop(name, None)


def test_the_sidecar_it_generates_never_imports_tandem(tmp_path):
    package = _new(tmp_path, "bins", "--sidecar")
    tree = ast.parse((package / "src" / "tandem_bins" / "sidecar.py").read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    # The planner's environment has only the standard library, the kit tandem puts on the path, and
    # whatever the planner itself brings.
    assert roots <= set(sys.stdlib_module_names) | {"__future__", "tandem_sidecar"}, sorted(roots)


def test_the_scaffold_refuses_what_it_cannot_write_safely(tmp_path):
    for bad in ("Bins", "1bins", "my bins", "pkg.mod"):
        with pytest.raises(TandemError, match="cannot be a planner's name"):
            planners_cli.scaffold(bad, tmp_path / "x")
    # A name that is taken would be shadowed by the planner that has it, and never used.
    with pytest.raises(TandemError, match="already a planner named 'tiptop'") as caught:
        planners_cli.scaffold("tiptop", tmp_path / "x")
    assert "built-in" in caught.value.message

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "notes.txt").write_text("mine")
    result = CliRunner().invoke(app, ["planners", "new", "bins", "--dir", str(occupied)])
    assert result.exit_code == 1 and "not empty" in result.exception.message
    assert sorted(p.name for p in occupied.iterdir()) == ["notes.txt"], "nothing was written"
    assert not (tmp_path / "x").exists()

    # An empty directory is fine: it is where a person made room for it.
    empty = tmp_path / "empty"
    empty.mkdir()
    written = planners_cli.scaffold("bins", empty)
    assert {p.relative_to(empty).as_posix() for p in written} == {
        "pyproject.toml",
        "README.md",
        ".gitignore",
        "src/tandem_bins/__init__.py",
        "src/tandem_bins/planner.py",
        "tests/test_conformance.py",
    }


def test_a_name_becomes_a_module_a_class_a_distribution_and_a_title():
    values = planners_cli.scaffold_values("my-arm")
    assert (values["module"], values["class_name"], values["dist"], values["title"]) == (
        "tandem_my_arm",
        "MyArmPlanner",
        "tandem-my-arm",
        "My Arm",
    )
    assert planners_cli.scaffold_values("fast_planner")["class_name"] == "FastPlanner"
    assert planners_cli.scaffold_values("bins")["class_name"] == "BinsPlanner"


def test_every_template_is_filled_completely_and_ships_in_the_wheel():
    globs = [
        str(g)
        for g in tomlkit.parse((SRC.parent / "pyproject.toml").read_text())["tool"]["setuptools"][
            "package-data"
        ]["tandem"]
    ]
    for sidecar in (False, True):
        values = planners_cli.scaffold_values("bins", sidecar=sidecar)
        for template, _target, _variant in planners_cli._FILES:
            text = planners_cli.read_template(template)
            rendered = planners_cli.render_template(text, values, template=template)
            assert "{{" not in rendered and "}}" not in rendered, template
            relative = f"resources/{planners_cli.SCAFFOLD}/{template}"
            assert any(fnmatch(relative, glob) for glob in globs), f"{relative} is not package data"
            # Never a name CI's wheel check reads as a planner's own sources.
            assert "pixi.toml" not in relative and "pixi.lock" not in relative
            # Templates are data, not modules: nothing may take one for tandem's code.
            assert not template.endswith(".py")
    with pytest.raises(TandemError, match="uses nobody"):
        planners_cli.render_template("{{nobody}}", {}, template="t")


@pytest.mark.skipif(
    shutil.which("ruff") is None and not (Path(sys.executable).parent / "ruff").is_file(),
    reason="ruff is not installed here",
)
def test_the_generated_code_is_lint_clean_under_its_own_settings(tmp_path):
    ruff = shutil.which("ruff") or str(Path(sys.executable).parent / "ruff")
    for extra in ((), ("--sidecar",)):
        package = _new(tmp_path / ("sidecar" if extra else "plain"), "bins", *extra)
        for command in (["check", "."], ["format", "--check", "."]):
            result = subprocess.run([ruff, *command], cwd=package, capture_output=True, text=True)
            assert result.returncode == 0, result.stdout + result.stderr
