"""What the installed package must be, for `pip install tandem-tamp` to mean anything."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Anything here would drag CUDA, a camera SDK or a robot client into the base install and
# break `pip install tandem-tamp && tandem ui` on a laptop.
FORBIDDEN = ("torch", "cv2", "pyzed", "open3d", "curobo", "cutamp", "tiptop", "warp", "rerun_sdk")

# Every module the light path touches. Imported in a subprocess with the heavy packages
# blocked, so an accidental top-level import fails here rather than on a user's laptop.
LIGHT_MODULES = [
    "tandem",
    "tandem.cli.app",
    "tandem.cli.theme",
    "tandem.core.paths",
    "tandem.core.settings",
    "tandem.core.secrets",
    "tandem.core.profiles",
    "tandem.core.render",
    "tandem.core.trajectories",
    "tandem.core.series",
    "tandem.core.nonidle",
    "tandem.core.merge",
    "tandem.core.events",
    "tandem.core.session",
    "tandem.core.runtime",
    "tandem.core.probe",
    "tandem.core.importers",
    "tandem.server.app",
    "tandem.teleop",
    # Phase planning and the backend protocol are on the laptop path too. The planner package used
    # to be the planner's own code and imported cuTAMP's symbolic layer for its atoms; the reason it
    # is tandem's now is exactly so this list can contain it.
    "tandem.planning.symbols",
    "tandem.planning.structs",
    "tandem.planning.prompts",
    "tandem.planning.config",
    "tandem.planning.proposal",
    "tandem.planning.grounding",
    "tandem.planning.feasibility",
    "tandem.planning.drift",
    "tandem.planning.plan",
    "tandem.planning.objects",
    "tandem.planners.base",
    "tandem.planners.registry",
    "tandem.planners.rpc",
    "tandem.planners.tiptop",
    "tandem.planners.tiptop.backend",
    "tandem.cli.plan",
]


def test_no_heavy_dependency_in_the_base_install():
    """The laptop path is the whole reason for the thin-pip design: `tandem ui` has to work
    with no GPU, no robot and no cameras. A stray top-level `import torch` anywhere in these
    modules silently ends that."""
    script = f"""
import builtins, sys
forbidden = {FORBIDDEN!r}
real_import = builtins.__import__

def guard(name, *args, **kwargs):
    root = name.split(".")[0]
    if root in forbidden:
        raise AssertionError(f"base install imported a heavy dependency: {{name}}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guard
for module in {LIGHT_MODULES!r}:
    real_import(module)
print("ok")
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout


def test_package_data_is_present():
    """Files that only exist as package data — a wheel that drops them installs a CLI that
    cannot create a profile or serve a page."""
    import tandem

    root = Path(tandem.__file__).parent
    assert (root / "resources" / "profile_template.yml").is_file()
    assert (root / "server" / "static" / "index.html").is_file()
    assert (root / "server" / "static" / "app.js").is_file()
    assert (root / "server" / "static" / "theme.css").is_file()
    assert (root / "server" / "static" / "pages" / "collect.js").is_file()


def test_the_ui_has_no_external_asset_references():
    """No CDN, no bundler, no npm. If a script or stylesheet ever points off-host, the UI
    stops working on the air-gapped workstation it is meant for."""
    import tandem

    static = Path(tandem.__file__).parent / "server" / "static"
    for path in static.rglob("*"):
        if path.suffix not in {".html", ".js", ".css"}:
            continue
        text = path.read_text()
        for marker in ("https://cdn", "http://cdn", "unpkg.com", "jsdelivr", "googleapis.com"):
            assert marker not in text, f"{path.name} references an external asset: {marker}"


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / "src" / "tandem" / "_vendor" / "VENDOR.toml").is_file(),
    reason="vendored sources are not present in this checkout",
)
def test_vendored_sources_are_pinned_and_trimmed():
    """Provenance is not optional here: two of the three vendored trees are under a licence
    that governs redistribution, and the wheel has to stay a reasonable size."""
    import tomlkit

    vendor = Path(__file__).parent.parent / "src" / "tandem" / "_vendor"
    manifest = tomlkit.parse((vendor / "VENDOR.toml").read_text())

    for component in ("tiptop", "cuTAMP", "curobo"):
        assert component in manifest, f"{component} is missing from VENDOR.toml"
        assert len(manifest[component]["commit"]) == 40, f"{component} is not pinned to a full commit"
        assert (vendor / component).is_dir()
        # Each tree keeps its own licence — the NVIDIA License requires it, and MIT does too.
        assert (vendor / component / "LICENSE").is_file(), f"{component} lost its LICENSE"

    total = sum(f.stat().st_size for f in vendor.rglob("*") if f.is_file())
    assert total < 90e6, (
        f"the vendor tree is {total / 1e6:.0f} MB; the trim list in tools/vendor.py has stopped working"
    )


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / "src" / "tandem" / "_vendor" / "tiptop").is_dir(),
    reason="vendored sources are not present in this checkout",
)
def test_tiptop_patches_are_applied():
    """Without these, a profile cannot supply its own config and every session would read
    whatever tiptop.yml happens to sit in the shared runtime."""
    config = (
        Path(__file__).parent.parent
        / "src" / "tandem" / "_vendor" / "tiptop" / "tiptop" / "config" / "__init__.py"
    ).read_text()
    assert "TIPTOP_CONFIG" in config
    assert "TIPTOP_CALIBRATION" in config


def test_every_module_attribute_the_cli_and_server_reach_for_actually_exists():
    """Catches the class of breakage a refactor causes and no test notices.

    Python resolves ``some_module.CONSTANT`` at call time, so deleting a constant one module deep
    breaks a caller only when that line runs — and the line that broke this way was on
    ``tandem collect``'s normal exit path, reached after a whole session had been driven. Importing
    the module proves nothing; this walks the references.
    """
    import ast
    import importlib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "tandem"
    # Module aliases the CLI and server import and then reach into by attribute.
    aliases = {
        "session_mod": "tandem.core.session",
        "settings_mod": "tandem.core.settings",
        "merge_mod": "tandem.core.merge",
        "runtime_mod": "tandem.core.runtime",
        "events_mod": "tandem.core.events",
        "theme": "tandem.cli.theme",
        "paths": "tandem.core.paths",
        "render": "tandem.core.render",
        "profiles": "tandem.core.profiles",
        "secrets": "tandem.core.secrets",
        "trajectories": "tandem.core.trajectories",
    }
    resolved = {alias: dir(importlib.import_module(name)) for alias, name in aliases.items()}

    missing: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if "_vendor" in path.parts:
            continue
        source = path.read_text()
        # Only check a file that actually imports the alias, so a local variable of the same name
        # in an unrelated module is not mistaken for it.
        tree = ast.parse(source)
        imported = {
            (a.asname or a.name)
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for a in node.names
        }
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)):
                continue
            alias = node.value.id
            if alias not in resolved or alias not in imported:
                continue
            if node.attr not in resolved[alias]:
                missing.append(f"{path.relative_to(root.parent.parent)}:{node.lineno} {alias}.{node.attr}")

    assert not missing, "references to names that no longer exist:\n  " + "\n  ".join(missing)


def test_a_runtime_is_re_copied_when_the_vendored_planner_changes(tmp_path):
    """"Already present" is only safe when the sources are the SAME sources.

    Skipping the copy while stamping the new commits anyway is the worst of both: the runtime is
    built from the old planner and claims to be the new one. That surfaces as an ImportError
    partway into a warm-up, with an operator standing next to the arm — the sidecar calls functions
    that exist only in the newer tree.
    """
    import json

    from tandem.core.runtime import Runtime

    def vendor_tree(root, commit: str):
        for name in ("tiptop", "cuTAMP", "curobo"):
            (root / name).mkdir(parents=True, exist_ok=True)
            (root / name / "marker.txt").write_text(commit)
        (root / "VENDOR.toml").write_text(
            "\n".join(
                f'[{name}]\ncommit = "{commit}"\nurl = ""\nversion = "{commit[:7]}"'
                for name in ("tiptop", "cuTAMP", "curobo")
            )
        )
        return root

    first = vendor_tree(tmp_path / "vendor-a", "1111111111111111")
    runtime = Runtime(tmp_path / "runtime")
    runtime.materialize(first)
    assert (runtime.root / "tiptop" / "marker.txt").read_text() == "1111111111111111"

    # Re-vendored: same paths, different commits.
    second = vendor_tree(tmp_path / "vendor-b", "2222222222222222")
    runtime.materialize(second)
    assert (runtime.root / "tiptop" / "marker.txt").read_text() == "2222222222222222", (
        "the runtime kept the old planner while the stamp claimed the new one"
    )
    stamp = json.loads(runtime.stamp_file.read_text())
    assert stamp["vendor"]["tiptop"]["commit"] == "2222222222222222"
    # A source change invalidates the compiled kernels and the installed packages with it.
    assert stamp["built_at"] is None

    # Unchanged sources are still left alone — this must not turn every init into a full re-copy.
    marker = runtime.root / "tiptop" / "untouched"
    marker.write_text("x")
    runtime.materialize(second)
    assert marker.is_file(), "an unchanged component should not be re-copied"


def test_the_planner_build_pins_curobos_version(tmp_path, monkeypatch):
    """cuRobo takes its version from setuptools_scm and a vendored tree has no SCM metadata.

    Vendoring extracts with `git archive` precisely so no VCS state rides along, so the editable
    install fails with "unable to detect version" before the 5-20 minute CUDA kernel build even
    starts — and `tandem init` ends with "the build finished but the runtime still looks
    incomplete", which names neither the cause nor the fix.
    """
    import json

    from tandem.core.runtime import Runtime

    runtime = Runtime(tmp_path / "runtime")
    runtime.root.mkdir(parents=True)
    runtime.stamp_file.write_text(
        json.dumps({"vendor": {"curobo": {"commit": "abc123", "version": "3a90ff4"}}, "built_at": None})
    )

    captured: dict = {}
    monkeypatch.setattr(
        Runtime, "_pixi", lambda self, args, log=None, extra_env=None, what="": captured.update(extra_env or {})
    )
    monkeypatch.setattr(Runtime, "_touch_built", lambda self: None)
    runtime.build_planners()

    version = captured.get("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO")
    assert version, "cuRobo's version is not pinned, so its editable install cannot resolve one"
    # Carries the vendored commit, so the installed package says which sources it is.
    assert "3a90ff4" in version
