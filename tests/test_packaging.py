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
