"""What the installed package must be, for `pip install tandem-tamp` to mean anything."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# Anything here would drag CUDA, a camera SDK or a robot client into the base install and
# break `pip install tandem-tamp && tandem ui` on a laptop.
FORBIDDEN = ("torch", "cv2", "pyzed", "open3d", "curobo", "cutamp", "tiptop", "warp", "rerun_sdk")

# Every module the light path touches. Imported in a subprocess with the heavy packages
# blocked, so an accidental top-level import fails here rather than on a user's laptop.
LIGHT_MODULES = [
    "tandem",
    # The library entry point (`tandem.plan_task`): a decomposition from a photo, on a laptop.
    "tandem.api",
    "tandem.cli.app",
    "tandem.cli.theme",
    "tandem.core.paths",
    "tandem.core.settings",
    "tandem.core.secrets",
    "tandem.core.profiles",
    "tandem.core.trajectories",
    "tandem.core.series",
    "tandem.core.nonidle",
    "tandem.core.merge",
    "tandem.core.events",
    # The rig and moving old profiles: `tandem rig show` and `tandem profile migrate` run on a laptop.
    "tandem.core.rig",
    "tandem.core.layout",
    "tandem.cli.rig",
    "tandem.core.session",
    "tandem.core.probe",
    "tandem.core.phase_loop",
    "tandem.core.episodes",
    "tandem.server.app",
    "tandem.teleop",
    "tandem.teleop.child",
    # Who carries out a human phase. The registry is read when a profile loads, so it and the one
    # executor that ships have to stay as light as the profile itself.
    "tandem.executors",
    "tandem.executors.base",
    "tandem.executors.teleop",
    # Phase planning and the backend protocol are on the laptop path too. The planner package used
    # to be the planner's own code and imported cuTAMP's symbolic layer for its atoms; the reason it
    # is tandem's now is exactly so this list can contain it.
    "tandem.planning.symbols",
    "tandem.planning.structs",
    "tandem.planning.prompts",
    "tandem.planning.config",
    "tandem.planning.proposal",
    "tandem.planning.grounding",
    "tandem.planning.contracts",
    "tandem.planning.feasibility",
    "tandem.planning.drift",
    "tandem.planning.plan",
    "tandem.planning.objects",
    "tandem.planners.base",
    "tandem.planners.registry",
    "tandem.planners.rpc",
    "tandem.planners.tiptop",
    "tandem.planners.tiptop.backend",
    "tandem.planners.tiptop.factory",
    # TiPToP's own config, schema, probes and doctor rows: read to validate a profile and the rig, and to
    # run doctor, all of which a laptop does.
    "tandem.planners.tiptop.render",
    "tandem.planners.tiptop.tamp_keys",
    "tandem.planners.tiptop.options",
    "tandem.planners.tiptop.doctor",
    "tandem.planners.tiptop.runtime",
    "tandem.planners.tiptop.probe",
    # Deprecated, and only a re-export; kept light like what it re-exports.
    "tandem.core.runtime",
    # A planner's runtime recipe is read to list planners, so fetching and building stay behind calls.
    "tandem.planners.runtime",
    "tandem.planners.tiptop.recipe",
    # The planner SDK: a planner class is imported to list planners, and a plugin's test suite imports
    # the conformance kit on a laptop. (Not tandem_sidecar: it is not a tandem module, and importing
    # it takes over stdout -- it runs only inside a planner's sidecar.)
    "tandem.planners.sdk",
    "tandem.planners.sidecar",
    "tandem.planners.sidecar_kit",
    "tandem.planners.testing",
    "tandem.cli.runtime",
    "tandem.cli.plan",
    # The catalogs: listing planners and executors is the laptop's question as much as the rig's.
    "tandem.core.names",
    # `tandem profile create --preset` and `tandem profile presets` run on a laptop too.
    "tandem.core.presets",
    "tandem.cli.planners",
    "tandem.cli.executors",
    "tandem.server.routes.planners",
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


SRC = Path(__file__).resolve().parents[1] / "src" / "tandem"


def _package_data_globs() -> list[str]:
    import tomlkit

    pyproject = tomlkit.parse((SRC.parents[1] / "pyproject.toml").read_text())
    return [str(g) for g in pyproject["tool"]["setuptools"]["package-data"]["tandem"]]


def test_no_planner_source_ships_inside_the_package():
    """The wheel is pure Python and small: a planner's sources are fetched by its runtime recipe.

    Two of the three trees tandem drives are under NVIDIA's licence; not redistributing them is half
    the point, and a stray glob or a stray copy would quietly undo it.
    """
    assert not (SRC / "_vendor").exists(), "src/tandem/_vendor is back"
    for marker in ("tiptop_run.py", "tamp_domain.py", "curobolib"):
        found = [str(p.relative_to(SRC)) for p in SRC.rglob(marker)]
        assert not found, f"planner sources inside the package: {found}"
    assert not any("_vendor" in glob for glob in _package_data_globs())


def test_tiptops_patches_and_checkpoints_ship_as_package_data():
    """What a planner package ships is its own: the patches its recipe applies, and the two DATAFARM
    checkpoints no public repository has. A wheel that dropped them installs a runtime that cannot be
    patched, or a VAE cost that fails at the first plan."""
    from fnmatch import fnmatch

    from tandem.planners.tiptop.recipe import RECIPE

    shipped = [p for s in RECIPE.sources for p in s.patches] + [a.source for a in RECIPE.assets]
    # One patch (0001, $TIPTOP_CALIBRATION) and two checkpoints. 0002 was dropped with the bump to
    # tiptop 1c6daf3: it made tiptop's pixi.lock stale (tests/test_tiptop_bump.py).
    assert len(shipped) == 3
    globs = _package_data_globs()
    for path in shipped:
        assert path.is_file(), f"{path} is missing"
        relative = path.relative_to(SRC).as_posix()
        assert any(fnmatch(relative, glob.replace("**/", "")) or fnmatch(relative, glob) for glob in globs), (
            f"{relative} matches no package-data glob, so the wheel would not carry it"
        )
    sizes = {a.source.name: a.source.stat().st_size for a in RECIPE.assets}
    assert 1e6 < sizes["vae_full_v2.pt"] < 2e6 and 2e6 < sizes["rnd_droid.pt"] < 4e6


def test_the_sidecar_kit_ships_as_a_file_a_planners_environment_can_import():
    """Every sidecar imports tandem_sidecar from the directory tandem puts on its path, so it has to be
    in the wheel as a plain file -- and must import nothing a planner's environment might not have."""
    import ast

    from tandem.planners import sidecar_kit

    kit = sidecar_kit.path()
    assert kit.is_file() and kit.parent == SRC / "planners" / "sidecar_kit"
    # A module of a package that setuptools finds, so the wheel carries it without a package-data glob.
    assert (kit.parent / "__init__.py").is_file()
    roots = set()
    for node in ast.walk(ast.parse(kit.read_text())):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots <= set(sys.stdlib_module_names) | {"__future__"}, sorted(roots)


def test_tiptops_patches_are_plain_diffs():
    """Run from inside a repository, `git apply` filters a `diff --git` patch by the current prefix and
    reports success while applying nothing. tandem guards against that; the patches avoid it too."""
    from tandem.planners.tiptop.recipe import RECIPE

    for patch in (p for s in RECIPE.sources for p in s.patches):
        text = patch.read_text()
        assert "\ndiff --git " not in text and not text.startswith("diff --git ")
        assert "\n--- a/" in text and "\n+++ b/" in text


def test_tiptops_patches_apply_to_the_pinned_tiptop(tmp_path):
    """Without them a profile cannot supply its own calibration, and every session would read whatever
    sits in the shared runtime. Checked against the real tree wherever one is to hand."""
    from planner_sources import planner_sources

    from tandem.planners import runtime as rt_mod
    from tandem.planners.tiptop.recipe import RECIPE, TIPTOP

    root = planner_sources("tiptop/tiptop/config/__init__.py")
    if "TIPTOP_CALIBRATION" in (root / "tiptop" / "tiptop" / "config" / "__init__.py").read_text():
        # An installed runtime, whose tree is patched already: that it is IS the check.
        return
    tree = tmp_path / "tiptop"
    shutil.copytree(root / "tiptop", tree, symlinks=True, ignore=shutil.ignore_patterns(".git", ".pixi"))
    applied = rt_mod.apply_patches(tree, TIPTOP.patches, name="tiptop")
    assert [a["name"] for a in applied] == [p.name for p in RECIPE.source("tiptop").patches]
    config = (tree / "tiptop" / "config" / "__init__.py").read_text()
    assert "TIPTOP_CONFIG" in config and "TIPTOP_CALIBRATION" in config


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
        "runtime_mod": "tandem.planners.tiptop.runtime",
        "events_mod": "tandem.core.events",
        "theme": "tandem.cli.theme",
        "paths": "tandem.core.paths",
        "render": "tandem.planners.tiptop.render",
        "profiles": "tandem.core.profiles",
        "secrets": "tandem.core.secrets",
        "trajectories": "tandem.core.trajectories",
    }
    resolved = {alias: dir(importlib.import_module(name)) for alias, name in aliases.items()}

    missing: list[str] = []
    for path in sorted(root.rglob("*.py")):
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
