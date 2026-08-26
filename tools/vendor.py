#!/usr/bin/env python3
"""Re-vendor tiptop, cuTAMP and cuRobo into src/tandem/_vendor/.

The three planner components ship inside the wheel so `pip install tandem-tamp` is the only
install anyone runs — `tandem init` then needs no network and no git to build the runtime.

They are trimmed on the way in. cuRobo alone carries 152 MB of robot assets for embodiments
this pipeline does not support, and tiptop's docs carry 22 MB of screen recordings; keeping
them would triple the wheel for no benefit.

Usage:
    python tools/vendor.py --source /path/to/hitl-tamp-vla
    python tools/vendor.py --source /path/to/hitl-tamp-vla --component curobo

Everything is extracted with `git archive`, so a build artifact left in someone's working
tree can never end up in a release.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VENDOR = REPO / "src" / "tandem" / "_vendor"

# Paths dropped from each component, relative to its root. Every entry is either an
# embodiment this pipeline cannot drive, or documentation media.
TRIM = {
    "tiptop": [
        "docs/_static",  # 22 MB of screen recordings and screenshots
        "docs/blogs",
        ".github",
    ],
    "cuTAMP": [
        "cutamp/robots/assets/yam_description",  # 11 MB; the bimanual YAM is out of scope
        "docs",
    ],
    "curobo": [
        # 115 MB of robot meshes for arms this pipeline does not support. Franka and UR stay:
        # cuTAMP's fr3_robotiq and ur5e configs reference them.
        "src/curobo/content/assets/robot/techman",
        "src/curobo/content/assets/robot/iiwa_allegro_description",
        "src/curobo/content/assets/robot/jaco",
        "src/curobo/content/assets/robot/kinova",
        # An nvblox demo scene (a UR10 bins mesh). nvblox_torch is not installed and the world
        # here comes from the ZED point cloud, so nothing reads it.
        "src/curobo/content/assets/scene/nvblox",
        "images",
        "benchmark",
        "docker",
        ".github",
    ],
}

# Small checkpoints that live OUTSIDE the three components in the source monorepo. The
# cuRobo fork's VAE and RND costs default to finding them at these paths relative to what
# they take to be a repo root, and tandem's runtime reproduces that layout — so vendoring
# them here means the manifold and novelty costs work out of the box.
EXTRA_FILES = [
    ("vae/checkpoints/vae_full_v2.pt", "vae/checkpoints/vae_full_v2.pt"),
    ("rnd/checkpoints/rnd_droid.pt", "rnd/checkpoints/rnd_droid.pt"),
]

# Where each component's sources come from inside the monorepo. tiptop is usually a
# submodule whose working tree may not be checked out, so it is read from its git dir.
SUBMODULES = {"tiptop": ".git/modules/tiptop", "cuTAMP": ".git/modules/cuTAMP", "curobo": ".git/modules/curobo"}

# Which ref each component is vendored from. Named here rather than taken from whatever the
# checkout's HEAD happens to be, and EXPLICIT for all three rather than `HEAD`: a source checkout
# is usually a working monorepo whose submodules sit on a detached head or a local branch, so
# `HEAD` vendors whatever someone was last working on. It has already produced a build carrying an
# unpushed commit, which is unreproducible by anyone else and unrecorded except as a hash.
#
# All three are the clean upstreams, with no phase-planning logic in any of them. That logic is
# tandem's own now (src/tandem/planning), and tandem drives an unmodified planner through
# src/tandem/planners — which is the whole point: a planner tandem has to fork is a planner tandem
# has to keep forking.
#
# tiptop and cuTAMP MUST move together. Upstream tiptop passes `pick_transparent` and `q_return`,
# which older cuTAMP trees do not accept; pinning one forward and not the other is a TypeError on
# every plan.
REFS = {
    "tiptop": "origin/main",
    "cuTAMP": "origin/main",
    "curobo": "origin/main",
}

PATCHES = REPO / "tools" / "patches"

# Which component each patch applies to, by filename prefix.
PATCH_TARGET = {"0001": "tiptop", "0002": "tiptop"}


def git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed in {cwd}:\n{result.stderr}")
    return result.stdout.strip()


def resolve_git_dir(source: Path, component: str) -> tuple[Path, list[str]]:
    """Return (cwd, extra git args) for reading `component` out of `source`."""
    worktree = source / component
    if (worktree / ".git").is_dir():
        return worktree, []
    module = source / SUBMODULES[component]
    if module.is_dir():
        return source, [f"--git-dir={module}"]
    if (worktree / ".git").is_file():
        # A submodule whose .git is a gitdir pointer, with the objects up in the parent.
        return source, [f"--git-dir={source / SUBMODULES[component]}"]
    raise SystemExit(f"Cannot find git history for {component} under {source}")


def export(source: Path, component: str, dest: Path, ref: str) -> dict:
    cwd, extra = resolve_git_dir(source, component)
    try:
        commit = git([*extra, "rev-parse", ref], cwd)
    except SystemExit:
        raise SystemExit(
            f"{component}: no ref {ref!r} in {source}. Fetch it first "
            f"(git -C {source} submodule foreach git fetch --all), or pass --ref {component}=<ref>."
        ) from None
    url = git([*extra, "config", "--get", "remote.origin.url"], cwd) or ""
    described = git([*extra, "describe", "--tags", "--always", commit], cwd)
    print(f"  {component}: {ref} -> {commit[:12]}")

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    with tempfile.NamedTemporaryFile(suffix=".tar") as archive:
        subprocess.run(
            ["git", *extra, "archive", "--format=tar", "-o", archive.name, commit],
            cwd=cwd, check=True,
        )
        with tarfile.open(archive.name) as tar:
            tar.extractall(dest, filter="data")

    dropped = []
    for relative in TRIM.get(component, []):
        target = dest / relative
        if target.is_dir():
            size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            shutil.rmtree(target)
            dropped.append((relative, size))
        elif target.is_file():
            dropped.append((relative, target.stat().st_size))
            target.unlink()

    kept = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"  {component}: {kept / 1e6:.1f} MB kept, {sum(s for _, s in dropped) / 1e6:.1f} MB trimmed")
    for relative, size in dropped:
        print(f"      - {relative}  ({size / 1e6:.1f} MB)")

    patches = apply_patches(component)

    return {
        "commit": commit,
        "url": url,
        "version": described,
        "ref": ref,
        "trimmed": [relative for relative, _ in dropped],
        "patches": patches,
        "bytes": kept,
    }


def apply_patches(component: str) -> list[str]:
    """Apply this component's patches to the freshly-exported tree.

    Kept as unified diffs under tools/patches/ rather than edited in place, so re-vendoring a
    newer upstream is mechanical and a patch that no longer applies fails loudly instead of
    being silently lost.
    """
    if not PATCHES.is_dir():
        return []
    applied = []
    for patch in sorted(PATCHES.glob("*.patch")):
        if PATCH_TARGET.get(patch.name[:4]) != component:
            continue
        result = subprocess.run(
            ["git", "apply", "--verbose", "--unsafe-paths", f"--directory={component}", str(patch)],
            cwd=VENDOR, capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            raise SystemExit(
                f"{patch.name} does not apply to the new {component}:\n{result.stderr}\n"
                "Upstream moved. Rewrite the patch against the new source before re-vendoring."
            )
        # Exit 0 is NOT enough. Run from inside a repository, `git apply` filters a git-style diff
        # (one carrying `diff --git` headers) by the current prefix and quietly drops everything
        # outside it -- reporting "Skipped patch" on stderr and success to the shell. A patch that
        # went nowhere is exactly the failure this whole mechanism exists to make loud, and it has
        # already shipped a vendor tree missing a patch that was reported as applied.
        skipped = [ln for ln in result.stderr.splitlines() if ln.startswith("Skipped patch")]
        if skipped:
            raise SystemExit(
                f"{patch.name} was skipped rather than applied to {component}:\n"
                + "\n".join(f"  {ln}" for ln in skipped)
                + "\nA plain unified diff (--- a/path, no `diff --git` header) is not filtered this way."
            )
        applied.append(patch.name)
        print(f"      patch: {patch.name}")
    return applied


def copy_extras(source: Path) -> list[str]:
    copied = []
    for relative, target in EXTRA_FILES:
        src = source / relative
        if not src.is_file():
            print(f"  ! {relative} not found under {source} — the cost that needs it will fail at plan time")
            continue
        dst = VENDOR / target
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(target)
        print(f"  checkpoint: {target}  ({src.stat().st_size / 1e6:.1f} MB)")
    return copied


def read_manifest() -> dict:
    """The manifest as it stands, so a partial re-vendor keeps the other components' entries."""
    path = VENDOR / "VENDOR.toml"
    if not path.is_file():
        return {}
    try:
        import tomlkit

        return dict(tomlkit.parse(path.read_text()))
    except Exception:
        return {}


def write_manifest(entries: dict, extras: list[str]) -> None:
    # Re-vendoring one component must not erase the provenance of the others. Losing that is
    # not cosmetic: two of the three trees are under a licence that governs redistribution,
    # and the commit is the only record of which planner a build actually contains.
    existing = read_manifest()
    merged = {name: meta for name, meta in existing.items() if name in TRIM}
    merged.update(entries)
    entries = {name: merged[name] for name in TRIM if name in merged}
    if not extras:
        extras = list((existing.get("checkpoints") or {}).get("files") or [])

    lines = [
        "# Provenance for the vendored planner sources.",
        "#",
        "# Regenerate with:  python tools/vendor.py --source /path/to/hitl-tamp-vla",
        "#",
        "# cuRobo and cuTAMP are under the NVIDIA License, which permits redistribution under",
        "# the same terms and restricts use to research or evaluation. Their LICENSE files are",
        "# kept intact inside each tree; see NOTICE at the repository root.",
        "",
    ]
    for name, meta in entries.items():
        lines.append(f"[{name}]")
        lines.append(f'url = "{meta["url"]}"')
        lines.append(f'commit = "{meta["commit"]}"')
        lines.append(f'version = "{meta["version"]}"')
        lines.append(f'ref = "{meta.get("ref", "")}"')
        lines.append(f"bytes = {meta['bytes']}")
        trimmed = ", ".join(f'"{t}"' for t in meta["trimmed"])
        lines.append(f"trimmed = [{trimmed}]")
        patches = ", ".join(f'"{p}"' for p in meta.get("patches") or [])
        lines.append(f"patches = [{patches}]")
        lines.append("")
    if extras:
        lines.append("[checkpoints]")
        lines.append("files = [" + ", ".join(f'"{e}"' for e in extras) + "]")
        lines.append("")
    (VENDOR / "VENDOR.toml").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path, help="Path to a hitl-tamp-vla checkout")
    parser.add_argument("--component", action="append", choices=list(TRIM), help="Only re-vendor these")
    parser.add_argument(
        "--ref", action="append", default=[], metavar="COMPONENT=REF",
        help=f"Override the ref for one component. Defaults: {REFS}",
    )
    args = parser.parse_args()

    refs = dict(REFS)
    for override in args.ref:
        if "=" not in override:
            raise SystemExit(f"--ref wants COMPONENT=REF, got {override!r}")
        component, _, ref = override.partition("=")
        if component not in TRIM:
            raise SystemExit(f"unknown component {component!r}; one of {sorted(TRIM)}")
        refs[component] = ref

    source = args.source.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"{source} is not a directory")

    VENDOR.mkdir(parents=True, exist_ok=True)
    components = args.component or list(TRIM)

    print(f"Vendoring from {source}")
    entries = {}
    for component in components:
        entries[component] = export(source, component, VENDOR / component, refs[component])

    extras = copy_extras(source)
    write_manifest(entries, extras)

    total = sum(f.stat().st_size for f in VENDOR.rglob("*") if f.is_file())
    print(f"\nVendor tree: {total / 1e6:.1f} MB  ->  {VENDOR}")
    if total > 90e6:
        print("! That is large for a wheel. Check the trim list in tools/vendor.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
