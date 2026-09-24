#!/usr/bin/env python3
"""Make an offline bundle of a planner's pinned sources, for a workstation with no network.

tandem fetches a planner's sources when its runtime is installed. A machine that cannot reach GitHub
installs from a directory instead -- ``tandem planners install NAME --sources DIR``, or
``$TANDEM_PLANNER_SOURCES=DIR`` -- and this makes that directory, on a machine that can:

    DIR/
        tiptop/     exactly the pinned commit, exported with `git archive`, trimmed
        cuTAMP/
        curobo/

Each export carries a ``.tandem-source.json`` marker naming its commit and a digest of its files, and
an install checks both: the commit against its own pins, so a bundle made for one version of tandem is
refused by another instead of quietly building the wrong planner, and the files against the digest,
so a bundle edited or damaged since is not recorded as that commit. Patches are NOT applied here: the
install applies them, the same way whether the tree came from the network or from a bundle.

The pins, the trims and the fetching are the recipe's own (tandem/planners/runtime.py and the
planner's recipe), so a bundle is byte-for-byte what an online install would fetch.

Usage:
    python tools/bundle.py --planner tiptop --out /media/usb/planner-sources
    python tools/bundle.py --planner tiptop --out DIR --only tiptop --only cuTAMP
    python tools/bundle.py --planner tiptop --out DIR --from tiptop=~/src/tiptop   # a local checkout
    python tools/bundle.py --planner tiptop --out DIR --archive                    # also DIR.tar.gz
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tandem.core.errors import TandemError  # noqa: E402
from tandem.planners import registry  # noqa: E402
from tandem.planners.runtime import (  # noqa: E402
    SOURCE_MARKER,
    RecipeRuntime,
    export_from_tree,
    export_pinned,
    tree_digest,
    trim,
)


def recipe_of(planner: str):
    runtime = registry.runtime(planner)
    if runtime is None:
        raise SystemExit(f"{planner} is pure Python; it has no sources to bundle")
    if not isinstance(runtime, RecipeRuntime):
        raise SystemExit(
            f"{planner}'s runtime is not built from a recipe, so there is nothing to bundle from"
        )
    return runtime.recipe


def bundle(planner: str, out: Path, *, only: list[str], local: dict[str, Path], log=print) -> dict:
    recipe = recipe_of(planner)
    names = [s.name for s in recipe.sources]
    unknown = sorted(set(only) - set(names)) + sorted(set(local) - set(names))
    if unknown:
        raise SystemExit(f"{planner} has no source named {', '.join(unknown)}; it has {', '.join(names)}")

    out.mkdir(parents=True, exist_ok=True)
    scratch = out / ".bundle-scratch"
    written = {}
    try:
        for source in recipe.sources:
            if only and source.name not in only:
                continue
            dest = out / source.name
            staged = scratch / source.name
            shutil.rmtree(staged, ignore_errors=True)
            if source.name in local:
                # A checkout is an object store here too: the pinned commit is exported out of it,
                # whatever its working tree holds.
                checkout = local[source.name].expanduser().resolve()
                origin = export_from_tree(source.pin, checkout, staged, scratch=scratch, log=log)
                if not origin.get("verified"):
                    raise SystemExit(
                        f"{checkout} is not a git checkout; a bundle has to be of a verified commit"
                    )
            else:
                origin = export_pinned(source.pin, staged, scratch=scratch, log=log)
            dropped = trim(staged, source.trim, name=source.name, log=log)
            marker = {
                "name": source.name,
                "url": source.pin.url,
                "commit": source.pin.commit,
                "planner": planner,
                "trimmed": dropped,
                "fetched_from": origin.get("origin"),
                # Of the files as they leave here, trimmed. The install recomputes it, so the commit
                # above is recorded as verified only for the files that were really exported from it.
                "sha256": tree_digest(staged),
            }
            (staged / SOURCE_MARKER).write_text(json.dumps(marker, indent=2) + "\n")
            shutil.rmtree(dest, ignore_errors=True)
            staged.rename(dest)
            size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file() and not f.is_symlink())
            log(f"{source.name}: {source.pin.short()} -> {dest}  ({size / 1e6:.1f} MB)")
            written[source.name] = marker
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--planner", default="tiptop", help="Whose sources to bundle (default: tiptop)")
    parser.add_argument("--out", required=True, type=Path, help="The directory to write the bundle into")
    parser.add_argument("--only", action="append", default=[], metavar="SOURCE", help="Only these sources")
    parser.add_argument(
        "--from",
        dest="local",
        action="append",
        default=[],
        metavar="SOURCE=PATH",
        help="Export this source from a local checkout instead of fetching it",
    )
    parser.add_argument("--archive", action="store_true", help="Also write OUT.tar.gz, for carrying it over")
    args = parser.parse_args()

    local = {}
    for item in args.local:
        name, sep, path = item.partition("=")
        if not sep or not path:
            raise SystemExit(f"--from wants SOURCE=PATH, got {item!r}")
        local[name] = Path(path)

    out = args.out.expanduser().resolve()
    try:
        written = bundle(args.planner, out, only=args.only, local=local)
    except TandemError as exc:
        raise SystemExit(f"{exc.message}\n{exc.hint or ''}".strip()) from None

    if args.archive:
        archive = out.with_name(out.name + ".tar.gz")
        with tarfile.open(archive, "w:gz") as tar:
            for name in written:
                tar.add(out / name, arcname=name)
        print(f"archive: {archive}  ({archive.stat().st_size / 1e6:.1f} MB)")
    print(f"\nInstall from it with:  tandem planners install {args.planner} --sources {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
