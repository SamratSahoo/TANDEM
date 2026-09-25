#!/usr/bin/env python3
"""Make an offline bundle of a planner's pinned sources, from a checkout of this repository.

The bundler itself is part of tandem -- ``tandem planners bundle NAME --out DIR`` (tandem/planners/
bundle.py) -- so a pip or pipx install can make a bundle that matches its own pins. This script is the
same thing for a checkout without an install (CI runs it): it puts the checkout's ``src`` first on the
path, so the bundle matches THIS checkout's pins.

A bundle carries the planner's sources only. The environment is still solved from conda-forge and PyPI
at install, and TiPToP's SAM-2 comes from GitHub (see tandem/planners/bundle.py).

Usage:
    python tools/bundle.py --planner tiptop --out /media/usb/planner-sources
    python tools/bundle.py --planner tiptop --out DIR --only tiptop --only cuTAMP
    python tools/bundle.py --planner tiptop --out DIR --from tiptop=~/src/tiptop   # a local checkout
    python tools/bundle.py --planner tiptop --out DIR --archive                    # also DIR.tar.gz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tandem.core.errors import TandemError  # noqa: E402
from tandem.planners.bundle import archive, bundle_sources, missing_sources, recipe_of  # noqa: E402,F401


def bundle(planner: str, out: Path, *, only: list[str], local: dict[str, Path], log=print) -> dict:
    """Kept for scripts written against this file: ``tandem.planners.bundle.bundle_sources``."""
    return bundle_sources(planner, out, only=only, local=local, log=log)


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
        written = bundle_sources(args.planner, out, only=args.only, local=local)
        missing = missing_sources(args.planner, out)
    except TandemError as exc:
        raise SystemExit(f"{exc.message}\n{exc.hint or ''}".strip()) from None

    if args.archive:
        path = archive(out, written)
        print(f"archive: {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    if missing:
        print(f"\n{out} does not hold {', '.join(missing)} yet: an install from it needs every source.")
    else:
        print(f"\nInstall from it with:  tandem planners install {args.planner} --sources {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
