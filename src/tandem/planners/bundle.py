"""An offline bundle of a planner's pinned sources: `tandem planners bundle NAME --out DIR`.

tandem fetches a planner's sources when its runtime is installed. A machine that cannot reach GitHub
installs them from a directory instead -- ``tandem planners install NAME --sources DIR``, or
``$TANDEM_PLANNER_SOURCES=DIR`` -- and this makes that directory, on a machine that can::

    DIR/
        tiptop/     exactly the pinned commit, exported with `git archive`, trimmed
        cuTAMP/
        curobo/

Each export carries a ``.tandem-source.json`` marker naming its commit and a digest of its files, and
an install checks both: the commit against its own pins, so a bundle made for one version of tandem is
refused by another instead of quietly building the wrong planner, and the files against the digest,
so a bundle edited or damaged since is not recorded as that commit. That is why this lives in the package rather than only
in the repository: a bundle has to be made by the SAME tandem that installs from it, and a pip or
pipx install has no repository to run a script from. Patches are not applied here: the install
applies them, the same way whether the tree came from the network or from a bundle.

What a bundle does NOT carry: the planner's environment. `pixi install` still solves it from
conda-forge and PyPI (and TiPToP's builds SAM-2 from GitHub), and TiPToP's first warm-up downloads
the SAM-2 checkpoint. A bundle spares the source fetch, not the network.
"""

from __future__ import annotations

import json
import shutil
import tarfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from tandem.core.errors import TandemError


def recipe_of(planner: str):
    """The ``RuntimeRecipe`` behind ``planner``'s runtime, or a TandemError saying why there is none."""
    from tandem.planners import registry
    from tandem.planners.runtime import RecipeRuntime

    runtime = registry.runtime(planner)
    if runtime is None:
        raise TandemError(f"The {planner} planner is pure Python: it has no sources to bundle.")
    if not isinstance(runtime, RecipeRuntime):
        raise TandemError(
            f"The {planner} planner's runtime is not built from a recipe, so tandem cannot say what to bundle.",
            hint="A planner declares its sources with a RuntimeRecipe (docs/ADDING_A_PLANNER.md).",
        )
    return runtime.recipe


def bundle_sources(
    planner: str,
    out: Path,
    *,
    only: Iterable[str] = (),
    local: Mapping[str, Path] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, dict]:
    """Write ``planner``'s pinned sources into ``out``, one marked export per source. Returns the markers.

    ``only`` limits it to those sources. ``local`` maps a source to a git checkout to export it from
    instead of fetching it -- a checkout is an object store here too: the pinned commit is exported out
    of it, whatever its working tree holds.
    """
    from tandem.planners.runtime import SOURCE_MARKER, export_from_tree, export_pinned, tree_digest, trim

    recipe = recipe_of(planner)
    only = list(only)
    local = dict(local or {})
    names = [s.name for s in recipe.sources]
    unknown = sorted(set(only) - set(names)) + sorted(set(local) - set(names))
    if unknown:
        raise TandemError(
            f"The {planner} planner has no source named {', '.join(unknown)}.",
            hint=f"Its sources are {', '.join(names)}.",
        )

    out = Path(out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    scratch = out / ".bundle-scratch"
    written: dict[str, dict] = {}
    try:
        for source in recipe.sources:
            if only and source.name not in only:
                continue
            dest = out / source.name
            staged = scratch / source.name
            shutil.rmtree(staged, ignore_errors=True)
            if source.name in local:
                checkout = Path(local[source.name]).expanduser().resolve()
                origin = export_from_tree(source.pin, checkout, staged, scratch=scratch, log=log)
                if not origin.get("verified"):
                    raise TandemError(
                        f"{checkout} is not a git checkout, so there is no telling which commit it is.",
                        hint="A bundle has to be of a verified commit: point --from at a git checkout.",
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


def missing_sources(planner: str, out: Path) -> list[str]:
    """The sources of ``planner`` that ``out`` does not hold: what an install from it would still lack."""
    recipe = recipe_of(planner)
    return [source.name for source in recipe.sources if not (Path(out) / source.name).is_dir()]


def archive(out: Path, names: Iterable[str]) -> Path:
    """``out`` as ``<out>.tar.gz``, holding the named exports, for carrying it over."""
    out = Path(out)
    path = out.with_name(out.name + ".tar.gz")
    with tarfile.open(path, "w:gz") as tar:
        for name in names:
            tar.add(out / name, arcname=name)
    return path
