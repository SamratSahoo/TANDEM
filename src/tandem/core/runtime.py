"""TiPToP's GPU runtime, under the name and the interface the rest of tandem has always used for it.

tandem itself is pure Python and installs with one pip command. Everything heavy lives in a runtime
instead: torch, the compiled cuRobo CUDA kernels, cuTAMP, tiptop, SAM-2, open3d and the ZED bindings.
That split is what lets a plain install plus `tandem ui` work on a laptop.

How a runtime is built is no longer this module's business. TiPToP declares its runtime as a recipe
(``tandem/planners/tiptop/recipe.py``: pinned commits, patches, checkpoints, a pixi environment and a
build step), and ``tandem/planners/runtime.py`` fetches, patches and builds any recipe, TiPToP's
included. This is a thin facade over that, kept so the TiPToP-specific callers -- the backend that
spawns its sidecar, ``tandem traj open``, the session's merge -- keep the attributes they reach for
(``tiptop_dir``, ``command``, ``python``, a status with ``kernels_built``) while they move to the
planner-agnostic path.

A new caller should not start here. ``tandem.planners.registry.runtime(name, settings)`` gives the
runtime of whichever planner a profile names.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tandem.planners import runtime as recipe_runtime
from tandem.planners.tiptop.factory import TiptopRuntime
from tandem.planners.tiptop.recipe import RECIPE

# Directory names inside the runtime, as the recipe names its sources.
TIPTOP = "tiptop"
CUTAMP = "cuTAMP"
CUROBO = "curobo"

STAMP_FILE = recipe_runtime.MANIFEST_FILE


@dataclass
class RuntimeStatus:
    exists: bool = False
    sources_present: bool = False
    env_built: bool = False
    kernels_built: bool = False
    ready: bool = False
    # What each source tree was installed from: {name: {url, commit, version, origin, ...}}.
    vendor: dict | None = None
    built_at: str | None = None
    problems: list[str] | None = None

    def to_dict(self) -> dict:
        return {
            "exists": self.exists,
            "sources_present": self.sources_present,
            "env_built": self.env_built,
            "kernels_built": self.kernels_built,
            "ready": self.ready,
            "vendor": self.vendor,
            "built_at": self.built_at,
            "problems": self.problems or [],
        }


class Runtime:
    """Locate, inspect, build and invoke TiPToP's runtime."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser()
        self.recipe_runtime = TiptopRuntime(self.root)

    # ---- layout ------------------------------------------------------------

    @property
    def tiptop_dir(self) -> Path:
        return self.root / TIPTOP

    @property
    def cutamp_dir(self) -> Path:
        return self.root / CUTAMP

    @property
    def curobo_dir(self) -> Path:
        return self.root / CUROBO

    @property
    def stamp_file(self) -> Path:
        return self.root / STAMP_FILE

    @property
    def pixi_env(self) -> Path:
        # Through the tree's .pixi link, which is the path the environment was built at.
        return self.tiptop_dir / ".pixi" / "envs" / "default"

    # ---- status ------------------------------------------------------------

    def status(self) -> RuntimeStatus:
        st = self.recipe_runtime.inspect()
        if not st.exists:
            return RuntimeStatus(problems=list(st.problems))
        return RuntimeStatus(
            exists=True,
            sources_present=st.sources_present,
            env_built=st.environment_built,
            kernels_built=dict(st.steps).get("planners", False),
            ready=st.ready,
            vendor=_installed(self.recipe_runtime) or None,
            built_at=st.built_at,
            problems=list(st.problems),
        )

    def pending_vendor(self) -> dict:
        """What a build WOULD install: the recipe's pins, before anything is fetched.

        Shown before anything is built so `tandem runtime status` can answer "which cuRobo am I
        about to compile?" without first spending twenty minutes compiling it.
        """
        return {
            source.name: {
                "url": source.pin.url,
                "commit": source.pin.commit,
                "version": source.pin.short(),
                "trimmed": list(source.trim),
                "patches": [Path(p).name for p in source.patches],
            }
            for source in RECIPE.sources
        }

    def is_ready(self) -> bool:
        return self.status().ready

    def require_ready(self) -> None:
        self.recipe_runtime.require_ready()

    # ---- invocation --------------------------------------------------------

    def command(self, args: list[str]) -> list[str]:
        """Wrap a console-script invocation in `pixi run`, from the tiptop manifest dir."""
        return self.recipe_runtime.command(args)

    def python(self) -> Path:
        return self.recipe_runtime.python()

    # ---- build -------------------------------------------------------------

    def materialize(self, sources_dir: Path | None = None, *, force: bool = False, log=None) -> None:
        """Put the pinned sources and the checkpoints in place: fetched, or from ``sources_dir``."""
        self.recipe_runtime.fetch(sources_dir=sources_dir, force=force, log=log)
        self.recipe_runtime.place_assets(log=log)

    def build_env(self, *, log=None, extra_env: dict | None = None) -> None:
        """`pixi install` — solve and materialise the conda environment."""
        self.recipe_runtime.build_environment(log=log, extra_env=extra_env)

    def build_planners(self, *, log=None, extra_env: dict | None = None) -> None:
        """`pixi run setup-planners` — build cuRobo's CUDA kernels, then install cuTAMP."""
        for step in RECIPE.steps:
            self.recipe_runtime.run_step(step, log=log, extra_env=extra_env)
        self.recipe_runtime.record_built()


def _installed(rt: recipe_runtime.RecipeRuntime) -> dict:
    """The record of each installed tree, in the shape the old vendor manifest had (plus where from)."""
    record = rt.record()
    out = {}
    for name, entry in record["sources"].items():
        if not isinstance(entry, dict) or not entry.get("commit"):
            continue
        out[name] = {
            "url": entry.get("url") or "",
            "commit": entry["commit"],
            "version": str(entry["commit"])[:7],
            "origin": entry.get("origin"),
            "verified": entry.get("verified"),
            "trimmed": list(entry.get("trimmed") or []),
            "patches": [p.get("name") if isinstance(p, dict) else str(p) for p in entry.get("patches") or []],
        }
    return out


def default() -> Runtime:
    from tandem.core import settings as settings_mod

    return Runtime(settings_mod.load().resolved_runtime_dir())
