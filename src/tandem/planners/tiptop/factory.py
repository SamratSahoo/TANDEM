"""How TiPToP is built for a session, installed on a machine, and described in a catalog.

Everything TiPToP-specific about STARTING a session lives here, and nowhere in tandem's core: which
runtime it runs in, the ``TIPTOP_*`` environment its sidecar reads, the ``tiptop.yml`` and cuRobo
cost overrides rendered from the profile, and the checks that catch a broken asset before a
twenty-second warm-up does. The session only hands over a ``BackendContext`` and gets a backend back.

This module is imported to list planners and to read TiPToP's capabilities, both of which happen on a
laptop. So it imports the declarations -- capabilities, runtime recipe -- and the protocol types and
nothing else at module level; the renderer and the backend itself are imported inside the calls that
need them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tandem.core.errors import TandemError
from tandem.planners.base import (
    BackendContext,
    Capabilities,
    PlannerInfo,
    SourcePin,
)
from tandem.planners.runtime import RecipeRuntime
from tandem.planners.tiptop.capabilities import CAPABILITIES
from tandem.planners.tiptop.recipe import RECIPE

# What an install builds the runtime from: the commits the recipe fetches. Read from the recipe rather
# than restated, because a catalog that names a commit the install does not deliver is a false
# statement about every dataset collected with it.
SOURCES: tuple[SourcePin, ...] = RECIPE.pins

INFO = PlannerInfo(
    name="tiptop",
    display_name="TiPToP",
    summary=(
        "GPU task and motion planning with cuTAMP and cuRobo, perceiving with Gemini, SAM-2 and M2T2 "
        "grasps, and executing pick-and-place on a real arm."
    ),
    homepage="https://github.com/SamratSahoo/tiptop",
    requires=(
        "Linux with an NVIDIA GPU, CUDA 12 or newer and a recent driver",
        "pixi, and about 25 GB of free disk for the runtime",
        "a Franka FR3 (or UR5) with a Robotiq 2F-85, over the bamboo-polymetis shim",
        "2-3 ZED cameras and the ZED SDK",
        "an M2T2 grasp server",
        "a Gemini API key",
    ),
    sources=SOURCES,
)


class TiptopFactory:
    """The registry's handle on TiPToP. Stateless: one instance serves every session."""

    info = INFO

    def capabilities(self) -> Capabilities:
        return CAPABILITIES

    def runtime(self, settings: Any = None) -> TiptopRuntime:
        return TiptopRuntime(_settings(settings).resolved_runtime_dir())

    def create(self, ctx: BackendContext):
        """A TiptopBackend for this session, with its config rendered and its assets checked.

        What ``Session._build_backend`` and part of ``Session.start`` used to do inline, moved here
        unchanged so the session no longer knows any of it is TiPToP's.
        """
        from tandem.core import render
        from tandem.core.runtime import Runtime
        from tandem.planners.tiptop.backend import TiptopBackend

        if ctx.options:
            # Nothing reads them yet, and an option that does nothing is a setting the operator
            # believes is in force. Refused rather than ignored.
            raise TandemError(
                "The tiptop planner takes no planner.options, but the profile sets "
                + ", ".join(sorted(map(str, ctx.options)))
                + ".",
                hint="TiPToP is configured by the profile's robot, cameras, perception and tamp blocks. "
                "Remove planner.options.",
            )

        # The runtime the caller already resolved wins, so a session and its planner can never be
        # looking at two different runtimes.
        root = ctx.runtime_dir if ctx.runtime_dir is not None else _settings(ctx.settings).resolved_runtime_dir()
        runtime = Runtime(Path(root))
        profile = ctx.profile

        # Problems that would otherwise surface minutes into a warmed session: a checkpoint the VAE
        # cost loads lazily, a blending key that does nothing. Missing extrinsics are fatal, because
        # tiptop raises for them at warm-up with the arm already moving to its capture pose.
        problems = render.check_assets(profile, runtime_dir=runtime.root)
        fatal = [p for p in problems if p.startswith("no camera extrinsics")]
        if fatal:
            raise TandemError(
                "\n".join(fatal),
                hint="Extrinsics are keyed by camera serial; add them before collecting.",
            )
        for problem in problems:
            ctx.log(f"warning: {problem}")

        # tiptop.yml (per profile, where $TIPTOP_CONFIG points) and the cuRobo cost overrides (in the
        # session's own directory). Keyed by the session's id, which is what names ctx.session_dir.
        files = render.prepare_session_files(
            profile, ctx.session_id or Path(ctx.session_dir).name, runtime_dir=runtime.root
        )
        env = render.render_env(
            profile,
            events_file=ctx.events_file if ctx.events_file is not None else files["events_file"],
            task=ctx.task or None,
            runtime_dir=runtime.root,
        )
        return TiptopBackend(
            runtime,
            env=env,
            output_dir=ctx.output_dir,
            execute=ctx.execute,
            record=ctx.record,
            cost_overrides_file=files.get("overrides_file"),
            on_log=ctx.on_log,
        )


class TiptopRuntime(RecipeRuntime):
    """TiPToP's runtime: the generic recipe runtime, bound to TiPToP's recipe.

    Its root is the ``runtime_dir`` setting rather than a directory of its own under the runtimes
    root, because TiPToP's runtime lived there before tandem drove more than one planner, and a built
    pixi environment cannot be moved: its own absolute path is baked into it.
    """

    def __init__(self, root: Path) -> None:
        super().__init__(RECIPE, root)


def _settings(settings: Any):
    if settings is not None:
        return settings
    from tandem.core import settings as settings_mod

    return settings_mod.load()


FACTORY = TiptopFactory()
