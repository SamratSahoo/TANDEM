"""How TiPToP is built for a session, installed on a machine, and described in a catalog.

Everything TiPToP-specific about STARTING a session lives here, and nowhere in tandem's core: which
runtime it runs in, the ``TIPTOP_*`` environment its sidecar reads, the ``tiptop.yml`` and cuRobo
cost overrides rendered from the profile, and the checks that catch a broken asset before a
twenty-second warm-up does. The session only hands over a ``BackendContext`` and gets a backend back.

This module is imported to list planners and to read TiPToP's capabilities, both of which happen on a
laptop. So it imports the declaration and the protocol types and nothing else at module level; the
runtime, the renderer and the backend itself are imported inside the calls that need them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tandem.core.errors import TandemError
from tandem.planners.base import (
    BackendContext,
    Capabilities,
    PlannerInfo,
    RuntimeStatus,
    SourcePin,
)
from tandem.planners.tiptop.capabilities import CAPABILITIES

# What an install builds the runtime from. Today that is the tree vendored into the wheel, and these
# are the commits _vendor/VENDOR.toml records for it -- a test holds the two together, because a
# catalog that names a commit the install does not deliver is a false statement about every dataset
# collected with it. When the runtime is fetched instead of copied, these become the pins it fetches.
SOURCES: tuple[SourcePin, ...] = (
    SourcePin("tiptop", "https://github.com/SamratSahoo/tiptop.git", "4db8f92671b431de4e5a456523dc84b5246401ee"),
    SourcePin("cuTAMP", "https://github.com/SamratSahoo/cuTAMP.git", "7b0aeaea452f13a4ee73d95f2aacbb3af720ad0f"),
    SourcePin("curobo", "https://github.com/SamratSahoo/curobo.git", "3a90ff49eee169d9636b2a679d98457a2592fb52"),
)

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
        from tandem.core.runtime import Runtime

        return TiptopRuntime(Runtime(_settings(settings).resolved_runtime_dir()))

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


class TiptopRuntime:
    """TiPToP's runtime as a catalog sees it: status, install, uninstall.

    A thin adapter over ``tandem.core.runtime.Runtime``, which still does the work -- copy the vendored
    sources, solve the pixi environment, compile cuRobo's kernels. Kept thin on purpose: fetching
    pinned commits instead of copying a vendored tree replaces what is behind these three methods,
    and nothing that calls them should have to change when it does.
    """

    def __init__(self, runtime) -> None:
        self._runtime = runtime

    @property
    def root(self) -> Path:
        return self._runtime.root

    def status(self) -> RuntimeStatus:
        st = self._runtime.status()
        detail = []
        if st.exists:
            detail.append("sources present" if st.sources_present else "sources missing")
            detail.append("pixi env built" if st.env_built else "pixi env not built")
            detail.append("cuRobo kernels compiled" if st.kernels_built else "cuRobo kernels not compiled")
            if st.built_at:
                detail.append(f"built {st.built_at}")
        else:
            detail.append("not created")
        return RuntimeStatus(
            installed=bool(st.ready),
            path=str(self._runtime.root),
            pins=_pins(st.vendor),
            detail=" · ".join(detail),
            problems=tuple(st.problems or ()),
        )

    def install(self, *, on_progress=None, sources_dir: Path | None = None, force: bool = False) -> None:
        """Copy the sources in, solve the environment, compile the kernels. 5-20 minutes the first time.

        Every step skips what is already done, so an interrupted install resumes. ``sources_dir`` is
        a directory holding ``tiptop/``, ``cuTAMP/`` and ``curobo/`` -- the vendored tree by default.
        """
        from tandem.core import paths

        say = on_progress or (lambda _line: None)
        source_root = Path(sources_dir) if sources_dir is not None else paths.vendor_dir()
        say(f"sources: {source_root}")
        self._runtime.materialize(source_root, force=force, log=say)
        say("pixi env: solving")
        self._runtime.build_env(log=say)
        say("planners: compiling cuRobo's CUDA kernels and installing cuTAMP")
        self._runtime.build_planners(log=say)
        st = self._runtime.status()
        if not st.ready:
            raise TandemError(
                "The build finished but the runtime still looks incomplete: " + "; ".join(st.problems or []),
                hint="Run the install again; every finished step is skipped.",
            )

    def uninstall(self) -> None:
        import shutil

        root = self._runtime.root
        if not root.exists():
            return
        if not _looks_like_a_runtime(root):
            # The runtime directory is a setting. Pointed at the wrong place by mistake, an
            # unconditional rmtree would delete whatever is there.
            raise TandemError(
                f"{root} does not look like a TiPToP runtime, so it was not deleted.",
                hint="Check runtime_dir in `tandem config`, and delete it by hand if it really is one.",
            )
        shutil.rmtree(root)


def _looks_like_a_runtime(root: Path) -> bool:
    from tandem.core import runtime as runtime_mod

    if not any(root.iterdir()):
        return True
    markers = (runtime_mod.STAMP_FILE, runtime_mod.TIPTOP, runtime_mod.CUTAMP, runtime_mod.CUROBO)
    return any((root / marker).exists() for marker in markers)


def _pins(vendor: dict | None) -> tuple[SourcePin, ...]:
    """The sources a runtime's stamp says it was built from, in the order it records them."""
    pins = []
    for name, meta in (vendor or {}).items():
        if isinstance(meta, dict) and meta.get("commit"):
            pins.append(SourcePin(str(name), str(meta.get("url") or ""), str(meta["commit"])))
    return tuple(pins)


def _settings(settings: Any):
    if settings is not None:
        return settings
    from tandem.core import settings as settings_mod

    return settings_mod.load()


FACTORY = TiptopFactory()
