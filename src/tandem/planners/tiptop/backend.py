"""TiPToP as a tandem backend: the parent-process half.

A ``SidecarPlanner``: every protocol call becomes a request on a child that runs inside the GPU
runtime, and everything generic about that -- launching, the wire, timeouts, crashes, custody -- is
the base class's. What is TiPToP's is only how it is launched and warmed: through the runtime's
``pixi run`` from the tiptop tree, with the ``TIPTOP_*`` environment the factory rendered, and
handed the cuRobo cost overrides when it warms.

The child is ``sidecar.py``, which lives in this package -- it is TANDEM's code executed in TiPToP's
environment, not TiPToP's code. That is the whole trick, and it is why the planner repositories stay
untouched: everything tandem needs from TiPToP is a call to a public function of it, made from a file
tandem ships.

Nothing in this module imports torch, cuTAMP or tiptop, and nothing ever will. It is on the path of
``pip install tandem-tamp`` on a laptop.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tandem.planners.base import BackendContext
from tandem.planners.sidecar import SidecarPlanner
from tandem.planners.tiptop.capabilities import CAPABILITIES
from tandem.planners.tiptop.factory import INFO, TiptopFactory
from tandem.planners.tiptop.recipe import RECIPE
from tandem.planners.tiptop.runtime import TiptopRuntime

_log = logging.getLogger(__name__)

# Warming builds cuRobo's CUDA kernels' solvers, loads SAM-2 and opens the cameras.
WARM_TIMEOUT = 900.0
# Perception is a detection call, a depth/grasp call, and scene geometry.
PERCEIVE_TIMEOUT = 300.0
# Planning is bounded by max_planning_time, but the symbolic search in front of it is not.
PLAN_TIMEOUT = 900.0
# Execution is a whole trajectory on a real arm, plus writing the video out.
EXECUTE_TIMEOUT = 1800.0
# Releasing two cameras measures ~14s of SDK teardown, plus waiting for the arm to stop moving.
HARDWARE_TIMEOUT = 180.0


def sidecar_path() -> Path:
    """The sidecar script, as a path the runtime's interpreter can be pointed at.

    Passed by path rather than imported as a module: the runtime environment has tiptop and cuTAMP on
    its path but not tandem, and adding tandem to it would mean installing tandem's own dependencies
    into the planner's environment for no reason. The sidecar is written to need nothing but the
    standard library, ``tandem_sidecar`` (which the launch puts on its path) and what tiptop has.
    """
    return TiptopBackend.sidecar_script()


class TiptopBackend(SidecarPlanner):
    """Drives TiPToP for one collection session.

    ``execute`` ignores ``should_stop``: upstream's execute_cutamp_plan takes a whole trajectory and
    has no mid-plan seam, which is exactly what ``supports_cooperative_stop=False`` says, and what
    keeps the base class from offering the sidecar a stop file. ``movables`` and ``return_home`` go
    on the wire because the capabilities declare both; the sidecar honours them (see ``Sidecar.plan``).
    """

    info = INFO
    CAPABILITIES = CAPABILITIES
    OPTIONS = TiptopFactory.OPTIONS
    recipe = RECIPE
    name = "tiptop"
    SIDECAR = "sidecar.py"
    TIMEOUTS = {
        "warm": WARM_TIMEOUT,
        "perceive": PERCEIVE_TIMEOUT,
        "plan": PLAN_TIMEOUT,
        "execute": EXECUTE_TIMEOUT,
        "capture_frame": HARDWARE_TIMEOUT,
        "home": HARDWARE_TIMEOUT,
        "release_hardware": HARDWARE_TIMEOUT,
        "reacquire_hardware": HARDWARE_TIMEOUT,
    }

    def __init__(
        self,
        runtime: TiptopRuntime,
        *,
        env: dict[str, str],
        output_dir: Path,
        execute: bool = True,
        record: bool = True,
        cost_overrides_file: Path | None = None,
        on_log: Callable[[str, str], None] | None = None,
        perception_urls: dict[str, str] | None = None,
        settings: Any = None,
    ) -> None:
        # Built by the factory, which has already located the runtime and rendered the environment
        # from the profile; see factory.TiptopFactory.create.
        super().__init__(None, runtime=runtime, env=env, on_log=on_log)
        self._output_dir = Path(output_dir)
        self._execute = execute
        self._record = record
        self._cost_overrides_file = cost_overrides_file
        # The M2T2 and FoundationStereo servers perception calls: started before warming and before each
        # perception pass if they are down, and stopped at close if this session started them (servers.py).
        self._servers = None
        if perception_urls:
            from tandem.planners.tiptop.servers import ServerManager

            self._servers = ServerManager(
                perception_urls, log=lambda text: self.log(text), settings=settings
            )

    # ---- the class as a factory: TiPToP's lives in factory.py, so both routes build the same thing ----

    @classmethod
    def create(cls, ctx: BackendContext) -> TiptopBackend:
        from tandem.planners.tiptop.factory import FACTORY

        return FACTORY.create(ctx)

    @classmethod
    def runtime(cls, settings: Any = None):
        from tandem.planners.tiptop.factory import FACTORY

        return FACTORY.runtime(settings)

    @classmethod
    def validate_options(cls, options):
        from tandem.planners.tiptop.factory import FACTORY

        return FACTORY.validate_options(options)

    @classmethod
    def describe_options(cls, profile, *, settings=None):
        from tandem.planners.tiptop.factory import FACTORY

        return FACTORY.describe_options(profile, settings=settings)

    @classmethod
    def doctor_checks(cls, profile, *, settings=None, probe_hardware=True):
        from tandem.planners.tiptop.factory import FACTORY

        return FACTORY.doctor_checks(profile, settings=settings, probe_hardware=probe_hardware)

    @classmethod
    def services(cls, settings=None):
        from tandem.planners.tiptop.factory import FACTORY

        return FACTORY.services(settings)

    @classmethod
    def replay(cls, rollout_dir, *, settings=None):
        from tandem.planners.tiptop.factory import FACTORY

        FACTORY.replay(rollout_dir, settings=settings)

    # ---- how TiPToP's sidecar is started -----------------------------------

    def launch_cwd(self) -> Path | None:
        # The tiptop tree, where its pixi manifest is and where tiptop resolves its relative paths.
        return self._runtime.tiptop_dir

    def warm(self) -> None:
        # The sidecar's warm-up checks both servers, so they come up first.
        if self._servers is not None:
            self._servers.ensure()
        super().warm()

    def perceive(
        self,
        *,
        task_hint: str,
        save_dir: Path,
        reset_arm: bool = True,
        open_gripper: bool = False,
    ):
        # One that died since (out of GPU memory, say) is started again before the pass that needs it.
        if self._servers is not None:
            self._servers.ensure()
        return super().perceive(
            task_hint=task_hint, save_dir=save_dir, reset_arm=reset_arm, open_gripper=open_gripper
        )

    def close(self) -> None:
        try:
            super().close()
        finally:
            if self._servers is not None:
                self._servers.stop_started()

    def warm_args(self) -> dict[str, Any]:
        # The cost overrides are the same file an ordinary tiptop-run reads its knobs from.
        return {
            "output_dir": str(self._output_dir),
            "execute": self._execute,
            "record": self._record,
            "cost_overrides": str(self._cost_overrides_file) if self._cost_overrides_file else None,
        }
