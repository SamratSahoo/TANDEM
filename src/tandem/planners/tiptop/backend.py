"""TiPToP as a tandem backend: the parent-process half.

All this does is turn protocol calls into requests on a child that runs inside the GPU runtime.
The child is ``sidecar.py``, which lives in this package -- it is TANDEM's code executed in TiPToP's
environment, not TiPToP's code. That is the whole trick, and it is why the planner repositories stay
untouched: everything tandem needs from TiPToP is a call to a public function of it, made from a file
tandem ships.

Nothing in this module imports torch, cuTAMP or tiptop, and nothing ever will. It is on the path of
``pip install tandem-tamp`` on a laptop.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from tandem.core.runtime import Runtime
from tandem.planners.base import (
    Capabilities,
    ExecuteResult,
    GoalAtom,
    LegSpec,
    PlanResult,
    SceneView,
)
from tandem.planners.rpc import HostedBackendChannel
from tandem.planners.tiptop.capabilities import CAPABILITIES

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
    standard library and what tiptop already has.
    """
    return Path(__file__).resolve().parent / "sidecar.py"


class TiptopBackend:
    """Drives TiPToP for one collection session."""

    name = "tiptop"

    def __init__(
        self,
        runtime: Runtime,
        *,
        env: dict[str, str],
        output_dir: Path,
        execute: bool = True,
        record: bool = True,
        cost_overrides_file: Path | None = None,
        on_log: Callable[[str, str], None] | None = None,
    ) -> None:
        self._runtime = runtime
        self._env = env
        self._output_dir = Path(output_dir)
        self._execute = execute
        self._record = record
        self._cost_overrides_file = cost_overrides_file
        self._on_log = on_log
        self._channel: HostedBackendChannel | None = None

    # ---- what this planner is ----------------------------------------------

    def capabilities(self) -> Capabilities:
        return CAPABILITIES

    def require_ready(self) -> None:
        self._runtime.require_ready()

    # ---- lifecycle ---------------------------------------------------------

    def warm(self) -> None:
        if self._channel is None:
            argv = self._runtime.command(["python", str(sidecar_path())])
            self._on_log and self._on_log("tandem", "$ " + " ".join(argv))
            self._channel = HostedBackendChannel(
                argv,
                cwd=self._runtime.tiptop_dir,
                env=self._env,
                on_log=self._on_log,
            ).start()
        self._call(
            "warm",
            timeout=WARM_TIMEOUT,
            output_dir=str(self._output_dir),
            execute=self._execute,
            record=self._record,
            cost_overrides=str(self._cost_overrides_file) if self._cost_overrides_file else None,
        )

    def close(self) -> None:
        channel, self._channel = self._channel, None
        if channel is not None:
            channel.stop()

    # ---- hardware custody --------------------------------------------------

    def release_hardware(self) -> None:
        self._call("release_hardware", timeout=HARDWARE_TIMEOUT)

    def reacquire_hardware(self) -> None:
        self._call("reacquire_hardware", timeout=HARDWARE_TIMEOUT)

    def capture_frame(self, *, camera: str = "external") -> str:
        return str(self._call("capture_frame", timeout=HARDWARE_TIMEOUT, camera=camera)["path"])

    def home(self) -> None:
        self._call("home", timeout=HARDWARE_TIMEOUT)

    # ---- the sub-goal cycle ------------------------------------------------

    def perceive(
        self,
        *,
        task_hint: str,
        save_dir: Path,
        reset_arm: bool = True,
        open_gripper: bool = False,
    ) -> SceneView:
        data = self._call(
            "perceive",
            timeout=PERCEIVE_TIMEOUT,
            task_hint=task_hint,
            save_dir=str(save_dir),
            reset_arm=reset_arm,
            open_gripper=open_gripper,
        )
        return SceneView.from_dict(data)

    def plan(
        self,
        scene_id: str,
        goal: Sequence[GoalAtom],
        *,
        surfaces: frozenset[str] = frozenset(),
        movables: frozenset[str] | None = None,
        return_home: bool = True,
        save_dir: Path,
        reuse_skeleton: Any = None,
    ) -> PlanResult:
        # Both are honoured in the sidecar (see Sidecar.plan), which is what CAPABILITIES declares.
        # None and an empty set are different requests -- "anything may be picked" against "nothing
        # may" -- so the None survives the wire as null rather than becoming [].
        data = self._call(
            "plan",
            timeout=PLAN_TIMEOUT,
            scene_id=scene_id,
            goal=[a.to_dict() for a in goal],
            surfaces=sorted(surfaces),
            movables=sorted(movables) if movables is not None else None,
            return_home=bool(return_home),
            save_dir=str(save_dir),
        )
        return PlanResult.from_dict(data)

    def execute(
        self,
        plan_handle: Any,
        leg: LegSpec,
        *,
        save_dir: Path,
        should_stop: Callable[[], bool] | None = None,
    ) -> ExecuteResult:
        # should_stop is ignored: upstream's execute_cutamp_plan takes a whole trajectory and has no
        # mid-plan seam, which is exactly what capabilities().supports_cooperative_stop says. The
        # parameter stays on the protocol because a backend that CAN stop should be able to.
        data = self._call(
            "execute",
            timeout=EXECUTE_TIMEOUT,
            plan_handle=plan_handle,
            leg=leg.to_dict(),
            save_dir=str(save_dir),
        )
        return ExecuteResult.from_dict(data)

    # ---- plumbing ----------------------------------------------------------

    def _call(self, verb: str, *, timeout: float, **args: Any) -> Any:
        from tandem.planners.base import BackendError

        if self._channel is None:
            raise BackendError("the tiptop backend has not been warmed")
        return self._channel.call(verb, timeout=timeout, **args)
