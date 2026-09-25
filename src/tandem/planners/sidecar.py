"""A planner that runs in an environment of its own, driven from tandem over JSON lines.

Most task and motion planners need torch, CUDA kernels, a camera SDK or a robot client, and tandem
needs none of them -- ``pip install tandem-tamp`` has to keep working on a laptop. So such a planner
runs as a child process inside its own runtime, and this is the parent half of it, for any planner:

    parent: tandem (pure Python)                   child: the planner's environment
      MyPlanner(SidecarPlanner).plan(goal) --JSON-->  my_sidecar.py: plan(scene_id, goal, ...)
                                           <--JSON--  {"ok": true, "plan_handle": ...}

A planner written this way is two files. The class, here in tandem's process::

    class MyPlanner(SidecarPlanner):
        info = PlannerInfo(name="mine", ...)
        CAPABILITIES = Capabilities(name="mine", ...)
        recipe = RuntimeRecipe(planner="mine", ...)     # the environment the sidecar runs in
        SIDECAR = "my_sidecar.py"                       # next to this module

and the sidecar script, run by the runtime's interpreter, which imports nothing from tandem -- only
``tandem_sidecar``, a standard-library-only helper tandem puts on its ``PYTHONPATH``
(``tandem/planners/sidecar_kit/tandem_sidecar.py`` has the protocol and a template).

What this class does, so a planner does not have to:

- **launches** the sidecar inside the runtime (``recipe``'s environment, via ``pixi run``; or this
  interpreter when there is no runtime), in its own process group, with the kit on its path;
- **maps every protocol verb** onto a request and the reply back onto tandem's types. ``movables``,
  ``return_home`` and ``reuse_skeleton`` go on the wire only when the capabilities declare them, so a
  sidecar that supports neither never has to accept them -- and a caller that passes one to a
  planner that did not declare it gets an error, not a silently unrestricted plan;
- **falls back** to ``Planner``'s defaults for a verb the sidecar does not answer (it lists the
  verbs it answers when it starts), and refuses a sidecar that does not answer
  ``perceive``/``plan``/``execute`` before the session relies on it;
- **streams** the sidecar's log lines and stderr into the session log, and its events into the
  session's events file;
- **times out** every verb (``TIMEOUTS``), reporting a wedged sidecar as one that may still hold
  the robot and then stopping it, with every process it started, so it holds nothing; reports a
  crash with its exit code; and starts a fresh sidecar at the next ``warm()`` after one has died or
  been stopped (``warm()`` on a sidecar running and warm does nothing, so it is safe to call again);
- **stops cooperatively** when the capabilities declare ``supports_cooperative_stop``: while an
  execution runs, ``should_stop`` is polled here and a stop is passed to the sidecar as a file whose
  existence ``tandem_sidecar.should_stop()`` reports -- no second request in flight, which the
  protocol does not have;
- **holds custody** the way the protocol needs: ``release_hardware``/``reacquire_hardware`` are
  requests with their own generous timeout, and ``close`` asks the sidecar to quit, then makes sure
  the whole process group has gone, so nothing is left holding a camera -- also when the sidecar
  exited by itself, or crashed, and left helpers of its own behind.

TiPToP's backend is this class plus TiPToP's own launch details (``tandem/planners/tiptop/backend.py``).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from tandem.core.errors import TandemError
from tandem.planners import rpc, sidecar_kit
from tandem.planners.base import (
    BackendContext,
    BackendError,
    ExecuteResult,
    GoalAtom,
    LegSpec,
    PlanResult,
    SceneView,
)
from tandem.planners.sdk import Planner

#: How long each verb may take before the sidecar is reported as wedged. Generous, because the
#: slow ones are genuinely slow: warming builds CUDA solvers and opens cameras, perception is a
#: detector and a grasp model, execution is a whole trajectory on a real arm plus writing the videos
#: out, and releasing two cameras measures ~14s of SDK teardown. A subclass overrides any of them
#: in ``TIMEOUTS``.
DEFAULT_TIMEOUTS: Mapping[str, float] = {
    "warm": 900.0,
    "perceive": 300.0,
    "plan": 900.0,
    "execute": 1800.0,
    "capture_frame": 180.0,
    "home": 180.0,
    "release_hardware": 180.0,
    "reacquire_hardware": 180.0,
}

#: Verbs a sidecar must answer. Everything else has a default on tandem's side.
REQUIRED_VERBS = ("perceive", "plan", "execute")

#: The environment variable naming the stop file; ``tandem_sidecar.should_stop`` reads it.
STOP_FILE_ENV = "TANDEM_SIDECAR_STOP_FILE"

_UNSET: Any = object()


class SidecarPlanner(Planner, abstract=True):
    """A ``Planner`` whose work is done by a sidecar script in the planner's own environment.

    Declare ``SIDECAR`` (the script, relative to the module defining the subclass) alongside the
    usual ``info``/``CAPABILITIES``/``recipe``. Override the ``launch_*`` hooks when the sidecar
    needs something the defaults do not give it, and ``warm_args`` to hand it planner-specific
    settings when it warms.
    """

    #: The sidecar script: absolute, or relative to the directory of the module defining the class --
    #: or to the working directory, for a class defined with no module file (a notebook, `python -c`).
    SIDECAR: ClassVar[str] = ""
    #: Per-verb timeouts in seconds, over DEFAULT_TIMEOUTS.
    TIMEOUTS: ClassVar[Mapping[str, float]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls._planner_base:
            return
        # Said at definition, like every other declaration: a planner whose sidecar is not where it
        # says would otherwise list, start a session, and fail at warm-up with the operator waiting.
        if not cls.SIDECAR and cls.sidecar_script.__func__ is SidecarPlanner.sidecar_script.__func__:
            raise TandemError(
                f"{cls.__module__}.{cls.__qualname__} declares no SIDECAR.",
                hint="Set SIDECAR to the sidecar script's path, relative to the module the class is in.",
            )
        script = cls.sidecar_script()
        if not script.is_file():
            raise TandemError(
                f"{cls.__module__}.{cls.__qualname__}'s sidecar script {script} does not exist "
                f"(SIDECAR {cls.SIDECAR!r}, resolved against {cls._sidecar_base()}).",
                hint="SIDECAR is resolved against the directory of the module that defines the class, or "
                "against the working directory for a class defined where there is no module file (a "
                "notebook, `python -c`). An absolute path is used as it is.",
            )

    def __init__(
        self,
        ctx: BackendContext | None = None,
        *,
        runtime: Any = _UNSET,
        env: Mapping[str, str] | None = None,
        on_log: Callable[[str, str], None] | None = None,
    ) -> None:
        """``runtime`` defaults to the declared recipe's; pass one to launch in another (or None).

        ``env`` is the environment the sidecar starts with (this process's when None), before the
        kit's path and the stop file are added to it.
        """
        super().__init__(ctx)
        if on_log is not None:
            self._on_log = on_log
        self._runtime = type(self).runtime(self.settings) if runtime is _UNSET else runtime
        self._env = env
        self._channel: rpc.HostedBackendChannel | None = None
        # The channel whose sidecar has been warmed, so warm() on it again is a no-op (see warm).
        self._warmed: rpc.HostedBackendChannel | None = None
        self._stop_dir: Path | None = None

    # ---- launching: override what the defaults get wrong for a planner ---------------------------

    @classmethod
    def sidecar_script(cls) -> Path:
        """The script the runtime's interpreter runs."""
        script = Path(cls.SIDECAR)
        if not script.is_absolute():
            script = cls._sidecar_base() / script
        return script

    @classmethod
    def _sidecar_base(cls) -> Path:
        """What a relative SIDECAR is resolved against: the directory of the module that SAID it.

        The class that said it, so a subclass defined elsewhere inherits the script rather than looking
        for one next to itself. A module with no file -- a notebook cell, `python -c`, an embedded
        interpreter -- resolves against the working directory, which is where such a session was
        started and where its author put the script. It used to be `Path(".").resolve().parent`: the
        directory ABOVE that, where the script never is.
        """
        owner = next((k for k in cls.__mro__ if "SIDECAR" in vars(k)), cls)
        module = sys.modules.get(owner.__module__)
        file = getattr(module, "__file__", None)
        return Path(file).resolve().parent if file else Path.cwd()

    def launch_command(self) -> list[str]:
        """argv for the sidecar: the runtime's ``python`` when there is a runtime, else this interpreter."""
        script = str(self.sidecar_script())
        command = getattr(self._runtime, "command", None)
        if callable(command):
            return list(command(["python", script]))
        return [sys.executable, script]

    def launch_cwd(self) -> Path | None:
        """Where the sidecar starts: the runtime's working directory, when it has one."""
        workdir = getattr(self._runtime, "workdir", None)
        return Path(workdir) if workdir is not None else None

    def launch_env(self) -> dict[str, str]:
        """The sidecar's environment: ``env``, with ``tandem_sidecar`` put first on its PYTHONPATH."""
        env = dict(self._env) if self._env is not None else dict(os.environ)
        kit = str(sidecar_kit.DIRECTORY)
        rest = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != kit]
        env["PYTHONPATH"] = os.pathsep.join([kit, *rest])
        if self._stop_dir is not None:
            env[STOP_FILE_ENV] = str(self._stop_file)
        return env

    def warm_args(self) -> dict[str, Any]:
        """What the sidecar's ``warm`` is handed: where legs go and whether to execute and record them."""
        ctx = self.ctx
        return {
            "output_dir": str(ctx.output_dir) if ctx is not None else None,
            "execute": ctx.execute if ctx is not None else True,
            "record": ctx.record if ctx is not None else True,
        }

    def on_event(self, event: Mapping[str, Any]) -> None:
        """An event the sidecar sent: appended to the session's events file, or logged outside one."""
        events_file = self.ctx.events_file if self.ctx is not None else None
        if events_file is None:
            self.log(f"event: {json.dumps(dict(event))}", stream="backend")
            return
        with open(events_file, "a") as handle:
            handle.write(json.dumps(dict(event)) + "\n")

    def timeout(self, verb: str) -> float:
        return float({**DEFAULT_TIMEOUTS, **type(self).TIMEOUTS}.get(verb, rpc.DEFAULT_TIMEOUT))

    # ---- lifecycle ---------------------------------------------------------------------------------

    def require_ready(self) -> None:
        # The runtime this sidecar will actually launch in, which is not necessarily the declared
        # one: None means this interpreter, and there is nothing to check.
        self.check_runtime(self._runtime)

    def warm(self) -> None:
        """Start the sidecar and warm it -- or, when it is running and warm already, do nothing.

        Safe to call again at any time, which is what makes it the way back from a crash: tandem calls
        it after a verb raised (``Session``), and a sidecar that died, or was stopped for not answering,
        is relaunched and warmed here. One that is running is not warmed twice: warming opens the
        cameras and connects the robot, and doing that over a sidecar holding them already is the
        failure, not the recovery.
        """
        channel = self._channel
        if channel is not None and channel.alive and self._warmed is channel:
            return
        self._start()
        if self._answers("warm"):
            self._request("warm", self.warm_args())
        self._warmed = self._channel

    def close(self) -> None:
        channel, self._channel = self._channel, None
        self._warmed = None
        if channel is not None:
            channel.stop()
        if self._stop_dir is not None:
            shutil.rmtree(self._stop_dir, ignore_errors=True)
            self._stop_dir = None

    # ---- hardware custody --------------------------------------------------------------------------

    def release_hardware(self) -> None:
        if self._answers("release_hardware"):
            self._request("release_hardware", {})
        else:
            super().release_hardware()

    def reacquire_hardware(self) -> None:
        if self._answers("reacquire_hardware"):
            self._request("reacquire_hardware", {})
        else:
            super().reacquire_hardware()

    def capture_frame(self, *, camera: str = "external") -> str:
        if not self._answers("capture_frame"):
            return super().capture_frame(camera=camera)
        return str(self._request("capture_frame", {"camera": camera})["path"])

    def home(self) -> None:
        if self._answers("home"):
            self._request("home", {})
        else:
            super().home()

    # ---- the sub-goal cycle ------------------------------------------------------------------------

    def perceive(
        self,
        *,
        task_hint: str,
        save_dir: Path,
        reset_arm: bool = True,
        open_gripper: bool = False,
    ) -> SceneView:
        data = self._request(
            "perceive",
            {
                "task_hint": task_hint,
                "save_dir": str(save_dir),
                "reset_arm": reset_arm,
                "open_gripper": open_gripper,
            },
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
        caps = self.capabilities()
        args: dict[str, Any] = {
            "scene_id": scene_id,
            "goal": [a.to_dict() for a in goal],
            "surfaces": sorted(surfaces),
        }
        # Each of these goes on the wire only when the planner said it honours it -- a sidecar that
        # does not may leave it out of its handler's signature -- and asking for one anyway is an
        # error rather than a plan quietly made without it.
        if caps.supports_movable_restriction:
            # None and an empty set are different requests -- "anything may be picked" against
            # "nothing may" -- so None travels as null rather than as [].
            args["movables"] = sorted(movables) if movables is not None else None
        elif movables is not None:
            raise self._undeclared("movables", "supports_movable_restriction")
        if caps.supports_return_home:
            args["return_home"] = bool(return_home)
        elif not return_home:
            raise self._undeclared("return_home=False", "supports_return_home")
        if caps.supports_skeleton_reuse:
            args["reuse_skeleton"] = reuse_skeleton
        elif reuse_skeleton is not None:
            raise self._undeclared("reuse_skeleton", "supports_skeleton_reuse")
        args["save_dir"] = str(save_dir)
        return PlanResult.from_dict(self._request("plan", args))

    def execute(
        self,
        plan_handle: Any,
        leg: LegSpec,
        *,
        save_dir: Path,
        should_stop: Callable[[], bool] | None = None,
    ) -> ExecuteResult:
        args = {"plan_handle": plan_handle, "leg": leg.to_dict(), "save_dir": str(save_dir)}
        stop_file = self._stop_file if self._stop_dir is not None else None
        if should_stop is None or stop_file is None:
            # Without cooperative stop a preempt is an abort, which is what the capabilities say.
            return ExecuteResult.from_dict(self._request("execute", args))

        stop_file.unlink(missing_ok=True)
        asked = False

        def poll() -> None:
            nonlocal asked
            if asked:
                return
            try:
                stop = bool(should_stop())
            except Exception as exc:  # the caller's predicate failing is not a reason to stop the arm
                self.log(f"should_stop raised {type(exc).__name__}: {exc}; carrying on")
                return
            if stop:
                stop_file.touch()
                asked = True
                self.log(f"asked the {self.name} planner to stop at its next step boundary")

        try:
            return ExecuteResult.from_dict(self._request("execute", args, poll=poll))
        finally:
            stop_file.unlink(missing_ok=True)

    # ---- for subclasses: a verb of the sidecar's own ------------------------------------------------

    def call(self, verb: str, *, timeout: float | None = None, **args: Any) -> Any:
        """Send the sidecar any verb it answers -- a debugging or planner-specific one included."""
        return self._ask(verb, args, timeout=self.timeout(verb) if timeout is None else timeout)

    @property
    def sidecar_verbs(self) -> frozenset[str] | None:
        """The verbs the running sidecar said it answers, or None when it did not say (or is not running)."""
        channel = self._channel
        verbs = channel.hello.get("verbs") if channel is not None else None
        return frozenset(verbs) if isinstance(verbs, list) else None

    # ---- plumbing ----------------------------------------------------------------------------------

    @property
    def _stop_file(self) -> Path:
        assert self._stop_dir is not None
        return self._stop_dir / "stop"

    def _start(self) -> None:
        """Launch the sidecar unless one is running. One that died is replaced, and the log says so."""
        channel = self._channel
        if channel is not None and channel.alive:
            return
        if channel is not None:
            self.log(f"the {self.name} sidecar is no longer running; starting a new one")
            channel.stop()
            self._channel = None
        if self.capabilities().supports_cooperative_stop and self._stop_dir is None:
            self._stop_dir = Path(tempfile.mkdtemp(prefix=f"tandem-{self.name}-"))
        argv = self.launch_command()
        self.log("$ " + " ".join(argv))
        channel = rpc.HostedBackendChannel(
            argv,
            cwd=self.launch_cwd(),
            env=self.launch_env(),
            on_log=self._on_log,
            on_event=self.on_event,
        ).start()
        verbs = channel.hello.get("verbs")
        if isinstance(verbs, list):
            missing = [verb for verb in REQUIRED_VERBS if verb not in verbs]
            if missing:
                channel.stop()
                raise BackendError(
                    f"the {self.name} sidecar ({self.sidecar_script().name}) does not answer "
                    f"{', '.join(missing)}, which every planner must; it answers {', '.join(verbs) or 'nothing'}"
                )
        self._channel = channel

    def _answers(self, verb: str) -> bool:
        """Whether the sidecar answers ``verb``. Asking before it is warm is an error, as every verb is."""
        self._channel_or_raise()
        verbs = self.sidecar_verbs
        # A sidecar that does not list its verbs is taken to answer the whole protocol, as every
        # sidecar did before they were listed.
        return verbs is None or verb in verbs

    def _request(self, verb: str, args: dict[str, Any], *, poll: Callable[[], None] | None = None) -> Any:
        return self._ask(verb, args, timeout=self.timeout(verb), poll=poll)

    def _ask(
        self, verb: str, args: dict[str, Any], *, timeout: float, poll: Callable[[], None] | None = None
    ) -> Any:
        """One request. A sidecar that does not answer it in time is stopped, with all it started.

        Left running, a wedged sidecar goes on holding the robot and the cameras -- possibly still
        moving the arm -- while the session carries on around it: a person is handed an arm it has not
        let go of, and every later request queues behind the one it never answered. Nothing else can
        be known about what it holds, so it is ended (SIGTERM, then SIGKILL, to its process group), and
        the next ``warm()`` starts a fresh one, as it does after a crash.
        """
        channel = self._channel_or_raise()
        try:
            return channel.request(verb, args, timeout=timeout, poll=poll)
        except rpc.BackendTimeout as exc:
            self.log(
                f"the {self.name} sidecar did not answer {verb} within {timeout:.0f}s; stopping it and "
                "every process it started, so nothing is left holding the robot or the cameras"
            )
            channel.kill()
            raise rpc.BackendTimeout(
                f"{exc}. It was stopped, with every process it started; the next warm-up starts a fresh one"
            ) from exc

    def _channel_or_raise(self) -> rpc.HostedBackendChannel:
        if self._channel is None:
            raise BackendError(f"the {self.name} backend has not been warmed")
        return self._channel

    def _undeclared(self, what: str, flag: str) -> TandemError:
        return TandemError(
            f"The {self.info.title} planner was asked for {what}, but its capabilities do not declare {flag}.",
            hint=f"tandem passes it only to a planner declaring {flag}. Declare it (and honour it in the "
            "sidecar), or do not pass it.",
        )
