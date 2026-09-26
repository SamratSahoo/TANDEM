"""TiPToP's two perception servers: built as runtimes, and started on demand while collecting.

TiPToP asks an M2T2 server for grasps and a FoundationStereo server for depth on every rollout. Each is a
runtime of its own -- a pinned commit of SamratSahoo's fork (its TANDEM branch, which adds the pixi tasks
this needs), a pixi environment, and build steps -- installed by ``tandem init`` (``install``) beside
TiPToP's.

A session does not need them running when it starts. Before warming, and again before every perception
pass, the backend calls ``ServerManager.ensure``: a server whose URL points at this machine and that does
not answer ``/health`` is started from its runtime, on the port its URL names, and waited for. Servers the
session started are stopped when it closes (a session collects many trials, so each is loaded once per
session); one that was already running is left alone. A server on another machine is never started: its
URL is the rig's (``planners.tiptop.perception.<name>.url``), and it is that machine's to run.

Every server started here writes a pid file under ``<state>/servers/``, so ``tandem servers stop`` can end
one a crashed session left behind.

Imported by the backend and the CLI, never on the path of listing planners: nothing here imports torch.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from tandem.core import paths
from tandem.core.errors import TandemError
from tandem.planners.base import SourcePin
from tandem.planners.runtime import BuildStep, PixiEnvironment, RecipeRuntime, RuntimeRecipe, Source

#: The command that builds both servers.
INSTALL_COMMAND = "tandem servers install"

# Both forks' TANDEM branches are their master plus the pixi tasks tandem runs (setup, download-weights,
# server) and tandem_setup.sh, which is build_server.sh without its own pixi calls. M2T2's compiles
# pointnet2_ops for this machine's GPUs (nvidia-smi), not a fixed architecture.
M2T2_RECIPE = RuntimeRecipe(
    planner="m2t2",
    title="M2T2 grasp server",
    install_command=INSTALL_COMMAND,
    sources=(
        Source(
            SourcePin(
                "M2T2",
                "https://github.com/SamratSahoo/M2T2.git",
                "cb63d01b7f3fe3912754e7ea60facc00f309a42d",
                ref="TANDEM",
            ),
            trim=("figures", "sample_data"),
            marker="server.py",
            # The checkpoint (download-weights), kept out of the tree so a pin bump does not download it again.
            persistent=("weights",),
        ),
    ),
    environment=PixiEnvironment(manifest="M2T2/pixi.toml", home="env"),
    steps=(
        BuildStep(
            name="setup",
            task="setup",
            produces=("env/envs/default/lib/python3*/site-packages/pointnet2_ops*",),
            description="installing torch and compiling pointnet2_ops for this GPU — a few minutes",
            label="pointnet2_ops",
            done="compiled",
            todo="not compiled",
            problem="M2T2's pointnet2_ops has not been compiled",
        ),
        BuildStep(
            name="weights",
            task="download-weights",
            produces=("M2T2/weights/m2t2.pth",),
            description="downloading the M2T2 checkpoint from Hugging Face",
            label="checkpoint",
            done="downloaded",
            todo="not downloaded",
            problem="the M2T2 checkpoint has not been downloaded",
        ),
    ),
    notes=("Building the M2T2 grasp server: torch 2.7.1 (CUDA 12.8), pointnet2_ops, and its checkpoint.",),
)

FOUNDATION_STEREO_RECIPE = RuntimeRecipe(
    planner="foundation_stereo",
    title="FoundationStereo depth server",
    install_command=INSTALL_COMMAND,
    sources=(
        Source(
            SourcePin(
                "FoundationStereo",
                "https://github.com/SamratSahoo/FoundationStereo.git",
                "8353e9d790f37be119503f02510a4d6e4cd429eb",
                ref="TANDEM",
            ),
            trim=("teaser",),
            marker="server.py",
            persistent=("pretrained_models",),
        ),
    ),
    environment=PixiEnvironment(manifest="FoundationStereo/pixi.toml", home="env"),
    steps=(
        BuildStep(
            name="setup",
            task="setup",
            produces=("env/envs/default/lib/python3*/site-packages/timm",),
            description="installing torch and the server's dependencies",
            label="dependencies",
            done="installed",
            todo="not installed",
            problem="the FoundationStereo server's dependencies are not installed",
        ),
        BuildStep(
            name="weights",
            task="download-weights",
            produces=("FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth",),
            description="downloading the FoundationStereo weights from Google Drive",
            label="weights",
            done="downloaded",
            todo="not downloaded",
            problem="the FoundationStereo weights have not been downloaded",
        ),
    ),
    notes=("Building the FoundationStereo depth server: torch 2.7.1 (CUDA 12.8) and its weights.",),
)


@dataclass(frozen=True)
class Server:
    name: str  # the rig's key: planners.tiptop.perception.<name>.url
    recipe: RuntimeRecipe
    source: str  # the tree server.py is in

    @property
    def title(self) -> str:
        return self.recipe.display_name

    def runtime(self, settings: Any = None) -> RecipeRuntime:
        return RecipeRuntime(self.recipe, paths.runtimes_dir() / self.name)


SERVERS: tuple[Server, ...] = (
    Server("m2t2", M2T2_RECIPE, "M2T2"),
    Server("foundation_stereo", FOUNDATION_STEREO_RECIPE, "FoundationStereo"),
)

# Loading a checkpoint onto the GPU: M2T2 in seconds, FoundationStereo's ViT-large in up to a minute or two.
START_TIMEOUT = 300.0


def by_name(name: str) -> Server:
    for server in SERVERS:
        if server.name == name:
            return server
    raise TandemError(
        f"There is no perception server called {name!r}.", hint=f"They are {', '.join(s.name for s in SERVERS)}."
    )


def rig_urls(rig: Any = None) -> dict[str, str]:
    """Each server's URL, as this machine's rig sets it (``planners.tiptop.perception.<name>.url``)."""
    from tandem.core import rig as rig_mod
    from tandem.planners.tiptop.options import TiptopRigOptions

    rig = rig if rig is not None else rig_mod.load()
    perception = TiptopRigOptions.model_validate(rig_mod.planner_options(rig, "tiptop")).perception
    return {server.name: str(getattr(perception, server.name).url) for server in SERVERS}


def is_local(url: str) -> bool:
    """Whether ``url`` points at this machine, so tandem may start the server behind it."""
    host = (urlparse(url).hostname or "").lower()
    if host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}:
        return True
    try:
        return host in {socket.gethostname().lower(), socket.getfqdn().lower()}
    except OSError:
        return False


def port_of(url: str) -> int:
    parsed = urlparse(url)
    return parsed.port or (443 if parsed.scheme == "https" else 80)


def healthy(url: str, *, timeout: float = 2.0) -> bool:
    """Whether the server at ``url`` answers ``/health`` with its model loaded."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as response:
            body = json.loads(response.read().decode() or "{}")
    except (OSError, urllib.error.URLError, ValueError):
        return False
    return body.get("status") == "healthy"


def pid_file(name: str) -> Path:
    return paths.state_dir() / "servers" / f"{name}.pid"


def log_file(name: str) -> Path:
    return paths.log_dir() / f"server-{name}.log"


def running_pid(name: str) -> int | None:
    """The process group tandem started this server in, if it is still alive."""
    path = pid_file(name)
    try:
        pid = int(path.read_text().strip())
        os.killpg(pid, 0)
    except (OSError, ValueError):
        return None
    return pid


def start(server: Server, url: str, *, settings: Any = None) -> subprocess.Popen:
    """Launch ``server`` from its runtime on ``url``'s port, detached into its own process group."""
    rt = server.runtime(settings)
    rt.require_ready()
    log = log_file(server.name)
    paths.ensure_dir(log.parent)
    paths.ensure_dir(pid_file(server.name).parent)
    handle = log.open("a")
    handle.write(f"\n== {time.strftime('%Y-%m-%d %H:%M:%S')} starting on {url}\n")
    handle.flush()
    proc = subprocess.Popen(
        rt.command(["python", "server.py", "--host", "0.0.0.0", "--port", str(port_of(url))]),
        cwd=str(rt.source_dir(server.source)),
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    pid_file(server.name).write_text(str(proc.pid))
    return proc


def stop(name: str, *, grace: float = 10.0) -> bool:
    """End the server tandem started under ``name`` (SIGTERM, then SIGKILL). Returns whether one was running."""
    pid = running_pid(name)
    pid_file(name).unlink(missing_ok=True)
    if pid is None:
        return False
    try:
        os.killpg(pid, signal.SIGTERM)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            os.killpg(pid, 0)
            time.sleep(0.2)
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        pass  # gone
    return True


class ServerManager:
    """The perception servers one session needs: started if down, stopped at the end if it started them."""

    def __init__(self, urls: dict[str, str], *, log: Callable[[str], None], settings: Any = None) -> None:
        self._urls = urls
        self._log = log
        self._settings = settings
        self._started: dict[str, subprocess.Popen] = {}

    def ensure(self, *, timeout: float = START_TIMEOUT) -> None:
        """Every local server answering ``/health``, starting the ones that are not. Raises if one never comes up."""
        waiting: list[tuple[Server, str]] = []
        for server in SERVERS:
            url = self._urls.get(server.name)
            if not url or healthy(url):
                continue
            if not is_local(url):
                # Another machine's to run; TiPToP's warm-up says so if it is not answering.
                continue
            if not server.runtime(self._settings).is_ready():
                # Not installed here: a server run by hand, or none. TiPToP's warm-up says so if it is not
                # answering, and `tandem servers install` is how tandem would start it.
                self._log(f"the {server.title} is not answering {url}, and its runtime is not installed "
                          f"(`{INSTALL_COMMAND}`), so it is not started")
                continue
            proc = self._started.get(server.name)
            if proc is not None and proc.poll() is None:
                waiting.append((server, url))  # started by us and still loading
                continue
            self._log(f"starting the {server.title} on {url} (log: {log_file(server.name)})")
            self._started[server.name] = start(server, url, settings=self._settings)
            waiting.append((server, url))

        deadline = time.monotonic() + timeout
        for server, url in waiting:
            while not healthy(url):
                proc = self._started.get(server.name)
                if proc is not None and proc.poll() is not None:
                    raise TandemError(
                        f"The {server.title} exited with code {proc.returncode} while starting.",
                        hint=f"Its log says why: {log_file(server.name)}",
                    )
                if time.monotonic() > deadline:
                    raise TandemError(
                        f"The {server.title} did not answer {url}/health within {int(timeout)} s.",
                        hint=f"Its log says why: {log_file(server.name)}",
                    )
                time.sleep(1.0)
            self._log(f"the {server.title} is up on {url}")

    def stop_started(self) -> None:
        """Stop the servers this session started. A server that was already running is not touched."""
        for name in list(self._started):
            self._started.pop(name)
            if stop(name):
                self._log(f"stopped the {by_name(name).title}")


class Service:
    """One server, as ``registry.services`` hands it out: what `tandem servers` and `tandem init` ask of it.

    The duck type every planner's services follow: ``name``, ``title``, ``runtime(settings)``, ``url()``,
    ``local()``, ``healthy()``, ``started_pid()``, ``log_path``, ``start()`` and ``stop()``.
    """

    def __init__(self, server: Server, *, settings: Any = None) -> None:
        self._server = server
        self._settings = settings
        self.name = server.name
        self.title = server.title
        self.log_path = log_file(server.name)

    def runtime(self, settings: Any = None) -> RecipeRuntime:
        return self._server.runtime(settings if settings is not None else self._settings)

    def url(self) -> str:
        return rig_urls()[self.name]

    def local(self) -> bool:
        return is_local(self.url())

    def healthy(self) -> bool:
        return healthy(self.url())

    def started_pid(self) -> int | None:
        return running_pid(self.name)

    def start(self, *, timeout: float = START_TIMEOUT) -> None:
        """Start it if it is down and local, and wait for it. It keeps running until ``stop``."""
        ServerManager({self.name: self.url()}, log=lambda _text: None, settings=self._settings).ensure(
            timeout=timeout
        )

    def stop(self) -> bool:
        return stop(self.name)


def services(settings: Any = None) -> list[Service]:
    return [Service(server, settings=settings) for server in SERVERS]

