"""Preflight probes.

Every check answers three things: did it pass, what did we find, and — when it did not —
what should the user do. The last part is the whole point; a red cross with no next step is
just a slower way of saying "it broke".

Shared by `tandem doctor` and `tandem init`, so the wizard and the diagnostic can never
disagree about what "ready" means.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

OK = "ok"
WARN = "warn"
FAIL = "fail"
SKIP = "skip"


@dataclass
class Check:
    name: str
    state: str
    detail: str = ""
    hint: str = ""
    # Checks that only matter for collecting; a laptop running `tandem ui` skips them.
    group: str = "core"
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "state": self.state,
            "detail": self.detail,
            "hint": self.hint,
            "group": self.group,
            **({"data": self.data} if self.data else {}),
        }


def _run(cmd: list[str], timeout: float = 10.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return proc.returncode, (proc.stdout or proc.stderr or "").strip()
    except FileNotFoundError:
        return 127, ""
    except subprocess.TimeoutExpired:
        return 124, ""
    except OSError as exc:
        return 1, str(exc)


# --------------------------------------------------------------------------- core


def check_python() -> Check:
    version = platform.python_version()
    # requires-python keeps pip from installing on an older interpreter, but running a source
    # checkout directly bypasses that — and a syntax-error traceback is a poor way to find out.
    if sys.version_info < (3, 10):  # noqa: UP036
        return Check("python", FAIL, version, "tandem needs Python 3.10 or newer.")
    return Check("python", OK, version)


def check_platform() -> Check:
    system = platform.system()
    if system != "Linux":
        return Check(
            "operating system",
            WARN,
            system,
            "Collection needs Linux (CUDA, the ZED SDK and the robot stack). Visualization works anywhere.",
        )
    return Check("operating system", OK, f"{system} {platform.release()}")


def check_disk(path: Path, need_gb: float = 25.0) -> Check:
    target = path
    while not target.exists() and target != target.parent:
        target = target.parent
    try:
        usage = shutil.disk_usage(target)
    except OSError as exc:
        return Check("disk space", WARN, str(exc), group="runtime")
    free_gb = usage.free / 1e9
    detail = f"{free_gb:.0f} GB free at {target}"
    if free_gb < need_gb:
        return Check(
            "disk space",
            FAIL if free_gb < 10 else WARN,
            detail,
            f"The runtime needs roughly {need_gb:.0f} GB (CUDA toolkit, torch, compiled kernels).",
            group="runtime",
        )
    return Check("disk space", OK, detail, group="runtime")


# --------------------------------------------------------------------------- gpu


def check_nvidia_driver() -> Check:
    code, out = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    if code == 127:
        return Check(
            "nvidia driver",
            FAIL,
            "nvidia-smi not found",
            "Install the NVIDIA driver. Collection needs a CUDA 12 GPU; visualization does not.",
            group="gpu",
        )
    if code != 0 or not out:
        return Check("nvidia driver", FAIL, "nvidia-smi failed", "Check the driver installation.", group="gpu")
    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    name = parts[0] if parts else "GPU"
    driver = parts[1] if len(parts) > 1 else "?"
    memory = parts[2] if len(parts) > 2 else "?"
    return Check("nvidia driver", OK, f"{name} · driver {driver} · {memory}", group="gpu",
                 data={"name": name, "driver": driver, "memory": memory})


def check_cuda_runtime() -> Check:
    code, out = _run(["nvidia-smi", "--query", "-d", "COMPUTE"])
    if code != 0:
        return Check("cuda runtime", SKIP, group="gpu")
    version = ""
    code2, smi = _run(["nvidia-smi"])
    if code2 == 0:
        for line in smi.splitlines():
            if "CUDA Version" in line:
                version = line.split("CUDA Version:")[-1].strip().rstrip("|").strip()
                break
    if not version:
        return Check("cuda runtime", WARN, "version not reported", group="gpu")
    major = version.split(".")[0]
    if major.isdigit() and int(major) < 12:
        return Check(
            "cuda runtime",
            FAIL,
            version,
            "The planner stack is built against CUDA 12. Upgrade the driver.",
            group="gpu",
        )
    return Check("cuda runtime", OK, version, group="gpu")


def check_nvcc() -> Check:
    code, out = _run(["nvcc", "--version"])
    if code == 127:
        return Check(
            "nvcc",
            WARN,
            "not on PATH",
            "A planner that compiles CUDA kernels at install time (cuRobo's, for TiPToP) needs it. A runtime "
            "recipe brings its own cuda-toolkit, so this is only a problem if that build fails.",
            group="gpu",
        )
    version = ""
    for line in out.splitlines():
        if "release" in line:
            version = line.split("release")[-1].split(",")[0].strip()
    return Check("nvcc", OK, version or "present", group="gpu")


# --------------------------------------------------------------------------- tooling


def check_pixi() -> Check:
    exe = find_pixi()
    if not exe:
        return Check(
            "pixi",
            FAIL,
            "not found",
            "tandem init can install it, or: curl -fsSL https://pixi.sh/install.sh | bash",
            group="runtime",
        )
    code, out = _run([str(exe), "--version"])
    return Check("pixi", OK, out or str(exe), group="runtime", data={"path": str(exe)})


def find_pixi() -> Path | None:
    """pixi installs to ~/.pixi/bin, which is often not on a non-login shell's PATH."""
    found = shutil.which("pixi")
    if found:
        return Path(found)
    candidate = Path.home() / ".pixi" / "bin" / "pixi"
    return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None


def check_tool(name: str, *, group: str = "runtime", hint: str = "", required: bool = True) -> Check:
    found = shutil.which(name)
    if found:
        return Check(name, OK, found, group=group)
    return Check(name, FAIL if required else WARN, "not found", hint, group=group)


def check_ffmpeg() -> Check:
    return check_tool(
        "ffmpeg",
        group="runtime",
        hint="Needed to join the legs of a teleop hand-off into one trajectory. apt install ffmpeg",
        required=False,
    )


# --------------------------------------------------------------------------- hardware
#
# Which hardware a session needs is its planner's to say (a planner's ``doctor_checks``), so the probes
# of one rig's robot shim or grasp server live with that planner -- TiPToP's in
# ``tandem/planners/tiptop/probe.py``. What stays here is the one thing every such probe is made of.


def check_port(name: str, host: str, port: int, *, timeout: float = 1.5, group: str = "hardware", hint: str = "") -> Check:
    """Whether ``host:port`` accepts a connection. A closed port is a warning: the thing is off, not broken."""
    return _check_port(name, host, port, timeout, group=group, hint=hint)


def _check_port(name: str, host: str, port: int, timeout: float, *, group: str, hint: str) -> Check:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return Check(name, OK, f"{host}:{port} reachable", group=group)
    except OSError as exc:
        return Check(name, WARN, f"{host}:{port} — {exc.__class__.__name__}", hint, group=group)


# --------------------------------------------------------------------------- credentials


def check_gemini_key() -> Check:
    from tandem.core import secrets

    source = secrets.gemini_key_source()
    if source == "none":
        return Check(
            "gemini api key",
            FAIL,
            "not set",
            "Run `tandem config set-gemini-key`. Perception calls Gemini once per rollout to turn "
            "the task string into objects and goal predicates, so collection cannot run without it.",
            group="credentials",
        )
    return Check(
        "gemini api key",
        OK,
        f"{secrets.mask(secrets.gemini_api_key())} (from {source})",
        group="credentials",
    )
