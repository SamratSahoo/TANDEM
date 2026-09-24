"""TiPToP's hardware probes: its cameras' SDK, the robot shim it drives, and the servers perception asks.

Run by TiPToP's ``doctor_checks`` (``doctor.py``), which is how `tandem doctor` learns what this
planner needs of a machine. They were in tandem's own probe module while TiPToP was the only planner;
a rig whose robot is not behind the bamboo-polymetis shim has nothing to learn from them.

Built from ``tandem.core.probe``'s generic parts, so a row here reads like every other row.
"""

from __future__ import annotations

from pathlib import Path

from tandem.core import probe


def check_zed_sdk() -> probe.Check:
    root = Path("/usr/local/zed")
    if not root.is_dir():
        return probe.Check(
            "zed sdk",
            probe.WARN,
            "not installed",
            "Cameras need the ZED SDK from stereolabs.com. Only required to collect.",
            group="hardware",
        )
    version_file = root / "settings" / "ZED_SDK_version.txt"
    detail = version_file.read_text().strip() if version_file.is_file() else str(root)
    return probe.Check("zed sdk", probe.OK, detail, group="hardware")


def check_robot(host: str, port: int, *, timeout: float = 1.5) -> probe.Check:
    """The bamboo shim's control port. A closed port is normal when the robot is off."""
    return probe.check_port(
        "robot control", host, port, timeout=timeout, hint="Start the bamboo-polymetis shim on the NUC."
    )


def check_robot_state_port(host: str, port: int, *, timeout: float = 1.5) -> probe.Check:
    return probe.check_port(
        "robot state port",
        host,
        port,
        timeout=timeout,
        hint=(
            "Start the bamboo-polymetis shim on the NUC: it serves the encoders on this port, so they can "
            "be read while the arm moves. Without it, capture aborts rather than falling back to plan "
            "positions."
        ),
    )


def _check_server(
    label: str, url: str, *, example: str, setting: str, hint: str, timeout: float = 1.5
) -> probe.Check:
    """Reach a server perception calls, or say why its address cannot even be used.

    A malformed URL is reported, never raised. `doctor` is the command you run *because*
    something is wrong, so a probe that throws takes down the one tool that was going to tell
    you what to fix — and it hides every check after it.
    """
    from urllib.parse import urlparse

    fix = f"Set it to a plain URL, such as {example}: `tandem rig set {setting} URL`."
    try:
        parsed = urlparse(url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        host = parsed.hostname
    except ValueError as exc:
        return probe.Check(
            label,
            probe.FAIL,
            f"{url} — {exc}",
            "That is not a usable address. An unresolved ${oc.env:...} here means OmegaConf's own syntax was "
            f"left behind. {fix}",
            group="hardware",
        )
    if not host:
        return probe.Check(label, probe.FAIL, f"{url or '(empty)'} — no host", fix, group="hardware")
    return probe.check_port(label, host, port, timeout=timeout, hint=hint)


def check_m2t2(url: str, *, timeout: float = 1.5) -> probe.Check:
    return _check_server(
        "m2t2 grasp server",
        url,
        example="http://localhost:8123",
        setting="planners.tiptop.perception.m2t2.url",
        hint="Start the M2T2 server; perception asks it for grasps every rollout. Another address: "
        "`tandem rig set planners.tiptop.perception.m2t2.url URL`.",
        timeout=timeout,
    )


def check_foundation_stereo(url: str, *, timeout: float = 1.5) -> probe.Check:
    return _check_server(
        "foundation stereo depth server",
        url,
        example="http://localhost:1234",
        setting="planners.tiptop.perception.foundation_stereo.url",
        hint="Start the FoundationStereo server (`pixi run server` in its checkout); TiPToP estimates the "
        "ZEDs' depth with it every rollout. Another address: "
        "`tandem rig set planners.tiptop.perception.foundation_stereo.url URL`.",
        timeout=timeout,
    )
