"""A sidecar as a process: its pipes, its process group, and the parent going away under it.

What a sidecar holds -- a robot connection, cameras -- is released only when its process, and every
process it started, has gone, and it stays responsive only while tandem keeps draining what it
prints. Each test here is one way that used to fail quietly:

- one byte that is not UTF-8 on its stderr ended tandem's drain, and ~64 KB later the sidecar blocked
  writing, answering nothing, holding the robot, with nothing in the log (the teleop driver's pipe too);
- a helper it started (a camera server, a recorder) outlived close(), and a crash, because only the
  sidecar itself was ever waited for;
- a sidecar that did not answer in time was left running, possibly still moving the arm;
- tandem dying mid-request made the sidecar's reply raise BrokenPipeError out of serve(), and the
  planner's close() -- what releases the hardware -- never ran.

Every sidecar here is a few lines on the real kit (``tandem_sidecar``), launched with this interpreter.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from dataclasses import replace
from pathlib import Path

import pytest
from helpers import wait_for
from toy_planner import TOY_CAPABILITIES

from tandem.core.settings import Settings
from tandem.executors.base import ExecutorContext
from tandem.executors.teleop import _ChildHost
from tandem.planners import LegSpec, PlannerInfo, SidecarPlanner, rpc, sidecar_kit
from tandem.planners.base import BackendError
from tandem.teleop import child as child_mod

SIDECAR = textwrap.dedent(
    r'''
    import os
    import subprocess
    import sys
    import time

    import tandem_sidecar
    from tandem_sidecar import log, serve

    HELPERS = []
    WARMS = []


    def warm(**_):
        WARMS.append(True)
        if "--helper" in sys.argv and not HELPERS:
            # A process of the sidecar's own, in its process group: a camera server, say.
            helper = subprocess.Popen(
                ["sleep", "300"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            HELPERS.append(helper.pid)
        return {}


    def helper():
        return HELPERS[0] if HELPERS else None


    def warms():
        return len(WARMS)


    def crash():
        os._exit(3)


    def slow(seconds):
        time.sleep(seconds)
        return {"slept": seconds}


    def bad_bytes():
        os.write(2, b"caf\xe9 \xff and on\n")
        return {}


    def burst(kb):
        line = b"x" * 99 + b"\n"
        for _ in range(kb * 10):
            os.write(2, line)
        os.write(2, b"the last line of the burst\n")
        return {"kb": kb}


    def perceive(**_):
        return {"scene_id": "s", "object_labels": [], "table_label": "floor"}


    def plan(**_):
        return {"ok": False, "failure_reason": "a stand-in plans nothing"}


    def execute(**_):
        return {"ok": False, "failure_reason": "a stand-in executes nothing"}


    if __name__ == "__main__":
        if "--bad-byte-before-hello" in sys.argv:
            os.write(tandem_sidecar._PROTOCOL_OUT.fileno(), b"\xff\xfe not json\n")
        verbs = [warm, helper, warms, crash, slow, bad_bytes, burst, perceive, plan, execute]
        raise SystemExit(serve({fn.__name__: fn for fn in verbs}))
    '''
)


@pytest.fixture
def script(tmp_path) -> Path:
    path = tmp_path / "sidecar.py"
    path.write_text(SIDECAR)
    return path


def _env() -> dict:
    return {**os.environ, "PYTHONPATH": str(sidecar_kit.DIRECTORY)}


@pytest.fixture
def channel(script):
    """A channel to the stand-in sidecar, with every log line it forwarded; stopped at the end."""
    made: list = []

    def open_(*flags: str):
        seen: list[tuple[str, str]] = []
        ch = rpc.HostedBackendChannel(
            [sys.executable, str(script), *flags],
            env=_env(),
            on_log=lambda stream, text: seen.append((stream, text)),
        ).start()
        ch.seen = seen
        made.append(ch)
        return ch

    yield open_
    for ch in made:
        ch.stop()


def _alive(pid: int) -> bool:
    """Whether ``pid`` is a live process -- a zombie waiting to be reaped is not."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout
    return bool(state.strip()) and not state.strip().startswith("Z")


@pytest.fixture
def helpers_killed():
    """Every helper pid a test registers is killed at the end, whatever the test found."""
    pids: list[int] = []
    yield pids
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


# --- the pipes ------------------------------------------------------------------------------------------


def test_a_byte_that_is_not_utf8_on_stderr_does_not_stop_the_drain(channel):
    ch = channel()
    ch.call("bad_bytes", timeout=10)
    # ~200 KB after it: several times a pipe's buffer. With the drain dead, the sidecar blocked writing
    # and this never came back.
    assert ch.call("burst", timeout=10, kb=200) == {"kb": 200}
    stderr = [text for stream, text in ch.seen if stream == "backend-stderr"]
    assert any(text.startswith("caf") and "\\xe9" in text and "and on" in text for text in stderr), stderr[:3]
    assert wait_for(lambda: "the last line of the burst" in [t for s, t in ch.seen if s == "backend-stderr"])


def test_a_byte_that_is_not_utf8_before_the_hello_is_logged_and_the_sidecar_starts(channel):
    ch = channel("--bad-byte-before-hello")
    assert ch.hello["ready"] is True
    assert any(stream == "backend" and "not json" in text for stream, text in ch.seen)
    assert ch.call("slow", timeout=10, seconds=0) == {"slept": 0}


def test_a_log_sink_that_fails_does_not_stop_the_drain(script):
    calls: list[str] = []

    def sink(stream, text):
        calls.append(text)
        if len(calls) == 1:
            raise RuntimeError("the log is full")

    ch = rpc.HostedBackendChannel([sys.executable, str(script)], env=_env(), on_log=sink).start()
    try:
        ch.call("bad_bytes", timeout=10)
        assert ch.call("burst", timeout=10, kb=200) == {"kb": 200}
    finally:
        ch.stop()


def test_a_late_answer_on_a_channel_still_running_is_not_read_as_the_next_reply(channel):
    ch = channel()
    with pytest.raises(rpc.BackendTimeout, match="did not answer within 0s"):
        ch.call("slow", timeout=0.3, seconds=1.0)
    # The channel itself leaves a slow child running -- what to do about it is its owner's call.
    assert ch.alive
    assert ch.call("burst", timeout=10, kb=0) == {"kb": 0}
    assert any("discarding a late reply" in text for _, text in ch.seen)


def test_the_teleop_drivers_output_is_drained_past_a_byte_that_is_not_utf8(tmp_path, monkeypatch):
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import os, sys, time\n"
        "os.write(1, b'caf\\xe9 \\xff and on\\n')\n"
        "line = b'x' * 99 + b'\\n'\n"
        "for _ in range(2000):\n"
        "    os.write(1, line)\n"
        "os.write(1, b'the last line of the burst\\n')\n"
        "time.sleep(30)\n"
    )
    from tandem import teleop as teleop_pkg

    monkeypatch.setattr(teleop_pkg, "driver_path", lambda: driver)
    seen: list[tuple[str, str]] = []
    profile = type("P", (), {"cameras": type("C", (), {"configured": staticmethod(dict)})()})()
    host = _ChildHost(
        ExecutorContext(profile=profile, session_dir=tmp_path, on_log=lambda s, t: seen.append((s, t))),
        LegSpec(trajectory_id="t-1", instruction="x", segment_source="teleop"),
        tmp_path / "legs",
        tmp_path / "scratch",
    )
    (tmp_path / "scratch").mkdir()
    settings = Settings()
    settings.teleop.enabled = True
    settings.teleop.python = sys.executable
    settings.teleop.droid_dir = str(tmp_path)
    teleop = child_mod.TeleopChild(host, settings).start()
    try:
        assert wait_for(lambda: ("teleop", "the last line of the burst") in seen, timeout=10), seen[:3]
        assert any(stream == "teleop" and "\\xe9" in text for stream, text in seen)
    finally:
        teleop.kill()


# --- the process group --------------------------------------------------------------------------------------


@pytest.fixture
def helper_planner(script):
    """A SidecarPlanner over the stand-in sidecar, which starts a helper process when it warms."""

    class WithHelper(SidecarPlanner):
        info = PlannerInfo(name="with-helper")
        CAPABILITIES = replace(TOY_CAPABILITIES, name="with-helper", supports_cooperative_stop=False)
        SIDECAR = str(script)
        TIMEOUTS = {"slow": 0.5}

        def launch_command(self) -> list[str]:
            return [*super().launch_command(), "--helper"]

    made: list = []

    def make():
        planner = WithHelper(None, on_log=lambda stream, text: None)
        made.append(planner)
        return planner

    yield make
    for planner in made:
        planner.close()


def test_close_ends_what_the_sidecar_started_as_well_as_the_sidecar(helper_planner, helpers_killed):
    planner = helper_planner()
    planner.warm()
    helper = planner.call("helper")
    helpers_killed.append(helper)
    assert _alive(helper)
    planner.close()
    # The sidecar quit when asked, within EXIT_GRACE -- and its helper used to be left holding the camera.
    assert wait_for(lambda: not _alive(helper), timeout=15)


def test_a_crashed_sidecars_helpers_are_gone_before_its_replacement_starts(helper_planner, helpers_killed):
    planner = helper_planner()
    planner.warm()
    first = planner._channel.hello["pid"]
    helper = planner.call("helper")
    helpers_killed.append(helper)
    with pytest.raises(BackendError, match=r"exited \(code 3\)"):
        planner.call("crash")
    planner.warm()  # the relaunch path
    helpers_killed.append(planner.call("helper"))
    assert planner._channel.hello["pid"] != first
    assert wait_for(lambda: not _alive(helper), timeout=15), "the old sidecar's helper still holds the camera"


def test_close_after_a_crash_ends_what_the_crashed_sidecar_started(helper_planner, helpers_killed):
    planner = helper_planner()
    planner.warm()
    helper = planner.call("helper")
    helpers_killed.append(helper)
    with pytest.raises(BackendError, match=r"exited \(code 3\)"):
        planner.call("crash")
    planner.close()
    assert wait_for(lambda: not _alive(helper), timeout=15)


def test_a_sidecar_that_does_not_answer_in_time_is_stopped_with_everything_it_started(
    helper_planner, helpers_killed
):
    planner = helper_planner()
    planner.warm()
    sidecar = planner._channel.hello["pid"]
    helper = planner.call("helper")
    helpers_killed.append(helper)
    with pytest.raises(rpc.BackendTimeout, match="It was stopped, with every process it started"):
        planner.call("slow", seconds=30)
    assert wait_for(lambda: not _alive(sidecar) and not _alive(helper), timeout=15)
    with pytest.raises(BackendError, match="not running"):
        planner.call("helper")
    planner.warm()
    helpers_killed.append(planner.call("helper"))
    assert planner._channel.hello["pid"] != sidecar


def test_warming_a_sidecar_that_is_running_and_warm_does_nothing(helper_planner, helpers_killed):
    # tandem warms again after a verb raised; a sidecar still up must not open its hardware twice.
    planner = helper_planner()
    planner.warm()
    pid = planner._channel.hello["pid"]
    helpers_killed.append(planner.call("helper"))
    planner.warm()
    assert planner._channel.hello["pid"] == pid
    assert planner.call("warms") == 1, "the sidecar was warmed twice"


# --- the parent going away ----------------------------------------------------------------------------------

PARENT_GONE = textwrap.dedent(
    """
    import sys
    import time
    from pathlib import Path

    from tandem_sidecar import log, serve

    MARKER = Path(sys.argv[1])


    def execute(**_):
        time.sleep(0.5)
        log("executed")
        return {"ok": True, "n_frames": 3}


    def close():
        log("releasing the robot")  # nobody reads it any more; it must not stop what follows
        print("and a print of the planner's own")
        MARKER.write_text("released")


    if __name__ == "__main__":
        raise SystemExit(serve({"execute": execute, "close": close}))
    """
)


def _launch_parent_gone(tmp_path: Path) -> tuple[subprocess.Popen, Path]:
    script = tmp_path / "gone.py"
    script.write_text(PARENT_GONE)
    marker = tmp_path / "released"
    proc = subprocess.Popen(
        [sys.executable, str(script), str(marker)],
        env=_env(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    hello = json.loads(proc.stdout.readline())
    assert hello["ready"] is True
    return proc, marker


def _hang_up(proc: subprocess.Popen) -> None:
    """What tandem dying looks like from the sidecar: every pipe to it closes."""
    for stream in (proc.stdout, proc.stderr, proc.stdin):
        stream.close()


def test_tandem_dying_mid_request_still_runs_the_planners_close(tmp_path):
    proc, marker = _launch_parent_gone(tmp_path)
    proc.stdin.write(json.dumps({"id": 1, "verb": "execute", "args": {}}).encode() + b"\n")
    proc.stdin.flush()
    time.sleep(0.1)  # the handler is running
    _hang_up(proc)
    assert proc.wait(timeout=15) == 0
    assert marker.read_text() == "released"


def test_tandem_dying_between_requests_still_runs_the_planners_close(tmp_path):
    proc, marker = _launch_parent_gone(tmp_path)
    _hang_up(proc)
    assert proc.wait(timeout=15) == 0
    assert marker.read_text() == "released"
