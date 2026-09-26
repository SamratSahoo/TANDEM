"""TiPToP's perception servers: built as runtimes, started when a session needs them, stopped when it ends.

Nothing is built and no model is loaded: ``/health`` is answered by a stand-in HTTP server, and starting a
server is a stand-in process. What is checked is the policy -- which servers are started, when, and which
are stopped at the end -- and that tandem reaches them only through the registry.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from tandem.cli.app import app
from tandem.core.errors import TandemError
from tandem.planners import registry
from tandem.planners.tiptop import servers


@pytest.fixture
def health():
    """A /health endpoint on a free local port; set ``.status`` to what it answers."""

    class Handler(BaseHTTPRequestHandler):
        status = "healthy"

        def do_GET(self):  # noqa: N802
            body = json.dumps({"status": Handler.status}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield SimpleNamespace(url=f"http://localhost:{httpd.server_port}", handler=Handler)
    httpd.shutdown()


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("TANDEM_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("TANDEM_RUNTIMES_DIR", str(tmp_path / "runtimes"))
    monkeypatch.setenv("TANDEM_CONFIG_DIR", str(tmp_path / "config"))


def test_both_servers_are_pinned_to_samrats_forks_and_built_by_tandem_servers_install():
    for server in servers.SERVERS:
        pin = server.recipe.sources[0].pin
        assert pin.url.startswith("https://github.com/SamratSahoo/") and pin.ref == "TANDEM"
        assert len(pin.commit) == 40
        assert server.recipe.build_command == "tandem servers install"
    assert [service.name for service in registry.services("tiptop")] == ["m2t2", "foundation_stereo"]


def test_only_a_server_on_this_machine_is_one_tandem_may_start():
    assert servers.is_local("http://localhost:8123")
    assert servers.is_local("http://127.0.0.1:1234")
    assert not servers.is_local("http://gpu-box.example.org:8123")
    assert servers.port_of("http://localhost:1234") == 1234


def test_healthy_means_the_model_is_loaded(health):
    assert servers.healthy(health.url)
    health.handler.status = "unconfigured"  # FoundationStereo with no weights answers, but cannot serve
    assert not servers.healthy(health.url)
    assert not servers.healthy("http://localhost:9")  # nothing listening


def test_a_server_already_answering_or_on_another_machine_is_never_started(health, monkeypatch):
    started: list[str] = []
    monkeypatch.setattr(servers, "start", lambda server, url, settings=None: started.append(server.name))
    manager = servers.ServerManager(
        {"m2t2": health.url, "foundation_stereo": "http://gpu-box.example.org:1234"}, log=lambda _t: None
    )
    manager.ensure(timeout=1)
    assert started == []


def test_a_local_server_that_is_not_installed_is_left_to_tiptops_own_warm_up_message(monkeypatch):
    said: list[str] = []
    monkeypatch.setattr(servers, "start", lambda *a, **k: pytest.fail("started without a runtime"))
    manager = servers.ServerManager({"m2t2": "http://localhost:9"}, log=said.append)
    manager.ensure(timeout=1)
    assert any("tandem servers install" in line for line in said)


def test_a_down_server_is_started_waited_for_and_stopped_only_if_this_session_started_it(health, monkeypatch):
    health.handler.status = "loading"
    stopped: list[str] = []

    class Proc:
        returncode = None

        def poll(self):
            return None

    def start(server, url, settings=None):
        # The model finishes loading a moment after launch.
        threading.Timer(0.3, lambda: setattr(health.handler, "status", "healthy")).start()
        return Proc()

    monkeypatch.setattr(servers.RecipeRuntime, "is_ready", lambda self: True)
    monkeypatch.setattr(servers, "start", start)
    monkeypatch.setattr(servers, "stop", lambda name: stopped.append(name) or True)
    manager = servers.ServerManager({"m2t2": health.url}, log=lambda _t: None)
    manager.ensure(timeout=10)
    assert servers.healthy(health.url)
    manager.ensure(timeout=1)  # up now: nothing more to do
    manager.stop_started()
    assert stopped == ["m2t2"]


def test_a_server_that_exits_while_starting_says_where_its_log_is(monkeypatch):
    class Dead:
        returncode = 1

        def poll(self):
            return 1

    monkeypatch.setattr(servers.RecipeRuntime, "is_ready", lambda self: True)
    monkeypatch.setattr(servers, "start", lambda *a, **k: Dead())
    manager = servers.ServerManager({"m2t2": "http://localhost:9"}, log=lambda _t: None)
    with pytest.raises(TandemError) as caught:
        manager.ensure(timeout=5)
    assert "exited with code 1" in caught.value.message
    assert "server-m2t2.log" in caught.value.hint


def test_stop_ends_the_process_group_tandem_started(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    servers.pid_file("m2t2").parent.mkdir(parents=True, exist_ok=True)
    servers.pid_file("m2t2").write_text(str(proc.pid))
    assert servers.running_pid("m2t2") == proc.pid
    assert servers.stop("m2t2", grace=5)
    proc.wait(timeout=10)
    assert servers.running_pid("m2t2") is None
    assert not servers.stop("m2t2"), "nothing left to stop"


def test_the_backend_brings_the_servers_up_before_it_warms_and_stops_its_own_at_close(monkeypatch):
    from tandem.planners.sidecar import SidecarPlanner
    from tandem.planners.tiptop.backend import TiptopBackend

    order: list[str] = []

    class Manager:
        def __init__(self, urls, *, log, settings=None):
            pass

        def ensure(self):
            order.append("ensure")

        def stop_started(self):
            order.append("stop_started")

    monkeypatch.setattr(servers, "ServerManager", Manager)
    monkeypatch.setattr(SidecarPlanner, "warm", lambda self: order.append("warm"))
    monkeypatch.setattr(SidecarPlanner, "close", lambda self: order.append("close"))
    backend = TiptopBackend(
        SimpleNamespace(root=Path("/nonexistent")),
        env={},
        output_dir=Path("/tmp"),
        perception_urls={"m2t2": "http://localhost:8123"},
    )
    backend.warm()
    backend.close()
    assert order == ["ensure", "warm", "close", "stop_started"]


def test_servers_status_lists_both_from_the_rig():
    result = CliRunner().invoke(app, ["servers", "status", "--json"])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    assert [(row["name"], row["url"]) for row in rows] == [
        ("m2t2", "http://localhost:8123"),
        ("foundation_stereo", "http://localhost:1234"),
    ]
    assert all(row["local"] and not row["installed"] for row in rows)
