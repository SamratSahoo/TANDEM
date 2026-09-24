"""A session collects on this machine's rig: its cameras, and the planner's machine settings, come from rig.yml.

The profile is the task; the session reads the rig once, when it starts, and every part of it that
touches hardware -- the planner through its BackendContext, the teleop driver through the executor's
context -- is handed that one rig. A rig a session could not record from is refused before anything is
warmed, with the command that fixes it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from helpers import use_fake_backend

from tandem.core import rig as rig_mod
from tandem.core.errors import TandemError
from tandem.core.session import Session
from tandem.core.settings import Settings
from tandem.executors.base import ExecutorContext
from tandem.executors.teleop import _ChildHost
from tandem.planners.base import LegSpec
from tandem.teleop import child as child_mod


def test_a_session_on_a_rig_with_no_cameras_is_refused_with_the_way_to_add_them(profile, monkeypatch):
    use_fake_backend(monkeypatch)
    rig_mod.update({"cameras.hand": None, "cameras.external": None})
    with pytest.raises(TandemError) as caught:
        Session(profile, task="x").start()
    assert "rig has no cameras configured" in caught.value.message
    assert "tandem rig set cameras.external.serial" in caught.value.hint


def test_a_session_whose_perception_camera_is_not_there_is_refused(profile, monkeypatch):
    use_fake_backend(monkeypatch)
    rig_mod.update({"cameras.external": None})
    with pytest.raises(TandemError) as caught:
        Session(profile, task="x").start()
    assert "cameras.perception is 'external' but cameras.external is not configured" in caught.value.message
    assert "tandem rig set cameras.perception" in caught.value.hint


def test_a_session_reads_the_rig_when_it_starts_and_hands_it_to_the_planner(profile, monkeypatch):
    use_fake_backend(monkeypatch)
    from tandem.planners import registry

    session = Session(profile, task="x")
    session.start()
    try:
        factory = registry.factory("tiptop")
        (ctx,) = factory.contexts
        assert ctx.rig is session.rig
        assert ctx.rig.cameras.configured().keys() == {"hand", "external"}
        assert session._executor_context().rig is session.rig
    finally:
        session.stop(park=False)
        session.wait(timeout=5)


def test_a_rig_given_to_the_session_is_the_one_it_uses(profile, monkeypatch):
    use_fake_backend(monkeypatch)
    given = rig_mod.parse_text("cameras:\n  external: {serial: '555'}\n", source="elsewhere.yml")
    session = Session(profile, task="x", rig=given)
    session.start()
    try:
        assert session.rig is given
    finally:
        session.stop(park=False)
        session.wait(timeout=5)


def _launch(host, tmp_path, monkeypatch) -> list[str]:
    cfg = Settings()
    cfg.teleop.enabled = True
    cfg.teleop.python = sys.executable
    cfg.teleop.droid_dir = str(tmp_path)
    launched: list[list[str]] = []

    class _Proc:
        stdout = ()  # the executor drains it on a thread of its own: nothing to drain

    monkeypatch.setattr(child_mod.subprocess, "Popen", lambda args, **kwargs: launched.append(list(args)) or _Proc())
    monkeypatch.setattr(child_mod.events_mod, "EventTailer", lambda *a, **k: SimpleNamespace(start=lambda: None))
    child_mod.TeleopChild(host, cfg).start()
    return launched[0][2:]


def test_the_teleop_driver_records_from_the_rigs_cameras(profile, machine_rig, tmp_path, monkeypatch):
    rig = rig_mod.update({"cameras.external_2.serial": "31425515"})
    ctx = ExecutorContext(profile=profile, session_dir=tmp_path, rig=rig)
    leg = LegSpec(trajectory_id="traj-1", instruction="fold it", segment_source="teleop")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    argv = _launch(_ChildHost(ctx, leg, tmp_path / "legs", scratch), tmp_path, monkeypatch)
    for flag, serial in (
        ("--hand-camera-id", "14846828"),
        ("--external-camera-id", "32439448"),
        ("--external-2-camera-id", "31425515"),
    ):
        assert argv[argv.index(flag) + 1] == serial, flag
    assert argv[argv.index("--output-root") + 1] == str(tmp_path / "legs")


def test_an_executor_context_without_a_rig_reads_this_machines(profile, machine_rig, tmp_path, monkeypatch):
    ctx = ExecutorContext(profile=profile, session_dir=Path(tmp_path))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    host = _ChildHost(ctx, LegSpec(trajectory_id="t"), tmp_path / "legs", scratch)
    assert host.profile.cameras.configured().keys() == {"hand", "external"}
