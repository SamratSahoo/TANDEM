"""The teleop half of a hand-off: does the human's demonstration actually get recorded?

The driver records NOTHING until it is told to. It parks at a task prompt and drops every stdin line
that is not ``{"cmd":"start"}`` — ``end_and_quit`` included — so a hand-off that never sends one
produces an episode with no frames in it AND a driver that never exits, which then times out
"return control" sixty seconds later.

That is not hypothetical: it was the behaviour of every hand-off tandem ran, and the suite stayed
green because the planner's stand-in driver was more capable than the real one. These tests drive
the real ``TeleopChild`` against a stand-in that reproduces the driver's actual gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from helpers import wait_for

from tandem.core.settings import Settings
from tandem.teleop import child as child_mod

FAKE_TELEOP = Path(__file__).parent / "fake_teleop.py"


class _StubSession:
    """Just enough Session for TeleopChild: a scratch dir, a task, and somewhere to log."""

    def __init__(self, tmp_path: Path, trajectory_dir: Path) -> None:
        self.id = "stub"
        self.task = "fold the cloth over the toy"
        # The language label the episode carries, which is not always the goal the planner is given.
        self.instruction = "fold the cloth over the toy"
        # Minted by the session, and what the hand-off must stamp the leg with. Reading it back off
        # a leg's _meta.json instead is what left the demonstration orphaned.
        self._trajectory_id = "traj-abc123"
        self.handoff_error: str | None = None
        self.logs: list[str] = []
        self.events: list[dict] = []
        self._files = {"session_dir": tmp_path}
        self._trajectory_dir = trajectory_dir
        self.current = type("R", (), {"dir": str(trajectory_dir)})()
        self.profile = type("P", (), {
            "trajectories_dir": lambda _self=None: trajectory_dir,
            "cameras": type("C", (), {"configured": staticmethod(lambda: {})})(),
        })()

    def _log(self, stream: str, text: str) -> None:
        self.logs.append(text)

    def _emit(self, payload: dict) -> None:
        self.events.append(payload)

    def _pump(self, stream, name: str) -> None:
        # The real one starts a thread; draining is enough here and keeps the child from blocking.
        import threading

        threading.Thread(target=lambda: [None for _ in stream], daemon=True).start()


def _settings(extra: list[str] | None = None) -> Settings:
    cfg = Settings()
    cfg.teleop.enabled = True
    cfg.teleop.python = sys.executable
    cfg.teleop.droid_dir = str(Path(__file__).parent)
    return cfg


@pytest.fixture
def child(tmp_path, monkeypatch):
    """A TeleopChild wired to the stand-in driver instead of the DROID one."""
    trajectories = tmp_path / "trajectories"
    trajectories.mkdir()
    stub = _StubSession(tmp_path, trajectories)

    from tandem import teleop as teleop_pkg

    monkeypatch.setattr(teleop_pkg, "driver_path", lambda: FAKE_TELEOP)
    made = []

    def build(*extra_args):
        original = child_mod.TeleopChild.start

        def start(self):
            self._extra = list(extra_args)
            return original(self)

        monkeypatch.setattr(child_mod.TeleopChild, "start", start)
        instance = child_mod.TeleopChild(stub, _settings())
        # The stand-in takes one extra flag; append it to the argv the real code built.
        real_popen = child_mod.subprocess.Popen

        def popen(args, **kwargs):
            return real_popen([*args, *extra_args], **kwargs)

        monkeypatch.setattr(child_mod.subprocess, "Popen", popen)
        made.append(instance.start())
        return made[-1], stub

    yield build
    for instance in made:
        instance.kill()


def test_a_handoff_leg_is_actually_recorded(child):
    teleop, stub = child()

    # The driver announces its prompt; tandem must answer it, or nothing is ever recorded.
    assert wait_for(lambda: teleop._recording), "tandem never told the driver to start recording"

    teleop.finish()
    assert teleop.wait(timeout=20.0), "the driver did not exit after `end_and_quit`"
    assert teleop.n_frames and teleop.n_frames >= 2, "the leg saved no frames"

    leg = Path(teleop.leg_dir)
    meta = json.loads((leg / "_meta.json").read_text())
    assert meta["segment_source"] == "teleop"
    assert meta["trajectory_id"] == "traj-abc123", "the leg must carry the trajectory it belongs to"
    assert stub.handoff_error is None


def test_finishing_before_anything_was_recorded_still_ends_the_driver(child):
    """`end_and_quit` is only honoured MID-RECORDING — at the prompt the driver drops it.

    So a hand-off the operator ends immediately has to be closed with `q` instead, or the process
    lives on holding the robot and both cameras and `resume_from_teleop` raises after its grace.
    """
    teleop, _ = child()
    assert wait_for(lambda: teleop._recording)

    # Put it back at the prompt, then end the session from there.
    teleop.discard()
    assert wait_for(lambda: not teleop._recording)
    teleop._start_attempts = teleop.MAX_START_ATTEMPTS  # stop it being handed another episode

    teleop.finish()
    assert teleop.wait(timeout=20.0), "the driver did not exit when ended from its task prompt"


def test_a_controller_that_does_not_wake_is_retried_then_reported(child):
    """The driver's own remedy is "wake it and press Start again", so retrying is right.

    Bounded, though: a headset left in a drawer would otherwise spin the person waiting for the arm
    in a loop with nothing on screen and no way out.
    """
    teleop, stub = child("--refuse-first")

    assert wait_for(lambda: teleop._recording), "the retry should have got there on the second go"
    assert any("VR controller not detected" in text for text in stub.logs)
    assert teleop._start_attempts == 2

    teleop.finish()
    assert teleop.wait(timeout=20.0)
    assert teleop.n_frames and teleop.n_frames >= 2


def test_the_leg_is_stamped_even_when_no_planner_leg_was_recorded_first(child, monkeypatch):
    """The id comes from the session, not from a leg's ``_meta.json``.

    Reading it back off disk was correct only while the PLANNER minted it. Now tandem does, and
    there are three ordinary ways to reach a hand-off with nothing on disk to read: a plan whose
    first phase is the person's, an operator-requested hand-off on the first leg, and — the default
    policy — a phase the planner could not plan, where a rollout directory exists but nothing was
    ever recorded into it.

    An unstamped leg is never merged, so the demonstration is orphaned; and the driver, seeing no
    trajectory id, prompts for a verdict nobody answers and holds the robot until it is killed.
    """
    teleop, stub = child()
    # Exactly the state a hand-off before any recorded leg is in.
    stub.current = None

    teleop.kill()
    teleop.wait(timeout=5.0)
    fresh = child_mod.TeleopChild(stub, _settings()).start()
    try:
        assert wait_for(lambda: fresh._recording)
        fresh.finish()
        assert fresh.wait(timeout=20.0)
        meta = json.loads((Path(fresh.leg_dir) / "_meta.json").read_text())
        assert meta["trajectory_id"] == "traj-abc123"
    finally:
        fresh.kill()


def test_an_unstamped_leg_would_block_the_driver_and_is_answered(child, monkeypatch):
    """Belt and braces for the case above.

    A leg with no trajectory id is not a leg of anything, so the driver asks for a success/failure
    verdict — and blocks there, holding the robot and both cameras, because nothing in a hand-off
    ever sends one. tandem answers it so the arm comes back instead of being taken back by force.
    """
    teleop, stub = child()
    teleop.kill()
    teleop.wait(timeout=5.0)

    stub._trajectory_id = ""
    stub.current = None
    fresh = child_mod.TeleopChild(stub, _settings()).start()
    try:
        assert wait_for(lambda: fresh._recording)
        fresh.finish()
        # Without the answer this wait would burn the full grace and end in a kill.
        assert fresh.wait(timeout=20.0), "the driver blocked at its label prompt"
        assert any("asked for a success/failure label" in text for text in stub.logs)
    finally:
        fresh.kill()
