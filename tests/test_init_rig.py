"""`tandem init`: the robot and cameras go to the rig, and the paper's five tasks become the profiles.

    tandem init [--robot-host HOST] [--robot-type TYPE] [--camera ROLE=SERIAL]... [--profile NAME]

At a terminal it asks for the robot's address (the NUC), the arm and each camera's serial; without one (or
with --yes) the flags say what it would have asked. The planner's runtime build is a stand-in here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tandem.cli import init as init_cli
from tandem.cli.app import app
from tandem.core import paths, probe, profiles, zed
from tandem.core import rig as rig_mod
from tandem.core import settings as settings_mod


@pytest.fixture(autouse=True)
def machine(monkeypatch):
    """A workstation whose checks pass and whose runtime builds instantly, recording each build."""
    builds: list[dict] = []
    monkeypatch.setattr(init_cli, "_preflight", lambda viz_only: [])
    monkeypatch.setattr(init_cli, "_planner_preflight", lambda planner, **kw: None)
    monkeypatch.setattr(probe, "find_pixi", lambda: Path("/opt/pixi"))
    monkeypatch.setattr("tandem.cli.runtime.run_build", lambda rt, **kw: builds.append(kw))
    # No ZED SDK here: the listing knows nothing, and each serial is typed.
    monkeypatch.setattr(zed, "detect", lambda pythons: None)
    return builds


def _run(*args: str):
    return CliRunner().invoke(app, ["init", *args])


def _said(result) -> str:
    return " ".join(result.output.split())


# --------------------------------------------------------------------------- the rig, from flags


def test_the_flags_write_the_rig_and_leave_the_planners_defaults_its_own(machine):
    result = _run(
        "-y", "--robot-host", "10.0.0.5", "--robot-type", "panda_robotiq", "--camera", "hand=111", "--camera",
        "external=222",
    )
    assert result.exit_code == 0, result.output
    rig = rig_mod.load(force=True)
    assert (rig.robot.host, rig.robot.type) == ("10.0.0.5", "panda_robotiq")
    assert {role: cam.serial for role, cam in rig.cameras.configured().items()} == {"hand": "111", "external": "222"}
    # TiPToP's machine settings are not written: its defaults stay its own, so the file is the few lines a
    # person set, and a later default reaches this machine. They are still what it runs with.
    assert rig.planners == {}
    tiptop = rig_mod.planner_options(rig, "tiptop")
    assert tiptop["perception"]["m2t2"]["url"] == "http://localhost:8123"
    assert tiptop["perception"]["foundation_stereo"]["url"] == "http://localhost:1234"
    text = paths.rig_file().read_text()
    assert "gemini" not in text and "depth_trunc" not in text and len(text.splitlines()) < 30
    assert "serial: '111'" in text, "a serial is written as text, so it reads back as one"
    assert json.loads(rig.calibration_file().read_text()) == {}
    assert "Rig: panda_robotiq at 10.0.0.5" in _said(result)
    assert "0 of 2 with extrinsics in" in _said(result)
    # TiPToP's runtime, then its two perception servers' (M2T2, FoundationStereo).
    assert len(machine) == 3, "the runtime and both perception servers were built"
    assert machine[0].get("planner") == "tiptop"


def test_a_rig_is_written_even_with_nothing_given_and_says_what_it_lacks():
    result = _run("-y")
    assert result.exit_code == 0, result.output
    assert rig_mod.exists()
    rig = rig_mod.load(force=True)
    assert rig.robot.host == "172.16.0.2" and not rig.cameras.configured()
    assert "No cameras yet" in _said(result) and "tandem rig set cameras.hand.serial SERIAL" in _said(result)


def test_a_rig_already_there_keeps_what_it_says_and_takes_what_the_flags_change():
    rig_mod.update({"robot.host": "10.1.1.1", "cameras.hand.serial": "111", "planners.tiptop.robot.port": 6000})
    result = _run("-y", "--camera", "external=222")
    assert result.exit_code == 0, result.output
    rig = rig_mod.load(force=True)
    assert rig.robot.host == "10.1.1.1" and rig.cameras.hand.serial == "111"
    assert rig.cameras.external.serial == "222"
    assert rig.planners["tiptop"]["robot"]["port"] == 6000, "its planner block is the user's, never re-defaulted"


@pytest.mark.parametrize(
    ("flags", "said"),
    [
        (["--camera", "wrist=111"], "'wrist' is not a camera role"),
        (["--camera", "externall=111"], "did you mean 'external'?"),
        (["--camera", "hand"], "is not ROLE=SERIAL"),
        (["--robot-host", "http://10.0.0.5"], "must be a hostname or IP address"),
        (["--robot-host", "10.0.0.5:5555"], "no port"),
    ],
)
def test_a_bad_flag_is_refused_before_anything_is_built(machine, flags, said):
    result = _run("-y", *flags)
    assert result.exit_code == 1
    assert said in result.exception.message
    assert machine == [] and not rig_mod.exists() and profiles.list_names() == []


def test_a_laptop_writes_no_rig():
    result = _run("--viz-only", "-y", "--robot-host", "10.0.0.5")
    assert result.exit_code == 0, result.output
    assert not rig_mod.exists()
    assert "--robot-host, --robot-type and --camera were not applied" in _said(result)
    assert "rig none (visualization only)" in _said(result)
    rig_mod.update({"robot.host": "10.0.0.5"})
    again = _run("--viz-only", "-y")
    assert "rig none" not in _said(again), "a rig that is there is shown, not said to be missing"


def test_the_flags_that_are_gone_are_refused():
    for flags in (["--preset", "paper"], ["--import-from", "/tmp"], ["--tamp-config", "x.yml"]):
        result = _run("-y", *flags)
        assert result.exit_code == 2 and "No such option" in result.output


# --------------------------------------------------------------------------- the rig, asked


class Answers:
    """typer.prompt and typer.confirm at a terminal, answered from a script; records each question."""

    def __init__(self, answers: dict[str, list[str]]) -> None:
        self.answers = {key: list(values) for key, values in answers.items()}
        self.asked: list[tuple[str, object]] = []

    def prompt(self, text, default=None, **_kw):
        self.asked.append((text.strip(), default))
        for key, values in self.answers.items():
            if key in text and values:
                return values.pop(0)
        return "" if default is None else default

    def confirm(self, text, default=False, **_kw):
        self.asked.append((text.strip(), default))
        return "Build it now" in text


def _at_a_terminal(monkeypatch, answers: dict[str, list[str]]) -> Answers:
    script = Answers(answers)
    monkeypatch.setattr(init_cli.theme, "is_tty", lambda: True)
    monkeypatch.setattr(init_cli.typer, "prompt", script.prompt)
    monkeypatch.setattr(init_cli.typer, "confirm", script.confirm)
    return script


def test_at_a_terminal_it_asks_for_the_nuc_and_each_camera(monkeypatch):
    script = _at_a_terminal(
        monkeypatch,
        {
            # A bad address is asked again, not written.
            "Robot address": ["http://172.16.0.9", "172.16.0.9"],
            "Wrist camera serial": ["14846828"],
            "External camera serial": ["32439448"],
            "Second external camera serial": ["none"],
            "perception": ["hand"],
        },
    )
    result = _run()
    assert result.exit_code == 0, result.output
    rig = rig_mod.load(force=True)
    assert rig.robot.host == "172.16.0.9" and rig.robot.type == "fr3_robotiq"
    assert {role: cam.serial for role, cam in rig.cameras.configured().items()} == {
        "hand": "14846828",
        "external": "32439448",
    }
    assert rig.cameras.perception == "hand"
    questions = [text for text, _ in script.asked]
    assert questions.count("Robot address (the NUC)") == 2
    assert dict(script.asked)["Arm type"] == "fr3_robotiq", "each question defaults to what the rig says now"
    assert "must be a hostname or IP address" in _said(result)


def test_at_a_terminal_a_rig_already_set_up_is_asked_again_only_with_repair(monkeypatch):
    rig_mod.update({"robot.host": "10.1.1.1", "cameras.hand.serial": "111"})
    script = _at_a_terminal(monkeypatch, {"Robot address": ["10.2.2.2"]})
    assert _run().exit_code == 0
    assert not any("Robot address" in text for text, _ in script.asked)
    assert rig_mod.load(force=True).robot.host == "10.1.1.1"

    result = _run("--repair")
    assert result.exit_code == 0, result.output
    assert ("Robot address (the NUC)", "10.1.1.1") in script.asked
    assert ("Wrist camera serial ('none' if there is none)", "111") in script.asked
    assert rig_mod.load(force=True).robot.host == "10.2.2.2"
    assert rig_mod.load(force=True).cameras.hand.serial == "111", "Enter keeps what was there"


def test_the_cameras_the_zed_sdk_lists_are_offered_to_confirm(monkeypatch):
    found = [
        zed.ZedCamera("32439448", "ZED 2i", "AVAILABLE"),
        zed.ZedCamera("14846828", "ZED-M", "AVAILABLE"),
        zed.ZedCamera("31425515", "ZED 2i", "AVAILABLE"),
    ]
    monkeypatch.setattr(zed, "detect", lambda pythons: found)
    # Enter at every camera question: the suggestions stand.
    script = _at_a_terminal(monkeypatch, {"Robot address": ["172.16.0.2"]})
    result = _run()
    assert result.exit_code == 0, result.output
    asked = dict(script.asked)
    # The Mini is offered for the wrist; the others, in the SDK's order, for the external roles.
    assert asked["Wrist camera serial ('none' if there is none)"] == "14846828"
    assert asked["External camera serial ('none' if there is none)"] == "32439448"
    assert asked["Second external camera serial ('none' if there is none)"] == "31425515"
    cams = rig_mod.load(force=True).cameras.configured()
    assert {role: cam.serial for role, cam in cams.items()} == {
        "hand": "14846828",
        "external": "32439448",
        "external_2": "31425515",
    }
    assert "The ZED SDK sees 3 camera(s)" in _said(result)


def test_a_typed_serial_overrides_the_suggestion_and_one_camera_fills_one_role(monkeypatch):
    found = [zed.ZedCamera("14846828", "ZED-M"), zed.ZedCamera("32439448", "ZED 2i")]
    monkeypatch.setattr(zed, "detect", lambda pythons: found)
    _at_a_terminal(
        monkeypatch,
        {
            "Wrist camera serial": ["55555555"],
            # The wrist's serial again is refused and asked again.
            "External camera serial": ["55555555", "32439448"],
            "Second external camera serial": ["none"],
        },
    )
    result = _run()
    assert result.exit_code == 0, result.output
    cams = rig_mod.load(force=True).cameras.configured()
    assert {role: cam.serial for role, cam in cams.items()} == {"hand": "55555555", "external": "32439448"}
    said = _said(result)
    assert "55555555 is already the hand camera" in said
    assert "55555555 is not connected right now" in said


# --------------------------------------------------------------------------- the profiles


def test_init_adds_the_papers_five_and_makes_the_first_active():
    result = _run("-y")
    assert result.exit_code == 0, result.output
    assert profiles.list_names() == sorted(profiles.BUILTIN)
    assert settings_mod.load(force=True).active_profile == "cover-bread-rolls"
    assert "Added the paper's five tasks" in _said(result)
    assert "profile cover-bread-rolls" in _said(result)
    assert not profiles.exists("default"), "no profile of init's own any more"


def test_a_re_run_keeps_edits_brings_back_a_deleted_task_and_keeps_the_active_profile():
    assert _run("-y").exit_code == 0
    path = profiles.path_of("sort-and-cover-snacks")
    path.write_text(path.read_text().replace("target_episodes: 20", "target_episodes: 33"))
    profiles.delete("open-obstructed-book")
    profiles.create("mine", prompt="stack the cups")
    cfg = settings_mod.load()
    cfg.active_profile = "mine"
    settings_mod.save(cfg)

    result = _run("-y")
    assert result.exit_code == 0, result.output
    assert "Added the paper's open-obstructed-book" in _said(result)
    assert profiles.load("sort-and-cover-snacks").task.target_episodes == 33
    assert settings_mod.load(force=True).active_profile == "mine"


def test_profile_makes_one_active_and_must_exist(machine):
    result = _run("-y", "--profile", "store-bread-in-closed-box")
    assert result.exit_code == 0, result.output
    assert settings_mod.load(force=True).active_profile == "store-bread-in-closed-box"

    builds = len(machine)
    missing = _run("-y", "--profile", "store-bread-in-closed-bx")
    assert missing.exit_code == 1
    assert "Did you mean 'store-bread-in-closed-box'?" in missing.exception.hint
    assert "tandem profile create store-bread-in-closed-bx" in missing.exception.hint
    assert len(machine) == builds, "said before the runtime step"


def test_the_rig_flags_are_written_before_the_planners_checks_can_stop_init(monkeypatch):
    """A missing GPU driver stops init at the planner's checks; the robot and cameras it was given are
    written first, not dropped without a word."""

    def no_driver(planner, *, interactive, repair=False):
        raise init_cli.TandemError("Preflight found blocking problems: nvidia driver")

    monkeypatch.setattr(init_cli, "_planner_preflight", no_driver)
    result = _run("-y", "--robot-host", "172.16.0.9", "--camera", "hand=111", "--camera", "external=222")
    assert result.exit_code == 1 and "nvidia driver" in result.exception.message
    rig = rig_mod.load(force=True)
    assert rig.robot.host == "172.16.0.9" and set(rig.cameras.configured()) == {"hand", "external"}


def test_tandems_own_preflight_says_the_rig_flags_were_not_applied(monkeypatch):
    monkeypatch.setattr(init_cli, "_preflight", lambda viz_only: [probe.Check("ffmpeg", probe.FAIL, "not found")])
    monkeypatch.setattr(init_cli, "_ffmpeg_install_command", lambda interactive: None)
    result = _run("-y", "--robot-host", "172.16.0.9")
    assert result.exit_code == 1
    assert "--robot-host, --robot-type and --camera were not applied" in result.exception.hint
    assert not rig_mod.exists()


# --------------------------------------------------------------------------- ffmpeg


def _no_ffmpeg(monkeypatch, *, installs: bool) -> list[list[str]]:
    """A machine with no ffmpeg on PATH, whose package manager is a stand-in that records each command."""
    ran: list[list[str]] = []
    missing = probe.Check("ffmpeg", probe.WARN, "not found", group="runtime")
    monkeypatch.setattr(init_cli, "_preflight", lambda viz_only: [missing])
    monkeypatch.setattr(init_cli, "_ffmpeg_install_command", lambda interactive: ["pkg", "install", "ffmpeg"])

    def install(command, *, interactive):
        ran.append(command)
        return [] if installs else ["E: Unable to locate package ffmpeg"]

    monkeypatch.setattr(init_cli, "_install_ffmpeg", install)
    after = probe.Check("ffmpeg", probe.OK, "/usr/bin/ffmpeg") if installs else missing
    monkeypatch.setattr(probe, "check_ffmpeg", lambda: after)
    return ran


def test_init_installs_a_missing_ffmpeg(monkeypatch):
    ran = _no_ffmpeg(monkeypatch, installs=True)
    result = _run("-y")
    assert result.exit_code == 0, result.output
    assert ran == [["pkg", "install", "ffmpeg"]]
    assert "ffmpeg installed" in _said(result)


def test_an_ffmpeg_that_does_not_install_is_said_and_init_goes_on(monkeypatch):
    ran = _no_ffmpeg(monkeypatch, installs=False)
    result = _run("-y")
    assert result.exit_code == 0, result.output
    assert ran and "ffmpeg did not install" in _said(result)
    assert "Unable to locate package ffmpeg" in _said(result)


def test_an_ffmpeg_already_there_is_not_installed(monkeypatch):
    ran = _no_ffmpeg(monkeypatch, installs=True)
    monkeypatch.setattr(init_cli, "_preflight", lambda viz_only: [probe.Check("ffmpeg", probe.OK, "/usr/bin/ffmpeg")])
    assert _run("-y").exit_code == 0
    assert ran == []


def test_sudo_is_never_left_waiting_on_a_password_without_a_terminal(monkeypatch):
    import shutil
    import sys

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr("os.geteuid", lambda: 1000)
    apt = ["env", "DEBIAN_FRONTEND=noninteractive", "apt-get", "install", "-y", "ffmpeg"]
    assert init_cli._ffmpeg_install_command(interactive=False) == ["sudo", "-n", *apt]
    assert init_cli._ffmpeg_install_command(interactive=True) == ["sudo", *apt]
    monkeypatch.setattr("os.geteuid", lambda: 0)
    assert init_cli._ffmpeg_install_command(interactive=False) == apt
