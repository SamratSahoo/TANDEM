"""`tandem rig`: show, set, edit and path, and what doctor says about the rig.

One line changes one thing on this machine for every profile at once -- `tandem rig set robot.host
172.16.0.5` is how the NUC's address is set -- so each change is checked before it is written, and what
it breaks for collection is said the moment it is made.
"""

from __future__ import annotations

import json

import pytest
from helpers import isolate_registry
from typer.testing import CliRunner

from tandem.cli.app import app
from tandem.core import paths, probe
from tandem.core import rig as rig_mod


@pytest.fixture(autouse=True)
def _registry(monkeypatch):
    isolate_registry(monkeypatch)


def _run(*args: str):
    return CliRunner().invoke(app, list(args))


def _flat(output: str) -> str:
    return " ".join(output.split())


def test_show_on_a_machine_without_a_rig_says_so(isolated_env):
    shown = _run("rig", "show")
    assert shown.exit_code == 0, shown.output
    output = _flat(shown.output)
    assert "not written yet" in output and "fr3_robotiq" in output and "172.16.0.2" in output
    assert "tandem rig set cameras.ROLE.serial SERIAL" in output


def test_show_lists_the_robot_the_cameras_what_is_calibrated_and_each_planners_block(machine_rig):
    rig_mod.update({"cameras.external_2.serial": "31425515", "planners.tiptop.robot.time_dilation_factor": 0.3})
    shown = _run("rig", "show")
    assert shown.exit_code == 0, shown.output
    output = _flat(shown.output)
    for needle in ("14846828", "32439448", "31425515", "2 of 3 camera(s)", "planners.tiptop"):
        assert needle in output, needle
    # One row per setting, as `tandem rig set` names it; what the file leaves out is marked as the default.
    assert "robot.time_dilation_factor 0.3 " in output + " "
    assert "robot.port 5555 (default)" in output
    assert "perception.m2t2.url http://localhost:8123 (default)" in output
    assert "perception.foundation_stereo.url http://localhost:1234 (default)" in output

    payload = json.loads(_run("rig", "show", "--json").output)
    assert payload["file"] == str(paths.rig_file()) and payload["exists"] is True
    assert payload["rig"]["robot"]["host"] == "172.16.0.2"
    assert payload["calibrated"] == ["14846828", "32439448"]
    assert payload["missing_calibration"] == ["31425515"]
    tiptop = payload["planners"]["tiptop"]
    assert tiptop["installed"] and tiptop["problem"] is None
    assert set(tiptop["declared"]) == {"robot", "perception"}
    assert tiptop["options"]["robot"]["time_dilation_factor"] == 0.3
    assert tiptop["set"] == ["robot.time_dilation_factor"], "what the file sets, apart from the defaults"


def test_set_changes_one_setting_and_keeps_serials_as_text(isolated_env):
    result = _run("rig", "set", "robot.host", "172.16.0.5")
    assert result.exit_code == 0, result.output
    assert "robot.host = 172.16.0.5" in _flat(result.output)
    assert rig_mod.load().robot.host == "172.16.0.5"

    assert _run("rig", "set", "cameras.hand.serial", "14846828").exit_code == 0
    assert _run("rig", "set", "cameras.external.serial", "0123").exit_code == 0
    rig = rig_mod.load()
    assert rig.cameras.hand.serial == "14846828" and rig.cameras.external.serial == "0123"

    assert _run("rig", "set", "planners.tiptop.perception.m2t2.url", "http://gpu:8123").exit_code == 0
    assert rig_mod.planner_options(rig_mod.load(), "tiptop")["perception"]["m2t2"]["url"] == "http://gpu:8123"

    assert _run("rig", "set", "cameras.external_2.serial", "31425515").exit_code == 0
    assert _run("rig", "set", "cameras.external_2", "null").exit_code == 0
    assert rig_mod.load().cameras.external_2 is None


def test_set_refuses_what_is_not_a_rig_setting_and_writes_nothing(isolated_env):
    assert _run("rig", "set", "robot.host", "10.0.0.5").exit_code == 0
    before = paths.rig_file().read_bytes()

    typo = _run("rig", "set", "robot.hots", "10.0.0.6")
    assert typo.exit_code == 1 and "did you mean robot.host" in typo.exception.message
    absent = _run("rig", "set", "planners.shelfbot.port", "1")
    assert absent.exit_code == 1 and "No planner named 'shelfbot' is installed" in absent.exception.message
    bad = _run("rig", "set", "robot.host", "http://10.0.0.6")
    assert bad.exit_code == 1 and "no http://" in bad.exception.message
    assert paths.rig_file().read_bytes() == before


def test_set_says_at_once_what_the_change_breaks_for_collection(machine_rig):
    result = _run("rig", "set", "robot.type", "franka")
    assert result.exit_code == 0, result.output
    output = _flat(result.output)
    assert "robot type: unsupported robot type 'franka'" in output and "did you mean panda" in output

    result = _run("rig", "set", "cameras.perception", "hand")
    assert result.exit_code == 0
    assert "not configured" not in _flat(result.output), "the wrist camera is there to read"
    rig_mod.update({"cameras.hand": None})
    result = _run("rig", "set", "robot.type", "fr3_robotiq")
    assert "cameras.perception is 'hand' but cameras.hand is not configured" in _flat(result.output)


def test_edit_validates_on_save_and_restores_the_rig_when_it_does_not(isolated_env, monkeypatch, tmp_path):
    rig_mod.update({"robot.host": "10.0.0.5"})
    before = paths.rig_file().read_text()

    breaker = tmp_path / "break.sh"
    breaker.write_text("#!/bin/sh\nprintf 'robot: {host: \"http://x\"}\\n' > \"$1\"\n")
    breaker.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(breaker))
    rejected = _run("rig", "edit")
    assert rejected.exit_code == 1
    assert "Your edit was rejected and the previous rig restored" in rejected.exception.message
    assert paths.rig_file().read_text() == before
    assert not paths.rig_file().with_name("rig.yml.bak").exists()
    # The person's text is not thrown away: it is kept beside the file, and the error says where.
    kept = paths.rig_file().with_name("rig.yml.rejected")
    assert kept.read_text() == 'robot: {host: "http://x"}\n' and str(kept) in rejected.exception.message

    fixer = tmp_path / "fix.sh"
    fixer.write_text("#!/bin/sh\nprintf 'version: 1\\nrobot: {host: nuc.lab}\\n' > \"$1\"\n")
    fixer.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(fixer))
    accepted = _run("rig", "edit")
    assert accepted.exit_code == 0, accepted.output
    assert rig_mod.load().robot.host == "nuc.lab"


def test_edit_on_a_machine_without_a_rig_starts_from_the_template_and_leaves_nothing_if_refused(
    isolated_env, monkeypatch, tmp_path
):
    breaker = tmp_path / "break.sh"
    breaker.write_text("#!/bin/sh\nprintf 'robot: [\\n' > \"$1\"\n")
    breaker.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(breaker))
    assert _run("rig", "edit").exit_code == 1
    assert not rig_mod.exists(), "a refused first edit leaves the machine as it was: no rig.yml"

    monkeypatch.setenv("EDITOR", "true")
    assert _run("rig", "edit").exit_code == 0
    assert paths.rig_file().read_text() == rig_mod.template()


def test_path_prints_the_rig_and_its_calibration(machine_rig):
    assert _run("rig", "path").output.strip() == str(paths.rig_file())
    assert _run("rig", "path", "--calibration").output.strip() == str(machine_rig.calibration_file())


def test_config_list_says_where_the_rig_is(isolated_env):
    payload = json.loads(_run("config", "list", "--json").output)
    assert payload["resolved"]["rig_file"] == str(paths.rig_file())


# --- doctor's rig rows ------------------------------------------------------------------------------------------


def _rows(**kwargs):
    from tandem.cli.doctor import collect_checks

    return {c.name: c for c in collect_checks(probe_hardware=False, **kwargs)}


def test_doctor_says_a_machine_has_no_rig_yet(isolated_env):
    rows = _rows()
    assert rows["rig"].state == probe.WARN and rows["rig"].detail == "not set up"
    assert "tandem rig set robot.host" in rows["rig"].hint
    assert rows["cameras"].state == probe.SKIP and rows["cameras"].group == "rig"


def test_doctor_shows_the_rig_and_fails_a_camera_perception_cannot_read(profile):
    rows = _rows()
    assert rows["rig"].state == probe.OK and "fr3_robotiq at 172.16.0.2" in rows["rig"].detail
    assert rows["cameras"].state == probe.OK and "perception reads external" in rows["cameras"].detail
    rig_mod.update({"cameras.external": None})
    rows = _rows()
    assert rows["cameras"].state == probe.FAIL
    assert "cameras.perception is 'external' but cameras.external is not configured" in rows["cameras"].detail


def test_doctor_fails_a_rig_that_does_not_load(profile):
    paths.rig_file().write_text("robot: {host: 'http://x'}\n")
    rows = _rows()
    assert rows["rig"].state == probe.FAIL and "robot.host" in rows["rig"].detail
    assert "tandem rig edit" in rows["rig"].hint


def test_doctor_shows_the_rig_group_between_credentials_and_profile(profile):
    from tandem.cli import doctor

    assert doctor.GROUPS == ("core", "runtime", "gpu", "credentials", "rig", "profile", "hardware")
    shown = _run("doctor", "--no-hardware")
    headings = [line.strip().lstrip("◆").strip() for line in shown.output.splitlines() if "◆" in line]
    assert headings.index("credentials") < headings.index("rig") < headings.index("profile")
