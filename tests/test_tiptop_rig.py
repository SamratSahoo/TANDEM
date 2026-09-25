"""TiPToP's settings, split: what the task says (planner.options.tamp), and what this machine has (the rig).

TiPToP declares ``tamp`` as its task's and ``robot`` and ``perception`` as its machine's. Its robot's
address and arm are the rig's own ``robot.host`` and ``robot.type``, which every planner reads; its
cameras and their extrinsics are the rig's too. What it renders for the pinned tiptop -- tiptop.yml, the
environment -- is put back together from both halves, and FoundationStereo's address is a machine
setting doctor checks like M2T2's.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tandem.core import probe, secrets
from tandem.core import rig as rig_mod
from tandem.core.errors import TandemError
from tandem.planners import registry
from tandem.planners.tiptop import FACTORY, doctor, render
from tandem.planners.tiptop.options import resolve, resolve_profile


def test_the_task_options_are_tamp_and_the_machines_are_robot_and_perception():
    assert set(FACTORY.OPTIONS) == {"tamp"}
    assert set(FACTORY.RIG_OPTIONS) == {"robot", "perception"}
    assert FACTORY.validate_options({"tamp": {"num_particles": 64}}) == {"tamp": {"num_particles": 64}}
    for key in ("robot", "perception"):
        with pytest.raises(ValidationError, match=rf"{key} is a machine setting.*tandem rig set planners.tiptop.{key}"):
            FACTORY.validate_options({key: {}})


def test_the_machine_options_fill_every_default():
    machine = FACTORY.validate_rig_options({})
    assert machine["robot"] == {
        "dof": 7,
        "port": 5555,
        "gripper_port": 5559,
        "state_port": 5557,
        "time_dilation_factor": 0.2,
        "q_home": [0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.0],
        "q_capture": [-0.034, 0.090, 0.080, -1.319, -0.003, 1.253, 0.030],
    }
    assert machine["perception"]["foundation_stereo"] == {"url": "http://localhost:1234"}
    assert machine["perception"]["m2t2"]["url"] == "http://localhost:8123"
    assert FACTORY.validate_rig_options(machine) == machine
    with pytest.raises(ValidationError, match="needs a scheme and a host"):
        FACTORY.validate_rig_options({"perception": {"foundation_stereo": {"url": "localhost"}}})


@pytest.mark.parametrize("key", ["host", "type"])
def test_the_arms_address_and_type_are_the_rigs_own(key):
    with pytest.raises(ValidationError) as caught:
        FACTORY.validate_rig_options({"robot": {key: "x"}})
    assert f"robot.{key} is the rig's own robot.{key}" in str(caught.value)
    assert f"tandem rig set robot.{key}" in str(caught.value)


def test_resolve_refuses_an_arm_tiptop_does_not_drive_and_names_the_ones_it_does(machine_rig):
    rig = rig_mod.update({"robot.type": "kuka"})
    with pytest.raises(TandemError) as caught:
        resolve(rig)
    assert "rig.yml's robot.type" in caught.value.message
    for arm in ("fr3_robotiq", "panda_robotiq", "panda", "ur5"):
        assert arm in caught.value.message
    assert "tandem rig set robot.type" in caught.value.hint
    # Doctor still wants the rest of the rows: it resolves without the check, and says so in a row.
    assert resolve(rig, check_type=False).robot.type == "kuka"


def test_tiptop_yml_takes_the_arm_the_cameras_and_the_servers_from_the_rig(profile, machine_rig):
    rig = rig_mod.update(
        {
            "robot.host": "10.0.0.5",
            "robot.type": "panda_robotiq",
            "cameras.external_2.serial": "31425515",
            "planners.tiptop.robot.port": 6000,
            "planners.tiptop.perception.foundation_stereo.url": "http://depth-box:1234",
            "planners.tiptop.perception.m2t2.url": "http://gpu:8123",
        }
    )
    config = render.render_tiptop_config(rig, resolve_profile(profile, rig))
    assert config["robot"]["host"] == "10.0.0.5" and config["robot"]["type"] == "panda_robotiq"
    assert config["robot"]["port"] == 6000
    assert config["cameras"]["hand"]["serial"] == "14846828"
    assert config["cameras"]["external_2"]["serial"] == "31425515"
    assert config["cameras"]["perception"] == "external"
    assert config["perception"]["foundation_stereo"] == {"url": "http://depth-box:1234"}
    assert config["perception"]["m2t2"]["url"] == "http://gpu:8123"


def test_the_environment_points_tiptop_at_the_rigs_calibration_and_cameras(profile, machine_rig, tmp_path):
    options = resolve_profile(profile, machine_rig)
    env = render.render_env(profile, machine_rig, options, events_file=tmp_path / "e.jsonl", base={})
    assert env["TIPTOP_CALIBRATION"] == str(machine_rig.calibration_file())
    assert env["TIPTOP_HAND_CAMERA_ID"] == "14846828" and env["TIPTOP_EXTERNAL_CAMERA_ID"] == "32439448"
    assert "TIPTOP_EXTERNAL_2_CAMERA_ID" not in env


def test_missing_extrinsics_are_found_against_the_rig(profile, machine_rig):
    options = resolve_profile(profile, machine_rig)
    assert not [p for p in render.check_assets(profile, machine_rig, options) if p.startswith(render.MISSING_EXTRINSICS)]
    rig = rig_mod.update({"cameras.external.serial": "99999999"})  # a camera swapped for another unit
    (problem,) = [p for p in render.check_assets(profile, rig, options) if p.startswith(render.MISSING_EXTRINSICS)]
    assert "99999999" in problem and str(rig.calibration_file()) in problem


def test_a_session_without_the_rigs_calibration_file_is_told_how_to_make_one(profile, machine_rig, tmp_path):
    machine_rig.calibration_file().unlink()
    with pytest.raises(TandemError) as caught:
        render.prepare_session_files(profile, machine_rig, tmp_path / "s", resolve_profile(profile, machine_rig))
    assert str(machine_rig.calibration_file()) in caught.value.message and "tandem init" in caught.value.hint


# --- doctor --------------------------------------------------------------------------------------------------


@pytest.fixture
def ports(monkeypatch):
    """Every port doctor would probe, answered without the network: what was asked, and an OK."""
    asked: list[tuple[str, str, int]] = []

    def check_port(name, host, port, *, timeout=1.5, group="hardware", hint=""):
        asked.append((name, host, port))
        return probe.Check(name, probe.OK, f"{host}:{port} reachable", group=group)

    monkeypatch.setattr(probe, "check_port", check_port)
    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "test-key")
    return asked


def test_doctor_probes_the_robot_and_the_servers_at_the_rigs_addresses(profile, ports):
    rig_mod.update(
        {
            "robot.host": "10.0.0.5",
            "planners.tiptop.robot.port": 6000,
            "planners.tiptop.perception.foundation_stereo.url": "http://depth-box:4321",
        }
    )
    rows = {c.name: c for c in doctor.doctor_checks(profile, settings=None, runtime_ready=True, probe_hardware=True)}
    assert ("robot control", "10.0.0.5", 6000) in ports
    assert ("robot state port", "10.0.0.5", 5557) in ports
    assert ("m2t2 grasp server", "localhost", 8123) in ports
    assert rows["m2t2 grasp server"].state == probe.OK
    assert ("foundation stereo depth server", "depth-box", 4321) in ports
    assert rows["foundation stereo depth server"].state == probe.OK
    assert rows["foundation stereo depth server"].group == "hardware"


def test_doctors_rig_rows_are_in_the_rig_group_and_an_arm_it_does_not_drive_fails(profile, ports):
    rows = {c.name: c for c in doctor.doctor_checks(profile, settings=None, runtime_ready=True, probe_hardware=False)}
    for name in ("robot type", "tiptop cameras", "camera calibration"):
        assert rows[name].group == "rig" and rows[name].state == probe.OK, name
    assert rows["tamp settings"].group == "profile"

    rig_mod.update({"robot.type": "franka"})
    rows = {c.name: c for c in doctor.doctor_checks(profile, settings=None, runtime_ready=True, probe_hardware=False)}
    assert rows["robot type"].state == probe.FAIL and "did you mean panda" in rows["robot type"].detail
    assert "tandem rig set robot.type" in rows["robot type"].hint
    assert rows["tamp settings"].state == probe.OK, "the rest is still checked"


def test_doctor_says_tiptop_cannot_be_checked_on_a_rig_that_does_not_load(profile, ports):
    rig_mod.paths.rig_file().write_text("planners:\n  tiptop:\n    robot: {dof: 0}\n    oops: 1\n")
    rows = {c.name: c for c in doctor.doctor_checks(profile, settings=None, runtime_ready=True, probe_hardware=False)}
    assert rows["tiptop rig settings"].state == probe.FAIL and rows["tiptop rig settings"].group == "rig"


def test_the_catalog_names_where_the_servers_are_set():
    requires = " ".join(registry.info("tiptop").requires)
    assert "planners.tiptop.perception.m2t2.url" in requires
    assert "planners.tiptop.perception.foundation_stereo.url" in requires and "http://localhost:1234" in requires


# --- how a profile's TiPToP settings are shown ---------------------------------------------------------------------


def test_describe_shows_the_machines_robot_and_perception_and_the_tasks_tamp(profile, machine_rig):
    rig_mod.update(
        {
            "robot.host": "10.0.0.5",
            "planners.tiptop.perception.m2t2.url": "http://g:1",
            "planners.tiptop.perception.foundation_stereo.url": "http://d:1",
        }
    )
    view = registry.describe_options("tiptop", profile)
    sections = {section.title: section for section in view.sections}
    assert list(sections) == ["robot", "perception", "tamp"]
    assert sections["robot"].subtitle == "this machine's (rig.yml)"
    assert ("address", "10.0.0.5:5555") in sections["robot"].rows
    assert sections["perception"].subtitle.startswith("this machine's (rig.yml)")
    assert ("grasps", "http://g:1") in sections["perception"].rows
    assert ("depth", "http://d:1") in sections["perception"].rows
    assert view.summary.startswith("fr3_robotiq at 10.0.0.5")
    assert view.receives["num_particles"] == 256, "what the task gives the planner"
