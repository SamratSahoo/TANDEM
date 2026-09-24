"""The rig: this machine's robot, cameras and calibration, in rig.yml beside config.toml.

Every profile shares it, so a mistake in it is a mistake in every session on the machine: it is
validated whole, refused loudly with the line located, and written only once it validates. It is also a
file people annotate, so a programmatic change keeps what they wrote in it.
"""

from __future__ import annotations

import json
import logging

import pytest
from helpers import isolate_registry

from tandem.core import paths
from tandem.core import rig as rig_mod
from tandem.core.errors import RigInvalid, TandemError
from tandem.core.rig import Rig


@pytest.fixture(autouse=True)
def _registry(monkeypatch):
    isolate_registry(monkeypatch)


def _write(text: str):
    path = paths.rig_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# --- where it is, and what a machine without one has ----------------------------------------------------


def test_the_rig_lives_beside_config_toml(isolated_env):
    assert paths.rig_file() == isolated_env / "config" / "rig.yml"
    assert paths.rig_file().parent == paths.config_file().parent


def test_a_machine_without_a_rig_has_the_defaults_and_no_cameras(isolated_env):
    assert not rig_mod.exists()
    rig = rig_mod.load()
    assert rig.robot.type == "fr3_robotiq" and rig.robot.host == "172.16.0.2"
    assert rig.cameras.configured() == {}
    assert rig.planners == {}
    assert rig.calibration_file() == isolated_env / "config" / "calibration.json"
    assert rig.extrinsics() == {} and rig.missing_calibration() == []
    assert rig_mod.read_text() == rig_mod.template(), "`tandem rig edit` starts from the commented template"


def test_the_template_is_a_valid_rig():
    rig = rig_mod.parse_text(rig_mod.template(), source="rig_template.yml")
    assert rig.version == rig_mod.RIG_VERSION and rig.cameras.configured() == {}


# --- update: a round trip that keeps what a person wrote --------------------------------------------------


def test_update_writes_the_file_keeps_its_comments_and_creates_the_calibration(isolated_env):
    rig = rig_mod.update({"robot.host": "10.0.0.5", "cameras.hand.serial": "14846828"})
    assert rig.robot.host == "10.0.0.5" and rig.cameras.hand.serial == "14846828"
    text = paths.rig_file().read_text()
    assert "# the robot computer: the NUC" in text, "the template's comments survive"
    assert rig.calibration_file().read_text() == "{}\n", "created empty, for the calibration script to fill"

    # A person's own note survives the next change too.
    paths.rig_file().write_text(text.replace("version: 1", "version: 1   # the lab's FR3, bench 2"))
    rig_mod.update({"cameras.external.serial": "32439448"})
    text = paths.rig_file().read_text()
    assert "the lab's FR3, bench 2" in text
    assert rig_mod.load().cameras.configured().keys() == {"hand", "external"}


def test_a_serial_set_through_update_is_written_quoted_and_read_back_as_text(isolated_env):
    rig_mod.update({"cameras.hand": {"serial": "14846828"}, "cameras.external.serial": "0123"})
    text = paths.rig_file().read_text()
    assert "'14846828'" in text and "'0123'" in text
    rig = rig_mod.load(force=True)
    assert rig.cameras.hand.serial == "14846828" and rig.cameras.external.serial == "0123"


def test_none_removes_a_key_and_a_bad_update_writes_nothing(isolated_env):
    rig_mod.update({"cameras.hand.serial": "1", "cameras.external.serial": "2", "cameras.external_2.serial": "3"})
    rig = rig_mod.update({"cameras.external_2": None})
    assert set(rig.cameras.configured()) == {"hand", "external"}
    before = paths.rig_file().read_bytes()
    with pytest.raises(RigInvalid) as caught:
        rig_mod.update({"robot.host": "http://10.0.0.5:5555"})
    assert "Nothing was written" in caught.value.hint
    assert paths.rig_file().read_bytes() == before


def test_the_load_cache_is_dropped_when_the_file_changes(isolated_env):
    rig_mod.update({"robot.host": "10.0.0.5"})
    first = rig_mod.load()
    assert rig_mod.load() is first, "cached while the file is as it was"
    text = paths.rig_file().read_text().replace("10.0.0.5", "10.0.0.6")
    paths.rig_file().write_text(text + "\n")  # another process: a `tandem rig set` in a terminal
    assert rig_mod.load().robot.host == "10.0.0.6"


def test_write_text_validates_before_it_writes(isolated_env):
    rig_mod.update({"robot.host": "10.0.0.5"})
    before = paths.rig_file().read_bytes()
    with pytest.raises(RigInvalid):
        rig_mod.write_text("version: 1\nrobot: {host: 'two words'}\n")
    assert paths.rig_file().read_bytes() == before
    rig = rig_mod.write_text("version: 1\nrobot: {host: nuc.lab}\n")
    assert rig.robot.host == "nuc.lab" and rig.calibration_file().is_file()


# --- validation, located and loud --------------------------------------------------------------------------


def test_an_unquoted_serial_is_refused_with_the_way_to_write_it(isolated_env):
    with pytest.raises(RigInvalid) as caught:
        rig_mod.parse_text("cameras:\n  hand: {serial: 14846828}\n", source="rig.yml")
    assert "cameras.hand.serial" in caught.value.message
    assert "quote it: '14846828'" in caught.value.message


def test_one_camera_cannot_fill_two_roles():
    with pytest.raises(RigInvalid, match="is both cameras.hand and cameras.external"):
        rig_mod.parse_text(
            "cameras:\n  hand: {serial: '1'}\n  external: {serial: '1'}\n", source="rig.yml"
        )


@pytest.mark.parametrize("host", ["http://172.16.0.2", "172.16.0.2:5555", "nuc/1", "two words", ""])
def test_the_robot_host_is_an_address_and_nothing_else(host):
    with pytest.raises(RigInvalid) as caught:
        rig_mod.parse_text(f"robot: {{host: '{host}'}}\n", source="rig.yml")
    assert "robot.host" in caught.value.message and "no http://, no port" in caught.value.message


@pytest.mark.parametrize("host", ["172.16.0.2", "nuc.lab.example", "fe80::1", "localhost"])
def test_hostnames_and_addresses_are_accepted(host):
    assert rig_mod.parse_text(f"robot: {{host: '{host}'}}\n", source="rig.yml").robot.host == host


def test_a_newer_rig_is_refused_as_one():
    with pytest.raises(RigInvalid, match="written by a newer tandem"):
        rig_mod.parse_text("version: 2\n", source="rig.yml")


def test_an_unknown_key_names_the_nearest_real_one():
    with pytest.raises(RigInvalid) as caught:
        rig_mod.parse_text("robt: {host: 10.0.0.5}\n", source="rig.yml")
    assert "'robt' is not a rig setting (did you mean 'robot'?)" in caught.value.message
    with pytest.raises(RigInvalid, match=r"robot: 'hots' is not a robot setting \(did you mean 'host'\?\)"):
        rig_mod.parse_text("robot: {hots: 10.0.0.5}\n", source="rig.yml")


def test_an_empty_or_unreadable_rig_is_refused_not_read_as_defaults(isolated_env):
    for text, message in (("", "is empty"), ("- a\n", "must be a mapping"), ("robot: [\n", "not valid YAML")):
        _write(text)
        with pytest.raises(RigInvalid, match=message) as caught:
            rig_mod.load(force=True)
        assert "tandem rig edit" in caught.value.hint


def test_the_perception_camera_may_be_missing_while_a_rig_is_set_up():
    """`tandem rig set` adds one camera at a time. The session and doctor refuse a rig perception cannot
    read (tests/test_session_rig.py, tests/test_rig_cli.py); the file itself may be in between."""
    rig = rig_mod.parse_text("cameras:\n  perception: external\n  hand: {serial: '1'}\n", source="rig.yml")
    assert rig.cameras.perception_missing() == "cameras.perception is 'external' but cameras.external is not configured"


# --- each planner's block ------------------------------------------------------------------------------------


def test_tiptops_block_is_validated_and_normalised(isolated_env):
    rig = rig_mod.update({"planners.tiptop.robot.time_dilation_factor": 0.3})
    block = rig.planners["tiptop"]
    assert block["robot"]["time_dilation_factor"] == 0.3
    assert block["robot"]["port"] == 5555, "defaults filled in, in memory"
    assert block["perception"]["m2t2"]["url"] == "http://localhost:8123"
    assert "port:" not in paths.rig_file().read_text(), "the file keeps what was written, and only that"
    assert rig_mod.planner_options(rig, "tiptop") == block


def test_tiptops_block_is_refused_with_its_line_located():
    with pytest.raises(RigInvalid) as caught:
        rig_mod.parse_text(
            "planners:\n  tiptop:\n    robot: {q_home: [0, 0]}\n    perception: {m2t2: {url: nope}}\n",
            source="rig.yml",
        )
    message = caught.value.message
    assert "planners.tiptop.robot: q_home has 2 values but robot.dof is 7" in message
    assert "planners.tiptop.perception.m2t2.url" in message


def test_the_robots_address_in_tiptops_block_is_pointed_at_the_rigs_own():
    with pytest.raises(RigInvalid) as caught:
        rig_mod.parse_text("planners:\n  tiptop:\n    robot: {host: 10.0.0.5}\n", source="rig.yml")
    assert "robot.host is the rig's own robot.host" in caught.value.message
    assert "tandem rig set robot.host" in caught.value.message


def test_a_task_setting_in_the_rig_is_refused_as_the_profiles():
    with pytest.raises(RigInvalid) as caught:
        rig_mod.parse_text("planners:\n  tiptop:\n    tamp: {num_particles: 5}\n", source="rig.yml")
    assert "tamp is a task setting" in caught.value.message
    assert "planner.options" in caught.value.message


def test_a_planner_that_is_not_installed_keeps_its_block_and_is_said_once(isolated_env, caplog):
    _write("planners:\n  shelfbot: {bins: [a, b]}\n")
    with caplog.at_level(logging.WARNING, logger="tandem.core.rig"):
        rig = rig_mod.load(force=True)
        rig_mod.load(force=True)
    assert rig.planners["shelfbot"] == {"bins": ["a", "b"]}
    notices = [r.getMessage() for r in caplog.records if "shelfbot" in r.getMessage()]
    assert len(notices) == 1 and "not installed here" in notices[0]
    assert rig_mod.planner_options(rig, "shelfbot") == {"bins": ["a", "b"]}, "as written"


def test_a_planner_that_will_not_load_keeps_its_block(isolated_env, monkeypatch):
    import importlib.metadata

    from tandem.planners import registry

    declared = [importlib.metadata.EntryPoint("gone", "tandem_no_such_planner:FACTORY", registry.GROUP)]
    real = importlib.metadata.entry_points
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda **params: list(declared) if params.get("group") == registry.GROUP else real(**params),
    )
    _write("planners:\n  gone: {anything: 1}\n")
    assert rig_mod.load(force=True).planners["gone"] == {"anything": 1}


def test_planner_options_validates_an_absent_block_as_empty(isolated_env):
    rig = rig_mod.load()
    options = rig_mod.planner_options(rig, "tiptop")
    assert set(options) == {"robot", "perception"} and options["robot"]["dof"] == 7


# --- calibration -------------------------------------------------------------------------------------------


def test_the_calibration_is_beside_rig_yml_or_where_it_says(isolated_env, tmp_path):
    rig = rig_mod.update({"cameras.hand.serial": "1", "cameras.external.serial": "2"})
    assert rig.calibration_file() == paths.config_dir() / "calibration.json"
    rig.calibration_file().write_text(json.dumps({"1": {"pose": [0] * 6}}))
    assert rig_mod.load().missing_calibration() == ["2"]

    elsewhere = tmp_path / "shared" / "extrinsics.json"
    rig = rig_mod.update({"calibration": str(elsewhere)})
    assert rig.calibration_file() == elsewhere and elsewhere.read_text() == "{}\n"
    assert rig.missing_calibration() == ["1", "2"]


def test_a_calibration_file_that_is_not_json_is_said_to_be(isolated_env):
    rig = rig_mod.update({"cameras.hand.serial": "1"})
    rig.calibration_file().write_text("{not json")
    with pytest.raises(RigInvalid, match="is not valid JSON"):
        rig.missing_calibration()


# --- `tandem rig set KEY VALUE`: what the value means ----------------------------------------------------------


def test_coerce_keeps_text_as_text_and_reads_the_rest_as_yaml(isolated_env):
    assert rig_mod.coerce("cameras.hand.serial", "14846828") == "14846828"
    assert rig_mod.coerce("cameras.hand.serial", "0123") == "0123"
    assert rig_mod.coerce("robot.host", "172.16.0.5") == "172.16.0.5"
    assert rig_mod.coerce("cameras.external_2", "null") is None
    assert rig_mod.coerce("planners.tiptop.robot.q_home", "[0, 1, 2]") == [0, 1, 2]
    assert rig_mod.coerce("planners.tiptop.robot.port", "5556") == 5556
    assert rig_mod.coerce("planners.tiptop.perception.m2t2.apply_bounds", "false") is False


def test_coerce_refuses_a_key_that_is_not_a_rig_setting(isolated_env):
    with pytest.raises(TandemError, match="did you mean robot.host"):
        rig_mod.coerce("robot.hots", "x")
    with pytest.raises(TandemError, match="did you mean cameras.external"):
        rig_mod.coerce("cameras.extrenal.serial", "1")
    with pytest.raises(TandemError, match="tandem writes it|tandem's to write"):
        rig_mod.coerce("version", "2")
    with pytest.raises(TandemError, match="did you mean 'tiptop'"):
        rig_mod.coerce("planners.tiptopp.robot.port", "1")


def test_a_rig_built_by_hand_resolves_against_the_config_dir(isolated_env):
    assert Rig().calibration_file() == paths.config_dir() / "calibration.json"
    assert Rig().file() == paths.rig_file()
