"""The conformance kit builds every backend for a machine, and machine settings reach a planner through it.

- The kit's context had no rig, so a planner doing what the scaffold says -- ``self.rig.robot.host`` in
  ``__init__`` -- failed the kit with AttributeError; and one that fell back to ``rig.load()`` read the
  developer's own rig.yml.
- TiPToP ignored the rig_options a context gave it whenever the context had no rig.
- No toy planner declared or read a machine setting, so nothing showed one reaching a sidecar.
- The hint for a machine setting put in a profile told `tandem rig set planners.toy.sensor.KEY VALUE` for a
  setting that is one value.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from helpers import isolate_registry
from toy_planner import DEFAULT_STATION, ToyPlanner, ToySidecarPlanner

from tandem.core import rig as rig_mod
from tandem.core.errors import TandemError
from tandem.planners import PlannerInfo, registry
from tandem.planners.base import BackendContext
from tandem.planners.testing import PlannerConformance, stand_in_rig


class HostReader(ToyPlanner):
    """What the scaffold's comment tells a planner to do: read the arm's address from the rig, unguarded."""

    info = PlannerInfo(name="host-reader", display_name="Host reader", summary="Reads the rig in __init__.")
    CAPABILITIES = replace(ToyPlanner.CAPABILITIES, name="host-reader")

    def __init__(self, ctx=None) -> None:
        super().__init__(ctx)
        self.host = self.rig.robot.host


class TestAPlannerThatReadsTheRigInItsConstructorConforms(PlannerConformance):
    planner = HostReader
    rig_options = {"station": "bench-3"}


def test_the_kit_builds_for_a_stand_in_machine_not_this_one(tmp_path, isolated_env):
    rig_mod.update({"robot.host": "10.9.9.9"})  # this machine's rig: the kit must not read it
    kit = TestAPlannerThatReadsTheRigInItsConstructorConforms()
    backend = kit.make_backend(tmp_path)
    assert backend.host == stand_in_rig(tmp_path).robot.host == "172.16.0.2"
    assert backend.station == "bench-3"
    assert backend.rig.calibration_file().parent == tmp_path, "its calibration is the test's, not the machine's"
    assert set(backend.rig.cameras.configured()) == {"hand", "external"}


def test_a_sidecar_is_handed_the_machines_settings_when_it_warms(tmp_path):
    class Kit(PlannerConformance):
        planner = ToySidecarPlanner
        rig_options = {"station": "bench-2"}

    kit = Kit()
    with kit.warmed(tmp_path) as backend:
        sent = backend.call("last_args", of="warm")
    assert sent["station"] == "bench-2" and sent["robot_host"] == "172.16.0.2"

    bare = ToySidecarPlanner()
    assert bare.warm_args()["station"] == DEFAULT_STATION and bare.warm_args()["robot_host"] is None


def test_tiptop_uses_the_rig_options_a_context_gives_it_even_without_a_rig(monkeypatch, isolated_env, tmp_path):
    from tandem.core import secrets
    from tandem.planners.tiptop import FACTORY
    from tandem.planners.tiptop import factory as tiptop_factory

    monkeypatch.setattr(secrets, "gemini_api_key", lambda: "key")
    rig_mod.update({"cameras.hand.serial": "1", "cameras.external.serial": "2"})
    seen = {}
    real = tiptop_factory._resolve

    def spy(rig, rig_options, options):
        seen["rig_options"] = dict(rig_options or {})
        return real(rig, rig_options, options)

    monkeypatch.setattr(tiptop_factory, "_resolve", spy)
    given = {"robot": {"port": 6001}}
    with pytest.raises(TandemError):  # no extrinsics on this stand-in machine: stopped after resolving
        FACTORY.create(
            BackendContext(
                profile=None,
                session_dir=tmp_path / "s",
                output_dir=tmp_path / "o",
                execute=False,
                rig_options=given,
                runtime_dir=tmp_path / "rt",
            )
        )
    assert seen["rig_options"] == given


def test_a_plain_machine_setting_in_a_profile_is_told_to_rig_set_itself_not_a_key_inside_it(monkeypatch):
    isolate_registry(monkeypatch)

    class Sensor(ToyPlanner):
        info = PlannerInfo(name="sensor-toy", display_name="Sensor toy", summary="x")
        CAPABILITIES = replace(ToyPlanner.CAPABILITIES, name="sensor-toy")
        RIG_OPTIONS = {"sensor": "the sensor's address", "robot": "a block of the robot's settings"}

    with pytest.raises(TandemError) as plain:
        Sensor.validate_options({"sensor": "10.0.0.3"})
    assert "`tandem rig set planners.sensor-toy.sensor VALUE`" in plain.value.hint
    with pytest.raises(TandemError) as block:
        Sensor.validate_options({"robot": {"port": 1}})
    assert "`tandem rig set planners.sensor-toy.robot.KEY VALUE`" in block.value.hint

    registry.register_backend("sensor-toy", Sensor)
    with pytest.raises(TandemError) as through_registry:
        registry.validate_options("sensor-toy", {"sensor": "10.0.0.3"})
    assert through_registry.value.hint == plain.value.hint, "one wording, whichever check finds it"


def test_a_rig_given_to_the_kit_is_the_one_it_builds_for(tmp_path):
    class Elsewhere(TestAPlannerThatReadsTheRigInItsConstructorConforms):
        def rig(self, tmp_path: Path):
            rig = stand_in_rig(tmp_path)
            rig.robot.host = "nuc.lab"
            return rig

    assert Elsewhere().make_backend(tmp_path).host == "nuc.lab"
