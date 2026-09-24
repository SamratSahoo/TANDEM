"""A planner's settings are of two kinds, and it says which: its task's (OPTIONS) and its machine's (RIG_OPTIONS).

A task's settings live in each profile's planner.options; a machine's -- a robot shim's ports, a server's
address -- in rig.yml's ``planners.<name>``, once, for every profile. Nothing in tandem knows which of
TiPToP's settings are which: TiPToP declares it, as any planner does, and the SDK, the registry, the rig
and the conformance kit hold every planner to its declaration.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers import isolate_registry
from toy_planner import ToyPlanner

from tandem.core import profiles
from tandem.core import rig as rig_mod
from tandem.core.errors import RigInvalid, TandemError
from tandem.planners import registry
from tandem.planners.base import BackendContext
from tandem.planners.testing import ConformanceError, PlannerConformance


@pytest.fixture(autouse=True)
def _registry(monkeypatch):
    isolate_registry(monkeypatch)


class RigToy(ToyPlanner):
    """The toy planner with a machine setting: the address of the bin sensor on this machine's table."""

    RIG_OPTIONS = {"sensor": "the bin sensor's address on this machine"}


def _context(**fields) -> BackendContext:
    return BackendContext(profile=None, session_dir=Path("."), output_dir=Path("."), **fields)


# --- the SDK ---------------------------------------------------------------------------------------------


def test_the_default_check_refuses_a_machine_setting_it_does_not_declare():
    assert RigToy.validate_rig_options({"sensor": "10.0.0.7"}) == {"sensor": "10.0.0.7"}
    with pytest.raises(TandemError, match="does not read the machine settings sensr") as caught:
        RigToy.validate_rig_options({"sensr": "10.0.0.7"})
    assert "Did you mean 'sensor'?" in caught.value.hint
    with pytest.raises(TandemError, match="reads no machine settings|does not read the machine settings"):
        ToyPlanner.validate_rig_options({"sensor": "x"})


def test_a_setting_in_the_wrong_place_is_told_where_it_belongs():
    with pytest.raises(TandemError) as caught:
        RigToy.validate_options({"sensor": "10.0.0.7"})
    assert "machine setting" in caught.value.message and "rig.yml" in caught.value.message
    assert "tandem rig set planners.toy.sensor" in caught.value.hint
    with pytest.raises(TandemError) as caught:
        RigToy.validate_rig_options({"items": ["duck"]})
    assert "task setting" in caught.value.message and "planner.options" in caught.value.message


def test_a_key_cannot_be_both_the_tasks_and_the_machines():
    with pytest.raises(TandemError, match="in both OPTIONS and RIG_OPTIONS"):
        type("Both", (ToyPlanner,), {"__module__": __name__, "RIG_OPTIONS": {"items": "the items, again"}})
    with pytest.raises(TandemError, match="RIG_OPTIONS must map"):
        type("Bad", (ToyPlanner,), {"__module__": __name__, "RIG_OPTIONS": {"sensor": 3}})


def test_create_hands_the_backend_both_kinds_as_checked():
    class Normalising(RigToy):
        @classmethod
        def validate_rig_options(cls, options):
            checked = super().validate_rig_options(options)
            return {"sensor": checked.get("sensor", "localhost")}

    built = Normalising.create(_context(options={"items": ["duck"]}, rig_options={}))
    assert built.options == {"items": ["duck"]} and built.rig_options == {"sensor": "localhost"}
    assert built.rig is None, "a context built by hand has no rig"
    with pytest.raises(TandemError, match="sensr"):
        RigToy.create(_context(rig_options={"sensr": "x"}))


# --- the registry ---------------------------------------------------------------------------------------------


def test_the_registry_refuses_each_kind_where_it_does_not_belong_whatever_the_hook_says():
    class Lenient:
        """A hand-written factory that takes anything, but declares which of its settings are which."""

        info = ToyPlanner.info
        OPTIONS = {"items": "the items"}
        RIG_OPTIONS = {"sensor": "the sensor"}

        def validate_options(self, options):
            return dict(options)

        def validate_rig_options(self, options):
            return dict(options)

    with pytest.raises(TandemError) as caught:
        registry.options_for(Lenient(), {"sensor": "10.0.0.7"})
    assert "is a machine setting of the Toy planner" in caught.value.message
    assert "tandem rig set planners.toy.sensor VALUE" in caught.value.hint
    with pytest.raises(TandemError, match="is a task setting of the Toy planner"):
        registry.rig_options_for(Lenient(), {"items": ["duck"]})
    assert registry.rig_options_for(Lenient(), {"sensor": "x"}) == {"sensor": "x"}


def test_the_registry_refuses_machine_settings_that_rig_yml_cannot_store():
    class PathSensor(RigToy):
        @classmethod
        def validate_rig_options(cls, options):
            return {"sensor": Path("/dev/sensor")}

    with pytest.raises(TandemError, match="rig.yml cannot store: options.sensor"):
        registry.rig_options_for(PathSensor, {})


def test_a_factory_without_the_hook_takes_its_machine_settings_as_written():
    from helpers import FakeFactory

    registry.register_backend("plain", FakeFactory("plain"))
    assert registry.validate_rig_options("plain", {"anything": 1}) == {"anything": 1}
    assert registry.rig_options_declared("plain") is None, "it does not say, which is not saying none"
    registry.register_backend("toy", RigToy)
    assert registry.rig_options_declared("toy") == {"sensor": "the bin sensor's address on this machine"}


def test_planners_info_lists_both_kinds_and_says_where_each_is_set(isolated_env):
    from typer.testing import CliRunner

    from tandem.cli.app import app
    from tandem.cli.planners import info_payload

    registry.register_backend("toy", RigToy)
    payload = info_payload("toy")
    assert payload["options"] == dict(RigToy.OPTIONS)
    assert payload["rig_options"] == {"sensor": "the bin sensor's address on this machine"}
    shown = CliRunner().invoke(app, ["planners", "info", "toy"], env={"COLUMNS": "200"})
    assert shown.exit_code == 0, shown.output
    assert "machine settings (rig.yml planners.toy)" in shown.output
    assert "tandem rig set planners.toy.KEY VALUE" in shown.output


# --- the rig asks the planner -----------------------------------------------------------------------------------


def test_the_rig_asks_each_installed_planner_to_check_its_block(isolated_env):
    registry.register_backend("toy", RigToy)
    rig = rig_mod.update({"planners.toy.sensor": "10.0.0.7"})
    assert rig.planners["toy"] == {"sensor": "10.0.0.7"}
    assert rig_mod.planner_options(rig, "toy") == {"sensor": "10.0.0.7"}
    with pytest.raises(RigInvalid) as caught:
        rig_mod.update({"planners.toy.sensr": "10.0.0.8"})
    assert "planners.toy" in caught.value.message and "sensr" in caught.value.message


def test_a_session_hands_the_planner_the_rig_and_its_block(profile, monkeypatch):
    from tandem.core.session import Session

    contexts: list[BackendContext] = []

    class Recording(RigToy):
        @classmethod
        def create(cls, ctx):
            contexts.append(ctx)
            return super().create(ctx)

    registry.register_backend("toy", Recording)
    rig_mod.update({"planners.toy.sensor": "10.0.0.7"})
    profile.planner = profiles.PlannerSpec(backend="toy")
    session = Session(profile, task="put the duck in the bin")
    session.start()
    try:
        (ctx,) = contexts
        assert ctx.rig is session.rig and ctx.rig.cameras.hand.serial == "14846828"
        assert dict(ctx.rig_options) == {"sensor": "10.0.0.7"}
        assert session._executor_context().rig is session.rig
    finally:
        session.stop(park=False)
        session.wait(timeout=5)


# --- the conformance kit ------------------------------------------------------------------------------------------


def _kit(planner, **attributes):
    return type("Kit", (PlannerConformance,), {"planner": planner, **attributes})()


def test_the_kit_checks_the_machine_settings_as_it_checks_the_tasks(tmp_path):
    kit = _kit(RigToy, rig_options={"sensor": "10.0.0.7"})
    kit.test_its_rig_options_check_accepts_what_it_returns()
    kit.test_its_rig_options_check_handles_no_options()
    assert kit.context(tmp_path).rig_options == {"sensor": "10.0.0.7"}


def test_the_kit_fails_a_check_that_does_not_accept_its_own_output():
    class Drifting(RigToy):
        @classmethod
        def validate_rig_options(cls, options):
            return {"sensor": str(dict(options).get("sensor", "")) + "!"}

    with pytest.raises(ConformanceError, match="must accept it unchanged"):
        _kit(Drifting).test_its_rig_options_check_accepts_what_it_returns()


def test_the_kit_fails_a_check_that_crashes_on_an_empty_block_and_accepts_one_that_says_what_it_needs():
    class Crashing(RigToy):
        @classmethod
        def validate_rig_options(cls, options):
            return {"sensor": options["sensor"]}  # KeyError on {}

    with pytest.raises(ConformanceError, match="given no machine settings it raised KeyError"):
        _kit(Crashing).test_its_rig_options_check_handles_no_options()

    class Requiring(RigToy):
        @classmethod
        def validate_rig_options(cls, options):
            if "sensor" not in (options or {}):
                raise TandemError("planners.toy.sensor is required: the bin sensor's address")
            return dict(options)

    _kit(Requiring).test_its_rig_options_check_handles_no_options()


def test_tiptop_passes_the_kits_machine_settings_checks():
    kit = _kit("tiptop")
    kit.test_its_rig_options_check_accepts_what_it_returns()
    kit.test_its_rig_options_check_handles_no_options()
    kit.test_its_options_check_accepts_what_it_returns()


# --- a new planner is scaffolded with both ---------------------------------------------------------------------------


def test_the_scaffold_declares_the_machines_settings_beside_the_tasks():
    from tandem import resources

    for template in ("scaffold/package/planner.py.tmpl", "scaffold/package/sidecar_planner.py.tmpl"):
        text = resources.read(template)
        assert "RIG_OPTIONS = {}" in text and "OPTIONS = {}" in text, template
        assert "tandem rig set" in text and "planners.{{name}}" in text, template
    assert "rig_options = {}" in resources.read("scaffold/tests/test_conformance.py.tmpl")
