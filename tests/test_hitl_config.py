"""The declarative surface of phase planning: the `hitl` keys, and what a planner declares.

Two things are defined twice on purpose and have to agree. The `hitl` block is a pydantic model in
the profile (it is what a person edits, and what is validated with a message they can act on) and a
plain dataclass in ``tandem.planning`` (so the phase planner runs with no profile at all). A key the
dataclass has and the profile does not is a setting nobody can turn; a default that differs between
them means ``tandem plan`` and a live session disagree about the same profile. The shipped template
is the third copy, and the one people actually read.

The defaults are the paper's. Each one pinned here is a decision somebody made for a reason, and a
test that fails when it moves is how that reason gets reread before it is overturned.
"""

from __future__ import annotations

import dataclasses

import pytest
from pydantic import ValidationError
from ruamel.yaml import YAML

from tandem import resources
from tandem.core.profiles import HitlSpec, Profile
from tandem.planners.base import Capabilities
from tandem.planners.tiptop.capabilities import CAPABILITIES as TIPTOP
from tandem.planning.config import ON_FAILURE_CHOICES, PlanningConfig
from tandem.planning.symbols import Parameter, Predicate

# Every key this pass added, at the default it was given.
NEW_DEFAULTS = {
    "on_verification_failure": "exclude",
    "check_plan_effects": True,
    "check_human_preconditions": False,
    "check_human_effects": True,
    "check_tamp_preconditions": False,
    "check_tamp_effects": False,
    "precondition_enforced": False,
    "verify_final_phase": True,
    "conjoin_robot_phases": True,
    "human_executor": "teleop",
    "allow_unrecorded_human_phase": False,
}


# --- defaults -------------------------------------------------------------------------------------


@pytest.mark.parametrize("build", [lambda: PlanningConfig(enabled=True), lambda: HitlSpec(enabled=True)])
def test_only_the_check_nothing_else_can_answer_is_on_by_default(build):
    # A human phase's effects are the only evidence the step happened at all. Every other half of
    # the contract is either provable for free at plan time or something the robot already knows,
    # and each costs a model call with the arm parked -- so they are opt-in.
    cfg = build()
    assert cfg.check_human_effects is True
    assert (cfg.check_human_preconditions, cfg.check_tamp_preconditions, cfg.check_tamp_effects) == (
        False,
        False,
        False,
    )
    # A failed precondition is observational until asked for otherwise.
    assert cfg.precondition_enforced is False
    # The free one stays on: symbolic, no model call, and it feeds the repair loop.
    assert cfg.check_plan_effects is True


@pytest.mark.parametrize("build", [PlanningConfig, HitlSpec])
def test_every_new_key_defaults_to_the_decided_value(build):
    cfg = build()
    assert {key: getattr(cfg, key) for key in NEW_DEFAULTS} == NEW_DEFAULTS


@pytest.mark.parametrize("build", [PlanningConfig, HitlSpec])
def test_a_phase_the_planner_cannot_plan_ends_the_trial_by_default(build):
    # The paper counts a TAMP failure as a trial failure. A teleop fallback by default would turn
    # the robot's phase into the operator's and inflate the human effort a dataset is said to cost.
    assert build().on_robot_phase_failure == "abort"
    # The other two stay available; `abort` is a default, not the only policy.
    assert set(ON_FAILURE_CHOICES) == {"abort", "teleop", "replan"}
    for policy in ON_FAILURE_CHOICES:
        assert build(on_robot_phase_failure=policy).on_robot_phase_failure == policy


def test_a_trial_that_fails_verification_is_excluded_by_default_and_labeling_is_opt_in():
    assert HitlSpec().on_verification_failure == "exclude"
    assert HitlSpec(on_verification_failure="label").to_planning_config().on_verification_failure == "label"


# --- the two definitions agree --------------------------------------------------------------------


def test_the_profile_and_the_planner_know_the_same_keys():
    profile_keys = set(HitlSpec.model_fields)
    planner_keys = {f.name for f in dataclasses.fields(PlanningConfig)}
    assert profile_keys == planner_keys, (
        f"only in the profile: {sorted(profile_keys - planner_keys)}; "
        f"only in the planner: {sorted(planner_keys - profile_keys)}"
    )


def test_the_profile_and_the_planner_agree_on_every_default():
    # Compared as a whole, so a key added to both with different defaults cannot slip through.
    assert HitlSpec().to_planning_config() == PlanningConfig()


def test_every_key_survives_the_trip_to_the_planner():
    # Every value flipped from its default, so a key to_planning_config dropped would come back as
    # the default and fail here instead of silently in a session.
    changed = {
        "enabled": True,
        "proposal_model": "some-other-model",
        "vlm_model": "another-model",
        "max_attempts": 5,
        "classify_initial": True,
        "verify_retries": 3,
        "verify_enforced": False,
        "on_verification_failure": "label",
        "verify_final_phase": False,
        "check_human_effects": False,
        "check_human_preconditions": True,
        "check_tamp_preconditions": True,
        "check_tamp_effects": True,
        "precondition_enforced": True,
        "check_plan_effects": False,
        "save_vlm_io": False,
        "cache_path": "proposals.sqlite",
        "on_robot_phase_failure": "replan",
        "conjoin_robot_phases": False,
        "human_executor": "diffusion-policy",
        "allow_unrecorded_human_phase": True,
        "verification_camera": "hand",
    }
    assert set(changed) == set(HitlSpec.model_fields), "a key was added without being tested here"
    resolved = HitlSpec(**changed).to_planning_config()
    assert dataclasses.asdict(resolved) == changed


# --- the shipped template -------------------------------------------------------------------------


def _template_hitl_block() -> dict:
    with resources.path("profile_template.yml").open() as fh:
        return dict(YAML(typ="safe").load(fh)["hitl"])


def test_the_template_spells_out_every_hitl_key():
    # Loading fills a missing key with its default, so a key absent from the template still works --
    # it is merely undiscoverable, which for a setting that changes what a dataset contains is the
    # same as not having it.
    block = _template_hitl_block()
    missing = sorted(set(HitlSpec.model_fields) - set(block))
    assert not missing, f"the template does not document: {missing}"
    assert not set(block) - set(HitlSpec.model_fields)


def test_the_template_ships_the_same_defaults_the_code_has():
    # `tandem init` writes the template, not HitlSpec(). A value that differs between them means a new
    # profile and a profile with the key deleted behave differently.
    block = _template_hitl_block()
    defaults = HitlSpec().model_dump(mode="python")
    assert {key: block[key] for key in defaults} == defaults


# --- validation -----------------------------------------------------------------------------------


def test_an_unknown_verification_policy_is_refused_in_both_places():
    with pytest.raises(ValidationError, match="on_verification_failure"):
        Profile.model_validate({"name": "x", "hitl": {"on_verification_failure": "ignore"}})
    with pytest.raises(ValueError, match="on_verification_failure must be one of exclude, label"):
        PlanningConfig(on_verification_failure="ignore")


@pytest.mark.parametrize("name", ["", " teleop", "teleop\n", "1teleop", "tele op", "../teleop", "pkg.mod:Cls"])
def test_a_human_executor_that_is_not_a_name_is_refused_in_both_places(name):
    # A path or an import string here is somebody configuring the registry by hand, and a stray
    # space is a typo -- all of them would otherwise surface as "no such executor" minutes into a
    # session, or worse, be stripped and resolve to something else.
    with pytest.raises(ValidationError, match="human_executor"):
        Profile.model_validate({"name": "x", "hitl": {"human_executor": name}})
    with pytest.raises(ValueError, match="human_executor must be the name"):
        PlanningConfig(human_executor=name)


@pytest.mark.parametrize("name", ["teleop", "diffusion-policy", "act_v2", "ACT"])
def test_an_executor_is_checked_for_shape_not_for_being_installed(name):
    # Whether anything is registered under the name is the executor registry's question. A profile
    # naming a third-party executor must still load on a machine that has not installed it -- the
    # laptop reading its trajectories, say.
    assert HitlSpec(human_executor=name).to_planning_config().human_executor == name


def test_a_misspelled_new_key_is_still_an_error():
    # `extra: forbid` is the only validation layer the block has. A misspelling of one of the new
    # switches must not silently leave the check it was meant to turn off running.
    with pytest.raises(ValidationError, match="check_human_precondition"):
        Profile.model_validate({"name": "x", "hitl": {"check_human_precondition": True}})


# --- what a planner declares ----------------------------------------------------------------------


def test_a_bare_declaration_claims_nothing_it_was_not_told():
    # A new backend that declares only its name must not inherit tiptop's answers. Every one of these
    # makes the phase planner DO something -- restrict the planner's movables, drop the trip home,
    # derive a delete effect -- and doing it to a planner that never agreed is a silent behaviour
    # change, not a sensible default.
    caps = Capabilities(name="bare")
    assert caps.prompt_fragments == {}
    assert caps.exclusive_arguments == {}
    assert caps.moved_arguments == {}
    assert caps.robot_operators == ()
    assert caps.supports_movable_restriction is False
    assert caps.supports_return_home is False


def test_tiptop_declares_what_its_planner_does():
    assert TIPTOP.exclusive_arguments == {"On": 0}
    assert TIPTOP.moved_arguments == {"On": 0, "Holding": 0}
    assert TIPTOP.robot_operators == ("Pick(?obj: movable)", "Place(?obj: movable, ?surface: surface)")
    assert TIPTOP.supports_movable_restriction is True
    assert TIPTOP.supports_return_home is True
    # Filled together with the prompt it belongs to, not before.
    assert TIPTOP.prompt_fragments == {}


def _argument_is_a(caps: Capabilities, predicate: str, index: int, type_: str) -> bool:
    parameters = caps.goal_predicates[predicate].parameters
    return 0 <= index < len(parameters) and parameters[index].type == type_


def test_tiptops_argument_positions_point_at_the_objects_it_moves():
    # The positions are bare integers, so an off-by-one would make On(toy, box) "move" the box --
    # restricting the planner to the wrong objects and blaming the wrong phase for wasted motion.
    # Checked against the declared goal language rather than restated.
    for mapping in (TIPTOP.exclusive_arguments, TIPTOP.moved_arguments):
        for predicate, index in mapping.items():
            assert predicate in TIPTOP.goal_predicates
            assert _argument_is_a(TIPTOP, predicate, index, TIPTOP.movable_type), (predicate, index)


def test_tiptops_robot_operators_are_stated_in_its_own_types():
    declared_types = {TIPTOP.movable_type, TIPTOP.surface_type}
    for signature in TIPTOP.robot_operators:
        name, _, rest = signature.partition("(")
        assert name.isidentifier() and rest.endswith(")"), signature
        types = {arg.split(":")[1].strip() for arg in rest[:-1].split(",")}
        assert types <= declared_types, signature


def test_a_backend_with_its_own_vocabulary_declares_these_the_same_way():
    # The same fields carry a different planner's facts with no change on tandem's side.
    inside = Predicate("Inside", (Parameter("obj", "thing"), Parameter("bin", "container")))
    caps = Capabilities(
        name="toy-planner",
        goal_predicates={"Inside": inside},
        movable_type="thing",
        surface_type="container",
        exclusive_arguments={"Inside": 0},
        moved_arguments={"Inside": 0},
        robot_operators=("Drop(?obj: thing, ?bin: container)",),
    )
    assert _argument_is_a(caps, "Inside", caps.moved_arguments["Inside"], "thing")
    assert caps.supports_movable_restriction is False
