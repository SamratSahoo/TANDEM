"""Which labels a drifted object may be re-bound to: through the phase loop, not beside it.

The pool is ``detected - scene_types.all_names``: every label perception produced that is NOT already
one of the plan's objects -- including objects only a COMPLETED phase named, such as a toy the robot
has already put away. ``detected - objects_named()`` covers only the phases still to come, so the
put-away toy would join the pool, and ``match_drifted_names(["green_toy"], ["toy"])`` folds the two
into one: the next leg is planned against the object that is already sorted.

tests/test_planning.py states the set facts this rests on; these tests drive the loop, so replacing the
pool in core/phase_loop.py with either wrong one fails here.
"""

from __future__ import annotations

import test_robot_legs
from test_robot_legs import OPEN_THE_BOX, goals, proposal, ran

# The robot-leg tests' phase-loop rig, shared rather than copied.
rig = test_robot_legs.rig

PUT_AWAY = {
    "executor": "robot",
    "description": "put the toy away on the table",
    "atoms": [{"predicate": "On", "args": ["toy", "table"]}],
}
GREEN_IN = {
    "executor": "robot",
    "description": "put the green toy in the box",
    "atoms": [{"predicate": "On", "args": ["green_toy", "white_box"]}],
}
PLAN = proposal(PUT_AWAY, OPEN_THE_BOX, GREEN_IN)
LABELS = ("toy", "green_toy", "white_box")


def test_an_object_a_finished_phase_owns_is_never_what_a_drifted_one_is_rebound_to(rig):
    # After the person's step, green_toy is not seen any more; toy -- put away by the first phase, and
    # named by no phase still to come -- is. It is still the plan's object, so it is not a candidate,
    # and with no other the trial ends, rather than the leg re-planning the put-away toy.
    r = rig(plan=PLAN, backend_kwargs={"labels": LABELS, "drifted_labels": ("toy", "white_box")})
    outcome = r.run()

    assert ran(r.backend) == [0], "the last leg was planned against the object already put away"
    assert (outcome.outcome, outcome.failure_stage) == ("failure", "tamp_planning")
    assert "green_toy" in outcome.reason


def test_a_drifted_object_is_rebound_to_the_one_new_label_and_not_to_an_owned_one(rig):
    # The same, with the green toy seen again under a longer name: the one label the plan does not own.
    r = rig(plan=PLAN, backend_kwargs={"labels": LABELS, "drifted_labels": ("toy", "small_green_toy", "white_box")})
    outcome = r.run()

    assert outcome.plan.finished
    assert goals(r.backend)[-1] == [["on", "small_green_toy", "white_box"]]
