"""What TiPToP (over cuTAMP) can be asked for, declared.

Every constant here used to be an import from ``cutamp`` inside the phase planner. That import is
what tied the orchestration layer to one planner, and it could not survive tandem's promise that
``pip install tandem-tamp`` works on a laptop: the parent process has no cuTAMP on its path and never
will. So the facts are stated, and ``tests/test_planners.py`` pins them against the real cuTAMP
whenever the vendored tree happens to be importable -- declared here, verified there.

The facts themselves come from ``cutamp/tamp_domain.py``: six operators (MoveFree, MoveHolding, Pick,
Place, Push, PushStick) over nineteen fluents, of which ``create_tamp_environment`` reads exactly two
as goals.
"""

from __future__ import annotations

from tandem.planners.base import Capabilities
from tandem.planning.symbols import Parameter, Predicate

# The goal language. `create_tamp_environment` reads on(...) and holding(...) and supplies HandEmpty
# itself, which is why HandEmpty is statable but has no wire name: a proposal may say it, a human is
# shown it, and it is dropped on the way to the planner rather than sent and ignored.
ON = Predicate("On", (Parameter("obj", "movable"), Parameter("surface", "surface")))
HOLDING = Predicate("Holding", (Parameter("obj", "movable"),))
HAND_EMPTY = Predicate("HandEmpty", ())

# Every fluent in the cuTAMP domain. Broader than the goal language on purpose: motion bookkeeping
# (At, CanMove, JustMoved) and type declarations (IsMovable, IsSurface) are not goal-statable, but a
# proposal that invented a predicate under one of these names would collide with the planner's own.
ALL_FLUENTS = frozenset(
    {
        "At",
        "ButtonPushed",
        "CanMove",
        "CanPush",
        "HandEmpty",
        "HasNotPickedUp",
        "HeldByGiver",
        "HeldByGiverGrasp",
        "HeldByTaker",
        "HeldByTakerGrasp",
        "Holding",
        "HoldingWithGrasp",
        "IsButton",
        "IsMovable",
        "IsStick",
        "IsSurface",
        "JustMoved",
        "On",
        "PushedWithStick",
    }
)

# What some operator can add, plus what a fresh scene already has. An atom over anything else can
# never become true, however long the search runs -- and cuTAMP's search has no bound of any kind, so
# "never" means "does not terminate", not "fails".
ACHIEVABLE = frozenset(
    # add effects of MoveFree / MoveHolding / Pick / Place / Push / PushStick
    {
        "At",
        "ButtonPushed",
        "CanMove",
        "HandEmpty",
        "Holding",
        "HoldingWithGrasp",
        "JustMoved",
        "On",
        "PushedWithStick",
    }
    # true in get_initial_state() before anything runs
    | {"HasNotPickedUp", "IsMovable", "IsSurface"}
)

# The paragraphs of the evaluated phase-segmentation prompt (the paper's Appendix B, LJ tiptop
# cf75a68) that are statements about cuTAMP's goal language, VERBATIM -- tests/test_prompts.py holds
# the render to that prompt byte for byte, so a word changed here is a change to the method. The slot
# contract is tandem.planning.prompts.PROMPT_SLOTS. Every slot is filled: a generic paragraph in any
# of them would be a prompt nobody evaluated this planner with.
PROMPT_FRAGMENTS = {
    # What replaced the worked examples: the loose-placement / precise-fit line, stated as a rule.
    "placement_semantics": (
        "WHAT On MEANS. On({0}, {1}) means {0} has been let go of and is RESTING somewhere on or inside "
        "{1}, and nothing more. That covers setting something down on a surface AND dropping it into "
        "anything with an opening it fits through -- a bin, a basket, a pot, a drawer that is pulled "
        'out. Words like "in", "into" and "inside" in an instruction usually mean exactly this: if the '
        "object only has to end up loosely somewhere within the container, it is On(object, container) "
        "and it is the robot's job -- including when a human phase first had to open, uncover or empty "
        "the container to make room for it. What On CANNOT say is a precise fit. The robot places by "
        "opening its gripper above a spot, so it cannot fit, insert, slot, thread, plug, screw, seat, "
        "close, or align one thing to another. If a clause needs the object in one particular position "
        "or orientation -- a key in a lock, a plug in a socket, a lid seated on a jar, a peg in its "
        "matching hole -- that is a HUMAN phase, however much it looks like a pick-and-place. The same "
        "goes for anything soft that has to be spread, draped, wrapped or folded over something: that "
        "is manipulation, not placement. Say either with an invented predicate, not with On."
    ),
    "precondition_vocabulary": (
        "Write them with On, Holding, HandEmpty or a predicate you invented. HandEmpty() belongs here "
        "whenever the person needs the robot to be out of the way -- which is almost always."
    ),
    "delete_effect_example": (
        "If the person moves something off a surface, the old On(...) is a delete effect; if they "
        "unplug what an earlier phase plugged in, the IsPluggedIn(...) is."
    ),
    "work_division": (
        "- Give the robot every pick-and-place. A human phase that includes moving an object from A to "
        "B is taking the robot's work away from it."
    ),
    "intermediate_state_example": (
        '- Intermediate states are fine and often necessary. "Take the mug off the notebook, sign the '
        'notebook, put the mug back" is three phases, and the first one ends with the mug somewhere '
        "else -- On(mug, table). Each robot phase is planned fresh, so an object may be picked up in "
        "more than one phase."
    ),
    "robot_phase_rules": (
        "- Give a whole pick-and-place ONE phase, ending with On(?obj, ?surface). Do not split it into "
        'a "pick it up" phase and a "put it down" phase; the robot does both as one piece of work.\n'
        "- Two ROBOT phases in a row must not move the same object twice. The second placement throws "
        "the first one away, so the first is wasted motion -- and it almost always means a step that "
        "is not really a pick-and-place was given to the robot. If a human phase belongs between them, "
        "put it there; if the second phase is the one the robot cannot do, make IT the human phase."
    ),
}

CAPABILITIES = Capabilities(
    name="tiptop",
    goal_predicates={"On": ON, "Holding": HOLDING, "HandEmpty": HAND_EMPTY},
    robot_description="pick an object up and place it on a surface",
    goal_predicate_wire_names={"On": "on", "Holding": "holding"},
    achievable_predicates=ACHIEVABLE,
    reserved_predicate_names=ALL_FLUENTS,
    movable_type="movable",
    surface_type="surface",
    predicate_descriptions={
        "On": "{0} is resting on top of {1}",
        "Holding": "the robot's gripper is holding {0}",
        "HandEmpty": "the robot's gripper is empty",
    },
    # A camera can settle On. Holding and HandEmpty are deliberately absent: verification runs on a
    # third-person frame in which the gripper is usually out of shot, and the classifier is told to
    # answer false when it cannot see the statement to be true -- so checking them would fail a phase
    # over something the robot knows exactly.
    checkable_predicates=frozenset({"On"}),
    # cuTAMP's Pick requires and DELETES HasNotPickedUp(obj), and nothing re-adds it, so one plan
    # picks each object at most once. Two phases that move the same object cannot be conjoined.
    one_pick_per_object=True,
    # get_initial_state() contains no On atom at all: it is a pure function of the object names.
    # Every goal is therefore planned from the same clean state, which is what makes conjoining
    # consecutive robot phases sound -- and what lets an object be picked up in more than one phase.
    initial_state_is_clean=True,
    # Upstream's execute_cutamp_plan takes no should_stop and has no mid-plan seam, so a preempt is
    # an abort. The fork that had one is what this refactor is un-forking.
    supports_cooperative_stop=False,
    # Upstream cuTAMP has no reuse_plan_skeleton and never did; the vendored tree's copy was a local
    # commit that was never pushed. Every leg pays a fresh symbolic search.
    supports_skeleton_reuse=False,
    # The Appendix-B paragraphs that spell out cuTAMP's goal language; see PROMPT_FRAGMENTS above.
    prompt_fragments=PROMPT_FRAGMENTS,
    # An object rests on one thing at a time. cuTAMP's Place deletes no On atom -- its initial state
    # has none to delete -- but in the world the toy is no longer where it was, and the contract check
    # reasons about the world across legs, not about one leg's plan.
    exclusive_arguments={"On": 0},
    # The object placed or held is the one that moves; the surface it lands on does not.
    moved_arguments={"On": 0, "Holding": 0},
    # The two cuTAMP operators a pick-and-place goal is achieved with, with their motion-level
    # parameters (conf, traj, grasp) dropped. MoveFree/MoveHolding are the motion between them, and
    # Push/PushStick serve goals create_tamp_environment never builds.
    robot_operators=("Pick(?obj: movable)", "Place(?obj: movable, ?surface: surface)"),
    # Both can be honoured in the sidecar without touching tiptop: `movables` by rebuilding cuTAMP's
    # TAMPEnvironment with every other object demoted to a static, `return_home=False` by trimming
    # the plan's trailing GoToInitial (tiptop.goal_clearing.drop_return_to_initial). Neither needs
    # a fork.
    supports_movable_restriction=True,
    supports_return_home=True,
)
