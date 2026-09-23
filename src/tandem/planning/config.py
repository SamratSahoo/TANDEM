"""Resolved phase-planning settings, as the planner package sees them.

A plain frozen dataclass rather than the pydantic model in ``tandem.core.profiles``, for one reason:
this package has to be usable with no profile, no runtime and no robot -- that is what makes
``tandem plan`` able to check a decomposition before anyone goes near the arm. ``HitlSpec`` builds
one of these; so does a test, in one line.
"""

from __future__ import annotations

from dataclasses import dataclass

from tandem.core import names

# Planning a task into phases is reasoning, not spatial grounding, so it does NOT reuse the detection
# model. A detector runs with thinking disabled because it is localising boxes; asking that same
# configuration to sequence a task and invent a predicate gets a worse answer than a general model
# with reasoning left on.
DEFAULT_PROPOSAL_MODEL = "gemini-2.5-pro"
# Grounding ("is the cloth folded?") is a visual judgement over one image. Flash is enough, and it is
# the query a run pays for repeatedly.
DEFAULT_VLM_MODEL = "gemini-2.5-flash"

# What to do when a robot phase cannot be planned. `abort` is the default, for the reason given at
# PlanningConfig.on_robot_phase_failure. `teleop` is the one the old design could not express at all:
# the executor split was frozen at proposal time inside the planner's process, so a phase the planner
# turned out not to be able to do could only end the attempt.
ON_FAILURE_CHOICES = ("abort", "teleop", "replan")

# What becomes of a trial whose human phase never verifies. See
# PlanningConfig.on_verification_failure.
ON_VERIFICATION_FAILURE_CHOICES = ("exclude", "label")

# What a human executor may be called: a name, not a path or an import string. Only the SHAPE is
# checked here -- whether anything is registered under it is the executor registry's question, and it
# is the only thing that knows. The profile asks it (HitlSpec._executor_name, like planner.backend), so
# a misspelt name is refused when the profile loads; a PlanningConfig built in code is checked by the
# registry when the executor is created, not here, so this module stays free of the registry. The shape
# is a planner's too (tandem.core.names): lowercase, so `ACT` and `act` cannot name two executors.
HUMAN_EXECUTOR_NAME = names.NAME


@dataclass(frozen=True)
class PlanningConfig:
    """Resolved settings. ``enabled`` False means nothing in this package ever runs."""

    enabled: bool = False
    proposal_model: str = DEFAULT_PROPOSAL_MODEL
    vlm_model: str = DEFAULT_VLM_MODEL
    # Reprompts allowed when a proposal comes back unparseable or fails validation. The error message
    # is fed back to the model, which is what makes a second attempt worth making at all. It also
    # bounds `on_robot_phase_failure: replan`: at most this many re-plans per trial.
    max_attempts: int = 3
    # Classify the plan's invented predicates on the FIRST image, before anything runs. Off by
    # default: a human is being asked precisely because the predicate is false, so it costs one model
    # call per grounding to learn what was already assumed. Worth turning on for a scene that may
    # start already solved.
    classify_initial: bool = False
    # Extra chances the operator gets at a human phase the check says did not happen.
    # 1 = show what is missing and hand the arm back once more; 0 = fail on the first bad verdict.
    verify_retries: int = 1
    # Treat a failed verification as a rollout failure. False records the verdict and carries on,
    # which is what you want while calibrating the classifier prompts. What then becomes of the
    # failed trial is on_verification_failure.
    verify_enforced: bool = True
    # What becomes of a trial whose human phase still does not verify once its retries are spent
    # (and verify_enforced is on).
    #   exclude  the paper's rule: the trial is terminated and kept out of the dataset. It is filed
    #            under failure/ with `excluded: true` and failure_stage "verification", its failing
    #            verdicts and raw legs are kept for inspection, and there is no label prompt.
    #   label    the operator is asked, as for any other trial, and their answer decides.
    # `exclude` is the default because a check the operator can overrule by answering "success" is
    # not a filter on what the dataset contains. `label` is for calibrating the classifier, where
    # every disagreement between the operator and the check is the data point.
    on_verification_failure: str = "exclude"
    # Verify the LAST phase too, when it is a human one. The reference implementation skipped it, on
    # the grounds that the operator's label settles the same question a moment later. Under
    # `on_verification_failure: exclude` it no longer does -- the label is never asked for a trial
    # that failed a check -- and the paper verifies every human phase. False leaves that phase to the
    # label alone.
    verify_final_phase: bool = True

    # ---- Which halves of the operator contract are put to a camera ----
    # Every human phase declares an operator -- preconditions, add effects, delete effects -- and a
    # robot phase's goal is its own add effects. These four say which halves of that contract a
    # camera is actually asked about. Each costs one model call per checkable atom, paid with the arm
    # parked, which is why only ONE is on by default:
    #
    #   * a human phase's EFFECTS are the only evidence the step happened at all. Nothing else in the
    #     system can tell you whether the box got opened, so this stays on. Add effects must hold;
    #     delete effects must no longer hold.
    #   * a human phase's PRECONDITIONS are usually redundant: check_plan_effects below has already
    #     proved, symbolically and for free, that the plan establishes them. The paper's experiments
    #     ran with it off. Worth turning on when a human phase keeps failing and it is not clear
    #     whether it was ever set up properly.
    #   * a robot leg's PRECONDITIONS guard against planning onto a stale belief -- the previous human
    #     phase may not have done what it was verified as doing. Real, but it lands between perception
    #     and the planner with the arm parked, and a collection run would rather spend that time
    #     collecting.
    #   * a robot leg's EFFECTS are the planner's own business: it either executed a plan for
    #     On(toy, box) or reported that it could not, and a third-person camera is a worse witness to
    #     that than the arm's own report.
    #
    # A failed check is recorded in hitl.json either way. Whether it STOPS the trial is
    # verify_enforced (human effects) and precondition_enforced (either precondition); nothing
    # enforces a robot leg's effects.
    check_human_effects: bool = True
    check_human_preconditions: bool = False
    check_tamp_preconditions: bool = False
    check_tamp_effects: bool = False
    # Treat an unmet precondition as a reason not to proceed. Off by default, for the rule that one
    # classifier call must not cost the operator a demonstration: the verdict is recorded and the
    # phase goes ahead. On, an unmet precondition ends the trial before the arm is handed over
    # (human) or before the planner is asked for a plan (robot).
    precondition_enforced: bool = False
    # Check the declared operators against each other before the arm moves: walk the phase list
    # symbolically and refuse a plan that deletes a precondition a later phase needs -- "the human
    # closes the box" ordered before "the robot puts the toy in the box". Costs no model call, and the
    # refusal goes back to the proposer through the repair loop, which is why it is on: a plan that
    # does not hang together is fixed while it is still text.
    check_plan_effects: bool = True

    # Write every image sent to the model, and a rendered PNG of what it answered, into `vlm/` beside
    # each episode (plus index.jsonl with the full prompts and replies). Rejected attempts included.
    # On by default: when a run goes wrong the question is almost always "what did the model actually
    # see, and what did it say", and that is unanswerable after the fact without this.
    save_vlm_io: bool = True
    # SQLite cache for PROPOSAL responses only, keyed on the model, the prompt and a noise-robust
    # hash of the image. Worth setting while iterating on prompts, where the same scene and
    # instruction are proposed over and over. Never applied to grounding or verification -- see
    # cache.ProposalCache for why that would be unsafe.
    cache_path: str | None = None

    # ---- Carrying the plan out ----
    # What happens when the planner cannot plan a robot phase.
    #   abort   the trial ends as a failure (failure_stage "tamp_planning")
    #   teleop  the sub-goal is described to the operator, who does it by hand, verified exactly as
    #           any other human phase, and the task carries on
    #   replan  the failure is fed back to the proposer, which decomposes the task again (at most
    #           max_attempts times per trial, then as abort)
    # `abort` is the default because it is what the method's numbers mean: the paper counts a TAMP
    # failure as a trial failure. A teleop fallback turns a robot phase into a human one, so a dataset
    # collected with it credits the method with trials it did not complete as designed and
    # understates the human effort they cost. `teleop` is still the right choice for a collection run
    # that only wants demonstrations.
    #
    # There is deliberately no such choice for a leg that was planned and then failed to EXECUTE:
    # that always ends the trial (failure_stage "tamp_execution"). The arm is somewhere no plan put
    # it, and the scene is no longer the one the next phase was planned against.
    on_robot_phase_failure: str = "abort"
    # Hand consecutive robot phases to the planner as ONE goal wherever that is sound (see
    # feasibility.conjoinable_run for when it is not): one plan, one continuous motion, and no
    # re-perception in the middle for the object labels to drift across. False gives every robot
    # phase its own leg and its own perception pass: the paper's "re-perceive after every phase" read
    # strictly, at the cost of the arm stopping between phases.
    conjoin_robot_phases: bool = True
    # Who carries out a human phase, by registered name (tandem.executors; another package registers
    # one through the `tandem.human_executors` entry point). "teleop" is a person driving the arm --
    # the executor the method was built around, and the only one that ships. This selects how a phase
    # is DONE, never what it asks for: the phase, its operator and the check afterwards are the same
    # whoever carries it out.
    human_executor: str = "teleop"
    # Accept "done" for a human phase that was never teleoperated. With recording on, that phase has
    # no leg, so the episode is missing exactly the demonstration the trial exists to capture while
    # looking complete. Always allowed when nothing is being recorded; with recording on, only when
    # this is set -- for staging part of a scene by hand mid-task, knowingly.
    allow_unrecorded_human_phase: bool = False
    # Which camera the verification frame comes from. A third-person view by default: after a
    # hand-off the arm is wherever the operator left it, so a wrist view points nowhere useful.
    verification_camera: str = "external"

    def __post_init__(self) -> None:
        if self.on_robot_phase_failure not in ON_FAILURE_CHOICES:
            raise ValueError(
                f"on_robot_phase_failure must be one of {', '.join(ON_FAILURE_CHOICES)}, "
                f"got {self.on_robot_phase_failure!r}"
            )
        if self.on_verification_failure not in ON_VERIFICATION_FAILURE_CHOICES:
            raise ValueError(
                f"on_verification_failure must be one of {', '.join(ON_VERIFICATION_FAILURE_CHOICES)}, "
                f"got {self.on_verification_failure!r}"
            )
        if not isinstance(self.human_executor, str) or not HUMAN_EXECUTOR_NAME.fullmatch(self.human_executor):
            raise ValueError(
                f"human_executor must be the name of a registered human executor ({names.RULE}), "
                f"got {self.human_executor!r}"
            )
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {self.max_attempts}")
        if self.verify_retries < 0:
            raise ValueError(f"verify_retries must be >= 0, got {self.verify_retries}")
