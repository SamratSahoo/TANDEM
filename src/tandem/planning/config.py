"""Resolved phase-planning settings, as the planner package sees them.

A plain frozen dataclass rather than the pydantic model in ``tandem.core.profiles``, for one reason:
this package has to be usable with no profile, no runtime and no robot -- that is what makes
``tandem plan`` able to check a decomposition before anyone goes near the arm. ``HitlSpec`` builds
one of these; so does a test, in one line.
"""

from __future__ import annotations

from dataclasses import dataclass

# Planning a task into phases is reasoning, not spatial grounding, so it does NOT reuse the detection
# model. A detector runs with thinking disabled because it is localising boxes; asking that same
# configuration to sequence a task and invent a predicate gets a worse answer than a general model
# with reasoning left on.
DEFAULT_PROPOSAL_MODEL = "gemini-2.5-pro"
# Grounding ("is the cloth folded?") is a visual judgement over one image. Flash is enough, and it is
# the query a run pays for repeatedly.
DEFAULT_VLM_MODEL = "gemini-2.5-flash"

# What to do when a robot phase cannot be planned. `teleop` is the one the old design could not
# express at all: the executor split was frozen at proposal time inside the planner's process, so a
# phase the planner turned out not to be able to do could only end the attempt.
ON_FAILURE_CHOICES = ("teleop", "abort", "replan")


@dataclass(frozen=True)
class PlanningConfig:
    """Resolved settings. ``enabled`` False means nothing in this package ever runs."""

    enabled: bool = False
    proposal_model: str = DEFAULT_PROPOSAL_MODEL
    vlm_model: str = DEFAULT_VLM_MODEL
    # Reprompts allowed when a proposal comes back unparseable or fails validation. The error message
    # is fed back to the model, which is what makes a second attempt worth making at all.
    max_attempts: int = 3
    # Classify the plan's invented predicates on the FIRST image, before anything runs. Off by
    # default: a human is being asked precisely because the predicate is false, so it costs one model
    # call per grounding to learn what was already assumed. Worth turning on for a scene that may
    # start already solved.
    classify_initial: bool = False
    # Extra chances the operator gets at a human phase the check says did not happen.
    # 1 = show what is missing and hand the arm back once more; 0 = fail on the first bad verdict.
    verify_retries: int = 1
    # Treat a failed verification as a rollout failure (the operator still labels the episode, so a
    # false negative is recoverable by answering the label prompt). False records the verdict and
    # carries on, which is what you want while calibrating the classifier prompts.
    verify_enforced: bool = True
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
    # What happens when the planner cannot plan a robot phase. `teleop` describes the sub-goal to the
    # operator and lets them do it by hand, verified exactly as any other human phase; `abort` ends
    # the attempt; `replan` feeds the failure back to the proposer.
    on_robot_phase_failure: str = "teleop"
    # Which camera the verification frame comes from. A third-person view by default: after a
    # hand-off the arm is wherever the operator left it, so a wrist view points nowhere useful.
    verification_camera: str = "external"

    def __post_init__(self) -> None:
        if self.on_robot_phase_failure not in ON_FAILURE_CHOICES:
            raise ValueError(
                f"on_robot_phase_failure must be one of {', '.join(ON_FAILURE_CHOICES)}, "
                f"got {self.on_robot_phase_failure!r}"
            )
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {self.max_attempts}")
        if self.verify_retries < 0:
            raise ValueError(f"verify_retries must be >= 0, got {self.verify_retries}")
