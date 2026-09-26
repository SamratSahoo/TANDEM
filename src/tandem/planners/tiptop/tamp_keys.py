"""The authoritative set of TAMP override keys, mirrored from tiptop.

These are the knobs that reach the planner's solvers as cost overrides. The
list is transcribed from the code that reads them, so a key here is a key that actually does
something:

  * ``tiptop/motion_planning.py``  — apply_cost_overrides, apply_model_overrides,
    resolve_time_dilation_factor, resolve_traj_length_norm, resolve_grasp_orientation_cost,
    resolve_grasp_center_cost, resolve_grasp_rank_conf_weight, resolve_transit_apex,
    resolve_posture_selection, resolve_ik_num_seeds, resolve_require_m2t2_grasps,
    resolve_max_motion_refine_attempts, resolve_placement_support,
    apply_perception_overrides, summarize_curobo_config
  * ``tiptop/planning.py``          — run_planning's grasp soft-cost weights
  * ``tiptop/trajectory_blending.py`` — resolve_blend_config
  * ``tiptop/tiptop_run.py`` — num_particles / opt_steps_per_skeleton

tiptop now lists them itself too (``tiptop/override_keys.py``, SUPPORTED_OVERRIDE_KEYS) and refuses any
other key when it loads the overrides. tandem accepts that set minus the keys in ``REFUSED``.

...as of the tiptop the TiPToP recipe pins (planners/tiptop/recipe.py). "Reads" is not enough on its
own: the key also has to be read on a path tandem's sidecar actually runs. tiptop's interactive
rollout loop and its websocket server read a few more, and tandem runs neither -- those are in
``REFUSED`` below, with the reason, rather than accepted and then ignored.
``tests/test_tiptop_bump.py`` re-derives the whole set from the pinned sources, so a bump that adds,
drops or re-homes a knob fails there instead of in a dataset.

We keep tandem's profile keys IDENTICAL to tiptop's rather than inventing a prettier nested
schema. A translation layer between two large key sets is exactly how the upstream project
lost overrides silently (for a while `tiptop-run` had no --curobo-overrides flag at all, so
every cfg/tamp/*.yml produced identical trajectories). Instead we validate strictly against
this set, which catches typos — the real problem — without the lossy mapping. It also means
an existing ``cfg/tamp/*.yml`` imports verbatim.
"""

from __future__ import annotations

# Scalar knobs: name -> expected python type for validation/coercion.
SCALAR_KEYS: dict[str, type] = {
    # --- cuTAMP solver effort (read by tiptop_run / tiptop_websocket_server) ---
    "num_particles": int,
    "opt_steps_per_skeleton": int,
    # --- TAMP-config knobs (not cuRobo cost weights) ---
    "time_dilation_factor": float,
    "time_dilation_factor_literal": float,
    "grasp_pose_change_weight": float,
    # Off-centre grasp soft cost: truthy turns cuTAMP's grasp_center_cost on AND is its weight
    # (resolve_grasp_center_cost; run_planning). Metres, so it wants tens, not units.
    "grasp_center_weight": float,
    # Rank satisfying particles on soft cost minus this times grasp confidence, instead of on
    # confidence alone -- without it the two grasp costs above never change the executed grasp.
    # 0.0 is meaningful (soft cost alone); absent keeps cuTAMP's confidence-only ranking.
    "grasp_rank_conf_weight": float,
    # Lift-traverse-descend transits: an apex this high above the higher end-effector position
    # (resolve_transit_apex). 0 (the default) is off; the min distance defaults to 0.10 m.
    "transit_apex_height": float,
    "transit_apex_min_dist": float,
    # Teleop-posture IK branch selection (resolve_posture_selection). Off unless the seed count is
    # above 1, and the four posture_* keys below it are read ONLY then.
    "posture_selection_seeds": int,
    "posture_grasp_roll": bool,
    "posture_pos_tol": float,
    "posture_rot_tol": float,
    # IK seeds the solver optimizes (resolve_ik_num_seeds). Unset, it follows posture selection:
    # max(12, 3 * seeds) when that is on, the robot's own default when it is off.
    "ik_num_seeds": int,
    # Fail the plan instead of substituting collision-sphere grasps for an object M2T2 proposed
    # nothing for (resolve_require_m2t2_grasps). The failure comes back as an ordinary plan failure.
    "require_m2t2_grasps": bool,
    # Satisfying particles cuTAMP tries motion refinement on before giving up; tiptop's default is
    # 32. tiptop reads `null` as "try every one", but a tandem profile treats a null as "not set",
    # so null is refused rather than quietly meaning 32 (see NULL_REFUSED). Use a large number.
    "max_motion_refine_attempts": int,
    # --- gradient_trajopt cost weights ---
    "uniform_velocity_weight": float,
    # The trajectory-encoder manifold cost's weight (encoder_path is its checkpoint, below).
    "encoder_weight": float,
    "rnd_novelty_weight": float,
    "rnd_novelty_log": bool,
    "joint_density_weight": float,
    "self_collision_weight": float,
    "cspace_weight": float,
    "primitive_collision_activation_distance": float,
    "run_weight_acceleration": float,
    "run_weight_jerk": float,
    "run_vec_weight": float,
    # --- trajopt model knobs ---
    "horizon": int,
    "base_dt": float,
    # --- MotionGen scaling ---
    "velocity_scale": float,
    "acceleration_scale": float,
    "jerk_scale": float,
    # --- stroke re-timing (trajectory_blending.resolve_blend_config) ---
    # The trajectory encoder re-times each stroke of the plan against the DROID cluster it was
    # trained on; encoder is the only mode, and it needs encoder_path.
    "retime_trajectory": bool,
    "retime_mode": str,
    "retime_smoothing": float,
    "retime_vel_slack": float,
    "retime_acc_slack": float,
    "retime_boundary_speed": float,
    "retime_speed_scale": float,
    "retime_max_duration_mult": float,
    # Aim each stroke at a latent DRAWN from the DROID cluster instead of its mean, which restores the
    # between-stroke timing variance the mean target collapses.
    "retime_sample_target": bool,
    # What an operation whose stroke cannot be re-timed inside the vel/accel caps gets. Off (tiptop's
    # default): the re-timer gives up on it and the run keeps its original segments at the plan's own
    # timing. On: the stroke is slowed past retime_max_duration_mult until it fits, and a run whose
    # re-timing failed is slowed into the same caps -- which can make a stroke many times slower than
    # the planner's. LJ1356's tiptop always did this; hitl-tamp-vla's toy-puzzle and bread/box configs
    # were tuned with it on. tiptop refuses a quoted "false" (it takes a boolean, or 0/1); tandem
    # takes a boolean, like every other switch here.
    "retime_stretch_to_caps": bool,
    # --- surface-fitted placement (resolve_placement_support) ---
    # Where an object may be put down. Off (the default), the region is the surface's oriented bounding
    # box, with the object's bottom at the box's TOP: right for a slab, wrong for anything with
    # structure -- for an open box it is the top of the folded-back lid, for a plate the rim. On,
    # cuTAMP fits the region to the surface's OBSERVED points: the level patches that would hold this
    # object's footprint, at that patch's own height. "Store Bread in Closed Box" and "Solve
    # Constrained Puzzle" set it; the bread was released ~19 cm up without it. The six keys after it
    # are read ONLY when it is on (PLACEMENT_GATED).
    "placement_support": bool,
    # Surface the object must keep around its footprint, in metres. Default 0.01. >= 0 (cuTAMP's
    # support_margin).
    "placement_support_margin": float,
    # How much the surface under a footprint may vary and still count as one level patch, in metres.
    # Default 0.008; > 0 (support_flatness_tol). Absorbs stereo noise as well as real relief: the
    # bread/box config raises it to 0.012 for a tray whose floor came back with a 1.7 cm spread.
    "placement_flatness_tol": float,
    # When NO patch of a goal surface would hold the object: true (default) fails the plan with that as
    # the reason -- an ordinary plan failure, so `hitl.on_robot_phase_failure` decides what happens --
    # false falls back to the bounding box and logs (placement_support_required).
    "placement_support_required": bool,
    # Whether a placed object may overlap the surface it was placed on in the collision cost. Default
    # true; placing INSIDE a container needs it, since perception reconstructs one as a filled hull
    # (placement_ignores_target_surface).
    "placement_into_surface": bool,
    # Whether the unobserved cells inside a surface's outline count as floor: a camera looking across a
    # box does not see its floor. Default false -- the one setting that places onto surface nobody
    # saw (support_fill_occluded).
    "placement_fill_occluded": bool,
    # The fraction of every footprint that must be genuinely observed. Default 0.25; in [0, 1]
    # (support_min_seen_frac). It is what guards placement_fill_occluded.
    "placement_min_seen_frac": float,
    # --- perception (PERCEPTION_KEYS below says where each one lands) ---
    "contact_threshold_m": float,
    "grasp_threshold": float,
    "m2t2_num_runs": int,
    "voxel_downsample_size": float,
    # How RANSAC picks the table among its candidate planes. Off (tiptop's default): an object votes for
    # a plane when its contact point is within 3 cm of it either side, which also counts objects BELOW
    # one. On: only objects resting ON a plane vote, ties broken by the plane's size (LJ1356's tiptop
    # always did this).
    "table_plane_support_vote": bool,
    # Whether object meshes and point clouds come from DISJOINT masks, every pixel two masks claim
    # going to the smaller object, so a container's hull stops at what rests on it. Off (tiptop's
    # default) keeps SAM-2's masks; the placement support points use disjoint masks either way.
    # LJ1356's tiptop always did this.
    "disjoint_object_masks": bool,
}

# Path-valued knobs. tandem resolves these to absolute paths before handing them over, so a
# relative path means "relative to the profile", not "relative to whatever cwd tiptop ran in".
# posture_ref is cuTAMP's baked posture prior (an .npz); unset, cuTAMP uses its own posture_ref.npz
# or $CUTAMP_POSTURE_REF.
PATH_KEYS: frozenset[str] = frozenset({"encoder_path", "posture_ref"})

# List-valued knobs.
LIST_KEYS: frozenset[str] = frozenset({"retime_ops"})

# Per-index dict knobs: {index -> value} applied into a cuRobo vector cost field.
INDEX_MAP_KEYS: frozenset[str] = frozenset(
    {"smooth_weight", "bound_weight", "bound_activation_distance", "pose_weight"}
)

# traj_length_norm is special: it must serialise as the STRING "inf" for the infinity norm,
# because the overrides dict round-trips through JSON, which has no Infinity literal.
SPECIAL_KEYS: frozenset[str] = frozenset({"traj_length_norm"})

ALL_KEYS: frozenset[str] = frozenset(
    set(SCALAR_KEYS) | PATH_KEYS | LIST_KEYS | INDEX_MAP_KEYS | SPECIAL_KEYS
)

# Knobs that retune PERCEPTION rather than the solver, and where in tiptop.yml each one lives. This
# is tiptop's own _PERCEPTION_OVERRIDE_KEYS table: the grasp candidates perception hands cuTAMP bound
# what any downstream cost can choose between, so a data-gen config sets them next to the solver
# knobs. tandem writes them into the rendered tiptop.yml (render.py) as well as passing them on
# with the rest, so the tiptop.yml copied into every leg states the values that were in force --
# upstream's copy shows the file on disk, not the override. Where the rig's perception block (rig.yml,
# planners.tiptop.perception) has the same setting, the profile's `tamp:` value wins, as it does in tiptop.
PERCEPTION_KEYS: dict[str, tuple[str, ...]] = {
    "contact_threshold_m": ("perception", "contact_threshold_m"),
    "grasp_threshold": ("perception", "m2t2", "grasp_threshold"),
    "m2t2_num_runs": ("perception", "m2t2", "num_runs"),
    "voxel_downsample_size": ("perception", "voxel_downsample_size"),
    "table_plane_support_vote": ("perception", "table_plane_support_vote"),
    "disjoint_object_masks": ("perception", "disjoint_object_masks"),
}

# The placement keys resolve_placement_support reads only when placement_support is on. Set without it
# they are accepted and passed on, and change nothing -- which render.check_assets says out loud.
PLACEMENT_GATED: tuple[str, ...] = (
    "placement_support_margin",
    "placement_flatness_tol",
    "placement_support_required",
    "placement_into_surface",
    "placement_fill_occluded",
    "placement_min_seen_frac",
)

# Keys a cfg/tamp file can carry that tandem REFUSES, each with the reason, because accepting one
# would be a setting that does nothing. The profile loader says why instead of "unknown setting".
REFUSED: dict[str, str] = {
    # Read by tiptop, but only on paths tandem's sidecar never runs.
    "auto_mode": (
        "is read only by tiptop's own interactive rollout loop (tiptop_run.async_entrypoint, via "
        "auto_mode.resolve_auto_mode), which tandem does not run: tandem's session decides when a "
        "trial starts and ends"
    ),
    "reset_placement_region": (
        "is read only by the scene reset in tiptop's own interactive rollout loop "
        "(scene_reset.reset_placement_region), which tandem does not run"
    ),
    "clear_goal_surfaces": (
        "is read only by tiptop's own rollout loop and its websocket server (tiptop_run.async_entrypoint, "
        "tiptop_websocket_server), which tandem does not run. tandem plans exactly the goal each phase "
        "states, and a leg may move only the objects that phase allows"
    ),
}

# Keys whose `null` means something to tiptop that a tandem profile cannot say: a profile drops a
# null as "not set", which here would silently mean tiptop's default instead.
NULL_REFUSED: dict[str, str] = {
    "max_motion_refine_attempts": (
        "max_motion_refine_attempts: null means 'try every satisfying particle' to tiptop, but a tandem "
        "profile reads null as 'not set', which is tiptop's default of 32. Set a large number instead"
    ),
}

# Enumerated string knobs -> their legal values (tiptop's override_keys.check_override_keys).
ENUMS: dict[str, frozenset[str]] = {
    "retime_mode": frozenset({"encoder"}),
}

# Keys tiptop renamed when stroke re-timing became the trajectory encoder's alone (tiptop 1d3dedf), old
# name -> new. Each is the same setting under a new name, so a profile written before the rename still
# loads: validate_tamp reads the old name as the new one. `tandem profile edit` shows the new names.
RENAMED: dict[str, str] = {
    "vae_path": "encoder_path",
    "vae_manifold_weight": "encoder_weight",
    "blend_trajectory": "retime_trajectory",
    "blend_smoothing": "retime_smoothing",
    "blend_vel_slack": "retime_vel_slack",
    "blend_acc_slack": "retime_acc_slack",
    "blend_boundary_speed": "retime_boundary_speed",
    "blend_speed_scale": "retime_speed_scale",
    "blend_ops": "retime_ops",
    "blend_max_duration_mult": "retime_max_duration_mult",
    "blend_vae_sample_target": "retime_sample_target",
    "blend_stretch_to_caps": "retime_stretch_to_caps",
}

# Settings of modes tiptop removed in the same change: the VAE-owned trajectory clock (vae_retiming and
# its three guard knobs) and the spline and flow re-timing modes. key -> (the value that describes what
# tiptop still does, so it is dropped; or None when no value does). Any other value is refused: that
# behavior is gone, and reading the setting as something else would change what the profile does.
REMOVED: dict[str, object] = {
    "vae_retiming": False,
    "retime_scale": None,
    "retime_smooth_weight": None,
    "retime_limit_weight": None,
    "blend_mode": "vae",
    "blend_boundary_window": None,
    "blend_boundary_window_sec": None,
    "blend_boundary_mode": None,
    "blend_pace": None,
    "blend_pace_scale": None,
    "blend_profile_end_sec": None,
    "blend_flow_steps": None,
    "blend_flow_retime_only": None,
    "blend_seed": None,
    "blend_model_path": None,
    "blend_stats_path": None,
}
REMOVED_BECAUSE = (
    "tiptop removed it: stroke re-timing is now the trajectory encoder's alone (retime_trajectory, "
    "encoder_path, the retime_* keys), and the VAE-owned clock and the spline and flow modes are gone"
)

# Range limits, each the check the planner itself applies -- only later, at warm-up or at the first
# plan (resolve_blend_config, apply_perception_overrides, cuTAMP's validate_tamp_config), and with
# the arm already moving to its capture pose. Three more are range-checked in options.py because
# their range is not a sign: traj_length_norm (>= 1, or inf) and the two time_dilation_factor keys
# ((0, 1]).
POSITIVE_KEYS: tuple[str, ...] = (
    "num_particles",
    "opt_steps_per_skeleton",
    "retime_speed_scale",
    "contact_threshold_m",
    "grasp_threshold",
    "m2t2_num_runs",
    "voxel_downsample_size",
    "ik_num_seeds",
    "max_motion_refine_attempts",
    "posture_pos_tol",
    "posture_rot_tol",
    "placement_flatness_tol",
)
NON_NEGATIVE_KEYS: tuple[str, ...] = (
    "transit_apex_height",
    "transit_apex_min_dist",
    "posture_selection_seeds",
    "grasp_rank_conf_weight",
    "placement_support_margin",
)
# A fraction, in [0, 1] both ends included, as cuTAMP's validate_tamp_config has it.
UNIT_INTERVAL_KEYS: tuple[str, ...] = ("placement_min_seen_frac",)

# Operations retime_ops may name (cuTAMP plan operation names).
RETIME_OPS = ("Pick", "Place", "MoveFree", "MoveHolding", "GoToInitial")

INFINITY_ALIASES = frozenset({"inf", "infinity", "max"})


def suggest(unknown: str, limit: int = 3) -> list[str]:
    """Closest known keys to a typo, for the 'did you mean…' hint."""
    import difflib

    return difflib.get_close_matches(unknown, sorted(ALL_KEYS), n=limit, cutoff=0.6)
