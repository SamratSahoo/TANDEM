"""The authoritative set of TAMP override keys, mirrored from tiptop.

These are the knobs that reach the planner's solvers as cost overrides. The
list is transcribed from the code that reads them, so a key here is a key that actually does
something:

  * ``tiptop/motion_planning.py``  — apply_cost_overrides, apply_model_overrides,
    resolve_time_dilation_factor, resolve_traj_length_norm, resolve_grasp_orientation_cost,
    resolve_grasp_center_cost, resolve_grasp_rank_conf_weight, resolve_transit_apex,
    resolve_posture_selection, resolve_ik_num_seeds, resolve_require_m2t2_grasps,
    resolve_max_motion_refine_attempts, resolve_vae_retiming, apply_perception_overrides,
    summarize_curobo_config
  * ``tiptop/planning.py``          — run_planning's grasp soft-cost weights
  * ``tiptop/trajectory_blending.py`` — resolve_blend_config
  * ``tiptop/tiptop_run.py`` — num_particles / opt_steps_per_skeleton

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
    "vae_manifold_weight": float,
    # The VAE manifold cost owns the trajectory clock: each waypoint interval's duration becomes a
    # trajopt variable, and time_dilation_factor, cuRobo's optimize_dt and blending are all switched
    # off (resolve_vae_retiming). Ignored, with a warning from tiptop, when vae_manifold_weight is 0.
    "vae_retiming": bool,
    # The three guard knobs of that retiming, read only when it is on (apply_cost_overrides).
    "retime_scale": float,
    "retime_smooth_weight": float,
    "retime_limit_weight": float,
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
    # --- trajectory blending (resolve_blend_config) ---
    "blend_trajectory": bool,
    "blend_mode": str,
    "blend_smoothing": float,
    "blend_vel_slack": float,
    "blend_acc_slack": float,
    "blend_boundary_speed": float,
    "blend_speed_scale": float,
    "blend_boundary_window": float,
    "blend_boundary_window_sec": float,
    "blend_boundary_mode": str,
    "blend_pace": str,
    "blend_pace_scale": float,
    "blend_profile_end_sec": float,
    "blend_max_duration_mult": float,
    "blend_flow_steps": int,
    "blend_flow_retime_only": bool,
    # blend_mode: vae only -- aim each stroke at a latent DRAWN from the DROID cluster instead of its
    # mean, which restores the between-stroke timing variance the mean target collapses.
    "blend_vae_sample_target": bool,
    "blend_seed": int,
    # --- perception (PERCEPTION_KEYS below says where each one lands) ---
    "contact_threshold_m": float,
    "grasp_threshold": float,
    "m2t2_num_runs": int,
    "voxel_downsample_size": float,
}

# Path-valued knobs. tandem resolves these to absolute paths before handing them over, so a
# relative path means "relative to the profile", not "relative to whatever cwd tiptop ran in".
# posture_ref is cuTAMP's baked posture prior (an .npz); unset, cuTAMP uses its own posture_ref.npz
# or $CUTAMP_POSTURE_REF.
PATH_KEYS: frozenset[str] = frozenset({"vae_path", "blend_model_path", "blend_stats_path", "posture_ref"})

# List-valued knobs.
LIST_KEYS: frozenset[str] = frozenset({"blend_ops"})

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
# upstream's copy shows the file on disk, not the override. Where the profile's `perception:` block
# has the same setting, the `tamp:` value wins, as it does in tiptop.
PERCEPTION_KEYS: dict[str, tuple[str, ...]] = {
    "contact_threshold_m": ("perception", "contact_threshold_m"),
    "grasp_threshold": ("perception", "m2t2", "grasp_threshold"),
    "m2t2_num_runs": ("perception", "m2t2", "num_runs"),
    "voxel_downsample_size": ("perception", "voxel_downsample_size"),
}

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
# Surface-fitted placement is a feature of LJ1356's fork of tiptop (resolve_placement_support) and its
# cuTAMP, not of the SamratSahoo trees tandem pins; the monorepo's paper-era box and puzzle configs
# carry these keys. Refused rather than dropped: a config that relied on them places differently
# without them (the bread released ~19 cm up), and that should be a decision, not a surprise.
for _key in (
    "placement_support",
    "placement_support_margin",
    "placement_support_required",
    "placement_into_surface",
    "placement_fill_occluded",
    "placement_min_seen_frac",
    "placement_flatness_tol",
):
    REFUSED[_key] = (
        "is read only by LJ1356's fork of tiptop (surface-fitted placement), not by the SamratSahoo "
        "tiptop tandem runs, so it would do nothing. Remove it, or port that placement support upstream"
    )
del _key

# Keys whose `null` means something to tiptop that a tandem profile cannot say: a profile drops a
# null as "not set", which here would silently mean tiptop's default instead.
NULL_REFUSED: dict[str, str] = {
    "max_motion_refine_attempts": (
        "max_motion_refine_attempts: null means 'try every satisfying particle' to tiptop, but a tandem "
        "profile reads null as 'not set', which is tiptop's default of 32. Set a large number instead"
    ),
}

# Enumerated string knobs -> their legal values (from resolve_blend_config).
ENUMS: dict[str, frozenset[str]] = {
    "blend_mode": frozenset({"spline", "flow", "vae"}),
    "blend_pace": frozenset({"plan", "droid"}),
    "blend_boundary_mode": frozenset({"const", "droid"}),
}

# Range limits, each the check the planner itself applies -- only later, at warm-up or at the first
# plan (resolve_blend_config, apply_perception_overrides, cuTAMP's validate_tamp_config), and with
# the arm already moving to its capture pose. Three more are range-checked in options.py because
# their range is not a sign: traj_length_norm (>= 1, or inf) and the two time_dilation_factor keys
# ((0, 1]).
POSITIVE_KEYS: tuple[str, ...] = (
    "num_particles",
    "opt_steps_per_skeleton",
    "blend_speed_scale",
    "blend_pace_scale",
    # tiptop raises for <= 0 here, not < 0 like its sibling blend_boundary_window.
    "blend_boundary_window_sec",
    "contact_threshold_m",
    "grasp_threshold",
    "m2t2_num_runs",
    "voxel_downsample_size",
    "ik_num_seeds",
    "max_motion_refine_attempts",
    "posture_pos_tol",
    "posture_rot_tol",
)
NON_NEGATIVE_KEYS: tuple[str, ...] = (
    "blend_boundary_window",
    "blend_profile_end_sec",
    "transit_apex_height",
    "transit_apex_min_dist",
    "posture_selection_seeds",
    "grasp_rank_conf_weight",
)

# Operations blend_ops may name (cuTAMP plan operation names).
BLEND_OPS = ("Pick", "Place", "MoveFree", "MoveHolding", "GoToInitial")

INFINITY_ALIASES = frozenset({"inf", "infinity", "max"})


def suggest(unknown: str, limit: int = 3) -> list[str]:
    """Closest known keys to a typo, for the 'did you mean…' hint."""
    import difflib

    return difflib.get_close_matches(unknown, sorted(ALL_KEYS), n=limit, cutoff=0.6)
