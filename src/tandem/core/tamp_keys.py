"""The authoritative set of TAMP override keys, mirrored from tiptop.

These are the knobs that reach the planner's solvers as cost overrides. The
list is transcribed from the code that reads them, so a key here is a key that actually does
something:

  * ``tiptop/motion_planning.py``  — apply_cost_overrides, apply_model_overrides,
    resolve_time_dilation_factor, resolve_traj_length_norm, resolve_grasp_orientation_cost,
    summarize_curobo_config
  * ``tiptop/trajectory_blending.py`` — resolve_blend_config
  * ``tiptop/tiptop_run.py`` — num_particles / opt_steps_per_skeleton

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
    # --- gradient_trajopt cost weights ---
    "uniform_velocity_weight": float,
    "vae_manifold_weight": float,
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
    "blend_seed": int,
}

# Path-valued knobs. tandem resolves these to absolute paths before handing them over, so a
# relative path means "relative to the profile", not "relative to whatever cwd tiptop ran in".
PATH_KEYS: frozenset[str] = frozenset({"vae_path", "blend_model_path", "blend_stats_path"})

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

# Enumerated string knobs -> their legal values (from resolve_blend_config).
ENUMS: dict[str, frozenset[str]] = {
    "blend_mode": frozenset({"spline", "flow"}),
    "blend_pace": frozenset({"plan", "droid"}),
    "blend_boundary_mode": frozenset({"const", "droid"}),
}

# Operations blend_ops may name (cuTAMP plan operation names).
BLEND_OPS = ("Pick", "Place", "MoveFree", "MoveHolding", "GoToInitial")

INFINITY_ALIASES = frozenset({"inf", "infinity", "max"})


def suggest(unknown: str, limit: int = 3) -> list[str]:
    """Closest known keys to a typo, for the 'did you mean…' hint."""
    import difflib

    return difflib.get_close_matches(unknown, sorted(ALL_KEYS), n=limit, cutoff=0.6)
