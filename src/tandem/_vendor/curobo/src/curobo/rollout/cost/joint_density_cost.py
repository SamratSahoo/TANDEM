#
# Joint-position density-matching cost (tamp-vla / tiptop-viz manifold study).
#
# Pulls a trajopt segment's PER-JOINT position density toward the DROID human-teleop dataset
# (lerobot/droid_1.0.1). The distance is the 1-D Wasserstein-1 between the segment's own per-joint
# marginal and DROID's -- the exact quantity analysis2/stats.py::_w1 reports as
# wasserstein1_vs_droid_rad in joint_coverage.csv, made differentiable.
#
# For each of the 7 arm joints, the segment's H waypoint positions are SORTED (q_(1)<=...<=q_(H))
# and compared against DROID's baked inverse-CDF (quantile function) Q_j at the mid-rank levels
# u_k = (k-0.5)/H:  W1_j = mean_k huber(q_(k) - Q_j(u_k)). This is the discrete order-statistics
# form of 1-D W1. MINIMIZING it makes the segment's marginal look like DROID's -- and because the
# zero-cost target is q_(1)~DROID low tail ... q_(H)~DROID high tail, the optimum reproduces DROID's
# SPREAD rather than collapsing to its mode (the failure mode a per-waypoint NLL/energy prior has).
#
# The reference is one self-contained artifact: joint_density_ref.npz next to this file (baked by
# analysis2/bake_joint_density_ref.py from droid_aggregate.npz["h1"] over 27.6 M DROID frames).
# Override with the JOINT_DENSITY_REF env var. weight == 0 disables the cost (CostBase).
#
# NOTE (population level): this matches PER SEGMENT -- each seed's own horizon is the sample. DROID's
# h1 is the whole-corpus marginal over 94,774 episodes, so a single narrow pick/place arc is
# over-constrained against it; use a MODEST weight (regularizer regime) and gate on the W1 table +
# sim feasibility. See tamp-vla scripts/sweep_spec.py for the OFAT band.
#
# Third Party
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# CuRobo
from curobo.types.base import TensorDeviceType

# Local Folder
from .cost_base import CostBase, CostConfig

# Default reference: joint_density_ref.npz next to this file (self-contained, ~110 KB). Resolved
# relative to __file__ so it works on any clone; override with JOINT_DENSITY_REF.
DEFAULT_JOINT_DENSITY_REF = os.environ.get(
    "JOINT_DENSITY_REF", str(Path(__file__).resolve().parent / "joint_density_ref.npz")
)


# --------------------------------------------------------------------------- #
# reference loading (cached per (path, device, dtype))                         #
# --------------------------------------------------------------------------- #
_REF_CACHE = {}


def load_joint_density(ref_path: str, tensor_args: TensorDeviceType):
    """Load (and cache) DROID's per-joint quantile table + per-joint weights. Frozen, no grad."""
    key = (ref_path, str(tensor_args.device), str(tensor_args.dtype))
    if key in _REF_CACHE:
        return _REF_CACHE[key]

    blob = np.load(ref_path, allow_pickle=True)
    if "q_table" not in blob:
        raise KeyError(
            f"{ref_path} has no q_table; rebuild it with analysis2/bake_joint_density_ref.py."
        )

    def _t(x):
        return torch.as_tensor(np.asarray(x), device=tensor_args.device, dtype=tensor_args.dtype)

    pack = {
        "n_joints": int(blob["n_joints"]),
        "q_table": _t(blob["q_table"]),         # (J, M) inverse-CDF (quantile function)
        "joint_w": _t(blob["joint_w"]),         # (J,) per-joint cost weights
        "m_levels": int(np.asarray(blob["q_table"]).shape[1]),
    }
    _REF_CACHE[key] = pack
    return pack


# --------------------------------------------------------------------------- #
# DROID quantile nodes at the H sorted-rank levels (cached per (H, device, dtype))            #
# --------------------------------------------------------------------------- #
def quantile_ref(pack: dict, horizon: int, n_joints: int, device, dtype) -> torch.Tensor:
    """DROID quantile Q_j(u_k) at u_k=(k-0.5)/H for k=1..H -> [H, n_joints], cached per horizon.

    q_table is Q_j sampled on the uniform grid u_grid[i]=(i+0.5)/M, so u_k maps to the fractional
    index pos = u_k*M - 0.5 and we linearly interpolate (flat extrapolation past the end levels)."""
    cache = pack.setdefault("_qref_cache", {})
    ck = (horizon, str(device), str(dtype))
    if ck in cache:
        return cache[ck]
    M = pack["m_levels"]
    qt = pack["q_table"][:n_joints]                                     # [J, M]
    u = (torch.arange(horizon, device=device, dtype=dtype) + 0.5) / horizon  # [H] mid-ranks
    pos = (u * M - 0.5).clamp(0, M - 1)                                 # [H] index into u_grid
    lo = pos.floor().to(torch.long)
    hi = (lo + 1).clamp(max=M - 1)
    frac = (pos - lo.to(dtype)).clamp(0, 1)                             # [H]
    q_lo = qt.index_select(1, lo)                                       # [J, H]
    q_hi = qt.index_select(1, hi)                                       # [J, H]
    qref = (q_lo * (1 - frac) + q_hi * frac).transpose(0, 1).contiguous()  # [H, J]
    cache[ck] = qref
    return qref


def segment_w1(position: torch.Tensor, pack: dict, n_joints: int, huber_delta: float) -> torch.Tensor:
    """Joint-summed 1-D Wasserstein-1 of each segment's marginal to DROID -> [B]. Differentiable."""
    B, H, _ = position.shape
    q = position[..., :n_joints]                                       # [B, H, J]
    q_sorted, _ = torch.sort(q, dim=1)                                 # [B, H, J] ascending per joint
    qref = quantile_ref(pack, H, n_joints, position.device, position.dtype)  # [H, J]
    d = q_sorted - qref.unsqueeze(0)                                   # [B, H, J]
    # Huber/pseudo-abs: ~|d| for |d|>>delta, ~d^2/(2 delta) near 0; gradient bounded in (-1, 1) and
    # smooth at d=0, so L-BFGS' multi-scale line search stays well-conditioned. Shifted to be >= 0.
    huber = torch.sqrt(d * d + huber_delta * huber_delta) - huber_delta
    w1 = huber.mean(dim=1)                                             # [B, J] mean over sorted ranks
    return (w1 * pack["joint_w"][:n_joints]).sum(dim=1)               # [B]


# --------------------------------------------------------------------------- #
# cost config + module                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class JointDensityCostConfig(CostConfig):
    ref_path: str = DEFAULT_JOINT_DENSITY_REF
    n_joints: int = 7
    huber_delta: float = 0.05       # rad; smooths the W1 L1 kink so L-BFGS stays well-conditioned

    def __post_init__(self):
        return super().__post_init__()


class JointDensityCost(CostBase, JointDensityCostConfig):
    """Per-segment joint-position density-matching cost (1-D Wasserstein-1 to DROID's marginals).

    forward(position) -> [batch, horizon]. Each segment's joint-summed W1 distance to DROID is spread
    uniformly over the horizon (so the horizon sum equals weight * W1, and gradients flow through the
    sort to every waypoint). weight == 0 disables the cost (CostBase)."""

    def __init__(self, config: Optional[JointDensityCostConfig] = None):
        if config is not None:
            JointDensityCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
        self._init_post_config()
        self._pack = None  # lazy-loaded on first enabled forward

    def _get_pack(self):
        if self._pack is None:
            self._pack = load_joint_density(self.ref_path, self.tensor_args)
        return self._pack

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        # position: [batch, horizon, dof]
        horizon = position.shape[1]
        pack = self._get_pack()
        # curobo's clique tensor-step C++ backward asserts grad w.r.t. position is contiguous, but the
        # sort -> transpose chain hands back a NON-contiguous grad; coerce it via a hook.
        if position.requires_grad:
            position.register_hook(lambda g: g.contiguous() if g is not None else g)
        seg = segment_w1(position, pack, self.n_joints, self.huber_delta)  # [batch]
        # repeat (not expand) -> contiguous cost tensor for the downstream cat_sum / reductions
        return (self.weight * (seg / horizon)).unsqueeze(1).repeat(1, horizon)


# --------------------------------------------------------------------------- #
# per-segment trace (for the tiptop-viz cost-over-time plot)                   #
# --------------------------------------------------------------------------- #
def trajectory_density_trace(
    position: torch.Tensor,
    *,
    ref_path: str = DEFAULT_JOINT_DENSITY_REF,
    n_joints: int = 7,
    huber_delta: float = 0.05,
    tensor_args: Optional[TensorDeviceType] = None,
):
    """Per-timestep joint-density W1 for one trajectory segment (length == T).

    W1 is a whole-segment scalar (the marginal is a property of the whole segment), so the trace is
    that scalar broadcast over the segment's T timesteps -- exactly the cost the optimizer sees.
    position: [T, dof]. Returns a float64 numpy array [T] in radians (joint-summed W1)."""
    if tensor_args is None:
        tensor_args = TensorDeviceType()
    pack = load_joint_density(ref_path, tensor_args)
    if isinstance(position, torch.Tensor):
        pos = position.detach().to(device=tensor_args.device, dtype=tensor_args.dtype)
    else:
        pos = torch.as_tensor(np.asarray(position), device=tensor_args.device, dtype=tensor_args.dtype)
    T = pos.shape[0]
    with torch.no_grad():
        s = float(segment_w1(pos.unsqueeze(0), pack, n_joints, huber_delta).item())
    return np.full(T, s, dtype=np.float64)
