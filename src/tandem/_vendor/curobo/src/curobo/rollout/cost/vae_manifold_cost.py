#
# VAE motion-manifold cost (tamp-vla / tiptop-viz manifold study).
#
# Scores how DROID-like a trajopt segment's joint-space motion is, using the Filterbank VAE
# trained in tamp-vla/vae/ (see vae/train.py, class FilterbankVAE). The segment positions
# [batch, horizon, dof] are resampled from the trajopt rate (base_dt) to the VAE's 15 Hz, the
# joint metric [q|v|a|j] (28-D for 7 joints) is rebuilt by central differencing, standardized
# with the VAE's per-channel stats, and encoded by the filterbank into one latent per segment
# (variable length, masked global pooling -- no windowing).
#
# The score is the squared Mahalanobis distance of that latent to the DROID human-teleop
# latent cluster. MINIMIZING it pulls the segment's motion style toward DROID. (The filterbank
# latent SEPARATES DROID from cuTAMP, so cuTAMP segments start far from the DROID mean.)
#
# RETIMING (`retiming: True`, driven by the `vae_retiming` tamp_override). By default the segment
# is assumed to run at a constant `source_dt`, so the only thing the optimizer can move is the
# waypoints. With retiming on, the cost additionally takes `theta` -- one free scalar per WAYPOINT
# INTERVAL, carried as extra rows of the trajopt action tensor -- and turns it into a per-interval
# duration vector. The resample to 15 Hz then reads the segment off that non-uniform clock with a
# differentiable gather+lerp instead of F.interpolate, so ONE backward pass produces both
# d(maha^2)/d(waypoint) and d(maha^2)/d(dt_i) and LBFGS moves geometry and timing together.
#
# This matters because the cost is far more sensitive to timing than to geometry: on real cuTAMP
# segments, retiming the SAME path at the SAME total duration moves maha^2 by ~184x, while
# perturbing the waypoints by the same displacement moves it ~4x. Mechanistically, cuTAMP's
# standardized accel/jerk sit ~2.4x/5.7x BELOW DROID's, and redistributing time is what raises
# them; amplitude (the only lever a fixed-dt cost has) raises them by inflating joint spans.
#
# The total duration is held at (horizon - 1) * source_dt and only its DISTRIBUTION is free. That
# is deliberate: the resample count stays fixed (so the integer sample-count staircase never
# enters the gradient), the encoder stays inside the segment-length band its DROID reference was
# baked on, and the cost's preferred global time scale -- which implies ~5x DROID's median joint
# speed -- never gets to set the pace.
#
# Everything -- encoder weights, channel stats, and DROID latent mean/precision -- is loaded
# from one self-contained checkpoint: default = tamp-vla/vae/checkpoints/vae_full.pt, resolved
# relative to this file (override with the VAE_MANIFOLD_CKPT env var). The DROID stats are
# baked in by vae/train.py::droid_latent_stats.
#
# Third Party
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# CuRobo
from curobo.types.base import TensorDeviceType

# Local Folder
from .cost_base import CostBase, CostConfig

# Default checkpoint: tamp-vla/vae/checkpoints/vae_full.pt, resolved RELATIVE to this file so it
# works on any clone of the tamp-vla monorepo (curobo is a submodule at tamp-vla/curobo).
# .../tamp-vla/curobo/src/curobo/rollout/cost/vae_manifold_cost.py -> parents[5] == tamp-vla.
_REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_VAE_MANIFOLD_CKPT = os.environ.get(
    "VAE_MANIFOLD_CKPT", str(_REPO_ROOT / "vae" / "checkpoints" / "vae_full.pt")
)

# The rate the VAE was trained at (vae/data.py COMMON_RATE); segments are resampled to it.
VAE_RATE_HZ = 15.0


# --------------------------------------------------------------------------- #
# Filterbank VAE replica (mirrors tamp-vla/vae/train.py:FilterbankVAE so the    #
# saved state_dict loads strictly).                                            #
# --------------------------------------------------------------------------- #
def _masked_mean_std(x, m):                                # x:(B,C,T) m:(B,1,T) -> (mean, std)
    s = m.sum(-1).clamp(min=1.0)
    mean = (x * m).sum(-1) / s
    std = (((x - mean.unsqueeze(-1)) ** 2 * m).sum(-1) / s).clamp(min=1e-8).sqrt()
    return mean, std


def _masked_stats(x, m):                                   # masked mean | std | max over time
    mean, std = _masked_mean_std(x, m)
    return torch.cat([mean, std, (x + (1 - m) * -1e9).amax(-1)], 1)


class _FilterbankVAE(nn.Module):
    def __init__(self, ch, d, n_target, emb=96, p=0.2):
        super().__init__()
        specs = [(3, 1), (7, 1), (15, 1), (7, 2), (15, 4), (15, 8)]
        self.fb = nn.ModuleList([nn.Conv1d(ch, 32, k, dilation=dl, padding=dl * (k - 1) // 2)
                                 for k, dl in specs])
        self.fbn = nn.ModuleList([nn.BatchNorm1d(32) for _ in specs])
        self.t = nn.Sequential(
            nn.Conv1d(ch, 64, 5, 2, 2), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 96, 5, 2, 2), nn.BatchNorm1d(96), nn.GELU(),
            nn.Conv1d(96, 96, 3, 2, 1), nn.BatchNorm1d(96), nn.GELU())
        self.fc = nn.Sequential(nn.Linear(len(specs) * 32 * 2 + 96 * 3, 256), nn.LayerNorm(256),
                                nn.GELU(), nn.Dropout(p),
                                nn.Linear(256, emb), nn.LayerNorm(emb), nn.GELU())
        self.to_lat = nn.Linear(emb, 2 * d)
        self.aux = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, n_target))
        self.d, self.ch = d, ch

    def _embed(self, x, m):
        fbp = []
        for b, bn in zip(self.fb, self.fbn):
            mean, std = _masked_mean_std(bn(b(x)).abs(), m)
            fbp += [mean, std]
        tt = self.t(x)
        mt = m[:, :, ::8][:, :, :tt.shape[-1]]
        if mt.shape[-1] < tt.shape[-1]:
            mt = F.pad(mt, (0, tt.shape[-1] - mt.shape[-1]))
        return self.fc(torch.cat(fbp + [_masked_stats(tt, mt)], 1))

    def encode_mu(self, x, m):
        return self.to_lat(self._embed(x, m)).chunk(2, dim=1)[0]   # latent mean only

    def encode_mu_masked(self, x, m):
        """encode_mu on a zero-padded window, made EXACTLY equal to encoding the prefix alone.

        ``_embed`` is already prefix-exact for the filterbank branch: it is a single conv layer, so
        input zeros past the mask present the same view as the conv's own zero padding does at the
        end of an unpadded sequence. The strided branch is not, because a conv over zeros emits its
        BIAS (then BatchNorm, then GELU) -- a nonzero constant -- where the unpadded forward sees
        true zeros, and layers 2-3 then mix that leakage back into the valid region. Re-zeroing the
        activations outside the (subsampled) mask after every strided layer restores equality.

        This is what makes a duration a differentiable variable rather than a sample COUNT: a soft
        mask gives a continuous effective length, and a batch of DIFFERENT durations can share one
        padded forward pass without the padding changing anyone's score.

        x: [B, C, T] standardized features, zero beyond the mask. m: [B, 1, T], soft mask in [0, 1].
        """
        fbp = []
        for b, bn in zip(self.fb, self.fbn):
            mean, std = _masked_mean_std(bn(b(x)).abs(), m)
            fbp += [mean, std]

        h, mh = x, m
        for mod in self.t:
            h = mod(h)
            if isinstance(mod, nn.Conv1d):
                mh = mh[:, :, :: mod.stride[0]][:, :, : h.shape[-1]]
                if mh.shape[-1] < h.shape[-1]:
                    mh = F.pad(mh, (0, h.shape[-1] - mh.shape[-1]))
            h = h * mh

        mean, std = _masked_mean_std(h, mh)
        # Hard + detached inside the max: a soft mask here would push a 1e9 gradient through
        # whichever frame happens to be the argmax.
        hard = (mh > 0.5).to(h.dtype).detach()
        mx = (h + (hard - 1.0) * 1e9).amax(-1)
        return self.to_lat(self.fc(torch.cat(fbp + [torch.cat([mean, std, mx], 1)], 1))).chunk(2, dim=1)[0]


# --------------------------------------------------------------------------- #
# checkpoint loading (cached per (path, device, dtype))                        #
# --------------------------------------------------------------------------- #
_PACK_CACHE = {}


def load_vae_manifold(checkpoint_path: str, tensor_args: TensorDeviceType):
    """Load (and cache) the filterbank encoder + channel stats + DROID latent mean/precision."""
    key = (checkpoint_path, str(tensor_args.device), str(tensor_args.dtype))
    if key in _PACK_CACHE:
        return _PACK_CACHE[key]

    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "droid_latent_mean" not in blob:
        raise KeyError(
            f"{checkpoint_path} has no DROID latent stats; rebuild vae_full.pt with "
            "vae/train.py (it bakes them in via droid_latent_stats)."
        )
    ch, d, n_feat = int(blob["ch"]), int(blob["latent"]), int(blob["n_feat"])
    model = _FilterbankVAE(ch, d, n_feat)
    model.load_state_dict(blob["state_dict"])
    model = model.to(device=tensor_args.device, dtype=tensor_args.dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def _t(x, shape=None):
        t = torch.as_tensor(np.asarray(x), device=tensor_args.device, dtype=tensor_args.dtype)
        return t.view(*shape) if shape is not None else t

    pack = {
        "model": model,
        "n_joints": int(blob.get("n_joints", 7)),
        "chan_mu": _t(blob["chan_mu"], (1, 1, ch)),
        "chan_sd": _t(blob["chan_sd"], (1, 1, ch)),
        "droid_mean": _t(blob["droid_latent_mean"], (1, d)),
        "droid_prec": _t(blob["droid_latent_precision"], (d, d)),
    }
    _PACK_CACHE[key] = pack
    return pack


# --------------------------------------------------------------------------- #
# motion -> standardized [q|v|a|j] segment (matches vae/data.py preprocessing)  #
# --------------------------------------------------------------------------- #
def _grad_time(x: torch.Tensor, h) -> torch.Tensor:
    """d x / d t along dim=1, central in the interior + one-sided edges -- identical to
    numpy.gradient(edge_order=1), which vae/data.py uses. x: [B, T, C].

    ``h`` may be a float (the fixed 15 Hz step) or a per-batch tensor broadcastable to [B, 1, 1].
    The tensor form is what carries d(cost)/d(duration): the sample GRID is normalized (sample k
    always sits at fraction k/(N-1) of the segment), so if h were constant the derivative channels
    would not depend on the segment's duration at all and the gradient w.r.t. it would be exactly
    zero almost everywhere -- duration would only enter through the integer sample count."""
    interior = (x[:, 2:] - x[:, :-2]) / (2.0 * h)
    first = (x[:, 1:2] - x[:, 0:1]) / h
    last = (x[:, -1:] - x[:, -2:-1]) / h
    return torch.cat([first, interior, last], dim=1)


def interval_durations(theta: torch.Tensor, nominal_dt: float, scale: float) -> torch.Tensor:
    """[B, H-1] free knots -> [B, H-1] strictly positive per-interval durations.

        d_i = nominal_dt * exp(scale * tanh(theta_i))

    Each interval is independent, so the knots control BOTH the local distribution of time and the
    segment's total duration D = sum(d_i) -- speeding the whole trajectory up is just every knot
    moving the same way. That matters: on the tiptop TDF sweep, duration alone moves maha^2 by 16x
    (6.18 s -> 435, 2.06 s -> 27), so a parameterization that pins the total (an earlier softmax
    version of this function did) throws away the largest single lever the cost has.

    ``theta == 0`` reproduces the uniform nominal clock exactly. ``tanh`` is a structural rail
    rather than a penalty: for ANY theta the optimizer proposes, each interval stays within
    exp(+-scale) of nominal -- so D stays in [D_nom/exp(scale), D_nom*exp(scale)] and the
    longest/shortest interval ratio is bounded by exp(2*scale). The LBFGS box bounds cannot do this
    (they are per-JOINT-COLUMN and shared by every action row)."""
    return nominal_dt * torch.exp(scale * torch.tanh(theta))


def _resample_at(q: torch.Tensor, tau: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Linear resample of ``q`` [B, H, J] sampled at times ``tau`` [B, H] onto times ``t`` [B, n2].

    Differentiable in BOTH q (through the lerp weights) and tau (through the interval endpoints),
    which is what carries d(cost)/d(dt) back to the duration knots. The interval index comes from a
    comparison and so is piecewise constant -- correct almost everywhere, exactly as for any
    linear-interpolation gradient."""
    n_int = tau.shape[1] - 1
    idx = (t.unsqueeze(-1) >= tau[:, :-1].unsqueeze(1)).sum(-1) - 1      # [B, n2]
    idx = idx.clamp(0, n_int - 1)
    t0 = torch.gather(tau, 1, idx)
    t1 = torch.gather(tau, 1, idx + 1)
    u = ((t - t0) / (t1 - t0).clamp(min=1e-9)).clamp(0.0, 1.0).unsqueeze(-1)
    gi = idx.unsqueeze(-1).expand(-1, -1, q.shape[-1])
    q0 = torch.gather(q, 1, gi)
    q1 = torch.gather(q, 1, gi + 1)
    return q0 + (q1 - q0) * u


def warp_positions(
    position: torch.Tensor,
    theta: torch.Tensor,
    nominal_dt: float,
    scale: float,
    n_out: Optional[int] = None,
):
    """Read ``position`` [B, H, J] off the per-interval clock ``theta`` defines, sampled uniformly.

    Returns ``(warped [B, n_out, J], durations [B, H-1], total [B])``. This is the one place the
    retiming is defined: the cost calls it to score the segment, and TrajOptSolver calls it again
    with ``n_out == H`` to BAKE the optimized clock into the emitted waypoints. Baking is what lets
    a non-uniform clock survive: every stage downstream of trajopt -- scale_by_dt, the interpolation
    kernel, cuTAMP's plan step, the executor and the LeRobot export -- carries one scalar dt per
    trajectory, so the SHAPE of the clock has to live in the position samples. Its overall SCALE
    does survive as a scalar, and TrajOptSolver emits it as the trajectory's dt."""
    d = interval_durations(theta, nominal_dt, scale)
    tau = F.pad(torch.cumsum(d, dim=-1), (1, 0))                             # [B, H]
    total = tau[:, -1]                                                       # [B]
    n = position.shape[1] if n_out is None else n_out
    frac = torch.linspace(0.0, 1.0, n, device=position.device, dtype=position.dtype)
    t = frac.unsqueeze(0) * total.unsqueeze(-1)                              # [B, n]
    return _resample_at(position, tau, t), d, total


def positions_to_input(
    position: torch.Tensor,
    source_dt: float,
    pack: dict,
    *,
    theta: Optional[torch.Tensor] = None,
    retime_scale: float = 0.7,
):
    """[B, H, >=n_joints] joint positions -> ([B, 28, n2] standardized series, [B, 1, n2] mask).

    Resamples the segment from its native rate (1/source_dt) to VAE_RATE_HZ, rebuilds the
    joint metric [q|v|a|j] (28-D for 7 joints) by central differencing, and standardizes with
    the VAE's per-channel stats. The filterbank pools over the whole (variable-length) segment,
    so the mask is all-ones.

    ``theta`` [B, H-1] switches the resample from a uniform clock to the per-interval one
    ``interval_durations`` builds. The sample COUNT stays at the nominal n2 for the whole batch (it
    has to -- one integer is shared by every particle, and a count that moved with the duration
    would be a non-differentiable staircase), and the differencing step becomes h = D/(n2-1) per
    segment. That is what makes the derivative channels -- and therefore the cost -- depend on how
    long the segment takes, which is the whole point: with a constant h, sample k sits at fraction
    k/(n2-1) of the path regardless of D, so d(cost)/d(D) would be identically zero."""
    B, H, _ = position.shape
    J = pack["n_joints"]
    if position.shape[-1] < J:
        raise ValueError(
            f"VAE-manifold cost expects >= {J} joints (the VAE was trained on {J}-DOF Franka "
            f"joint metrics) but got dof={position.shape[-1]}"
        )
    q = position[..., :J]
    nominal_total = (H - 1) * float(source_dt)
    n2 = max(2, int(round(nominal_total * VAE_RATE_HZ)) + 1)
    if theta is None:
        if n2 != H:
            q = F.interpolate(
                q.transpose(1, 2), size=n2, mode="linear", align_corners=True
            ).transpose(1, 2)
        h = 1.0 / VAE_RATE_HZ
    else:
        if theta.shape[-1] != H - 1:
            raise ValueError(
                f"VAE-manifold retiming expects one duration knot per waypoint interval "
                f"({H - 1} for horizon {H}) but got theta with {theta.shape[-1]}"
            )
        q, _, total = warp_positions(q, theta, float(source_dt), retime_scale, n_out=n2)
        h = (total / (n2 - 1)).view(B, 1, 1)
    v = _grad_time(q, h)
    a = _grad_time(v, h)
    j = _grad_time(a, h)
    feats = torch.cat([q, v, a, j], dim=-1)                 # [B, n2, 28]
    feats = (feats - pack["chan_mu"]) / pack["chan_sd"]
    x = feats.transpose(1, 2).contiguous()                  # [B, 28, n2]
    mask = torch.ones(B, 1, n2, device=x.device, dtype=x.dtype)
    return x, mask


def segment_maha2(
    position: torch.Tensor,
    source_dt: float,
    pack: dict,
    *,
    theta: Optional[torch.Tensor] = None,
    retime_scale: float = 0.7,
) -> torch.Tensor:
    """Squared Mahalanobis distance of each segment's latent to the DROID cluster -> [B]."""
    x, mask = positions_to_input(
        position, source_dt, pack, theta=theta, retime_scale=retime_scale
    )
    mu = pack["model"].encode_mu(x, mask)
    dz = mu - pack["droid_mean"]
    return torch.einsum("ni,ij,nj->n", dz, pack["droid_prec"], dz)


def warp_limit_penalty(
    position: torch.Tensor,
    durations: torch.Tensor,
    max_vel: torch.Tensor,
    max_acc: torch.Tensor,
) -> torch.Tensor:
    """Hinge penalty [B] for a retimed segment that exceeds the arm's velocity/acceleration limits.

    Load-bearing, not cosmetic. cuRobo's own BoundCost differentiates the waypoints at the solver's
    FIXED traj_dt, so it is blind to the warp: compressing an interval raises the executed speed
    without changing anything BoundCost sees. And with ``vae_retiming`` the downstream time-optimal
    retiming is bypassed, so nothing after this point will slow a too-fast segment back down either.

    Evaluated on the piecewise-linear waypoint path, which is exact for it and independent of the
    15 Hz resample count: v_i = (q_{i+1} - q_i) / d_i, a_i from the neighbouring interval speeds."""
    dq = position[:, 1:] - position[:, :-1]                                  # [B, H-1, J]
    v = dq / durations.unsqueeze(-1).clamp(min=1e-6)
    dt_mid = (0.5 * (durations[:, 1:] + durations[:, :-1])).unsqueeze(-1).clamp(min=1e-6)
    a = (v[:, 1:] - v[:, :-1]) / dt_mid                                      # [B, H-2, J]
    over_v = torch.relu(v.abs() / max_vel.view(1, 1, -1) - 1.0)
    over_a = torch.relu(a.abs() / max_acc.view(1, 1, -1) - 1.0)
    return (over_v ** 2).sum(dim=(1, 2)) + (over_a ** 2).sum(dim=(1, 2))


def warp_smoothness_penalty(theta: torch.Tensor, scale: float) -> torch.Tensor:
    """Mean squared second difference of the log interval durations -> [B].

    One free knot per interval is exactly what was asked for, and it is also enough freedom to
    express per-sample chatter -- a sawtooth clock that inflates the jerk channel the encoder
    weights most without producing motion any human made. This is the term that makes the warp a
    band-limited redistribution of time rather than a dither; ``retime_smooth_weight`` sets how
    hard. Computed on the log durations, whose constant term (log nominal_dt) second-differences
    away -- so this penalizes only the SHAPE of the clock and is blind to the overall speed, which
    is exactly the split we want: chatter is penalized, going uniformly faster is not."""
    log_d = scale * torch.tanh(theta)
    d2 = log_d[:, 2:] - 2.0 * log_d[:, 1:-1] + log_d[:, :-2]
    return (d2 ** 2).mean(dim=-1)


# --------------------------------------------------------------------------- #
# cost config + module                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class VaeManifoldCostConfig(CostConfig):
    checkpoint_path: str = DEFAULT_VAE_MANIFOLD_CKPT
    n_joints: int = 7
    source_dt: float = 0.15         # trajopt base_dt (gradient_trajopt.yml model.dt_traj_params.base_dt)

    # --- per-interval retiming (the `vae_retiming` tamp_override) ------------------------------ #
    # When True the cost accepts one duration knot per waypoint interval and optimizes the timing
    # jointly with the waypoints. ArmReacher carries the knots as extra rows of the action tensor
    # and TrajOptSolver emits the resulting non-uniform clock instead of its own retiming.
    retiming: bool = False
    # Rail on the clock: every interval stays within exp(+-retime_scale) of the nominal source_dt,
    # so the segment's total duration stays in [D/exp(s), D*exp(s)] and the longest/shortest
    # interval ratio is bounded by exp(2s). 1.1 -> duration free over [0.33x, 3.0x] nominal, which
    # covers the tiptop TDF sweep's measured optimum (~2.1 s against a 4.65 s nominal, i.e. 0.44x)
    # with margin on both sides.
    retime_scale: float = 1.1
    # Weights of the two guard terms, RELATIVE to maha^2 (the whole bracket is multiplied by
    # `weight`, so sweeping vae_manifold_weight keeps their balance).
    retime_smooth_weight: float = 10.0
    retime_limit_weight: float = 500.0

    def __post_init__(self):
        return super().__post_init__()


class VaeManifoldCost(CostBase, VaeManifoldCostConfig):
    """Per-segment VAE motion-manifold cost (DROID Mahalanobis distance).

    forward(position, theta) -> [batch, horizon]. Each segment is encoded to one latent; its
    squared Mahalanobis distance to the DROID cluster is spread uniformly over the horizon (so the
    horizon sum equals weight * distance, and gradients flow through the encoder to every
    waypoint). weight == 0 disables the cost (CostBase).

    With ``retiming`` on, ``theta`` [batch, horizon - 1] additionally sets the duration of each
    waypoint interval, and the returned cost carries the warp guard terms. One backward pass then
    yields d(cost)/d(waypoint) AND d(cost)/d(dt_i), so LBFGS co-adapts geometry and timing."""

    def __init__(self, config: Optional[VaeManifoldCostConfig] = None):
        if config is not None:
            VaeManifoldCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
        self._init_post_config()
        self._pack = None  # lazy-loaded on first enabled forward
        self._max_vel = None
        self._max_acc = None

    def _get_pack(self):
        if self._pack is None:
            self._pack = load_vae_manifold(self.checkpoint_path, self.tensor_args)
            if self.n_joints != self._pack["n_joints"]:
                self._pack = dict(self._pack, n_joints=self.n_joints)
        return self._pack

    def set_joint_limits(self, max_velocity: torch.Tensor, max_acceleration: torch.Tensor):
        """Arm limits for the retiming guard (ArmReacher hands these over from state_bounds).

        Without them the warp is unconstrained, and since ``vae_retiming`` also bypasses cuRobo's
        time-optimal retiming there would be nothing left downstream to slow a segment back down."""
        J = self.n_joints
        self._max_vel = max_velocity.detach()[:J].to(self.tensor_args.device).clamp(min=1e-3)
        self._max_acc = max_acceleration.detach()[:J].to(self.tensor_args.device).clamp(min=1e-3)

    def forward(self, position: torch.Tensor, theta: Optional[torch.Tensor] = None) -> torch.Tensor:
        # position: [batch, horizon, dof]; theta: [batch, horizon - 1] or None
        horizon = position.shape[1]
        pack = self._get_pack()
        # curobo's clique tensor-step C++ backward asserts grad w.r.t. position is contiguous, but our
        # resample -> transpose chain hands back a NON-contiguous grad; coerce it via a hook.
        if position.requires_grad:
            position.register_hook(lambda g: g.contiguous() if g is not None else g)
        if not self.retiming:
            theta = None
        seg = segment_maha2(
            position, self.source_dt, pack, theta=theta, retime_scale=self.retime_scale
        )  # [batch]
        if theta is not None:
            J = pack["n_joints"]
            durations = interval_durations(theta, float(self.source_dt), self.retime_scale)
            seg = seg + self.retime_smooth_weight * warp_smoothness_penalty(theta, self.retime_scale)
            if self._max_vel is not None:
                seg = seg + self.retime_limit_weight * warp_limit_penalty(
                    position[..., :J], durations, self._max_vel, self._max_acc
                )
        # repeat (not expand) -> contiguous cost tensor for the downstream cat_sum / reductions
        return (self.weight * (seg / horizon)).unsqueeze(1).repeat(1, horizon)


# --------------------------------------------------------------------------- #
# per-segment trace (for the tiptop-viz cost-over-time plot)                   #
# --------------------------------------------------------------------------- #
def trajectory_score_trace(
    position: torch.Tensor,
    source_dt: float,
    *,
    checkpoint_path: str = DEFAULT_VAE_MANIFOLD_CKPT,
    n_joints: int = 7,
    tensor_args: Optional[TensorDeviceType] = None,
):
    """Per-timestep VAE-manifold score for one trajectory segment (length == T).

    The filterbank scores the WHOLE segment as one scalar (no per-window decomposition), so
    the trace is that scalar broadcast over the segment's T timesteps -- exactly the cost the
    optimizer sees. position: [T, dof]. Returns a float64 numpy array [T]."""
    if tensor_args is None:
        tensor_args = TensorDeviceType()
    pack = dict(load_vae_manifold(checkpoint_path, tensor_args), n_joints=n_joints)
    if isinstance(position, torch.Tensor):
        pos = position.detach().to(device=tensor_args.device, dtype=tensor_args.dtype)
    else:
        pos = torch.as_tensor(np.asarray(position), device=tensor_args.device, dtype=tensor_args.dtype)
    T = pos.shape[0]
    with torch.no_grad():
        s = float(segment_maha2(pos.unsqueeze(0), source_dt, pack).item())
    return np.full(T, s, dtype=np.float64)
