#
# RND trajectory-novelty cost (tamp-vla / tiptop-viz manifold study).
#
# Scores how POORLY the DROID human-teleop dataset covers a trajopt segment's joint-space motion,
# with the Random Network Distillation (RND) coverage model trained in tamp-vla/rnd/ (see
# rnd/train.py -> rnd/checkpoints/rnd_droid.pt). Unlike VaeManifoldCost (which MINIMIZES distance
# to the DROID cluster -> pulls motion TOWARD DROID), this cost MAXIMIZES novelty -> pushes motion
# toward regions DROID does NOT cover.
#
# The segment is encoded to the DROID Filterbank-VAE latent EXACTLY as VaeManifoldCost does (same
# resample -> [q|v|a|j] -> standardize -> filterbank encode_mu), because rnd_droid.pt was trained
# on that same vae_full.pt latent. The RND novelty of the latent is the mean predictor-vs-target
# error over the frozen ensemble (rnd/rnd.py::rnd_load): higher = less covered by DROID.
#
# Because the trajopt solver MINIMIZES cost, the cost is weight * (-log(novelty)) so that lowering
# it DRIVES novelty UP. log() (default) tames RND's large dynamic range (raw novelty spans ~1e-3 to
# ~1e4 across the DROID<->cuTAMP gap); set use_log=False to maximize raw novelty instead. log() is
# also bounded above (RND error saturates far off-support), so -log(novelty) is bounded below and
# the reward is self-limiting -- no runaway. weight == 0 disables the cost (CostBase).
#
# Third Party
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# CuRobo
from curobo.types.base import TensorDeviceType

# Local Folder
from .cost_base import CostBase, CostConfig
from .vae_manifold_cost import (
    DEFAULT_VAE_MANIFOLD_CKPT,
    load_vae_manifold,
    positions_to_input,
)

# Default RND checkpoint: tamp-vla/rnd/checkpoints/rnd_droid.pt, resolved RELATIVE to this file so
# it works on any clone of the tamp-vla monorepo (curobo is a submodule at tamp-vla/curobo).
# .../tamp-vla/curobo/src/curobo/rollout/cost/rnd_novelty_cost.py -> parents[5] == tamp-vla.
_REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_RND_NOVELTY_CKPT = os.environ.get(
    "RND_NOVELTY_CKPT", str(_REPO_ROOT / "rnd" / "checkpoints" / "rnd_droid.pt")
)


# --------------------------------------------------------------------------- #
# RND ensemble replica (mirrors rnd/rnd.py::_mlp / rnd_load so the saved         #
# checkpoint loads strictly and scores identically).                           #
# --------------------------------------------------------------------------- #
def _rnd_mlp(in_dim: int, out_dim: int, hidden: int) -> nn.Sequential:
    """Same topology as rnd/rnd.py::_mlp. The random init/seed is irrelevant here: every weight is
    overwritten by load_state_dict below."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


_RND_CACHE = {}


def load_rnd(checkpoint_path: str, tensor_args: TensorDeviceType):
    """Load (and cache) the frozen RND ensemble + input normalization from an rnd_droid.pt-style
    checkpoint (written by rnd/rnd.py::rnd_save). Returns a pack matching rnd_load's score math."""
    key = (checkpoint_path, str(tensor_args.device), str(tensor_args.dtype))
    if key in _RND_CACHE:
        return _RND_CACHE[key]

    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    keep = np.asarray(ck["keep"], bool)
    dev, dtype = tensor_args.device, tensor_args.dtype

    def _t(x):
        return torch.as_tensor(np.asarray(x), device=dev, dtype=dtype)

    members = []
    for m in ck["members"]:
        tgt = _rnd_mlp(cfg["in_dim"], cfg["out_dim"], cfg["target_hidden"])
        tgt.load_state_dict(m["target"])
        g = _rnd_mlp(cfg["in_dim"], cfg["out_dim"], cfg["pred_hidden"])
        g.load_state_dict(m["pred"])
        tgt = tgt.to(device=dev, dtype=dtype).eval()
        g = g.to(device=dev, dtype=dtype).eval()
        for p in list(tgt.parameters()) + list(g.parameters()):
            p.requires_grad_(False)
        members.append((tgt, g, _t(m["tmu"]), _t(m["tsd"])))

    pack = {
        "members": members,
        # keep-mask over the RAW latent dims -> the kept-dim indices RND was standardized on.
        "keep_idx": torch.as_tensor(np.nonzero(keep)[0], device=dev, dtype=torch.long),
        "mu": _t(ck["mu"]),   # z-score of the kept latent dims (train-only stats)
        "sd": _t(ck["sd"]),
    }
    _RND_CACHE[key] = pack
    return pack


def latent_novelty(latent: torch.Tensor, rnd_pack: dict) -> torch.Tensor:
    """RND novelty of each latent -> [B]. Reproduces rnd/rnd.py::rnd_load's score exactly and is
    differentiable in `latent`. Higher = less covered by DROID (more novel)."""
    X = (latent.index_select(1, rnd_pack["keep_idx"]) - rnd_pack["mu"]) / rnd_pack["sd"]
    errs = [((g(X) - (tgt(X) - tmu) / tsd) ** 2).mean(dim=1)
            for tgt, g, tmu, tsd in rnd_pack["members"]]
    return torch.stack(errs, dim=0).mean(dim=0)


def segment_novelty(position: torch.Tensor, source_dt: float, vae_pack: dict, rnd_pack: dict) -> torch.Tensor:
    """RND novelty of each segment's motion -> [B]. Encodes to the DROID VAE latent (identical to
    VaeManifoldCost) then applies the RND ensemble."""
    x, mask = positions_to_input(position, source_dt, vae_pack)
    latent = vae_pack["model"].encode_mu(x, mask)   # [B, latent_dim] -- the latent RND was trained on
    return latent_novelty(latent, rnd_pack)


# --------------------------------------------------------------------------- #
# cost config + module                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class RndNoveltyCostConfig(CostConfig):
    vae_checkpoint_path: str = DEFAULT_VAE_MANIFOLD_CKPT   # encoder (shared with VaeManifoldCost)
    rnd_checkpoint_path: str = DEFAULT_RND_NOVELTY_CKPT     # RND heads (rnd/checkpoints/rnd_droid.pt)
    n_joints: int = 7
    source_dt: float = 0.15         # trajopt base_dt (gradient_trajopt.yml model.dt_traj_params.base_dt)
    use_log: bool = True            # maximize log(novelty) (default, tames the range) vs raw novelty
    log_eps: float = 1e-6

    def __post_init__(self):
        return super().__post_init__()


class RndNoveltyCost(CostBase, RndNoveltyCostConfig):
    """Per-segment RND novelty *reward* -- MAXIMIZE how poorly DROID covers a segment's motion.

    forward(position) -> [batch, horizon]. Each segment is encoded to one latent; its RND novelty
    is turned into the cost weight * (-log(novelty)) (or -novelty when use_log=False) and spread
    uniformly over the horizon (so the horizon sum equals weight * (-log(novelty)), and gradients
    flow through the encoder + RND heads to every waypoint). Because trajopt MINIMIZES cost, this
    drives novelty UP. weight == 0 disables the cost (CostBase)."""

    def __init__(self, config: Optional[RndNoveltyCostConfig] = None):
        if config is not None:
            RndNoveltyCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
        self._init_post_config()
        self._vae_pack = None  # lazy-loaded on first enabled forward
        self._rnd_pack = None

    def _get_packs(self):
        if self._vae_pack is None:
            self._vae_pack = load_vae_manifold(self.vae_checkpoint_path, self.tensor_args)
            if self.n_joints != self._vae_pack["n_joints"]:
                self._vae_pack = dict(self._vae_pack, n_joints=self.n_joints)
            self._rnd_pack = load_rnd(self.rnd_checkpoint_path, self.tensor_args)
        return self._vae_pack, self._rnd_pack

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        # position: [batch, horizon, dof]
        horizon = position.shape[1]
        vae_pack, rnd_pack = self._get_packs()
        # curobo's clique tensor-step C++ backward asserts grad w.r.t. position is contiguous, but our
        # resample -> transpose chain hands back a NON-contiguous grad; coerce it via a hook.
        if position.requires_grad:
            position.register_hook(lambda g: g.contiguous() if g is not None else g)
        nov = segment_novelty(position, self.source_dt, vae_pack, rnd_pack)   # [batch], >= 0
        reward = torch.log(nov + self.log_eps) if self.use_log else nov
        # MINIMIZE -reward -> MAXIMIZE novelty. repeat (not expand) -> contiguous cost tensor.
        return (self.weight * (-reward / horizon)).unsqueeze(1).repeat(1, horizon)


# --------------------------------------------------------------------------- #
# per-segment trace (for the tiptop-viz cost-over-time plot)                   #
# --------------------------------------------------------------------------- #
def trajectory_novelty_trace(
    position: torch.Tensor,
    source_dt: float,
    *,
    vae_checkpoint_path: str = DEFAULT_VAE_MANIFOLD_CKPT,
    rnd_checkpoint_path: str = DEFAULT_RND_NOVELTY_CKPT,
    n_joints: int = 7,
    tensor_args: Optional[TensorDeviceType] = None,
):
    """Per-timestep RND novelty for one trajectory segment (length == T).

    The filterbank scores the WHOLE segment as one scalar (no per-window decomposition), so the
    trace is that novelty broadcast over the segment's T timesteps. Reports the raw novelty (the
    model output the cost maximizes). position: [T, dof]. Returns a float64 numpy array [T]."""
    if tensor_args is None:
        tensor_args = TensorDeviceType()
    vae_pack = dict(load_vae_manifold(vae_checkpoint_path, tensor_args), n_joints=n_joints)
    rnd_pack = load_rnd(rnd_checkpoint_path, tensor_args)
    if isinstance(position, torch.Tensor):
        pos = position.detach().to(device=tensor_args.device, dtype=tensor_args.dtype)
    else:
        pos = torch.as_tensor(np.asarray(position), device=tensor_args.device, dtype=tensor_args.dtype)
    T = pos.shape[0]
    with torch.no_grad():
        s = float(segment_novelty(pos.unsqueeze(0), source_dt, vae_pack, rnd_pack).item())
    return np.full(T, s, dtype=np.float64)
