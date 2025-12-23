from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Callable, Any

import numpy as np
import torch
import torch.nn as nn


def _flatten_rep(x: torch.Tensor) -> torch.Tensor:
    # [B, ...] -> [B, D]
    return x.flatten(start_dim=1)


def cosine_rsa(reps: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    reps: [N, D] float32/float64
    returns: [N, N] cosine similarity matrix
    """
    import pdb; pdb.set_trace()
    X = reps.astype(np.float32, copy=False)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(nrm, eps)
    return Xn @ Xn.T


@dataclass
class LayerReps:
    layer_names: List[str]
    reps: Dict[str, np.ndarray]  # layer -> [N, D]


def select_default_layers(net: nn.Module) -> List[str]:
    """
    Default: all leaf Conv2d and Linear modules (good “standard RL metrics” choice).
    You can customize to include ReLUs etc if you want.
    """
    names: List[str] = []
    for name, m in net.named_modules():
        if name == "":
            continue
        is_leaf = len(list(m.children())) == 0
        if not is_leaf:
            continue
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            names.append(name)
    # stable order
    return names


@torch.no_grad()
def compute_layer_reps(
    net: nn.Module,
    obs_batch: torch.Tensor,
    layer_names: Optional[List[str]] = None,
    device: str | torch.device = "cpu",
    forward_fn: Optional[Callable[[torch.Tensor], Any]] = None,
) -> LayerReps:
    """
    Hooks module outputs for requested layers.

    - net: torch.nn.Module
    - obs_batch: [N, ...] torch tensor, already in the same preprocessed format the net expects.
    - forward_fn: optional callable. If not provided, uses net(obs_batch).
    """
    net = net.to(device)
    net.eval()

    if layer_names is None:
        layer_names = select_default_layers(net)

    # map name -> module
    name_to_mod = dict(net.named_modules())
    missing = [n for n in layer_names if n not in name_to_mod]
    if missing:
        raise KeyError(f"Missing layers in net: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    outputs: Dict[str, List[torch.Tensor]] = {n: [] for n in layer_names}
    hooks = []

    def _make_hook(nm: str):
        def hook(_m, _inp, out):
            if isinstance(out, (tuple, list)):
                out0 = out[0]
            else:
                out0 = out
            outputs[nm].append(_flatten_rep(out0).detach().cpu())
        return hook

    for nm in layer_names:
        hooks.append(name_to_mod[nm].register_forward_hook(_make_hook(nm)))

    obs_batch = obs_batch.to(device)
    if forward_fn is None:
        _ = net(obs_batch)
    else:
        _ = forward_fn(obs_batch)

    for h in hooks:
        h.remove()

    reps: Dict[str, np.ndarray] = {}
    for nm in layer_names:
        if len(outputs[nm]) != 1:
            # we expect one forward pass
            cat = torch.cat(outputs[nm], dim=0) if outputs[nm] else torch.empty((obs_batch.shape[0], 0))
        else:
            cat = outputs[nm][0]
        reps[nm] = cat.numpy()

    return LayerReps(layer_names=layer_names, reps=reps)


@dataclass
class PairStats:
    within_mean: float
    within_var: float
    between_mean: float
    between_var: float
    within_pairs: int
    between_pairs: int


def rsa_group_pair_stats(sim: np.ndarray, groups: np.ndarray) -> PairStats:
    """
    Compute mean/var of similarities for pairs (i<j) within same group vs different groups.
    groups: [N] int group id. States with group < 0 are ignored.
    """
    import pdb; pdb.set_trace()
    assert sim.shape[0] == sim.shape[1]
    N = sim.shape[0]
    assert groups.shape[0] == N

    valid = groups >= 0
    idx = np.nonzero(valid)[0]
    if len(idx) < 2:
        return PairStats(
            within_mean=float("nan"), within_var=float("nan"),
            between_mean=float("nan"), between_var=float("nan"),
            within_pairs=0, between_pairs=0,
        )

    sim_v = sim[np.ix_(idx, idx)]
    g = groups[idx]

    # use upper triangle only (exclude diagonal)
    iu = np.triu_indices(sim_v.shape[0], k=1)
    svals = sim_v[iu]
    gi = g[iu[0]]
    gj = g[iu[1]]
    same = (gi == gj)

    within = svals[same]
    between = svals[~same]

    def mean_var(x: np.ndarray) -> Tuple[float, float]:
        if x.size == 0:
            return float("nan"), float("nan")
        m = float(x.mean())
        v = float(x.var(ddof=0))
        return m, v

    w_m, w_v = mean_var(within)
    b_m, b_v = mean_var(between)

    return PairStats(
        within_mean=w_m, within_var=w_v,
        between_mean=b_m, between_var=b_v,
        within_pairs=int(within.size), between_pairs=int(between.size),
    )
