from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Hashable, List, Tuple
import numpy as np

from rlgrid.mdp.empirical import EmpiricalMDP

def sinkhorn(a, b, M, reg=1e-2, num_iters=200):
    '''
    Sinkhorn distance between two discrete distributions a,b over same support with cost matrix M.
    This is an entropic-regularized OT proxy to the Kantorovich metric.
    '''
    a = a / (a.sum() + 1e-12)
    b = b / (b.sum() + 1e-12)
    K = np.exp(-M / max(1e-12, reg))
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(num_iters):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)
    P = np.outer(u, v) * K
    return float(np.sum(P * M))

def approximate_bisimulation_distance(
    mdp: EmpiricalMDP,
    gamma: float = 0.99,
    reg: float = 1e-2,
    n_iters: int = 50,
) -> Dict[Tuple[Hashable, Hashable], float]:
    '''
    Approximate bisimulation metric on a finite MDP using a Sinkhorn OT proxy.

    Iteration:
      d_{k+1}(s,t) = max_a |R(s,a)-R(t,a)| + gamma * W_dk(P(.|s,a), P(.|t,a))

    where W is an OT distance using cost matrix = current distances. Sinkhorn is used for scalability.

    Returns pairwise distances for all observed states (O(|S|^2) memory).
    Intended for small/medium empirical MDPs (e.g., <= 2-5k states).
    '''
    S = list(mdp.states)
    idx = {s:i for i,s in enumerate(S)}
    A = mdp.actions
    n = len(S)
    D = np.zeros((n, n), dtype=np.float32)

    # precompute transition vectors per (s,a) in common support
    Pvec = {}
    for s in S:
        for a in A:
            v = np.zeros((n,), dtype=np.float32)
            for sp, p in mdp.P.get(s, {}).get(a, {}).items():
                if sp in idx:
                    v[idx[sp]] += float(p)
            Pvec[(s,a)] = v

    for _ in range(n_iters):
        Dnew = np.zeros_like(D)
        for i, s in enumerate(S):
            for j, t in enumerate(S):
                if i == j:
                    continue
                best = 0.0
                for a in A:
                    rdiff = abs(float(mdp.R.get(s, {}).get(a, 0.0)) - float(mdp.R.get(t, {}).get(a, 0.0)))
                    # OT distance between next-state distributions using current D as cost matrix
                    w = sinkhorn(Pvec[(s,a)], Pvec[(t,a)], D, reg=reg, num_iters=80)
                    best = max(best, rdiff + gamma * w)
                Dnew[i, j] = best
        D = Dnew

    out = {}
    for i,s in enumerate(S):
        for j,t in enumerate(S):
            if i < j:
                out[(s,t)] = float(D[i,j])
    return out
