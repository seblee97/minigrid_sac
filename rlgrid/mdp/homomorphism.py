from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple
import numpy as np

from rlgrid.mdp.empirical import EmpiricalMDP

@dataclass
class Aggregation:
    phi: Dict[Hashable, int]          # state -> abstract state id
    n_abs_states: int
    # empirical check scores
    reward_mse: float
    trans_tv_mean: float

def build_state_aggregation_from_partition(partition_cls: Dict[Hashable, int]) -> Aggregation:
    blocks = {}
    for s,b in partition_cls.items():
        blocks.setdefault(b, []).append(s)
    return Aggregation(phi=partition_cls, n_abs_states=len(blocks), reward_mse=0.0, trans_tv_mean=0.0)

def check_homomorphism(mdp: EmpiricalMDP, agg: Aggregation) -> Aggregation:
    '''
    Checks how well phi induces an approximate MDP homomorphism by measuring:
      - reward MSE across states within each abstract state, per action
      - transition total variation between states within each abstract state, per action, after aggregation
    '''
    phi = agg.phi
    A = mdp.actions

    # aggregated transition operator
    # for each concrete state s, action a -> distribution over abstract states
    def agg_trans(s, a):
        dist = {}
        for sp, p in mdp.P.get(s, {}).get(a, {}).items():
            dist[phi.get(sp, -1)] = dist.get(phi.get(sp, -1), 0.0) + float(p)
        return dist

    # group by abstract state
    groups = {}
    for s, z in phi.items():
        groups.setdefault(z, []).append(s)

    reward_errs = []
    tv_errs = []

    for z, states in groups.items():
        if len(states) <= 1:
            continue
        for a in A:
            rs = [float(mdp.R.get(s, {}).get(a, 0.0)) for s in states]
            reward_errs.append(np.var(rs))

            # TV between all pairs in group
            for i in range(len(states)):
                for j in range(i+1, len(states)):
                    di = agg_trans(states[i], a)
                    dj = agg_trans(states[j], a)
                    keys = set(di.keys()) | set(dj.keys())
                    tv = 0.5 * sum(abs(di.get(k,0.0) - dj.get(k,0.0)) for k in keys)
                    tv_errs.append(tv)

    agg.reward_mse = float(np.mean(reward_errs)) if reward_errs else 0.0
    agg.trans_tv_mean = float(np.mean(tv_errs)) if tv_errs else 0.0
    return agg
