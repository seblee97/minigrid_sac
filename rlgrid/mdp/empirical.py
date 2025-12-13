from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Tuple, Callable, Optional
import numpy as np

@dataclass
class EmpiricalMDP:
    # finite sets inferred from sampling
    states: List[Hashable]
    actions: List[int]
    # transitions: P[s][a] -> dict{s': p}
    P: Dict[Hashable, Dict[int, Dict[Hashable, float]]]
    # expected reward: R[s][a] -> float
    R: Dict[Hashable, Dict[int, float]]

def default_state_key(obs: np.ndarray) -> bytes:
    # Works for small uint8 images. For safety, copy to C-contiguous.
    return np.ascontiguousarray(obs).tobytes()

def collect_empirical_mdp(env, steps: int, state_key: Callable[[Any], Hashable] = default_state_key,
                          seed: int = 0) -> EmpiricalMDP:
    '''
    Collects an empirical MDP from an environment by sampling transitions.

    For MiniGrid with partial observability, this yields an *observation-MDP*.
    To target the true underlying MDP, provide a state_key that hashes the full simulator state.
    '''
    obs, _ = env.reset(seed=seed)
    A = env.action_space.n
    actions = list(range(A))

    # counts
    trans_counts: Dict[Hashable, Dict[int, Dict[Hashable, int]]] = {}
    reward_sums: Dict[Hashable, Dict[int, float]] = {}
    sa_counts: Dict[Hashable, Dict[int, int]] = {}

    for _ in range(steps):
        s = state_key(obs)
        a = env.action_space.sample()
        next_obs, r, terminated, truncated, _ = env.step(a)
        sp = state_key(next_obs)
        done = terminated or truncated

        trans_counts.setdefault(s, {}).setdefault(a, {}).setdefault(sp, 0)
        trans_counts[s][a][sp] += 1
        reward_sums.setdefault(s, {}).setdefault(a, 0.0)
        reward_sums[s][a] += float(r)
        sa_counts.setdefault(s, {}).setdefault(a, 0)
        sa_counts[s][a] += 1

        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs

    # build probabilities
    P: Dict[Hashable, Dict[int, Dict[Hashable, float]]] = {}
    R: Dict[Hashable, Dict[int, float]] = {}
    states = set()

    for s, a_dict in trans_counts.items():
        states.add(s)
        P[s] = {}
        R[s] = {}
        for a, sp_dict in a_dict.items():
            n = sum(sp_dict.values())
            P[s][a] = {sp: c / n for sp, c in sp_dict.items()}
            R[s][a] = reward_sums[s][a] / max(1, sa_counts[s][a])
            for sp in sp_dict.keys():
                states.add(sp)

    return EmpiricalMDP(states=list(states), actions=actions, P=P, R=R)
