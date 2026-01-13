from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Hashable, List

import numpy as np
from rlgrid.mdp.empirical import EmpiricalMDP


@dataclass(frozen=True)
class FiniteMDP:
    P: np.ndarray   # (S, A, S) one-hot
    R: np.ndarray   # (S, A)
    states: List[Hashable]
    actions: List[Hashable]
    s2i: Dict[Hashable, int]
    a2i: Dict[Hashable, int]


def prepare_abstraction(mdp: EmpiricalMDP) -> FiniteMDP:
    """
    Convert EmpiricalMDP into deterministic finite-MDP arrays (P, R).

    Assumes the MDP is deterministic.
    Raises if any (s,a) has multiple next states.
    """
    states = list(mdp.states)
    actions = list(mdp.actions)
    S, A = len(states), len(actions)

    s2i = {s: i for i, s in enumerate(states)}
    a2i = {a: i for i, a in enumerate(actions)}

    P = np.zeros((S, A, S), dtype=np.float64)
    R = np.zeros((S, A), dtype=np.float64)

    for s in states:
        si = s2i[s]
        for a in actions:
            ai = a2i[a]
            rew = float(mdp.R.get(s, {}).get(a, 0.0))
            if rew != 0.0:
                R[si, ai] = 1.0

            dist = mdp.P.get(s, {}).get(a, {})
            if len(dist) == 0:
                continue
            if len(dist) != 1:
                raise ValueError(
                    f"Non-deterministic transition at state {s}, action {a}: {dist}"
                )

            sp, _ = next(iter(dist.items()))
            P[si, ai, s2i[sp]] = 1.0

    return FiniteMDP(P=P, R=R, states=states, actions=actions, s2i=s2i, a2i=a2i)
