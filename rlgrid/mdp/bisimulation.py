from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Hashable, List, Tuple, Set
import numpy as np

from rlgrid.mdp.empirical import EmpiricalMDP

@dataclass
class Partition:
    # class_id per state
    cls: Dict[Hashable, int]
    # reverse index: class -> states
    blocks: Dict[int, List[Hashable]]

def exact_bisimulation_partition(mdp: EmpiricalMDP, gamma: float = 0.99, tol: float = 0.0) -> Partition:
    '''
    Exact partition refinement for a *finite* MDP given explicit P and R.

    Two states are equivalent if, for all actions:
      - rewards match (within tol)
      - transition probabilities aggregated by current partition match (within tol)

    This is the classic iterative refinement scheme (Paige-Tarjan-style optimization omitted for clarity).
    '''
    states = list(mdp.states)
    actions = mdp.actions

    # Initial partition: by reward vector over actions
    def reward_signature(s):
        return tuple(round(float(mdp.R.get(s, {}).get(a, 0.0))/max(1e-12, tol) if tol>0 else float(mdp.R.get(s, {}).get(a, 0.0)), 12) for a in actions)

    sig_to_block: Dict[Tuple, int] = {}
    cls: Dict[Hashable, int] = {}
    for s in states:
        sig = reward_signature(s)
        if sig not in sig_to_block:
            sig_to_block[sig] = len(sig_to_block)
        cls[s] = sig_to_block[sig]

    changed = True
    while changed:
        changed = False
        # build blocks
        blocks: Dict[int, List[Hashable]] = {}
        for s, b in cls.items():
            blocks.setdefault(b, []).append(s)

        new_cls: Dict[Hashable, int] = {}
        new_sig_to_block: Dict[Tuple, int] = {}

        for s in states:
            b = cls[s]
            # build transition signature: for each action, probability mass to each block
            t_sig = []
            for a in actions:
                sp_dict = mdp.P.get(s, {}).get(a, {})
                mass = {}
                for sp, p in sp_dict.items():
                    mass[cls.get(sp, -1)] = mass.get(cls.get(sp, -1), 0.0) + float(p)
                # deterministic signature ordering
                t_sig.append(tuple(sorted((k, round(v, 12)) for k, v in mass.items())))
            sig = (cls[s], reward_signature(s), tuple(t_sig))
            if sig not in new_sig_to_block:
                new_sig_to_block[sig] = len(new_sig_to_block)
            new_cls[s] = new_sig_to_block[sig]
            if new_cls[s] != cls[s]:
                changed = True

        cls = new_cls

    # final blocks
    blocks: Dict[int, List[Hashable]] = {}
    for s, b in cls.items():
        blocks.setdefault(b, []).append(s)
    return Partition(cls=cls, blocks=blocks)
