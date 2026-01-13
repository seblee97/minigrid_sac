from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
from collections import defaultdict

import numpy as np


def next_state(P: np.ndarray, s: int, a: int) -> int:
    return int(np.argmax(P[s, a]))


def deterministic_bisimulation(P: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Coarsest bisimulation partition for deterministic finite MDPs.
    """
    S, A, _ = P.shape
    label = np.arange(S)

    changed = True
    while changed:
        changed = False
        sig_to_id: Dict[Tuple, int] = {}
        new_label = np.zeros_like(label)
        next_id = 0

        remap = {u: i for i, u in enumerate(np.unique(label))}

        for s in range(S):
            sig = tuple(
                (R[s, a], remap[label[next_state(P, s, a)]])
                for a in range(A)
            )
            if sig not in sig_to_id:
                sig_to_id[sig] = next_id
                next_id += 1
            new_label[s] = sig_to_id[sig]

        if not np.array_equal(new_label, label):
            label = new_label
            changed = True

    return label


@dataclass(frozen=True)
class JointSAAbstraction:
    phi: np.ndarray      # (S,)
    sa: np.ndarray       # (S,A)
    n_abs_states: int
    n_abs_actions: int


def deterministic_joint_sa_abstraction(P: np.ndarray, R: np.ndarray) -> JointSAAbstraction:
    """
    Coarsest joint state–action abstraction (deterministic).
    """
    S, A, _ = P.shape
    phi = np.arange(S)
    sa = np.arange(S * A).reshape(S, A)

    changed = True
    while changed:
        changed = False

        # refine SA-classes
        remap = {u: i for i, u in enumerate(np.unique(phi))}
        sa_sig_to_id: Dict[Tuple, int] = {}
        new_sa = np.zeros_like(sa)
        next_id = 0

        for s in range(S):
            for a in range(A):
                sig = (R[s, a], remap[phi[next_state(P, s, a)]])
                if sig not in sa_sig_to_id:
                    sa_sig_to_id[sig] = next_id
                    next_id += 1
                new_sa[s, a] = sa_sig_to_id[sig]

        if not np.array_equal(new_sa, sa):
            sa = new_sa
            changed = True

        # refine states by SA-set
        phi_sig_to_id: Dict[Tuple[int, ...], int] = {}
        new_phi = np.zeros_like(phi)
        next_id = 0

        for s in range(S):
            sig = tuple(sorted(sa[s]))
            if sig not in phi_sig_to_id:
                phi_sig_to_id[sig] = next_id
                next_id += 1
            new_phi[s] = phi_sig_to_id[sig]

        if not np.array_equal(new_phi, phi):
            phi = new_phi
            changed = True

    return JointSAAbstraction(
        phi=phi,
        sa=sa,
        n_abs_states=int(phi.max()) + 1,
        n_abs_actions=int(sa.max()) + 1,
    )


def joint_state_action_abstraction(P, R, *, atol=0.0):
    """
    Compute the coarsest exact MDP homomorphism (joint state+action abstraction).

    Inputs
    ------
    P : (S, A, S) ndarray
        Transition probabilities P[s, a, s'].
    R : (S, A) ndarray
        Immediate expected rewards R[s, a].
    atol : float, default 0.0
        Tolerance for equality (use 0 for strict exactness; small >0 for fp noise).

    Returns
    -------
    state_blocks : list[list[int]]
        Partition of states; each inner list is an abstract state (block of base states).
    sa_blocks : list[list[tuple[int,int]]]
        Partition of state-actions; each inner list is an abstract action class
        (pairs (s,a)) that are equivalent.
    state_label : (S,) ndarray[int]
        Abstract-state id for each base state s.
    sa_label : (S, A) ndarray[int]
        Abstract SA-class id for each base (s,a).
    P_tilde : (K, A_b_max, K) ndarray
        Abstract transitions. Rows beyond each state's actual abstract-action count
        are zero. K = #abstract states; A_b_max = max #abstract actions over blocks.
    R_tilde : (K, A_b_max) ndarray
        Abstract rewards aligned with P_tilde.
    action_index_per_block : list[dict[int,int]]
        For block b: a dict mapping global sa_class_id -> local abstract-action index (0..A_b-1).
        Use: a_tilde = action_index_per_block[b][sa_label[s,a]] for any s in block b.
    """
    S, A, S2 = P.shape
    assert S == S2 and R.shape == (S, A)

    def key_array(x):
        if atol == 0.0:
            return tuple(x.tolist())
        return tuple(np.round(x / max(atol, 1e-12)).astype(np.int64).tolist())

    # Initialize all states in one block; SA classes uninitialized (-1)
    state_label = np.zeros(S, dtype=int)
    sa_label = -np.ones((S, A), dtype=int)

    def mass_to_blocks(current_state_label):
        K = current_state_label.max() + 1
        M = np.zeros((S, A, K), dtype=np.float64)
        for s in range(S):
            for a in range(A):
                for sp in range(S):
                    M[s, a, current_state_label[sp]] += P[s, a, sp]
        return M

    changed = True
    while changed:
        changed = False
        # Step 1: refine SA classes given current state blocks
        M = mass_to_blocks(state_label)  # (S, A, K)
        sig2class = {}
        next_id = 0
        new_sa_label = np.empty_like(sa_label)

        for s in range(S):
            for a in range(A):
                sig = (key_array(np.array([R[s, a]])), key_array(M[s, a]))
                if sig not in sig2class:
                    sig2class[sig] = next_id
                    next_id += 1
                new_sa_label[s, a] = sig2class[sig]

        if not np.array_equal(new_sa_label, sa_label):
            changed = True
            sa_label = new_sa_label

        # Step 2: refine state blocks from SA classes (treat actions as unlabeled)
        # For state s, signature is the *set* of SA class ids it offers (order-invariant).
        # Using set (presence) rather than counts ensures a valid g(s,·) mapping exists.
        sig2block = {}
        next_block = 0
        new_state_label = np.empty_like(state_label)
        for s in range(S):
            # presence bitset-ish signature via sorted tuple of unique classes
            sig = tuple(sorted(set(sa_label[s, :].tolist())))
            if sig not in sig2block:
                sig2block[sig] = next_block
                next_block += 1
            new_state_label[s] = sig2block[sig]

        if not np.array_equal(new_state_label, state_label):
            changed = True
            state_label = new_state_label

    # Build output partitions
    K = state_label.max() + 1
    state_blocks = [[] for _ in range(K)]
    for s in range(S):
        state_blocks[state_label[s]].append(s)

    C = sa_label.max() + 1
    sa_blocks = [[] for _ in range(C)]
    for s in range(S):
        for a in range(A):
            sa_blocks[sa_label[s, a]].append((s, a))

    # For each abstract state block b, determine which SA classes appear there,
    # and create a local index mapping for abstract actions of b.
    action_index_per_block = []
    max_actions = 0
    for b in range(K):
        classes = sorted({sa_label[s, a] for s in state_blocks[b] for a in range(A)})
        mapping = {cls: i for i, cls in enumerate(classes)}
        action_index_per_block.append(mapping)
        max_actions = max(max_actions, len(classes))

    # Construct abstract MDP (P_tilde, R_tilde)
    P_tilde = np.zeros((K, max_actions, K), dtype=np.float64)
    R_tilde = np.zeros((K, max_actions), dtype=np.float64)

    for b in range(K):
        reps = state_blocks[b]
        assert len(reps) > 0
        s0 = reps[0]  # any representative state in block b
        mapping = action_index_per_block[b]

        # Sanity: the set of SA classes must be identical for all states in the block
        base_set = {sa_label[s0, a] for a in range(A)}
        for s in reps[1:]:
            assert {
                sa_label[s, a] for a in range(A)
            } == base_set, "Inconsistent SA-classes within a state block (should not happen at fixpoint)."

        # For each SA class available in this block, pick any (s,a) in the block with that class
        inv_map = defaultdict(list)
        for a in range(A):
            inv_map[sa_label[s0, a]].append((s0, a))

        for cls, idx in mapping.items():
            # pick representative (s,a) for this abstract action
            s_rep, a_rep = inv_map[cls][0]
            # reward
            R_tilde[b, idx] = R[s_rep, a_rep]
            # transitions to abstract states
            for sp in range(S):
                bp = state_label[sp]
                P_tilde[b, idx, bp] += P[s_rep, a_rep, sp]

        # Optional consistency checks (remove if speed matters):
        # All (s,a) in this cls within this block must produce the same R and P_tilde row
        # due to construction; if you like, assert here.

    return (
        state_blocks,
        sa_blocks,
        state_label,
        sa_label,
        P_tilde,
        R_tilde,
        action_index_per_block,
    )