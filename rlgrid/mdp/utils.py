import numpy as np


def dist_dict_to_matrix(states, dist_dict):
    """
    dist_dict: {(s,t): d} for s<t (as returned by approximate_bisimulation_distance)
    returns: D [N,N] symmetric with zeros on diagonal
    """
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    D = np.zeros((n, n), dtype=np.float32)
    for (s, t), d in dist_dict.items():
        i = idx.get(s, None)
        j = idx.get(t, None)
        if i is None or j is None:
            continue
        D[i, j] = float(d)
        D[j, i] = float(d)
    return D


def cluster_eps_unionfind(D: np.ndarray, eps: float) -> np.ndarray:
    """
    Simple threshold clustering: connect i~j if D[i,j] <= eps, return component labels.
    """
    n = D.shape[0]
    parent = np.arange(n, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # O(N^2) scan (fine for a few thousand states)
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= eps:
                union(i, j)

    # compress + relabel
    roots = [find(i) for i in range(n)]
    remap = {}
    labels = np.zeros(n, dtype=np.int64)
    k = 0
    for i, r in enumerate(roots):
        if r not in remap:
            remap[r] = k
            k += 1
        labels[i] = remap[r]
    return labels


def cluster_kmedoids(D: np.ndarray, k: int, iters: int = 10, seed: int = 0) -> np.ndarray:
    """
    Lightweight k-medoids using only distances (no sklearn dependency).
    Complexity ~ O(iters * k * N^2) worst-case; fine for moderate N and modest k.
    """
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    k = max(1, min(k, n))

    # init medoids uniformly
    medoids = rng.choice(n, size=k, replace=False)

    def assign(medoids_):
        # assign each point to nearest medoid
        dist_to_m = D[:, medoids_]
        return np.argmin(dist_to_m, axis=1)

    labels = assign(medoids)

    for _ in range(iters):
        changed = False
        for mi in range(k):
            members = np.where(labels == mi)[0]
            if members.size == 0:
                continue
            # choose member that minimizes sum distance within cluster
            subD = D[np.ix_(members, members)]
            costs = subD.sum(axis=1)
            best_member = members[np.argmin(costs)]
            if best_member != medoids[mi]:
                medoids[mi] = best_member
                changed = True
        new_labels = assign(medoids)
        if np.all(new_labels == labels) and not changed:
            break
        labels = new_labels

    return labels

def reward_homomorphism_partition(mdp):
    """
    Partition states by exact reward signature across actions.
    """
    sig_to_gid = {}
    state_to_gid = {}
    gid = 0

    for s in mdp.states:
        sig = tuple(
            mdp.R[s][a] if a in mdp.R[s] else 0.0
            for a in mdp.actions
        )
        if sig not in sig_to_gid:
            sig_to_gid[sig] = gid
            gid += 1
        state_to_gid[s] = sig_to_gid[sig]

    return state_to_gid


def transition_homomorphism_partition(mdp, tol=1e-3):
    """
    Partition states by aggregated transition signatures.
    """
    sig_to_gid = {}
    state_to_gid = {}
    gid = 0

    for s in mdp.states:
        sig = []
        for a in mdp.actions:
            dist = mdp.P[s].get(a, {})
            # sorted to canonicalize
            sig.append(tuple(sorted(dist.items())))
        sig = tuple(sig)

        if sig not in sig_to_gid:
            sig_to_gid[sig] = gid
            gid += 1
        state_to_gid[s] = sig_to_gid[sig]

    return state_to_gid
