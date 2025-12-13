from __future__ import annotations
import argparse
import gymnasium as gym
import minigrid
import numpy as np

from rlgrid.envs.wrappers import make_minigrid_wrapped
from rlgrid.mdp import (
    collect_empirical_mdp,
    exact_bisimulation_partition,
    approximate_bisimulation_distance,
    build_state_aggregation_from_partition,
    check_homomorphism,
)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", dest="env_id", required=True)
    p.add_argument("--obs", choices=["image","image_dir","rgb"], default="image")
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", choices=["exact_bisim","sinkhorn_bisim"], default="sinkhorn_bisim")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--reg", type=float, default=1e-2)
    args = p.parse_args()

    env = make_minigrid_wrapped(gym.make(args.env_id), obs_mode=args.obs)
    mdp = collect_empirical_mdp(env, steps=args.steps, seed=args.seed)

    print(f"Empirical MDP: |S|={len(mdp.states)} |A|={len(mdp.actions)}")

    if args.method == "exact_bisim":
        part = exact_bisimulation_partition(mdp, gamma=args.gamma, tol=0.0)
        agg = build_state_aggregation_from_partition(part.cls)
        agg = check_homomorphism(mdp, agg)
        print(f"Exact partition blocks: {len(part.blocks)}")
        print(f"Reward var (mean across blocks/actions): {agg.reward_mse:.6f}")
        print(f"Aggregated transition TV (mean within-block): {agg.trans_tv_mean:.6f}")
    else:
        D = approximate_bisimulation_distance(mdp, gamma=args.gamma, reg=args.reg, n_iters=20)
        # Build a simple clustering by thresholding distances (very simple heuristic).
        # For research usage, replace with hierarchical clustering / k-medoids / etc.
        S = mdp.states
        # adjacency if distance < eps
        eps = np.quantile(list(D.values()), 0.05) if len(D) > 0 else 0.0
        parent = {s:s for s in S}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x,y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        for (s,t), d in D.items():
            if d <= eps:
                union(s,t)

        cls = {}
        block_id = {}
        for s in S:
            r = find(s)
            if r not in block_id:
                block_id[r] = len(block_id)
            cls[s] = block_id[r]

        agg = build_state_aggregation_from_partition(cls)
        agg = check_homomorphism(mdp, agg)
        print(f"Sinkhorn-bisim clustering eps={eps:.6f}: |Z|={agg.n_abs_states}")
        print(f"Reward var (mean across blocks/actions): {agg.reward_mse:.6f}")
        print(f"Aggregated transition TV (mean within-block): {agg.trans_tv_mean:.6f}")

if __name__ == "__main__":
    main()
