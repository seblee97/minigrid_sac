from __future__ import annotations

import argparse
import glob
import json
import os
import time
from typing import Dict, Any, Hashable, List, Tuple, Optional

import numpy as np
import gymnasium as gym
import torch
import minigrid  # ensure MiniGrid envs register

from rlgrid.envs.wrappers import make_minigrid_wrapped
from rlgrid.mdp.empirical import collect_empirical_mdp, default_state_key
from rlgrid.mdp.bisimulation import exact_bisimulation_partition
from rlgrid.mdp.homomorphism import build_state_aggregation_from_partition, check_homomorphism
from rlgrid.mdp.approx_bisim_metric import approximate_bisimulation_distance
from rlgrid.mdp.utils import (
    dist_dict_to_matrix,
    cluster_eps_unionfind,
    cluster_kmedoids,
    reward_homomorphism_partition,
    transition_homomorphism_partition,
)

from rlgrid.analysis.rsa import compute_layer_reps, cosine_rsa, rsa_group_pair_stats


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_run_dir(path: str) -> str:
    """
    Accept either:
      - exact run directory containing config.json (preferred), or
      - a parent directory that contains one timestamp subdir with config.json (fallback)
    """
    path = os.path.abspath(path)
    if os.path.isfile(os.path.join(path, "config.json")):
        return path

    # Fallback: search immediate children
    kids = sorted(glob.glob(os.path.join(path, "*")))
    for k in kids:
        if os.path.isdir(k) and os.path.isfile(os.path.join(k, "config.json")):
            return k

    raise FileNotFoundError(
        f"Could not find config.json in run-dir '{path}' or its immediate subdirectories."
    )


def collect_state_obs_map(env, steps: int, seed: int = 0) -> Dict[Hashable, np.ndarray]:
    """
    Collect a representative observation for each visited state_key.
    This corresponds to "all states in the original MDP" as discovered by sampling.
    """
    obs, _ = env.reset(seed=seed)
    obs_map: Dict[Hashable, np.ndarray] = {}

    for _ in range(steps):
        k = default_state_key(obs)
        if k not in obs_map:
            obs_map[k] = np.array(obs, copy=True)
        a = env.action_space.sample()
        next_obs, _, terminated, truncated, _ = env.step(a)
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

    return obs_map


def rollout_states(env, steps: int, seed: int = 123) -> Tuple[List[Hashable], np.ndarray]:
    """
    Collect a test rollout’s states (keys + obs tensor).
    """
    obs, _ = env.reset(seed=seed)
    keys: List[Hashable] = []
    obss: List[np.ndarray] = []
    for _ in range(steps):
        k = default_state_key(obs)
        keys.append(k)
        obss.append(np.array(obs, copy=True))
        a = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(a)
        if terminated or truncated:
            obs, _ = env.reset(seed=seed)
    return keys, np.stack(obss, axis=0)


def load_model_and_net(algo: str, ckpt_path: str, env):
    """
    Loads checkpoint and returns (model, torch_module_to_hook, forward_fn).
    forward_fn is used so we can hook layers even if forward() returns tuples.
    """
    algo = algo.lower()
    if algo == "ppo":
        from rlgrid.algos import PPO

        model = PPO.load(ckpt_path, env=env, device="cpu")
        net = model.ac
        forward_fn = lambda x: net.forward(x)
        return model, net, forward_fn
    if algo == "a2c":
        from rlgrid.algos import A2C

        model = A2C.load(ckpt_path, env=env, device="cpu")
        net = model.ac
        forward_fn = lambda x: net.forward(x)
        return model, net, forward_fn
    if algo == "dqn":
        from rlgrid.algos import DQN

        model = DQN.load(ckpt_path, env=env, device="cpu")
        net = model.q
        forward_fn = lambda x: net.forward(x)
        return model, net, forward_fn
    if algo == "qrdqn":
        from rlgrid.algos import QRDQN

        model = QRDQN.load(ckpt_path, env=env, device="cpu")
        net = model.q
        forward_fn = lambda x: net.forward(x)
        return model, net, forward_fn
    raise ValueError(f"Unknown algo: {algo}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to a timestamped run folder containing config.json (e.g. runs/exp/13-12-2025-14-37/). "
        "If you pass the parent (runs/exp/), we try to infer a child with config.json.",
    )

    # Analysis controls (everything else inferred from run-dir/config.json)
    p.add_argument("--mdp-steps", type=int, default=200000)
    p.add_argument("--mdp-seed", type=int, default=-1, help="override; -1 uses run config seed")
    p.add_argument(
        "--reduction",
        choices=[
            "exact_bisim",
            "bisim_metric_eps",
            "bisim_metric_kmedoids",
            "homomorphism_reward",
            "homomorphism_transition",
        ],
        default="exact_bisim",
    )
    p.add_argument("--gamma", type=float, default=0.99, help="for bisimulation-style reductions")

    p.add_argument("--test-steps", type=int, default=5000)
    p.add_argument("--test-seed", type=int, default=123)

    p.add_argument(
        "--max-states", type=int, default=0, help="cap number of MDP states for RSA (0 = no cap)"
    )
    p.add_argument(
        "--save-rsa",
        action="store_true",
        help="save RSA matrices (.npz). Otherwise save only stats.",
    )
    p.add_argument(
        "--layers", type=str, default="", help="comma-separated layer names to hook; empty = auto"
    )
    p.add_argument(
        "--centered",
        action="store_true",
        help="use centered cosine similarity (subtract mean before computing similarity)"
    )

    args = p.parse_args()

    run_dir = _infer_run_dir(args.run_dir)
    cfg_path = os.path.join(run_dir, "config.json")
    cfg = _read_json(cfg_path)

    # Infer key settings from config.json written by train.py
    algo = str(cfg["algo"]).lower()
    env_id = cfg["env_id"]
    obs_mode = cfg.get("obs_mode", cfg.get("obs", "image"))
    seed = int(cfg.get("seed", 0)) if args.mdp_seed == -1 else int(args.mdp_seed)

    # locate checkpoints
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ckpt_glob = os.path.join(ckpt_dir, "*.pt")
    ckpts = sorted(glob.glob(ckpt_glob))
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoints found under {ckpt_dir}. Expected something like {ckpt_glob}.\n"
            f"Tip: ensure training used --checkpoint-freq > 0 or saved checkpoints into that folder."
        )

    # output path inside the same run folder
    ts = time.strftime("%d-%m-%Y-%H-%M")
    out_root = os.path.join(run_dir, "analysis", "rsa_reduction", ts)
    os.makedirs(out_root, exist_ok=True)

    # environment
    env = gym.make(env_id)
    env = make_minigrid_wrapped(env, obs_mode=obs_mode)

    # empirical MDP + representative obs per state
    mdp = collect_empirical_mdp(env, steps=args.mdp_steps, seed=seed)
    obs_map = collect_state_obs_map(env, steps=args.mdp_steps, seed=seed)

    # Choose which states to include for "all-states RSA"
    states = [s for s in list(mdp.states) if s in obs_map]
    if args.max_states and len(states) > args.max_states:
        rng = np.random.default_rng(0)
        states = list(rng.choice(states, size=args.max_states, replace=False))

    obs_all = np.stack([obs_map[s] for s in states], axis=0)

    # reduction => state -> group
    if args.reduction == "exact_bisim":
        part = exact_bisimulation_partition(mdp, gamma=args.gamma, tol=0.0)
        state_to_group = part.cls
    elif args.reduction in ("bisim_metric_eps", "bisim_metric_kmedoids"):
        # 1) compute approximate bisimulation distances (Sinkhorn OT proxy)
        dist = approximate_bisimulation_distance(
            mdp,
            gamma=args.gamma,
            reg=args.bisim_reg,
            n_iters=args.bisim_iters,
        )

        # 2) build full distance matrix over the states used for RSA
        D = dist_dict_to_matrix(states, dist)

        # 3) cluster -> labels -> mapping
        if args.reduction == "bisim_metric_eps":
            labels = cluster_eps_unionfind(D, eps=float(args.bisim_eps))
        else:
            labels = cluster_kmedoids(
                D, k=int(args.bisim_k), iters=int(args.bisim_kmedoids_iters), seed=0
            )

        state_to_group = {s: int(labels[i]) for i, s in enumerate(states)}

    elif args.reduction == "homomorphism_reward":
        state_to_group = reward_homomorphism_partition(mdp)

    elif args.reduction == "homomorphism_transition":
        state_to_group = transition_homomorphism_partition(mdp)

    else:
        raise ValueError(f"Unknown reduction: {args.reduction}")

    agg = build_state_aggregation_from_partition(state_to_group)
    agg = check_homomorphism(mdp, agg)

    hom_stats = {
        "reward_mse": agg.reward_mse,
        "transition_tv_mean": agg.trans_tv_mean,
        # "transition_tv_max": agg.trans_tv_max,
    }
    groups_all = np.array([state_to_group.get(s, -1) for s in states], dtype=np.int64)

    # test rollout
    roll_keys, roll_obs = rollout_states(env, steps=args.test_steps, seed=args.test_seed)
    groups_roll = np.array([state_to_group.get(k, -1) for k in roll_keys], dtype=np.int64)

    # save metadata including original run config
    meta = {
        "run_dir": run_dir,
        "run_config": cfg,
        "inferred": {
            "algo": algo,
            "env_id": env_id,
            "obs_mode": obs_mode,
            "seed_used_for_mdp": seed,
        },
        "analysis": {
            "mdp_steps": args.mdp_steps,
            "reduction": args.reduction,
            "gamma": args.gamma,
            "n_states_for_rsa": int(len(states)),
            "n_groups": int(len(set(g for g in groups_all.tolist() if g >= 0))),
            "test_steps": args.test_steps,
            "test_seed": args.test_seed,
            "save_rsa": bool(args.save_rsa),
            "centered": bool(args.centered),
            "homomorphism_stats": hom_stats,
        },
        "checkpoints_found": [os.path.basename(x) for x in ckpts],
    }
    with open(os.path.join(out_root, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # prepare tensors
    obs_all_t = torch.from_numpy(obs_all)
    roll_obs_t = torch.from_numpy(roll_obs)

    # layer selection
    layer_names = [s.strip() for s in args.layers.split(",") if s.strip()] if args.layers else None

    results: Dict[str, Any] = {"meta": meta, "checkpoints": {}}

    for ckpt_path in ckpts:
        ckpt_name = os.path.basename(ckpt_path)
        print(f"[RSA] checkpoint: {ckpt_name}")

        model, net, forward_fn = load_model_and_net(algo, ckpt_path, env=env)

        reps_all = compute_layer_reps(
            net, obs_all_t, layer_names=layer_names, device="cpu", forward_fn=forward_fn
        )
        reps_roll = compute_layer_reps(
            net, roll_obs_t, layer_names=reps_all.layer_names, device="cpu", forward_fn=forward_fn
        )

        ckpt_out = {"layers": {}}

        for layer in reps_all.layer_names:
            X_all = reps_all.reps[layer]
            X_roll = reps_roll.reps[layer]

            sim_all = cosine_rsa(X_all, centered=args.centered)
            sim_roll = cosine_rsa(X_roll, centered=args.centered)

            stats_all = rsa_group_pair_stats(sim_all, groups_all)
            stats_roll = rsa_group_pair_stats(sim_roll, groups_roll)

            ckpt_out["layers"][layer] = {
                "all_states": {
                    "within_mean": stats_all.within_mean,
                    "within_var": stats_all.within_var,
                    "between_mean": stats_all.between_mean,
                    "between_var": stats_all.between_var,
                    "within_pairs": stats_all.within_pairs,
                    "between_pairs": stats_all.between_pairs,
                },
                "rollout": {
                    "within_mean": stats_roll.within_mean,
                    "within_var": stats_roll.within_var,
                    "between_mean": stats_roll.between_mean,
                    "between_var": stats_roll.between_var,
                    "within_pairs": stats_roll.within_pairs,
                    "between_pairs": stats_roll.between_pairs,
                },
            }

            if args.save_rsa:
                layer_safe = layer.replace("/", "_").replace(".", "_")
                np.savez_compressed(
                    os.path.join(out_root, f"{ckpt_name}__{layer_safe}.npz"),
                    sim_all=sim_all.astype(np.float32),
                    groups_all=groups_all,
                    sim_roll=sim_roll.astype(np.float32),
                    groups_roll=groups_roll,
                )

        results["checkpoints"][ckpt_name] = ckpt_out

        try:
            model.close()
        except Exception:
            pass

    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[RSA] Done. Results in: {out_root}")


if __name__ == "__main__":
    main()
