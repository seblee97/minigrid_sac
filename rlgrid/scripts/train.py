from __future__ import annotations
import argparse
import gymnasium as gym
import numpy as np
import minigrid

from rlgrid.envs.make_env import EnvConfig, make_vec_env, make_single_env
from rlgrid.algos import PPO, PPOConfig, A2C, A2CConfig, DQN, DQNConfig, QRDQN, QRDQNConfig

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo","a2c","dqn","qrdqn"], required=True)
    p.add_argument("--env", dest="env_id", required=True)
    p.add_argument("--policy", choices=["CnnPolicy","MlpPolicy"], default="CnnPolicy")
    p.add_argument("--obs", choices=["image","image_dir","rgb"], default="image")
    p.add_argument("--total-steps", type=int, default=200000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--save", type=str, default="")
    args = p.parse_args()

    if args.algo in ["ppo","a2c"]:
        cfg_env = EnvConfig(env_id=args.env_id, seed=args.seed, n_envs=args.n_envs, obs_mode=args.obs)
        env = make_vec_env(cfg_env)
    else:
        env = gym.make(args.env_id)
        from rlgrid.envs.wrappers import make_minigrid_wrapped
        env = make_minigrid_wrapped(env, obs_mode=args.obs)

    if args.algo == "ppo":
        cfg = PPOConfig(seed=args.seed, device=args.device, n_envs=args.n_envs)
        model = PPO(args.policy, env, cfg)
    elif args.algo == "a2c":
        cfg = A2CConfig(seed=args.seed, device=args.device, n_envs=args.n_envs)
        model = A2C(args.policy, env, cfg)
    elif args.algo == "dqn":
        cfg = DQNConfig(seed=args.seed, device=args.device)
        model = DQN(args.policy, env, cfg)
    else:
        cfg = QRDQNConfig(seed=args.seed, device=args.device)
        model = QRDQN(args.policy, env, cfg)

    model.learn(total_timesteps=args.total_steps)

    if args.save:
        model.save(args.save)
        print(f"Saved to: {args.save}")

if __name__ == "__main__":
    main()
