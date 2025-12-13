from __future__ import annotations
import argparse
import gymnasium as gym
import numpy as np

from rlgrid.envs.wrappers import make_minigrid_wrapped
from rlgrid.algos import PPO, A2C, DQN, QRDQN

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo","a2c","dqn","qrdqn"], required=True)
    p.add_argument("--env", dest="env_id", required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--obs", choices=["image","image_dir","rgb"], default="image")
    args = p.parse_args()

    env = make_minigrid_wrapped(gym.make(args.env_id), obs_mode=args.obs)

    algo_map = {"ppo": PPO, "a2c": A2C, "dqn": DQN, "qrdqn": QRDQN}
    cls = algo_map[args.algo]
    model = cls.load(args.model, env=env, device=args.device)

    rets = []
    for _ in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            a = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(int(a))
            done = bool(term or trunc)
            ep_ret += float(r)
        rets.append(ep_ret)
    print(f"Mean return over {args.episodes} eps: {np.mean(rets):.3f} +/- {np.std(rets):.3f}")

if __name__ == "__main__":
    main()
