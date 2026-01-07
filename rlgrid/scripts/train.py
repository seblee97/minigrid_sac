from __future__ import annotations
import argparse
import os
import time
import gymnasium as gym
import minigrid  # ensure MiniGrid envs register

from rlgrid.envs.make_env import EnvConfig, make_vec_env
from rlgrid.algos import PPO, PPOConfig, A2C, A2CConfig, DQN, DQNConfig, QRDQN, QRDQNConfig
from rlgrid.common.logging import LogWriter, LogConfig
from rlgrid.common.rendering import render_static_env_image

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
    # logging
    p.add_argument("--logdir", type=str, default="runs")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--log-interval", type=int, default=10, help="updates (PPO/A2C) or env-steps (DQN/QRDQN)")
    p.add_argument("--checkpoint-freq", type=int, default=50000,
               help="save model checkpoint every N env-steps (0 disables)")
    p.add_argument("--checkpoint-prefix", type=str, default="",
               help="optional prefix for checkpoint filenames")
    # rendering and video
    p.add_argument("--render-static", action="store_true", 
               help="render static image of environment at start of training")
    p.add_argument("--video-freq", type=int, default=0,
               help="record episode video every N env-steps (0 disables)")
    p.add_argument("--video-fps", type=int, default=30,
               help="FPS for recorded videos")
    
    args = p.parse_args()


    base_name = args.run_name or f"{args.algo}_{args.env_id}"
    ts_subdir = time.strftime("%d-%m-%Y-%H-%M")
    run_name = os.path.join(base_name, ts_subdir)

    # run_name = args.run_name or f"{args.algo}_{args.env_id}_{int(time.time())}"
    writer = LogWriter(
        LogConfig(log_dir=args.logdir, run_name=run_name, tensorboard=(not args.no_tensorboard)),
        config_payload={
            "algo": args.algo,
            "env_id": args.env_id,
            "policy": args.policy,
            "obs_mode": args.obs,
            "total_steps": args.total_steps,
            "seed": args.seed,
            "device": args.device,
            "n_envs": args.n_envs,
        },
    )

    # Create single env for rendering if needed
    render_env = None
    if args.render_static or args.video_freq > 0:
        render_env = gym.make(args.env_id, render_mode='rgb_array')
        from rlgrid.envs.wrappers import make_minigrid_wrapped
        render_env = make_minigrid_wrapped(render_env, obs_mode=args.obs)

    # Render static environment image at start
    if args.render_static and writer and render_env:
        static_path = writer.static_image_path("env_initial.png")
        render_static_env_image(render_env, static_path, title=f"{args.env_id} - Initial State")
        print(f"Saved initial environment image to: {static_path}")

    if args.algo in ["ppo","a2c"]:
        cfg_env = EnvConfig(env_id=args.env_id, seed=args.seed, n_envs=args.n_envs, obs_mode=args.obs)
        env = make_vec_env(cfg_env)
    else:
        env = gym.make(args.env_id)
        from rlgrid.envs.wrappers import make_minigrid_wrapped
        env = make_minigrid_wrapped(env, obs_mode=args.obs)

    if args.algo == "ppo":
        cfg = PPOConfig(seed=args.seed, device=args.device, n_envs=args.n_envs)
        model = PPO(args.policy, env, cfg, writer=writer)
    elif args.algo == "a2c":
        cfg = A2CConfig(seed=args.seed, device=args.device, n_envs=args.n_envs)
        model = A2C(args.policy, env, cfg, writer=writer)
    elif args.algo == "dqn":
        cfg = DQNConfig(seed=args.seed, device=args.device)
        model = DQN(args.policy, env, cfg, writer=writer)
    else:
        cfg = QRDQNConfig(seed=args.seed, device=args.device)
        model = QRDQN(args.policy, env, cfg, writer=writer)

    model._checkpoint_freq = int(args.checkpoint_freq)
    model._checkpoint_prefix = args.checkpoint_prefix or args.algo
    model._video_freq = int(args.video_freq)
    model._video_fps = int(args.video_fps)
    model._render_env = render_env

    model.learn(total_timesteps=args.total_steps, log_interval=args.log_interval)

    if args.save:
        model.save(args.save)
        print(f"Saved to: {args.save}")

    model.close()
    print(f"Logs in: {os.path.join(args.logdir, run_name)}")

if __name__ == "__main__":
    main()
