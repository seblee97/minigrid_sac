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
from rlgrid.envs.wrappers import make_minigrid_wrapped

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
    p.add_argument("--eval-freq", type=int, default=0,
                help="run one evaluation episode every N env-steps (0 disables); logs eval/* metrics")
    p.add_argument("--eval-video-freq", type=int, default=0,
               help="record an evaluation episode video every N env-steps (0 disables). If 0, --video-freq is used as a legacy alias.")
    p.add_argument("--eval-max-steps", type=int, default=2048,
               help="max steps per evaluation episode")
    p.add_argument("--eval-stochastic", action="store_true",
               help="use stochastic actions during evaluation (default: deterministic)")
    # observability settings
    p.add_argument("--full-obs", action="store_true",
               help="use full observability instead of partial (makes learning easier but less robust)")
    
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

    # Create single envs for evaluation and/or rendering if needed
    eval_env = None
    render_env = None

    # Backward compatible: --video-freq acts as a legacy alias for eval-video-freq
    eval_freq = int(args.eval_freq)
    eval_video_freq = int(args.eval_video_freq) if int(args.eval_video_freq) > 0 else int(args.video_freq)

    if eval_freq > 0 or eval_video_freq > 0:
        eval_env = gym.make(args.env_id)
        eval_env = make_minigrid_wrapped(eval_env, obs_mode=args.obs, full_obs=args.full_obs)

    if args.render_static or eval_video_freq > 0:
        render_env = gym.make(args.env_id, render_mode='rgb_array')
        render_env = make_minigrid_wrapped(render_env, obs_mode=args.obs, full_obs=args.full_obs)

    # Render static environment image at start
    if args.render_static and writer and render_env:
        static_path = writer.static_image_path("env_initial.png")
        render_static_env_image(render_env, static_path, title=f"{args.env_id} - Initial State")
        print(f"Saved initial environment image to: {static_path}")

    if args.algo in ["ppo","a2c"]:
        cfg_env = EnvConfig(env_id=args.env_id, seed=args.seed, n_envs=args.n_envs, obs_mode=args.obs, full_obs=args.full_obs)
        env = make_vec_env(cfg_env)
    else:
        env = gym.make(args.env_id)
        env = make_minigrid_wrapped(env, obs_mode=args.obs, full_obs=args.full_obs)

    if args.algo == "ppo":
        cfg = PPOConfig(seed=args.seed, device=args.device, n_envs=args.n_envs)
        model = PPO(
            args.policy,
            env,
            cfg,
            policy_kwargs={"obs_mode": args.obs},
            writer=writer,
        )
    elif args.algo == "a2c":
        cfg = A2CConfig(seed=args.seed, device=args.device, n_envs=args.n_envs)
        model = A2C(
            args.policy,
            env,
            cfg,
            policy_kwargs={"obs_mode": args.obs},
            writer=writer,
        )
    elif args.algo == "dqn":
        cfg = DQNConfig(seed=args.seed, device=args.device)
        model = DQN(args.policy, env, cfg, policy_kwargs={"obs_mode": args.obs}, writer=writer)
    elif args.algo == "qrdqn":
        cfg = QRDQNConfig(seed=args.seed, device=args.device)
        model = QRDQN(args.policy, env, cfg, policy_kwargs={"obs_mode": args.obs}, writer=writer)

    model._checkpoint_freq = int(args.checkpoint_freq)
    model._checkpoint_prefix = args.checkpoint_prefix or args.algo
    
    # Evaluation scheduling (metrics) and evaluation video recording
    model._eval_env = eval_env
    model._eval_freq = int(eval_freq)
    model._eval_video_freq = int(eval_video_freq)
    model._eval_max_steps = int(args.eval_max_steps)
    model._eval_deterministic = (not args.eval_stochastic)
    # Rendering/video settings
    model._video_fps = int(args.video_fps)
    model._render_env = render_env

    # Legacy alias retained for older code paths (should be unused by the algos after this patch)
    model._video_freq = int(eval_video_freq)

    model.learn(total_timesteps=args.total_steps, log_interval=args.log_interval)

    if args.save:
        model.save(args.save)
        print(f"Saved to: {args.save}")

    model.close()
    print(f"Logs in: {os.path.join(args.logdir, run_name)}")

if __name__ == "__main__":
    main()
