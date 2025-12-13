from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import gymnasium as gym
import minigrid
import numpy as np
from gymnasium.vector import SyncVectorEnv

from rlgrid.envs.wrappers import MiniGridObs, make_minigrid_wrapped

ObsMode = Literal["image", "image_dir", "rgb"]

@dataclass
class EnvConfig:
    env_id: str
    seed: int = 0
    n_envs: int = 1
    obs_mode: ObsMode = "image"
    max_episode_steps: Optional[int] = None  # if you want TimeLimit override

def make_single_env(cfg: EnvConfig, idx: int = 0) -> Callable[[], gym.Env]:
    def _thunk() -> gym.Env:
        env = gym.make(cfg.env_id)
        if cfg.max_episode_steps is not None:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=cfg.max_episode_steps)
        env = make_minigrid_wrapped(env, obs_mode=cfg.obs_mode)
        env.reset(seed=cfg.seed + idx)
        return env
    return _thunk

def make_vec_env(cfg: EnvConfig) -> SyncVectorEnv:
    return SyncVectorEnv([make_single_env(cfg, i) for i in range(cfg.n_envs)])

def obs_shape(env: gym.Env) -> tuple[int, ...]:
    space = env.observation_space
    if hasattr(space, "shape") and space.shape is not None:
        return tuple(space.shape)
    raise ValueError(f"Unsupported observation space for obs_shape: {space}")

def act_dim(env: gym.Env) -> int:
    space = env.action_space
    if isinstance(space, gym.spaces.Discrete):
        return int(space.n)
    raise ValueError("This codebase currently supports Discrete action spaces (MiniGrid default).")
