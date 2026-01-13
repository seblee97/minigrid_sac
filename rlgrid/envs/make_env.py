from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import gymnasium as gym
import minigrid  # ensure MiniGrid envs register
import numpy as np
from gymnasium.vector import SyncVectorEnv

from rlgrid.envs.wrappers import MiniGridObs, make_minigrid_wrapped, KeyDoorGymnasiumWrapper

ObsMode = Literal["image", "image_dir", "rgb"]

@dataclass
class EnvConfig:
    env_id: str
    seed: int = 0
    n_envs: int = 1
    obs_mode: ObsMode = "image"
    max_episode_steps: Optional[int] = None  # if you want TimeLimit override
    full_obs: bool = False
    # key_door specific parameters
    key_door_map_ascii: Optional[str] = None
    key_door_map_yaml: Optional[str] = None
    key_door_map_yaml_test: Optional[str] = None  # Optional separate YAML for testing
    key_door_representation: str = "agent_position"  # or "pixel" or "po_pixel"

def _is_key_door_env(env_id: str) -> bool:
    """Check if the env_id refers to a key_door environment."""
    return env_id.startswith("KeyDoor") or "key_door" in env_id.lower()

def make_single_env(cfg: EnvConfig, idx: int = 0) -> Callable[[], gym.Env]:
    def _thunk() -> gym.Env:
        if _is_key_door_env(cfg.env_id):
            # Create key_door environment
            from key_door.key_door_env import KeyDoorEnv
            
            # Require map paths for key_door environments
            if cfg.key_door_map_ascii is None or cfg.key_door_map_yaml is None:
                raise ValueError(
                    f"key_door environment '{cfg.env_id}' requires "
                    "key_door_map_ascii and key_door_map_yaml parameters"
                )
            
            # Create the key_door environment
            key_door_env = KeyDoorEnv(
                map_ascii_path=cfg.key_door_map_ascii,
                map_yaml_path=cfg.key_door_map_yaml,
                representation=cfg.key_door_representation,
                episode_timeout=cfg.max_episode_steps,
            )
            
            # Wrap it to be Gymnasium-compatible, optionally with separate test YAML
            env = KeyDoorGymnasiumWrapper(
                key_door_env, 
                test_yaml_path=cfg.key_door_map_yaml_test
            )
        else:
            # Create MiniGrid environment
            env = gym.make(cfg.env_id)
            if cfg.max_episode_steps is not None:
                env = gym.wrappers.TimeLimit(env, max_episode_steps=cfg.max_episode_steps)
            env = make_minigrid_wrapped(env, obs_mode=cfg.obs_mode, full_obs=cfg.full_obs)
        
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
