from __future__ import annotations

from typing import Literal, Tuple
import gymnasium as gym
import numpy as np
from minigrid.wrappers import FullyObsWrapper

ObsMode = Literal["image", "image_dir", "rgb"]

class MiniGridObs(gym.ObservationWrapper):
    '''
    MiniGrid observations are often dicts with 'image' and sometimes 'mission'.
    This wrapper produces a pure numeric observation suitable for neural nets.

    Modes:
    - 'image': returns the default MiniGrid image observation (H,W,C) uint8
    - 'image_dir': concatenates agent direction as a 1x1 channel (H,W,C+1)
    - 'rgb': uses RGB image if available via MiniGrid wrappers
    '''
    def __init__(self, env: gym.Env, mode: ObsMode = "image"):
        super().__init__(env)
        self.mode = mode

        # Determine base image space
        base_space = env.observation_space
        if isinstance(base_space, gym.spaces.Dict):
            img_space = base_space["image"]
        else:
            img_space = base_space

        assert isinstance(img_space, gym.spaces.Box)
        assert img_space.dtype == np.uint8, "MiniGrid image obs are typically uint8."

        h, w, c = img_space.shape
        if mode == "image":
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(h, w, c), dtype=np.uint8)
        elif mode == "image_dir":
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(h, w, c + 1), dtype=np.uint8)
        elif mode == "rgb":
            # Assumes upstream wrapper provides RGB image in obs (H,W,3)
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)
        else:
            raise ValueError(f"Unknown obs mode: {mode}")

    def observation(self, obs):
        if isinstance(obs, dict):
            img = obs["image"]
            direction = obs.get("direction", None)
            rgb = obs.get("rgb", None)
        else:
            img = obs
            direction = None
            rgb = None

        if self.mode == "image":
            return img
        if self.mode == "image_dir":
            if direction is None:
                # Try unwrapped attribute (MiniGrid usually has agent_dir)
                direction = getattr(self.unwrapped, "agent_dir", 0)
            dir_chan = np.full(img.shape[:2] + (1,), int(direction) % 4, dtype=np.uint8)
            return np.concatenate([img, dir_chan], axis=-1)
        if self.mode == "rgb":
            if rgb is not None:
                return rgb
            # fallback: convert object encoding to fake RGB via simple lookup
            # (kept intentionally simple; prefer MiniGrid's RGB wrappers if needed)
            return img[..., :3]
        raise RuntimeError("unreachable")

def make_minigrid_wrapped(env: gym.Env, obs_mode: ObsMode = "image", full_obs: bool = False) -> gym.Env:
    # Common, lightweight wrappers:
    # - RecordEpisodeStatistics for returns/length
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # # Only use full observability if explicitly requested
    if full_obs:
        env = FullyObsWrapper(env)

    # Many MiniGrid envs yield dict obs; adapt it.
    env = MiniGridObs(env, mode=obs_mode)

    # Normalize to float for networks by default in the policy (not here).
    return env
