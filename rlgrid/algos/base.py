from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import os
import gymnasium as gym
import torch

from rlgrid.common.torch_utils import set_seed
from rlgrid.common.logger import Logger
from rlgrid.common.logging import LogWriter
from rlgrid.common.rendering import EpisodeVideoRecorder

@dataclass
class AlgoConfig:
    seed: int = 0
    device: str = "cpu"
    gamma: float = 0.99
    lr: float = 3e-5
    verbose: int = 1

class BaseAlgorithm:
    def __init__(self, env: gym.Env, cfg: AlgoConfig, writer: Optional[LogWriter] = None):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.logger = Logger()
        self.writer = writer
        set_seed(cfg.seed)

        # Video recording attributes
        self._video_freq = 0
        self._video_fps = 30
        self._render_env = None
        self._last_video_step = 0

        self._eval_freq = 0                 # env-steps between eval episodes (0 disables)
        self._eval_video_freq = 0           # env-steps between eval videos (0 disables)
        self._eval_max_steps = 2048
        self._eval_deterministic = True
        self._eval_env = None               # single env for metric-only eval
        self._last_eval_step = 0
        self._last_eval_video_step = 0

    def should_eval(self, step: int) -> bool:
        """Check if we should run an evaluation episode (metrics only) at this step."""
        if self._eval_freq <= 0 or self._eval_env is None or self.writer is None:
            return False
        return step - self._last_eval_step >= self._eval_freq
    
    def should_record_eval_video(self, step: int) -> bool:
        """Check if we should record an evaluation episode video at this step."""
        if self._eval_video_freq <= 0 or self._render_env is None or self.writer is None:
            return False
        return step - self._last_eval_video_step >= self._eval_video_freq

    def _run_eval_episode(self, step: int, record_video: bool) -> None:
        """Run a single evaluation episode and log metrics. Optionally record video."""
        if self.writer is None:
            return
        env = self._render_env if record_video else self._eval_env
        if env is None:
            return
        # Seed eval episodes for reproducibility but without perturbing training env RNG.
        seed = int(self.cfg.seed) + int(step)
        obs, _ = env.reset(seed=seed)
        recorder = None
        if record_video:
            recorder = EpisodeVideoRecorder(env, self.writer.video_dir(), fps=self._video_fps)
            recorder.start_recording()
            try:
                recorder.capture_frame()
            except Exception:
                pass
        
        ep_ret = 0.0
        ep_len = 0
        done = False

        while not done and ep_len < int(self._eval_max_steps):
            action = self.predict(obs, deterministic=self._eval_deterministic)
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += float(r)
            ep_len += 1
            done = bool(terminated or truncated)
           
            if recorder is not None:
                try:
                    recorder.capture_frame()
                except Exception:
                    pass

        # Log scalars
        self.writer.dump(step, {
            "eval/ep_return": ep_ret,
            "eval/ep_length": ep_len,
        })

        # Save video if requested
        if recorder is not None and recorder.recording:
            filename = f"eval_step_{step}_r_{ep_ret:.2f}_l_{ep_len}"
            recorder.stop_recording(filename)

    def maybe_eval(self, step: int) -> None:
        """Run evaluation and/or eval-video if their respective schedules are due."""
        do_video = self.should_record_eval_video(step)
        do_eval = self.should_eval(step) or do_video  # if video is due, also do eval/logging

        if not (do_eval or do_video):
            return

        self._run_eval_episode(step, record_video=do_video)

        if do_eval:
            self._last_eval_step = step
        if do_video:
            self._last_eval_video_step = step

    def learn(self, total_timesteps: int, log_interval: int = 10) -> "BaseAlgorithm":
        raise NotImplementedError

    def predict(self, obs, deterministic: bool = True):
        raise NotImplementedError

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self._get_state(), path)

    @classmethod
    def load(cls, path: str, env: gym.Env, device: str = "cpu") -> "BaseAlgorithm":
        payload = torch.load(path, map_location=device)
        return cls._from_state(payload, env, device=device)
    
    def maybe_checkpoint(self, step: int, checkpoint_freq: int, prefix: str = "ckpt") -> None:
        """
        Save periodic checkpoints based on *threshold crossing* (robust to large step jumps).
        checkpoint_freq is in environment steps. If <= 0, checkpointing is disabled.
        """
        if checkpoint_freq is None or checkpoint_freq <= 0:
            return
        if self.writer is None:
            return

        # Initialize next checkpoint threshold on first use
        if not hasattr(self, "_next_ckpt_step"):
            self._next_ckpt_step = checkpoint_freq

        # Save checkpoints for all crossed thresholds
        while step >= self._next_ckpt_step:
            path = self.writer.checkpoint_path(self._next_ckpt_step, prefix=prefix)
            self.save(path)
            self._next_ckpt_step += checkpoint_freq

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()

    def _get_state(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def _from_state(cls, state: Dict[str, Any], env: gym.Env, device: str):
        raise NotImplementedError
