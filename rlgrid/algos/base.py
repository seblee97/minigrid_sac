from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import os
import gymnasium as gym
import torch

from rlgrid.common.torch_utils import set_seed
from rlgrid.common.logger import Logger
from rlgrid.common.logging import LogWriter

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
