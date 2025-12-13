from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import os
import json

import gymnasium as gym
import numpy as np
import torch

from rlgrid.common.logger import Logger
from rlgrid.common.torch_utils import set_seed

@dataclass
class AlgoConfig:
    seed: int = 0
    device: str = "cpu"
    gamma: float = 0.99
    lr: float = 3e-4
    verbose: int = 1

class BaseAlgorithm:
    def __init__(self, env: gym.Env, cfg: AlgoConfig):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.logger = Logger()
        set_seed(cfg.seed)

    def learn(self, total_timesteps: int, log_interval: int = 10) -> "BaseAlgorithm":
        raise NotImplementedError

    def predict(self, obs, deterministic: bool = True):
        raise NotImplementedError

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = self._get_state()
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, env: gym.Env, device: str = "cpu") -> "BaseAlgorithm":
        payload = torch.load(path, map_location=device)
        obj = cls._from_state(payload, env, device=device)
        return obj

    def _get_state(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def _from_state(cls, state: Dict[str, Any], env: gym.Env, device: str):
        raise NotImplementedError
