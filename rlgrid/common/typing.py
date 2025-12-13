from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Protocol

import numpy as np
import torch

Obs = Any
Act = Any

@dataclass
class StepBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    next_obs: torch.Tensor
    infos: Optional[Any] = None

class Policy(Protocol):
    def forward(self, obs: torch.Tensor) -> Any: ...
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
