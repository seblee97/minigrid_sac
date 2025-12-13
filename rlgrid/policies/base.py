from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlgrid.policies.distributions import CategoricalDist

PolicyType = Literal["MlpPolicy", "CnnPolicy"]

def preprocess_obs(obs: torch.Tensor) -> torch.Tensor:
    # Expect uint8 images from MiniGrid wrappers; scale to [0,1]
    if obs.dtype == torch.uint8:
        return obs.float() / 255.0
    return obs.float()

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Tuple[int, int] = (256, 256)):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # small, fast CNN for MiniGrid-style small images
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.out_dim = 64 * 2 * 2  # will be inferred if needed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,H,W,C) -> (B,C,H,W)
        x = x.permute(0, 3, 1, 2).contiguous()
        y = self.conv(x)
        return y.flatten(1)

class ActorCritic(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...], n_actions: int, policy_type: PolicyType):
        super().__init__()
        self.policy_type = policy_type
        self.obs_shape = obs_shape
        self.n_actions = n_actions

        if policy_type == "MlpPolicy":
            in_dim = int(np.prod(obs_shape))
            self.backbone = MLP(in_dim)
            feat_dim = 256
        elif policy_type == "CnnPolicy":
            c = obs_shape[-1]
            self.backbone = SimpleCNN(c)
            # infer feat dim lazily
            feat_dim = None
        else:
            raise ValueError(f"Unknown policy_type: {policy_type}")

        self._feat_dim = feat_dim
        self.actor = None
        self.critic = None
        self._init_heads()

    def _infer_feat_dim(self) -> int:
        if self._feat_dim is not None:
            return self._feat_dim
        # infer by dummy forward
        with torch.no_grad():
            dummy = torch.zeros((1,) + self.obs_shape, dtype=torch.float32)
            y = self.backbone(dummy)
            self._feat_dim = y.shape[-1]
        return self._feat_dim

    def _init_heads(self):
        feat_dim = self._infer_feat_dim()
        self.actor = nn.Linear(feat_dim, self.n_actions)
        self.critic = nn.Linear(feat_dim, 1)

    def forward(self, obs: torch.Tensor):
        obs = preprocess_obs(obs)
        if self.policy_type == "MlpPolicy":
            obs = obs.flatten(1)
        feat = self.backbone(obs)
        logits = self.actor(feat)
        value = self.critic(feat).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        logits, value = self.forward(obs)
        dist = CategoricalDist(logits=logits)
        if deterministic:
            actions = dist.mode()
        else:
            actions = dist.sample()
        logp = dist.log_prob(actions)
        return actions, logp, value
