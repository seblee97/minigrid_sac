from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlgrid.policies.base import MLP, SimpleCNN, preprocess_obs

NetType = Literal["MlpPolicy", "CnnPolicy"]

class QNetwork(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...], n_actions: int, net_type: NetType):
        super().__init__()
        self.net_type = net_type
        self.obs_shape = obs_shape
        self.n_actions = n_actions

        if net_type == "MlpPolicy":
            in_dim = int(np.prod(obs_shape))
            self.backbone = MLP(in_dim, hidden=(256, 256))
            feat_dim = 256
        elif net_type == "CnnPolicy":
            c = obs_shape[-1]
            self.backbone = SimpleCNN(c)
            feat_dim = None
        else:
            raise ValueError(f"Unknown net_type: {net_type}")

        self._feat_dim = feat_dim
        self.head = None
        self._init_head()

    def _infer_feat_dim(self) -> int:
        if self._feat_dim is not None:
            return self._feat_dim
        with torch.no_grad():
            dummy = torch.zeros((1,) + self.obs_shape, dtype=torch.float32)
            y = self.backbone(dummy)
            self._feat_dim = y.shape[-1]
        return self._feat_dim

    def _init_head(self):
        feat_dim = self._infer_feat_dim()
        self.head = nn.Linear(feat_dim, self.n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = preprocess_obs(obs)
        if self.net_type == "MlpPolicy":
            obs = obs.flatten(1)
        feat = self.backbone(obs)
        return self.head(feat)

class QuantileQNetwork(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...], n_actions: int, net_type: NetType, n_quantiles: int = 50):
        super().__init__()
        self.net_type = net_type
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles

        if net_type == "MlpPolicy":
            in_dim = int(np.prod(obs_shape))
            self.backbone = MLP(in_dim, hidden=(256, 256))
            feat_dim = 256
        elif net_type == "CnnPolicy":
            c = obs_shape[-1]
            self.backbone = SimpleCNN(c)
            feat_dim = None
        else:
            raise ValueError(f"Unknown net_type: {net_type}")

        self._feat_dim = feat_dim
        self.head = None
        self._init_head()

    def _infer_feat_dim(self) -> int:
        if self._feat_dim is not None:
            return self._feat_dim
        with torch.no_grad():
            dummy = torch.zeros((1,) + self.obs_shape, dtype=torch.float32)
            y = self.backbone(dummy)
            self._feat_dim = y.shape[-1]
        return self._feat_dim

    def _init_head(self):
        feat_dim = self._infer_feat_dim()
        self.head = nn.Linear(feat_dim, self.n_actions * self.n_quantiles)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = preprocess_obs(obs)
        if self.net_type == "MlpPolicy":
            obs = obs.flatten(1)
        feat = self.backbone(obs)
        q = self.head(feat)  # (B, A*N)
        q = q.view(q.shape[0], self.n_actions, self.n_quantiles)  # (B,A,N)
        return q
