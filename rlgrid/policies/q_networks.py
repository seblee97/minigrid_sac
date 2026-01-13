from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlgrid.policies.base import (
    MLP, SimpleCNN, 
    preprocess_obs, effective_obs_shape, 
    DEFAULT_N_OBJ, DEFAULT_N_COL, DEFAULT_N_STATE
)


NetType = Literal["MlpPolicy", "CnnPolicy"]
ObsMode = Literal["image", "image_dir", "rgb"]

class QNetwork(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        net_type: NetType,
        obs_mode: ObsMode = "image",
        n_obj: int = DEFAULT_N_OBJ,
        n_col: int = DEFAULT_N_COL,
        n_state: int = DEFAULT_N_STATE,
    ):
        super().__init__()
        self.net_type = net_type
        self.obs_shape_raw = obs_shape
        self.obs_mode = obs_mode
        self.n_actions = n_actions
        self.n_obj, self.n_col, self.n_state = n_obj, n_col, n_state

        # network input shape after preprocessing (e.g. one-hot depth)
        self.obs_shape = effective_obs_shape(obs_shape, obs_mode, n_obj, n_col, n_state)

        if net_type == "MlpPolicy":
            in_dim = int(np.prod(self.obs_shape))
            self.backbone = MLP(in_dim, hidden=(256, 256))
            feat_dim = 256
        elif net_type == "CnnPolicy":
            c = self.obs_shape[-1]
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
        # Q-network head: backbone -> FC layer -> output layer
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_actions)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = preprocess_obs(obs, obs_mode=self.obs_mode, n_obj=self.n_obj, n_col=self.n_col, n_state=self.n_state)
        if self.net_type == "MlpPolicy":
            x = x.flatten(1)
        feat = self.backbone(x)
        return self.head(feat)

class QuantileQNetwork(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        net_type: NetType,
        n_quantiles: int = 50,
        obs_mode: ObsMode = "image",
        n_obj: int = DEFAULT_N_OBJ,
        n_col: int = DEFAULT_N_COL,
        n_state: int = DEFAULT_N_STATE,
    ):
        super().__init__()
        self.net_type = net_type
        self.obs_shape_raw = obs_shape
        self.obs_mode = obs_mode
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        self.n_obj, self.n_col, self.n_state = n_obj, n_col, n_state

        self.obs_shape = effective_obs_shape(obs_shape, obs_mode, n_obj, n_col, n_state)

        if net_type == "MlpPolicy":
            in_dim = int(np.prod(self.obs_shape))
            self.backbone = MLP(in_dim, hidden=(256, 256))
            feat_dim = 256
        elif net_type == "CnnPolicy":
            c = self.obs_shape[-1]
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
        # Quantile Q-network head: backbone -> FC layer -> output layer
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_actions * self.n_quantiles)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = preprocess_obs(obs, obs_mode=self.obs_mode, n_obj=self.n_obj, n_col=self.n_col, n_state=self.n_state)
        if self.net_type == "MlpPolicy":
            x = x.flatten(1)
        feat = self.backbone(x)
        q = self.head(feat)  # (B, A*N)
        return q.view(q.shape[0], self.n_actions, self.n_quantiles)
