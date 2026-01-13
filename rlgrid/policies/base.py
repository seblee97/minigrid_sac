from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlgrid.policies.distributions import CategoricalDist

PolicyType = Literal["MlpPolicy", "CnnPolicy"]
ObsMode = Literal["image", "image_dir", "rgb"]

# MiniGrid defaults (stable across most releases)
DEFAULT_N_OBJ = 11     # unseen, empty, wall, floor, door, key, ball, box, goal, lava, agent
DEFAULT_N_COL = 6      # red, green, blue, purple, yellow, grey
DEFAULT_N_STATE = 3    # door states etc.

# def preprocess_obs(obs: torch.Tensor) -> torch.Tensor:
#     # Expect uint8 images from MiniGrid wrappers; scale to [0,1]
#     if obs.dtype == torch.uint8:
#         return obs.float() / 255.0
#     return obs.float()

def preprocess_obs(
    obs: torch.Tensor,
    obs_mode: ObsMode,
    n_obj: int = DEFAULT_N_OBJ,
    n_col: int = DEFAULT_N_COL,
    n_state: int = DEFAULT_N_STATE,
) -> torch.Tensor:
    """
    Mode-consistent preprocessing:

    - rgb: uint8 pixels -> float in [0,1]
    - image / image_dir: uint8 categorical codes -> one-hot features (float)
    """
    if obs_mode == "rgb":
        # true pixels
        if obs.dtype == torch.uint8:
            return obs.float() / 255.0
        return obs.float()

    # symbolic MiniGrid encoding: (obj, color, state) (+ optional dir)
    # Expected shape: (B,H,W,C)
    if obs.dtype != torch.long:
        x = obs.long()
    else:
        x = obs

    obj = x[..., 0].clamp(0, n_obj - 1)
    col = x[..., 1].clamp(0, n_col - 1)
    st  = x[..., 2].clamp(0, n_state - 1)

    obj_oh = F.one_hot(obj, num_classes=n_obj).float()
    col_oh = F.one_hot(col, num_classes=n_col).float()
    st_oh  = F.one_hot(st,  num_classes=n_state).float()

    feats = torch.cat([obj_oh, col_oh, st_oh], dim=-1)  # (B,H,W,n_obj+n_col+n_state)

    if obs_mode == "image_dir":
        # direction channel appended at the end by your wrapper; values 0..3
        d = x[..., 3].clamp(0, 3)
        d_oh = F.one_hot(d, num_classes=4).float()
        feats = torch.cat([feats, d_oh], dim=-1)

    return feats


def effective_obs_shape(
    raw_obs_shape: tuple[int, ...],
    obs_mode: ObsMode,
    n_obj: int = DEFAULT_N_OBJ,
    n_col: int = DEFAULT_N_COL,
    n_state: int = DEFAULT_N_STATE,
) -> tuple[int, ...]:
    """
    What shape the *network* actually receives after preprocess_obs.
    """
    h, w = raw_obs_shape[0], raw_obs_shape[1]
    if obs_mode == "rgb":
        return (h, w, 3)

    base_c = n_obj + n_col + n_state
    if obs_mode == "image_dir":
        base_c += 4
    return (h, w, base_c)


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
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        policy_type: PolicyType,
        obs_mode: ObsMode = "image",
        n_obj: int = DEFAULT_N_OBJ,
        n_col: int = DEFAULT_N_COL,
        n_state: int = DEFAULT_N_STATE,
    ):
        super().__init__()
        self.policy_type = policy_type
        self.obs_shape_raw = obs_shape
        self.obs_mode = obs_mode
        self.n_actions = n_actions
        self.n_obj, self.n_col, self.n_state = n_obj, n_col, n_state

        # IMPORTANT: network sees the post-processed shape
        self.obs_shape = effective_obs_shape(obs_shape, obs_mode, n_obj, n_col, n_state)

        if policy_type == "MlpPolicy":
            in_dim = int(np.prod(self.obs_shape))
            self.backbone = MLP(in_dim)
            feat_dim = 256
        elif policy_type == "CnnPolicy":
            c = self.obs_shape[-1]
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
        # Actor head: backbone -> FC layer -> output layer
        self.actor = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_actions)
        )
        # Critic head: backbone -> FC layer -> output layer
        self.critic = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs: torch.Tensor):
        obs = preprocess_obs(
            obs,
            obs_mode=self.obs_mode,
            n_obj=self.n_obj,
            n_col=self.n_col,
            n_state=self.n_state,
        )
        if self.policy_type == "MlpPolicy":
            obs = obs.flatten(1)
        feat = self.backbone(obs)
        logits = self.actor(feat)
        value = self.critic(feat).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        logits, value = self.forward(obs)
        dist = CategoricalDist(logits=logits)
        actions = dist.mode() if deterministic else dist.sample()
        logp = dist.log_prob(actions)
        return actions, logp, value
