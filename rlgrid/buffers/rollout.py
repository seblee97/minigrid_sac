from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch

@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    logp: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor

class RolloutBuffer:
    def __init__(self, n_steps: int, n_envs: int, obs_shape: tuple[int, ...], device: torch.device):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.device = device

        self.obs = np.zeros((n_steps, n_envs) + obs_shape, dtype=np.uint8)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.logp = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(self, obs, actions, logp, rewards, dones, values):
        self.obs[self.pos] = obs
        self.actions[self.pos] = actions
        self.logp[self.pos] = logp
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones
        self.values[self.pos] = values
        self.pos += 1
        if self.pos >= self.n_steps:
            self.full = True

    def compute_returns_and_advantages(self, last_values: torch.Tensor, last_dones: torch.Tensor,
                                       gamma: float, gae_lambda: float, normalize_adv: bool = True):
        assert self.full
        advantages = np.zeros_like(self.rewards)
        last_gae = np.zeros((self.n_envs,), dtype=np.float32)

        for t in reversed(range(self.n_steps)):
            next_nonterminal = 1.0 - (last_dones.cpu().numpy().astype(np.float32) if t == self.n_steps - 1 else self.dones[t + 1])
            next_values = last_values.cpu().numpy() if t == self.n_steps - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_values * next_nonterminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values
        adv = torch.from_numpy(advantages).to(self.device)
        ret = torch.from_numpy(returns).to(self.device)

        if normalize_adv:
            # More stable advantage normalization with epsilon and per-batch
            adv_mean = adv.mean()
            adv_std = adv.std()
            # Prevent division by very small numbers
            if adv_std > 1e-6:
                adv = (adv - adv_mean) / (adv_std + 1e-8)
            else:
                adv = adv - adv_mean

        return adv, ret

    def get(self) -> RolloutBatch:
        assert self.full
        obs = torch.from_numpy(self.obs.reshape((-1,) + self.obs.shape[2:])).to(self.device)
        actions = torch.from_numpy(self.actions.reshape(-1)).to(self.device)
        logp = torch.from_numpy(self.logp.reshape(-1)).to(self.device)
        values = torch.from_numpy(self.values.reshape(-1)).to(self.device)
        return obs, actions, logp, values
