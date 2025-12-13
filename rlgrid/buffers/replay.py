from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.obs = np.zeros((capacity,) + obs_shape, dtype=np.uint8)
        self.next_obs = np.zeros((capacity,) + obs_shape, dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(self, obs, action, reward, done, next_obs):
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def sample(self, batch_size: int):
        n = len(self)
        idx = np.random.randint(0, n, size=batch_size)
        obs = torch.from_numpy(self.obs[idx]).to(self.device)
        next_obs = torch.from_numpy(self.next_obs[idx]).to(self.device)
        actions = torch.from_numpy(self.actions[idx]).to(self.device)
        rewards = torch.from_numpy(self.rewards[idx]).to(self.device)
        dones = torch.from_numpy(self.dones[idx]).to(self.device)
        return obs, actions, rewards, dones, next_obs
