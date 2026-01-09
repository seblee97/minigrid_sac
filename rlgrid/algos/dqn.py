from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import copy

import gymnasium as gym
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import trange

from rlgrid.algos.base import AlgoConfig, BaseAlgorithm
from rlgrid.buffers.replay import ReplayBuffer
from rlgrid.common.torch_utils import to_tensor
from rlgrid.policies.q_networks import QNetwork, NetType

@dataclass
class DQNConfig(AlgoConfig):
    buffer_size: int = 100000
    learning_starts: int = 5000
    batch_size: int = 256
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 1000
    exploration_fraction: float = 0.2
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    max_grad_norm: float = 10.0

class DQN(BaseAlgorithm):
    def __init__(self, policy: NetType, env: gym.Env, cfg: DQNConfig, policy_kwargs: Optional[dict] = None, writer=None):
        super().__init__(env, cfg, writer=writer)
        self.policy_type = policy
        self.policy_kwargs = policy_kwargs or {}
        obs_shape = env.observation_space.shape
        n_actions = env.action_space.n
        
        obs_mode = self.policy_kwargs.get("obs_mode", "image")
        n_obj = self.policy_kwargs.get("n_obj", 11)
        n_col = self.policy_kwargs.get("n_col", 6)
        n_state = self.policy_kwargs.get("n_state", 3)

        self.q = QNetwork(obs_shape, n_actions, policy, obs_mode=obs_mode, n_obj=n_obj, n_col=n_col, n_state=n_state).to(self.device)

        self.q_targ = copy.deepcopy(self.q).to(self.device)
        self.optim = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.rb = ReplayBuffer(cfg.buffer_size, obs_shape, self.device)
        self.total_steps = 0
        self.eps = cfg.exploration_initial_eps

    def _update_eps(self):
        cfg: DQNConfig = self.cfg  # type: ignore
        frac = min(1.0, self.total_steps / max(1, int(cfg.exploration_fraction * self.total_steps_target)))
        # We'll set total_steps_target at start of learn()
        self.eps = cfg.exploration_initial_eps + frac * (cfg.exploration_final_eps - cfg.exploration_initial_eps)

    def predict(self, obs, deterministic: bool = True):
        obs_t = to_tensor(obs, self.device)
        if obs_t.ndim == 3:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            q = self.q(obs_t)
            a = torch.argmax(q, dim=-1)
        return a.squeeze(0).cpu().numpy()


    def learn(self, total_timesteps: int, log_interval: int = 200):
        cfg: DQNConfig = self.cfg  # type: ignore
        env = self.env
        self.total_steps_target = total_timesteps

        obs, _ = env.reset(seed=cfg.seed)
        ep_ret = 0.0
        ep_len = 0
        t0 = time.time()

        pbar = trange(total_timesteps, desc="DQN", leave=True)
        for t in pbar:
            self.total_steps += 1
            self._update_eps()

            if np.random.rand() < self.eps:
                action = env.action_space.sample()
            else:
                action = int(self.predict(obs, deterministic=True))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            self.rb.add(obs, action, reward, float(done), next_obs)
            obs = next_obs
            ep_ret += float(reward)
            ep_len += 1

            if done:
                if self.writer is not None:
                    self.writer.log_episode(self.total_steps, ep_ret, ep_len)
                self.logger.log("rollout/ep_rew", ep_ret)
                self.logger.log("rollout/ep_len", ep_len)
                
                # Reset training environment
                obs, _ = env.reset()
                
                # Reset render environment with same seed if it exists
                if self._render_env is not None and not render_episode_active:
                    self._render_env.reset()
                
                ep_ret = 0.0
                ep_len = 0

            if self.total_steps > cfg.learning_starts and self.total_steps % cfg.train_freq == 0:
                last_loss = None
                for _ in range(cfg.gradient_steps):
                    o, a, r, d, no = self.rb.sample(cfg.batch_size)
                    with torch.no_grad():
                        q_next = self.q_targ(no).max(dim=1).values
                        target = r + cfg.gamma * (1.0 - d) * q_next
                    q_sa = self.q(o).gather(1, a.view(-1, 1)).squeeze(1)
                    loss = F.smooth_l1_loss(q_sa, target)

                    self.optim.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.q.parameters(), cfg.max_grad_norm)
                    self.optim.step()
                    last_loss = float(loss.detach().cpu().item())
                if last_loss is not None:
                    self.logger.log("train/q_loss", last_loss)

            if self.total_steps % cfg.target_update_interval == 0:
                self.q_targ.load_state_dict(self.q.state_dict())

            fps = int(self.total_steps / max(1e-6, (time.time() - t0)))
            self.logger.log("time/fps", float(fps))
            self.logger.log("train/epsilon", float(self.eps))
            self.logger.log("time/total_steps", float(self.total_steps))

            if self.writer is not None and ((t + 1) % log_interval == 0 or t == total_timesteps - 1):
                self.writer.dump(self.total_steps, self.logger.summary())

            self.maybe_checkpoint(self.total_steps, getattr(self, "_checkpoint_freq", 0),
                      prefix=getattr(self, "_checkpoint_prefix", "dqn"))

            if (t + 1) % log_interval == 0:
                summ = self.logger.summary()
                pbar.set_postfix({"eps": round(self.eps, 3), "fps": int(summ.get("time/fps", 0.0))})
                self.logger.reset()

        return self
    def _get_state(self) -> Dict[str, Any]:
        return {
            "algo": "DQN",
            "cfg": self.cfg.__dict__,
            "policy_type": self.policy_type,
            "policy_kwargs": self.policy_kwargs,
            "q": self.q.state_dict(),
            "q_targ": self.q_targ.state_dict(),
        }

    @classmethod
    def _from_state(cls, state: Dict[str, Any], env: gym.Env, device: str):
        cfg = DQNConfig(**state["cfg"])
        cfg.device = device
        obj = cls(state["policy_type"], env, cfg, policy_kwargs=state.get("policy_kwargs"))
        obj.q.load_state_dict(state["q"])
        obj.q_targ.load_state_dict(state["q_targ"])
        obj.q.to(torch.device(device))
        obj.q_targ.to(torch.device(device))
        return obj
