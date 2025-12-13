from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import copy

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import trange

from rlgrid.algos.base import AlgoConfig, BaseAlgorithm
from rlgrid.buffers.replay import ReplayBuffer
from rlgrid.common.torch_utils import to_tensor
from rlgrid.policies.q_networks import QuantileQNetwork, NetType
from rlgrid.policies.distributions import huber_quantile_loss

@dataclass
class QRDQNConfig(AlgoConfig):
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
    n_quantiles: int = 50
    kappa: float = 1.0

class QRDQN(BaseAlgorithm):
    def __init__(self, policy: NetType, env: gym.Env, cfg: QRDQNConfig, policy_kwargs: Optional[dict] = None):
        super().__init__(env, cfg)
        self.policy_type = policy
        self.policy_kwargs = policy_kwargs or {}
        obs_shape = env.observation_space.shape
        n_actions = env.action_space.n
        self.q = QuantileQNetwork(obs_shape, n_actions, policy, n_quantiles=cfg.n_quantiles).to(self.device)
        self.q_targ = copy.deepcopy(self.q).to(self.device)
        self.optim = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.rb = ReplayBuffer(cfg.buffer_size, obs_shape, self.device)
        self.total_steps = 0
        self.eps = cfg.exploration_initial_eps
        # quantile midpoints
        self.taus = torch.linspace(0.5 / cfg.n_quantiles, 1 - 0.5 / cfg.n_quantiles, cfg.n_quantiles, device=self.device)

    def _update_eps(self):
        cfg: QRDQNConfig = self.cfg  # type: ignore
        frac = min(1.0, self.total_steps / max(1, int(cfg.exploration_fraction * self.total_steps_target)))
        self.eps = cfg.exploration_initial_eps + frac * (cfg.exploration_final_eps - cfg.exploration_initial_eps)

    def predict(self, obs, deterministic: bool = True):
        obs_t = to_tensor(obs, self.device)
        if obs_t.ndim == 3:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            q = self.q(obs_t).mean(dim=-1)  # expectation over quantiles
            a = torch.argmax(q, dim=-1)
        return a.squeeze(0).cpu().numpy()

    def learn(self, total_timesteps: int, log_interval: int = 200):
        cfg: QRDQNConfig = self.cfg  # type: ignore
        env = self.env
        self.total_steps_target = total_timesteps

        obs, _ = env.reset(seed=cfg.seed)
        ep_ret = 0.0
        returns = []

        pbar = trange(total_timesteps, desc="QRDQN", leave=True)
        for t in pbar:
            self.total_steps += 1
            self._update_eps()

            if np.random.rand() < self.eps:
                action = env.action_space.sample()
            else:
                action = int(self.predict(obs, deterministic=True))

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            self.rb.add(obs, action, reward, float(done), next_obs)
            obs = next_obs
            ep_ret += float(reward)

            if done:
                returns.append(ep_ret)
                self.logger.log("rollout/ep_rew_mean", float(np.mean(returns[-50:])))
                obs, _ = env.reset()
                ep_ret = 0.0

            # train
            if self.total_steps > cfg.learning_starts and self.total_steps % cfg.train_freq == 0:
                for _ in range(cfg.gradient_steps):
                    o, a, r, d, no = self.rb.sample(cfg.batch_size)

                    # current quantiles: (B,A,N) -> choose action a: (B,N)
                    q_quant = self.q(o)
                    qa = q_quant.gather(1, a.view(-1, 1, 1).expand(-1, 1, cfg.n_quantiles)).squeeze(1)

                    with torch.no_grad():
                        # Double DQN action selection on expectation
                        q_next = self.q(no).mean(dim=-1)
                        next_a = torch.argmax(q_next, dim=1)
                        targ_quant = self.q_targ(no)
                        targ = targ_quant.gather(1, next_a.view(-1, 1, 1).expand(-1, 1, cfg.n_quantiles)).squeeze(1)
                        target = r.view(-1, 1) + cfg.gamma * (1.0 - d.view(-1, 1)) * targ  # (B,N)

                    # pairwise td errors for quantile regression: (B,N,N')
                    td = target.unsqueeze(1) - qa.unsqueeze(2)
                    loss = huber_quantile_loss(td, self.taus, kappa=cfg.kappa)

                    self.optim.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.q.parameters(), cfg.max_grad_norm)
                    self.optim.step()

            if self.total_steps % cfg.target_update_interval == 0:
                self.q_targ.load_state_dict(self.q.state_dict())

            if (t + 1) % log_interval == 0:
                summ = self.logger.summary()
                pbar.set_postfix({"eps": round(self.eps, 3), "ep_rew": round(summ.get("rollout/ep_rew_mean", 0.0), 3)})
                self.logger.reset()

        return self

    def _get_state(self) -> Dict[str, Any]:
        return {
            "algo": "QRDQN",
            "cfg": self.cfg.__dict__,
            "policy_type": self.policy_type,
            "policy_kwargs": self.policy_kwargs,
            "q": self.q.state_dict(),
            "q_targ": self.q_targ.state_dict(),
        }

    @classmethod
    def _from_state(cls, state: Dict[str, Any], env: gym.Env, device: str):
        cfg = QRDQNConfig(**state["cfg"])
        cfg.device = device
        obj = cls(state["policy_type"], env, cfg, policy_kwargs=state.get("policy_kwargs"))
        obj.q.load_state_dict(state["q"])
        obj.q_targ.load_state_dict(state["q_targ"])
        obj.q.to(torch.device(device))
        obj.q_targ.to(torch.device(device))
        return obj
