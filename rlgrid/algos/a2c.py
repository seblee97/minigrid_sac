from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import trange

from rlgrid.algos.base import AlgoConfig, BaseAlgorithm
from rlgrid.common.torch_utils import to_tensor
from rlgrid.policies.base import ActorCritic, PolicyType
from rlgrid.policies.distributions import CategoricalDist

@dataclass
class A2CConfig(AlgoConfig):
    n_steps: int = 5
    n_envs: int = 8
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    gae_lambda: float = 1.0  # if 1.0, becomes n-step return without GAE-ish smoothing

class A2C(BaseAlgorithm):
    def __init__(self, policy: PolicyType, env: gym.Env, cfg: A2CConfig, policy_kwargs: Optional[dict] = None):
        super().__init__(env, cfg)
        self.policy_type = policy
        self.policy_kwargs = policy_kwargs or {}
        obs_shape = env.single_observation_space.shape if hasattr(env, "single_observation_space") else env.observation_space.shape
        n_actions = env.single_action_space.n if hasattr(env, "single_action_space") else env.action_space.n
        self.ac = ActorCritic(obs_shape, n_actions, policy).to(self.device)
        self.optim = torch.optim.Adam(self.ac.parameters(), lr=cfg.lr)

    def predict(self, obs, deterministic: bool = True):
        obs_t = to_tensor(obs, self.device)
        if obs_t.ndim == 3:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            a, _, _ = self.ac.act(obs_t, deterministic=deterministic)
        return a.squeeze(0).cpu().numpy()

    def learn(self, total_timesteps: int, log_interval: int = 10):
        cfg: A2CConfig = self.cfg  # type: ignore
        env = self.env
        obs, _ = env.reset(seed=cfg.seed)
        ep_info_buffer = []

        n_updates = total_timesteps // (cfg.n_steps * cfg.n_envs)
        pbar = trange(n_updates, desc="A2C", leave=True)

        for update in pbar:
            obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf = [], [], [], [], [], []

            for _ in range(cfg.n_steps):
                obs_t = to_tensor(obs, self.device)
                logits, values = self.ac.forward(obs_t)
                dist = CategoricalDist(logits=logits)
                actions = dist.sample()
                logp = dist.log_prob(actions)

                next_obs, rewards, terms, truncs, infos = env.step(actions.cpu().numpy())
                dones = np.logical_or(terms, truncs).astype(np.float32)

                if isinstance(infos, dict) and "final_info" in infos and infos["final_info"] is not None:
                    for fi in infos["final_info"]:
                        if fi is not None and "episode" in fi:
                            ep_info_buffer.append(fi["episode"])

                obs_buf.append(obs_t)
                act_buf.append(actions)
                rew_buf.append(to_tensor(rewards, self.device))
                done_buf.append(to_tensor(dones, self.device))
                val_buf.append(values)
                logp_buf.append(logp)

                obs = next_obs

            with torch.no_grad():
                last_obs_t = to_tensor(obs, self.device)
                _, last_values = self.ac.forward(last_obs_t)

            # compute returns (n-step with bootstrap)
            returns = []
            R = last_values
            for t in reversed(range(cfg.n_steps)):
                R = rew_buf[t] + cfg.gamma * (1.0 - done_buf[t]) * R
                returns.append(R)
            returns = list(reversed(returns))

            # flatten
            obs_b = torch.cat(obs_buf, dim=0)
            act_b = torch.cat(act_buf, dim=0)
            logp_b = torch.cat(logp_buf, dim=0)
            val_b = torch.cat(val_buf, dim=0)
            ret_b = torch.cat(returns, dim=0)

            adv = (ret_b - val_b).detach()
            logits, values = self.ac.forward(obs_b)
            dist = CategoricalDist(logits=logits)
            entropy = dist.entropy().mean()

            policy_loss = -(dist.log_prob(act_b) * adv).mean()
            value_loss = F.mse_loss(values, ret_b)
            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.ac.parameters(), cfg.max_grad_norm)
            self.optim.step()

            if len(ep_info_buffer) > 0:
                mean_ret = float(np.mean([e["r"] for e in ep_info_buffer[-100:]]))
                self.logger.log("rollout/ep_rew_mean", mean_ret)

            if (update + 1) % log_interval == 0:
                summ = self.logger.summary()
                pbar.set_postfix({k: round(v, 3) for k, v in summ.items() if "ep_rew" in k})
                self.logger.reset()

        return self

    def _get_state(self) -> Dict[str, Any]:
        return {
            "algo": "A2C",
            "cfg": self.cfg.__dict__,
            "policy_type": self.policy_type,
            "policy_kwargs": self.policy_kwargs,
            "state_dict": self.ac.state_dict(),
        }

    @classmethod
    def _from_state(cls, state: Dict[str, Any], env: gym.Env, device: str):
        cfg = A2CConfig(**state["cfg"])
        cfg.device = device
        obj = cls(state["policy_type"], env, cfg, policy_kwargs=state.get("policy_kwargs"))
        obj.ac.load_state_dict(state["state_dict"])
        obj.ac.to(torch.device(device))
        return obj
