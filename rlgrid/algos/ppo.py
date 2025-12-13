from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import trange

from rlgrid.algos.base import AlgoConfig, BaseAlgorithm
from rlgrid.buffers.rollout import RolloutBuffer
from rlgrid.common.torch_utils import explained_variance, to_tensor
from rlgrid.policies.base import ActorCritic, PolicyType
from rlgrid.policies.distributions import CategoricalDist

@dataclass
class PPOConfig(AlgoConfig):
    n_steps: int = 128
    n_envs: int = 8
    batch_size: int = 256
    n_epochs: int = 4
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    gae_lambda: float = 0.95
    normalize_adv: bool = True

class PPO(BaseAlgorithm):
    def __init__(self, policy: PolicyType, env: gym.Env, cfg: PPOConfig, policy_kwargs: Optional[dict] = None):
        super().__init__(env, cfg)
        self.policy_type = policy
        self.policy_kwargs = policy_kwargs or {}
        obs_shape = env.single_observation_space.shape if hasattr(env, "single_observation_space") else env.observation_space.shape
        n_actions = env.single_action_space.n if hasattr(env, "single_action_space") else env.action_space.n
        self.ac = ActorCritic(obs_shape, n_actions, policy).to(self.device)
        self.optim = torch.optim.Adam(self.ac.parameters(), lr=cfg.lr)
        self.buf = RolloutBuffer(cfg.n_steps, cfg.n_envs, obs_shape, self.device)

    def predict(self, obs, deterministic: bool = True):
        obs_t = to_tensor(obs, self.device)
        if obs_t.ndim == 3:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            a, _, _ = self.ac.act(obs_t, deterministic=deterministic)
        return a.squeeze(0).cpu().numpy()

    def learn(self, total_timesteps: int, log_interval: int = 10):
        cfg: PPOConfig = self.cfg  # type: ignore
        env = self.env

        obs, _ = env.reset(seed=cfg.seed)
        ep_info_buffer = []

        n_updates = total_timesteps // (cfg.n_steps * cfg.n_envs)
        pbar = trange(n_updates, desc="PPO", leave=True)

        for update in pbar:
            self.buf.pos = 0
            self.buf.full = False

            for step in range(cfg.n_steps):
                obs_t = to_tensor(obs, self.device)
                with torch.no_grad():
                    actions, logp, values = self.ac.act(obs_t, deterministic=False)

                actions_np = actions.cpu().numpy()
                next_obs, rewards, terms, truncs, infos = env.step(actions_np)
                dones = np.logical_or(terms, truncs).astype(np.float32)

                # record episode stats
                if "episode" in infos:
                    # vector env: infos["episode"] is list/dict depending on wrapper
                    pass
                if isinstance(infos, dict) and "final_info" in infos and infos["final_info"] is not None:
                    for fi in infos["final_info"]:
                        if fi is not None and "episode" in fi:
                            ep_info_buffer.append(fi["episode"])

                self.buf.add(
                    obs=obs,
                    actions=actions_np,
                    logp=logp.cpu().numpy(),
                    rewards=rewards,
                    dones=dones,
                    values=values.cpu().numpy(),
                )
                obs = next_obs

            # bootstrap
            with torch.no_grad():
                last_obs_t = to_tensor(obs, self.device)
                _, last_values = self.ac.forward(last_obs_t)
                last_dones = to_tensor(dones, self.device)  # from last step

            adv, ret = self.buf.compute_returns_and_advantages(
                last_values=last_values,
                last_dones=last_dones,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
                normalize_adv=cfg.normalize_adv,
            )
            obs_b, actions_b, old_logp_b, old_values_b = self.buf.get()
            adv_b = adv.reshape(-1)
            ret_b = ret.reshape(-1)

            # PPO epochs
            n_samples = obs_b.shape[0]
            inds = np.arange(n_samples)

            for epoch in range(cfg.n_epochs):
                np.random.shuffle(inds)
                for start in range(0, n_samples, cfg.batch_size):
                    mb = inds[start:start+cfg.batch_size]
                    logits, values = self.ac.forward(obs_b[mb])
                    dist = CategoricalDist(logits=logits)
                    logp = dist.log_prob(actions_b[mb])
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(logp - old_logp_b[mb])
                    pg_loss1 = -adv_b[mb] * ratio
                    pg_loss2 = -adv_b[mb] * torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    v_loss = F.mse_loss(values, ret_b[mb])
                    loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy

                    self.optim.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ac.parameters(), cfg.max_grad_norm)
                    self.optim.step()

            # logging
            if len(ep_info_buffer) > 0:
                mean_ret = float(np.mean([e["r"] for e in ep_info_buffer[-100:]]))
                mean_len = float(np.mean([e["l"] for e in ep_info_buffer[-100:]]))
                self.logger.log("rollout/ep_rew_mean", mean_ret)
                self.logger.log("rollout/ep_len_mean", mean_len)

            with torch.no_grad():
                y_pred = old_values_b
                y_true = ret_b
                self.logger.log("train/explained_variance", explained_variance(y_pred, y_true))

            if (update + 1) % log_interval == 0:
                summ = self.logger.summary()
                pbar.set_postfix({k: round(v, 3) for k, v in summ.items() if "ep_rew" in k or "expl" in k})
                self.logger.reset()

        return self

    def _get_state(self) -> Dict[str, Any]:
        return {
            "algo": "PPO",
            "cfg": self.cfg.__dict__,
            "policy_type": self.policy_type,
            "policy_kwargs": self.policy_kwargs,
            "state_dict": self.ac.state_dict(),
        }

    @classmethod
    def _from_state(cls, state: Dict[str, Any], env: gym.Env, device: str):
        cfg = PPOConfig(**state["cfg"])
        cfg.device = device
        obj = cls(state["policy_type"], env, cfg, policy_kwargs=state.get("policy_kwargs"))
        obj.ac.load_state_dict(state["state_dict"])
        obj.ac.to(torch.device(device))
        return obj
