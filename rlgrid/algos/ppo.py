from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import time
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
    def __init__(self, policy: PolicyType, env: gym.Env, cfg: PPOConfig, policy_kwargs: Optional[dict] = None, writer=None):
        super().__init__(env, cfg, writer=writer)
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
        ep_returns, ep_lengths = [], []

        ep_ret = np.zeros(cfg.n_envs, dtype=np.float32)
        ep_len = np.zeros(cfg.n_envs, dtype=np.int32)

        global_step = 0
        t0 = time.time()

        n_updates = total_timesteps // (cfg.n_steps * cfg.n_envs)
        pbar = trange(n_updates, desc="PPO", leave=True)

        # Video recording setup
        video_recorder = None
        recording_env_idx = 0  # Record from first environment

        for update in pbar:
            self.buf.pos = 0
            self.buf.full = False

            # Check if we should start recording a video
            if self.should_record_video(global_step) and video_recorder is None:
                video_recorder = self._setup_video_recording(global_step)
                if video_recorder is not None:
                    video_recorder.start_recording()
                    # Reset the render environment to sync with training
                    if self._render_env is not None:
                        self._render_env.reset(seed=cfg.seed)

            for step in range(cfg.n_steps):
                global_step += cfg.n_envs
                obs_t = to_tensor(obs, self.device)
                with torch.no_grad():
                    actions, logp, values = self.ac.act(obs_t, deterministic=False)

                next_obs, rewards, terms, truncs, infos = env.step(actions.cpu().numpy())
                dones = np.logical_or(terms, truncs).astype(np.float32)

                # Capture video frame if recording
                if video_recorder is not None and video_recorder.recording:
                    # For vectorized environments, sync with the first environment
                    try:
                        if hasattr(env, 'envs') and len(env.envs) > 0:
                            # Take action in render env to sync with training env
                            action_to_take = actions[recording_env_idx].cpu().numpy()
                            self._render_env.step(action_to_take)
                        
                        video_recorder.capture_frame()
                    except Exception as e:
                        print(f"Warning: Could not capture video frame: {e}")

                ep_ret += rewards
                ep_len += 1

                for i in range(cfg.n_envs):
                    if dones[i]:
                        r_i = float(ep_ret[i])
                        l_i = int(ep_len[i])
                        ep_returns.append(r_i)
                        ep_lengths.append(l_i)
                        if self.writer is not None:
                            self.writer.log_episode(global_step, r_i, l_i)
                        
                        # Stop recording if this is the recording environment and episode ended
                        if video_recorder is not None and i == recording_env_idx and video_recorder.recording:
                            filename = f"episode_step_{global_step}_env_{i}"
                            success = video_recorder.stop_recording(filename)
                            if success:
                                print(f"Saved training video: {filename}.mp4")
                            video_recorder = None
                        
                        ep_ret[i] = 0.0
                        ep_len[i] = 0

                if isinstance(infos, dict) and "final_info" in infos and infos["final_info"] is not None:
                    for fi in infos["final_info"]:
                        if fi is not None and "episode" in fi:
                            ep = fi["episode"]
                            ep_returns.append(float(ep["r"]))
                            ep_lengths.append(int(ep["l"]))
                            if self.writer is not None:
                                self.writer.log_episode(global_step, float(ep["r"]), int(ep["l"]))

                self.buf.add(
                    obs=obs,
                    actions=actions.cpu().numpy(),
                    logp=logp.cpu().numpy(),
                    rewards=rewards,
                    dones=dones,
                    values=values.cpu().numpy(),
                )
                obs = next_obs

            with torch.no_grad():
                last_obs_t = to_tensor(obs, self.device)
                _, last_values = self.ac.forward(last_obs_t)
                last_dones = to_tensor(dones, self.device)

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

            n_samples = obs_b.shape[0]
            inds = np.arange(n_samples)

            last_pg = last_v = last_ent = None
            for _epoch in range(cfg.n_epochs):
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

                    last_pg = float(pg_loss.detach().cpu().item())
                    last_v = float(v_loss.detach().cpu().item())
                    last_ent = float(entropy.detach().cpu().item())

            fps = int(global_step / max(1e-6, (time.time() - t0)))
            if ep_returns:
                self.logger.log("rollout/ep_rew_mean", float(np.mean(ep_returns[-100:])))
                self.logger.log("rollout/ep_len_mean", float(np.mean(ep_lengths[-100:])))
            with torch.no_grad():
                self.logger.log("train/explained_variance", explained_variance(old_values_b, ret_b))

            if last_pg is not None:
                self.logger.log("train/policy_loss", last_pg)
            if last_v is not None:
                self.logger.log("train/value_loss", last_v)
            if last_ent is not None:
                self.logger.log("train/entropy", last_ent)

            self.logger.log("time/fps", float(fps))
            self.logger.log("time/total_steps", float(global_step))

            if self.writer is not None and ((update + 1) % log_interval == 0 or update == n_updates - 1):
                self.writer.dump(global_step, self.logger.summary())
            self.maybe_checkpoint(global_step, getattr(self, "_checkpoint_freq", 0),
                    prefix=getattr(self, "_checkpoint_prefix", "ppo"))

            if (update + 1) % log_interval == 0:
                summ = self.logger.summary()
                pbar.set_postfix({"ep_rew": round(summ.get("rollout/ep_rew_mean", 0.0), 3),
                                  "fps": int(summ.get("time/fps", 0.0))})
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
