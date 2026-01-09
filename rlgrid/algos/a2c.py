from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import time
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
    def __init__(self, policy: PolicyType, env: gym.Env, cfg: A2CConfig, policy_kwargs: Optional[dict] = None, writer=None):
        super().__init__(env, cfg, writer=writer)
        self.policy_type = policy
        self.policy_kwargs = policy_kwargs or {}
        obs_shape = env.single_observation_space.shape if hasattr(env, "single_observation_space") else env.observation_space.shape
        n_actions = env.single_action_space.n if hasattr(env, "single_action_space") else env.action_space.n

        obs_mode = self.policy_kwargs.get("obs_mode", "image")  # "image"|"image_dir"|"rgb"
        n_obj = self.policy_kwargs.get("n_obj", 11)
        n_col = self.policy_kwargs.get("n_col", 6)
        n_state = self.policy_kwargs.get("n_state", 3)

        self.ac = ActorCritic(
            obs_shape,
            n_actions,
            policy,
            obs_mode=obs_mode,
            n_obj=n_obj,
            n_col=n_col,
            n_state=n_state,
        ).to(self.device)
        
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

        global_step = 0
        t0 = time.time()
        ep_returns, ep_lengths = [], []

        n_updates = total_timesteps // (cfg.n_steps * cfg.n_envs)
        pbar = trange(n_updates, desc="A2C", leave=True)

        # Video recording setup
        video_recorder = None
        recording_env_idx = 0  # Record from first environment

        for update in pbar:
            obs_buf, act_buf, rew_buf, done_buf, val_buf = [], [], [], [], []

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
                logits, values = self.ac.forward(obs_t)
                dist = CategoricalDist(logits=logits)
                actions = dist.sample()

                next_obs, rewards, terms, truncs, infos = env.step(actions.cpu().numpy())
                dones = np.logical_or(terms, truncs).astype(np.float32)

                # Capture video frame if recording
                if video_recorder is not None and video_recorder.recording:
                    try:
                        if hasattr(env, 'envs') and len(env.envs) > 0:
                            # Take action in render env to sync with training env
                            action_to_take = actions[recording_env_idx].cpu().numpy()
                            self._render_env.step(action_to_take)
                        
                        video_recorder.capture_frame()
                    except Exception as e:
                        print(f"Warning: Could not capture video frame: {e}")

                # Handle episode endings and video recording
                for i in range(cfg.n_envs):
                    if dones[i]:
                        # Stop recording if this is the recording environment and episode ended
                        if video_recorder is not None and i == recording_env_idx and video_recorder.recording:
                            filename = f"episode_step_{global_step}_env_{i}"
                            success = video_recorder.stop_recording(filename)
                            if success:
                                print(f"Saved training video: {filename}.mp4")
                            video_recorder = None

                if isinstance(infos, dict) and "final_info" in infos and infos["final_info"] is not None:
                    for fi in infos["final_info"]:
                        if fi is not None and "episode" in fi:
                            ep = fi["episode"]
                            ep_returns.append(float(ep["r"]))
                            ep_lengths.append(int(ep["l"]))
                            if self.writer is not None:
                                self.writer.log_episode(global_step, float(ep["r"]), int(ep["l"]))

                obs_buf.append(obs_t)
                act_buf.append(actions)
                rew_buf.append(to_tensor(rewards, self.device))
                done_buf.append(to_tensor(dones, self.device))
                val_buf.append(values)
                obs = next_obs

            with torch.no_grad():
                last_obs_t = to_tensor(obs, self.device)
                _, last_values = self.ac.forward(last_obs_t)

            returns = []
            R = last_values
            for t in reversed(range(cfg.n_steps)):
                R = rew_buf[t] + cfg.gamma * (1.0 - done_buf[t]) * R
                returns.append(R)
            returns = list(reversed(returns))

            obs_b = torch.cat(obs_buf, dim=0)
            act_b = torch.cat(act_buf, dim=0)
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

            fps = int(global_step / max(1e-6, (time.time() - t0)))
            if ep_returns:
                self.logger.log("rollout/ep_rew_mean", float(np.mean(ep_returns[-100:])))
                self.logger.log("rollout/ep_len_mean", float(np.mean(ep_lengths[-100:])))
            self.logger.log("train/policy_loss", float(policy_loss.detach().cpu().item()))
            self.logger.log("train/value_loss", float(value_loss.detach().cpu().item()))
            self.logger.log("train/entropy", float(entropy.detach().cpu().item()))
            self.logger.log("time/fps", float(fps))
            self.logger.log("time/total_steps", float(global_step))

            if self.writer is not None and ((update + 1) % log_interval == 0 or update == n_updates - 1):
                self.writer.dump(global_step, self.logger.summary())

            self.maybe_checkpoint(self.total_steps, getattr(self, "_checkpoint_freq", 0),
                      prefix=getattr(self, "_checkpoint_prefix", "dqn"))

            if (update + 1) % log_interval == 0:
                summ = self.logger.summary()
                pbar.set_postfix({"ep_rew": round(summ.get("rollout/ep_rew_mean", 0.0), 3),
                                  "fps": int(summ.get("time/fps", 0.0))})
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
