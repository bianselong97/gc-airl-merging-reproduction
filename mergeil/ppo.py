"""PPO（论文 3.3.4 节，Eq.14–15）：采样、GAE 优势估计和 PPO-clip 更新。奖励由调用方给出。"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import AdvConfig
from .env import MergeEnv
from .policy import GaussianPolicy, ValueNet


def collect(env: MergeEnv, pi: GaussianPolicy, n_steps: int):
    keys = ("obs", "next_obs", "act", "prev_act", "z", "logp", "term", "end", "goal_now")
    buf = {k: [] for k in keys}
    stats = {"episodes": 0, "collision": 0, "success": 0}
    obs = env.reset()
    for t in range(n_steps):
        a, z, logp = pi.act(obs)
        prev = env.prev_act.copy()
        nxt, term, trunc, info = env.step(a)
        buf["obs"].append(obs)
        buf["next_obs"].append(nxt)
        buf["act"].append(env.prev_act.copy())  # 截断后的实际动作
        buf["prev_act"].append(prev)
        buf["z"].append(z)
        buf["logp"].append(logp)
        buf["term"].append(term)
        buf["end"].append(term or trunc or t == n_steps - 1)
        buf["goal_now"].append(info["goal_now"])
        if term or trunc:
            stats["episodes"] += 1
            stats["collision"] += info["collision"]
            stats["success"] += info["success"]
            obs = env.reset()
        else:
            obs = nxt
    return {k: torch.as_tensor(np.asarray(v), dtype=torch.float32) for k, v in buf.items()}, stats


def gae(rew, val, next_val, term, end, gamma, lam):
    adv = torch.zeros_like(rew)
    last = 0.0
    for t in reversed(range(len(rew))):
        delta = rew[t] + gamma * next_val[t] * (1 - term[t]) - val[t]
        last = delta + gamma * lam * (1 - end[t]) * last
        adv[t] = last
    return adv, adv + val


class PPO:
    def __init__(self, pi: GaussianPolicy, vf: ValueNet, cfg: AdvConfig):
        self.pi, self.vf, self.cfg = pi, vf, cfg
        self.pi_opt = torch.optim.Adam(pi.parameters(), lr=cfg.pi_lr)
        self.vf_opt = torch.optim.Adam(vf.net.parameters(), lr=cfg.vf_lr)

    def update(self, b: dict, rew: torch.Tensor):
        cfg = self.cfg
        with torch.no_grad():
            val, next_val = self.vf(b["obs"]), self.vf(b["next_obs"])
        adv, ret = gae(rew, val, next_val, b["term"], b["end"], cfg.gamma, cfg.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(adv)
        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(n)
            for i in range(0, n, cfg.minibatch):
                mb = perm[i:i + cfg.minibatch]
                logp = self.pi.dist(b["obs"][mb]).log_prob(b["z"][mb]).sum(-1)
                ratio = (logp - b["logp"][mb]).exp()
                pi_loss = -torch.min(ratio * adv[mb], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv[mb]).mean()
                self.pi_opt.zero_grad()
                pi_loss.backward()
                nn.utils.clip_grad_norm_(self.pi.parameters(), cfg.max_grad_norm)
                self.pi_opt.step()

                v_loss = ((self.vf(b["obs"][mb]) - ret[mb]) ** 2).mean()
                self.vf_opt.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.vf.parameters(), cfg.max_grad_norm)
                self.vf_opt.step()
        return {"pi_loss": pi_loss.item(), "v_loss": v_loss.item()}
