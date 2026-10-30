"""行为克隆（BC）基线：在专家 (obs, act) 上做最大似然。"""
from __future__ import annotations

import numpy as np
import torch

from .config import BCConfig
from .policy import GaussianPolicy


def train_bc(obs: np.ndarray, act: np.ndarray, cfg: BCConfig, val_frac: float = 0.1,
             seed: int = 0, log_every: int = 20) -> GaussianPolicy:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs_t = torch.as_tensor(obs)
    act_t = torch.as_tensor(act)
    pi = GaussianPolicy(obs.shape[1], act.shape[1], cfg.hidden)
    pi.fit_normalizers(obs_t, act_t)

    idx = rng.permutation(len(obs))
    n_val = int(len(obs) * val_frac)
    val, tr = idx[:n_val], idx[n_val:]
    opt = torch.optim.Adam(pi.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for ep in range(1, cfg.epochs + 1):
        perm = rng.permutation(tr)
        for i in range(0, len(perm), cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            z = pi.act_norm(act_t[b])
            loss = -pi.dist(obs_t[b]).log_prob(z).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % log_every == 0 or ep == cfg.epochs:
            with torch.no_grad():
                mu = pi.act_norm.inverse(pi.dist(obs_t[val]).mean)
                rmse = ((mu - act_t[val]) ** 2).mean(0).sqrt().numpy()
            print(f"[BC] epoch {ep:4d}  nll {loss.item():.3f}  "
                  f"val RMSE ax {rmse[0]:.3f}  ay {rmse[1]:.3f} m/s^2")
    return pi
