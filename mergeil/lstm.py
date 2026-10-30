"""LSTM 基线（论文 5.3.4 节）：输入与 GC-AIRL 相同（状态 + 目标），输出二维加速度。
2 层 LSTM，每层 256，dropout 0.2，MSE 损失，RMSprop 优化。闭环行驶时逐步喂入观测并保留隐状态。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import EnvConfig, LSTMConfig
from .env import expert_actions, obs_dim, observe
from .policy import Normalizer


class LSTMPolicy(nn.Module):
    def __init__(self, n_obs: int, n_act: int, cfg: LSTMConfig):
        super().__init__()
        self.obs_norm, self.act_norm = Normalizer(n_obs), Normalizer(n_act)
        self.lstm = nn.LSTM(n_obs, cfg.hidden, cfg.layers, batch_first=True, dropout=cfg.dropout)
        self.head = nn.Linear(cfg.hidden, n_act)
        self.state = None

    def forward(self, seq, state=None):
        out, state = self.lstm(self.obs_norm(seq), state)
        return self.head(out), state

    def reset(self):
        self.state = None

    @torch.no_grad()
    def act(self, obs, deterministic: bool = True):
        x = torch.as_tensor(obs, dtype=torch.float32)[None, None]
        z, self.state = self(x, self.state)
        return self.act_norm.inverse(z[0, 0]).numpy(), None, 0.0


def train_lstm(scenes, env_cfg: EnvConfig, cfg: LSTMConfig, seed: int = 0, log_every: int = 25) -> LSTMPolicy:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    seqs = []
    for sc in scenes:
        o = np.stack([observe(sc.ego[k], sc.others[k], sc.meta, sc.goal) for k in range(sc.horizon)])
        seqs.append((torch.as_tensor(o), torch.as_tensor(expert_actions(sc, env_cfg))))
    all_o = torch.cat([s[0] for s in seqs])
    all_a = torch.cat([s[1] for s in seqs])
    net = LSTMPolicy(obs_dim(True), 2, cfg)
    net.obs_norm.fit(all_o)
    net.act_norm.fit(all_a)
    opt = torch.optim.RMSprop(net.parameters(), lr=cfg.lr)

    for ep in range(1, cfg.epochs + 1):
        net.train()
        perm = rng.permutation(len(seqs))
        total = 0.0
        for i in range(0, len(perm), cfg.batch_size):
            batch = [seqs[j] for j in perm[i:i + cfg.batch_size]]
            o = nn.utils.rnn.pad_sequence([s[0] for s in batch], batch_first=True)
            a = nn.utils.rnn.pad_sequence([s[1] for s in batch], batch_first=True)
            mask = nn.utils.rnn.pad_sequence([torch.ones(len(s[0])) for s in batch], batch_first=True)
            pred, _ = net(o)
            err = ((pred - net.act_norm(a)) ** 2).sum(-1)
            loss = (err * mask).sum() / mask.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(batch)
        if ep % log_every == 0 or ep == cfg.epochs:
            print(f"[lstm] epoch {ep:4d}  mse {total / len(seqs):.4f}")
    net.eval()
    return net
