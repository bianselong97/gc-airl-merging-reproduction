"""高斯策略、价值网络和输入归一化。归一化统计量取自专家数据，保存在模型里。"""
from __future__ import annotations

import torch
import torch.nn as nn


def mlp(inp: int, out: int, hidden=(128, 128), act=nn.Tanh) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class Normalizer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def fit(self, x: torch.Tensor):
        self.mean.copy_(x.mean(0))
        std = x.std(0)
        # 近似常数的特征（如合成数据中的目标横向位置）不缩放，否则偏离常数的输入会被放大成极端值
        self.std.copy_(torch.where(std < 1e-2, torch.ones_like(std), std))
        return self

    def forward(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean


class GaussianPolicy(nn.Module):
    """输出物理单位的动作；网络内部在归一化后的观测/动作空间中工作。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden=(128, 128), init_log_std: float = -0.5):
        super().__init__()
        self.obs_norm = Normalizer(obs_dim)
        self.act_norm = Normalizer(act_dim)
        self.net = mlp(obs_dim, act_dim, hidden)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def fit_normalizers(self, obs: torch.Tensor, act: torch.Tensor):
        self.obs_norm.fit(obs)
        self.act_norm.fit(act)

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        """归一化动作空间中的分布。"""
        mu = self.net(self.obs_norm(obs))
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        d = self.dist(obs_t)
        z = d.mean if deterministic else d.sample()
        logp = d.log_prob(z).sum(-1)
        return self.act_norm.inverse(z)[0].numpy(), z[0].numpy(), float(logp)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden=(128, 128)):
        super().__init__()
        self.obs_norm = Normalizer(obs_dim)
        self.net = mlp(obs_dim, 1, hidden)

    def forward(self, obs):
        return self.net(self.obs_norm(obs)).squeeze(-1)
