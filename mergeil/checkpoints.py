"""保存/加载各类策略（高斯策略或 LSTM）。"""
from __future__ import annotations

import torch

from .config import LSTMConfig
from .env import obs_dim
from .lstm import LSTMPolicy
from .policy import GaussianPolicy


def save_lstm(net: LSTMPolicy, cfg: LSTMConfig, path: str):
    torch.save({"kind": "lstm", "model": "lstm", "use_goal": True, "cfg": cfg.__dict__,
                "state_dict": net.state_dict()}, path)


def save_gaussian(pi: GaussianPolicy, use_goal: bool, model: str, hidden, path: str):
    torch.save({"kind": "gaussian", "model": model, "use_goal": use_goal, "hidden": tuple(hidden),
                "state_dict": pi.state_dict()}, path)


def load_policy(path: str):
    """返回 (policy, use_goal, model 名)。"""
    ck = torch.load(path, weights_only=False)
    if ck["kind"] == "lstm":
        net = LSTMPolicy(obs_dim(True), 2, LSTMConfig(**ck["cfg"]))
    else:
        net = GaussianPolicy(obs_dim(ck["use_goal"]), 2, tuple(ck["hidden"]))
    net.load_state_dict(ck["state_dict"])
    return net.eval(), ck["use_goal"], ck["model"]
