"""对抗式模仿学习：GAIL、AIRL，以及论文提出的 GC-AIRL 和两个消融版本，外加 RL 基线。

模型          判别器   目标条件(G)   置信度(C)
gail          GAIL     -             -
airl          AIRL     -             -
gairl         AIRL     ✓             -
cairl         AIRL     -             ✓
gcairl        AIRL     ✓             ✓        ← 论文主模型
rl            -（手工奖励，论文 Eq.20）

AIRL 判别器（论文 Eq.7–8）：
    f(s,a,g,s') = c(s,a,g) + γ·h(s') − h(s)，  D = exp(f) / (exp(f) + π(a|s))
生成器奖励（Eq.9）：r = c(s,a,g) + r_goal，r_goal = 1[到达中间目标]。

置信度（Eq.10–13，参考 Zhang et al. 2021 的 CAIL）：每条专家轨迹一个置信度 β（均值为 1），
用于加权判别器的专家项。β 通过双层优化更新：先用当前 β 对奖励网络 c 做一步虚拟梯度更新，
再用更新后的 c 计算各专家轨迹的回报 R̂，与真实排序得分比较得到排序损失 RK，对 β 求梯度。
"""
from __future__ import annotations

import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from .checkpoints import save_gaussian
from .config import Config
from .env import BASE_DIM, MergeEnv, expert_transitions, obs_dim
from .evaluate import evaluate, expert_results
from .metrics import ranking_score
from .policy import GaussianPolicy, ValueNet, mlp
from .ppo import PPO, collect

MODELS = {
    "gail": dict(disc="gail", goal=False, conf=False),
    "airl": dict(disc="airl", goal=False, conf=False),
    "gairl": dict(disc="airl", goal=True, conf=False),
    "cairl": dict(disc="airl", goal=False, conf=True),
    "gcairl": dict(disc="airl", goal=True, conf=True),
    "rl": dict(disc=None, goal=True, conf=False),
}


class GAILDisc(nn.Module):
    def __init__(self, pi: GaussianPolicy, hidden):
        super().__init__()
        self.obs_norm, self.act_norm = pi.obs_norm, pi.act_norm
        n_obs, n_act = pi.obs_norm.mean.numel(), pi.act_norm.mean.numel()
        self.net = mlp(n_obs + n_act, 1, hidden)

    def logit(self, obs, act, next_obs, done, logpi, c_params=None):
        return self.net(torch.cat([self.obs_norm(obs), self.act_norm(act)], -1)).squeeze(-1)


class AIRLDisc(nn.Module):
    def __init__(self, pi: GaussianPolicy, hidden, gamma: float, use_goal: bool = True):
        super().__init__()
        self.obs_norm, self.act_norm = pi.obs_norm, pi.act_norm
        n_obs, n_act = pi.obs_norm.mean.numel(), pi.act_norm.mean.numel()
        self.n_c_obs = n_obs if use_goal else BASE_DIM
        self.c = mlp(self.n_c_obs + n_act, 1, hidden)  # 奖励估计器 c(s, a, g)
        self.h = mlp(BASE_DIM, 1, hidden)              # 势函数 h(s)，只看状态不看目标
        self.gamma = gamma

    def reward_c(self, obs, act, c_params=None):
        x = torch.cat([self.obs_norm(obs)[:, :self.n_c_obs], self.act_norm(act)], -1)
        out = functional_call(self.c, c_params, (x,)) if c_params is not None else self.c(x)
        return out.squeeze(-1)

    def logit(self, obs, act, next_obs, done, logpi, c_params=None):
        h = self.h(self.obs_norm(obs)[:, :BASE_DIM]).squeeze(-1)
        h_next = self.h(self.obs_norm(next_obs)[:, :BASE_DIM]).squeeze(-1)
        f = self.reward_c(obs, act, c_params) + self.gamma * (1 - done) * h_next - h
        return f - logpi


def rank_loss(r_hat_i, r_hat_j, sign, eps):
    """论文 Eq.10：sign = 1 若真实排序 i 优于 j，否则 -1。"""
    diff = r_hat_i - r_hat_j
    hard = torch.clamp(-sign * diff, min=0)
    soft = torch.clamp((sign * diff - eps) ** 2 / (4 * eps), min=0)
    return torch.where(diff.abs() > eps, hard, soft)


class RLReward:
    """论文 Eq.20：效率 + 安全 + 舒适 + 目标，各项做尺度归一化。"""

    def __init__(self, expert_obs: np.ndarray, dt: float, lane_width: float):
        self.dt = dt
        self.x_scale = float(np.ptp(expert_obs[:, 0])) or 1.0
        self.y_scale = lane_width

    def __call__(self, b):
        nobs = b["next_obs"]
        eff = nobs[:, 2] / 30.0
        gap, dvx = nobs[:, 4], nobs[:, 6]
        exists = (nobs[:, 4:8].abs().sum(-1) > 0) & (dvx < 0)
        ttc = torch.where(exists, gap / (-dvx).clamp_min(1e-3), torch.full_like(gap, 1e3))
        safety = torch.where(ttc < 4, ttc / 4, torch.ones_like(ttc))
        jerk = (b["act"] - b["prev_act"]) / self.dt / 10.0
        comfort = -(jerk ** 2).sum(-1)
        goal = -(((nobs[:, BASE_DIM] - nobs[:, 0]) / self.x_scale) ** 2 +
                 ((nobs[:, BASE_DIM + 1] - nobs[:, 1]) / self.y_scale) ** 2)
        return eff + safety + comfort + goal


def train(model: str, train_scenes, val_scenes, cfg: Config, out_dir: str, seed: int = 0) -> GaussianPolicy:
    spec = MODELS[model]
    ac = cfg.adv
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    E = expert_transitions(train_scenes, cfg.env, spec["goal"])
    e = {k: torch.as_tensor(v) for k, v in E.items()}
    n_obs = obs_dim(spec["goal"])
    pi = GaussianPolicy(n_obs, 2, ac.pi_hidden, ac.init_log_std)
    pi.fit_normalizers(e["obs"], e["act"])
    if ac.bc_init:
        ck = torch.load(ac.bc_init, weights_only=False)
        pi.load_state_dict(ck["state_dict"])
        with torch.no_grad():
            pi.log_std.fill_(ac.init_log_std)
    vf = ValueNet(n_obs, ac.pi_hidden)
    vf.obs_norm.fit(e["obs"])
    ppo = PPO(pi, vf, ac)
    # 对抗式模型的奖励可能为负，训练时用吸收态代替碰撞终止；RL 的奖励为正，碰撞终止本身就是惩罚
    terminate = cfg.env.train_terminate_on_collision or spec["disc"] is None
    env = MergeEnv(train_scenes, cfg.env, use_goal=spec["goal"], seed=seed, terminate_on_collision=terminate)

    disc, d_opt, rl_reward = None, None, None
    if spec["disc"] == "gail":
        disc = GAILDisc(pi, ac.d_hidden)
    elif spec["disc"] == "airl":
        disc = AIRLDisc(pi, ac.d_hidden, ac.gamma, use_goal=ac.disc_goal)
    else:
        rl_reward = RLReward(E["obs"], train_scenes[0].dt, train_scenes[0].meta["lane_width"])
    if disc is not None:
        d_opt = torch.optim.Adam(disc.parameters(), lr=ac.d_lr)

    # 置信度：每条专家轨迹一个参数，β = n·softmax(θ)，均值为 1
    n_traj = len(train_scenes)
    theta = torch.zeros(n_traj, requires_grad=True)
    beta_opt = torch.optim.Adam([theta], lr=ac.beta_lr)
    if spec["conf"]:
        scores = np.array([ranking_score(m, ac.score_weights) for m in expert_results(train_scenes, cfg)])
        traj_rows = [np.where(E["traj"] == i)[0] for i in range(n_traj)]
        # expert_transitions 按轨迹顺序排列，因此各行在轨迹内的时间步可直接拼接得到
        t_in_traj = np.concatenate([np.arange(len(r)) for r in traj_rows])
        disc_w = torch.as_tensor(ac.gamma ** t_in_traj, dtype=torch.float32)

    def beta():
        return n_traj * torch.softmax(theta, 0)

    def d_loss_fn(ei, b, pi_i, logpi_e, logpi_p, weights, c_params=None):
        le = disc.logit(e["obs"][ei], e["act"][ei], e["next_obs"][ei], e["done"][ei], logpi_e, c_params)
        lp = disc.logit(b["obs"][pi_i], b["act"][pi_i], b["next_obs"][pi_i], b["term"][pi_i], logpi_p, c_params)
        return (weights * F.softplus(-le)).mean() + F.softplus(lp).mean(), le

    def logpi(obs, act):
        with torch.no_grad():
            return pi.dist(obs).log_prob(pi.act_norm(act)).sum(-1)

    log_f = open(os.path.join(out_dir, "train_log.csv"), "w", newline="", encoding="utf-8")
    writer = None
    best = -np.inf

    for it in range(1, ac.iterations + 1):
        t0 = time.time()
        b, st = collect(env, pi, ac.steps_per_iter)
        row = {"iter": it, "episodes": st["episodes"]}

        if disc is not None:
            for _ in range(ac.d_steps):
                ei = torch.as_tensor(rng.integers(len(e["obs"]), size=ac.d_batch))
                pi_i = torch.as_tensor(rng.integers(len(b["obs"]), size=ac.d_batch))
                lp_e = logpi(e["obs"][ei], e["act"][ei])
                lp_p = logpi(b["obs"][pi_i], b["act"][pi_i])
                w = beta().detach()[e["traj"][ei]] if spec["conf"] else torch.ones(len(ei))
                obs_e = e["obs"][ei].clone().requires_grad_(True)
                le = disc.logit(obs_e, e["act"][ei], e["next_obs"][ei], e["done"][ei], lp_e)
                lp = disc.logit(b["obs"][pi_i], b["act"][pi_i], b["next_obs"][pi_i], b["term"][pi_i], lp_p)
                loss = (w * F.softplus(-le)).mean() + F.softplus(lp).mean()
                gp = (torch.autograd.grad(le.sum(), obs_e, create_graph=True)[0] ** 2).sum(-1).mean()
                d_opt.zero_grad()
                (loss + ac.grad_penalty * gp).backward()
                d_opt.step()
            row.update(d_loss=loss.item(), d_acc_e=(le > 0).float().mean().item(),
                       d_acc_p=(lp < 0).float().mean().item())

            if spec["conf"]:  # 双层优化更新置信度 β
                ei = torch.as_tensor(rng.integers(len(e["obs"]), size=ac.d_batch))
                pi_i = torch.as_tensor(rng.integers(len(b["obs"]), size=ac.d_batch))
                params = dict(disc.c.named_parameters())
                inner, _ = d_loss_fn(ei, b, pi_i, logpi(e["obs"][ei], e["act"][ei]),
                                     logpi(b["obs"][pi_i], b["act"][pi_i]), beta()[e["traj"][ei]], params)
                grads = torch.autograd.grad(inner, list(params.values()), create_graph=True)
                new = {k: p - ac.inner_lr * g for (k, p), g in zip(params.items(), grads)}
                trajs = rng.choice(n_traj, size=min(ac.rank_trajs, n_traj), replace=False)
                rows = torch.as_tensor(np.concatenate([traj_rows[i] for i in trajs]))
                r = disc.reward_c(e["obs"][rows], e["act"][rows], new)
                if spec["goal"]:
                    r = r + ac.goal_reward * e["goal_now"][rows]
                r = r * disc_w[rows]
                seg = torch.as_tensor(np.concatenate([[k] * len(traj_rows[i]) for k, i in enumerate(trajs)]))
                R = torch.zeros(len(trajs)).index_add(0, seg, r)
                ii, jj = np.triu_indices(len(trajs), 1)
                sign = torch.as_tensor(np.sign(scores[trajs][ii] - scores[trajs][jj]), dtype=torch.float32)
                out_loss = rank_loss(R[ii], R[jj], sign, ac.rank_eps).sum()
                beta_opt.zero_grad()
                out_loss.backward()
                beta_opt.step()
                disc.zero_grad()
                bt = beta().detach()
                row.update(rank_loss=out_loss.item(), beta_min=bt.min().item(), beta_max=bt.max().item())

            with torch.no_grad():
                if spec["disc"] == "gail":
                    lg = disc.logit(b["obs"], b["act"], None, None, None)
                    rew = -F.softplus(-lg) if ac.gail_reward == "logD" else F.softplus(lg)
                else:
                    if ac.airl_reward == "c":
                        rew = disc.reward_c(b["obs"], b["act"])
                    else:
                        rew = disc.logit(b["obs"], b["act"], b["next_obs"], b["term"], logpi(b["obs"], b["act"]))
        else:
            with torch.no_grad():
                rew = rl_reward(b)
        if spec["goal"] and spec["disc"] is not None:
            rew = rew + ac.goal_reward * b["goal_now"]

        ppo.update(b, rew)
        eps = max(1, st["episodes"])
        row.update(train_success=st["success"] / eps, train_collision=st["collision"] / eps,
                   reward=rew.mean().item(), log_std=pi.log_std.mean().item(), secs=time.time() - t0)

        msg = ""
        if it % ac.eval_every == 0 or it == ac.iterations:
            m, _ = evaluate(val_scenes, cfg, pi, spec["goal"], n_runs=1)
            row.update(val_success=m["success_rate"], val_collision=m["collision_rate"])
            key = m["success_rate"] - m["collision_rate"]
            if key >= best:
                best = key
                save_gaussian(pi, spec["goal"], model, ac.pi_hidden, os.path.join(out_dir, "best.pt"))
            msg = f" | val succ {m['success_rate']:.2f} coll {m['collision_rate']:.2f}"
        print(f"[{model}] it {it:4d} | succ {row['train_success']:.2f} coll {row['train_collision']:.2f} "
              f"| r {row['reward']:.3f} | std {row['log_std']:.2f} | {row['secs']:.1f}s{msg}", flush=True)

        if writer is None:
            fields = list(row) + [k for k in ("val_success", "val_collision") if k not in row]
            writer = csv.DictWriter(log_f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
        writer.writerow(row)
        log_f.flush()

    log_f.close()
    save_gaussian(pi, spec["goal"], model, ac.pi_hidden, os.path.join(out_dir, "last.pt"))
    if spec["conf"]:
        np.save(os.path.join(out_dir, "beta.npy"), beta().detach().numpy())
    return pi

