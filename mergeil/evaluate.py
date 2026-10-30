"""闭环评估：策略从专家轨迹的初始状态出发行驶到专家轨迹的最后一步，按论文指标统计。"""
from __future__ import annotations

import numpy as np
import torch

from .config import Config
from .env import MergeEnv, expert_actions, neighbors
from .metrics import aggregate, trajectory_metrics


def rollout(env: MergeEnv, idx: int, policy, deterministic: bool, ecfg) -> dict:
    obs = env.reset(idx)
    if hasattr(policy, "reset"):
        policy.reset()
    actions = []
    while True:
        a = policy.act(obs, deterministic)[0]
        obs, term, trunc, info = env.step(a)
        actions.append(env.prev_act.copy())
        if term or trunc:
            break
    traj = env.trajectory()
    m = trajectory_metrics(traj, np.asarray(actions), env.leaders, env.scene.dt, ecfg)
    m.update(idx=idx, traj=traj, success=info["success"], collision=info["collision"])
    return m


def expert_results(scenes, cfg: Config) -> list[dict]:
    """观测数据本身的指标（论文表格中的 Observed 行）。"""
    out = []
    for i, sc in enumerate(scenes):
        leaders = [neighbors(sc.ego[k], sc.others[k], sc.meta)["l"] for k in range(len(sc.ego))]
        m = trajectory_metrics(sc.ego[:, :4], expert_actions(sc, cfg.env), leaders, sc.dt, cfg.eval)
        m.update(idx=i, traj=sc.ego[:, :4], success=True, collision=False)
        out.append(m)
    return out


def evaluate(scenes, cfg: Config, policy, use_goal: bool, n_runs: int | None = None,
             deterministic: bool = False, seed: int = 0) -> tuple[dict, list]:
    """论文：每个模型评估 5 次（随机策略采样）取平均。返回 (平均指标, 第一次评估的逐条结果)。"""
    n_runs = n_runs or cfg.eval.n_runs
    env = MergeEnv(scenes, cfg.env, use_goal=use_goal)
    runs, first = [], None
    for r in range(n_runs):
        torch.manual_seed(seed + r)
        rs = [rollout(env, i, policy, deterministic, cfg.eval) for i in range(len(scenes))]
        runs.append(aggregate(rs))
        first = first or rs
    mean = {}
    for k in runs[0]:
        vals = [m[k] for m in runs if not np.isnan(m[k])]
        mean[k] = float(np.mean(vals)) if vals else np.nan  # 某次评估没有成功轨迹时该次不计入
    return mean, first


def plot(results: dict, scenes, path: str, n_traj: int = 4):
    """results: {模型名: 逐条结果列表}。画论文图 4–7 风格的图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    # 轨迹：横向 vs 纵向（论文图 4 的方向）
    ax = axes[0, 0]
    obs = results[names[0]]
    for i, r in enumerate(obs[:n_traj]):
        line = scenes[r["idx"]].meta["lane_line_y"]
        ax.plot(r["traj"][:, 1] - line, r["traj"][:, 0], "-", color=f"C{i}", lw=2, alpha=0.6)
        for n in names[1:2]:
            rr = next((q for q in results[n] if q["idx"] == r["idx"]), None)
            if rr is not None:
                ax.plot(rr["traj"][:, 1] - line, rr["traj"][:, 0], "--", color=f"C{i}", lw=1.2)
    ax.axvline(0, color="gray", ls="--")
    ax.set(title=f"trajectories (solid = {names[0]}, dashed = {names[1] if len(names) > 1 else '-'})",
           xlabel="lateral position rel. lane line (m)", ylabel="longitudinal (m)")

    def collect(name, key):
        return np.concatenate([np.asarray(r[key], float) for r in results[name] if r["success"]] or [np.zeros(0)])

    for ax, key, title in ((axes[0, 1], "_v_all", "speed (m/s)"), (axes[0, 2], "_ttc_all", "TTC (s)"),
                           (axes[1, 0], "_hw_all", "time headway (s)"), (axes[1, 1], "_jerk_all", "jerk (m/s^3)")):
        data = [collect(n, key) for n in names]
        hi = max([d.max() for d in data if len(d)] + [1.0])
        bins = np.linspace(0, min(hi, 25 if key == "_v_all" else hi), 30)
        for n, d in zip(names, data):
            if len(d):
                ax.hist(d, bins, alpha=0.45, density=True, label=n)
        ax.set(title=title)
        ax.legend(fontsize=8)

    ax = axes[1, 2]
    rates = [(np.mean([r["success"] for r in results[n]]), np.mean([r["collision"] for r in results[n]]))
             for n in names]
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, [s for s, _ in rates], 0.4, label="success")
    ax.bar(xs + 0.2, [c for _, c in rates], 0.4, label="collision")
    ax.set_xticks(xs, names, rotation=30)
    ax.set(title="success / collision rate", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
