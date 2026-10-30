"""论文 5.1 节的评价指标，以及置信度估计器使用的轨迹排序得分。

单条轨迹（只看纵向，论文认为纵向行为对交通流影响更大）：
  平均速度 v̄、速度标准差 σ、平均 TTC、平均车头时距、平均纵向 jerk。
TTC 和车头时距都相对前车 l 计算（Eq.17–18）。TTC 只在逼近前车时有定义；
论文未说明如何处理无穷大和极端值，这里只统计 (0, ttc_max] 与 (0, headway_max] 内的值（与论文图 6 的横轴范围一致）。
汇总：成功率、碰撞率按全部轨迹计算；其余指标只对成功的轨迹取平均（论文 5.2 节）。
"""
from __future__ import annotations

import numpy as np

from .config import EvalConfig

KEYS = ["success_rate", "avg_speed", "speed_std", "ttc", "headway", "collision_rate", "jerk"]
LABELS = {"success_rate": "Merging success", "avg_speed": "Avg speed (m/s)",
          "speed_std": "Speed dev (m/s)", "ttc": "TTC (s)", "headway": "Headway (s)",
          "collision_rate": "Collision rate", "jerk": "Long. jerk (m/s^3)"}


def trajectory_metrics(traj: np.ndarray, actions: np.ndarray, leaders: list, dt: float,
                       cfg: EvalConfig) -> dict:
    """traj: (T, 4) x, y, vx, vy；actions: (T-1, 2)；leaders: 长度 T，每步前车 (gap, dy, dvx, dvy) 或 None。"""
    v = traj[:, 2]
    ttc, hw = [], []
    for k, ld in enumerate(leaders):
        if ld is None:
            continue
        gap, _, dvx, _ = ld
        if gap <= 0:
            continue
        if dvx < 0:  # 前车更慢，正在逼近
            t = gap / -dvx
            if t <= cfg.ttc_max:
                ttc.append(t)
        if v[k] > 0.1:
            h = gap / v[k]
            if h <= cfg.headway_max:
                hw.append(h)
    jerk = np.abs(np.diff(actions[:, 0])) / dt if len(actions) > 1 else np.zeros(1)
    return {
        "avg_speed": float(v.mean()), "speed_std": float(v.std()),
        "ttc": float(np.mean(ttc)) if ttc else np.nan,
        "headway": float(np.mean(hw)) if hw else np.nan,
        "jerk": float(jerk.mean()),
        "_ttc_sum": float(np.sum(ttc)), "_jerk_sum": float(jerk.sum()), "_n": len(traj),
        "_ttc_all": ttc, "_hw_all": hw, "_jerk_all": jerk.tolist(), "_v_all": v.tolist(),
    }


def ranking_score(m: dict, weights=(1.0, 1.0, 5.0)) -> float:
    """论文：score = (v̄ - σ) + ΣTTC/N - 5·ΣJerk/N。"""
    a1, a2, a3 = weights
    return a1 * (m["avg_speed"] - m["speed_std"]) + a2 * m["_ttc_sum"] / m["_n"] - a3 * m["_jerk_sum"] / m["_n"]


def aggregate(results: list[dict]) -> dict:
    succ = [r for r in results if r["success"]]
    out = {"n": len(results),
           "success_rate": float(np.mean([r["success"] for r in results])),
           "collision_rate": float(np.mean([r["collision"] for r in results]))}
    for k in ("avg_speed", "speed_std", "ttc", "headway", "jerk"):
        vals = [r[k] for r in succ if not np.isnan(r[k])]
        out[k] = float(np.mean(vals)) if vals else np.nan
    return out


def format_table(rows: dict) -> str:
    head = f"{'':<12}" + "".join(f"{LABELS[k]:>20}" for k in KEYS)
    lines = [head]
    for name, m in rows.items():
        lines.append(f"{name:<12}" + "".join(f"{m[k]:>20.3f}" for k in KEYS))
    return "\n".join(lines)
