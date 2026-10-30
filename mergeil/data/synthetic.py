"""合成的匝道汇入数据，用于在拿到 highD/NGSIM 之前把整条流程跑通。

路段：两条主线车道 + 一条长 ramp_len 的加速车道。
- 所有车辆纵向用 IDM 跟驰（前车 = 前方横向重叠的最近车辆），参数因人而异；
- 匝道车辆按间隙接受模型选择汇入时机，横向按余弦曲线完成换道；
- 主线车辆不换道，但会对切入的汇入车辆做出反应。

这只是“看起来像”的专家数据，不能代替真实数据得出研究结论。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import compute_group_meta, validate

LANE_W = 3.75
LANE_Y = np.array([0.0, LANE_W, 2 * LANE_W])  # 匝道、主线右、主线左


def _idm(v, v_lead, gap, p):
    s_star = p["s0"] + np.maximum(0.0, v * p["T"] + v * (v - v_lead) / (2 * np.sqrt(p["a"] * p["b"])))
    return p["a"] * (1 - (v / p["v0"]) ** 4 - (s_star / np.maximum(gap, 0.1)) ** 2)


def simulate_group(group: str, seed: int, duration: float = 300.0, dt: float = 0.1,
                   ramp_len: float = 250.0, main_headway: float = 2.2,
                   ramp_headway: float = 7.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x_start, x_end = -200.0, 600.0
    vehs: list[dict] = []
    rows = []
    next_id = 0
    next_spawn = {1: 0.0, 2: 0.0, 0: 3.0}

    def new_vehicle(lane, v):
        nonlocal next_id
        next_id += 1
        return {
            "id": next_id, "x": x_start if lane else 0.0, "y": LANE_Y[lane], "vx": v, "vy": 0.0,
            "len": rng.uniform(4.2, 5.2), "wid": rng.uniform(1.75, 2.0),
            "ramp": lane == 0, "merging": None,
            "p": {"v0": rng.uniform(26, 34), "T": rng.uniform(1.0, 1.8), "s0": rng.uniform(1.5, 3.0),
                  "a": rng.uniform(1.0, 2.0), "b": rng.uniform(1.5, 2.5)},
            "g_lead": rng.uniform(5, 15), "g_lag": rng.uniform(5, 15), "dur": rng.uniform(3.0, 5.0),
        }

    n_steps = int(duration / dt)
    for k in range(n_steps):
        t = k * dt
        # 生成车辆
        for lane in (1, 2, 0):
            if t < next_spawn[lane]:
                continue
            x0 = x_start if lane else 0.0
            same = [v for v in vehs if abs(v["y"] - LANE_Y[lane]) < 1.0 and v["x"] - x0 < 40]
            if same:
                continue
            v_init = rng.uniform(22, 28) if lane else rng.uniform(15, 20)
            vehs.append(new_vehicle(lane, v_init))
            hw = main_headway if lane else ramp_headway
            next_spawn[lane] = t + rng.exponential(hw) + 0.8

        if not vehs:
            continue
        X = np.array([v["x"] for v in vehs])
        Y = np.array([v["y"] for v in vehs])
        V = np.array([v["vx"] for v in vehs])
        L = np.array([v["len"] for v in vehs])

        acc = np.zeros(len(vehs))
        for i, v in enumerate(vehs):
            dx = X - X[i]
            ahead = (dx > 0) & (np.abs(Y - Y[i]) < 2.2)
            if v["merging"] is not None:  # 换道过程中同时跟随目标车道前车
                ahead |= (dx > 0) & (np.abs(Y - LANE_Y[1]) < 1.5)
            gap, v_lead = 500.0, v["vx"]
            if ahead.any():
                j = np.where(ahead)[0][np.argmin(dx[ahead])]
                gap, v_lead = dx[j] - 0.5 * (L[i] + L[j]), V[j]
            if v["ramp"] and v["merging"] is None:
                g_end = ramp_len - v["x"] - 0.5 * v["len"]
                if g_end < gap:
                    gap, v_lead = g_end, 0.0
            acc[i] = np.clip(_idm(v["vx"], v_lead, gap, v["p"]), -8.0, 3.0)

            # 间隙接受：决定是否开始汇入
            if v["ramp"] and v["merging"] is None and v["x"] > 20:
                in_tgt = np.abs(Y - LANE_Y[1]) < 1.5
                lead_gap, lag_gap, lag_v, lead_v = 500.0, 500.0, 0.0, v["vx"]
                fwd, back = in_tgt & (dx > 0), in_tgt & (dx <= 0)
                if fwd.any():
                    j = np.where(fwd)[0][np.argmin(dx[fwd])]
                    lead_gap, lead_v = dx[j] - 0.5 * (L[i] + L[j]), V[j]
                if back.any():
                    j = np.where(back)[0][np.argmax(dx[back])]
                    lag_gap, lag_v = -dx[j] - 0.5 * (L[i] + L[j]), V[j]
                urgency = np.clip((ramp_len - v["x"]) / ramp_len, 0.3, 1.0)
                need_lag = max(4.0, v["g_lag"] * urgency) + max(0.0, lag_v - v["vx"]) * 1.2
                need_lead = max(4.0, v["g_lead"] * urgency) + max(0.0, v["vx"] - lead_v) * 1.5
                if lead_gap > need_lead and lag_gap > need_lag:
                    v["merging"] = (t, v["y"], LANE_Y[1])

        for i, v in enumerate(vehs):
            v["vx"] = max(0.0, v["vx"] + acc[i] * dt)
            v["x"] += v["vx"] * dt
            if v["merging"] is not None:
                t0, y0, y1 = v["merging"]
                tau = min(1.0, (t + dt - t0) / v["dur"])
                y_new = y0 + (y1 - y0) * (1 - np.cos(np.pi * tau)) / 2
                v["vy"] = (y_new - v["y"]) / dt
                v["y"] = y_new
                if tau >= 1.0:
                    v["merging"], v["ramp"], v["vy"] = None, False, 0.0

        vehs = [v for v in vehs if v["x"] < x_end]
        for v in vehs:
            lane = int(np.argmin(np.abs(LANE_Y - v["y"])))
            if lane == 0 and v["x"] > ramp_len + 5:
                lane = -1
            rows.append((group, v["id"], k, v["x"], v["y"], v["vx"], v["vy"], v["len"], v["wid"], lane))

    return pd.DataFrame(rows, columns=["group", "vid", "frame", "x", "y", "vx", "vy",
                                       "length", "width", "lane"])


def generate(n_groups: int = 4, duration: float = 300.0, seed: int = 0,
             ramp_len: float = 250.0) -> tuple[pd.DataFrame, dict]:
    dfs, metas = [], {}
    for gi in range(n_groups):
        group = f"synth{gi}"
        df = simulate_group(group, seed + gi, duration=duration, ramp_len=ramp_len)
        metas[group] = compute_group_meta(df, 0.1, LANE_W, ramp_end_x=ramp_len)
        dfs.append(df)
        print(f"[synthetic] {group}: {df.vid.nunique()} 辆车")
    return validate(pd.concat(dfs, ignore_index=True)), metas
