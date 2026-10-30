"""统一的轨迹数据格式。

各数据集适配器（highD / NGSIM / 合成数据）都转换成同一种 DataFrame，后续模块只认这种格式：

    group   str    录像/路段编号，不同 group 之间的车辆互不可见
    vid     int    车辆编号（group 内唯一）
    frame   int    帧号（group 内统一时钟）
    x, y    float  车辆中心坐标 (m)。x 沿行驶方向为正，y 向左为正
    vx, vy  float  速度 (m/s)
    length, width  float  车长、车宽 (m)
    lane    int    车道编码：0 = 加速车道（汇入车辆的起始车道），1 = 目标车道，
                   2, 3, ... 依次向左，-1 = 其他/忽略

每个 group 另有一份元数据 dict（见 compute_group_meta）。

注意：论文中 x 为横向、y 为纵向（NGSIM 的 Local_X / Local_Y）；本仓库内部 x 为纵向、y 为横向，
仅为命名不同，状态向量的内容与论文一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

COLUMNS = ["group", "vid", "frame", "x", "y", "vx", "vy", "length", "width", "lane"]


def smooth_kinematics(df: pd.DataFrame, data_dt: float, window_s: float = 1.0,
                      recompute_velocity: bool = True) -> pd.DataFrame:
    """对每辆车的位置做 Savitzky-Golay 平滑，并（可选）由平滑后的位置重新求速度。"""
    out = []
    win = max(5, int(round(window_s / data_dt)) | 1)  # 奇数窗口
    for _, g in df.groupby(["group", "vid"], sort=False):
        g = g.sort_values("frame").copy()
        n = len(g)
        if n >= 5:
            w = min(win, n if n % 2 == 1 else n - 1)
            if w >= 5:
                g["x"] = savgol_filter(g["x"].to_numpy(), w, 2)
                g["y"] = savgol_filter(g["y"].to_numpy(), w, 2)
                if recompute_velocity:
                    g["vx"] = savgol_filter(g["x"].to_numpy(), w, 2, deriv=1, delta=data_dt)
                    g["vy"] = savgol_filter(g["y"].to_numpy(), w, 2, deriv=1, delta=data_dt)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def kalman_smooth_batch(z: np.ndarray, mask: np.ndarray, dt: float, q: float, r: float,
                        chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """常加速度模型的 Kalman 滤波 + RTS 平滑，批量处理多条轨迹。

    z: (n, T) 观测位置，每行一条左对齐的轨迹；mask: (n, T) 是否有观测。
    q: 加加速度（jerk）白噪声强度；r: 位置观测噪声标准差。
    返回平滑后的位置和速度，形状均为 (n, T)。
    """
    F = np.array([[1, dt, 0.5 * dt * dt], [0, 1, dt], [0, 0, 1]])
    Q = q * np.array([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                      [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                      [dt ** 3 / 6, dt ** 2 / 2, dt]])
    R = r * r
    pos = np.zeros_like(z, dtype=np.float64)
    vel = np.zeros_like(z, dtype=np.float64)
    for s in range(0, len(z), chunk):
        zc, mc = z[s:s + chunk], mask[s:s + chunk]
        n, T = zc.shape
        xf = np.zeros((n, T, 3))
        Pf = np.zeros((n, T, 3, 3))
        xp = np.zeros((n, T, 3))
        Pp = np.zeros((n, T, 3, 3))
        x = np.zeros((n, 3))
        x[:, 0] = zc[:, 0]
        P = np.tile(np.diag([R, 25.0, 9.0]), (n, 1, 1))
        for t in range(T):
            if t > 0:
                x = x @ F.T
                P = F @ P @ F.T + Q
            xp[:, t], Pp[:, t] = x, P
            m = mc[:, t]
            if m.any():
                S = P[m, 0, 0] + R
                K = P[m, :, 0] / S[:, None]
                innov = zc[m, t] - x[m, 0]
                x[m] = x[m] + K * innov[:, None]
                P[m] = P[m] - K[:, :, None] * P[m, 0, :][:, None, :]
            xf[:, t], Pf[:, t] = x, P
        xs = xf.copy()
        for t in range(T - 2, -1, -1):
            C = Pf[:, t] @ F.T @ np.linalg.inv(Pp[:, t + 1])
            xs[:, t] = xf[:, t] + np.einsum("nij,nj->ni", C, xs[:, t + 1] - xp[:, t + 1])
        pos[s:s + chunk], vel[s:s + chunk] = xs[:, :, 0], xs[:, :, 1]
    return pos, vel


def kalman_smooth_df(df: pd.DataFrame, data_dt: float, q_x: float, r_x: float,
                     q_y: float, r_y: float) -> pd.DataFrame:
    """对 DataFrame 中每辆车的 x、y 分别做 Kalman 平滑，并写回平滑位置和速度。"""
    df = df.sort_values(["group", "vid", "frame"]).reset_index(drop=True)
    key = df.groupby(["group", "vid"], sort=False).ngroup().to_numpy()
    start = df.groupby(key)["frame"].transform("min").to_numpy()
    off = df["frame"].to_numpy() - start
    n, T = key.max() + 1, off.max() + 1
    mask = np.zeros((n, T), bool)
    mask[key, off] = True
    for col, vcol, q, r in (("x", "vx", q_x, r_x), ("y", "vy", q_y, r_y)):
        z = np.zeros((n, T))
        z[key, off] = df[col].to_numpy()  # 缺帧处为 0，滤波时由 mask 跳过
        p, v = kalman_smooth_batch(z, mask, data_dt, q, r)
        df[col] = p[key, off]
        df[vcol] = v[key, off]
    return df


def compute_group_meta(df: pd.DataFrame, data_dt: float, lane_width: float,
                       ramp_end_x: float | None = None) -> dict:
    """由车道编码估计道路几何：车道线位置（加速车道与目标车道之间）等。"""
    ramp = df[df.lane == 0]
    target = df[df.lane == 1]
    if ramp.empty or target.empty:
        raise ValueError("该 group 缺少加速车道(lane=0)或目标车道(lane=1)的数据")
    meta = {
        "data_dt": data_dt,
        "lane_width": lane_width,
        "target_lane_y": float(target.y.median()),
        "ramp_lane_y": float(ramp.y.median()),
        "ramp_end_x": float(ramp_end_x if ramp_end_x is not None else np.percentile(ramp.x, 99.5)),
    }
    meta["lane_line_y"] = 0.5 * (meta["target_lane_y"] + meta["ramp_lane_y"])
    return meta


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少列: {missing}")
    df = df[COLUMNS].copy()
    df["group"] = df["group"].astype(str)
    df["vid"] = df["vid"].astype(int)
    df["frame"] = df["frame"].astype(int)
    df["lane"] = df["lane"].astype(int)
    return df.sort_values(["group", "frame", "vid"]).reset_index(drop=True)
