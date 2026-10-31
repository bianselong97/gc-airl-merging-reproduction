"""从统一格式轨迹中提取汇入场景（论文 4.2 节）。

一个场景 = 一辆从加速车道(lane 0)换到目标车道(lane 1)的车辆（ego），
时间窗口为 [换道开始 - 2 s, 换道结束 + 2 s]，再加上同一时段内周围车辆的轨迹（回放用）。

换道起止点：论文用小波变换检测横向运动的突变点。这里用横向速度阈值近似：
从越过车道线的时刻向前、向后找横向速度降到 lc_vy_threshold 以下的位置。
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DataConfig

STATE_COLS = ["x", "y", "vx", "vy", "length", "width"]


@dataclass
class Scene:
    group: str
    vid: int
    meta: dict
    ego: np.ndarray            # (T, 6): x, y, vx, vy, length, width
    others: list               # 长度 T，每项 (n_k, 7)：STATE_COLS + vid，不含 ego
    cross_step: int            # ego 中心越过车道线的步号
    dt: float
    extra: dict = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        return len(self.ego) - 1

    @property
    def goal(self) -> np.ndarray:
        """论文：目标 g = 专家轨迹的终点位置。"""
        return self.ego[-1, :2]


def detect_lane_change(vy: np.ndarray, cross: int, thr: float, gap: int = 0,
                       stop: np.ndarray | None = None) -> tuple[int, int]:
    """返回换道执行的 (开始, 结束) 下标。vy 为指向目标车道的横向速度。

    gap：横向速度短暂低于阈值不超过 gap 帧时视为同一次换道（拥堵中常见的“停一下再并”）；
    stop：布尔数组，结束点不越过第一个为 True 的帧（例如已开始换到下一条车道）。
    """
    moving = vy > thr
    i0 = cross
    while i0 > 0 and moving[max(0, i0 - 1 - gap):i0].any():
        i0 -= 1
    while i0 < cross and not moving[i0]:
        i0 += 1
    last = len(vy) - 1
    if stop is not None and stop[cross:].any():
        last = cross + int(np.argmax(stop[cross:]))
    i1 = cross
    while i1 < last and moving[i1 + 1:i1 + 2 + gap].any():
        i1 += 1
    while i1 > cross and not moving[i1]:
        i1 -= 1
    return i0, i1


def extract_scenes(df: pd.DataFrame, metas: dict, cfg: DataConfig) -> list[Scene]:
    from .env import collides

    scenes, n_overlap, n_short = [], 0, 0
    for group, g in df.groupby("group"):
        meta = metas[group]
        data_dt = meta["data_dt"]
        stride = max(1, int(round(cfg.env_dt / data_dt)))
        dt = stride * data_dt
        line = meta["lane_line_y"]
        g = g.sort_values(["vid", "frame"])
        by_frame = None

        for vid, ev in g.groupby("vid"):
            lanes = ev.lane.to_numpy()
            # 只要 lane 0 -> lane 1 的直接换道（论文：lane 6 -> lane 5）
            trans = np.where((lanes[:-1] == 0) & (lanes[1:] == 1))[0]
            if len(trans) == 0:
                continue
            ys = ev.y.to_numpy()
            after = np.where(ys > line)[0]
            after = after[after > max(0, trans[0] - int(3 / data_dt))]
            if len(after) == 0:
                continue
            cross = int(after[0])
            i0, i1 = detect_lane_change(ev.vy.to_numpy(), cross, cfg.lc_vy_threshold,
                                        gap=int(round(cfg.lc_gap_s / data_dt)),
                                        stop=ys > line + meta["lane_width"])
            far = np.where(ys[cross:cross + int(round(cfg.lc_max_after_cross_s / data_dt))]
                           >= line + cfg.lc_end_offset)[0]
            if len(far):
                i1 = max(i1, cross + int(far[0]))
            s = max(0, i0 - int(round(cfg.pre_lc_s / data_dt)))
            e = min(len(ev) - 1, i1 + int(round(cfg.post_lc_s / data_dt)))
            seg = ev.iloc[s:e + 1:stride]
            if len(seg) < cfg.min_scene_steps or ys[s] > line:
                continue
            if seg.y.max() < line + cfg.min_cross_m:
                n_short += 1  # 专家没有越过车道线足够远，成功判定无从谈起
                continue
            if (np.diff(seg.frame.to_numpy()) != stride).any():
                continue  # 轨迹有缺帧，跳过

            if by_frame is None:
                by_frame = {f: d for f, d in g.groupby("frame")}
            others = []
            for f, ex in zip(seg.frame, seg.x):
                d = by_frame[f]
                d = d[(d.vid != vid) & (np.abs(d.x - ex) < cfg.neighbor_range)]
                others.append(d[STATE_COLS + ["vid"]].to_numpy(np.float32))
            ego = seg[STATE_COLS].to_numpy(np.float32)
            if any(collides(a, o) for a, o in zip(ego, others)):
                n_overlap += 1  # 专家轨迹本身与他车包围框重叠（标注噪声等），剔除
                continue
            scenes.append(Scene(group=group, vid=int(vid), meta=meta, ego=ego, others=others,
                                cross_step=(cross - s) // stride, dt=dt,
                                extra={"lc_start": (i0 - s) // stride, "lc_end": (i1 - s) // stride}))
    if n_overlap:
        print(f"剔除 {n_overlap} 个专家轨迹与他车重叠的场景")
    if n_short:
        print(f"剔除 {n_short} 个专家没有越过车道线 {cfg.min_cross_m} m 的场景")
    return scenes


def split(scenes: list[Scene], n_test: int, n_val: int, seed: int):
    """随机划分 train / val / test。n_test 为测试场景数（论文：40）。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(scenes))
    n_test = min(n_test, len(scenes) // 3)
    test = [scenes[i] for i in idx[:n_test]]
    rest = idx[n_test:]
    n_val = min(n_val, len(rest) // 5)
    val = [scenes[i] for i in rest[:n_val]]
    train = [scenes[i] for i in rest[n_val:]]
    return train, val, test


def save(path: str, **splits):
    with open(path, "wb") as f:
        pickle.dump(splits, f)


def load(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
