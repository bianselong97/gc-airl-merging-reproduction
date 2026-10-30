"""highD 数据集适配器 (https://levelxdata.com/highd-dataset/)。

highD 的坐标是图像坐标（y 向下），x/y 为包围框左上角，width 是车长、height 是车宽。
drivingDirection: 1 = 向左行驶（上半幅），2 = 向右行驶（下半幅）。

只有部分路段带加速车道。默认用启发式找匝道车道：某方向最右侧车道若只覆盖了
道路纵向长度的一部分（< coverage_ratio），就认为它是加速车道。也可以用 ramp_lane_ids 手动指定。
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from .common import compute_group_meta, smooth_kinematics, validate

DATA_DT = 1.0 / 25.0


def _lane_order(direction: int, lane_ids) -> list[int]:
    """返回某行驶方向上从右到左排序的 laneId。"""
    lane_ids = sorted(set(int(l) for l in lane_ids))
    # 向右行驶(2)：图像 y 越大越靠右，laneId 越大越靠右；向左行驶(1)相反
    return lane_ids[::-1] if direction == 2 else lane_ids


def load_recording(data_dir: str, rec_id: int, lane_width: float = 3.75,
                   coverage_ratio: float = 0.7,
                   ramp_lane_ids: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """读取一段 highD 录像，返回 (统一格式 DataFrame, {group: meta})。不含加速车道的方向会被跳过。"""
    p = os.path.join(data_dir, f"{rec_id:02d}_")
    tracks = pd.read_csv(p + "tracks.csv")
    tmeta = pd.read_csv(p + "tracksMeta.csv")
    tracks = tracks.merge(tmeta[["id", "drivingDirection"]], on="id")

    frames, metas = [], {}
    for direction in (1, 2):
        d = tracks[tracks.drivingDirection == direction].copy()
        if d.empty:
            continue
        xc = d.x + d.width / 2
        yc = d.y + d.height / 2
        sign = 1.0 if direction == 2 else -1.0
        d["X"] = sign * xc
        d["Y"] = -sign * yc
        d["VX"] = sign * d.xVelocity
        d["VY"] = -sign * d.yVelocity

        order = _lane_order(direction, d.laneId.unique())
        override = (ramp_lane_ids or {}).get(rec_id, {}).get(direction)
        if override is not None:
            ramp_lane = int(override)
        else:
            span = np.percentile(d.X, 99) - np.percentile(d.X, 1)
            r = d[d.laneId == order[0]].X
            r_span = np.percentile(r, 95) - np.percentile(r, 5) if len(r) > 50 else span
            ramp_lane = order[0] if r_span < coverage_ratio * span else None
        if ramp_lane is None:
            continue
        rest = [l for l in order if l != ramp_lane]
        code = {ramp_lane: 0, **{l: i + 1 for i, l in enumerate(rest)}}

        group = f"highD{rec_id:02d}_d{direction}"
        g = pd.DataFrame({
            "group": group, "vid": d.id, "frame": d.frame,
            "x": d.X, "y": d.Y, "vx": d.VX, "vy": d.VY,
            "length": d.width, "width": d.height,
            "lane": d.laneId.map(code).fillna(-1).astype(int),
        })
        g = smooth_kinematics(g, DATA_DT, window_s=0.6, recompute_velocity=False)
        metas[group] = compute_group_meta(g, DATA_DT, lane_width)
        frames.append(g)

    if not frames:
        return pd.DataFrame(), {}
    return validate(pd.concat(frames, ignore_index=True)), metas


def load_all(data_dir: str, **kw) -> tuple[pd.DataFrame, dict]:
    ids = sorted(int(os.path.basename(f)[:2]) for f in glob.glob(os.path.join(data_dir, "*_tracks.csv")))
    dfs, metas = [], {}
    for rid in ids:
        df, m = load_recording(data_dir, rid, **kw)
        if not df.empty:
            dfs.append(df)
            metas.update(m)
            print(f"[highD] 录像 {rid:02d}: 找到带加速车道的方向 {list(m)}")
    if not dfs:
        raise RuntimeError("没有找到带加速车道的 highD 录像，请检查数据目录或手动指定 ramp_lane_ids")
    return pd.concat(dfs, ignore_index=True), metas
