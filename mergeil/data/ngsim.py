"""NGSIM 数据集适配器（US-101 / I-80），与论文的数据处理保持一致。

数据来源：data.transportation.gov 上的
"Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data"（合并版 CSV，含 Location 列）。

论文的处理：
- 用 Kalman 滤波降低坐标测量误差，并计算速度；
- 只考虑从 lane 6 换到 lane 5 的车辆（两个地点都是如此）；
- US-101 用于训练和测试（164 条），I-80 用于迁移性测试（139 条）。

注意：
- 单位是英尺；Local_Y 是车头纵向位置，Local_X 是车辆横向中心到路段左边缘的距离；
- 每个地点有 3 个时段，Vehicle_ID 在时段之间会重复，因此用 Global_Time 的大间隔切分时段；
- I-80 的官方说明中 lane 7 才是匝道，lane 6 是主线最右侧车道。论文写的是两个地点都取 6→5，
  这里默认照论文；如需改成 7→6，传 lane_maps 参数覆盖即可。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import compute_group_meta, kalman_smooth_df, validate

FT = 0.3048
DATA_DT = 0.1

# NGSIM Lane_ID -> 统一车道编码（0 = 加速车道，1 = 目标车道，向左递增，-1 = 忽略）
PAPER_LANE_MAP = {6: 0, 5: 1, 4: 2, 3: 3, 2: 4, 1: 5, 7: -1, 8: -1}
LANE_MAPS = {"us-101": PAPER_LANE_MAP, "i-80": PAPER_LANE_MAP}

# Kalman 平滑参数：q 为 jerk 噪声强度，r 为位置测量噪声标准差 (m)。论文未给出，可按需调整
KALMAN = {"q_x": 2.0, "r_x": 1.0, "q_y": 0.5, "r_y": 0.3}


def load(csv_path: str, locations=("us-101",), lane_width: float = 3.66,
         period_gap_s: float = 60.0, lane_maps: dict | None = None) -> tuple[pd.DataFrame, dict]:
    lane_maps = lane_maps or LANE_MAPS
    use = ["Vehicle_ID", "Global_Time", "Local_X", "Local_Y", "v_length", "v_Width",
           "Lane_ID", "Location"]
    raw = pd.read_csv(csv_path, usecols=use)
    raw = raw[raw.Location.isin(locations)].drop_duplicates(["Location", "Vehicle_ID", "Global_Time"])
    if raw.empty:
        raise ValueError(f"CSV 中没有找到地点 {locations}，Location 列的取值有：{raw.Location.unique()}")

    frames, metas = [], {}
    for loc, d in raw.groupby("Location"):
        times = np.sort(d.Global_Time.unique())
        cuts = np.where(np.diff(times) > period_gap_s * 1000)[0]
        bounds = np.concatenate([[times[0]], times[cuts + 1]])
        d = d.assign(period=np.searchsorted(bounds, d.Global_Time, side="right") - 1)
        for period, dp in d.groupby("period"):
            t0 = dp.Global_Time.min()
            group = f"{loc}_p{period}"
            g = pd.DataFrame({
                "group": group,
                "vid": dp.Vehicle_ID,
                "frame": ((dp.Global_Time - t0) / (DATA_DT * 1000)).round().astype(int),
                "x": (dp.Local_Y - dp.v_length / 2) * FT,
                "y": -dp.Local_X * FT,
                "vx": 0.0, "vy": 0.0,
                "length": dp.v_length * FT, "width": dp.v_Width * FT,
                "lane": dp.Lane_ID.map(lane_maps[loc]).fillna(-1).astype(int),
            })
            g = kalman_smooth_df(g, DATA_DT, **KALMAN)
            metas[group] = compute_group_meta(g, DATA_DT, lane_width)
            frames.append(g)
            print(f"[NGSIM] {group}: {g.vid.nunique()} 辆车")
    return validate(pd.concat(frames, ignore_index=True)), metas
