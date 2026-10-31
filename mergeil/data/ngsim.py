"""NGSIM 数据集适配器（US-101 / I-80），与论文的数据处理保持一致。

支持两种数据来源：
- 各地点数据包中的分时段文件，如 US-101-LosAngeles-CA.zip 里的
  vehicle-trajectory-data/0750am-0805am/trajectories-0750am-0805am.csv（或同名 .txt，无表头、空格分隔）。
  --path 可以是单个文件，也可以是包含这些文件的目录；一个文件就是一个时段；
- data.transportation.gov 上的合并版 CSV（含 Location 列），用 Global_Time 的大间隔切分时段。

论文的处理：
- 用 Kalman 滤波降低坐标测量误差，并计算速度；
- 只考虑从 lane 6 换到 lane 5 的车辆（两个地点都是如此）；
- US-101 用于训练和测试（164 条），I-80 用于迁移性测试（139 条）。

注意：
- 单位是英尺；Local_Y 是车头纵向位置，Local_X 是车辆横向中心到路段左边缘的距离；
- Vehicle_ID 在时段之间会重复，所以每个时段单独处理；
- I-80 的官方说明中 lane 7 才是匝道，lane 6 是主线最右侧车道。论文写的是两个地点都取 6→5，
  这里默认照论文；如需改成 7→6，传 lane_maps 参数覆盖即可。
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

from .common import compute_group_meta, kalman_smooth_df, straighten_lanes, validate

FT = 0.3048
DATA_DT = 0.1

# NGSIM Lane_ID -> 统一车道编码（0 = 加速车道，1 = 目标车道，向左递增，-1 = 忽略）
PAPER_LANE_MAP = {6: 0, 5: 1, 4: 2, 3: 3, 2: 4, 1: 5, 7: -1, 8: -1}
LANE_MAPS = {"us-101": PAPER_LANE_MAP, "i-80": PAPER_LANE_MAP}

# Kalman 平滑参数：q 为 jerk 噪声强度，r 为位置测量噪声标准差 (m)。论文未给出，可按需调整
KALMAN = {"q_x": 2.0, "r_x": 1.0, "q_y": 0.5, "r_y": 0.3}

USE = ["Vehicle_ID", "Global_Time", "Local_X", "Local_Y", "v_length", "v_Width", "Lane_ID"]
# 原始 .txt 的 18 列（无表头）
TXT_COLUMNS = ["Vehicle_ID", "Frame_ID", "Total_Frames", "Global_Time", "Local_X", "Local_Y",
               "Global_X", "Global_Y", "v_length", "v_Width", "v_Class", "v_Vel", "v_Acc",
               "Lane_ID", "Preceding", "Following", "Space_Headway", "Time_Headway"]


def _guess_location(path: str, locations) -> str:
    key = re.sub(r"[^a-z0-9]", "", path.lower())
    for loc in locations:
        if re.sub(r"[^a-z0-9]", "", loc) in key:
            return loc
    if len(locations) == 1:
        return locations[0]
    raise ValueError(f"无法从路径 {path} 判断地点，请只传一个 --locations")


def _canonical(df: pd.DataFrame) -> pd.DataFrame:
    """列名统一成 USE 中的写法（不同来源大小写不一，如 v_Length / v_length、API 导出的全小写列名）。"""
    names = {c.lower(): c for c in USE + ["Location"]}
    return df.rename(columns={c: names[c.lower()] for c in df.columns if c.lower() in names})


def _read_period_file(path: str) -> pd.DataFrame:
    if path.endswith(".txt"):
        df = pd.read_csv(path, sep=r"\s+", header=None, names=TXT_COLUMNS, usecols=range(18))
    else:
        df = _canonical(pd.read_csv(path))
    return df[USE].drop_duplicates(["Vehicle_ID", "Global_Time"])


def _periods(path: str, locations, period_gap_s: float):
    """依次给出 (地点, 时段名, 该时段的原始数据)。"""
    if os.path.isdir(path):
        # 优先用 .txt：官方包里的 .csv 被 Excel 截断在 1,048,576 行
        files = sorted(glob.glob(os.path.join(path, "**", "trajectories-*.txt"), recursive=True))
        files = files or sorted(glob.glob(os.path.join(path, "**", "trajectories-*.csv"), recursive=True))
        if not files:
            raise FileNotFoundError(f"{path} 下没有 trajectories-*.csv / .txt")
    else:
        files = [path]
    combined = len(files) == 1 and not files[0].endswith(".txt") and \
        "location" in [c.lower() for c in pd.read_csv(files[0], nrows=0).columns]
    if not combined:
        for f in files:
            loc = _guess_location(os.path.abspath(f), locations)
            name = re.sub(r"^trajectories-", "", os.path.splitext(os.path.basename(f))[0])
            yield loc, name, _read_period_file(f)
        return

    raw = _canonical(pd.read_csv(files[0], usecols=lambda c: c.lower() in {u.lower() for u in USE + ["Location"]}))
    raw = raw[raw.Location.str.lower().isin(locations)].drop_duplicates(["Location", "Vehicle_ID", "Global_Time"])
    if raw.empty:
        raise ValueError(f"CSV 中没有找到地点 {locations}")
    for loc, d in raw.groupby("Location"):
        times = np.sort(d.Global_Time.unique())
        cuts = np.where(np.diff(times) > period_gap_s * 1000)[0]
        bounds = np.concatenate([[times[0]], times[cuts + 1]])
        d = d.assign(period=np.searchsorted(bounds, d.Global_Time, side="right") - 1)
        for period, dp in d.groupby("period"):
            yield loc, f"p{period}", dp


def load(path: str, locations=("us-101",), lane_width: float = 3.66,
         period_gap_s: float = 60.0, lane_maps: dict | None = None) -> tuple[pd.DataFrame, dict]:
    lane_maps = lane_maps or LANE_MAPS
    frames, metas = [], {}
    for loc, period, dp in _periods(path, tuple(locations), period_gap_s):
        if loc not in locations:
            continue
        t0 = dp.Global_Time.min()
        group = f"{loc}_{period}"
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
        g = straighten_lanes(g)
        g = kalman_smooth_df(g, DATA_DT, **KALMAN)
        metas[group] = compute_group_meta(g, DATA_DT, lane_width)
        frames.append(g)
        print(f"[NGSIM] {group}: {g.vid.nunique()} 辆车")
    if not frames:
        raise ValueError(f"没有读到地点 {locations} 的数据")
    return validate(pd.concat(frames, ignore_index=True)), metas
