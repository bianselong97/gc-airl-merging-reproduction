"""基于轨迹回放的匝道汇入环境（论文 3.1 节）。

状态（16 维，论文定义）：
  ego 4 维：纵向位置 px、相对车道线的横向位置 py、纵向速度 vx、横向速度 vy
  3 辆周围车 × 4 维：l = 加速车道前车，p = 目标车道前车，f = 目标车道后车，
           每辆为 (纵向净间距（扣除车长）, 横向距离, 纵向相对速度, 横向相对速度)，不存在时全为 0。
  越过车道线后，目标车道前车成为跟驰对象，只观测它（放在 l 槽位），p、f 置 0。
目标条件模型额外拼接目标 g = (gx, gy)（专家轨迹终点，横向同样相对车道线），共 18 维。

动作：二维加速度 (ax, ay)。车辆按论文 Eq.1 用欧拉法更新（位置用当前步速度）。
周围车辆按真实轨迹回放；cfg.reactive=True 时 ego 正后方的车改用 IDM（本仓库的扩展，论文没有）。

回合结束：碰撞（失败），或到达专家轨迹的最后一步。训练时可设 terminate_on_collision=False：
碰撞后进入吸收态（观测恒为全 0），一直持续到专家轨迹长度（参考 DAC, Kostrikov et al. 2019）。
这样回合长度固定，不会因负奖励产生“早撞早结束”的生存偏置；而专家数据中没有吸收态，
判别器会学到吸收态很差，从而对碰撞给出学习得到的惩罚。碰撞仍记为失败。
成功：越过车道线 goal_offset (1 m) 且全程未碰撞（论文 Eq.16）。
"""
from __future__ import annotations

import numpy as np

from .config import EnvConfig

SLOTS = ("l", "p", "f")
EGO_DIM = 4
SLOT_DIM = 4
BASE_DIM = EGO_DIM + len(SLOTS) * SLOT_DIM   # 16
GOAL_DIM = 2
ACT_DIM = 2


def obs_dim(use_goal: bool) -> int:
    return BASE_DIM + (GOAL_DIM if use_goal else 0)


def neighbors(ego: np.ndarray, others: np.ndarray, meta: dict) -> dict:
    """按位置（而非车辆编号）找 l / p / f 三辆周围车，返回 {槽位: (gap, dy, dvx, dvy) 或 None}。"""
    out = {s: None for s in SLOTS}
    if not len(others):
        return out
    x, y, vx, vy, L = ego[:5]
    line, w = meta["lane_line_y"], meta["lane_width"]
    oy = others[:, 1]
    in_tgt = (oy > line) & (oy < line + w)
    in_acc = (oy <= line) & (oy > line - w)
    dx = others[:, 0] - x
    half = 0.5 * (L + others[:, 4])

    def pick(mask, ahead):
        m = mask & ((dx > 0) if ahead else (dx <= 0))
        if not m.any():
            return None
        j = np.where(m)[0][np.argmin(np.abs(dx[m]))]
        gap = dx[j] - half[j] if ahead else dx[j] + half[j]
        return (float(gap), float(oy[j] - y), float(others[j, 2] - vx), float(others[j, 3] - vy))

    if y < line:
        out["l"], out["p"], out["f"] = pick(in_acc, True), pick(in_tgt, True), pick(in_tgt, False)
    else:
        out["l"] = pick(in_tgt, True)
    return out


def observe(ego: np.ndarray, others: np.ndarray, meta: dict, goal: np.ndarray | None = None) -> np.ndarray:
    line = meta["lane_line_y"]
    nb = neighbors(ego, others, meta)
    obs = np.zeros(obs_dim(goal is not None), np.float32)
    obs[:4] = (ego[0], ego[1] - line, ego[2], ego[3])
    for i, s in enumerate(SLOTS):
        if nb[s] is not None:
            obs[EGO_DIM + i * SLOT_DIM: EGO_DIM + (i + 1) * SLOT_DIM] = nb[s]
    if goal is not None:
        obs[BASE_DIM:] = (goal[0], goal[1] - line)
    return obs


def idm_accel(v, v_lead, gap, params) -> float:
    v0, T, s0, a, b = params
    s_star = s0 + max(0.0, v * T + v * (v - v_lead) / (2 * np.sqrt(a * b)))
    return a * (1 - (v / v0) ** 4 - (s_star / max(gap, 0.1)) ** 2)


def collides(ego: np.ndarray, others: np.ndarray) -> bool:
    if not len(others):
        return False
    ox = np.abs(others[:, 0] - ego[0]) < 0.5 * (ego[4] + others[:, 4])
    oy = np.abs(others[:, 1] - ego[1]) < 0.5 * (ego[5] + others[:, 5])
    return bool((ox & oy).any())


def expert_actions(scene, cfg: EnvConfig) -> np.ndarray:
    """由相邻两步速度差反推专家动作，与环境的欧拉积分一致。"""
    v = scene.ego[:, 2:4]
    a = (v[1:] - v[:-1]) / scene.dt
    a[:, 0] = np.clip(a[:, 0], *cfg.ax_bounds)
    a[:, 1] = np.clip(a[:, 1], *cfg.ay_bounds)
    return a.astype(np.float32)


def expert_transitions(scenes, cfg: EnvConfig, use_goal: bool):
    """把专家轨迹转成转移样本。返回 dict：obs, act, next_obs, done, traj（所属轨迹编号）, goal_now。"""
    out = {k: [] for k in ("obs", "act", "next_obs", "done", "traj", "goal_now")}
    for i, sc in enumerate(scenes):
        a = expert_actions(sc, cfg)
        g = sc.goal if use_goal else None
        obs = [observe(sc.ego[k], sc.others[k], sc.meta, g) for k in range(len(sc.ego))]
        target = sc.meta["lane_line_y"] + cfg.goal_offset
        reached = sc.ego[:, 1] >= target
        first = int(np.argmax(reached)) if reached.any() else -1
        for k in range(sc.horizon):
            out["obs"].append(obs[k])
            out["act"].append(a[k])
            out["next_obs"].append(obs[k + 1])
            out["done"].append(0.0)
            out["traj"].append(i)
            out["goal_now"].append(float(k + 1 == first))
    return {k: np.asarray(v, np.float32 if k != "traj" else np.int64) for k, v in out.items()}


class MergeEnv:
    def __init__(self, scenes, cfg: EnvConfig, use_goal: bool = False, seed: int = 0,
                 terminate_on_collision: bool = True):
        self.scenes = scenes
        self.cfg = cfg
        self.use_goal = use_goal
        self.terminate_on_collision = terminate_on_collision
        self.rng = np.random.default_rng(seed)
        self.low = np.array([cfg.ax_bounds[0], cfg.ay_bounds[0]], np.float32)
        self.high = np.array([cfg.ax_bounds[1], cfg.ay_bounds[1]], np.float32)

    def reset(self, idx: int | None = None) -> np.ndarray:
        self.idx = int(self.rng.integers(len(self.scenes))) if idx is None else idx
        self.scene = self.scenes[self.idx]
        self.k = 0
        self.state = self.scene.ego[0].copy()
        self.prev_act = np.zeros(2, np.float32)
        self.others = self.scene.others[0]
        self.reactive = {}
        self.reached = False
        self.collided = False
        self.traj = [self.state[:4].copy()]
        self.leaders = [neighbors(self.state, self.others, self.scene.meta)["l"]]
        return self._obs()

    def _obs(self) -> np.ndarray:
        if self.collided and not self.terminate_on_collision:
            return np.zeros(obs_dim(self.use_goal), np.float32)  # 吸收态
        return observe(self.state, self.others, self.scene.meta, self.scene.goal if self.use_goal else None)

    def _advance_others(self, ego_prev: np.ndarray) -> np.ndarray:
        """取下一帧回放车辆；已激活的反应车辆用 IDM 推进纵向状态，并检查是否有新的后车需要激活。"""
        dt = self.scene.dt
        nxt = self.scene.others[self.k].copy()
        if not self.cfg.reactive:
            return nxt
        prev = self.others
        updated = {}
        for vid, (x, v) in self.reactive.items():
            rows = np.where(nxt[:, 6] == vid)[0]
            i_prev = np.where(prev[:, 6] == vid)[0]
            if not len(rows) or not len(i_prev):
                continue  # 车辆已离开回放范围
            me = prev[i_prev[0]]
            cand = np.vstack([np.delete(prev, i_prev[0], 0)[:, :6], ego_prev[None, :6]])
            dx = cand[:, 0] - x
            lat = np.abs(cand[:, 1] - me[1]) < 0.5 * (cand[:, 5] + me[5]) + 0.5
            m = (dx > 0) & lat
            if m.any():
                j = np.where(m)[0][np.argmin(dx[m])]
                acc = idm_accel(v, cand[j, 2], dx[j] - 0.5 * (cand[j, 4] + me[4]), self.cfg.idm)
            else:
                acc = idm_accel(v, v, 1e3, self.cfg.idm)
            v_new = max(0.0, v + float(np.clip(acc, -9.0, 4.0)) * dt)
            x_new = x + 0.5 * (v + v_new) * dt
            if nxt[rows[0], 0] < x_new:  # 真实司机更保守时沿用回放，反应车只在需要避让时偏离回放
                x_new, v_new = float(nxt[rows[0], 0]), float(nxt[rows[0], 2])
            nxt[rows[0], 0], nxt[rows[0], 2] = x_new, v_new
            updated[vid] = (x_new, v_new)
        self.reactive = updated

        e = self.state
        dx = nxt[:, 0] - e[0]
        lat = np.abs(nxt[:, 1] - e[1]) < 0.5 * (nxt[:, 5] + e[5]) + 0.5
        m = (dx < 0) & (dx > -self.cfg.react_range) & lat
        if m.any():
            j = np.where(m)[0][np.argmax(dx[m])]
            vid = float(nxt[j, 6])
            if vid not in self.reactive:
                self.reactive[vid] = (float(nxt[j, 0]), float(nxt[j, 2]))
        return nxt

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), self.low, self.high)
        dt = self.scene.dt
        if self.collided:  # 吸收态：车辆不再运动，只推进时间
            self.k += 1
            self.prev_act = a
            truncated = self.k >= self.scene.horizon
            return self._obs(), False, truncated, {"collision": True, "goal_now": False, "success": False}
        ego_prev = self.state.copy()
        # 论文 Eq.1：v' = v + a·dt，p' = p + v·dt
        self.state[0:2] += self.state[2:4] * dt
        self.state[2:4] += a * dt
        self.state[2] = max(0.0, self.state[2])
        self.prev_act = a
        self.k += 1
        self.others = self._advance_others(ego_prev)
        self.traj.append(self.state[:4].copy())
        self.leaders.append(neighbors(self.state, self.others, self.scene.meta)["l"])

        meta = self.scene.meta
        goal_now = (not self.reached) and self.state[1] >= meta["lane_line_y"] + self.cfg.goal_offset
        self.reached |= goal_now
        self.collided |= collides(self.state, self.others)
        terminated = self.collided and self.terminate_on_collision
        truncated = (not terminated) and self.k >= self.scene.horizon
        info = {"collision": self.collided, "goal_now": bool(goal_now),
                "success": bool(truncated and self.reached and not self.collided)}
        return self._obs(), terminated, truncated, info

    def trajectory(self) -> np.ndarray:
        return np.asarray(self.traj)
