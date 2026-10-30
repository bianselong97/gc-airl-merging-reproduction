"""全局配置。默认值尽量按论文 Sun & Yang (2024) 设置，论文未给出的取常用值并在注释中标明。
命令行可用 --config xxx.json 覆盖部分字段，例如 {"adv": {"iterations": 500}}。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class DataConfig:
    env_dt: float = 0.1            # 论文：更新步长 0.1 s
    neighbor_range: float = 100.0  # 只保留纵向距离在此范围内的周围车辆 (m)
    lc_vy_threshold: float = 0.1   # 换道起止点检测的横向速度阈值 (m/s)；论文用小波变换，这里用阈值法近似
    pre_lc_s: float = 2.0          # 论文：换道执行前保留 2 s
    post_lc_s: float = 2.0         # 论文：换道执行后保留 2 s
    min_scene_steps: int = 20
    n_test: int = 40               # 论文：US-101 164 条中 40 条测试
    n_val: int = 12                # 从训练集中再留出一部分用于选模型（论文未说明）
    seed: int = 0


@dataclass
class EnvConfig:
    lane_width: float = 3.66       # NGSIM 车道宽约 12 ft
    goal_offset: float = 1.0       # 论文：中间目标 = 越过车道线 1 m
    ax_bounds: tuple = (-8.0, 4.0)   # 动作截断范围（论文未说明）
    ay_bounds: tuple = (-3.0, 3.0)
    reactive: bool = False         # 论文：周围车辆按真实轨迹回放；True 为本仓库的扩展
    react_range: float = 40.0
    idm: tuple = (33.0, 1.2, 2.0, 1.5, 2.0)


@dataclass
class AdvConfig:
    """对抗式模仿学习（GAIL / AIRL / G-AIRL / C-AIRL / GC-AIRL）和 RL 基线共用的 PPO 设置。"""
    pi_hidden: tuple = (128, 128)  # 论文：策略/价值网络 2 层 128，tanh
    d_hidden: tuple = (64, 64)     # 论文：奖励估计器 c 和势函数 h 均为 2 层 64
    iterations: int = 500
    steps_per_iter: int = 4096
    gamma: float = 0.99
    gae_lambda: float = 0.95       # 论文用 k 步回报的优势函数，这里用 GAE 近似
    ppo_epochs: int = 10
    minibatch: int = 256
    clip: float = 0.2              # 论文：PPO-clip ε = 0.2
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    max_grad_norm: float = 0.5
    init_log_std: float = -0.5
    d_lr: float = 3e-4
    d_steps: int = 5
    d_batch: int = 512
    grad_penalty: float = 1.0
    gail_reward: str = "logD"      # 论文：GAIL 以 log D 为奖励；可选 "-log(1-D)"
    airl_reward: str = "c"         # 论文 Eq.9：r = c(s,a,g) (+ r_goal)；可选 "f-logpi"
    goal_reward: float = 1.0       # 论文：到达中间目标奖励 1
    beta_lr: float = 1e-2          # 置信度参数的学习率（论文未给出）
    inner_lr: float = 1e-2         # 双层优化中内层一步更新的步长（论文未给出）
    rank_eps: float = 1e-5         # 论文 Eq.10 的 ε
    rank_trajs: int = 16           # 每次外层更新抽取的专家轨迹数
    score_weights: tuple = (1.0, 1.0, 5.0)  # 论文：排序得分权重 1-1-5
    eval_every: int = 10
    bc_init: str = ""


@dataclass
class LSTMConfig:
    hidden: int = 256              # 论文：2 层 256，dropout 0.2，MSE，RMSprop
    layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    epochs: int = 300
    batch_size: int = 16


@dataclass
class BCConfig:
    hidden: tuple = (128, 128)
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 512
    weight_decay: float = 1e-5


@dataclass
class EvalConfig:
    n_runs: int = 5                # 论文：每个模型评估 5 次取平均
    ttc_max: float = 20.0          # TTC 只统计 (0, 20] s（论文图 6 的范围，论文未明确说明）
    headway_max: float = 10.0      # 车头时距只统计 (0, 10] s（同上）


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    adv: AdvConfig = field(default_factory=AdvConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: str | None) -> "Config":
        cfg = cls()
        if not path:
            return cfg
        with open(path, encoding="utf-8") as f:
            overrides = json.load(f)
        for section, values in overrides.items():
            sub = getattr(cfg, section)
            for k, v in values.items():
                if not hasattr(sub, k):
                    raise KeyError(f"未知配置项 {section}.{k}")
                setattr(sub, k, tuple(v) if isinstance(getattr(sub, k), tuple) else v)
        return cfg
