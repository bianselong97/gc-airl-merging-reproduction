# GC-AIRL 匝道汇入行为复现

非官方复现：Jie Sun, Hai Yang (2024). **Learning two-dimensional merging behaviour from vehicle trajectories with imitation learning.** *Transportation Research Part C* 160, 104530. [doi:10.1016/j.trc.2024.104530](https://doi.org/10.1016/j.trc.2024.104530)

论文从 NGSIM 真实轨迹中学习匝道汇入车辆的二维（纵向 + 横向）驾驶行为。在 AIRL 的基础上加入两种机制，提出 **GC-AIRL**：
- **目标条件（goal-conditioned）**：把目的地 g 拼进状态，并在车辆越过车道线 1 m 时给予额外奖励；
- **置信度加权（confidence-aware）**：每条专家轨迹分配一个置信度 β，用轨迹排序损失做双层优化，降低次优人类示范的影响。

## 实现内容

| 模型 | 说明 | 代码 |
|---|---|---|
| **GC-AIRL** | 论文主模型 | `mergeil/adversarial.py` (`gcairl`) |
| G-AIRL / C-AIRL | 消融（论文表 3） | `gairl` / `cairl` |
| AIRL | 基线 | `airl` |
| GAIL | 基线，奖励 log D | `gail` |
| RL | 基线，手工奖励（论文 Eq.20）+ PPO | `rl` |
| LSTM | 基线，2 层 256，dropout 0.2，MSE，RMSprop | `mergeil/lstm.py` |
| BC | 额外基线（论文没有） | `mergeil/bc.py` |

论文与代码的对应关系：

| 论文 | 代码 |
|---|---|
| 3.1 状态（16 维：ego + l/p/f 三辆车）、动作（ax, ay）、Eq.1 欧拉更新 | `mergeil/env.py` |
| 3.3.1 生成器：高斯策略 + 价值网络，2×128 tanh，PPO-clip ε=0.2 | `mergeil/policy.py`, `mergeil/ppo.py` |
| 3.3.2 判别器：f = c(s,a,g) + γh(s′) − h(s)，c、h 均 2×64；奖励 r = c + r_goal | `mergeil/adversarial.py` |
| 3.3.3 置信度估计器：排序损失 RK（Eq.10），双层优化（Eq.11–13） | `mergeil/adversarial.py` |
| 4.2 数据处理：Kalman 滤波、lane 6→5、换道前后各 2 s | `mergeil/data/ngsim.py`, `mergeil/scenes.py` |
| 5.1 指标：成功率、平均速度、速度标准差、TTC、车头时距、碰撞率、jerk；排序得分 | `mergeil/metrics.py` |

## 运行

依赖：`numpy pandas scipy torch matplotlib`（`pip install -r requirements.txt`）。网络很小，CPU 即可。

```bash
# 1. 数据
python scripts/prepare_data.py --source ngsim --path <NGSIM.csv> --locations us-101 --out outputs/scenes.pkl
python scripts/prepare_data.py --source ngsim --path <NGSIM.csv> --locations i-80 --all-test --out outputs/i80.pkl
#    没有数据时可先用合成数据跑通流程：
python scripts/prepare_data.py --source synthetic

# 2. 训练（每个模型一条命令）
python scripts/train.py --model gcairl
python scripts/train.py --model airl     # gairl / cairl / gail / rl / lstm / bc 同理

# 3. 评估：论文表 2（US-101 测试集）和表 4（I-80 迁移性）
python scripts/evaluate.py --models outputs/gcairl/best.pt outputs/airl/best.pt outputs/gail/best.pt outputs/rl/best.pt outputs/lstm/best.pt
python scripts/evaluate.py --scenes outputs/i80.pkl --models outputs/gcairl/best.pt ...
```

超参数在 `mergeil/config.py`，可以写一个 JSON 只覆盖部分字段（如 `{"adv": {"iterations": 1000}}`），再用 `--config` 传入。

## 数据

论文只用了 NGSIM：US-101 共 164 条汇入轨迹（124 训练 / 40 测试），I-80 共 139 条（每次随机抽 40 条做迁移性测试）。
NGSIM 是公开数据，在 data.transportation.gov 搜索 “Next Generation Simulation (NGSIM) Vehicle Trajectories” 导出 CSV。数据不随仓库分发。

## 与论文的差异和假设

论文没有给出的细节，这里按下表处理。都可以在 `config.py` 中调整：

| 项目 | 论文 | 本仓库 |
|---|---|---|
| 换道起止点检测 | 小波变换 | 横向速度阈值（0.1 m/s） |
| Kalman 滤波参数 | 未给出 | 常加速度模型，参数见 `ngsim.py` 的 `KALMAN` |
| I-80 车道 | 论文写 lane 6→5 | 照论文；NGSIM 官方说明中 I-80 的 lane 7 才是匝道，可通过 `lane_maps` 改为 7→6 |
| 优势估计 | k 步回报 | GAE（λ=0.95） |
| 置信度粒度、外层学习率 | 未给出 | 每条专家轨迹一个 β = n·softmax(θ)，内外层学习率见配置 |
| TTC / 车头时距的统计范围 | 未说明 | 只统计 (0, 20] s 和 (0, 10] s（与论文图 6 的横轴一致） |
| 训练迭代次数、批大小 | 未给出 | 500 轮 × 4096 步 |
| 输入归一化 | 位置 [0,1]，其他 [-1,1] | 按专家数据做 z-score |
| 模型选择 | 未说明 | 从训练集留出 12 条做验证集，按“成功率 − 碰撞率”选最优 |
| 状态中的纵向位置 | 绝对位置 | 绝对位置（NGSIM Local_Y），横向位置相对车道线 |

扩展（论文没有，默认关闭）：`env.reactive=True` 时，ego 正后方的回放车会改用 IDM 跟驰。

## 目录

```
mergeil/
  data/        数据适配器：ngsim（论文用）、highd、synthetic（调试用）
  scenes.py    汇入场景提取与划分
  env.py       回放环境、状态构造
  ppo.py       PPO
  adversarial.py  GAIL / AIRL / G-AIRL / C-AIRL / GC-AIRL / RL
  lstm.py, bc.py  监督学习基线
  metrics.py, evaluate.py  论文指标与评估
scripts/       prepare_data / train / evaluate
```

## 结果

待用 NGSIM 数据训练后补充。合成数据（`--source synthetic`）只用于验证代码，不能代表论文结果。
