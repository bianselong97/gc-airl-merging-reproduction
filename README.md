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

| 训练时的碰撞处理 | 未说明 | 对抗式模型：碰撞后进入吸收态直到专家轨迹结束；RL：碰撞即结束回合（见下文调试记录） |

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

### NGSIM（论文设定）

待补充。

### 合成数据（只用于验证代码）

合成数据由 IDM 加间隙接受规则生成，137 个汇入场景（85 训练 / 12 验证 / 40 测试），每个模型在测试集上评估 5 次取平均：

| 模型 | 成功率 | 平均速度 (m/s) | 速度标准差 (m/s) | TTC (s) | 车头时距 (s) | 碰撞率 | 纵向 jerk (m/s³) |
|---|---|---|---|---|---|---|---|
| Observed | 1.000 | 18.01 | 1.76 | 10.67 | 1.71 | 0.000 | 0.88 |
| GC-AIRL | 0.850 | 21.27 | 1.89 | 12.09 | 1.07 | 0.150 | 2.87 |
| G-AIRL | 0.935 | 19.71 | 1.78 | 10.08 | 1.24 | 0.050 | 3.73 |
| C-AIRL | 0.825 | 20.03 | 1.52 | 15.04 | 1.50 | 0.175 | 2.84 |
| AIRL | 0.930 | 19.01 | 1.56 | 8.67 | 1.52 | 0.060 | 5.53 |
| GAIL | 0.850 | 18.14 | 1.44 | 8.77 | 1.61 | 0.050 | 2.72 |
| RL | 0.725 | 18.98 | 0.55 | 10.65 | 1.82 | 0.200 | 0.39 |
| LSTM | 0.950 | 18.33 | 1.38 | 9.45 | 1.77 | 0.050 | 0.67 |
| BC | 0.405 | 19.71 | 1.68 | 9.11 | 1.49 | 0.080 | 4.16 |

![合成数据上的对比](docs/synthetic_comparison.png)

这些数字**不能和论文比较**：
- 合成的“专家”全部出自同一个规则模型，不存在论文要解决的次优示范，所以置信度机制（C）在这里没有用武之地；
- 场景少（测试 40 条），差距在几个百分点以内的排名不可靠；
- 随机策略评估时 jerk 主要来自动作采样噪声，偏高。

### 调试记录：碰撞处理与归一化

论文没有说明训练时如何处理碰撞。下面三个问题都会让模型学会撞车，记录在此：

| 问题 | 现象 | 处理 |
|---|---|---|
| 生存偏置：AIRL/GAIL 的奖励大多为负，碰撞立即结束回合反而能少拿负奖励 | AIRL 训练碰撞率 89% | 训练时回合长度固定，碰撞不提前结束 |
| 碰撞没有代价：固定长度后，车辆重叠还能继续行驶 | GC-AIRL 并道后追尾前车，平均速度 23.5 m/s（真实 18.0） | 碰撞后进入吸收态（观测恒为 0，参考 DAC, Kostrikov et al. 2019），由判别器学出碰撞的代价 |
| 归一化爆炸：目标横向位置近似常数，标准差下限 1e-3 把吸收态的 0 放大到约 −1900 | 带目标条件的模型成功率只有 14–49% | 标准差过小的特征不做缩放 |

另外做了一个对照：把目标从判别器输入中去掉（`adv.disc_goal=false`），结果和论文设定相近（成功率 0.855 对 0.845），所以默认保留论文设定。
