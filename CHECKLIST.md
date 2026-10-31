# 复现进度清单

## 你（用户）

- [x] 下载目标论文 PDF（放在上级目录，不进仓库）
- [x] 下载 AIRL 论文（2026，*AI for Transportation*），后续扩展用
- [x] 下载 NGSIM US-101（`../data/US-101-LosAngeles-CA.zip`，完整）
- [ ] **重新下载 `I-80-Emeryville-CA.zip`**：现在这个只有 82 MB，下载中断了，里面只有航拍图的前一部分，没有轨迹数据
- [x] 建 GitHub 仓库：https://github.com/bianselong97/gc-airl-merging-reproduction
- [ ] 确定目标：演示版还是完整复现加改进
- [ ] （不急）highD / exiD：论文没用，只在做额外迁移实验时才需要申请

## Claude

### 已完成
- [x] 按论文重构：16 维状态 + 目标、欧拉更新、成功判定（越线 1 m）、纯回放
- [x] NGSIM 适配器：Kalman 平滑、lane 6→5、按时段切分
- [x] 场景提取：换道起止点检测，前后各 2 s；train/val/test 划分（测试 40 条）
- [x] 模型：GC-AIRL、G-AIRL、C-AIRL、AIRL、GAIL、RL（Eq.20）、LSTM，以及额外的 BC
- [x] 论文指标：成功率、平均速度、速度标准差、TTC、车头时距、碰撞率、jerk；排序得分
- [x] 评估脚本：输出论文表 2 / 表 4 格式的对比表和图
- [x] 合成数据上的全流程冒烟测试

- [x] 合成数据上完整训练 8 个模型，排查并修复 3 个导致撞车的问题（生存偏置、碰撞无代价、归一化爆炸）
- [x] 结果和调试记录写进 README

### 拿到 NGSIM 后
- [x] NGSIM 适配器支持官方分时段文件（.csv / .txt）
- [x] 跑 US-101 数据处理：260 个场景（论文 164 条，差异见 README），车道线拉直、换道检测修正
- [x] 训练全部模型，得到表 2、表 3 的第一版结果（单种子）
- [x] 3 个随机种子：AIRL 0.767 ± 0.015，GC-AIRL 0.715 ± 0.045，GC-AIRL 的优势没有复现
- [x] 诊断 GC-AIRL：目标奖励被淹没，调到 30 后与 AIRL 打平；β 变平反而更差（README）
- [x] RL 基线：目标项权重扫描（README），调弱后不并道，保留原设置；差距仍未解决
- [ ] 表 4（I-80 迁移），等 I-80 数据

### 收尾
- [x] GitHub 仓库：git 初始化、首次提交、推送
- [x] NGSIM US-101 的复现结果和图写进 README
- [ ] 改进实验（例如用 Transformer 编码周围车辆，或对接 2026 年 AIRL 论文的可解释奖励分析）
- [ ] 给老师汇报的幻灯片
