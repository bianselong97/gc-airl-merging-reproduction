"""汇入过程的俯视动画：同一场景下每个模型各跑一次闭环仿真，上下并排；虚线框为真实司机（专家）的位置。

    # 先看每个测试场景各模型的结果，挑场景
    python scripts/animate.py --scenes outputs/us101.pkl --models A/best.pt B/best.pt --summary
    # 生成 GIF
    python scripts/animate.py --scenes outputs/us101.pkl --models A/best.pt B/best.pt --scene 7 --out docs/merge_7.gif

默认用策略均值（确定性动作），--stochastic 改为随机采样。
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mergeil import scenes as S  # noqa: E402
from mergeil.checkpoints import load_policy  # noqa: E402
from mergeil.config import Config  # noqa: E402
from mergeil.env import MergeEnv  # noqa: E402

NAMES = {"gcairl": "GC-AIRL", "gairl": "G-AIRL", "cairl": "C-AIRL", "airl": "AIRL", "gail": "GAIL",
         "rl": "RL", "lstm": "LSTM", "bc": "BC"}
COLORS = ["#2A78D6", "#EB6834", "#1BAF7A", "#8E5BD0"]  # 依次给各模型
ROAD, MARK, EDGE, OTHER, EXPERT, CRASH = "#E4E7EB", "#FFFFFF", "#5B6573", "#9AA3AD", "#1B2733", "#C62828"
HOLD_S = 1.5  # 结束后停留的时长 (s)


def snapshot(env: MergeEnv) -> dict:
    return dict(k=env.k, ego=env.state.copy(), others=env.others.copy(), reached=env.reached,
                collided=env.collided)


def record(env: MergeEnv, idx: int, pi, deterministic: bool, seed: int):
    """跑一次闭环仿真，返回每一步的快照和结果（成功 / 碰撞 / 未汇入）。"""
    torch.manual_seed(seed)
    obs = env.reset(idx)
    if hasattr(pi, "reset"):
        pi.reset()
    frames = [snapshot(env)]
    while True:
        a = pi.act(obs, deterministic)[0]
        obs, term, trunc, info = env.step(a)
        frames.append(snapshot(env))
        if term or trunc:
            break
    result = "成功" if info["success"] else ("碰撞" if info["collision"] else "未汇入")
    return frames, result


def draw(ax, sc, frames, t, color, label, result, goal_offset):
    from matplotlib.patches import Rectangle

    fr = frames[min(t, len(frames) - 1)]
    done = t >= len(frames) - 1
    meta = sc.meta
    line, w, ramp_end = meta["lane_line_y"], meta["lane_width"], meta["ramp_end_x"]
    ex = fr["ego"][0]
    x0, x1, y0, y1 = ex - 45, ex + 75, line - 1.3 * w, line + 3 * w
    ax.clear()
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")

    # 道路：上方为主线（目标车道及其左侧车道），下方为加速车道，到 ramp_end 结束
    ax.add_patch(Rectangle((x0, line), x1 - x0, 3 * w, color=ROAD, zorder=0))
    aux_end = min(ramp_end, x1)
    if aux_end > x0:
        ax.add_patch(Rectangle((x0, line - w), aux_end - x0, w, color=ROAD, zorder=0))
        ax.plot([x0, aux_end], [line - w] * 2, color=EDGE, lw=2)
        ax.plot([x0, aux_end], [line] * 2, color=MARK, lw=2, ls=(0, (6, 6)))
    if ramp_end < x1:
        ax.plot([max(ramp_end, x0), x1], [line] * 2, color=EDGE, lw=2)
    for k in (1, 2):
        ax.plot([x0, x1], [line + k * w] * 2, color=MARK, lw=1.5, ls=(0, (6, 6)))
    ax.plot([x0, x1], [line + goal_offset] * 2, color=EDGE, lw=0.8, ls=":")

    def car(s, **kw):
        ax.add_patch(Rectangle((s[0] - s[4] / 2, s[1] - s[5] / 2), s[4], s[5], **kw))

    for o in fr["others"]:
        if x0 - 5 < o[0] < x1 + 5 and line - w < o[1] < line + 3 * w:  # 只画画面内的几条车道
            car(o, fc=OTHER, ec="none", zorder=2)
    k = min(fr["k"], len(sc.ego) - 1)
    car(sc.ego[k], fc="none", ec=EXPERT, lw=1.5, ls="--", zorder=3)
    trail = np.array([f["ego"][:2] for f in frames[:t + 1]])
    ax.plot(trail[:, 0], trail[:, 1], color=color, lw=1.5, alpha=0.6, zorder=3)
    car(fr["ego"], fc=color, ec=CRASH if fr["collided"] else "none", lw=2.5, zorder=4)

    if done:
        status = {"成功": "成功", "碰撞": "失败：碰撞", "未汇入": "失败：没有汇入"}[result]
    else:
        status = "已越线 1 m" if fr["reached"] else "汇入中"
    ax.set_title(f"{label}    t = {fr['k'] * sc.dt:4.1f} s    v = {fr['ego'][2]:4.1f} m/s",
                 loc="left", fontsize=12)
    ax.set_title(status, loc="right", fontsize=12, fontweight="bold",
                 color=CRASH if status.startswith("失败") else "#1B2733")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="outputs/scenes.pkl")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=[], help="各模型在画面上的名字，默认取模型类型")
    ap.add_argument("--scene", type=int, default=0, help="场景在该划分中的序号")
    ap.add_argument("--summary", action="store_true", help="只打印每个场景各模型的结果，不画图")
    ap.add_argument("--stochastic", action="store_true", help="随机采样动作（默认用策略均值）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--dpi", type=int, default=72)
    ap.add_argument("--config", default="")
    ap.add_argument("--out", default="outputs/merge.gif")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    scenes = S.load(args.scenes)[args.split]
    models = [load_policy(p) for p in args.models]
    labels = args.labels or [NAMES.get(m, m) for _, _, m in models]
    envs = [MergeEnv(scenes, cfg.env, use_goal=g) for _, g, _ in models]
    det = not args.stochastic

    if args.summary:
        print("场景\t" + "\t".join(labels) + "\t时长 (s)")
        for i in range(len(scenes)):
            res = [record(e, i, pi, det, args.seed)[1] for e, (pi, _, _) in zip(envs, models)]
            print(f"{i}\t" + "\t".join(res) + f"\t{scenes[i].horizon * scenes[i].dt:.1f}")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]

    sc = scenes[args.scene]
    runs = [record(e, args.scene, pi, det, args.seed) for e, (pi, _, _) in zip(envs, models)]
    n = max(len(f) for f, _ in runs) + int(HOLD_S * args.fps)
    fig, axes = plt.subplots(len(runs), 1, figsize=(12, 1.9 * len(runs) + 0.5), squeeze=False)
    fig.text(0.01, 0.02, "彩色：模型控制的车    虚线框：真实司机    灰色：周围车辆（按真实轨迹回放）    点线：越线 1 m 的成功判定线",
             fontsize=11, color=EDGE)

    def update(t):
        for ax, (frames, result), color, label in zip(axes[:, 0], runs, COLORS, labels):
            draw(ax, sc, frames, t, color, label, result, cfg.env.goal_offset)

    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    FuncAnimation(fig, update, frames=n).save(args.out, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    print(f"场景 {args.scene}：" + "，".join(f"{lb} {r}" for lb, (_, r) in zip(labels, runs)) + f"；已保存到 {args.out}")


if __name__ == "__main__":
    main()
