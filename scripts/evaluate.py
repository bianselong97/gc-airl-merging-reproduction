"""在测试集上评估多个模型，输出论文表 2 / 表 4 格式的对比表和图。

    python scripts/evaluate.py --models outputs/gcairl/best.pt outputs/airl/best.pt outputs/gail/best.pt
    python scripts/evaluate.py --scenes outputs/i80.pkl --models ...      # 迁移性评估（论文表 4）
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mergeil import scenes as S  # noqa: E402
from mergeil.checkpoints import load_policy  # noqa: E402
from mergeil.config import Config  # noqa: E402
from mergeil.evaluate import evaluate, expert_results, plot  # noqa: E402
from mergeil.metrics import aggregate, format_table  # noqa: E402

NAMES = {"gcairl": "GC-AIRL", "gairl": "G-AIRL", "cairl": "C-AIRL", "airl": "AIRL", "gail": "GAIL",
         "rl": "RL", "lstm": "LSTM", "bc": "BC"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="outputs/scenes.pkl")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--n-runs", type=int, default=0, help="评估次数，默认取配置（论文：5）")
    ap.add_argument("--deterministic", action="store_true", help="用策略均值而非随机采样")
    ap.add_argument("--config", default="")
    ap.add_argument("--out", default="outputs/eval")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    scenes = S.load(args.scenes)[args.split]
    os.makedirs(args.out, exist_ok=True)

    obs = expert_results(scenes, cfg)
    table, per_traj = {"Observed": aggregate(obs)}, {"Observed": obs}
    for p in args.models:
        pi, use_goal, model = load_policy(p)
        name = NAMES.get(model, model)
        if name in table:  # 同类模型的多个检查点，用所在目录名区分
            name = f"{name}[{os.path.basename(os.path.dirname(os.path.abspath(p)))}]"
        table[name], per_traj[name] = evaluate(scenes, cfg, pi, use_goal, n_runs=args.n_runs or None,
                                               deterministic=args.deterministic)

    print(f"{len(scenes)} 个场景，每个模型评估 {args.n_runs or cfg.eval.n_runs} 次取平均")
    print(format_table(table))
    with open(os.path.join(args.out, "table.json"), "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    plot(per_traj, scenes, os.path.join(args.out, "comparison.png"))
    print(f"结果已保存到 {args.out}")


if __name__ == "__main__":
    main()
