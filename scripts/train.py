"""训练论文中的各个模型。

    python scripts/train.py --model gcairl        # 论文主模型 GC-AIRL
    python scripts/train.py --model gairl         # 消融：只有目标条件
    python scripts/train.py --model cairl         # 消融：只有置信度
    python scripts/train.py --model airl          # 基线
    python scripts/train.py --model gail          # 基线
    python scripts/train.py --model rl            # 基线：手工奖励 + PPO
    python scripts/train.py --model lstm          # 基线
    python scripts/train.py --model bc            # 额外基线（论文没有）
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mergeil import scenes as S  # noqa: E402
from mergeil.checkpoints import save_gaussian, save_lstm  # noqa: E402
from mergeil.config import Config  # noqa: E402

MODELS = ["gcairl", "gairl", "cairl", "airl", "gail", "rl", "lstm", "bc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--scenes", default="outputs/scenes.pkl")
    ap.add_argument("--out", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--iterations", type=int, default=0, help="覆盖配置中的迭代次数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = Config.load(args.config)
    if args.iterations:
        cfg.adv.iterations = args.iterations
    out = args.out or f"outputs/{args.model}"
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    torch.set_num_threads(4)

    d = S.load(args.scenes)
    train, val = d["train"], d["val"] or d["train"]
    print(f"训练场景 {len(train)}，验证场景 {len(val)}")

    if args.model == "lstm":
        from mergeil.lstm import train_lstm
        net = train_lstm(train, cfg.env, cfg.lstm, seed=args.seed)
        save_lstm(net, cfg.lstm, os.path.join(out, "best.pt"))
    elif args.model == "bc":
        from mergeil.bc import train_bc
        from mergeil.env import expert_transitions
        E = expert_transitions(train, cfg.env, use_goal=True)
        pi = train_bc(E["obs"], E["act"], cfg.bc, seed=args.seed)
        save_gaussian(pi, True, "bc", cfg.bc.hidden, os.path.join(out, "best.pt"))
    else:
        from mergeil.adversarial import train as train_adv
        train_adv(args.model, train, val, cfg, out, seed=args.seed)
    print(f"模型已保存到 {out}")


if __name__ == "__main__":
    main()
