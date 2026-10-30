"""把原始数据转换成汇入场景并划分 train / val / test。

    python scripts/prepare_data.py --source synthetic
    python scripts/prepare_data.py --source ngsim --path D:/data/NGSIM.csv --locations us-101 --out outputs/us101.pkl
    python scripts/prepare_data.py --source ngsim --path D:/data/NGSIM.csv --locations i-80 --all-test --out outputs/i80.pkl
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mergeil import scenes as S  # noqa: E402
from mergeil.config import Config  # noqa: E402
from mergeil.env import expert_actions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "highd", "ngsim"], required=True)
    ap.add_argument("--path", default="")
    ap.add_argument("--locations", nargs="*", default=["us-101"], help="NGSIM 地点：us-101 / i-80")
    ap.add_argument("--all-test", action="store_true", help="全部作为测试集（用于迁移性评估）")
    ap.add_argument("--out", default="outputs/scenes.pkl")
    ap.add_argument("--config", default="")
    ap.add_argument("--synthetic-groups", type=int, default=4)
    args = ap.parse_args()
    cfg = Config.load(args.config)

    if args.source == "synthetic":
        from mergeil.data import synthetic
        df, metas = synthetic.generate(n_groups=args.synthetic_groups, seed=cfg.data.seed)
    elif args.source == "highd":
        from mergeil.data import highd
        df, metas = highd.load_all(args.path, lane_width=cfg.env.lane_width)
    else:
        from mergeil.data import ngsim
        df, metas = ngsim.load(args.path, locations=tuple(args.locations), lane_width=cfg.env.lane_width)

    scenes = S.extract_scenes(df, metas, cfg.data)
    if not scenes:
        raise SystemExit("没有提取到汇入场景，请检查车道编码和车道线位置")
    if args.all_test:
        train, val, test = [], [], scenes
    else:
        train, val, test = S.split(scenes, cfg.data.n_test, cfg.data.n_val, cfg.data.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    S.save(args.out, train=train, val=val, test=test)

    lens = np.array([s.horizon for s in scenes]) * scenes[0].dt
    acts = np.concatenate([expert_actions(s, cfg.env) for s in scenes])
    print(f"共 {len(scenes)} 个汇入场景（train {len(train)} / val {len(val)} / test {len(test)}），"
          f"时长 {lens.mean():.1f}±{lens.std():.1f} s，dt={scenes[0].dt:.2f} s")
    print(f"专家动作范围 ax [{acts[:, 0].min():.2f}, {acts[:, 0].max():.2f}]，"
          f"ay [{acts[:, 1].min():.2f}, {acts[:, 1].max():.2f}] m/s^2")
    print(f"已保存到 {args.out}")


if __name__ == "__main__":
    main()
