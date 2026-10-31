"""汇总多个随机种子的评估结果（scripts/evaluate.py 输出的 table.json），给出均值 ± 标准差。

    python scripts/summarize_seeds.py outputs/ngsim/eval/table.json outputs/ngsim/seed1/eval/table.json ...
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mergeil.metrics import KEYS, LABELS  # noqa: E402

DIGITS = {"success_rate": 3, "collision_rate": 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tables", nargs="+")
    ap.add_argument("--out", default="", help="另存为 JSON")
    args = ap.parse_args()
    tables = []
    for p in args.tables:
        with open(p, encoding="utf-8") as f:
            tables.append(json.load(f))
    names = [n for n in tables[0] if all(n in t for t in tables)]

    summary = {}
    for n in names:
        summary[n] = {}
        for k in KEYS:
            v = np.array([t[n][k] for t in tables], dtype=float)
            v = v[~np.isnan(v)]
            summary[n][k] = (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0)

    print(f"{len(tables)} 个随机种子，均值 ± 标准差\n")
    print("| 模型 | " + " | ".join(LABELS[k] for k in KEYS) + " |")
    print("|---" * (len(KEYS) + 1) + "|")
    for n in names:
        cells = []
        for k in KEYS:
            m, s = summary[n][k]
            d = DIGITS.get(k, 2)
            cells.append(f"{m:.{d}f}" if n == "Observed" else f"{m:.{d}f} ± {s:.{d}f}")
        print(f"| {n} | " + " | ".join(cells) + " |")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
