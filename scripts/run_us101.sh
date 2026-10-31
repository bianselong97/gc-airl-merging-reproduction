#!/usr/bin/env bash
# US-101 全部实验：8 个模型 + GC-AIRL（目标奖励 30）各跑 3 个随机种子，再评估并汇总（论文表 2、表 3）。
#   bash scripts/run_us101.sh [场景文件] [输出目录]
# 每个种子的 9 个模型并行训练，CPU 上一个种子约 35 分钟。中断后重跑会跳过日志显示已保存的模型。
set -e
SCENES=${1:-outputs/us101.pkl}
OUT=${2:-outputs/ngsim}
MODELS="gcairl gairl cairl airl gail rl lstm bc"
export PYTHONIOENCODING=utf-8
done_() { grep -qs "模型已保存到" "$(dirname $1)/logs/$(basename $1).log"; }

for s in 0 1 2; do
  d=$OUT/seed$s
  mkdir -p $d/logs
  for m in $MODELS; do
    done_ $d/$m || python scripts/train.py --model $m --seed $s --scenes $SCENES --out $d/$m > $d/logs/$m.log 2>&1 &
  done
  done_ $d/gcairl_goal30 || python scripts/train.py --model gcairl --seed $s --config configs/goal30.json \
    --scenes $SCENES --out $d/gcairl_goal30 > $d/logs/gcairl_goal30.log 2>&1 &
  wait
done

for s in 0 1 2; do
  d=$OUT/seed$s
  ckpts="$(for m in $MODELS; do echo $d/$m/best.pt; done) $d/gcairl_goal30/best.pt"
  python scripts/evaluate.py --scenes $SCENES --models $ckpts --out $d/eval > $d/eval.log 2>&1 &
  python scripts/evaluate.py --scenes $SCENES --deterministic --models $ckpts --out $d/eval_det > $d/eval_det.log 2>&1 &
done
wait

for e in eval eval_det; do
  python scripts/summarize_seeds.py $OUT/seed0/$e/table.json $OUT/seed1/$e/table.json $OUT/seed2/$e/table.json \
    --out $OUT/summary_$e.json | tee $OUT/summary_$e.md
done
