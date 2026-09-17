#!/bin/bash
# Full study runner — executes preregistration §9 Amendment 7 exactly.
# Order is budget-prioritized: primary comparisons first, ablations last.
# Resumable: cached JSONL records are never regenerated.
set -x
cd "$(dirname "$0")/.."

CONDS="C0,C1,C2,C3,C4,C5,C6,C9"
CONC=64

# 1. Primary: MMLU-Pro 500q x seeds {0,1,2}
for S in 0 1 2; do
  python run_experiment.py --benchmark mmlu_pro --n-questions 500 --seed $S \
    --conditions $CONDS --max-concurrency $CONC 2>&1 | tail -12
done

# 2. Primary: SuperGPQA 500q x seeds {0,1}
for S in 0 1; do
  python run_experiment.py --benchmark supergpqa --n-questions 500 --seed $S \
    --conditions $CONDS --max-concurrency $CONC 2>&1 | tail -12
done

# 3. N ablation: MMLU-Pro 200q seed 0, N in {2,6,8}
for N in 2 6 8; do
  python run_experiment.py --benchmark mmlu_pro --n-questions 200 --seed 0 \
    --n-agents $N --conditions C1,C2,C3,C4,C5,C6,C9 --max-concurrency $CONC 2>&1 | tail -10
done

# 4. Pairing-seed variance: MMLU-Pro 200q seed 0, pairing seeds {1,2}
#    (pseed 2 reuses pseed 1's cached initials: same answer set, prereg §21)
P1_DIR="data/runs/mmlu_pro__n200__seed0__pseed1__N4__deepseek-flash__nothink"
python run_experiment.py --benchmark mmlu_pro --n-questions 200 --seed 0 \
  --pairing-seed 1 --conditions C3,C5 --max-concurrency $CONC 2>&1 | tail -8
python run_experiment.py --benchmark mmlu_pro --n-questions 200 --seed 0 \
  --pairing-seed 2 --conditions C3,C5 --reuse-initials-from "$P1_DIR" \
  --max-concurrency $CONC 2>&1 | tail -8

# 5. Distance ablations: C5 under D2, D3(l=0.5), D4 on 200q of both primaries
#    (D3/D4 reuse D2's cached initials: identical answer set across metrics)
for BM in mmlu_pro supergpqa; do
  D2_DIR="data/runs/${BM}__n200__seed0__pseed0__N4__deepseek-flash__nothink__D2"
  python run_experiment.py --benchmark $BM --n-questions 200 --seed 0 \
    --conditions C5 --distance-id D2 --max-concurrency $CONC 2>&1 | tail -6
  for DID in D3 D4; do
    python run_experiment.py --benchmark $BM --n-questions 200 --seed 0 \
      --conditions C5 --distance-id $DID --lambda-hybrid 0.5 \
      --reuse-initials-from "$D2_DIR" --max-concurrency $CONC 2>&1 | tail -6
  done
done

echo FULL_STUDY_DONE
