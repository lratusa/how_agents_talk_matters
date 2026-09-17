"""Paired cross-distance-metric comparisons on identical cached initials (local only).

Within each benchmark, D2/D3l50/D4 runs share the same cached initial responses
(D3/D4 were run with --reuse-initials-from D2). Compare C5 final answers across
distance metrics, paired by question.
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from src import evaluate as ev
from src.datasets import load_benchmark, is_mcq

RUNS = {
    "mmlu_pro": {
        "D2": "data/runs/mmlu_pro__n200__seed0__pseed0__N4__deepseek-flash__nothink__D2",
        "D3l50": "data/runs/mmlu_pro__n200__seed0__pseed0__N4__deepseek-flash__nothink__D3l50",
        "D4": None,
    },
    "supergpqa": {
        "D2": "data/runs/supergpqa__n200__seed0__pseed0__N4__deepseek-flash__nothink__D2",
        "D3l50": "data/runs/supergpqa__n200__seed0__pseed0__N4__deepseek-flash__nothink__D3l50",
        "D4": "data/runs/supergpqa__n200__seed0__pseed0__N4__deepseek-flash__nothink__D4",
    },
}


def load_c5(d):
    out = {}
    with open(f"{d}/synth.jsonl", encoding="utf-8") as fh:
        for l in fh:
            r = json.loads(l)
            if r["condition"] == "C5":
                out[str(r["qid"])] = r["parsed_answer"]
    return out


lines = []
for bench, runs in RUNS.items():
    src = load_benchmark(bench)
    gold = {str(r["qid"]): r["gold"] for r in src}
    mcq = is_mcq(bench)
    corr = {}
    for did, d in runs.items():
        if d is None:
            continue
        preds = load_c5(d)
        corr[did] = {q: int(p is not None and ev.is_correct(p, gold[q], mcq))
                     for q, p in preds.items() if q in gold}
    lines.append(f"\n# {bench}")
    dids = sorted(corr)
    for did in dids:
        acc = np.mean(list(corr[did].values()))
        lines.append(f"- C5/{did}: acc={acc:.3f} (n={len(corr[did])})")
    for a in dids:
        for b in dids:
            if a >= b:
                continue
            qs = sorted(set(corr[a]) & set(corr[b]))
            d = np.array([corr[a][q] - corr[b][q] for q in qs])
            _, lo, hi = ev.paired_bootstrap_ci(d)
            bcd = sum(1 for q in qs if corr[a][q] == 1 and corr[b][q] == 0)
            ccd = sum(1 for q in qs if corr[a][q] == 0 and corr[b][q] == 1)
            mc = ev.mcnemar_cc(bcd, ccd)
            lines.append(f"- {a} - {b}: {100*d.mean():+.2f}pp CI [{100*lo:+.2f}, "
                         f"{100*hi:+.2f}], McNemar b/c={bcd}/{ccd}, p={mc['p']} (n={len(qs)})")

report = "# Paired distance-metric comparisons (identical cached initials)\n" + "\n".join(lines) + "\n"
with open("results/distance_ablation_paired.md", "w", encoding="utf-8") as fh:
    fh.write(report)
print(report)
