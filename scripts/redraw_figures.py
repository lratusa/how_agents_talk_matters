#!/usr/bin/env python
"""Redraw all paper figures (F1-F6, F8) in publication style, offline.

Reads only stored data — figures/F*.json and results/full_analysis_*.json
(F3/F4 data are not fully stored in figures/*.json, so they are rebuilt
from the results tables; the paper's F3/F4 cover the two main benchmarks
MMLU-Pro and SuperGPQA, not the n=100 single-seed GSM8K pilot).
Writes figures/F*.{png,pdf,json} under the existing names. Never touches
results/ or paper/. No API calls.

Usage:  python scripts/redraw_figures.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pubstyle as ps  # noqa: E402
import figplots  # noqa: E402
import plot_f8  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(BASE, "figures")
RESULTS = os.path.join(BASE, "results")

# Benchmarks whose per-benchmark figures exist in figures/*.json.
PER_BENCH = ["mmlu_pro", "supergpqa", "gsm8k"]
# Benchmarks covered by the aggregate paper figures F3/F4.
MAIN_BENCHES = ["mmlu_pro", "supergpqa"]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(fig, name, data):
    ps.save_fig(fig, os.path.join(FIGDIR, name))
    if data is not None:
        with open(os.path.join(FIGDIR, name + ".json"), "w",
                  encoding="utf-8") as f:
            json.dump(data, f, indent=1)


def main():
    # F1 — static architecture
    save(figplots.plot_f1_architecture(None), "F1_architecture",
         {"stages": ["Stage A", "Stage B", "Stage C", "Stage D", "Stage E"]})

    # F2 / F5 / F6 — per benchmark, straight from stored JSON
    for bench in PER_BENCH:
        p = os.path.join(FIGDIR, "F2_distance_matchings_%s.json" % bench)
        if os.path.exists(p):
            data = load_json(p)
            save(figplots.plot_f2_distance_matchings(data, bench),
                 "F2_distance_matchings_%s" % bench, data)
        p = os.path.join(FIGDIR, "F5_decile_correction_%s.json" % bench)
        if os.path.exists(p):
            data = load_json(p)
            save(figplots.plot_f5_decile_rates(data, bench),
                 "F5_decile_correction_%s" % bench, data)
        p = os.path.join(FIGDIR, "F6_error_transitions_%s.json" % bench)
        if os.path.exists(p):
            data = load_json(p)
            save(figplots.plot_f6_error_transitions(data, bench),
                 "F6_error_transitions_%s" % bench, data)

    # F3 / F4 — rebuilt from results tables (main benchmarks only)
    f3, f4 = {}, {}
    for bench in MAIN_BENCHES:
        res = load_json(os.path.join(RESULTS, "full_analysis_%s.json" % bench))
        f3[bench] = {c: d["pooled"] for c, d in res["accuracy"].items()}
        f4[bench] = {c: {"total_tokens": res["tokens"][c]["total_tokens"],
                         "accuracy": res["accuracy"][c]["pooled"]["accuracy"]}
                     for c in res["tokens"] if c in res["accuracy"]}
    save(figplots.plot_f3_accuracy(f3), "F3_accuracy_by_condition", f3)
    save(figplots.plot_f4_accuracy_vs_tokens(f4), "F4_accuracy_vs_tokens", f4)

    # F8 — dual funnels (data: figures/F8_repair_funnels.json)
    plot_f8.main()
    print("done.")


if __name__ == "__main__":
    main()
