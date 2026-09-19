#!/usr/bin/env python
"""Full-study analysis + figures for the FPRR experiment (prereg §10/§12).

Usage:
  python scripts/analyze_full.py --benchmark mmlu_pro --pilot
  python scripts/analyze_full.py --benchmark mmlu_pro \
      --main "mmlu_pro__n500__seed*__pseed0__N4__deepseek-flash__nothink"

Inputs: run dirs under data/runs (glob patterns). Outputs:
  results/full_analysis_<benchmark>.{md,json}
  results/case_studies_<benchmark>.md
  figures/F*.{png,pdf,json}
Missing inputs (N-ablation / D-ablation dirs, partial runs) degrade with a
warning, never crash. No API calls. Deterministic (seeded bootstrap).

CAVEAT reported in all outputs: pooling across experimental seeds treats
(seed, qid) observations as independent; seeds share questions, so CIs and
p-values are anti-conservative. Report per-seed numbers alongside.
"""
import argparse
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figplots  # noqa: E402  publication-style figure bodies (pubstyle)
import pubstyle as ps  # noqa: E402

from src import analysis, config
from src.datasets import is_mcq, load_benchmark

FIGDIR = os.path.join(config.BASE_DIR, "figures")
MODEL_TAG = "deepseek-flash__nothink"


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def pilot_main_pattern(bench):
    return "%s__n100__seed*__pseed0__N4__%s" % (bench, MODEL_TAG)


def find_n_ablation(bench):
    """N in {2,6,8} at pseed0; N=4 from the pairing-seed base dir (pseed1)."""
    out = {}
    for n in (2, 6, 8):
        d = analysis.discover_runs(
            ["%s__n*__seed0__pseed0__N%d__%s" % (bench, n, MODEL_TAG)])
        if d:
            out[n] = (d, "pseed0")
    d4 = analysis.discover_runs(
        ["%s__n*__seed0__pseed1__N4__%s" % (bench, MODEL_TAG)])
    if d4:
        out[4] = (d4, "pseed1 (pairing-seed base dir; C3/C5 only)")
    return out


def find_d_ablation(bench):
    """Distance-variant dirs (__D2/__D3l25/.../__D4); D1 reference from the
    pseed1 base dir if present, else the main pseed0 run (labeled)."""
    variants = {}
    for d in analysis.discover_runs(
            ["%s__n*__seed*__pseed0__N4__%s__D*" % (bench, MODEL_TAG)]):
        tag = os.path.basename(d).split("__")[-1]  # D2 / D3l50 / D4 ...
        variants.setdefault(tag, []).append(d)
    d1_ref = analysis.discover_runs(
        ["%s__n*__seed0__pseed1__N4__%s" % (bench, MODEL_TAG)])
    return variants, d1_ref


# ---------------------------------------------------------------------------
# Per-benchmark analysis
# ---------------------------------------------------------------------------

def analyze_benchmark(bench, main_dirs):
    obs, mcq = analysis.load_observations(main_dirs, bench)
    if not obs:
        return None, None, None
    conds = analysis.conds_present(obs)
    res = {
        "benchmark": bench,
        "run_dirs": [os.path.basename(d) for d in main_dirs],
        "n_observations": len(obs),
        "seeds": sorted({o["seed"] for o in obs}),
        "conditions_present": conds,
        "accuracy": analysis.accuracy_table(obs, conds),
        "paired_comparisons": analysis.paired_comparisons(obs),
        "secondary": analysis.secondary_metrics(obs),
        "consensus_transitions": analysis.consensus_transitions(obs),
        "review_outcomes": analysis.review_outcomes(obs),
        "diversity": analysis.diversity_stats(obs),
        "tokens": analysis.token_table(obs),
        "mechanistic": analysis.mechanistic_reviews(obs, mcq),
        "failure_modes": analysis.failure_modes(obs),
    }
    return res, obs, mcq


def run_n_ablation(bench):
    dirs_by_n = find_n_ablation(bench)
    if not dirs_by_n:
        return None
    table = {}
    sources = {}
    for n, (dirs, src) in sorted(dirs_by_n.items()):
        obs, _ = analysis.load_observations(dirs, bench)
        acc = analysis.accuracy_table(obs)
        for cond in ("C1", "C2", "C3", "C4", "C5", "C6", "C9"):
            if cond in acc:
                table.setdefault(cond, {})[str(n)] = {
                    "accuracy": acc[cond]["pooled"]["accuracy"],
                    "n": acc[cond]["pooled"]["n"]}
        sources[str(n)] = src
    return {"accuracy_by_N": table, "sources": sources,
            "note": "N=4 point comes from the pseed1 pairing-seed base dir "
                    "(conditions C3/C5 only there); N in {2,6,8} from pseed0 "
                    "N-ablation dirs. Question subsets may differ across N."}


def run_d_ablation(bench, main_res):
    variants, d1_ref_dirs = find_d_ablation(bench)
    if not variants:
        return None
    rows = {}
    # D1 reference: pseed1 base dir if it exists, else main-run C5 (pseed0)
    d1_acc = None
    d1_src = None
    if d1_ref_dirs:
        obs, _ = analysis.load_observations(d1_ref_dirs, bench)
        acc = analysis.accuracy_table(obs)
        if "C5" in acc:
            d1_acc = acc["C5"]["pooled"]["accuracy"]
            d1_src = "%s (pairing seed 1)" % os.path.basename(d1_ref_dirs[0])
    if d1_acc is None and main_res and "C5" in main_res["accuracy"]:
        d1_acc = main_res["accuracy"]["C5"]["pooled"]["accuracy"]
        d1_src = "%s (main run, pairing seed 0)" % main_res["run_dirs"][0]
    if d1_acc is not None:
        rows["D1"] = {"accuracy": d1_acc, "source": d1_src}
    for tag, dirs in sorted(variants.items()):
        obs, _ = analysis.load_observations(dirs, bench)
        acc = analysis.accuracy_table(obs)
        if "C5" in acc:
            rows[tag] = {"accuracy": acc["C5"]["pooled"]["accuracy"],
                         "n": acc["C5"]["pooled"]["n"],
                         "source": os.path.basename(dirs[0])}
    return {"C5_accuracy_by_distance": rows,
            "note": "All rows are condition C5 (RGFM pairing) with the "
                    "distance metric varied; D1 = cosine on CONCLUSION+"
                    "JUSTIFICATION embeddings (primary). Pairing seed of the "
                    "D1 row is labeled in its source."}


# ---------------------------------------------------------------------------
# Case studies
# ---------------------------------------------------------------------------

def render_case_studies(bench, obs, mcq, path):
    qmap = {r["qid"]: r for r in load_benchmark(bench)}
    cats = analysis.pick_case_studies(obs)
    titles = {
        "successful_correction": "Successful correction by C5 (majority wrong -> C5 correct)",
        "minority_rescue": "Minority rescue (correct minority recovered by C5)",
        "harmful_outlier_pairing": "Harmful outlier pairing (majority right -> C5 wrong; unique-wrong agent in farthest pair)",
        "shared_misconception": "Shared misconception (unanimous wrong)",
    }
    lines = ["# Case studies — %s" % bench, ""]
    for key, title in titles.items():
        o = cats.get(key)
        lines.append("## %s" % title)
        lines.append("")
        if o is None:
            lines.append("_No example found in this dataset._\n")
            continue
        q = qmap.get(o["qid"], {})
        lines.append("**qid** %s (seed %s, run %s)" % (o["qid"], o["seed"], o["run_dir"]))
        lines.append("")
        lines.append("**Question:** %s" % _clip(q.get("question", "?"), 1200))
        if q.get("options"):
            lines.append("")
            for i, opt in enumerate(q["options"]):
                lines.append("- %s. %s" % (chr(65 + i), _clip(opt, 300)))
        lines.append("")
        lines.append("**Gold:** %s" % o["gold"])
        lines.append("")
        lines.append("| agent | answer | correct |")
        lines.append("|---|---|---|")
        for i, (a, c) in enumerate(zip(o["agent_answers"], o["agent_correct"])):
            lines.append("| %d | %s | %s |" % (i, a, "yes" if c else "no"))
        lines.append("")
        lines.append("Majority answer: %s (%s); C0 (agent 0): %s"
                     % (o["maj_answer"],
                        "correct" if o["maj_correct"] else "wrong",
                        "correct" if o["c0_correct"] else "wrong"))
        if o["dist"] is not None:
            lines.append("")
            lines.append("Distance matrix (cosine, D1):")
            lines.append("")
            lines.append("```")
            lines.append(np.array2string(o["dist"], precision=3,
                                         suppress_small=True))
            lines.append("```")
        for cond in ("C3", "C4", "C5", "C6"):
            if o["matchings"].get(cond):
                lines.append("- %s pairs: %s" % (cond, o["matchings"][cond]))
        lines.append("")
        c5reviews = [r for r in o["reviews"] if r["condition"] == "C5"]
        if c5reviews:
            lines.append("C5 reviews:")
            lines.append("")
            lines.append("| reviewer -> reviewee | verdict | updated answer |")
            lines.append("|---|---|---|")
            for r in sorted(c5reviews, key=lambda r: (r["reviewer_idx"],
                                                      r["reviewee_idx"])):
                lines.append("| %d -> %d | %s | %s |"
                             % (r["reviewer_idx"], r["reviewee_idx"],
                                r["review_verdict"], r["updated_answer"]))
            lines.append("")
        finals = {c: o["cond_pred"].get(c) for c in analysis.CONDITION_ORDER
                  if c in o["cond_pred"]}
        lines.append("Final answers: %s"
                     % ", ".join("%s=%s (%s)" % (c, finals[c],
                                "correct" if o["cond_correct"][c] else "wrong")
                                 for c in finals))
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return {k: (v["qid"] if v else None) for k, v in cats.items()}


def _clip(s, n):
    s = str(s).replace("\n", " ")
    return s if len(s) <= n else s[:n] + " ..."


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def render_md(res, n_abl, d_abl, case_qids):
    L = ["# Full analysis — %s" % res["benchmark"], ""]
    L.append("Run dirs pooled: %s" % ", ".join(res["run_dirs"]))
    L.append("Observations: %d (seeds: %s)"
             % (res["n_observations"], res["seeds"]))
    L.append("")
    L.append("> **Cross-seed dependence caveat:** (seed, qid) observations are "
             "pooled; the same questions appear under multiple seeds, so the "
             "observations are not fully independent and bootstrap CIs / "
             "McNemar p-values are anti-conservative. Per-seed accuracies are "
             "reported alongside pooled values.")
    L.append("")
    L.append("## 1. Accuracy per condition")
    L.append("")
    L.append("| condition | pooled acc | 95% CI | n | unparsed | per-seed acc |")
    L.append("|---|---|---|---|---|---|")
    for cond, d in sorted(res["accuracy"].items()):
        p = d["pooled"]
        ci = "[%0.3f, %0.3f]" % (p["ci95"][0], p["ci95"][1]) \
            if p["ci95"][0] is not None else "—"
        per_seed = ", ".join("s%s=%0.3f(n=%d)" % (s, v["accuracy"], v["n"])
                             for s, v in sorted(d["per_seed"].items()))
        L.append("| %s | %.3f | %s | %d | %d | %s |"
                 % (cond, p["accuracy"], ci, p["n"], p["n_unparsed"], per_seed))
    L.append("")
    L.append("## 2. Paired comparisons: C5 vs others")
    L.append("")
    L.append("| comparison | n | diff (pp) | 95% CI (pp) | McNemar b | c | chi2 | p | p Holm (primary family) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k, d in sorted(res["paired_comparisons"].items()):
        mc = d["mcnemar"]
        L.append("| %s | %d | %.2f | [%.2f, %.2f] | %d | %d | %s | %s | %s |"
                 % (k, d["n"], d["diff_pp"], d["ci95_pp"][0], d["ci95_pp"][1],
                    mc["b"], mc["c"],
                    "—" if mc["chi2"] is None else "%.3f" % mc["chi2"],
                    "—" if mc["p"] is None else "%.4g" % mc["p"],
                    "—" if d.get("p_holm") is None else "%.4g" % d["p_holm"]))
    L.append("")
    L.append("Holm-Bonferroni applied across the four primary comparisons "
             "(C5 vs C3, C4, C1, C9).")
    L.append("")
    L.append("## 3. Secondary metrics (pooled)")
    L.append("")
    L.append("| condition | maj-err recovery | maj corruption | minority rescue |")
    L.append("|---|---|---|---|")
    for cond, d in sorted(res["secondary"].items()):
        L.append("| %s | %s | %s | %s |"
                 % (cond, _f(d["majority_error_recovery_rate"]),
                    _f(d["correct_majority_corruption_rate"]),
                    _f(d["minority_rescue_rate"])))
    L.append("")
    ct = res["consensus_transitions"]
    L.append("### Consensus transitions (C0 agent-0 correctness -> final)")
    L.append("")
    L.append("| condition | w->c | c->w | w->w | c->c | w->c rate | c->w rate |")
    L.append("|---|---|---|---|---|---|---|")
    for cond, d in sorted(ct["by_condition"].items()):
        c = d["counts"]
        L.append("| %s | %d | %d | %d | %d | %s | %s |"
                 % (cond, c["wrong_to_correct"], c["correct_to_wrong"],
                    c["wrong_to_wrong"], c["correct_to_correct"],
                    _f(d["wrong_to_correct_rate"]), _f(d["correct_to_wrong_rate"])))
    L.append("")
    L.append("Unanimity over initial answers: %s" % ct["unanimity"])
    L.append("")
    L.append("### Review outcomes")
    L.append("")
    L.append("| condition | n reviews | correction rate (verdict INCORRECT \\| reviewee wrong) | false-opposition rate | verdicts |")
    L.append("|---|---|---|---|---|")
    for cond, d in sorted(res["review_outcomes"].items()):
        L.append("| %s | %d | %s | %s | %s |"
                 % (cond, d["n_reviews"], _f(d["pair_review_correction_rate"]),
                    _f(d["false_opposition_rate"]), d["verdict_counts"]))
    L.append("")
    L.append("### Diversity")
    L.append("")
    L.append("```json")
    L.append(json.dumps(res["diversity"], indent=2))
    L.append("```")
    L.append("")
    L.append("## 4. Token / cost accounting")
    L.append("")
    L.append("| condition | prompt tokens | completion tokens | total | accuracy / M tokens |")
    L.append("|---|---|---|---|---|")
    for cond, d in sorted(res["tokens"].items()):
        L.append("| %s | %d | %d | %d | %s |"
                 % (cond, d["prompt_tokens"], d["completion_tokens"],
                    d["total_tokens"],
                    "—" if d["accuracy_per_million_tokens"] is None
                    else "%.1f" % d["accuracy_per_million_tokens"]))
    L.append("")
    L.append("## 5. Mechanistic: distance-decile review behaviour (C3-C6)")
    L.append("")
    mech = res["mechanistic"]
    if mech.get("by_condition"):
        L.append("Decile edges (pooled C3-C6 pair distances): %s"
                 % mech["decile_edges"])
        L.append("")
        for cond, d in sorted(mech["by_condition"].items()):
            L.append("### %s (%d reviews) — pair composition: %s"
                     % (cond, d["n_reviews"], d["pair_composition"]))
            L.append("")
            L.append("| decile | dist range | n | n reviewee wrong | P(detect \\| wrong) | P(corrected \\| wrong) | P(false opp.) |")
            L.append("|---|---|---|---|---|---|---|")
            for row in d["deciles"]:
                L.append("| %d | [%0.3f, %0.3f] | %d | %d | %s | %s | %s |"
                         % (row["decile"], row["dist_range"][0],
                            row["dist_range"][1], row["n"],
                            row["n_reviewee_wrong"],
                            _f(row["p_detect_given_wrong"]),
                            _f(row["p_corrected_given_wrong"]),
                            _f(row["p_false_opposition"])))
            L.append("")
    else:
        L.append("_No review records with distances available._")
        L.append("")
    L.append("## 6. Failure modes")
    L.append("")
    sm = res["failure_modes"]["shared_misconception"]
    L.append("- Shared misconception (unanimous wrong): %d questions (rate %s); "
             "final still wrong: %s"
             % (sm["n_unanimous_wrong"], _f(sm["rate"]),
                {k: (None if v is None else round(v, 3))
                 for k, v in sm["final_still_wrong_by_condition"].items()}))
    L.append("")
    L.append("| condition | maj disruption n | rate | outlier amplification n | share of disruptions |")
    L.append("|---|---|---|---|---|")
    for cond, d in sorted(res["failure_modes"].items()):
        if cond == "shared_misconception":
            continue
        L.append("| %s | %d | %s | %d | %s |"
                 % (cond, d["correct_majority_disruption_n"],
                    _f(d["correct_majority_disruption_rate"]),
                    d["outlier_amplification_n"],
                    _f(d["outlier_amplification_share_of_disruptions"])))
    L.append("")
    L.append("## 7. Case studies")
    L.append("")
    L.append("See results/case_studies_%s.md — picked: %s"
             % (res["benchmark"], case_qids))
    L.append("")
    L.append("## 8. Agent-count ablation (N)")
    L.append("")
    if n_abl is None:
        L.append("_N-ablation run dirs not found; skipped._")
    else:
        L.append(n_abl["note"])
        L.append("")
        ns = sorted({n for c in n_abl["accuracy_by_N"].values() for n in c},
                    key=int)
        L.append("| condition | " + " | ".join("N=%s" % n for n in ns) + " |")
        L.append("|---" * (len(ns) + 1) + "|")
        for cond, row in sorted(n_abl["accuracy_by_N"].items()):
            L.append("| %s | %s |"
                     % (cond, " | ".join(
                         _f(row.get(n, {}).get("accuracy")) for n in ns)))
        L.append("")
        L.append("Sources: %s" % n_abl["sources"])
    L.append("")
    L.append("## 9. Distance ablation (C5 only)")
    L.append("")
    if d_abl is None:
        L.append("_Distance-variant run dirs not found; skipped._")
    else:
        L.append(d_abl["note"])
        L.append("")
        L.append("| distance | C5 accuracy | n | source |")
        L.append("|---|---|---|---|")
        for tag, d in sorted(d_abl["C5_accuracy_by_distance"].items()):
            L.append("| %s | %s | %s | %s |"
                     % (tag, _f(d["accuracy"]), d.get("n", "—"), d["source"]))
    L.append("")
    return "\n".join(L)


def _f(v):
    return "—" if v is None else "%.3f" % v


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save(fig, name, data):
    os.makedirs(FIGDIR, exist_ok=True)
    ps.save_fig(fig, os.path.join(FIGDIR, name))
    analysis.write_json(data, os.path.join(FIGDIR, name + ".json"))


def fig1_architecture():
    fig = figplots.plot_f1_architecture(None)
    _save(fig, "F1_architecture",
          {"stages": ["Stage A", "Stage B", "Stage C", "Stage D", "Stage E"]})


def fig2_question(obs, bench):
    """Heatmap of the 4x4 distance matrix + C3/C4/C5 matchings as edge
    lists; nodes colored by post-hoc correctness."""
    cand = None
    for o in sorted(obs, key=lambda x: x["qid"]):
        if o["dist"] is None or o["n_agents"] != 4:
            continue
        if len(set(o["agent_answers"])) > 1 and any(o["agent_correct"]) \
                and not all(o["agent_correct"]):
            cand = o
            break
    if cand is None:
        print("  F2: no split-disagreement question found; skipped")
        return None
    o = cand
    data = {"qid": o["qid"], "seed": o["seed"],
            "agent_answers": o["agent_answers"],
            "agent_correct": o["agent_correct"],
            "dist": o["dist"].tolist(),
            "matchings": {c: o["matchings"].get(c) for c in ("C3", "C4", "C5")}}
    fig = figplots.plot_f2_distance_matchings(data, bench)
    _save(fig, "F2_distance_matchings_%s" % bench, data)
    return o["qid"]


def fig3_accuracy(all_res):
    data = {}
    for bench, res in all_res.items():
        conds = [c for c in analysis.CONDITION_ORDER if c in res["accuracy"]]
        data[bench] = {c: res["accuracy"][c]["pooled"] for c in conds}
    fig = figplots.plot_f3_accuracy(data)
    _save(fig, "F3_accuracy_by_condition", data)


def fig4_accuracy_vs_tokens(all_res):
    data = {}
    for bench, res in all_res.items():
        data[bench] = {}
        for cond in analysis.CONDITION_ORDER:
            if cond not in res["tokens"] or cond not in res["accuracy"]:
                continue
            tok = res["tokens"][cond]["total_tokens"]
            if not tok:
                continue
            data[bench][cond] = {
                "total_tokens": tok,
                "accuracy": res["accuracy"][cond]["pooled"]["accuracy"]}
    fig = figplots.plot_f4_accuracy_vs_tokens(data)
    _save(fig, "F4_accuracy_vs_tokens", data)


def fig5_decile_correction(res, bench):
    mech = res["mechanistic"]
    if not mech.get("by_condition"):
        print("  F5: no mechanistic data; skipped")
        return
    data = {"decile_edges": mech["decile_edges"],
            "by_condition": {cond: {"deciles": d["deciles"]}
                             for cond, d in sorted(mech["by_condition"].items())}}
    fig = figplots.plot_f5_decile_rates(data, bench)
    _save(fig, "F5_decile_correction_%s" % bench, data)


def fig6_error_transitions(res, bench):
    ct = res["consensus_transitions"]["by_condition"]
    conds = [c for c in analysis.CONDITION_ORDER if c in ct]
    if not conds:
        print("  F6: no transitions; skipped")
        return
    data = {cond: ct[cond] for cond in conds}
    fig = figplots.plot_f6_error_transitions(data, bench)
    _save(fig, "F6_error_transitions_%s" % bench, data)


def fig7_n_ablation(n_abl, bench):
    if n_abl is None:
        print("  F7: no N-ablation dirs; skipped")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for cond in ("C1", "C3", "C5", "C6"):
        row = n_abl["accuracy_by_N"].get(cond)
        if not row:
            continue
        ns = sorted(row, key=int)
        ax.plot([int(n) for n in ns], [row[n]["accuracy"] for n in ns],
                marker="o", label=cond, lw=2.4 if cond == "C5" else 1.2)
        plotted = True
    if not plotted:
        plt.close(fig)
        print("  F7: no shared conditions in N-ablation; skipped")
        return
    ax.set_xlabel("N (number of agents)")
    ax.set_ylabel("accuracy")
    ax.set_xticks([2, 4, 6, 8])
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Accuracy vs N — %s (N=4 point from pseed1 base dir)" % bench)
    fig.tight_layout()
    _save(fig, "F7_agent_ablation_%s" % bench, n_abl)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", required=True,
                   help="mmlu_pro | supergpqa | gsm8k | math500, "
                        "comma-separated, or 'all'")
    p.add_argument("--main", default=None,
                   help="glob (under data/runs) for main run dirs; "
                        "default with --pilot: <bench>__n100__seed*__pseed0__N4__%s"
                        % MODEL_TAG)
    p.add_argument("--pilot", action="store_true",
                   help="use n100 pilot dirs (default main glob)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    benches = (["mmlu_pro", "supergpqa", "gsm8k", "math500"]
               if args.benchmark == "all"
               else [b.strip() for b in args.benchmark.split(",")])
    all_res = {}
    for bench in benches:
        pat = args.main or pilot_main_pattern(bench)
        if not args.pilot and args.main is None:
            print("warning: no --main given and not --pilot; using pilot glob")
        main_dirs = analysis.discover_runs([pat])
        if not main_dirs:
            print("WARNING [%s]: no main run dirs match %r; skipped"
                  % (bench, pat))
            continue
        print("[%s] main dirs: %s"
              % (bench, [os.path.basename(d) for d in main_dirs]))
        res, obs, mcq = analyze_benchmark(bench, main_dirs)
        if res is None:
            print("WARNING [%s]: no usable observations; skipped" % bench)
            continue
        all_res[bench] = res

        n_abl = run_n_ablation(bench)
        if n_abl is None:
            print("  N-ablation dirs not found; skipping analysis 8")
        d_abl = run_d_ablation(bench, res)
        if d_abl is None:
            print("  distance-variant dirs not found; skipping analysis 9")
        res["n_ablation"] = n_abl
        res["d_ablation"] = d_abl

        case_path = os.path.join(config.RESULTS_DIR,
                                 "case_studies_%s.md" % bench)
        case_qids = render_case_studies(bench, obs, mcq, case_path)
        res["case_studies"] = case_qids

        md = render_md(res, n_abl, d_abl, case_qids)
        md_path = os.path.join(config.RESULTS_DIR,
                               "full_analysis_%s.md" % bench)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        analysis.write_json(res, os.path.join(
            config.RESULTS_DIR, "full_analysis_%s.json" % bench))
        print("  wrote %s" % md_path)

        # per-benchmark figures
        fig2_question(obs, bench)
        fig5_decile_correction(res, bench)
        fig6_error_transitions(res, bench)
        fig7_n_ablation(n_abl, bench)

    if all_res:
        fig1_architecture()
        fig3_accuracy(all_res)
        fig4_accuracy_vs_tokens(all_res)
    print("done.")


if __name__ == "__main__":
    main()
