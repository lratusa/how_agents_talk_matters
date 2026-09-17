"""Evaluation and statistics (prereg §10).

- Exact-answer accuracy per condition (unparsed => incorrect, frozen rule).
- Paired bootstrap over questions, 10,000 resamples, 95% CIs for differences.
- McNemar (continuity-corrected) for C5 vs {C3, C4, C1, C9} + Holm-Bonferroni.
- Secondary metrics: majority-error recovery, correct-majority corruption,
  minority rescue, consensus transitions, pair-review correction rate,
  diversity statistics, token usage / accuracy per million tokens.

Writes results/<run_id>.json and results/<run_id>.md.
Gold labels are used ONLY here, post hoc — never in any prompt.
"""
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

from . import config
from .datasets import is_mcq, load_benchmark, normalize_value

BOOT_RESAMPLES = 10000
PRIMARY_COMPARISONS = ["C3", "C4", "C1", "C9"]  # C5 vs each (prereg §10)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_run(run_dir):
    return {
        "initial": _load_jsonl(os.path.join(run_dir, "initial.jsonl")),
        "distances": _load_jsonl(os.path.join(run_dir, "distances.jsonl")),
        "matchings": _load_jsonl(os.path.join(run_dir, "matchings.jsonl")),
        "reviews": _load_jsonl(os.path.join(run_dir, "reviews.jsonl")),
        "debate": _load_jsonl(os.path.join(run_dir, "debate.jsonl")),
        "synth": _load_jsonl(os.path.join(run_dir, "synth.jsonl")),
    }


def majority_vote(initials):
    """C1: deterministic majority over parsed answers; ties -> lowest agent
    index among tied answers; tie flag recorded. Unparsed (None) votes are
    ignored; all-None -> prediction None (incorrect)."""
    votes = [(r["agent_idx"], r["parsed_answer"]) for r in initials
             if r["parsed_answer"] is not None]
    if not votes:
        return None, False
    counts = Counter(a for _, a in votes)
    top = max(counts.values())
    tied = {a for a, c in counts.items() if c == top}
    tie = len(tied) > 1
    for idx, a in sorted(votes):  # lowest agent index among tied answers
        if a in tied:
            return a, tie
    return None, tie  # unreachable


def assemble_predictions(run, gold_by_qid):
    """Final answer per (qid, seed, condition). Returns
    preds[condition][(qid, seed)] = answer|None  and  tie flags for C1."""
    initials = defaultdict(dict)  # (qid, seed) -> agent_idx -> rec
    for r in run["initial"]:
        initials[(r["qid"], r["seed"])][r["agent_idx"]] = r

    preds = defaultdict(dict)
    ties = {}
    for key, agents in initials.items():
        ordered = [agents[i] for i in sorted(agents)]
        preds["C0"][key] = ordered[0]["parsed_answer"]  # C0 := agent_idx 0
        vote, tie = majority_vote(ordered)
        preds["C1"][key] = vote
        ties[key] = tie
    for r in run["synth"]:
        preds[r["condition"]][(r["qid"], r["seed"])] = r["parsed_answer"]
    return preds, ties, initials


def is_correct(pred, gold, mcq):
    if pred is None:
        return False
    if mcq:
        return str(pred).strip().upper() == str(gold).strip().upper()
    return normalize_value(pred) == gold  # gold already normalized at load


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(diff_vector, resamples=BOOT_RESAMPLES, seed=20260917):
    """Percentile bootstrap CI for the mean of per-question differences."""
    d = np.asarray(diff_vector, dtype=float)
    n = len(d)
    if n == 0:
        return (None, None, None)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(resamples, n))
    means = d[idx].mean(axis=1)
    return (float(d.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def mcnemar_cc(b, c):
    """McNemar, continuity-corrected: chi2 = (|b-c|-1)^2/(b+c)."""
    if b + c == 0:
        return {"b": b, "c": c, "chi2": None, "p": None}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "chi2": float(chi2),
            "p": float(stats.chi2.sf(chi2, 1))}


def holm_bonferroni(pvals):
    """Returns dict label -> adjusted p (Holm)."""
    items = [(k, v) for k, v in pvals.items() if v is not None]
    m = len(items)
    order = sorted(range(m), key=lambda i: items[i][1])
    adj = {}
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * items[i][1])
        running = max(running, val)
        adj[items[i][0]] = running
    for k, v in pvals.items():
        if v is None:
            adj[k] = None
    return adj


def entropy(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in counter.values() if c)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(run_dir, benchmark, out_dir=config.RESULTS_DIR, questions=None,
             mcq=None):
    """questions: optional in-memory question records (toy/smoke-test runs);
    mcq: optional override, inferred from the benchmark name otherwise."""
    run = load_run(run_dir)
    run_id = os.path.basename(run_dir.rstrip("/\\"))
    src = questions if questions is not None else load_benchmark(benchmark)
    gold_by_qid = {r["qid"]: r["gold"] for r in src}
    if mcq is None:
        mcq = is_mcq(benchmark)
    preds, ties, initials = assemble_predictions(run, gold_by_qid)

    keys = sorted(initials.keys())  # (qid, seed) pairs
    golds = {k: gold_by_qid[k[0]] for k in keys}
    correct = {}  # cond -> np.array of 0/1 over keys
    for cond, pmap in preds.items():
        correct[cond] = np.array(
            [1.0 if is_correct(pmap.get(k), golds[k], mcq) else 0.0
             for k in keys])

    n = len(keys)
    out = {"run_id": run_id, "benchmark": benchmark, "n_questions": n,
           "conditions": {}}

    # --- accuracy per condition + bootstrap CI of the mean ---
    for cond in sorted(correct):
        v = correct[cond]
        mean, lo, hi = paired_bootstrap_ci(v)
        out["conditions"][cond] = {
            "accuracy": mean, "ci95": [lo, hi], "n": n,
            "n_unparsed": int(sum(1 for k in keys
                                  if preds[cond].get(k) is None)),
        }

    # --- paired differences: C5 vs others (and C5 may be absent in subsets) ---
    out["paired_differences"] = {}
    if "C5" in correct:
        for other in sorted(correct):
            if other == "C5":
                continue
            diff = correct["C5"] - correct[other]
            mean, lo, hi = paired_bootstrap_ci(diff)
            out["paired_differences"]["C5-%s" % other] = {
                "diff_pp": None if mean is None else round(100 * mean, 3),
                "ci95_pp": [None if lo is None else round(100 * lo, 3),
                            None if hi is None else round(100 * hi, 3)],
            }

    # --- McNemar C5 vs {C3, C4, C1, C9} + Holm-Bonferroni ---
    out["mcnemar"] = {}
    if "C5" in correct:
        pvals = {}
        for other in PRIMARY_COMPARISONS:
            if other not in correct:
                continue
            b = int(((correct["C5"] == 1) & (correct[other] == 0)).sum())
            c = int(((correct["C5"] == 0) & (correct[other] == 1)).sum())
            res = mcnemar_cc(b, c)
            out["mcnemar"]["C5_vs_%s" % other] = res
            pvals["C5_vs_%s" % other] = res["p"]
        out["mcnemar_holm_adjusted_p"] = holm_bonferroni(pvals)

    # --- secondary metrics (prereg §10) ---
    sec = {}
    maj = correct.get("C1")
    for cond in sorted(correct):
        if cond == "C1" or maj is None:
            continue
        v = correct[cond]
        maj_wrong = maj == 0
        maj_right = maj == 1
        sec[cond] = {
            # majority-error recovery: majority wrong -> condition right
            "majority_error_recovery_rate": _rate(v[maj_wrong]),
            # correct-majority corruption: majority right -> condition wrong
            "correct_majority_corruption_rate":
                None if not maj_right.any() else float((v[maj_right] == 0).mean()),
            # minority rescue: gold held by < N/2 agents -> condition right
            "minority_rescue_rate": _minority_rescue(
                keys, initials, preds[cond], golds, mcq),
        }
    out["secondary"] = sec

    # consensus transitions
    ct = {"unanimous_correct": 0, "unanimous_wrong": 0, "split": 0,
          "split_accuracy_by_condition": {}}
    split_mask = np.zeros(n, dtype=bool)
    for ki, k in enumerate(keys):
        answers = [r["parsed_answer"] for r in initials[k].values()]
        if len(set(answers)) == 1:
            if is_correct(answers[0], golds[k], mcq):
                ct["unanimous_correct"] += 1
            else:
                ct["unanimous_wrong"] += 1
        else:
            ct["split"] += 1
            split_mask[ki] = True
    for cond in sorted(correct):
        ct["split_accuracy_by_condition"][cond] = _rate(correct[cond][split_mask])
    ct["n_majority_ties"] = int(sum(1 for k in keys if ties.get(k)))
    out["consensus_transitions"] = ct

    # pair-review correction / false-opposition rates (gold used post hoc only)
    corr, false_opp = defaultdict(list), defaultdict(list)
    verdict_by = defaultdict(Counter)
    for r in run["reviews"]:
        peer = initials.get((r["qid"], r["seed"]), {}).get(r["reviewee_idx"])
        if peer is None or r["review_verdict"] is None:
            continue
        peer_ok = is_correct(peer["parsed_answer"], gold_by_qid[r["qid"]],
                             mcq)
        cond = r["condition"]
        verdict_by[cond][r["review_verdict"]] += 1
        if not peer_ok:
            corr[cond].append(r["review_verdict"] == "PEER INCORRECT")
        else:
            false_opp[cond].append(r["review_verdict"] == "PEER INCORRECT")
    out["review_outcomes"] = {
        cond: {
            "pair_review_correction_rate": _rate(np.array(corr[cond], float)),
            "false_opposition_rate": _rate(np.array(false_opp[cond], float)),
            "verdict_counts": dict(verdict_by[cond]),
        } for cond in sorted(verdict_by)}

    # diversity statistics
    div = {}
    dist_by_key = {(d["qid"], d["seed"]): np.asarray(d["matrix"], float)
                   for d in run["distances"]}
    mean_d, max_d, min_d, uniq, ent = [], [], [], [], []
    for k in keys:
        answers = [r["parsed_answer"] for r in initials[k].values()]
        counter = Counter(a for a in answers if a is not None)
        uniq.append(len(counter))
        ent.append(entropy(counter))
        if k in dist_by_key:
            m = dist_by_key[k]
            iu = np.triu_indices(m.shape[0], k=1)
            vals = m[iu]
            mean_d.append(vals.mean()); max_d.append(vals.max()); min_d.append(vals.min())
    div = {
        "mean_pairwise_cosine_distance": _rate(np.array(mean_d)),
        "max_pairwise_cosine_distance": _rate(np.array(max_d)),
        "min_pairwise_cosine_distance": _rate(np.array(min_d)),
        "mean_unique_answers": _rate(np.array(uniq, float)),
        "mean_answer_entropy": _rate(np.array(ent, float)),
        "n_with_distances": len(mean_d),
    }
    out["diversity"] = div

    # --- token usage & accuracy per million tokens (prereg §11) ---
    tok = defaultdict(int)
    for cond in correct:
        tok[cond] = 0
    for k in keys:
        agents = initials[k]
        per_initial = {i: (r["usage"] or {}).get("total_tokens") or 0
                       for i, r in agents.items()}
        if "C0" in tok:
            tok["C0"] += per_initial.get(0, 0)
        all_init = sum(per_initial.values())
        for cond in tok:
            if cond == "C0":
                continue
            tok[cond] += all_init
    for r in run["reviews"]:
        tok[r["condition"]] += (r["usage"] or {}).get("total_tokens") or 0
    for r in run["debate"]:
        tok["C7"] += (r["usage"] or {}).get("total_tokens") or 0
    for r in run["synth"]:
        tok[r["condition"]] += (r["usage"] or {}).get("total_tokens") or 0
    out["tokens"] = {}
    for cond in sorted(tok):
        acc = out["conditions"].get(cond, {}).get("accuracy")
        out["tokens"][cond] = {
            "total_tokens": tok[cond],
            "accuracy_per_million_tokens":
                None if not tok[cond] or acc is None else round(acc * 1e6 / tok[cond], 2),
        }

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, run_id + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    md_path = os.path.join(out_dir, run_id + ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(out))
    return out, json_path, md_path


def _rate(v):
    v = np.asarray(v, dtype=float)
    return None if v.size == 0 else float(v.mean())


def _minority_rescue(keys, initials, pmap, golds, mcq):
    vals = []
    n_agents = max(len(v) for v in initials.values()) if initials else 0
    for k in keys:
        answers = [r["parsed_answer"] for r in initials[k].values()]
        n_correct = sum(1 for a in answers if is_correct(a, golds[k], mcq))
        if n_correct < len(answers) / 2:  # correct answer in the minority (or absent)
            vals.append(1.0 if is_correct(pmap.get(k), golds[k], mcq) else 0.0)
    return _rate(np.array(vals))


def render_markdown(out):
    lines = ["# Results — %s" % out["run_id"], "",
             "benchmark: %s | n=%d questions" % (out["benchmark"], out["n_questions"]),
             "", "## Accuracy per condition", "",
             "| condition | accuracy | 95% CI | n | unparsed |",
             "|---|---|---|---|---|"]
    for cond, d in sorted(out["conditions"].items()):
        ci = d["ci95"]
        ci_s = "[%.3f, %.3f]" % (ci[0], ci[1]) if ci[0] is not None else "—"
        acc = "%.3f" % d["accuracy"] if d["accuracy"] is not None else "—"
        lines.append("| %s | %s | %s | %d | %d |" % (cond, acc, ci_s, d["n"],
                                                     d["n_unparsed"]))
    if out["paired_differences"]:
        lines += ["", "## Paired differences (C5 − other), bootstrap 10k", "",
                  "| comparison | diff (pp) | 95% CI (pp) |", "|---|---|---|"]
        for k, d in sorted(out["paired_differences"].items()):
            if d["diff_pp"] is None:
                continue
            lines.append("| %s | %.2f | [%.2f, %.2f] |"
                         % (k, d["diff_pp"], d["ci95_pp"][0], d["ci95_pp"][1]))
    if out.get("mcnemar"):
        lines += ["", "## McNemar (continuity-corrected), Holm-adjusted", "",
                  "| comparison | b | c | chi2 | p | p_holm |", "|---|---|---|---|---|---|"]
        holm = out.get("mcnemar_holm_adjusted_p", {})
        for k, d in sorted(out["mcnemar"].items()):
            p = "—" if d["p"] is None else "%.4g" % d["p"]
            ph = holm.get(k)
            lines.append("| %s | %d | %d | %s | %s | %s |" % (
                k, d["b"], d["c"],
                "—" if d["chi2"] is None else "%.3f" % d["chi2"],
                p, "—" if ph is None else "%.4g" % ph))
    lines += ["", "## Secondary metrics", "",
              "| condition | maj-err recovery | maj corruption | minority rescue |",
              "|---|---|---|---|"]
    for cond, d in sorted(out["secondary"].items()):
        lines.append("| %s | %s | %s | %s |" % (
            cond, _fmt(d["majority_error_recovery_rate"]),
            _fmt(d["correct_majority_corruption_rate"]),
            _fmt(d["minority_rescue_rate"])))
    if out.get("review_outcomes"):
        lines += ["", "## Review outcomes", "",
                  "| condition | correction rate | false-opposition rate | verdicts |",
                  "|---|---|---|---|"]
        for cond, d in sorted(out["review_outcomes"].items()):
            lines.append("| %s | %s | %s | %s |" % (
                cond, _fmt(d["pair_review_correction_rate"]),
                _fmt(d["false_opposition_rate"]), d["verdict_counts"]))
    lines += ["", "## Diversity", "",
              "```json", json.dumps(out["diversity"], indent=2), "```",
              "", "## Consensus transitions", "",
              "```json", json.dumps(out["consensus_transitions"], indent=2), "```",
              "", "## Tokens", "",
              "| condition | total tokens | accuracy / M tokens |", "|---|---|---|"]
    for cond, d in sorted(out["tokens"].items()):
        lines.append("| %s | %d | %s |" % (
            cond, d["total_tokens"],
            "—" if d["accuracy_per_million_tokens"] is None
            else "%.1f" % d["accuracy_per_million_tokens"]))
    return "\n".join(lines) + "\n"


def _fmt(v):
    return "—" if v is None else "%.3f" % v
