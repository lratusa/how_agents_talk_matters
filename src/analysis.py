"""Pooled multi-run analysis for the FPRR full study (prereg §10/§12).

Read-only consumer of data/runs/* directories. Pools observations across
seed run dirs: each (seed, qid) pair is one observation. NOTE: observations
from different experimental seeds are NOT independent (same questions,
correlated sampling) — CIs/McNemar treat them as paired-but-independent;
this caveat is reported alongside the numbers by scripts/analyze_full.py.

Ground truth is used here ONLY post hoc. No API calls, no writes to run dirs.
"""
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np

from . import config
from .datasets import is_mcq, load_benchmark
from .evaluate import (entropy, holm_bonferroni, is_correct, load_run,
                       majority_vote, mcnemar_cc, paired_bootstrap_ci)

CONDITION_ORDER = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C9"]
PRIMARY_OTHERS = ["C3", "C4", "C1", "C9"]          # Holm family (prereg §10)
ALL_COMPARISONS = ["C3", "C4", "C1", "C9", "C2", "C6", "C0"]
REVIEW_CONDS = ["C3", "C4", "C5", "C6", "C9"]


# ---------------------------------------------------------------------------
# Run discovery / loading
# ---------------------------------------------------------------------------

def discover_runs(patterns, runs_dir=config.RUNS_DIR):
    """Glob run dirs (relative patterns under data/runs). Only dirs with at
    least initial.jsonl count. Returns sorted absolute paths."""
    out = []
    for pat in patterns:
        for d in sorted(glob.glob(os.path.join(runs_dir, pat))):
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "initial.jsonl")):
                if d not in out:
                    out.append(d)
    return out


def load_observations(run_dirs, benchmark):
    """Build one observation dict per (seed, qid) across all run dirs.

    Each obs: seed, qid, gold, n_agents, agent_answers, agent_correct,
    c0_correct, maj_answer, maj_correct, maj_tie, cond_pred, cond_correct,
    dist (np.array|None), matchings {cond: pairs}, reviews [recs],
    tokens {cond: {prompt,completion,total}}.
    """
    gold_by_qid = {r["qid"]: r["gold"] for r in load_benchmark(benchmark)}
    mcq = is_mcq(benchmark)
    obs = []
    for rd in run_dirs:
        run = load_run(rd)
        seed = os.path.basename(rd)
        by_qid = defaultdict(lambda: {"initial": {}, "synth": {}, "reviews": [],
                                      "debate": {}, "dist": None,
                                      "matchings": {}})
        rseed = None
        for r in run["initial"]:
            by_qid[r["qid"]]["initial"][r["agent_idx"]] = r
            rseed = r["seed"]
        for r in run["synth"]:
            by_qid[r["qid"]]["synth"][r["condition"]] = r
        for r in run["reviews"]:
            by_qid[r["qid"]]["reviews"].append(r)
        for r in run["debate"]:
            by_qid[r["qid"]]["debate"][r["agent_idx"]] = r
        for r in run["distances"]:
            by_qid[r["qid"]]["dist"] = np.asarray(r["matrix"], float)
        for r in run["matchings"]:
            by_qid[r["qid"]]["matchings"][r["condition"]] = [
                tuple(p) for p in r["pairs"]]
        for qid, b in sorted(by_qid.items()):
            if not b["initial"] or qid not in gold_by_qid:
                continue
            agents = [b["initial"][i] for i in sorted(b["initial"])]
            answers = [a["parsed_answer"] for a in agents]
            gold = gold_by_qid[qid]
            a_corr = [is_correct(a, gold, mcq) for a in answers]
            maj, tie = majority_vote(agents)
            cond_pred, cond_correct = {}, {}
            for cond, srec in b["synth"].items():
                cond_pred[cond] = srec["parsed_answer"]
                cond_correct[cond] = is_correct(srec["parsed_answer"], gold, mcq)
            cond_pred["C0"] = answers[0]
            cond_correct["C0"] = a_corr[0]
            cond_pred["C1"] = maj
            cond_correct["C1"] = is_correct(maj, gold, mcq)

            init_tok = [(a["usage"] or {}) for a in agents]
            def _tot(u, k):
                return sum((x.get(k) or 0) for x in u)
            tokens = {}
            for cond in CONDITION_ORDER:
                if cond not in cond_correct:
                    continue
                p = c = 0
                if cond == "C0":
                    p += init_tok[0].get("prompt_tokens") or 0
                    c += init_tok[0].get("completion_tokens") or 0
                else:
                    p += _tot(init_tok, "prompt_tokens")
                    c += _tot(init_tok, "completion_tokens")
                    if cond in REVIEW_CONDS:
                        for r in b["reviews"]:
                            if r["condition"] == cond:
                                u = r["usage"] or {}
                                p += u.get("prompt_tokens") or 0
                                c += u.get("completion_tokens") or 0
                    if cond == "C7":
                        for drec in b["debate"].values():
                            u = drec["usage"] or {}
                            p += u.get("prompt_tokens") or 0
                            c += u.get("completion_tokens") or 0
                    srec = b["synth"].get(cond)
                    if srec:
                        u = srec["usage"] or {}
                        p += u.get("prompt_tokens") or 0
                        c += u.get("completion_tokens") or 0
                tokens[cond] = {"prompt": p, "completion": c, "total": p + c}

            obs.append({
                "run_dir": os.path.basename(rd), "seed": rseed, "qid": qid,
                "gold": gold, "n_agents": len(agents),
                "agent_answers": answers, "agent_correct": a_corr,
                "agent_justifications": [a["justification"] for a in agents],
                "c0_correct": a_corr[0], "maj_answer": maj,
                "maj_correct": is_correct(maj, gold, mcq), "maj_tie": tie,
                "cond_pred": cond_pred, "cond_correct": cond_correct,
                "dist": b["dist"], "matchings": b["matchings"],
                "reviews": b["reviews"], "tokens": tokens,
            })
    return obs, mcq


def conds_present(obs):
    s = set()
    for o in obs:
        s |= set(o["cond_correct"])
    return [c for c in CONDITION_ORDER if c in s]


# ---------------------------------------------------------------------------
# 1-2. Accuracy and paired comparisons
# ---------------------------------------------------------------------------

def accuracy_table(obs, conds=None, boot_seed=20260917):
    """Per condition: accuracy per seed and pooled, 95% bootstrap CI."""
    conds = conds or conds_present(obs)
    seeds = sorted({o["seed"] for o in obs})
    table = {}
    for cond in conds:
        per_seed = {}
        for s in seeds:
            v = np.array([1.0 if o["cond_correct"].get(cond) else 0.0
                          for o in obs if o["seed"] == s and cond in o["cond_correct"]])
            if v.size:
                per_seed[str(s)] = {"n": int(v.size), "accuracy": float(v.mean())}
        v = np.array([1.0 if o["cond_correct"].get(cond) else 0.0
                      for o in obs if cond in o["cond_correct"]])
        mean, lo, hi = paired_bootstrap_ci(v, seed=boot_seed)
        table[cond] = {"pooled": {"n": int(v.size), "accuracy": mean,
                                  "ci95": [lo, hi],
                                  "n_unparsed": int(sum(
                                      1 for o in obs
                                      if cond in o["cond_pred"]
                                      and o["cond_pred"].get(cond) is None))},
                       "per_seed": per_seed}
    return table


def paired_comparisons(obs, base="C5", others=ALL_COMPARISONS,
                       boot_seed=20260917):
    """C5 vs others: paired bootstrap diff CI, McNemar-CC, Holm over the
    four primary comparisons (C3, C4, C1, C9)."""
    out = {}
    pvals = {}
    base_vec = {id(o): o["cond_correct"].get(base) for o in obs}
    for other in others:
        if other == base:
            continue
        pairs = [(1.0 if o["cond_correct"][base] else 0.0,
                  1.0 if o["cond_correct"][other] else 0.0)
                 for o in obs
                 if base in o["cond_correct"] and other in o["cond_correct"]]
        if not pairs:
            continue
        a = np.array([p[0] for p in pairs])
        bvec = np.array([p[1] for p in pairs])
        mean, lo, hi = paired_bootstrap_ci(a - bvec, seed=boot_seed)
        b = int(((a == 1) & (bvec == 0)).sum())
        c = int(((a == 0) & (bvec == 1)).sum())
        mc = mcnemar_cc(b, c)
        key = "%s_vs_%s" % (base, other)
        out[key] = {"n": len(pairs),
                    "diff_pp": round(100 * mean, 3),
                    "ci95_pp": [round(100 * lo, 3), round(100 * hi, 3)],
                    "mcnemar": mc}
        if other in PRIMARY_OTHERS:
            pvals[key] = mc["p"]
    holm = holm_bonferroni(pvals)
    for k, v in holm.items():
        out[k]["p_holm"] = v
    return out


# ---------------------------------------------------------------------------
# 3. Secondary metrics
# ---------------------------------------------------------------------------

def secondary_metrics(obs):
    conds = [c for c in conds_present(obs) if c != "C1"]
    maj = np.array([1.0 if o["maj_correct"] else 0.0 for o in obs])
    out = {}
    for cond in conds:
        mask = np.array([cond in o["cond_correct"] for o in obs])
        v = np.array([1.0 if o["cond_correct"].get(cond) else 0.0
                      for o in obs])[mask]
        m = maj[mask]
        entry = {
            "n": int(mask.sum()),
            "majority_error_recovery_rate": _mean(v[m == 0]),
            "correct_majority_corruption_rate":
                None if not (m == 1).any() else float((v[m == 1] == 0).mean()),
            "minority_rescue_rate": _minority_rescue(obs, cond, mask),
        }
        out[cond] = entry
    return out


def _minority_rescue(obs, cond, mask):
    vals = []
    for o, keep in zip(obs, mask):
        if not keep:
            continue
        n = len(o["agent_correct"])
        if sum(o["agent_correct"]) < n / 2:
            vals.append(1.0 if o["cond_correct"].get(cond) else 0.0)
    return _mean(np.array(vals))


def consensus_transitions(obs):
    """Per condition: C0(agent-0) -> final correctness transition counts and
    rates; plus unanimity stats over initial answers."""
    trans = {}
    for cond in conds_present(obs):
        cats = Counter()
        for o in obs:
            if cond not in o["cond_correct"]:
                continue
            before = o["c0_correct"]
            after = o["cond_correct"][cond]
            cats[("correct" if before else "wrong") + "_to_" +
                 ("correct" if after else "wrong")] += 1
        n = sum(cats.values())
        trans[cond] = {
            "counts": {k: cats.get(k, 0) for k in
                       ("wrong_to_correct", "correct_to_wrong",
                        "wrong_to_wrong", "correct_to_correct")},
            "n": n,
            "wrong_to_correct_rate": _safe_div(cats.get("wrong_to_correct", 0),
                                               cats.get("wrong_to_correct", 0)
                                               + cats.get("wrong_to_wrong", 0)),
            "correct_to_wrong_rate": _safe_div(cats.get("correct_to_wrong", 0),
                                               cats.get("correct_to_wrong", 0)
                                               + cats.get("correct_to_correct", 0)),
        }
    unanim = {"unanimous_correct": 0, "unanimous_wrong": 0, "split": 0}
    for o in obs:
        if len(set(o["agent_answers"])) == 1:
            unanim["unanimous_correct" if o["agent_correct"][0]
                   else "unanimous_wrong"] += 1
        else:
            unanim["split"] += 1
    return {"by_condition": trans, "unanimity": unanim}


def review_outcomes(obs):
    """Pair-review correction rate (verdict INCORRECT | reviewee wrong) and
    false-opposition rate (verdict INCORRECT | reviewee right), per condition."""
    out = {}
    for cond in REVIEW_CONDS:
        corr, false_opp, verdicts = [], [], Counter()
        for o in obs:
            for r in o["reviews"]:
                if r["condition"] != cond or r["review_verdict"] is None:
                    continue
                peer_ok = o["agent_correct"][r["reviewee_idx"]]
                verdicts[r["review_verdict"]] += 1
                if peer_ok:
                    false_opp.append(r["review_verdict"] == "PEER INCORRECT")
                else:
                    corr.append(r["review_verdict"] == "PEER INCORRECT")
        if verdicts:
            out[cond] = {
                "n_reviews": sum(verdicts.values()),
                "pair_review_correction_rate": _mean(np.array(corr, float)),
                "false_opposition_rate": _mean(np.array(false_opp, float)),
                "verdict_counts": dict(verdicts)}
    return out


def diversity_stats(obs):
    mean_d, max_d, min_d, uniq, ent = [], [], [], [], []
    for o in obs:
        cnt = Counter(a for a in o["agent_answers"] if a is not None)
        uniq.append(len(cnt))
        ent.append(entropy(cnt))
        if o["dist"] is not None:
            iu = np.triu_indices(o["dist"].shape[0], k=1)
            v = o["dist"][iu]
            mean_d.append(v.mean()); max_d.append(v.max()); min_d.append(v.min())
    def _r(v):
        return None if v is None else round(v, 4)
    return {"mean_pairwise_cosine_distance": _r(_mean(np.array(mean_d))),
            "max_pairwise_cosine_distance": _r(_mean(np.array(max_d))),
            "min_pairwise_cosine_distance": _r(_mean(np.array(min_d))),
            "mean_unique_answers": _r(_mean(np.array(uniq, float))),
            "mean_answer_entropy": _r(_mean(np.array(ent, float))),
            "n_with_distances": len(mean_d)}


# ---------------------------------------------------------------------------
# 4. Tokens
# ---------------------------------------------------------------------------

def token_table(obs):
    out = {}
    for cond in conds_present(obs):
        p = sum(o["tokens"][cond]["prompt"] for o in obs if cond in o["tokens"])
        c = sum(o["tokens"][cond]["completion"] for o in obs if cond in o["tokens"])
        n = sum(1 for o in obs if cond in o["cond_correct"])
        acc = (np.mean([1.0 if o["cond_correct"][cond] else 0.0
                        for o in obs if cond in o["cond_correct"]])
               if n else None)
        out[cond] = {"prompt_tokens": p, "completion_tokens": c,
                     "total_tokens": p + c,
                     "accuracy_per_million_tokens":
                     None if not (p + c) or acc is None
                     else round(acc * 1e6 / (p + c), 2)}
    return out


# ---------------------------------------------------------------------------
# 5. Mechanistic: distance-binned review behaviour
# ---------------------------------------------------------------------------

def mechanistic_reviews(obs, mcq, n_bins=10):
    """Join reviews with gold + pair distance. Decile binning over the pooled
    C3-C6 pair distances. Per condition per decile:
      - P(verdict PEER INCORRECT | reviewee actually wrong)  [detection]
      - P(updated answer == gold | reviewee actually wrong)  [correction]
    Also pair-composition counts (reviewer/reviewee correctness)."""
    rows = []
    for o in obs:
        if o["dist"] is None:
            continue
        for r in o["reviews"]:
            cond = r["condition"]
            if cond not in ("C3", "C4", "C5", "C6"):
                continue
            i, j = r["reviewer_idx"], r["reviewee_idx"]
            rows.append({
                "condition": cond,
                "distance": float(o["dist"][i, j]),
                "reviewer_correct": bool(o["agent_correct"][i]),
                "reviewee_correct": bool(o["agent_correct"][j]),
                "verdict": r["review_verdict"],
                "updated_correct": bool(is_correct(
                    r["updated_answer"], o["gold"], mcq))
                if r["updated_answer"] is not None else False,
            })
    if not rows:
        return {"n_reviews": 0}
    dists = np.array([r["distance"] for r in rows])
    edges = np.quantile(dists, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -1e-9, 1.0 + 1e-9
    by_cond = {}
    for cond in ("C3", "C4", "C5", "C6"):
        crows = [r for r in rows if r["condition"] == cond]
        if not crows:
            continue
        dec = []
        for b in range(n_bins):
            bin_rows = [r for r in crows
                        if edges[b] <= r["distance"] < edges[b + 1]]
            wrong = [r for r in bin_rows if not r["reviewee_correct"]]
            dec.append({
                "decile": b + 1,
                "dist_range": [round(float(edges[b]), 4),
                               round(float(edges[b + 1]), 4)],
                "n": len(bin_rows),
                "n_reviewee_wrong": len(wrong),
                "p_detect_given_wrong": _mean(np.array(
                    [r["verdict"] == "PEER INCORRECT" for r in wrong], float)),
                "p_corrected_given_wrong": _mean(np.array(
                    [r["updated_correct"] for r in wrong], float)),
                "p_false_opposition": _mean(np.array(
                    [r["verdict"] == "PEER INCORRECT"
                     for r in bin_rows if r["reviewee_correct"]], float)),
            })
        comp = Counter("%s-%s" % (
            "correct" if r["reviewer_correct"] else "incorrect",
            "correct" if r["reviewee_correct"] else "incorrect")
            for r in crows)
        by_cond[cond] = {"deciles": dec, "pair_composition": dict(comp),
                         "n_reviews": len(crows)}
    return {"n_reviews": len(rows),
            "decile_edges": [round(float(e), 4) for e in edges],
            "by_condition": by_cond}


# ---------------------------------------------------------------------------
# 6. Failure modes
# ---------------------------------------------------------------------------

def failure_modes(obs):
    """(a) outlier amplification, (b) correct-majority disruption,
    (c) shared misconception (unanimous wrong)."""
    out = {}
    n = len(obs)
    unanimous_wrong = [o for o in obs
                       if len(set(o["agent_answers"])) == 1
                       and not o["agent_correct"][0]]
    out["shared_misconception"] = {
        "n_unanimous_wrong": len(unanimous_wrong),
        "rate": _safe_div(len(unanimous_wrong), n),
        "final_still_wrong_by_condition": {
            cond: _mean(np.array([
                not o["cond_correct"].get(cond, True) for o in unanimous_wrong
                if cond in o["cond_correct"]], float))
            for cond in conds_present(obs)}}
    for cond in [c for c in conds_present(obs) if c not in ("C0", "C1")]:
        disruption = amplification = 0
        for o in obs:
            if cond not in o["cond_correct"]:
                continue
            if not o["maj_correct"] or o["cond_correct"][cond]:
                continue
            disruption += 1  # majority right -> final wrong
            # outlier amplification: a uniquely-wrong agent sits in the
            # farthest pair of this condition's matching
            pairs = o["matchings"].get(cond)
            if pairs and o["dist"] is not None:
                wrong_single = [i for i, (a, c) in enumerate(
                    zip(o["agent_answers"], o["agent_correct"]))
                    if not c and o["agent_answers"].count(a) == 1]
                if wrong_single:
                    far_pair = max(pairs, key=lambda p: o["dist"][p[0], p[1]])
                    if any(i in far_pair for i in wrong_single):
                        amplification += 1
        total = sum(1 for o in obs if cond in o["cond_correct"])
        out[cond] = {
            "correct_majority_disruption_n": disruption,
            "correct_majority_disruption_rate": _safe_div(disruption, total),
            "outlier_amplification_n": amplification,
            "outlier_amplification_share_of_disruptions":
                _safe_div(amplification, disruption),
        }
    return out


# ---------------------------------------------------------------------------
# 7. Case studies
# ---------------------------------------------------------------------------

def pick_case_studies(obs):
    """Pick one representative obs per category (deterministic: first match
    in sorted order)."""
    cats = {}
    for o in obs:
        if "C5" not in o["cond_correct"]:
            continue
        n_right = sum(o["agent_correct"])
        n = len(o["agent_correct"])
        if ("successful_correction" not in cats and not o["maj_correct"]
                and o["cond_correct"]["C5"]):
            cats["successful_correction"] = o
        if ("minority_rescue" not in cats and 0 < n_right < n / 2
                and o["cond_correct"]["C5"]):
            cats["minority_rescue"] = o
        if "harmful_outlier_pairing" not in cats and o["maj_correct"] \
                and not o["cond_correct"]["C5"] and o["dist"] is not None \
                and o["matchings"].get("C5"):
            wrong_single = [i for i, (a, c) in enumerate(
                zip(o["agent_answers"], o["agent_correct"]))
                if not c and o["agent_answers"].count(a) == 1]
            if wrong_single:
                far = max(o["matchings"]["C5"],
                          key=lambda p: o["dist"][p[0], p[1]])
                if any(i in far for i in wrong_single):
                    cats["harmful_outlier_pairing"] = o
        if ("shared_misconception" not in cats
                and len(set(o["agent_answers"])) == 1 and not o["agent_correct"][0]):
            cats["shared_misconception"] = o
    return cats


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mean(v):
    v = np.asarray(v, dtype=float)
    return None if v.size == 0 else float(v.mean())


def _safe_div(a, b):
    return None if not b else a / b


def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
