"""Deep mechanism analysis v2 (local only, no API calls).

Addresses reviewer feedback on the funnel writeup:
  A. selection oracle vs review-generated knowledge:
     P(all initials wrong AND some reviewer proposes a correct answer)
  B. forward + reverse funnel per condition, net Delta = P(rescue) - P(corrupt)
  C. stage-wise logistic regressions on pair distance:
     detection ~ d ; fix | detection ~ d ; adoption | fix ~ d
  D. within-question singleton comparison (3:1 splits) and
     P(FPRR-selected partner is wrong)
  E. disagreement-subset and recoverable-subset accuracy, C5 vs C3 paired diff
  F. C2 (no-review synth) vs C5 per-question transitions: rescue vs harm
"""
import collections
import glob
import json
import math
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, ".")
from src import evaluate as ev
from src.datasets import load_benchmark, is_mcq

BENCH_GLOBS = {
    "mmlu_pro": "data/runs/mmlu_pro__n500__seed*__pseed0__N4__deepseek-flash__nothink",
    "supergpqa": "data/runs/supergpqa__n500__seed*__pseed0__N4__deepseek-flash__nothink",
}
REVIEW_CONDS = ["C3", "C4", "C5", "C6", "C9"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def logit_fit(x, y):
    """IRLS logistic regression y ~ 1 + x. Returns (b0, b1, se1, p1)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(50):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-9, None)
        XtW = X.T * W
        H = XtW @ X
        g = X.T @ (y - p)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return beta[0], beta[1], float("nan"), float("nan")
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    eta = X @ beta
    p = 1 / (1 + np.exp(-eta))
    W = np.clip(p * (1 - p), 1e-9, None)
    H = (X.T * W) @ X
    try:
        cov = np.linalg.inv(H)
        se1 = math.sqrt(cov[1, 1])
        z = beta[1] / se1
        p1 = 2 * (1 - stats.norm.cdf(abs(z)))
    except np.linalg.LinAlgError:
        se1, p1 = float("nan"), float("nan")
    return beta[0], beta[1], se1, p1


def fmt_logit(name, x, y):
    if len(set(y)) < 2 or len(y) < 30:
        return f"    {name}: n={len(y)} (degenerate, skipped)"
    b0, b1, se1, p1 = logit_fit(x, y)
    return (f"    {name}: n={len(y)}  beta_dist={b1:+.2f} "
            f"(SE {se1:.2f}, p={p1:.4f})  "
            f"-> odds x{math.exp(b1):.2f} per +1.0 distance")


def main():
    lines = []
    for bench, pattern in BENCH_GLOBS.items():
        dirs = sorted(glob.glob(pattern))
        if not dirs:
            continue
        src = load_benchmark(bench)
        gold = {str(r["qid"]): r["gold"] for r in src}
        mcq = is_mcq(bench)

        initials, reviews, synths, dists, matchings = [], [], [], {}, {}
        for d in dirs:
            initials += load_jsonl(f"{d}/initial.jsonl")
            reviews += load_jsonl(f"{d}/reviews.jsonl")
            synths += load_jsonl(f"{d}/synth.jsonl")
            for r in load_jsonl(f"{d}/distances.jsonl"):
                dists[(r["seed"], str(r["qid"]))] = r["matrix"]
            for r in load_jsonl(f"{d}/matchings.jsonl"):
                matchings[(r["seed"], str(r["qid"]), r["condition"])] = r["pairs"]

        def corr(ans, qid):
            return ans is not None and ev.is_correct(ans, gold[str(qid)], mcq)

        ini = [r for r in initials if str(r["qid"]) in gold]
        ini_corr = {(r["seed"], str(r["qid"]), r["agent_idx"]): corr(r["parsed_answer"], r["qid"])
                    for r in ini}
        by_q = collections.defaultdict(dict)
        for r in ini:
            by_q[(r["seed"], str(r["qid"]))][r["agent_idx"]] = r
        final = {(s["seed"], str(s["qid"]), s["condition"]): corr(s["parsed_answer"], s["qid"])
                 for s in synths if str(s["qid"]) in gold}

        lines.append(f"\n# {bench} ({len(dirs)} dir(s), {len(by_q)} (seed,qid) obs)\n")

        # ---- A. review-generated knowledge beyond the initial set -------------
        lines.append("## A. Beyond the selection oracle (all 4 initials wrong)")
        for cond in REVIEW_CONDS:
            allwrong = [k for k, v in by_q.items()
                        if not any(ini_corr[(k[0], k[1], a)] for a in v)]
            gen_fix = 0
            rescued = 0
            for k in allwrong:
                rv = [r for r in reviews
                      if r["condition"] == cond and r["seed"] == k[0]
                      and str(r["qid"]) == k[1]]
                if any(corr(r["updated_answer"], k[1]) for r in rv):
                    gen_fix += 1
                if final.get((k[0], k[1], cond), False):
                    rescued += 1
            n = len(allwrong)
            lines.append(f"- {cond}: all-wrong questions={n}; review generated a correct "
                         f"answer absent from initials: {gen_fix} ({gen_fix/n:.3f}); "
                         f"final correct: {rescued} ({rescued/n:.3f})")

        # ---- B. forward & reverse funnel --------------------------------------
        lines.append("\n## B. Forward funnel (rescue) vs reverse funnel (corruption), review-level")
        for cond in REVIEW_CONDS:
            rv = [r for r in reviews if r["condition"] == cond and str(r["qid"]) in gold]
            e0 = e1 = e2 = 0
            c0 = c1 = c2 = 0
            for r in rv:
                rc = ini_corr.get((r["seed"], str(r["qid"]), r["reviewer_idx"]), False)
                ec = ini_corr.get((r["seed"], str(r["qid"]), r["reviewee_idx"]), False)
                uc = corr(r["updated_answer"], r["qid"]) if r["updated_answer"] is not None else rc
                if not ec:  # forward: wrong peer
                    e0 += 1
                    if r["review_verdict"] == "PEER INCORRECT":
                        e1 += 1
                        if rc and uc:  # correct reviewer detects & offers correct answer
                            e2 += 1
                if ec and rc:  # reverse: correct reviewer meets correct peer -> safe
                    pass
                if rc and not ec:
                    pass
                if rc:  # correct reviewer at risk
                    c0 += 1
                    if r["review_verdict"] == "PEER INCORRECT" and ec:
                        c1 += 1  # false opposition against a correct peer
                    if not uc:
                        c2 += 1  # correct reviewer flipped to wrong
            # question-level net
            wk = [k for k, v in by_q.items() if not all(ini_corr[(k[0], k[1], a)] for a in v)]
            res = sum(1 for k in wk if final.get((k[0], k[1], cond), False)
                      and not ini_corr[(k[0], k[1], 0)])
            cor = sum(1 for k in wk if not final.get((k[0], k[1], cond), False)
                      and ini_corr[(k[0], k[1], 0)])
            lines.append(f"- {cond}: forward {e0} ->detect {e1} ({e1/e0:.3f}) "
                         f"->fix by correct reviewer {e2} ({e2/e0:.3f}) | "
                         f"reverse: correct reviewers {c0} ->false-opp {c1} ->flipped wrong {c2} "
                         f"({c2/c0:.3f}) | question-level rescue(ag0 w->c)={res} vs corrupt(c->w)={cor} "
                         f"net={res - cor:+d}")

        # ---- C. stage-wise logistic regressions on distance --------------------
        lines.append("\n## C. Stage-wise logistic regressions on pair distance (pooled C3-C6)")
        x_det, y_det, x_fix, y_fix = [], [], [], []
        fix_q = collections.defaultdict(list)  # (seed,qid,cond) -> [dist of fixing reviews]
        for cond in ["C3", "C4", "C5", "C6"]:
            for r in reviews:
                if r["condition"] != cond or str(r["qid"]) not in gold:
                    continue
                key = (r["seed"], str(r["qid"]))
                m = dists.get(key)
                if m is None:
                    continue
                d = m[r["reviewer_idx"]][r["reviewee_idx"]]
                ec = ini_corr.get((r["seed"], str(r["qid"]), r["reviewee_idx"]), False)
                if ec:
                    continue
                det = r["review_verdict"] == "PEER INCORRECT"
                x_det.append(d)
                y_det.append(int(det))
                if det:
                    fix = corr(r["updated_answer"], r["qid"])
                    x_fix.append(d)
                    y_fix.append(int(fix))
                    if fix:
                        fix_q[(r["seed"], str(r["qid"]), cond)].append(d)
        lines.append(fmt_logit("P(detect | reviewee wrong) ~ d", x_det, y_det))
        lines.append(fmt_logit("P(correct fix | detected) ~ d", x_fix, y_fix))
        x_ad, y_ad = [], []
        for (seed, qid, cond), ds in fix_q.items():
            x_ad.append(float(np.mean(ds)))
            y_ad.append(int(final.get((seed, qid, cond), False)))
        lines.append(fmt_logit("P(final correct | fix exists) ~ mean fix-pair d", x_ad, y_ad))

        # ---- D. within-question singleton comparison + FPRR partner quality ----
        lines.append("\n## D. Singleton vs majority, within 3:1 splits only")
        s_corr, m_corr = [], []
        for k, agents in by_q.items():
            counts = collections.Counter(r["parsed_answer"] for r in agents.values())
            if len(counts) != 2 or sorted(counts.values()) != [1, 3]:
                continue
            for a, r in agents.items():
                if counts[r["parsed_answer"]] == 1:
                    s_corr.append(corr(r["parsed_answer"], k[1]))
                else:
                    m_corr.append(corr(r["parsed_answer"], k[1]))
        lines.append(f"- 3:1 questions: n_singletons={len(s_corr)}; "
                     f"P(singleton correct)={np.mean(s_corr):.3f} | "
                     f"n_majority={len(m_corr)}; P(majority correct)={np.mean(m_corr):.3f}")
        lines.append("- P(endpoint wrong) for the max-distance vs min-distance edge of each question's complete graph:")
        for cond_label in ["graph"]:
            mx_w, mn_w = [], []
            for k in by_q:
                m = dists.get(k)
                if m is None:
                    continue
                n = len(m)
                edges = [(m[i][j], i, j) for i in range(n) for j in range(i + 1, n)]
                edges.sort()
                for d, i, j in edges[:1]:
                    mn_w += [not ini_corr.get((k[0], k[1], i), False),
                             not ini_corr.get((k[0], k[1], j), False)]
                for d, i, j in edges[-1:]:
                    mx_w += [not ini_corr.get((k[0], k[1], i), False),
                             not ini_corr.get((k[0], k[1], j), False)]
            base_rate = 1 - np.mean([ini_corr[(k[0], k[1], a)]
                                     for k in by_q for a in by_q[k]])
            lines.append(f"  - max-distance edge endpoints wrong: {np.mean(mx_w):.3f} (n={len(mx_w)}) | "
                         f"min-distance edge endpoints wrong: {np.mean(mn_w):.3f} (n={len(mn_w)}) | "
                         f"base rate: {base_rate:.3f}")

        # ---- E. disagreement / recoverable subset ------------------------------
        lines.append("\n## E. Subset accuracy where topology has treatment")
        S, Sr = [], []
        for k, agents in by_q.items():
            answers = [r["parsed_answer"] for r in agents.values()]
            if len(set(answers)) > 1:
                S.append(k)
                votes = collections.Counter(answers).most_common(1)[0][0]
                maj_correct = ev.is_correct(votes, gold[k[1]], mcq) if votes is not None else False
                any_c = any(ini_corr[(k[0], k[1], a)] for a in agents)
                if not maj_correct and any_c:
                    Sr.append(k)
        lines.append(f"- disagreement subset |S|={len(S)} ({len(S)/len(by_q):.1%}); "
                     f"recoverable subset |S_r|={len(Sr)} ({len(Sr)/len(by_q):.1%})")
        for subset, name in ((S, "disagreement"), (Sr, "recoverable")):
            row = f"  - {name}: "
            for cond in ["C1", "C3", "C4", "C5", "C6", "C9"]:
                vals = [int(final.get((k[0], k[1], cond), False)) for k in subset]
                row += f"{cond}={np.mean(vals):.3f} "
            lines.append(row)
            for cond in ["C3", "C4"]:
                a = np.array([int(final.get((k[0], k[1], "C5"), False)) for k in subset])
                b = np.array([int(final.get((k[0], k[1], cond), False)) for k in subset])
                diff = a - b
                _, lo, hi = ev.paired_bootstrap_ci(diff)
                lines.append(f"    C5-{cond} on {name}: {100*diff.mean():+.2f}pp "
                             f"95% CI [{100*lo:+.2f}, {100*hi:+.2f}] (n={len(subset)})")

        # ---- F. C2 vs C5 transitions: active-but-net-zero vs redundant ---------
        lines.append("\n## F. Review contribution: C2 (no-review synth) vs C5 transitions")
        tt = collections.Counter()
        for k in by_q:
            c2 = final.get((k[0], k[1], "C2"))
            c5 = final.get((k[0], k[1], "C5"))
            if c2 is None or c5 is None:
                continue
            tt[(c2, c5)] += 1
        w2c = tt[(False, True)]
        c2w = tt[(True, False)]
        lines.append(f"- C2 wrong -> C5 correct (review rescue): {w2c} | "
                     f"C2 correct -> C5 wrong (review harm): {c2w} | "
                     f"unchanged: {tt[(True, True)] + tt[(False, False)]} | net={w2c - c2w:+d}")

    report = "\n".join(lines)
    print(report)
    with open("results/mechanism_deep_dive.md", "w", encoding="utf-8") as fh:
        fh.write("# Mechanism deep dive (post-hoc, cached runs only)\n" + report + "\n")


if __name__ == "__main__":
    main()
