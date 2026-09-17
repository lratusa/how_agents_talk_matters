"""Funnel / headroom analysis: why does semantic distance not convert into accuracy?

Reads cached run JSONL only (no API calls). For each benchmark:
  1. Oracle headroom: P(any of N initials correct) vs best condition accuracy.
  2. Review funnel per condition: reviewee-wrong -> detected -> reviewer proposes
     correct answer -> synthesizer final correct (given fix available vs not).
  3. Symmetric flow: does the correct side win cross-cluster review edges?
  4. Is farthest = worst? correctness of the most distant initial answer.
"""
import collections
import glob
import json
import sys

sys.path.insert(0, ".")
from src import evaluate as ev
from src.datasets import load_benchmark, is_mcq

BENCH_GLOBS = {
    "mmlu_pro": "data/runs/mmlu_pro__n500__seed*__pseed0__N4__deepseek-flash__nothink",
    "supergpqa": "data/runs/supergpqa__n500__seed0__pseed0__N4__deepseek-flash__nothink",
}
REVIEW_CONDS = ["C3", "C4", "C5", "C6", "C9"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def main():
    out = []
    for bench, pattern in BENCH_GLOBS.items():
        dirs = sorted(glob.glob(pattern))
        if not dirs:
            continue
        src = load_benchmark(bench)
        gold = {r["qid"]: r["gold"] for r in src}
        mcq = is_mcq(bench)

        initials, reviews, synths = [], [], []
        for d in dirs:
            initials += load_jsonl(f"{d}/initial.jsonl")
            reviews += load_jsonl(f"{d}/reviews.jsonl")
            synths += load_jsonl(f"{d}/synth.jsonl")

        def corr(ans, qid):
            return ans is not None and ev.is_correct(ans, gold[qid], mcq)

        # --- 1. oracle headroom ------------------------------------------------
        by_q = collections.defaultdict(list)
        for r in initials:
            if r["qid"] in gold:
                by_q[(r["seed"], r["qid"])].append(corr(r["parsed_answer"], r["qid"]))
        n_q = len(by_q)
        any_correct = sum(1 for v in by_q.values() if any(v)) / n_q
        all_correct = sum(1 for v in by_q.values() if all(v)) / n_q
        all_wrong = sum(1 for v in by_q.values() if not any(v)) / n_q

        acc = collections.defaultdict(list)
        for s in synths:
            if s["qid"] in gold:
                acc[s["condition"]].append(corr(s["parsed_answer"], s["qid"]))
        best_cond = max(acc, key=lambda c: sum(acc[c]) / len(acc[c]))
        best_acc = sum(acc[best_cond]) / len(acc[best_cond])
        c5_acc = sum(acc["C5"]) / len(acc["C5"])

        out.append(f"\n===== {bench} ({len(dirs)} run dir(s), {n_q} (seed,qid) obs) =====")
        out.append(f"oracle P(any of 4 correct) = {any_correct:.3f} | "
                   f"all-correct {all_correct:.3f} | all-wrong {all_wrong:.3f}")
        out.append(f"best condition {best_cond} = {best_acc:.3f} | C5 = {c5_acc:.3f} | "
                   f"unexploited headroom (oracle - best) = {any_correct - best_acc:+.3f}")

        # --- 2. review funnel ---------------------------------------------------
        final = {(s["seed"], s["qid"], s["condition"]): corr(s["parsed_answer"], s["qid"])
                 for s in synths if s["qid"] in gold}
        ini_corr = {(r["seed"], r["qid"], r["agent_idx"]): corr(r["parsed_answer"], r["qid"])
                    for r in initials if r["qid"] in gold}

        for cond in REVIEW_CONDS:
            rv = [r for r in reviews if r["condition"] == cond and r["qid"] in gold]
            n_w = det = det_fix = 0
            # split by whether reviewer itself was correct
            det_rev_correct = det_rev_wrong = 0
            fix_available_q = collections.defaultdict(bool)
            for r in rv:
                key = (r["seed"], r["qid"], r["reviewee_idx"])
                if not ini_corr.get(key, False):
                    n_w += 1
                    if r["review_verdict"] == "PEER INCORRECT":
                        det += 1
                        if ini_corr.get((r["seed"], r["qid"], r["reviewer_idx"]), False):
                            det_rev_correct += 1
                        else:
                            det_rev_wrong += 1
                        if corr(r["updated_answer"], r["qid"]):
                            det_fix += 1
                            fix_available_q[(r["seed"], r["qid"])] = True
            # gate C: final correct when a correct fix was proposed vs not
            # (restrict to questions with >=1 wrong initial, else trivially correct)
            wrong_q = {k for k, v in by_q.items() if not all(v)}
            with_fix = [k for k in wrong_q if fix_available_q[k]]
            without_fix = [k for k in wrong_q if not fix_available_q[k]]
            f_w = [final.get((k[0], k[1], cond), False) for k in with_fix]
            f_wo = [final.get((k[0], k[1], cond), False) for k in without_fix]
            p_w = sum(f_w) / len(f_w) if f_w else float("nan")
            p_wo = sum(f_wo) / len(f_wo) if f_wo else float("nan")
            out.append(
                f"[{cond}] funnel: wrong-reviewee={n_w} -> detected={det} "
                f"({det / n_w:.3f}) -> correct fix proposed={det_fix} "
                f"({det_fix / det:.3f} of detected; overall {det_fix / n_w:.3f}) | "
                f"detection by correct vs wrong reviewer: "
                f"{det_rev_correct}/{max(det_rev_correct + det_rev_wrong, 1)} = "
                f"{det_rev_correct / max(det, 1):.3f} correct-reviewer share | "
                f"P(final correct | fix exists, n={len(f_w)})={p_w:.3f} vs "
                f"P(final correct | no fix, n={len(f_wo)})={p_wo:.3f}"
            )

        # --- 3. symmetric flow on mixed pairs ------------------------------------
        for cond in ["C3", "C5"]:
            rv = [r for r in reviews if r["condition"] == cond and r["qid"] in gold]
            correct_wins = wrong_wins = 0
            for r in rv:
                rc = ini_corr.get((r["seed"], r["qid"], r["reviewer_idx"]), False)
                ec = ini_corr.get((r["seed"], r["qid"], r["reviewee_idx"]), False)
                if rc == ec:
                    continue
                upd = r["updated_answer"]
                if upd is None:
                    continue
                uc = corr(upd, r["qid"])
                # reviewer correct + stays correct, or reviewer wrong + adopts
                # correct -> correct side wins this direction
                if uc == rc:  # reviewer kept/achieved own correctness state
                    if rc:
                        correct_wins += 1
                    else:
                        wrong_wins += 1
                else:
                    if rc:
                        wrong_wins += 1  # correct reviewer flipped to wrong
                    else:
                        correct_wins += 1  # wrong reviewer flipped to correct
            tot = correct_wins + wrong_wins
            out.append(f"[{cond}] mixed-pair review outcomes: correct side wins "
                       f"{correct_wins}/{tot} = {correct_wins / tot:.3f} "
                       f"(chance = 0.5)")

        # --- 4. is farthest = worst? ----------------------------------------------
        # distance matrices are recomputed cheaply? They are cached in run dir?
        # Fallback: for each question, agent whose answer is unique (singleton)
        # vs agents in the majority cluster -> correctness rates.
        uniq_corr, maj_corr = [], []
        for k, _ in by_q.items():
            seed, qid = k
            ans = [r for r in initials
                   if r["seed"] == seed and r["qid"] == qid]
            counts = collections.Counter(r["parsed_answer"] for r in ans)
            for r in ans:
                c = corr(r["parsed_answer"], qid)
                if counts[r["parsed_answer"]] == 1 and len(counts) > 1:
                    uniq_corr.append(c)
                elif counts[r["parsed_answer"]] > 1:
                    maj_corr.append(c)
        out.append(f"singleton-answer agents correct {sum(uniq_corr)}/{len(uniq_corr)}"
                   f" = {sum(uniq_corr) / max(len(uniq_corr), 1):.3f} vs "
                   f"shared-answer agents {sum(maj_corr)}/{len(maj_corr)}"
                   f" = {sum(maj_corr) / max(len(maj_corr), 1):.3f}")

    print("\n".join(out))


if __name__ == "__main__":
    main()
