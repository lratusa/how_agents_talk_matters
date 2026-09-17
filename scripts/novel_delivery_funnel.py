"""Novel-correct delivery audit + strict question-level dual funnels (local only).

Part 1 (delivery): on all-initial-wrong questions, for each condition:
  - reviews proposing a NOVEL correct answer (value absent from the initial set)
  - final correct
  - adopted: final answer value equals some novel-correct proposal value
  - final correct WITHOUT any novel-correct proposal (synthesizer-generated)
  - proposed but NOT delivered

Part 2 (F8 funnels), question level, per condition:
  forward:  E0 >=1 wrong initial -> E1 >=1 true detection -> E2 >=1 correct fix
            -> E3 final correct
  reverse:  C0 >=1 correct initial -> C1 >=1 false opposition against a correct
            peer -> C2 >=1 correct reviewer destabilized (updated answer wrong)
            -> C3 final wrong
Saves JSON for figure F8 and prints a markdown report.
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
    "supergpqa": "data/runs/supergpqa__n500__seed*__pseed0__N4__deepseek-flash__nothink",
}
CONDS = ["C3", "C4", "C5", "C6", "C9"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def norm_val(ans, mcq):
    if ans is None:
        return None
    if mcq:
        return str(ans).strip().upper()
    return ev.normalize_value(ans)


def main():
    fig = {}
    lines = []
    for bench, pattern in BENCH_GLOBS.items():
        dirs = sorted(glob.glob(pattern))
        if not dirs:
            continue
        src = load_benchmark(bench)
        gold = {str(r["qid"]): r["gold"] for r in src}
        mcq = is_mcq(bench)

        initials, reviews, synths = [], [], []
        for d in dirs:
            initials += load_jsonl(f"{d}/initial.jsonl")
            reviews += load_jsonl(f"{d}/reviews.jsonl")
            synths += load_jsonl(f"{d}/synth.jsonl")

        def corr(ans, qid):
            return ans is not None and ev.is_correct(ans, gold[str(qid)], mcq)

        ini = [r for r in initials if str(r["qid"]) in gold]
        ini_corr = {(r["seed"], str(r["qid"]), r["agent_idx"]): corr(r["parsed_answer"], r["qid"])
                    for r in ini}
        by_q = collections.defaultdict(dict)
        for r in ini:
            by_q[(r["seed"], str(r["qid"]))][r["agent_idx"]] = r
        final_ans = {(s["seed"], str(s["qid"]), s["condition"]): s["parsed_answer"]
                     for s in synths if str(s["qid"]) in gold}
        rv_by = collections.defaultdict(list)
        for r in reviews:
            if str(r["qid"]) in gold:
                rv_by[(r["condition"], r["seed"], str(r["qid"]))].append(r)

        lines.append(f"\n# {bench} ({len(by_q)} obs)\n")

        # ---------------- Part 1: delivery audit on all-initial-wrong ----------
        lines.append("## 1. Novel-correct delivery audit (all 4 initials wrong)")
        lines.append("| cond | all-wrong q | novel correct proposed | final correct | "
                     "adopted the proposal | final correct w/o proposal | proposed, not delivered |")
        lines.append("|---|---|---|---|---|---|---|")
        fig.setdefault(bench, {})["delivery"] = {}
        for cond in CONDS:
            n_all = n_prop = n_fin = n_adopt = n_wo = n_lost = 0
            for k, agents in by_q.items():
                if any(ini_corr[(k[0], k[1], a)] for a in agents):
                    continue
                n_all += 1
                init_vals = {norm_val(r["parsed_answer"], mcq) for r in agents.values()}
                props = set()
                for r in rv_by.get((cond, k[0], k[1]), []):
                    ua = r["updated_answer"]
                    if ua is not None and corr(ua, k[1]) and norm_val(ua, mcq) not in init_vals:
                        props.add(norm_val(ua, mcq))
                if props:
                    n_prop += 1
                fans = final_ans.get((k[0], k[1], cond))
                fcorr = fans is not None and corr(fans, k[1])
                if fcorr:
                    n_fin += 1
                    if norm_val(fans, mcq) in props:
                        n_adopt += 1
                    else:
                        n_wo += 1
                if props and not (fcorr and norm_val(fans, mcq) in props):
                    n_lost += 1
            lines.append(f"| {cond} | {n_all} | {n_prop} | {n_fin} | {n_adopt} | "
                         f"{n_wo} | {n_lost} |")
            fig[bench]["delivery"][cond] = dict(
                all_wrong=n_all, novel_proposed=n_prop, final_correct=n_fin,
                adopted=n_adopt, correct_without_proposal=n_wo, lost=n_lost)

        # ---------------- Part 2: strict dual funnels ---------------------------
        lines.append("\n## 2. Question-level dual funnels")
        lines.append("| cond | E0 >=1 wrong | E1 detected | E2 fix proposed | E3 final correct "
                     "| C0 >=1 correct | C1 false-opp | C2 destabilized | C3 final wrong |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        fig[bench]["funnels"] = {}
        for cond in CONDS:
            E0 = E1 = E2 = E3 = C0 = C1 = C2 = C3 = 0
            for k, agents in by_q.items():
                rvs = rv_by.get((cond, k[0], k[1]), [])
                has_wrong = any(not ini_corr[(k[0], k[1], a)] for a in agents)
                has_correct = any(ini_corr[(k[0], k[1], a)] for a in agents)
                fans = final_ans.get((k[0], k[1], cond))
                fcorr = fans is not None and corr(fans, k[1])
                if has_wrong:
                    E0 += 1
                    det = any(r["review_verdict"] == "PEER INCORRECT"
                              and not ini_corr.get((k[0], k[1], r["reviewee_idx"]), True)
                              for r in rvs)
                    if det:
                        E1 += 1
                        fix = any(r["updated_answer"] is not None
                                  and corr(r["updated_answer"], k[1]) for r in rvs)
                        if fix:
                            E2 += 1
                            if fcorr:
                                E3 += 1
                if has_correct:
                    C0 += 1
                    fo = any(r["review_verdict"] == "PEER INCORRECT"
                             and ini_corr.get((k[0], k[1], r["reviewee_idx"]), False)
                             for r in rvs)
                    if fo:
                        C1 += 1
                        destab = any(ini_corr.get((k[0], k[1], r["reviewer_idx"]), False)
                                     and r["updated_answer"] is not None
                                     and not corr(r["updated_answer"], k[1]) for r in rvs)
                        if destab:
                            C2 += 1
                            if not fcorr:
                                C3 += 1
            lines.append(f"| {cond} | {E0} | {E1} ({E1/E0:.2f}) | {E2} ({E2/E0:.2f}) "
                         f"| {E3} ({E3/E0:.2f}) | {C0} | {C1} ({C1/C0:.2f}) "
                         f"| {C2} ({C2/C0:.2f}) | {C3} ({C3/C0:.2f}) |")
            fig[bench]["funnels"][cond] = dict(E0=E0, E1=E1, E2=E2, E3=E3,
                                               C0=C0, C1=C1, C2=C2, C3=C3)

    report = "# Novel-correct delivery audit + dual funnels\n" + "\n".join(lines) + "\n"
    with open("results/novel_delivery_funnels.md", "w", encoding="utf-8") as fh:
        fh.write(report)
    with open("figures/F8_repair_funnels.json", "w", encoding="utf-8") as fh:
        json.dump(fig, fh, indent=1)
    print(report)


if __name__ == "__main__":
    main()
