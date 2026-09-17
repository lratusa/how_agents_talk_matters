"""Figure F8: dual repair funnels (forward rescue vs reverse corruption).

2x2 grid: rows = benchmarks, cols = forward / reverse chain.
x = funnel stage, y = fraction of stage-0 questions. One line per condition,
C5 (FPRR) highlighted. Data: figures/F8_repair_funnels.json.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH_TITLES = {"mmlu_pro": "MMLU-Pro (n=1500)", "supergpqa": "SuperGPQA (seed 0, n=500)"}
CONDS = ["C3", "C4", "C5", "C6", "C9"]
LABELS = {"C3": "Random", "C4": "Nearest", "C5": "FPRR", "C6": "Max-weight", "C9": "Self-review"}
FWD = ["E0", "E1", "E2", "E3"]
REV = ["C0", "C1", "C2", "C3"]
FWD_X = ["≥1 wrong\ninitial", "error\ndetected", "correct fix\nproposed", "final answer\ncorrect"]
REV_X = ["≥1 correct\ninitial", "false\nopposition", "correct agent\ndestabilized", "final answer\nwrong"]

with open("figures/F8_repair_funnels.json", encoding="utf-8") as fh:
    data = json.load(fh)

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=False)
colors = {"C3": "#4878d0", "C4": "#6acc64", "C5": "#d65f5f", "C6": "#b47cc7", "C9": "#c4ad66"}

for row, bench in enumerate(["mmlu_pro", "supergpqa"]):
    for col, (stages, xlabels, title) in enumerate([
            (FWD, FWD_X, "Forward chain (error rescue)"),
            (REV, REV_X, "Reverse chain (corruption of correct answers)")]):
        ax = axes[row][col]
        for cond in CONDS:
            f = data[bench]["funnels"][cond]
            base = f[stages[0]]
            ys = [f[s] / base for s in stages]
            lw = 3.0 if cond == "C5" else 1.4
            alpha = 1.0 if cond == "C5" else 0.75
            ax.plot(range(4), ys, marker="o", lw=lw, alpha=alpha,
                    color=colors[cond], label=LABELS[cond])
        ax.set_xticks(range(4))
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.grid(alpha=0.3)
        if row == 0:
            ax.set_title(title, fontsize=11)
        if col == 0:
            ax.set_ylabel(f"{BENCH_TITLES[bench]}\nfraction of stage-0 questions", fontsize=9)
        if row == 0 and col == 0:
            ax.legend(fontsize=8, loc="upper right")

fig.suptitle("Review funnels: detection scales, resolution does not; corruption grows with distance",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("figures/F8_repair_funnels.pdf")
fig.savefig("figures/F8_repair_funnels.png", dpi=200)
print("saved figures/F8_repair_funnels.{pdf,png}")
