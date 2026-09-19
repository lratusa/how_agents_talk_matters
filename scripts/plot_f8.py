"""Figure F8: dual repair funnels (forward rescue vs reverse corruption).

2x2 grid: rows = benchmarks, cols = forward / reverse chain. Panels tagged
(a)-(d). x = funnel stage, y = fraction of stage-0 questions. One line per
condition, C5 (FPRR) highlighted in vermillion. Publication style via
pubstyle.py. Data: figures/F8_repair_funnels.json. No API calls.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pubstyle as ps  # noqa: E402

ps.apply()

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(BASE, "figures")

BENCH_TITLES = {"mmlu_pro": "MMLU-Pro (n=1500)",
                "supergpqa": "SuperGPQA (n=2000)"}
CONDS = ["C3", "C4", "C5", "C6", "C9"]
LABELS = {"C3": "C3 random", "C4": "C4 nearest", "C5": "C5 FPRR",
          "C6": "C6 max-weight", "C9": "C9 self-review"}
FWD = ["E0", "E1", "E2", "E3"]
REV = ["C0", "C1", "C2", "C3"]
FWD_X = ["≥1 wrong\ninitial", "error\ndetected", "correct fix\nproposed",
         "final answer\ncorrect"]
REV_X = ["≥1 correct\ninitial", "false\nopposition",
         "correct agent\ndestabilized", "final answer\nwrong"]
COL_TITLES = ["forward chain (error rescue)",
              "reverse chain (corruption)"]


def main():
    with open(os.path.join(FIGDIR, "F8_repair_funnels.json"),
              encoding="utf-8") as fh:
        data = json.load(fh)

    fig, axes = plt.subplots(2, 2, figsize=(ps.DOUBLE_COL, ps.mm(110)))
    tags = iter(["(a)", "(b)", "(c)", "(d)"])
    for row, bench in enumerate(["mmlu_pro", "supergpqa"]):
        for col, (stages, xlabels) in enumerate([(FWD, FWD_X),
                                                 (REV, REV_X)]):
            ax = axes[row][col]
            for cond in CONDS:
                f = data[bench]["funnels"][cond]
                base = f[stages[0]]
                ys = [f[s] / base for s in stages]
                big = cond == "C5"
                ax.plot(range(4), ys, marker="o", ms=3.4 if big else 2.6,
                        lw=1.8 if big else 0.9,
                        alpha=1.0 if big else 0.85,
                        color=ps.cond_color(cond), label=LABELS[cond],
                        zorder=3 if big else 2)
            ax.set_xticks(range(4))
            ax.set_xticklabels(xlabels)
            ax.set_ylim(0, 1.02)
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: "%.0f%%" % (v * 100)))
            ps.style_ax(ax, grid="y")
            ps.panel_label(ax, next(tags), x=-0.14, y=1.08)
            ax.set_title("%s: %s" % (BENCH_TITLES[bench], COL_TITLES[col]),
                         pad=4)
            if col == 0:
                ax.set_ylabel("Fraction of stage-0 questions")
            if row == 0 and col == 0:
                ax.legend(loc="upper right", borderaxespad=0.1)

    fig.tight_layout()
    ps.save_fig(fig, os.path.join(FIGDIR, "F8_repair_funnels"))


if __name__ == "__main__":
    main()
