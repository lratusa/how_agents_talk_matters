"""Repository banner / social preview image (1280x640).

Not a paper figure — README/Open Graph use only. Uses the shared Okabe-Ito
palette from pubstyle. Run:  python scripts/make_banner.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import pubstyle

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "figures", "banner.png")

STAGES = ["Independent\nsampling", "Semantic\ndistance", "Pairwise\nreview", "Synthesis", "Verdict"]

fig, ax = plt.subplots(figsize=(12.8, 6.4), dpi=100)
fig.patch.set_facecolor("#FFFFFF")
ax.set_xlim(0, 12.8)
ax.set_ylim(0, 6.4)
ax.axis("off")

# Title block
ax.text(0.55, 5.55, "Who Should Review Whom?", fontsize=30, fontweight="bold",
        color="#1a1a1a", va="center")
ax.text(0.55, 4.85, "Response-conditioned disassortative peer review for test-time LLM reasoning",
        fontsize=13.5, color="#444444", va="center")

# Pipeline: five rounded boxes with arrows
box_w, box_h, y0 = 1.85, 1.05, 3.35
xs = [0.55 + i * 2.5 for i in range(5)]
for i, (x, label) in enumerate(zip(xs, STAGES)):
    color = pubstyle.OI_VERMILLION if i == 2 else "#F2F2F2"
    tcolor = "white" if i == 2 else "#333333"
    edge = pubstyle.OI_VERMILLION if i == 2 else "#BBBBBB"
    box = FancyBboxPatch((x, y0), box_w, box_h,
                         boxstyle="round,pad=0.06,rounding_size=0.12",
                         fc=color, ec=edge, lw=1.2)
    ax.add_patch(box)
    ax.text(x + box_w / 2, y0 + box_h / 2, label, ha="center", va="center",
            fontsize=10.5, color=tcolor, fontweight="bold" if i == 2 else "normal")
    if i < 4:
        ax.add_patch(FancyArrowPatch((x + box_w + 0.12, y0 + box_h / 2),
                                     (x + 2.5 - 0.12, y0 + box_h / 2),
                                     arrowstyle="-|>", mutation_scale=14,
                                     color="#888888", lw=1.4))

# Headline result strip
ax.text(0.55, 2.55, "Controlled result (7 seeds, 3,500 questions, ~140k API calls): "
        "farthest pairing detects more errors —",
        fontsize=12.5, color="#1a1a1a", va="center")
ax.text(0.55, 2.05, "but false opposition rises just as fast. Peer review is active, yet net-zero.",
        fontsize=12.5, color="#1a1a1a", va="center")
ax.text(0.55, 1.35, "Most different \u2260 most instructive.",
        fontsize=17, fontweight="bold", color=pubstyle.OI_VERMILLION, va="center")

ax.text(0.55, 0.55, "Code + paper + full audit trail \u00b7 Apache-2.0 / CC BY 4.0",
        fontsize=10.5, color="#777777", va="center")

fig.savefig(OUT, dpi=100, facecolor="#FFFFFF", bbox_inches="tight", pad_inches=0.12)
print("wrote", OUT)
