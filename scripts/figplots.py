"""Figure bodies F1-F6 in publication style (see pubstyle.py).

Each plot_fN_*() takes a plain data dict (the same structure stored in
figures/F*.json) and returns a matplotlib Figure. Saving is left to the
caller (pubstyle.save_fig / analysis.write_json). Both
scripts/analyze_full.py (fresh data) and scripts/redraw_figures.py
(stored JSON) render through this module so the style lives in one place.
"""
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pubstyle as ps  # noqa: E402

ps.apply()

BENCH_TITLES = {
    "mmlu_pro": "MMLU-Pro",
    "supergpqa": "SuperGPQA",
    "gsm8k": "GSM8K",
    "math500": "MATH-500",
}


def bench_title(bench, n=None):
    t = BENCH_TITLES.get(bench, bench)
    return "%s (n=%d)" % (t, n) if n else t


def _conds_in_order(data_dict):
    order = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C9"]
    return [c for c in order if c in data_dict]


# ---------------------------------------------------------------------------
# F1 — pipeline architecture
# ---------------------------------------------------------------------------

def plot_f1_architecture(data):
    # Body lines pre-wrapped short; a measurement pass below shrinks the body
    # font uniformly if any line would exceed its box interior.
    stages = [
        ("Stage A", ["N independent", "initial samples", "(T=0.8,",
                     "top_p=0.95)"], False),
        ("Stage B", ["bge-m3", "embeddings:", "CONCLUSION +",
                     "JUSTIFICATION;", "cosine", "distance matrix"], False),
        ("Stage C", ["Perfect", "matching:", "random / nearest",
                     "RGFM (FPRR) /", "max-weight"], True),
        ("Stage D", ["Reciprocal", "peer review", "(both directions)"], False),
        ("Stage E", ["Synthesizer:", "anonymized", "S1..Sk;", "single best",
                     "answer"], False),
    ]
    fig, ax = plt.subplots(figsize=(ps.DOUBLE_COL, ps.mm(48)))
    ax.axis("off")
    w, h, y0 = 0.175, 0.70, 0.15
    step = 0.199
    pad = 0.003
    body_texts, boxes = [], []
    for i, (title, lines, highlight) in enumerate(stages):
        x = 0.006 + i * step
        ec = ps.OI_VERMILLION if highlight else "0.25"
        lw = 1.4 if highlight else 0.8
        box = FancyBboxPatch((x, y0), w, h,
                             boxstyle="round,pad=%f,rounding_size=0.015" % pad,
                             fc="white", ec=ec, lw=lw)
        ax.add_patch(box)
        boxes.append(box)
        ax.text(x + w / 2, y0 + h - 0.08, title, ha="center", va="center",
                fontsize=8, fontweight="bold",
                color=ps.OI_VERMILLION if highlight else "black")
        n = len(lines)
        for k, ln in enumerate(lines):
            ly = y0 + h - 0.21 - k * (h - 0.28) / max(n - 1, 1)
            is_fprr = ln.startswith("RGFM")
            t = ax.text(x + w / 2, ly, ln, ha="center", va="center",
                        fontsize=8,
                        color=ps.OI_VERMILLION if is_fprr else "0.15",
                        fontweight="bold" if is_fprr else "normal")
            body_texts.append(t)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + w + pad + 0.0015, y0 + h / 2),
                                         (x + step - pad - 0.0015, y0 + h / 2),
                                         arrowstyle="-|>", mutation_scale=7,
                                         lw=0.8, color="0.25", zorder=5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Uniform autofit: worst line must fit its box interior (minus 6 px).
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    scale = 1.0
    pairs = []
    pos = 0
    for bi, cnt in enumerate([len(s[1]) for s in stages]):
        for _ in range(cnt):
            pairs.append((body_texts[pos], boxes[bi]))
            pos += 1
    for t, box in pairs:
        tw = t.get_window_extent(rend).width
        bw = box.get_window_extent(rend).width - 6
        if tw > 0:
            scale = min(scale, bw / tw)
    if scale < 1.0:
        for t in body_texts:
            t.set_fontsize(max(6.0, 8.0 * scale * 0.98))
    return fig


# ---------------------------------------------------------------------------
# F2 — distance matrix + matchings for one example question
# ---------------------------------------------------------------------------

def plot_f2_distance_matchings(data, bench):
    dist = np.asarray(data["dist"], dtype=float)
    answers = data["agent_answers"]
    correct = data["agent_correct"]
    matchings = data["matchings"]
    n_agents = len(answers)

    fig, axes = plt.subplots(2, 2, figsize=(ps.SINGLE_COL, ps.mm(86)))
    ax_hm, ax_c3, ax_c4, ax_c5 = axes.ravel()

    im = ax_hm.imshow(dist, cmap=ps.HEAT_CMAP, vmin=0)
    ax_hm.set_xticks(range(n_agents))
    ax_hm.set_yticks(range(n_agents))
    vmax = float(np.nanmax(dist)) or 1.0
    for i in range(n_agents):
        for j in range(n_agents):
            v = dist[i, j]
            if abs(v) < 1e-12:
                v = 0.0  # avoid "-0.00" on the diagonal
            ax_hm.text(j, i, "%.2f" % v, ha="center", va="center",
                       fontsize=6.5,
                       color="white" if v > 0.55 * vmax else "black")
    for s in ax_hm.spines.values():
        s.set_visible(False)
    ax_hm.set_title("D1 cosine distance", pad=3)
    cb = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7, width=0.6, length=2.5)
    cb.outline.set_linewidth(0.6)
    cb.set_label("cosine distance", fontsize=7)

    angles = np.linspace(0, 2 * np.pi, n_agents, endpoint=False) + np.pi / 2
    pos = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}
    for ax, cond, title in ((ax_c3, "C3", "C3 random"),
                            (ax_c4, "C4", "C4 nearest"),
                            (ax_c5, "C5", "C5 RGFM (FPRR)")):
        ax.axis("off")
        ax.set_title(title, pad=3,
                     color=ps.OI_VERMILLION if cond == "C5" else "black",
                     fontweight="bold" if cond == "C5" else "normal")
        pairs = matchings.get(cond) or []
        for i, j in pairs:
            ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], "-",
                    lw=1.2, color=ps.cond_color(cond), zorder=1)
        for i in range(n_agents):
            ax.scatter(*pos[i], s=210, zorder=2, linewidths=0.6,
                       c=ps.CORRECT_COLOR if correct[i] else ps.WRONG_COLOR,
                       edgecolors="black")
            ax.text(pos[i][0], pos[i][1], str(i), ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold", zorder=3)
            ans = answers[i] if answers[i] is not None else "–"
            ax.text(pos[i][0] * 1.3, pos[i][1] * 1.3, str(ans),
                    ha="center", va="center", fontsize=7, color="0.15")
        ax.set_xlim(-1.62, 1.62)
        ax.set_ylim(-1.62, 1.62)
        ax.set_aspect("equal")

    for ax, tag in ((ax_hm, "(a)"), (ax_c3, "(b)"), (ax_c4, "(c)"),
                    (ax_c5, "(d)")):
        ps.panel_label(ax, tag, x=-0.08, y=1.06)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ps.CORRECT_COLOR,
               markeredgecolor="black", markeredgewidth=0.6, markersize=5,
               label="correct (post hoc)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ps.WRONG_COLOR,
               markeredgecolor="black", markeredgewidth=0.6, markersize=5,
               label="wrong (post hoc)"),
        Line2D([0], [0], color="0.25", lw=1.2, label="matched pair"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7,
               bbox_to_anchor=(0.5, 0.0), frameon=False,
               handletextpad=0.4, columnspacing=0.9)
    fig.text(0.5, 0.985, "%s — qid %s, seed %s"
             % (bench_title(bench), data["qid"], data.get("seed", "?")),
             ha="center", va="top", fontsize=8)
    fig.tight_layout(rect=[0.02, 0.055, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# F3 — accuracy by condition: point estimates + 95% CI (two benchmarks)
# ---------------------------------------------------------------------------

def plot_f3_accuracy(data):
    """data[bench][cond] = pooled dict {accuracy, ci95, n, ...}.

    Y axis is truncated to the observed CI range (0.52-0.88): with point
    estimates + CIs (forest-plot idiom) a zero baseline is not required —
    unlike bars, point markers carry no implied length from zero. This is
    deliberate and noted here; the axis limits are printed on the axis.
    """
    benches = list(data.keys())
    fig, axes = plt.subplots(1, len(benches), sharey=True,
                             figsize=(ps.DOUBLE_COL, ps.mm(52)))
    if len(benches) == 1:
        axes = [axes]
    all_lo, all_hi = 1.0, 0.0
    for ax, bench in zip(axes, benches):
        conds = _conds_in_order(data[bench])
        accs = [data[bench][c]["accuracy"] for c in conds]
        los = [a - data[bench][c]["ci95"][0] for c, a in zip(conds, accs)]
        his = [data[bench][c]["ci95"][1] - a for c, a in zip(conds, accs)]
        all_lo = min(all_lo, min(a - l for a, l in zip(accs, los)))
        all_hi = max(all_hi, max(a + h for a, h in zip(accs, his)))
        xs = np.arange(len(conds))
        for x, a, lo, hi, c in zip(xs, accs, los, his, conds):
            big = c == "C5"
            ax.errorbar(x, a, yerr=[[lo], [hi]], fmt="o",
                        ms=4.5 if big else 3.2,
                        mew=0.8 if big else 0.6,
                        color=ps.cond_color(c),
                        ecolor=ps.cond_color(c),
                        elinewidth=0.9, capsize=2, capthick=0.7,
                        mec="black" if big else "none",
                        zorder=3 if big else 2)
        ax.set_xticks(xs)
        ax.set_xticklabels(conds)
        ax.set_title(bench_title(bench, data[bench][conds[0]]["n"]), pad=3)
        ps.style_ax(ax, grid="y")
        ps.highlight_c5_tick(ax)
    axes[0].set_ylabel("Accuracy")
    lo = np.floor(all_lo * 20) / 20
    hi = np.ceil(all_hi * 20) / 20
    axes[0].set_ylim(lo, hi)
    for ax, tag in zip(axes, ("(a)", "(b)", "(c)")):
        ps.panel_label(ax, tag)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F4 — accuracy vs token cost
# ---------------------------------------------------------------------------

_MARKERS = ["o", "s", "^", "D"]


def plot_f4_accuracy_vs_tokens(data):
    """data[bench][cond] = {total_tokens, accuracy}."""
    fig, ax = plt.subplots(figsize=(ps.mm(132), ps.mm(84)))
    bench_handles = {}
    for bi, (bench, per_cond) in enumerate(data.items()):
        mk = _MARKERS[bi % len(_MARKERS)]
        for cond in _conds_in_order(per_cond):
            d = per_cond[cond]
            if not d.get("total_tokens"):
                continue
            big = cond == "C5"
            ax.scatter(d["total_tokens"] / 1e6, d["accuracy"], marker=mk,
                       s=52 if big else 26,
                       facecolors=ps.cond_color(cond),
                       edgecolors="black", linewidths=0.6,
                       zorder=3 if big else 2)
        bench_handles[bench] = Line2D(
            [0], [0], marker=mk, color="none", markerfacecolor="0.6",
            markeredgecolor="black", markeredgewidth=0.6, markersize=5,
            label=bench_title(bench))
    cond_handles = [Line2D([0], [0], marker="o", color="none",
                           markerfacecolor=ps.cond_color(c),
                           markeredgecolor="black", markeredgewidth=0.6,
                           markersize=5, label=ps.COND_LABELS[c])
                    for c in _conds_in_order(
                        {c: 1 for pc in data.values() for c in pc})]
    # Figure-level legends inside the reserved right margin (no clipping).
    leg1 = fig.legend(handles=cond_handles, loc="center left",
                      bbox_to_anchor=(0.76, 0.66), title="Condition",
                      title_fontsize=8, fontsize=7)
    fig.add_artist(leg1)
    fig.legend(handles=list(bench_handles.values()), loc="center left",
               bbox_to_anchor=(0.76, 0.22), title="Benchmark",
               title_fontsize=8, fontsize=7)
    ax.set_xlabel("Total tokens (millions)")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(left=0)
    ax.set_ylim(0.5, 0.9)
    ps.style_ax(ax, grid="both")
    fig.tight_layout(rect=[0, 0, 0.74, 1])
    return fig


# ---------------------------------------------------------------------------
# F5 — distance-decile detection / correction / false-opposition rates
# ---------------------------------------------------------------------------

_F5_METRICS = [
    ("p_detect_given_wrong", "P(detect | wrong)", "(a)"),
    ("p_corrected_given_wrong", "P(corrected | wrong)", "(b)"),
    ("p_false_opposition", "P(false opposition)", "(c)"),
]


def plot_f5_decile_rates(data, bench):
    """data = {decile_edges, by_condition: {cond: {deciles: [rows]}}}."""
    by_cond = data["by_condition"]
    fig, axes = plt.subplots(3, 1, sharex=True,
                             figsize=(ps.SINGLE_COL, ps.mm(112)))
    for ax, (key, ylab, tag) in zip(axes, _F5_METRICS):
        ymax = 0.0
        for cond in sorted(by_cond):
            rows = by_cond[cond]["deciles"]
            xs = [r["decile"] for r in rows]
            ys = [np.nan if r[key] is None else r[key] for r in rows]
            ymax = max(ymax, np.nanmax(ys) if len(ys) else 0.0)
            big = cond == "C5"
            ax.plot(xs, ys, marker="o", ms=3 if big else 2.4,
                    lw=1.6 if big else 0.9,
                    color=ps.cond_color(cond),
                    zorder=3 if big else 2, label=cond)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, min(1.0, ymax * 1.18 + 1e-9))
        ps.style_ax(ax, grid="y")
        ps.panel_label(ax, tag, x=-0.24, y=1.10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=4, columnspacing=0.9, handletextpad=0.4)
    axes[-1].set_xlabel("Distance decile (1 = nearest, 10 = farthest)")
    axes[-1].set_xticks(range(1, 11))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ---------------------------------------------------------------------------
# F6 — C0 -> final correctness transitions (stacked fractions)
# ---------------------------------------------------------------------------

_F6_CATS = [
    ("wrong_to_correct", "wrong→correct", ps.OI_GREEN),
    ("wrong_to_wrong", "wrong→wrong", ps.NEUTRAL_GREY),
    ("correct_to_correct", "correct→correct", ps.OI_BLUE),
    ("correct_to_wrong", "correct→wrong", ps.OI_VERMILLION),
]


def plot_f6_error_transitions(data, bench):
    """data[cond] = {counts: {...}, n: int, ...}."""
    conds = _conds_in_order(data)
    fig, ax = plt.subplots(figsize=(ps.SINGLE_COL, ps.mm(88)))
    xs = np.arange(len(conds))
    bottoms = np.zeros(len(conds))
    handles = []
    for cat, lab, col in _F6_CATS:
        vals = np.array([data[c]["counts"][cat] / data[c]["n"]
                         if data[c]["n"] else 0.0 for c in conds])
        b = ax.bar(xs, vals, bottom=bottoms, width=0.68, label=lab, color=col,
                   edgecolor="black", linewidth=0.3)
        handles.append(b)
        bottoms += vals
    ax.set_xticks(xs)
    ax.set_xticklabels(conds)
    ax.set_ylabel("Fraction of questions")
    ax.set_ylim(0, 1.0)
    ps.style_ax(ax, grid="y")
    ps.highlight_c5_tick(ax)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, fontsize=7, handlelength=1.0, handletextpad=0.4,
               columnspacing=0.8)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig
