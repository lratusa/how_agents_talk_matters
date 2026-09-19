"""Shared publication style for all paper figures (F1-F8).

Nature/Science-like defaults: Okabe-Ito colorblind-safe palette, DejaVu Sans
(base 8 pt, axis titles 8 pt, ticks 7 pt, legend 7 pt, panel labels 9 pt
bold), 0.6 pt spines (left/bottom only), 0.3 pt light-grey grids where
needed, no chartjunk (no 3D, no shadows).

Sizes: single column ~89 mm (3.5 in), double column ~183 mm (7.2 in).
save_fig() writes both PNG (dpi=300) and PDF (vector, editable text).

Condition colors are fixed across every figure; C5 (FPRR) is always
vermillion #D55E00.

Usage from sibling scripts:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pubstyle
    pubstyle.apply()
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Palette (Okabe-Ito, colorblind safe)
# ---------------------------------------------------------------------------

OI_BLUE = "#0072B2"
OI_VERMILLION = "#D55E00"
OI_GREEN = "#009E73"
OI_PURPLE = "#CC79A7"
OI_ORANGE = "#E69F00"
OI_SKY = "#56B4E9"
OI_YELLOW = "#F0E442"
OI_BLACK = "#000000"
NEUTRAL_GREY = "#999999"  # neutral control tone (C9), outside the palette

# Fixed condition -> color mapping, identical in every figure.
COND_COLORS = {
    "C0": OI_BLACK,
    "C1": OI_SKY,
    "C2": OI_GREEN,
    "C3": OI_BLUE,
    "C4": OI_ORANGE,
    "C5": OI_VERMILLION,   # FPRR — highlighted everywhere
    "C6": OI_PURPLE,
    "C7": OI_YELLOW,       # gsm8k pilot only
    "C9": NEUTRAL_GREY,
}

COND_LABELS = {
    "C0": "C0 single",
    "C1": "C1 majority vote",
    "C2": "C2 synth (no review)",
    "C3": "C3 random",
    "C4": "C4 nearest",
    "C5": "C5 FPRR",
    "C6": "C6 max-weight",
    "C7": "C7",
    "C9": "C9 self-review",
}

# Semantic colors for correctness / transitions (not condition encodings).
CORRECT_COLOR = OI_GREEN
WRONG_COLOR = OI_VERMILLION

# Sequential heatmap colormap derived from the palette (white -> OI blue).
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HEAT_CMAP = LinearSegmentedColormap.from_list("oi_white_blue",
                                              ["#FFFFFF", OI_BLUE])

# ---------------------------------------------------------------------------
# Sizes
# ---------------------------------------------------------------------------

MM_PER_INCH = 25.4
SINGLE_COL = 89 / MM_PER_INCH    # ~3.5 in
DOUBLE_COL = 183 / MM_PER_INCH   # ~7.2 in


def mm(v):
    return v / MM_PER_INCH


# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------

def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.0,
        "grid.linewidth": 0.3,
        "grid.alpha": 0.3,
        "grid.color": "0.75",
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        "errorbar.capsize": 2.0,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,      # TrueType, editable text in the PDF
        "ps.fonttype": 42,
        "figure.constrained_layout.use": False,
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def style_ax(ax, grid=None):
    """Despine + optional light grid ('x', 'y', 'both')."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, axis=grid)
        ax.set_axisbelow(True)
    return ax


def panel_label(ax, label, x=-0.16, y=1.05):
    """Nature-style panel tag '(a)' — 9 pt bold, top-left of the axes."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="right", clip_on=False)


def cond_color(cond):
    return COND_COLORS.get(cond, NEUTRAL_GREY)


def highlight_c5_tick(ax):
    """Color the C5 x-tick label vermillion (C5 highlight convention)."""
    for tick in ax.get_xticklabels():
        if tick.get_text() == "C5":
            tick.set_color(OI_VERMILLION)
            tick.set_fontweight("bold")


def save_fig(fig, path_base):
    """Write <path_base>.png (300 dpi) and <path_base>.pdf (vector)."""
    fig.savefig(path_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(path_base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("  figure: %s" % os.path.basename(path_base))
