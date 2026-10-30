"""Shared style for the draft-06 figures. Mirrors curious-qmoe/scripts/analysis/
generate_confidence_figure.py (fonts, ink/grid/axis greys, pdf.fonttype 42, 6.5 in
text width) and takes its hues from the expert card marks of
generate_architecture_figure.py (Q4 blue #137298, Q8 ochre #b9892d, BitNet
lavender #9b81d2). The three hues pass the dataviz validator on every pair
(worst CVD dE 11.3, normal-vision dE 18.5, all >= 3:1 on white)."""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.getLogger("fontTools").setLevel(logging.ERROR)

INK, INK2 = "#1a1a19", "#52514e"          # primary and secondary text
GRID, AXIS = "#e1e0d9", "#a9a89f"         # hairline grid, baseline
BLUE, OCHRE, LAV = "#137298", "#b9892d", "#9b81d2"   # Q4, Q8, BitNet mark colours
LAV_DARK = "#6140a0"                      # darker lavender for the emphasised marks
WIDTH = 6.5                               # exact text width, inches

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.5,
    "pdf.fonttype": 42,                   # embedded TrueType, no Type 3
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "legend.frameon": False,
}


def quiet_axes(ax, grid="y"):
    """Recessive spines, hairline grid, secondary-ink ticks."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    if grid in ("y", "both"):
        ax.yaxis.grid(True, color=GRID, lw=0.5)
    if grid in ("x", "both"):
        ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=7, length=2.5)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_color(INK2)


def panel_title(ax, text, letter=None):
    t = f"({letter}) {text}" if letter else text
    ax.set_title(t, loc="left", fontsize=8, color=INK, pad=3)


def save(fig, stem):
    """PDF (vector, TrueType fonts) and a PNG preview with the same name."""
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=220)
    plt.close(fig)
