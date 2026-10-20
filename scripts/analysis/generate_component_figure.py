#!/usr/bin/env python3
"""Component figure: (a) full model, (b) epistemic uncertainty, (c) precision prior.

Three panels side by side, drawn at print size for a full text width (6.5 in),
in the visual language of generate_architecture_figure.py (its helpers are
reused). Each panel works in inches: one data unit is one inch.

The bars in (b) and (c) are schematic and tagged "illustrative" in the figure.
The bars in (c) are computed from the precision prior with the paper's beta
values; alpha and the base distributions are made up to show the mechanism.
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse

from generate_architecture_figure import (EXPERTS, INK, ROSE_T, ROUTER_FC, SUBINK,
                                          beta, card, fmt_beta, router, wire)

FIG_W, DRAW_H, HEAD_H = 6.5, 3.0, 0.45
FIG_H = DRAW_H + HEAD_H
PANELS = [  # left edge, width (inches), title, subtitle
    (0.00, 2.30, "(a) Full model", "shared trunk, top-1 routing"),
    (2.38, 1.92, "(b) Epistemic uncertainty", "knowing when it doesn't know"),
    (4.36, 2.14, "(c) Precision prior", "deferring to precision when unsure"),
]
FS, SUBFS = 8.0, 7.0   # card title / subtitle sizes in print points
ALPHA_DEMO = 2.0       # illustrative strength for panel (c), not a fitted value
PANEL_FC = "#F6F6F8"


def panel(fig, left, width, title, subtitle):
    ax = fig.add_axes([left / FIG_W, 0.0, width / FIG_W, DRAW_H / FIG_H])
    ax.set_xlim(0, width); ax.set_ylim(0, DRAW_H); ax.axis("off")
    x = (left + width / 2) / FIG_W
    fig.text(x, (DRAW_H + 0.32) / FIG_H, title, ha="center", va="center", fontsize=9, color=INK)
    fig.text(x, (DRAW_H + 0.15) / FIG_H, subtitle, ha="center", va="center", fontsize=7.5,
             color=SUBINK)
    return ax


def c(ax, x, y, w, h, text, fc, **kw):
    card(ax, x, y, w, h, text, fc, fs=kw.pop("fs", FS), subfs=kw.pop("subfs", SUBFS),
         pad=0.03, lw=0.8, gap=0.6, **kw)


def w_(ax, p, q, rad=0.0, lw=0.9, **kw):
    wire(ax, p, q, rad=rad, lw=lw, ms=7, **kw)


def note(ax, x, y, s, ha="left", color=SUBINK, **kw):
    ax.text(x, y, s, ha=ha, va="center", fontsize=kw.pop("fs", 7), color=color, **kw)


def well(ax, x, y, w, h):
    """Neutral rounded area that holds the illustrative bars."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=PANEL_FC, ec="none",
                                zorder=0))


def bars(ax, x0, y0, vals, bw, gap, hmax=0.5, top=None, names=False):
    """Schematic routing distribution: one pastel bar per expert on a thin baseline."""
    span = 3 * bw + 2 * gap
    ax.plot([x0 - 0.03, x0 + span + 0.03], [y0, y0], color=SUBINK, lw=0.6, solid_capstyle="butt")
    for k, ((name, col, _, _), v) in enumerate(zip(EXPERTS, vals)):
        x = x0 + k * (bw + gap)
        ax.add_patch(FancyBboxPatch((x, y0), bw, v * hmax, boxstyle="round,pad=0,rounding_size=0.012",
                                    fc=col, ec=INK if k == top else "none", lw=0.9))
        if names:
            ax.text(x + bw / 2, y0 - 0.04, name, ha="center", va="top", fontsize=7, color=SUBINK)
    return span


def full_model(ax):
    cx = 1.15
    c(ax, 0.30, 2.80, 1.70, 0.16, "Audio embedding  $z$", "#FEF4B5")
    w_(ax, (0.52, 2.76), (0.52, 2.59))
    c(ax, 0.04, 2.25, 0.95, 0.30, "Shared 8-bit Trunk", "#EFEFF1", sub="once per sample")
    ry = 1.94
    a, b = router(ax, cx, ry, fs=7.5)
    w_(ax, (cx, 2.76), (cx, ry + b + 0.02))
    c(ax, 1.63, ry - 0.17, 0.63, 0.34, "Epistemic", "#FBEBE9", sub="uncertainty  $u$",
      dashed=True, tc=ROSE_T, subc=ROSE_T)
    w_(ax, (cx + a + 0.02, ry), (1.58, ry), color=ROSE_T)
    ty = 1.33
    c(ax, cx - 0.48, ty - 0.14, 0.96, 0.28, "Top-1", "#F6F6F8", sub="dispatch to head $i^*$")
    w_(ax, (cx, ry - b - 0.02), (cx, ty + 0.19))
    note(ax, cx + 0.05, (ry - b + ty + 0.17) / 2, "$p_i$", fs=7.5)
    # precision prior: uncertainty feeds back into routing
    w_(ax, (1.94, ry - 0.22), (cx + 0.53, ty), rad=0.42, color=ROSE_T, lw=1.1)
    note(ax, 2.00, 1.20, "precision\nprior", ha="center", color=ROSE_T, linespacing=1.0)
    # trunk features into the dispatch
    w_(ax, (0.52, 2.21), (cx - 0.53, ty), rad=0.35)
    note(ax, 0.52, 1.58, "features")
    ey = 0.60
    for k, (name, col, bits, _) in enumerate(EXPERTS):
        x = 0.04 + 0.775 * k
        c(ax, x, ey, 0.68, 0.30, name, col, sub=f"$\\beta={fmt_beta(bits)}$")
        w_(ax, (cx, ty - 0.18), (x + 0.34, ey + 0.34), rad=(0.16, 0.0, -0.16)[k])
        w_(ax, (x + 0.34, ey - 0.04), (cx, 0.31), rad=(-0.14, 0.0, 0.14)[k])
    c(ax, cx - 0.56, 0.03, 1.12, 0.25, "Output", "#EFEFF1",
      sub=r"head $i^*$ logits $\times\, p^{\mathrm{UA}}_{i^*}$")


def uncertainty(ax):
    W = 1.92; cx = W / 2
    c(ax, cx - 0.50, 2.80, 1.00, 0.16, "embedding  $z$", "#FEF4B5")
    # M stochastic passes: ghosted copies of the router behind the front one
    rx, ry = cx - 0.12, 2.30
    a, b = router(ax, rx, ry, fs=7.5, zorder=4)
    for k, alpha in ((2, 0.30), (1, 0.55)):
        ax.add_patch(Ellipse((rx + 0.07 * k, ry + 0.05 * k), 2 * a, 2 * b,
                             fc=ROUTER_FC, ec="white", lw=0.8, alpha=alpha, zorder=4 - k))
    w_(ax, (rx, 2.76), (rx, ry + b + 0.02))
    note(ax, rx + a + 0.20, ry, "$M$ dropout\npasses", linespacing=1.05)
    w_(ax, (rx, ry - b - 0.02), (rx, 1.79))
    # one routing distribution per pass
    well(ax, 0.04, 1.10, W - 0.08, 0.64)
    note(ax, 0.06, 1.67, "$p_i^{(m)}$ per pass")
    note(ax, W - 0.06, 1.67, "illustrative", ha="right")
    passes = [(0.55, 0.30, 0.15), (0.30, 0.45, 0.25), (0.40, 0.20, 0.40)]
    for x, v, lab in zip((0.14, 0.70, 1.36), passes, ("$m{=}1$", "$m{=}2$", "$m{=}M$")):
        span = bars(ax, x, 1.28, v, bw=0.11, gap=0.04, hmax=0.5)
        note(ax, x + span / 2, 1.19, lab, ha="center")
    note(ax, 1.235, 1.37, r"$\cdots$", ha="center", fs=8)
    w_(ax, (cx, 1.05), (cx, 0.89))
    c(ax, cx - 0.62, 0.57, 1.24, 0.29, "spread across passes", "#F6F6F8", fs=7.5,
      sub=r"$u_{\mathrm{raw}} = \Sigma_i\, \mathrm{Var}_m\, p_i^{(m)}$")
    w_(ax, (cx, 0.52), (cx, 0.36), color=ROSE_T)
    note(ax, cx + 0.06, 0.44, "running min/max", color=ROSE_T)
    c(ax, cx - 0.62, 0.03, 1.24, 0.29, "Epistemic uncertainty", "#FBEBE9",
      sub=r"$u \in [0, 1]$", dashed=True, tc=ROSE_T, subc=ROSE_T)


def precision_prior(ax):
    W = 2.14
    note(ax, W / 2, 2.88, r"$p_i^{\mathrm{UA}} \propto p_i \cdot \exp(\alpha\, u\, \beta_i)$",
         ha="center", color=ROSE_T, fs=8.5)
    note(ax, W / 2, 2.68, "    ".join(f"{n}  $\\beta={fmt_beta(bits)}$" for n, _, bits, _ in EXPERTS),
         ha="center")
    well(ax, 0.04, 0.58, W - 0.08, 1.92)
    note(ax, W / 2, 2.40, "illustrative", ha="center")
    cases = [  # column centre, heading, u, base distribution (schematic)
        (0.56, "confident, $u \\approx 0$", 0.0, (0.70, 0.20, 0.10), "no change"),
        (1.62, "uncertain, $u \\approx 1$", 1.0, (0.42, 0.33, 0.25), "shift to Q8"),
    ]
    bw, gap = 0.11, 0.15
    span = 3 * bw + 2 * gap
    for x, head, u, p, verb in cases:
        q = [pi * math.exp(ALPHA_DEMO * u * beta(bits)) for pi, (_, _, bits, _) in zip(p, EXPERTS)]
        q = [qi / sum(q) for qi in q]
        top = max(range(3), key=q.__getitem__)
        x0 = x - span / 2
        note(ax, x, 2.24, head, ha="center", color=INK, fs=7.5)
        bars(ax, x0, 1.68, p, bw=bw, gap=gap, names=True)
        note(ax, x0 - 0.05, 1.74, "$p_i$", ha="right", fs=7.5)
        w_(ax, (x, 1.48), (x, 1.28), color=ROSE_T, lw=1.1)
        note(ax, x - 0.06, 1.38, verb, ha="right", color=ROSE_T)
        bars(ax, x0, 0.80, q, bw=bw, gap=gap, top=top)
        note(ax, x0 - 0.05, 0.86, r"$p_i^{\mathrm{UA}}$", ha="right", fs=7.5)
        bx = x0 + top * (bw + gap) + bw / 2
        ax.text(bx, 0.80 + q[top] * 0.5 + 0.03, "top-1", ha="center", va="bottom", fontsize=7,
                color=INK)
        name, col, bits, kind = EXPERTS[top]
        w_(ax, (bx, 0.76), (x, 0.40))
        c(ax, x - 0.40, 0.04, 0.80, 0.30, name, col, sub=f"{kind}, $\\beta={fmt_beta(bits)}$")


def main():
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    axes = [panel(fig, *p) for p in PANELS]
    full_model(axes[0]); uncertainty(axes[1]); precision_prior(axes[2])
    fig.savefig("scripts/analysis/outputs-paper/architecture_panels.pdf")
    print("saved")


if __name__ == "__main__":
    main()
