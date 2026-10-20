#!/usr/bin/env python3
"""Figure 2: classifier confidence of the held-out samples routed to each expert.

Reads the routing_results.json written by routing_analysis.py (one file, or the
directory holding it, per router setting; one panel each, sharing the
confidence axis). Schema used: config.expert_names (experts in config order,
which is ascending precision in the paper runs), config.run_dir (the panel
title takes alpha from its "_a<alpha>_" tag) and routing[*].expert /
routing[*].confidence (the classifier's max class probability).

Each expert row shows the density of its samples' confidence (reflected at
1.0, cut at the observed range), the individual samples, the interquartile
range as a bar and the median as a dot, plus n and the share of the panel's
samples. --palette figure1 (default) fills each density with the exact
expert card color of Figure 1 (generate_architecture_figure.py) and draws
the outline, samples and IQR bar in a darker step of the same hue, so those
marks clear 3:1 on white and the three experts stay apart under red-green
color blindness. --palette ramp uses one lavender hue, light to dark with
precision. The p value is a two-sided Mann-Whitney U test of the
highest-precision expert against all other samples in the panel, computed here
from the plotted data.

Usage:
    python scripts/analysis/generate_confidence_figure.py \
        --routing-json scripts/analysis/outputs-paper/routing_a0.3 \
                       scripts/analysis/outputs-paper/routing_a1.0 \
        --output confidence_distribution.pdf
"""
import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.transforms import blended_transform_factory
from scipy import stats

try:
    from scripts.analysis.generate_architecture_figure import EXPERTS as FIG1_EXPERTS
except ImportError:                       # run as a script from scripts/analysis
    from generate_architecture_figure import EXPERTS as FIG1_EXPERTS

logging.getLogger("fontTools").setLevel(logging.ERROR)  # font subsetting chatter

INK, INK2 = "#1a1a19", "#52514e"          # primary and secondary text
GRID, AXIS = "#e1e0d9", "#a9a89f"         # hairline grid, baseline
LIGHT, DARK = "#b09ee3", "#6140a0"        # lavender ramp ends (OKLCH h=296)
RAMP3 = ["#b09ee3", "#896cc9", "#6140a0"]  # validated ordinal ramp, 3 experts
FIG1_FILL = {name: color for name, color, *_ in FIG1_EXPERTS}
# Card hues stepped down in OKLCH lightness (0.66, 0.52, 0.66) for the marks:
# all >= 3:1 on white, worst pair CVD dE 11.3 and normal-vision dE 18.5.
FIG1_MARK = {"BitNet": "#9b81d2", "Q4": "#137298", "Q8": "#b9892d"}

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.5,
    "pdf.fonttype": 42,                   # embedded TrueType, no Type 3
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "xtick.major.pad": 2,
}


def tint(c, a):
    """Opaque mix of color c with white (a = share of c), for print-safe fills."""
    return tuple(a * np.array(to_rgb(c)) + (1 - a))


def ramp(n):
    if n == 3:
        return RAMP3
    a, b = np.array(to_rgb(LIGHT)), np.array(to_rgb(DARK))
    return [tuple(a + (b - a) * t) for t in np.linspace(0, 1, max(n, 2))][:n]


def colors(names, palette):
    """(fill, mark) per expert. Figure 1 colors follow the expert by name."""
    if palette == "ramp":
        return [(tint(c, 0.35), c) for c in ramp(len(names))]
    missing = [n for n in names if n not in FIG1_MARK]
    if missing:
        raise SystemExit(f"--palette figure1 has no color for {missing}; use --palette ramp")
    return [(FIG1_FILL[n], FIG1_MARK[n]) for n in names]


def load_routing(path):
    """One panel's data from a routing_results.json (or its directory)."""
    p = Path(path)
    if p.is_dir():
        p = p / "routing_results.json"
    d = json.loads(p.read_text())
    cfg, recs = d.get("config", {}), d["routing"]
    names = cfg.get("expert_names")
    if not names:
        by_id = {r["expert"]: r.get("expert_name", f"E{r['expert']}") for r in recs}
        names = [by_id.get(i, f"E{i}") for i in range(max(by_id) + 1)]
    conf = [np.array([r["confidence"] for r in recs if r["expert"] == e], float)
            for e in range(len(names))]
    m = re.search(r"_a(\d+(?:\.\d+)?)(?:_|/|$)", str(cfg.get("run_dir") or "")) \
        or re.search(r"_a(\d+(?:\.\d+)?)$", p.parent.name)
    title = f"α = {m.group(1)}" if m else p.parent.name
    return {"title": title, "names": list(names), "conf": conf, "total": len(recs)}


def compare_top(panel):
    """Two-sided Mann-Whitney U: highest-precision expert vs every other sample."""
    if len(panel["conf"]) < 2:
        return None
    top, rest = panel["conf"][-1], np.concatenate(panel["conf"][:-1])
    if len(top) < 2 or len(rest) < 2:
        return None
    p = stats.mannwhitneyu(top, rest, alternative="two-sided").pvalue
    return {"name": panel["names"][-1], "n_top": len(top), "n_rest": len(rest),
            "median_top": float(np.median(top)), "median_rest": float(np.median(rest)),
            "p": float(p)}


def density(v, grid):
    """Gaussian KDE (Silverman bandwidth), reflected at 1.0 so mass piled
    at full confidence is not lost past the bound."""
    sd, iqr = v.std(ddof=1), np.subtract(*np.percentile(v, [75, 25]))
    s = min(sd, iqr / 1.34) if iqr > 0 else sd
    h = max(0.9 * s * len(v) ** -0.2, 0.012)
    z = np.concatenate([v, 2.0 - v])
    return np.exp(-0.5 * ((grid[:, None] - z[None, :]) / h) ** 2).sum(1)


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.2g}"   # 2 s.f.: 0.046 stays 0.046


def draw_panel(ax, panel, rng, palette):
    names = panel["names"]
    k = len(names)
    lab = blended_transform_factory(ax.transAxes, ax.transData)
    for e, (name, v, (fill, c)) in enumerate(zip(names, panel["conf"], colors(names, palette))):
        y = k - 1 - e                     # lowest precision on top
        share = 100 * len(v) / panel["total"] if panel["total"] else 0
        ax.text(-0.03, y + 0.05, name, transform=lab, ha="right", va="bottom",
                fontsize=8, color=INK)
        ax.text(-0.03, y + 0.01, f"n = {len(v):,} ({share:.0f}%)", transform=lab,
                ha="right", va="top", fontsize=7, color=INK2)
        if not len(v):
            continue
        if len(v) >= 5 and v.std() > 0:
            grid = np.linspace(v.min(), v.max(), 256)
            dens = density(v, grid)
            top = y + 0.13 + 0.36 * dens / dens.max()
            ax.fill_between(grid, y + 0.13, top, color=fill, lw=0, zorder=1)
            ax.plot(grid, top, color=c, lw=0.8, zorder=2, solid_capstyle="round")
        ax.scatter(v, y - 0.24 + rng.uniform(-0.08, 0.08, len(v)), s=1.4,
                   color=c, alpha=0.8 if len(v) <= 150 else 0.4, lw=0, zorder=2)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([q1, q3], [y, y], color=c, lw=3.0, solid_capstyle="butt", zorder=3)
        ax.scatter([med], [y], s=15, color=INK, edgecolor="white", lw=0.8, zorder=4)
    ax.set_ylim(-0.38, k - 0.48)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=INK2, labelsize=7, length=2.5)
    ax.set_title(panel["title"], loc="left", fontsize=8, color=INK, pad=2)
    t = compare_top(panel)
    if t:
        ax.set_title(f"{t['name']} vs others, Mann-Whitney {fmt_p(t['p'])}",
                     loc="right", fontsize=7, color=INK2, pad=2)
    return t


def make_figure(panels, output, stamp=None, palette="figure1"):
    """Exact 3.25 in width (half of a 6.5 in text width), so the point sizes
    set here are the printed sizes."""
    lo = min(min((v.min() for v in p["conf"] if len(v)), default=1.0) for p in panels)
    xmin = min(0.2, np.floor(lo * 10) / 10)
    rng = np.random.default_rng(0)
    W, ROW, LEFT, RIGHT, TITLE, GAP, BOTTOM = 3.25, 0.34, 0.80, 0.09, 0.17, 0.12, 0.36
    TOP = 0.03 + (0.15 if stamp else 0)
    heights = [ROW * len(p["names"]) for p in panels]
    H = TOP + sum(heights) + TITLE * len(panels) + GAP * (len(panels) - 1) + BOTTOM
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(W, H))
        axes, y = [], H - TOP
        for h in heights:
            y -= TITLE + h
            axes.append(fig.add_axes([LEFT / W, y / H, (W - LEFT - RIGHT) / W, h / H],
                                     sharex=axes[0] if axes else None))
            y -= GAP
        tests = [draw_panel(ax, p, rng, palette) for ax, p in zip(axes, panels)]
        for ax in axes[:-1]:
            ax.tick_params(axis="x", labelbottom=False, length=0)
        ax = axes[-1]
        ax.set_xlim(xmin - 0.01, 1.01)
        ax.set_xticks(np.round(np.arange(np.ceil(xmin * 5) / 5, 1.0001, 0.2), 1))
        ax.set_xlabel("Classifier confidence (max class probability)",
                      fontsize=7.5, color=INK, labelpad=3)
        if stamp:
            fig.text(0.06 / W, 1 - 0.03 / H, stamp, ha="left", va="top",
                     fontsize=7, color="#b42318", style="italic")
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
        plt.close(fig)
    return tests


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--routing-json", nargs="+", required=True,
                    help="routing_results.json file(s) or their directories, one panel each")
    ap.add_argument("--output", required=True, help="output PDF")
    ap.add_argument("--palette", choices=["figure1", "ramp"], default="figure1",
                    help="Figure 1 expert card colors, or one lavender hue by precision")
    ap.add_argument("--stamp", default=None, help="optional note printed above the plot")
    args = ap.parse_args(argv)
    panels = [load_routing(p) for p in args.routing_json]
    tests = make_figure(panels, args.output, args.stamp, args.palette)
    for p, t in zip(panels, tests):
        meds = ", ".join(f"{n} {np.median(v):.3f} (n={len(v)})" if len(v) else f"{n} - (n=0)"
                         for n, v in zip(p["names"], p["conf"]))
        print(f"{p['title']}: medians {meds}")
        if t:
            print(f"  {t['name']} vs others, two-sided Mann-Whitney U: p = {t['p']:.4g}")
    print("saved", args.output)


if __name__ == "__main__":
    main()
