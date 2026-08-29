#!/usr/bin/env python3
"""Confidence-distribution figure, redesigned in the architecture figure's
pastel language: each expert wears its card color, jittered sample points show
the raw distribution behind slim boxes, means as charcoal diamonds, honest
measured annotation. Usage:

    python generate_confidence_figure.py --routing-json <routing_results.json> \
        --output <out.pdf>
"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

INK, SUBINK = "#1F2937", "#6B7280"
FILLS = {0: "#E0D4FF", 1: "#B8E3FA", 2: "#FCC155"}
EDGES = {0: "#9C7FD4", 1: "#5FA8D3", 2: "#DE9A1F"}
NAMES = {0: "BitNet", 1: "Q4", 2: "Q8"}
QUANT = {0: "ternary", 1: "4-bit", 2: "8-bit"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
})

ap = argparse.ArgumentParser()
ap.add_argument("--routing-json", required=True)
ap.add_argument("--output", required=True)
args = ap.parse_args()

data = json.load(open(args.routing_json))
conf = {}
for r in data["routing"]:
    conf.setdefault(r["expert"], []).append(r["confidence"])
total = sum(len(v) for v in conf.values())

rng = np.random.default_rng(42)
fig, ax = plt.subplots(figsize=(8.2, 5.2))
for e in (0, 1, 2):
    vals = np.array(conf.get(e, []))
    if not len(vals):
        continue
    x = e + 1
    # jittered raw samples (capped) behind the box
    show = vals if len(vals) <= 260 else rng.choice(vals, 260, replace=False)
    ax.scatter(x + rng.uniform(-0.16, 0.16, len(show)), show, s=11,
               color=EDGES[e], alpha=0.22, linewidths=0, zorder=1)
    # slim pastel box
    bp = ax.boxplot([vals], positions=[x], widths=0.44, patch_artist=True,
                    showfliers=False, zorder=2,
                    boxprops=dict(fc=FILLS[e], ec=EDGES[e], lw=1.4, alpha=0.92),
                    medianprops=dict(color=INK, lw=2.0),
                    whiskerprops=dict(color=EDGES[e], lw=1.3),
                    capprops=dict(color=EDGES[e], lw=1.3))
    ax.plot(x, vals.mean(), marker="D", ms=6.5, color=INK, zorder=3)

# honest measured annotation, neutral chip style
q8 = np.array(conf.get(2, [])); others = np.array(conf.get(0, []) + conf.get(1, []))
t, pval = stats.ttest_ind(q8, others)
rel = (1 - q8.mean() / others.mean()) * 100
sig = f"p = {pval:.3f}" + (", n.s." if pval >= 0.05 else "")
ax.text(0.985, 0.965,
        f"Q8 mean {rel:.0f}% lower ({sig})",
        transform=ax.transAxes, ha="right", va="top", fontsize=10.5, color=SUBINK,
        bbox=dict(boxstyle="round,pad=0.45", fc="#F6F6F8", ec="none"))

ax.set_xticks([1, 2, 3])
ax.set_xticklabels([f"{NAMES[e]}\n{QUANT[e]}  ·  {100*len(conf.get(e,[]))/total:.0f}% of samples"
                    for e in (0, 1, 2)], fontsize=10.5, color=INK)
ax.set_ylabel("Prediction confidence", fontsize=12, color=INK)
lo = min(min(v) for v in conf.values())
ax.set_ylim(lo - 0.03, 1.03)
ax.yaxis.grid(True, color="#E5E7EB", lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#D1D5DB")
ax.tick_params(colors=SUBINK)

plt.tight_layout()
plt.savefig(args.output, bbox_inches="tight")
plt.savefig("/tmp/fig2_review.png", dpi=140, bbox_inches="tight")
print("saved", args.output)
