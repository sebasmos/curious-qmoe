#!/usr/bin/env python3
"""Architecture figure: shared-trunk MoE with uncertainty-directed precision prior.

Journal styling for Archives of Acoustics: serif (Times) fonts including math,
vector PDF output. Shows the real mechanism: uncertainty feeds the precision
prior, which reweights routing toward higher-precision heads (an earlier
diagram drew uncertainty as a dead-end side output).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

def box(ax, x, y, w, h, text, fc, fs=12.5, sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.055",
                                fc=fc, ec="0.15", lw=1.2))
    cy = y + h/2 + (0.17 if sub else 0)
    ax.text(x + w/2, cy, text, ha="center", va="center", fontsize=fs)
    if sub:
        ax.text(x + w/2, y + h/2 - 0.22, sub, ha="center", va="center",
                fontsize=9.5, style="italic", color="0.30")

def arrow(ax, p, q, color="0.15", lw=1.5, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}"))

fig, ax = plt.subplots(figsize=(8.6, 6.3))
ax.set_xlim(0, 10); ax.set_ylim(1.05, 10.05); ax.axis("off")

box(ax, 3.0, 9.1, 4.0, 0.75, "Audio embedding $z$  (1536-d)", "#f4f4f4")
box(ax, 2.2, 7.7, 5.6, 0.95, "Shared 8-bit trunk  $1536 \\times 640$", "#dbe9f9",
    sub="computed once per sample")
arrow(ax, (5.0, 9.08), (5.0, 8.73))

# router branch
box(ax, 0.30, 5.65, 2.55, 0.95, "Bayesian router", "#fdeacf",
    sub="$M$ MC-dropout passes")
box(ax, 0.30, 3.8, 2.55, 1.0, "Precision prior", "#fbd9a6",
    sub=r"$p_i^{\mathrm{UA}} \propto p_i\, e^{\,\alpha u \beta_i}$")
arrow(ax, (3.0, 7.66), (1.85, 6.68), rad=-0.14)
arrow(ax, (1.57, 5.60), (1.57, 4.88))
ax.text(1.78, 5.24, "base $p_i$,  uncertainty $u$", fontsize=9.5,
        style="italic", color="0.30", ha="left")

# heads
heads = [(3.55, "#f0e5f8", "BitNet head", r"ternary,  $\beta_i = 0$"),
         (5.75, "#e5d5f3", "Q4 head",     r"4-bit,  $\beta_i = 0.4$"),
         (7.95, "#d8c2ef", "Q8 head",     r"8-bit,  $\beta_i = 1$")]
for x, c, t, s_ in heads:
    box(ax, x, 5.65, 1.95, 1.0, t, c, fs=11.5, sub=s_)
    arrow(ax, (5.2, 7.66), (x + 0.975, 6.70))
ax.text(6.72, 5.38, r"increasing precision $\beta_i \longrightarrow$",
        fontsize=10, color="0.35", ha="center", style="italic")

# top-k sum
box(ax, 4.75, 3.05, 2.6, 0.8, "Top-$k$ weighted sum", "#dcefdc", fs=12)
for x, _, _, _ in heads:
    arrow(ax, (x + 0.975, 5.60), (6.05, 3.92))

# the mechanism arrow, clear of all text
arrow(ax, (2.88, 4.15), (4.70, 3.55), color="#a41e1e", lw=2.4, rad=-0.10)
ax.text(2.55, 3.20, "uncertain samples routed to\nhigher-precision heads",
        fontsize=10, color="#a41e1e", ha="center", style="italic")

box(ax, 5.2, 1.55, 1.7, 0.75, "Class logits", "#f4f4f4", fs=12)
arrow(ax, (6.05, 3.00), (6.05, 2.37))

plt.savefig("scripts/analysis/outputs-paper/architecture.pdf", bbox_inches="tight")
plt.savefig("/tmp/arch_review.png", dpi=150, bbox_inches="tight")
print("saved")
