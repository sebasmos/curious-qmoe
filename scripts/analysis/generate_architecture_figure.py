#!/usr/bin/env python3
"""Architecture figure in the original diagram's visual language.

Borderless pastel cards, soft yellow input, neutral chips, dashed rose
uncertainty card, portrait flow. Content corrected relative to the original:
three experts labeled with bit-width and beta, the shared trunk, and epistemic
uncertainty feeding BACK into routing through the precision prior (the
original drew it as a dead-end output).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

INK    = "#1F2937"
SUBINK = "#6B7280"
ROSE_T = "#8C3A34"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
})

def card(ax, x, y, w, h, text, fc, fs=18.2, sub=None, dashed=False, tc=INK, subc=SUBINK):
    kw = dict(fc=fc, ec=ROSE_T if dashed else "none", lw=1.6)
    if dashed: kw["ls"] = (0, (4, 3))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10", **kw))
    cy = y + h/2 + (0.155 if sub else 0)
    ax.text(x + w/2, cy, text, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=tc)
    if sub:
        ax.text(x + w/2, y + h/2 - 0.185, sub, ha="center", va="center",
                fontsize=13.5, color=subc)

def wire(ax, p, q, rad=0.0, color=INK, lw=1.7, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=15,
                                 color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))

fig, ax = plt.subplots(figsize=(7.0, 8.6))
ax.set_xlim(0, 10); ax.set_ylim(2.1, 13.55); ax.axis("off")

# 1. input
card(ax, 2.6, 12.35, 4.8, 0.95, "Audio Input", "#FEF4B5", sub="embeddings, 1536-d")
wire(ax, (5.0, 12.30), (5.0, 11.80))
ax.text(5.32, 12.07, "data flow", fontsize=12.8, color=SUBINK, ha="left", va="center")

# 2. shared trunk
card(ax, 2.15, 10.85, 5.7, 0.95, "Shared 8-bit Trunk", "#EFEFF1",
     sub="1536 × 640, computed once per sample")
wire(ax, (5.0, 10.80), (5.0, 10.28))

# 3. router circle + uncertainty card
ax.add_patch(Circle((5.0, 9.45), 0.78, fc="#8FCB9B", ec="none", zorder=2))
ax.text(5.0, 9.62, "Bayesian", ha="center", fontsize=15.5, fontweight="bold",
        color="#123C27", zorder=3)
ax.text(5.0, 9.28, "Router", ha="center", fontsize=15.5, fontweight="bold",
        color="#123C27", zorder=3)
ax.text(3.05, 9.62, "base routing $p_i$", fontsize=13.5, color=SUBINK, ha="right")
ax.text(3.05, 9.30, "MC dropout", fontsize=13.5, color=SUBINK, ha="right")

card(ax, 6.85, 8.95, 2.85, 1.05, "Epistemic", "#FBEBE9", fs=16.2,
     sub="uncertainty  $u$", dashed=True, tc=ROSE_T, subc=ROSE_T)
wire(ax, (5.82, 9.45), (6.80, 9.45), color=ROSE_T, lw=1.6)

# 4. precision prior: uncertainty feeds BACK into routing
wire(ax, (8.25, 8.88), (6.18, 7.45), rad=0.42, color=ROSE_T, lw=2.1)
ax.text(8.55, 7.72, "precision prior\n" + r"$p_i \cdot e^{\,\alpha u \beta_i}$",
        fontsize=14.2, color=ROSE_T, ha="center", va="center", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none"), zorder=4)

# 5. top-k chip
wire(ax, (5.0, 8.62), (5.0, 7.90))
card(ax, 3.9, 7.05, 2.2, 0.8, "Top-k", "#F6F6F8", fs=16.9)
ax.text(3.62, 7.45, "routing", fontsize=12.8, color=SUBINK, ha="right")

# 6. experts
experts = [
    (0.35, "#E0D4FF", "BitNet Expert", "ternary,  $\\beta=0$"),
    (3.65, "#B8E3FA", "Q4 Expert", "4-bit,  $\\beta=0.4$"),
    (6.95, "#FCC155", "Q8 Expert", "8-bit,  $\\beta=1$"),
]
for x, c, t, s_ in experts:
    card(ax, x, 5.0, 2.7, 1.05, t, c, fs=16.2, sub=s_)
    wire(ax, (5.0, 7.00), (x + 1.25, 6.12), rad=(0.16 if x < 3 else (-0.16 if x > 5 else 0.0)))


# 7. aggregate
for x, _, _, _ in experts:
    wire(ax, (x + 1.25, 4.95), (5.0, 3.35), rad=(-0.14 if x < 3 else (0.14 if x > 5 else 0.0)))
card(ax, 3.25, 2.45, 3.5, 0.9, "Aggregated Output", "#EFEFF1", fs=16.9,
     sub="class logits")

plt.savefig("scripts/analysis/outputs-paper/architecture.pdf", bbox_inches="tight")
plt.savefig("/tmp/arch_review.png", dpi=140, bbox_inches="tight")
print("saved")
