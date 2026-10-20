#!/usr/bin/env python3
"""Architecture figure in the original diagram's visual language.

Borderless pastel cards, soft yellow input, neutral chips, dashed rose
uncertainty card, portrait flow. The wiring follows QWave/moe.py with
shared_trunk=True and top_k=1: the embedding z feeds the shared trunk and,
in parallel, the Bayesian router; the router's epistemic uncertainty feeds
back into routing through the precision prior; top-1 sends the trunk
features to the selected head only, and its logits are scaled by its
routing probability.

The style helpers (card, wire, router, beta) are reused by
generate_component_figure.py.
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
from matplotlib.transforms import offset_copy

INK    = "#1F2937"
SUBINK = "#6B7280"
ROSE_T = "#8C3A34"
ROUTER_FC, ROUTER_TC = "#8FCB9B", "#123C27"
EXPERTS = [  # name, colour, bit-width (BitNet counts as 1.58 bits), subtitle prefix
    ("BitNet", "#E0D4FF", 1.58, "ternary"),
    ("Q4",     "#B8E3FA", 4.0,  "4-bit"),
    ("Q8",     "#FCC155", 8.0,  "8-bit"),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    # Write glyphs as outlines in SVG. The default stores <use> references
    # into a font table, which renderers lacking the font (GitHub) collapse
    # into solid blobs.
    "svg.fonttype": "path",
    # Embed TrueType (Type 42) fonts in PDFs instead of Type 3.
    "pdf.fonttype": 42,
})


def beta(bits, lo=1.58, hi=8.0):
    """Precision weight of QWave/moe.py: bit-width rescaled linearly to [0, 1]."""
    return (bits - lo) / (hi - lo)


def fmt_beta(bits):
    return f"{beta(bits):.2f}".rstrip("0").rstrip(".")


def _data_width(ax, artist):
    """Rendered width of an artist in x data units."""
    bb = artist.get_window_extent(ax.figure.canvas.get_renderer())
    x0, x1 = ax.get_xlim()
    return bb.width * (x1 - x0) / ax.bbox.width


def card(ax, x, y, w, h, text, fc, fs=18.2, sub=None, subfs=13.5, note=None, notefs=12.5,
         dashed=False, tc=INK, subc=SUBINK, pad=0.10, lw=1.6, gap=0.45):
    kw = dict(fc=fc, ec=ROSE_T if dashed else "none", lw=lw)
    if dashed: kw["ls"] = (0, (4, 3))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad}", **kw))
    # Stack title, subtitle and note about the card's centre line. Offsets are
    # in points, so the card keeps its proportions at any figure scale.
    lines = [(s, f, c, wt) for s, f, c, wt in [(text, fs, tc, "bold"), (sub, subfs, subc, "normal"),
                                              (note, notefs, subc, "normal")] if s]
    offs = [0.0]
    for i, ((_, f0, _, _), (_, f1, _, _)) in enumerate(zip(lines, lines[1:])):
        offs.append(offs[-1] - (gap if i == 0 else 0.55) * (f0 + f1))
    shift = -(offs[0] + offs[-1]) / 2
    for (s, f, c, wt), o in zip(lines, offs):
        t = ax.text(x + w/2, y + h/2, s, ha="center", va="center", fontsize=f, fontweight=wt,
                    color=c, transform=offset_copy(ax.transData, ax.figure, y=o + shift, units="points"))
        assert _data_width(ax, t) <= w, f"{s!r} overflows its card"


def wire(ax, p, q, rad=0.0, color=INK, lw=1.7, ls="-", ms=15):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))


def router(ax, cx, cy, text="Bayesian\nRouter", fs=15.5, aspect=1.12, margin=1.08, alpha=1.0, zorder=2):
    """Green router ellipse sized to its measured label so no text touches the edge.

    Returns the semi-axes (a, b) in data units, for attaching wires.
    """
    t = ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, fontweight="bold",
                color=ROUTER_TC, linespacing=1.1, zorder=zorder + 1)
    bb = t.get_window_extent(ax.figure.canvas.get_renderer())
    # Smallest ellipse of this display aspect that contains the text box, enlarged by `margin`.
    b = margin * math.hypot(bb.width / 2 / aspect, bb.height / 2)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    a = aspect * b * (x1 - x0) / ax.bbox.width
    b = b * (y1 - y0) / ax.bbox.height
    ax.add_patch(Ellipse((cx, cy), 2 * a, 2 * b, fc=ROUTER_FC, ec="none", alpha=alpha, zorder=zorder))
    return a, b


def main():
    print("beta: " + ", ".join(f"{n} {beta(b):.4f}" for n, _, b, _ in EXPERTS))

    fig, ax = plt.subplots(figsize=(7.0, 8.6))
    ax.set_xlim(0, 10); ax.set_ylim(2.1, 13.55); ax.axis("off")

    # 1. input: the embedding feeds the trunk and the router in parallel
    card(ax, 1.3, 12.35, 7.4, 0.95, "Audio embedding", "#FEF4B5", sub=r"$z \in \mathbb{R}^d$")
    wire(ax, (2.3, 12.30), (2.3, 11.85))

    # 2. shared trunk, computed once per sample
    card(ax, 0.1, 10.85, 4.4, 0.95, "Shared 8-bit Trunk", "#EFEFF1",
         sub=r"$d \times 640$, once per sample")

    # 3. router circle + uncertainty card
    ry = 9.45
    a, b = router(ax, 5.0, ry)
    wire(ax, (5.0, 12.30), (5.0, ry + b + 0.05))
    card(ax, 6.85, 8.80, 2.85, 1.30, "Epistemic", "#FBEBE9", fs=16.2, sub="uncertainty  $u$",
         note="MC dropout", dashed=True, tc=ROSE_T, subc=ROSE_T)
    wire(ax, (5.0 + a + 0.05, ry), (6.72, ry), color=ROSE_T, lw=1.6)

    # 4. top-1 chip: base routing from the router, prior from the uncertainty,
    # trunk features dispatched to the selected head
    ty = 7.45
    card(ax, 3.35, ty - 0.475, 3.3, 0.95, "Top-1", "#F6F6F8", fs=16.9, sub="dispatch to head $i^*$")
    wire(ax, (5.0, ry - b - 0.05), (5.0, ty + 0.62))
    ax.text(5.15, (ry - b + ty + 0.57) / 2, "$p_i$", fontsize=14.2, color=SUBINK, ha="left", va="center")

    # precision prior: uncertainty feeds BACK into routing
    wire(ax, (8.25, 8.68), (6.80, ty), rad=0.42, color=ROSE_T, lw=2.1)
    ax.text(8.75, 7.55, "precision prior\n" + r"$p_i^{\mathrm{UA}} \propto p_i\, e^{\,\alpha u \beta_i}$",
            fontsize=14.2, color=ROSE_T, ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none"), zorder=4)

    # trunk features into the dispatch
    wire(ax, (2.3, 10.73), (3.20, ty), rad=0.35)
    ax.text(1.80, 9.05, "features", fontsize=13.5, color=SUBINK, ha="right", va="center")

    # 5. experts: only head i* runs
    ey = 4.85
    for k, (name, c, bits, kind) in enumerate(EXPERTS):
        x = 0.20 + 3.35 * k
        card(ax, x, ey, 2.9, 1.05, f"{name} Expert", c, fs=16.2,
             sub=f"{kind},  $\\beta={fmt_beta(bits)}$")
        wire(ax, (5.0, ty - 0.60), (x + 1.45, ey + 1.12), rad=(0.16, 0.0, -0.16)[k])
        wire(ax, (x + 1.45, ey - 0.05), (5.0, 3.20), rad=(-0.14, 0.0, 0.14)[k])

    # 6. output: the selected head's logits, scaled by its routing probability
    card(ax, 2.90, 2.30, 4.2, 0.9, "Output", "#EFEFF1", fs=16.9,
         sub=r"head $i^*$ logits $\times\, p^{\mathrm{UA}}_{i^*}$")

    plt.savefig("scripts/analysis/outputs-paper/architecture.pdf", bbox_inches="tight")
    plt.savefig("/tmp/arch_review.png", dpi=140, bbox_inches="tight")
    # README copy as SVG: GitHub renders it as vector, so it stays sharp at any
    # zoom and on any display, unlike a rasterised PNG.
    plt.savefig("docs/architecture.svg", format="svg", bbox_inches="tight",
                facecolor="white", edgecolor="none")
    # PNG fallback at high DPI for contexts that cannot render SVG.
    plt.savefig("docs/architecture.png", dpi=400, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    # matplotlib still writes an alpha channel, which GitHub renders as a
    # transparent (so dark-mode-grey) ground. Flatten it onto white.
    try:
        from PIL import Image
        _im = Image.open("docs/architecture.png")
        if _im.mode == "RGBA":
            _bg = Image.new("RGBA", _im.size, (255, 255, 255, 255))
            _im = Image.alpha_composite(_bg, _im)
        _im.convert("RGB").save("docs/architecture.png", optimize=True)
    except ImportError:
        pass
    print("saved")


if __name__ == "__main__":
    main()
