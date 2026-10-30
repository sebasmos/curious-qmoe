#!/usr/bin/env python3
"""fig-gate-by-precision-v2: fig-gate-by-precision.py with the three group header rows set as headers (bold, flush
left at the figure's left edge, a thin grey rule above each, drawn across the label column and all four panels), so
they no longer read as data rows without points. v1 already asked for bold, but matplotlib drew Helvetica's regular face; the
headers now use Arial Bold. Same data, same layout; output names and JSON note changed.

v1 docstring: fig-gate-by-precision (paper version 08): a copy of draft 07's fig4_gate_precision_v1.py
(~/Documents/qmoe-results/2026-10-24_paper-draft-07_FIGURES_FINAL/), same data, same layout. Changed for version 08:
panel (d)'s title reads "TAU Urban Acoustic Scenes 2020 Mobile" (the original paper's dataset name; draft 07 had
"TAU 2020 Mobile"), the output file names, the JSON note and the PNG dpi (200 to 220). Nothing else.

Original docstring: Figure 4 (new), gate quality by precision.
Source (read only): 2026-10-24_1557_gate-by-precision_code-6cbc780/paired_vs_fp32.csv (+ systems.csv for the system
codes), metric auroc_margin, condition ALL (clean audio and the noise cells), paired difference to the FP32 reference
with the 95% t-interval over folds. Every plotted cell is asserted equal to the string in README.md's compact cross
table (the one the draft's Table gateprec was made from).
Names are the paper's (07-quantized-gate name table): AST-Q8pc is the 1024-frame Q8 row (per-channel 8-bit weights),
AST-Q4ch is per-channel 4-bit weights (the result file calls it Q4pc), AST-Q4 is group-wise, MN10-Q8 / MN20-Q8 are the
simulated Q8 networks, -INT8 real integer kernels with a retrained head, BitNet head against the FP32 head.
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from figstyle import BLUE, INK, INK2, LAV, OCHRE, RC, WIDTH, quiet_axes
from fig_v2_common import MID, place_label, report
from fig_v3_common import final_asserts

OUT = Path(__file__).resolve().parent
GP = Path.home() / "Documents/qmoe-results/2026-10-24_1557_gate-by-precision_code-6cbc780"
FS = 7
HEADER_X = 0.012   # figure x of the flush-left group headers
# matplotlib reads only the regular face of macOS Helvetica.ttc, so fontweight="bold" in Helvetica draws regular weight
# (that is why v1's headers did not look bold). Arial Bold is a separate TrueType file with Helvetica's metrics.
HEADER_FONT = ["Arial", "DejaVu Sans"]
TASKS = [("esc50", "ESC-50\n(50 classes)"), ("esc5", "ESC-50\n(5 groups)"), ("urban", "UrbanSound8K"), ("tau", "TAU Urban\nAcoustic Scenes\n2020 Mobile")]
README_COL = {"esc50": 1, "esc5": 2, "urban": 3, "tau": 4}
# (header or row): ("H", text) | ("R", label, panel, model, backbone, head, colour, README row label)
ROWS = [("H", "AST, 1024 frames, FP32 head"),
        ("R", "AST-Q8pc", "backbone", "ast1024", "Q8", "esc", BLUE, "AST 1024 frames, FP32 head: Q8"),
        ("R", "AST-Q6", "backbone", "ast1024", "Q6", "esc", BLUE, "AST 1024 frames, FP32 head: Q6"),
        ("R", "AST-Q5", "backbone", "ast1024", "Q5", "esc", BLUE, "AST 1024 frames, FP32 head: Q5"),
        ("R", "AST-Q4 (group-wise)", "backbone", "ast1024", "Q4", "esc", BLUE, "AST 1024 frames, FP32 head: Q4"),
        ("R", "AST-Q4ch (per-channel)", "backbone", "ast1024", "Q4pc", "esc", BLUE, "AST 1024 frames, FP32 head: Q4pc"),
        ("R", "AST-Q3", "backbone", "ast1024", "Q3", "esc", BLUE, "AST 1024 frames, FP32 head: Q3"),
        ("H", "Small CNNs, against their FP32"),
        ("R", "MN10-Q8", "backbone", "mn10", "Q8sim", "esc", OCHRE, "mn10 Q8sim"),
        ("R", "MN20-Q8", "backbone", "mn20", "Q8sim", "esc", OCHRE, "mn20 Q8sim"),
        ("R", "MN10-INT8 (retrained head)", "backbone", "mn10", "INT8", None, OCHRE, "mn10 INT8"),
        ("R", "MN20-INT8 (retrained head)", "backbone", "mn20", "INT8", None, OCHRE, "mn20 INT8"),
        ("H", "Head, against the FP32 head"),
        ("R", "BitNet head, MN10-FP32", "head", "mn10", "FP32", "bitnet", LAV, "BitNet head vs FP32 head, mn10 FP32"),
        ("R", "BitNet head, MN20-FP32", "head", "mn20", "FP32", "bitnet", LAV, "BitNet head vs FP32 head, mn20 FP32")]


def fmt(r):
    return f"{r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]{'*' if r['decided'] else ''}"


def readme_table():
    txt = (GP / "README.md").read_text()
    sec = txt.split("### Margin-gate AUROC for 'model is wrong'")[1].split("### Macro F1")[0]
    tab = {}
    for line in sec.splitlines():
        if line.startswith("|") and not line.startswith("|---") and not line.startswith("| precision"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            tab[cells[0]] = cells
    return tab


def load():
    st = pd.read_csv(GP / "paired_vs_fp32.csv")
    sy = pd.read_csv(GP / "systems.csv")
    st = st.merge(sy[["sid", "task", "model", "backbone", "head"]], on="sid")
    st = st[(st.cond == "ALL") & (st.metric == "auroc_margin")]
    tab = readme_table()
    out = []
    for row in ROWS:
        if row[0] == "H":
            out.append({"header": row[1]})
            continue
        _, label, panel, model, bb, head, col, rlabel = row
        cells = {}
        for task, _t in TASKS:
            x = st[(st.panel == panel) & (st.task == task) & (st.model == model) & (st.backbone == bb)]
            if head:
                x = x[x["head"] == head]
            if x.empty:
                cells[task] = None
                assert tab[rlabel][README_COL[task]] == "n/a", (label, task, tab[rlabel][README_COL[task]])
                continue
            assert len(x) == 1, (label, task, x[["sid", "head"]].values.tolist())
            r = x.iloc[0]
            assert r.lo <= r["diff"] <= r.hi and bool(r.decided) == (r.lo > 0 or r.hi < 0)
            assert fmt(r) == tab[rlabel][README_COL[task]], (label, task, fmt(r), tab[rlabel][README_COL[task]])
            cells[task] = {"diff": float(r["diff"]), "lo": float(r.lo), "hi": float(r.hi), "excludes_zero": bool(r.decided),
                           "n_folds": int(r.n), "system_id": r.sid}
        out.append({"label": label, "colour": col, "cells": cells})
    return out


def main():
    data = load()
    n = len(data)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 4.0), facecolor="white", sharey=True)
        fig.subplots_adjust(left=0.265, right=0.985, bottom=0.2, top=0.86, wspace=0.14)
        for ax, (task, title) in zip(axes, TASKS):
            ax.axvline(0, color=INK2, lw=0.8, zorder=2, label="zero")
            for i, d in enumerate(data):
                y = n - 1 - i
                if "header" in d:
                    continue
                c = d["cells"][task]
                if c is None and task == "tau" and d["label"].startswith("AST"):
                    continue
                if c is None:
                    t = ax.text(0.02, y, "n/a", transform=ax.get_yaxis_transform(), fontsize=FS, color=INK2, ha="left", va="center")
                    t._fc_owner = (-1e9, -1e9)
                    continue
                ax.plot([c["lo"], c["hi"]], [y, y], color=d["colour"], lw=1.4, solid_capstyle="butt", zorder=3, label=f"{d['label']} interval")
                ax.scatter([c["diff"]], [y], s=22, facecolor=d["colour"] if c["excludes_zero"] else "white", edgecolor=d["colour"],
                           lw=1.0 if not c["excludes_zero"] else 0.5, zorder=4, label=f"{d['label']} difference")
            for i, d in enumerate(data):  # thin rule above each group header, inside the panel (seen by the overlap check)
                if "header" in d:
                    ax.plot([0, 1], [n - 1 - i + 0.5] * 2, transform=ax.get_yaxis_transform(), color=MID, lw=0.5, zorder=1,
                            solid_capstyle="butt", label=f"rule above '{d['header']}'")
            ax.set_ylim(-0.6, n - 0.4)
            ax.set_xlim(-0.115, 0.035)
            ax.set_xticks([-0.10, -0.05, 0.0])
            ax.set_xticklabels(["-0.10", "-0.05", "0"])
            ax.set_yticks([])
            quiet_axes(ax, grid="x")
            ax.tick_params(labelsize=FS)
            ax.spines["left"].set_visible(False)
            ax.set_title(f"({'abcd'[[t for t, _ in TASKS].index(task)]}) {title}", loc="left", fontsize=8, color=INK, pad=3, linespacing=1.15)
        rule_ys = []
        for i, d in enumerate(data):
            y = n - 1 - i
            if "header" in d:
                t = axes[0].text(HEADER_X, y, d["header"], transform=blended_transform_factory(fig.transFigure, axes[0].transData),
                                 fontsize=FS, color=INK, ha="left", va="center", fontweight="bold",
                                 fontfamily=HEADER_FONT)
                # the same rule over the label column, from the header's left edge to panel (a)
                fig.add_artist(Line2D([HEADER_X, axes[0].get_position().x0], [y + 0.5] * 2, color=MID, lw=0.5,
                                      solid_capstyle="butt",
                                      transform=blended_transform_factory(fig.transFigure, axes[0].transData)))
                rule_ys.append(y + 0.5)
            else:
                t = axes[0].text(-0.03, y, d["label"], transform=axes[0].get_yaxis_transform(), fontsize=FS, color=INK, ha="right", va="center")
            t._fc_owner = (-1e9, -1e9)
        t = axes[3].text(-0.108, n - 1 - 3.5, "AST at\n1024 frames\nnot run", fontsize=FS,
                         color=INK2, ha="left", va="center")
        t._fc_owner = (-1e9, -1e9)
        fig.text(0.625, 0.125, "Margin AUROC for 'the model is wrong', difference to FP32 (95% interval over folds)",
                 fontsize=7.5, color=INK, ha="center", va="top")
        h = [Line2D([], [], marker="o", mfc=INK2, mec=INK2, ls="", ms=4.4, label="interval excludes zero"),
             Line2D([], [], marker="o", mfc="white", mec=INK2, mew=1.0, ls="", ms=4.4, label="interval spans zero"),
             Line2D([], [], color=BLUE, lw=1.4, label="transformer"), Line2D([], [], color=OCHRE, lw=1.4, label="small CNN"),
             Line2D([], [], color=LAV, lw=1.4, label="BitNet head")]
        fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.625, 0.075), ncol=5, fontsize=FS, handlelength=1.6, handletextpad=0.4,
                   labelcolor=INK, borderaxespad=0, columnspacing=1.0)
        fig.canvas.draw()
        check = report(fig, "fig-gate-by-precision-v2")
        # the overlap check sees only lines inside axes; test the label-column part of each rule against every text box
        from matplotlib.text import Text
        r = fig.canvas.get_renderer()
        for yr in rule_ys:
            ypx = axes[0].transData.transform((0, yr))[1]
            for t in fig.findobj(Text):
                if t.get_visible() and t.get_text().strip():
                    e = t.get_window_extent(r)
                    assert not (e.y0 - 1 <= ypx <= e.y1 + 1 and e.x0 < axes[0].bbox.x0), ("rule crosses text", t.get_text())
        check += f"\nlabel-column rules at rows {rule_ys}: clear of every text box (1 px pad)"
        extra = final_asserts(fig, {"blue": BLUE, "ochre": OCHRE, "lavender": LAV, "ink2": INK2, "rule grey": MID})
        print(extra)
        fig.savefig(OUT / "fig-gate-by-precision-v2.pdf", facecolor="white")
        fig.savefig(OUT / "fig-gate-by-precision-v2.png", dpi=220, facecolor="white")
        plt.close(fig)
    (OUT / "fig-gate-by-precision-v2.json").write_text(json.dumps(
        {"note": "fig-gate-by-precision-v2 (v1 with the three group headers set bold, flush left, with a thin rule above each; "
                 "v1 is a copy of draft 07 Figure 4 v1, TAU title renamed): margin-gate AUROC difference to FP32, condition ALL, 95% interval over folds; every cell asserted equal "
                 "to the README compact cross table",
         "source": str(GP / "paired_vs_fp32.csv"), "overlap_check": check, "asserts": extra, "rows": data}, indent=1))
    for d in data:
        if "label" in d:
            print(d["label"], {k: (None if v is None else round(v["diff"], 3)) for k, v in d["cells"].items()})


if __name__ == "__main__":
    main()
