#!/usr/bin/env python3
"""fig-humility: the humility curve of the two-expert QMoE. Escalation rate of MN10-Q8 (the cheap expert, FP32 head)
at one threshold fixed on clean validation clips, against noise level, for background, pink and white noise.

Starts from draft 07's fig2_humility_v4 (~/Documents/qmoe-results/2026-10-24_paper-draft-07_FIGURES_FINAL/): same noise
colours, markers, line widths, the +-0.08 category offsets, the dotted clean-rate line and the shaded field-mix band.
Draft 07's panels (a) to (c) drew the stage 3c cheap tiers (MN20-Q8 with a Q4 head, MN20-Q8 FP32 head, MN10-Q8 at 50%),
none of which is the version 08 design, and its panel (d) was unseen classes with MN20-Q8. Here every panel is MN10-Q8
with its FP32 head, the cheap expert of the version 08 design:

(a) ESC-50 (50 classes) and (b) UrbanSound8K: T5, threshold at the 30th percentile of the clean inner-validation margin
    (a 30% budget), clean then 20, 10, 5, 0 dB. Source: 2026-10-25_1406_moe-t3t5_code-81a3d6a/tables/t5_rate_by_cell_q30.csv
    (task esc50 / urban, system cheap), asserted against the rates printed in that run's README. No field-mix row exists
    in that table, so (a) and (b) have no field-mix point.
(c) FSDKaggle2018 and (d) FSD50K: T21, the pre-registered cascade, budget 20% and 30% picked on clean inner validation,
    clean then 10 and 0 dB, and the field mix. Source: 2026-10-26_1408_moe-t21_code-731570e/tables/{d1,d2}/
    primary_by_condition.csv (column rate), asserted against humility.csv and against the audit folders.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figstyle import BLUE, INK, INK2, LAV, OCHRE, RC, WIDTH, quiet_axes
import fig_v2_common as c2
from fig_v2_common import MID
from fig_v3_common import final_asserts

HERE = Path(__file__).resolve().parent
STEM = "fig-humility"
R = Path.home() / "Documents/qmoe-results"
T5 = R / "2026-10-25_1406_moe-t3t5_code-81a3d6a"
T21 = R / "2026-10-26_1408_moe-t21_code-731570e"
AUD = {"d1": R / "2026-10-26_1408_moe-t21_code-731570e-AUDIT/audit_by_condition.csv",
       "d2": R / "2026-10-26_1408_moe-t21_code-731570e-AUDIT-D2/audit_by_condition.csv"}

FS = 7
BAND = "#f1f0ea"
XPOS = {"clean": 0.0, 20: 1.0, 10: 2.0, 5: 3.0, 0: 4.0}
XFIELD, BAND_X0, XMAX = 5.85, 4.75, 6.75
NOISE = [("bg", "background noise", BLUE), ("pink", "pink noise", OCHRE), ("white", "white noise", LAV)]
JIT = {"bg": -0.08, "pink": 0.0, "white": 0.08}
STYLE = {"bg": dict(marker="o", s=17, lw=1.6), "pink": dict(marker="s", s=15, lw=1.2), "white": dict(marker="^", s=21, lw=0.9)}


def load_t5(task, title):
    f = T5 / "tables/t5_rate_by_cell_q30.csv"
    t = pd.read_csv(f)
    t = t[(t.task == task) & (t.system == "cheap")].set_index("cell")["rate"]
    snrs = [20, 10, 5, 0]
    rows = {"clean": float(t["clean"])}
    for k, _, _ in NOISE:
        for s in snrs:
            rows[f"{k}__{s}db"] = float(t[f"{k}__{s}db"])
    # cross-check: the rates printed in the T5 README, section (c)
    txt = (T5 / "README.md").read_text()
    sec = txt.split("**(c) Abstain rate")[1].split("Mixture:")[0]
    label = {"esc50": "ESC-50", "urban": "UrbanSound8K"}[task]
    part = sec.split(label + " bg")[1]
    nums = [float(x) for x in re.findall(r"\d\.\d{3}", part.split("UrbanSound8K bg")[0] if task == "esc50" else part)][:13]
    mine = [rows["clean"]] + [rows[f"bg__{s}db"] for s in snrs] + [rows[f"pink__{s}db"] for s in snrs] + [rows[f"white__{s}db"] for s in snrs]
    assert len(nums) == 13 and all(abs(a - b) < 6e-4 for a, b in zip(nums, mine)), (task, nums, mine)
    return {"title": title, "tier": "MN10-Q8, FP32 head, 30% budget", "snrs": snrs, "rows": rows, "field": None,
            "source": str(f) + f" (task == {task}, system == cheap)",
            "threshold": "30th percentile of the margin on clean inner validation, per seed and fold",
            "cross_check": str(T5 / "README.md") + " section (c), 13 rates equal to 3 decimals"}


def load_t21(d, title, budget):
    f = T21 / f"tables/{d}/primary_by_condition.csv"
    t = pd.read_csv(f).set_index("cond")
    assert (t.budget == budget).all()
    snrs = [10, 0]
    rows = {"clean": float(t.loc["clean", "rate"])}
    for k, _, _ in NOISE:
        for s in snrs:
            rows[f"{k}__{s}db"] = float(t.loc[f"{k}__{s}db", "rate"])
    field = float(t.loc["field", "rate"])
    h = pd.read_csv(T21 / f"tables/{d}/humility.csv")
    for _, r in h.iterrows():
        a, b = [x.strip() for x in r.step.split("->")]
        rr = {**rows, "field": field}
        assert abs(rr[a] * 100 - r.rate_from) < 1e-6 and abs(rr[b] * 100 - r.rate_to) < 1e-6, (d, r.step)
        assert r.rises and r.folds_rising == 5
    au = pd.read_csv(AUD[d]).set_index("cond")["rate"]
    for c, v in {**rows, "field": field}.items():
        assert abs(au[c] - v) < 1e-3, (d, c, au[c], v)
    return {"title": title, "tier": f"MN10-Q8, FP32 head, {budget[1:]}% budget", "snrs": snrs, "rows": rows, "field": field,
            "source": str(f) + " (column rate)", "threshold": "clean inner-validation budget, chosen before the test data",
            "cross_check": [str(T21 / f"tables/{d}/humility.csv") + " (every step rises in 5 of 5 folds)", str(AUD[d])]}


def panel(ax, d, letter):
    rows = d["rows"]
    clean = rows["clean"]
    has_field = d["field"] is not None
    xmax = XMAX
    ax.axvspan(BAND_X0, XMAX, color=BAND, zorder=0, lw=0)
    ax.axvline(BAND_X0, color=MID, lw=0.6, zorder=1, label="_sep")
    ax.text((BAND_X0 + XMAX) / 2, 0.985, "field mix", fontsize=FS, color=INK, ha="center", va="top",
            transform=ax.get_xaxis_transform())
    ax.axhline(clean, color=MID, lw=0.8, ls=(0, (1.5, 2)), zorder=1, label="clean rate")
    handles = []
    for key, name, col in NOISE:
        st = STYLE[key]
        xs = [XPOS["clean"]] + [XPOS[s] + JIT[key] for s in d["snrs"]]
        ys = [clean] + [rows[f"{key}__{s}db"] for s in d["snrs"]]
        ax.plot(xs, ys, color=col, lw=st["lw"], zorder=3, solid_capstyle="round", label=f"{name} line")
        ax.scatter(xs[1:], ys[1:], s=st["s"], marker=st["marker"], color=col, edgecolor="white", lw=0.5, zorder=4,
                   label=f"{name} points")
        handles.append(Line2D([], [], color=col, lw=st["lw"], marker=st["marker"], ms=np.sqrt(st["s"]) * 0.9, mec="white",
                              mew=0.4, label=name))
    ax.scatter([0], [clean], s=22, color=INK, edgecolor="white", lw=0.5, zorder=5, label="clean point")
    ax.text(4.6, clean - 0.025, f"clean rate {clean:.2f}", ha="right", va="top", fontsize=FS,
            color=INK2)
    if has_field:
        fr = d["field"]
        ax.scatter([XFIELD], [fr], s=38, marker="D", color=INK, edgecolor="white", lw=0.5, zorder=5, label="field mix")
        ax.annotate(f"{fr:.2f}", xy=(XFIELD, fr), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom",
                    fontsize=FS, color=INK2)
    else:
        ax.text((BAND_X0 + XMAX) / 2, 0.5, "not scored\nin this run", fontsize=FS, color=INK2, ha="center",
                va="center", linespacing=1.15)
    ticks = ["clean"] + d["snrs"]
    ax.set_xticks([XPOS[s] for s in ticks])
    ax.set_xticklabels([str(s) for s in ticks])
    ax.set_xlim(-0.55, xmax)
    ax.set_ylim(0, 1.2)        # headroom above 1.0 holds the in-panel legend
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    quiet_axes(ax, grid="y")
    ax.tick_params(labelsize=FS)
    ax.set_title(f"({letter}) {d['title']}", loc="left", fontsize=8, color=INK, pad=11)
    ax.text(0, 1.015, d["tier"], transform=ax.transAxes, fontsize=FS, color=INK2, ha="left", va="bottom")
    ax.set_xlabel("SNR (dB), louder noise to the right", fontsize=FS, color=INK, labelpad=3)
    ax.legend(handles=handles, loc="upper left", fontsize=FS, handlelength=2.0, handletextpad=0.5, labelcolor=INK,
              borderaxespad=0.15, labelspacing=0.3, frameon=False)


def overlap_check(fig):
    """fig2_humility_v4.overlap_check: the shared check, with each in-panel legend tested against the other panels,
    its own panel's lines and the panel edge instead of its own panel box."""
    bad, nt, nm, nl = c2.collisions(fig)
    bad = [b for b in bad if b[0] != "legend-panel"]
    r = fig.canvas.get_renderer()
    legs = [(ax, ax.get_legend()) for ax in fig.axes if ax.get_legend() is not None]
    for ax, leg in legs:
        lb = leg.get_window_extent(r)
        for ax2 in fig.axes:
            if ax2 is not ax and lb.overlaps(ax2.bbox):
                bad.append(("legend-panel", "legend", ax2.get_title(loc="left")))
        for path, pax, lab in c2._line_paths(fig):
            if pax is ax and c2._crosses(c2._pad(lb, 1.0), pax, path):
                bad.append(("legend-line", "legend", lab))
        if not ax.bbox.padded(2).contains(lb.x0, lb.y0) or not ax.bbox.padded(2).contains(lb.x1, lb.y1):
            bad.append(("legend-outside", "legend", ax.get_title(loc="left")))
    head = f"OVERLAP CHECK {STEM}: {len(bad)} collisions among {nt} text boxes, {nm} markers, {nl} lines, {len(legs)} legends"
    text = "\n".join([head] + [f"  {k}: {a!r} vs {b!r}" for k, a, b in bad])
    print(text)
    return text


def main():
    plt.rcParams.update(RC)
    data = [load_t5("esc50", "ESC-50 (50 classes)"), load_t5("urban", "UrbanSound8K"),
            load_t21("d1", "FSDKaggle2018", "b20"), load_t21("d2", "FSD50K", "b30")]
    fig = plt.figure(figsize=(WIDTH, 4.45), facecolor="white")
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.985, top=0.925, bottom=0.085, wspace=0.22, hspace=0.52,
                          )
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    # top row: T5 (ESC-50, UrbanSound8K); bottom row: T21 (FSDKaggle2018, FSD50K)
    order = [(axes[0], data[0], "a"), (axes[1], data[1], "b"), (axes[2], data[2], "c"), (axes[3], data[3], "d")]
    for ax, d, letter in order:
        panel(ax, d, letter)
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("Escalation rate", fontsize=7.5, color=INK)
    check = overlap_check(fig)
    asserts = final_asserts(fig, {"ink": INK, "ink2": INK2, "reference grey": MID, "background blue": BLUE,
                                  "pink ochre": OCHRE, "white lavender": LAV})
    print(asserts)
    assert "0 collisions" in check
    fig.savefig(HERE / f"{STEM}.pdf", facecolor="white")
    fig.savefig(HERE / f"{STEM}.png", dpi=220, facecolor="white")
    plt.close(fig)
    (HERE / f"{STEM}.json").write_text(json.dumps({
        "figure": "fig-humility, escalation rate of MN10-Q8 at a fixed clean threshold against noise level",
        "panels": {"a": data[0], "b": data[1], "c": data[2], "d": data[3]},
        "x_offsets_category_widths": JIT,
        "not_drawn": ["field mix for ESC-50 and UrbanSound8K: not in t5_rate_by_cell_q30.csv",
                      "20 and 5 dB for FSDKaggle2018 and FSD50K: T21 scored 10 and 0 dB only",
                      "draft 07 panel (d), unseen classes and the other dataset: run with MN20-Q8 on ESC-50, not the version 08 cheap expert",
                      "TAU Urban Acoustic Scenes 2020 Mobile: T5 has device shift, not noise severity (rates in the same table)"],
        "overlap_check": check, "asserts": asserts}, indent=1))
    print("saved", HERE / f"{STEM}.pdf")


if __name__ == "__main__":
    main()
