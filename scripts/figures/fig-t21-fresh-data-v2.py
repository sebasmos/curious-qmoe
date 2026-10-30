#!/usr/bin/env python3
"""fig-t21-fresh-data-v2: T21, the pre-registered cascade on data never used before, with T25's widened hull in (a).

Changed from fig-t21-fresh-data.py (v1), panel (a) only: T25's MN10-Q8 + MN20-Q8 average as a new marker, the widened
equal-cost hull (upper envelope of T25's 17 statics on the field mix) as a dashed grey line, the gain over it
("+0.27 over widened hull"), the old gain renamed "+2.83 over two-expert hull", and the legend updated. Panels (b) and
(c) are unchanged (FSD50K has no widened hull). T25 data: ~/Documents/qmoe-results/2026-10-28_2247_moe-t25_code-cc3b048/
tables/d1/{verdict.json, t25_statics_by_condition.csv, t25_by_condition.csv}.

v1 docstring follows.

(a) FSDKaggle2018, field mix: macro F1 against priced cost for MN10-Q8 alone, AST-Q4 alone, always-average, the
    equal-cost hull (time-sharing of MN10-Q8 and always-average, the hull pair the T21 tables report) and the
    cascade at its chosen budget (20%), with its gain over the hull.
(b) FSD50K single-label subset, the same plane, descriptive only: its precondition (AST-Q4 at least 2.0 points
    above MN10-Q8 on the field mix) failed, so H1 was not tested there.
(c) The budget sweep that exists in the T21 tables: gain over the hull on clean inner validation at each budget,
    the bed the budget was chosen on (20% on FSDKaggle2018, 30% on FSD50K). No test-set budget sweep exists for
    FSDKaggle2018 in the T21 tables (primary_by_seed_fold.csv holds the chosen budget only), so none is drawn.

Data: only ~/Documents/qmoe-results/2026-10-26_1408_moe-t21_code-731570e/tables/{d1,d2}/primary_by_condition.csv
and primary_budget_selection.csv; every F1 is cross-checked against the -AUDIT (D1) and -AUDIT-D2 folders.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

from figstyle import BLUE, INK, INK2, LAV_DARK, OCHRE, RC, WIDTH, quiet_axes
from fig_v2_common import MID, report
from fig_v3_common import final_asserts

HERE = Path(__file__).resolve().parent
STEM = "fig-t21-fresh-data-v2"
R = Path.home() / "Documents/qmoe-results"
T21 = R / "2026-10-26_1408_moe-t21_code-731570e"
AUD1 = R / "2026-10-26_1408_moe-t21_code-731570e-AUDIT"
AUD2 = R / "2026-10-26_1408_moe-t21_code-731570e-AUDIT-D2"
T25 = R / "2026-10-28_2247_moe-t25_code-cc3b048"
T = 1e12
NAMES = {"d1": "FSDKaggle2018", "d2": "FSD50K"}


def load_plane(d):
    f = T21 / f"tables/{d}/primary_by_condition.csv"
    t = pd.read_csv(f)
    r = t[t.cond == "field"]
    assert len(r) == 1
    r = r.iloc[0]
    hull_a, hull_b, lam = r.hull.split("|")
    assert (hull_a, hull_b) == ("mn", "avg"), r.hull
    out = {
        "source": str(f), "row": "cond == field", "budget": r.budget,
        "mn": {"f1": r.mn_f1, "cost": r.mn_cost},
        "ast": {"f1": r.ast_f1, "cost": r.ast_cost},
        # always-average is charged both tiers on every clip (audit README, section 3)
        "avg": {"f1": r.avg_f1, "cost": r.mn_cost + r.ast_cost},
        "cascade": {"f1": r.cascade_f1, "cost": r.cascade_cost, "rate": r.rate},
        "hull": {"pair": r.hull, "lambda": float(lam), "f1_at_cascade_cost": r.hull_f1},
        "d_hull": r.d_hull, "nb_ci95": [r.nb_lo_hull, r.nb_hi_hull], "p_nb": r.p_nb_hull,
        "folds_pos": int(r.folds_pos_hull), "n_folds": int(r.n_folds), "n_seeds": int(r.n_seeds),
        "random_gate_f1_not_plotted": r.random_gate_f1, "precondition_ast_minus_mn": r.d_ast_minus_mn,
    }
    # the per seed-fold table gives always-average's cost directly; check the sum above against it
    sf = pd.read_csv(T21 / f"tables/{d}/primary_by_seed_fold.csv")
    sf = sf[sf.cond == "field"].groupby("system")[["f1", "cost"]].mean()
    assert abs(sf.loc["avg", "cost"] / out["avg"]["cost"] - 1) < 1e-9, (sf.loc["avg", "cost"], out["avg"]["cost"])
    for k in ("mn", "ast", "avg", "cascade"):
        assert abs(sf.loc[k, "f1"] - out[k]["f1"]) < 1e-9
    # hull: the linear time-sharing segment between MN10-Q8 and always-average; AST-Q4 must sit below it
    lin = lambda c: out["mn"]["f1"] + (c - out["mn"]["cost"]) / (out["avg"]["cost"] - out["mn"]["cost"]) * (out["avg"]["f1"] - out["mn"]["f1"])
    assert out["ast"]["f1"] < lin(out["ast"]["cost"]), "AST-Q4 above the MN10-Q8 to always-average segment"
    lam_lin = (out["cascade"]["cost"] - out["mn"]["cost"]) / (out["avg"]["cost"] - out["mn"]["cost"])
    assert abs(lam_lin - out["hull"]["lambda"]) < 1e-4, (lam_lin, out["hull"]["lambda"])
    out["hull"]["f1_linear_at_cascade_cost"] = lin(out["cascade"]["cost"])
    out["hull"]["note"] = ("The line is the linear time-sharing segment between MN10-Q8 and always-average. The table's hull_f1 "
                           "(the expected macro F1 of the random mixture) is used for the stem end and the gain; it differs from "
                           f"the line at the cascade's cost by {abs(lin(out['cascade']['cost']) - r.hull_f1):.3f} points.")
    assert abs(out["cascade"]["f1"] - out["hull"]["f1_at_cascade_cost"] - out["d_hull"]) < 1e-9
    return out, lin


def audit_check(d, p):
    """F1s and the escalation rate must match the independent audit to 1e-3; costs may not (auditor's own MN10 count)."""
    a = pd.read_csv((AUD1 / "audit_by_condition.csv") if d == "d1" else (AUD2 / "audit_by_condition.csv"))
    a = a[a.cond == "field"].iloc[0]
    rows = {}
    for k, col in (("mn", "mn"), ("ast", "ast"), ("avg", "avg"), ("cascade", "cascade")):
        rows[k] = {"t21": p[k]["f1"], "audit": float(a[col if d == "d1" else f"{col}_f1"])}
        assert abs(rows[k]["t21"] - rows[k]["audit"]) < 1e-3, (d, k, rows[k])
    assert abs(p["cascade"]["rate"] - a["rate"]) < 1e-3
    rows["rate"] = {"t21": p["cascade"]["rate"], "audit": float(a["rate"])}
    rows["cascade_cost"] = {"t21": p["cascade"]["cost"], "audit": float(a["cascade_cost"]),
                            "note": "differs because the auditor counts MN10 cost about 1.31x higher; the hull is MN10-Q8 to "
                                    "always-average, so the gain does not depend on MN10's cost (D1 audit, section 3)"}
    gain_col = "d_linear" if d == "d1" else "d_hull_lin"
    rows["d_hull"] = {"t21": p["d_hull"], "audit_linear_hull": float(a[gain_col])}
    assert abs(rows["d_hull"]["t21"] - rows["d_hull"]["audit_linear_hull"]) < 0.05
    return {"source": str((AUD1 if d == "d1" else AUD2) / "audit_by_condition.csv"), "rows": rows}


def load_budget(d):
    f = T21 / f"tables/{d}/primary_budget_selection.csv"
    t = pd.read_csv(f)
    t["pct"] = t.budget.str[1:].astype(int)
    t = t.sort_values("pct")
    chosen = t.loc[t.gain_iv.idxmax(), "budget"]
    a = pd.read_csv((AUD1 / "audit_budget_selection.csv") if d == "d1" else (AUD2 / "audit_budget_selection.csv"))
    acol = "gain_iv" if d == "d1" else "gain_expected"
    a_best = a.loc[a[acol].idxmax(), "budget"]
    a_best = f"b{int(round(float(a_best) * 100)):02d}" if d == "d1" else a_best
    assert a_best == chosen, (d, a_best, chosen)
    return {"source": str(f), "bed": "clean inner validation", "budget_pct": t.pct.tolist(), "gain_iv": t.gain_iv.tolist(),
            "chosen": chosen, "audit_source": str((AUD1 if d == "d1" else AUD2) / "audit_budget_selection.csv"),
            "audit_gain": a[acol].tolist(), "audit_chosen": a_best}


def upper_envelope(pts):
    """Upper concave envelope of (cost, f1, name) points sorted by cost (monotone chain); time-sharing is linear in cost."""
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])  # noqa: E731
    up = []
    for q in sorted(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], q) >= 0:
            up.pop()
        up.append(q)
    return up


def load_wide(p1):
    """T25, FSDKaggle2018 field mix: the widened hull, the MN10+MN20 average and the gain over the widened hull."""
    fv, fs, fc = T25 / "tables/d1/verdict.json", T25 / "tables/d1/t25_statics_by_condition.csv", T25 / "tables/d1/t25_by_condition.csv"
    v = json.loads(fv.read_text())
    st = pd.read_csv(fs)
    st = st[st.cond == "field"].set_index("system")
    bc = pd.read_csv(fc)
    bc = bc[bc.cond == "field"].iloc[0]
    h1, bp = v["H1_wide"], v["best_pair"]["field"]
    # T25 rebuilt T21's cascade and narrow hull exactly
    assert abs(h1["cascade_f1"] - p1["cascade"]["f1"]) < 1e-9 and abs(h1["cascade_cost"] - p1["cascade"]["cost"]) < 1e-3
    assert abs(v["narrow_reproduced"]["field"] - p1["d_hull"]) < 1e-9
    for k in ("mn", "ast", "avg"):
        assert abs(st.loc[{"mn": "mn10"}.get(k, k), "f1"] - p1[k]["f1"]) < 1e-9
        assert abs(st.loc[{"mn": "mn10"}.get(k, k), "cost"] / p1[k]["cost"] - 1) < 1e-9
    # the pair point: statics row equals verdict.json best_pair.field
    assert bp["pair_chosen_on_iv"] == "mn10+mn20"
    assert abs(st.loc["mn10+mn20", "f1"] - bp["pair_f1"]) < 1e-9 and abs(st.loc["mn10+mn20", "cost"] / bp["pair_cost"] - 1) < 1e-9
    # the widened hull: the 17 statics (every system in the table except the cascade)
    statics = st.drop(index="cascade")
    assert len(statics) == 17, len(statics)
    env = upper_envelope([(r.cost, r.f1, s) for s, r in statics.iterrows()])
    cc = p1["cascade"]["cost"]
    seg = [(a, b) for a, b in zip(env, env[1:]) if a[0] <= cc <= b[0]]
    assert len(seg) == 1
    a, b = seg[0]
    lam = (cc - a[0]) / (b[0] - a[0])
    lo, hi, lam_t = h1["hull"].split("|")
    assert (a[2], b[2]) == (lo, hi) == (bc.hull_wide_lo, bc.hull_wide_hi), (a[2], b[2], h1["hull"])
    assert abs(lam - bc.hull_wide_lam) < 1e-4 and abs(lam - float(lam_t)) < 1e-4
    lin_at_cc = a[1] + lam * (b[1] - a[1])
    assert abs(h1["cascade_f1"] - h1["hull_f1"] - h1["d"]) < 1e-9
    assert abs(bc.d_wide - h1["d"]) < 1e-12 and abs(bc.hull_wide_f1 - h1["hull_f1"]) < 1e-9
    return {
        "sources": {"verdict": str(fv), "statics": str(fs), "by_condition": str(fc)},
        "row": "cond == field (FSDKaggle2018)",
        "mn10_mn20_average": {"f1": float(st.loc["mn10+mn20", "f1"]), "cost": float(st.loc["mn10+mn20", "cost"]),
                              "source": f"{fs} (system mn10+mn20), equal to {fv} best_pair.field pair_f1 / pair_cost",
                              "d_cascade_minus_pair": bp["d"], "nb_ci95": bp["nb_ci95"], "p_nb": bp["p_nb"],
                              "folds_positive": bp["folds_positive"]},
        "widened_statics_field": {s: {"f1": float(r.f1), "cost": float(r.cost)} for s, r in statics.iterrows()},
        "envelope_vertices": [{"system": s, "cost": float(c), "f1": float(f)} for c, f, s in env],
        "drawn_from": "mn10",
        "not_drawn_note": "envelope vertices mn04, dymn04 and mn05 (0.015 to 0.017 T, F1 68.2 to 69.5) lie left of the panel's "
                          "x range (from 0.018 T) and below its y range; the dashed line starts at MN10-Q8, which is on the envelope",
        "segment_at_cascade_cost": {"lo": lo, "hi": hi, "lambda_table": float(bc.hull_wide_lam), "lambda_recomputed": lam,
                                    "hull_f1_table_expected": h1["hull_f1"], "hull_f1_linear": lin_at_cc,
                                    "note": "the line is the linear time-sharing envelope; the gain uses the table's hull_f1 "
                                            f"(expected macro F1 of the mixture), {abs(lin_at_cc - h1['hull_f1']):.3f} points below the line"},
        "d_wide": h1["d"], "nb_ci95": h1["nb_ci95"], "p_nb": h1["p_nb"], "boot_ci95": h1["boot_ci95"],
        "folds_positive": h1["folds_positive"], "n_folds": h1["n_folds"], "fold_d": h1["fold_d"],
        "cascade": {"f1": h1["cascade_f1"], "cost": h1["cascade_cost"], "source": f"{fv} H1_wide"},
    }, env


SYS = [  # key, legend name, marker, colour, filled
    ("mn", "MN10-Q8 alone", "o", OCHRE),
    ("ast", "AST-Q4 alone", "^", BLUE),
    ("avg", "always-average (both experts)", "s", INK2),
    ("cascade", "cascade, chosen budget", "D", LAV_DARK),
]


PAIR_NAME = "MN10-Q8 + MN20-Q8 average"
PAIR_STYLE = dict(marker="h", s=38, c=OCHRE, edgecolors="white", linewidths=0.8, zorder=4)
WIDE_TEXT_OFFSET = (-5, 4)


def plane(ax, p, lin, title, sub, gain_text, wide=None, env=None):
    c0, c1 = p["mn"]["cost"], p["avg"]["cost"]
    xs = np.geomspace(c0, c1, 300)
    ax.plot(xs / T, [lin(x) for x in xs], color=MID, lw=1.1, zorder=1, label="_hull")
    if env is not None:  # widened hull: piecewise linear in cost from MN10-Q8 on, drawn like the two-expert hull
        k0 = [s for _, _, s in env].index(wide["drawn_from"])
        for a, b in zip(env[k0:], env[k0 + 1:]):
            xw = np.geomspace(a[0], b[0], 200)
            ax.plot(xw / T, a[1] + (xw - a[0]) / (b[0] - a[0]) * (b[1] - a[1]), color=MID, lw=1.1, ls=(0, (3, 1.6)),
                    zorder=1, label="_widehull")
        q = wide["mn10_mn20_average"]
        ax.scatter([q["cost"] / T], [q["f1"]], **PAIR_STYLE, label=PAIR_NAME)
    cc, cf, hf = p["cascade"]["cost"], p["cascade"]["f1"], p["hull"]["f1_at_cascade_cost"]
    ax.plot([cc / T, cc / T], [hf, cf], color=LAV_DARK, lw=0.9, ls=(0, (2, 1.5)), zorder=2, label="_stem")
    for k, name, m, col in SYS:
        ax.scatter([p[k]["cost"] / T], [p[k]["f1"]], marker=m, s=34 if m != "D" else 30, c=col, edgecolors="white",
                   linewidths=0.8, zorder=4, label=name)
    ax.set_xscale("log")
    ax.set_xlim(0.018, 5.0)
    ax.xaxis.set_major_locator(FixedLocator([0.03, 0.1, 0.3, 1, 3]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["0.03", "0.1", "0.3", "1", "3"])
    quiet_axes(ax)
    ax.set_xlabel("Priced cost per clip (T bit-ops, log)", fontsize=7.5, color=INK)
    ax.set_title(title, loc="left", fontsize=8, color=INK, pad=13)
    ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=7, color=INK2, va="bottom", ha="left")
    ax.annotate(gain_text, xy=(cc / T, (cf + hf) / 2), xytext=(-5, 0), textcoords="offset points", ha="right",
                va="center", fontsize=7, color=INK, linespacing=1.15)
    if wide is not None:
        ax.annotate(f"+{wide['d_wide']:.2f} over\nwidened hull", xy=(cc / T, cf), xytext=WIDE_TEXT_OFFSET,
                    textcoords="offset points", ha="right", va="bottom", fontsize=7, color=INK, linespacing=1.15)


def main():
    plt.rcParams.update(RC)
    p1, lin1 = load_plane("d1")
    wide, env = load_wide(p1)
    p2, lin2 = load_plane("d2")
    aud = {"d1": audit_check("d1", p1), "d2": audit_check("d2", p2)}
    b1, b2 = load_budget("d1"), load_budget("d2")
    assert p1["budget"] == b1["chosen"] == "b20" and p2["budget"] == b2["chosen"] == "b30"

    fig = plt.figure(figsize=(WIDTH, 2.75), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.78], left=0.075, right=0.985, top=0.855, bottom=0.315,
                          wspace=0.42)
    ax1, ax2, ax3 = (fig.add_subplot(gs[0, i]) for i in range(3))
    plane(ax1, p1, lin1, "(a) FSDKaggle2018, field mix", "pre-registered test, budget 20%",
          f"+{p1['d_hull']:.2f} over\ntwo-expert hull", wide=wide, env=env)
    plane(ax2, p2, lin2, "(b) FSD50K, field mix", "descriptive: precondition failed",
          f"+{p2['d_hull']:.2f}\nover hull")
    ax1.set_ylabel("Macro F1 (%)", fontsize=7.5, color=INK)
    ax1.set_ylim(71.5, 79.0)
    ax2.set_ylim(44.5, 50.5)
    ax2.yaxis.set_major_locator(FixedLocator([45, 46, 47, 48, 49, 50]))

    # (c) budget choice on clean inner validation
    for b, name, m, ls in ((b1, "FSDKaggle2018", "o", "-"), (b2, "FSD50K", "^", (0, (3, 1.6)))):
        ax3.plot(b["budget_pct"], b["gain_iv"], color=INK2, lw=1.0, ls=ls, zorder=2, label="_" + name)
        ax3.scatter(b["budget_pct"], b["gain_iv"], marker=m, s=20, facecolors="white", edgecolors=INK2, linewidths=0.9,
                    zorder=3, label="_pts")
        k = b["budget_pct"].index(int(b["chosen"][1:]))
        ax3.scatter([b["budget_pct"][k]], [b["gain_iv"][k]], marker=m, s=30, c=LAV_DARK, edgecolors="white",
                    linewidths=0.8, zorder=4, label="_chosen")
    quiet_axes(ax3)
    ax3.set_xlim(0, 55)
    ax3.set_ylim(0, 3.2)
    ax3.xaxis.set_major_locator(FixedLocator([5, 10, 20, 30, 50]))
    ax3.set_xticklabels(["5", "10", "20", "30", "50"])
    ax3.set_xlabel("Escalation budget (%)", fontsize=7.5, color=INK)
    ax3.set_ylabel("Gain over hull (points)", fontsize=7.5, color=INK)
    ax3.set_title("(c) Budget choice", loc="left", fontsize=8, color=INK, pad=13)
    ax3.text(0, 1.015, "clean inner validation", transform=ax3.transAxes, fontsize=7, color=INK2, va="bottom")
    ax3.text(52, b1["gain_iv"][-1] - 0.22, "FSDKaggle2018", ha="right", va="top", fontsize=7, color=INK2)
    ax3.text(3, 2.92, "filled: chosen", ha="left", va="top", fontsize=7, color=INK2)
    ax3.text(4, b2["gain_iv"][0] - 0.22, "FSD50K", ha="left", va="top", fontsize=7, color=INK2)

    handles = [Line2D([], [], ls="none", marker=m, ms=5.5, mfc=col, mec="white", mew=0.6, label=name)
               for _, name, m, col in SYS]
    handles.insert(2, Line2D([], [], ls="none", marker=PAIR_STYLE["marker"], ms=6.0, mfc=OCHRE, mec="white", mew=0.6,
                             label=PAIR_NAME))
    handles.insert(4, Line2D([], [], color=MID, lw=1.1, label="two-expert hull (time-sharing)"))
    handles.insert(5, Line2D([], [], color=MID, lw=1.1, ls=(0, (3, 1.6)), label="(a) widened hull, 17 statics"))
    handles.append(Line2D([], [], ls="none", marker="o", ms=4.8, mfc="white", mec=INK2, mew=0.9,
                          label="(c) budget not chosen"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 0.0),
               handletextpad=0.4, columnspacing=1.4, borderaxespad=0.3)
    check = report(fig, STEM, strict=False)
    colors = {"ink": INK, "ink2": INK2, "hull/reference grey": MID, "MN10-Q8 ochre": OCHRE, "AST-Q4 blue": BLUE,
              "cascade dark lavender": LAV_DARK}
    asserts = final_asserts(fig, colors)
    print(asserts)
    assert "0 collisions" in check
    fig.savefig(HERE / f"{STEM}.pdf", facecolor="white")
    fig.savefig(HERE / f"{STEM}.png", dpi=220, facecolor="white")
    plt.close(fig)
    (HERE / f"{STEM}.json").write_text(json.dumps({
        "figure": "fig-t21-fresh-data-v2", "units": {"cost": "priced bit-operations per clip (plotted in T = 1e12)", "f1": "macro F1, %"},
        "panel_a_fsdkaggle2018_field": p1, "panel_a_widened_hull_t25": wide, "changes_from_v1": [
            "(a) adds the MN10-Q8 + MN20-Q8 average (hexagon, ochre) and the widened hull (dashed grey, envelope from MN10-Q8 "
            "through mn05+dymn04, mn05+mn10 and mn10+mn20 to always-average; the two middle vertices carry no marker)",
            "(a) annotations: '+2.83 over two-expert hull' and '+0.27 over widened hull'",
            "(a) y upper limit 78.5 to 79.0 so the second annotation fits inside the panel",
            "legend: 4 columns (was 3), 'equal-cost hull' renamed 'two-expert hull', two entries added; (b) and (c) unchanged"],
        "panel_b_fsd50k_field_descriptive": p2,
        "panel_c_budget_choice": {"fsdkaggle2018": b1, "fsd50k": b2},
        "audit_cross_check": aud,
        "always_average_cost": "mn_cost + ast_cost from primary_by_condition.csv, asserted equal to the 'avg' cost in primary_by_seed_fold.csv",
        "not_drawn": ["a test-set (field mix) budget sweep for FSDKaggle2018: the T21 tables hold the chosen budget only",
                      "the FSD50K test-set sweep in AUDIT-D2/audit_every_budget_test.csv has no cost column and is an audit file",
                      "the random gate at the cascade's rate (field F1 in random_gate_f1_not_plotted)"],
        "overlap_check": check, "asserts": asserts}, indent=1, default=float))
    print("saved", HERE / f"{STEM}.pdf")


if __name__ == "__main__":
    main()
