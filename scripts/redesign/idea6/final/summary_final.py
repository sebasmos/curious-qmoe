"""Tables of the idea 6 final round: families with Holm, gate indicators, summary.md and verdict.json.

    python summary_final.py --r R --s2 S2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stage2"))
sys.path.insert(0, str(HERE))
from idea5b_common import holm  # noqa: E402
import corrupt as CR  # noqa: E402

TASKS = ["esc50", "urban"]
NAME = {"esc50": "ESC-50 (50 classes)", "urban": "UrbanSound8K"}
CONDS = ["clean"] + [CR.cond_label(k, lv) for k in CR.KINDS for lv in CR.LEVELS[k]]


def fmt(x, nd=3):
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return ""
        if x != 0 and abs(x) < 10 ** -nd:
            return f"{x:.1e}"
        return f"{x:.{nd}f}"
    return str(x)


def md(df, nd=3):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt(r[c], nd) for c in cols) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", required=True, type=Path)
    ap.add_argument("--s2", required=True, type=Path)
    a = ap.parse_args()
    A = a.r / "analysis"
    L, verdict = [], {}
    L.append("# Idea 6 final round: summary (every number is in the CSVs next to this file)\n")

    # ── Part B ──────────────────────────────────────────────────────────────
    b1 = pd.concat([pd.read_csv(A / f"b_{t}" / "b1_steps_raw.csv") for t in TASKS], ignore_index=True)
    b1["p_holm"] = holm(b1.p_greater.to_numpy())
    b1["family_size"] = len(b1)
    b1["step_rises"] = (b1.p_holm < 0.05) & (b1["diff"] > 0)
    b1.to_csv(A / "b1_trend_steps.csv", index=False)
    tr = b1.groupby(["task", "corruption"]).agg(steps_rising=("step_rises", "sum"), rate_clean=("rate_from", "first"),
                                                rate_severe=("rate_to", "last")).reset_index()
    tr["rises_with_severity"] = tr.steps_rising == 3
    tr.to_csv(A / "b1_trend.csv", index=False)
    verdict["B1"] = {f"{r.task}/{r.corruption}": {"rises": bool(r.rises_with_severity), "steps_rising": int(r.steps_rising)}
                     for r in tr.itertuples()}
    L.append("## B1. Escalation against severity, picked cheap tier at its budget, clean threshold (family of 18, Holm)\n")
    L.append("One-sided paired t-test over fold means of the escalation-rate step; `rises_with_severity` needs all three "
             "steps with Holm p below 0.05 and a positive difference.\n")
    L.append(md(tr) + "\n")
    L.append(md(b1[["task", "corruption", "step", "rate_from", "rate_to", "diff", "n_folds", "p_greater", "p_holm"]]) + "\n")

    b2 = []
    for t in TASKS:
        d = pd.read_csv(A / f"b_{t}" / "cascade_vs_comparators.csv")
        d.insert(0, "task", t)
        b2.append(d)
    b2 = pd.concat(b2, ignore_index=True)
    long = []
    for r in b2.itertuples():
        if r.condition == "clean":
            continue
        for nm in ("hull", "random", "noise"):
            long.append({"task": r.task, "condition": r.condition, "comparator": nm, "cascade_f1": r.f1,
                         "comparator_f1": getattr(r, f"{nm}_f1") if nm != "hull" else r.hull_f1,
                         "diff_pts": 100 * getattr(r, f"{nm}_diff"), "fold_diffs_min": 100 * getattr(r, f"{nm}_diff_fold_min"),
                         "fold_diffs_max": 100 * getattr(r, f"{nm}_diff_fold_max"), "p_t_fold": getattr(r, f"{nm}_p_t_fold"),
                         "p_nb": getattr(r, f"{nm}_p_nb"), "p_signflip": getattr(r, f"{nm}_p_signflip")})
    B2 = pd.DataFrame(long)
    B2["p_holm"] = holm(B2.p_t_fold.to_numpy())
    B2["family_size"] = len(B2)
    B2.to_csv(A / "b2_cascade_vs_comparators.csv", index=False)
    L.append("## B2. Picked cascade against the restricted hull, the random gate and the noise-detector gate (descriptive, family of 54, Holm on the fold-level t)\n")
    for t in TASKS:
        x = b2[b2.task == t]
        d_ = x[["condition", "rate", "cheap_f1", "ast_f1", "f1", "cost", "hull_f1", "random_f1", "noise_f1", "oracle_f1"]].copy()
        L.append(f"### {NAME[t]}: condition means (F1 at the cascade's cost; cost in bit-ops)\n")
        L.append(md(d_) + "\n")
        y = B2[B2.task == t].pivot(index="condition", columns="comparator", values=["diff_pts", "p_holm"])
        y.columns = [f"{c}_{k}" for c, k in y.columns]
        y = y.reindex([c for c in CONDS if c != "clean"]).reset_index()
        L.append(f"Cascade minus comparator, points of F1 and Holm p:\n")
        L.append(md(y[["condition", "diff_pts_hull", "p_holm_hull", "diff_pts_random", "p_holm_random", "diff_pts_noise", "p_holm_noise"]]) + "\n")

    # gate AUROC
    g_all = []
    for t in TASKS:
        meta = json.loads((A / f"b_{t}" / "meta.json").read_text())
        g = pd.read_csv(A / f"b_{t}" / "gate_auroc.csv")
        d = pd.read_csv(A / f"b_{t}" / "detect.csv")
        tier, kind = {"esc50": ("mn20", "4"), "urban": ("mn10", "esc")}[t]
        g = g[(g.tier == tier) & (g.kind == kind)].groupby("condition")[["error", "auroc_wrong", "auroc_wrong_noise_lr", "auroc_fixable"]].mean()
        d = d[(d.tier == tier) & (d.kind == kind)].groupby("condition")[["auroc_margin", "auroc_noise_lr"]].mean()
        d.columns = ["detect_margin", "detect_noise_lr"]
        g = g.join(d).reindex(CONDS).reset_index()
        g.insert(0, "task", t)
        g["random_gate"] = 0.5
        g["margin_beats_noise_lr_on_errors"] = g.auroc_wrong > g.auroc_wrong_noise_lr
        g_all.append(g)
    G = pd.concat(g_all, ignore_index=True)
    G.to_csv(A / "b3_gate_auroc.csv", index=False)
    L.append("## B3. Gate AUROC per condition, picked cheap tier (mean over seeds)\n")
    L.append("`auroc_wrong`: margin score for cheap-wrong clips; `auroc_wrong_noise_lr`: idea 7's detector on the same targets; "
             "`auroc_fixable`: margin for cheap-wrong, AST-right clips; `detect_*`: AUROC for telling the corrupted clips of a fold from its clean clips.\n")
    L.append(md(G.drop(columns=["margin_beats_noise_lr_on_errors"])) + "\n")

    # gate failure indicators
    fail = []
    for t in TASKS:
        for c in CONDS[1:]:
            r = B2[(B2.task == t) & (B2.condition == c) & (B2.comparator == "random")].iloc[0]
            g = G[(G.task == t) & (G.condition == c)].iloc[0]
            fail.append({"task": t, "condition": c, "i_not_beating_random_gate": bool(r.diff_pts <= 0 or r.p_holm >= 0.05),
                         "random_diff_pts": r.diff_pts, "random_p_holm": r.p_holm,
                         "ii_margin_auroc_le_noise_lr": bool(g.auroc_wrong <= g.auroc_wrong_noise_lr),
                         "auroc_margin": g.auroc_wrong, "auroc_noise_lr": g.auroc_wrong_noise_lr})
    F = pd.DataFrame(fail)
    F.to_csv(A / "b3_gate_failure_indicators.csv", index=False)
    verdict["gate_failures"] = {f"{r.task}/{r.condition}": {"i": bool(r.i_not_beating_random_gate), "ii": bool(r.ii_margin_auroc_le_noise_lr)}
                                for r in F.itertuples() if r.i_not_beating_random_gate or r.ii_margin_auroc_le_noise_lr}
    L.append("### Gate failure indicators (plan: (i) cascade does not beat the random gate with Holm p below 0.05; (ii) margin AUROC at or below the noise-detector's)\n")
    L.append(md(F) + "\n")

    # escalation of every tier and head kind
    L.append("## Escalation rate at the clean threshold, every cheap tier (mean over seeds and folds)\n")
    for t in TASKS:
        e = pd.read_csv(A / f"b_{t}" / "escalation.csv")
        e = e.groupby(["tier", "kind", "budget", "condition"]).rate.mean().unstack("condition")[CONDS].reset_index()
        e.to_csv(A / f"b_{t}" / "escalation_summary.csv", index=False)
        L.append(f"### {NAME[t]}, head kinds esc and 4, budgets b05 and b30 (full table in b_{t}/escalation_summary.csv)\n")
        L.append(md(e[e.kind.isin(["esc", "4"]) & e.budget.isin(["b05", "b30"])]) + "\n")
    (A / "verdict_b.json").write_text(json.dumps(verdict, indent=1))
    (A / "summary_b.md").write_text("\n".join(L))
    print("\n".join(L)[:6000])


if __name__ == "__main__":
    main()
