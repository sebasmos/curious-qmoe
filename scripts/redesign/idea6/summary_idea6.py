"""Idea 6 stage 1 summary: the pre-specified primary analysis that decides the
go rule, the exploratory family, and the tables the README quotes.

Primary analysis (decides; fp32 view, EfficientNet at 32 x 32 bits, what the
embeddings are): for each task, ONE design (cheap kind, expensive config,
rule, operating point) is picked from inner validation only: the largest
gain_iv in selection_<view>.csv (inner-validation F1 minus the time-sharing
hull at the design's inner-validation cost; ties go to the cheaper design).
That design alone is then tested on the test folds against the hull at its
test cost (comparisons_<view>.csv), fold level (seeds averaged per fold),
paired t-test, Holm over the 3 tasks. GO when all three have a positive mean
difference and Holm p < alpha. The fold-level Wilcoxon is reported raw and
with Holm over 3 but cannot decide on ESC-50: its exact two-sided minimum is
2/2^n, 0.0625 with 5 folds (0.00195 with UrbanSound8K's 10). Also reported:
for each fold, the design its own inner validation split alone would pick,
since the pooled pick also reads other folds' inner-validation rows, which
can hold this fold's test clips (scored by other folds' heads).

Exploratory: every cascade point of every task against the hull (tasks x
cheap kinds x expensive configs x rules x points = 432 per cost view), Holm
within that family separately for each p-value series. Not used for the
verdict. The int8 view repeats both with its own families.

    python scripts/redesign/idea6/summary_idea6.py R/analysis --out R/analysis/summary.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TASKS = ["esc50", "esc5", "urban"]
NAMES = {"esc50": "ESC-50, 50 classes", "esc5": "ESC-50, 5 classes", "urban": "UrbanSound8K"}
DESIGN = ["cheap", "expensive", "rule", "point"]
SERIES = ["p_t_fold", "p_w_fold", "p_t_pair", "p_w_pair"]


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p, kind="stable")
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj


def family(A: Path, view: str, alpha: float) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(A / t / f"comparisons_{view}.csv").assign(task=t) for t in TASKS], ignore_index=True)
    for base in ("hull", "static"):
        for s in SERIES:
            df[f"{base}_{s}_holm"] = holm(df[f"{base}_{s}"])
        df[f"{base}_pass_fold"] = (df[f"{base}_diff"] > 0) & (df[f"{base}_p_t_fold_holm"] < alpha)
        df[f"{base}_pass_pair"] = ((df[f"{base}_diff"] > 0) & (df[f"{base}_p_t_pair_holm"] < alpha)
                                   & (df[f"{base}_p_w_pair_holm"] < alpha))
    df["family_size"] = len(df)
    return df


def primary(A: Path, view: str, alpha: float) -> pd.DataFrame:
    """Each task's inner-validation pick, its test comparison with the hull, Holm over the tasks."""
    rows = []
    for t in TASKS:
        sel = pd.read_csv(A / t / f"selection_{view}.csv", dtype={"cheap": str})
        pick = sel.sort_values(["gain_iv", "cost_iv"], ascending=[False, True]).iloc[0]
        comp = pd.read_csv(A / t / f"comparisons_{view}.csv", dtype={"cheap": str})
        r = comp[(comp.cheap == pick.cheap) & (comp.expensive == pick.expensive) & (comp.rule == pick.rule)
                 & (comp.point == pick.point)].iloc[0]
        # leakage check: the design each fold's own inner validation split would pick
        per_fold = {}
        for _, s in sel.iterrows():
            for f, gval in json.loads(s.gain_iv_per_fold).items():
                if f not in per_fold or gval > per_fold[f][0]:
                    per_fold[f] = (gval, tuple(s[k] for k in DESIGN))
        own = {f: v[1] for f, v in per_fold.items()}
        same = sum(v == tuple(pick[k] for k in DESIGN) for v in own.values())
        # nested variant: each test fold uses the design its own inner split picks (no row of that fold's
        # test set is read by its selection); the fold's difference is that design's test difference
        # against its own hull on that fold
        folds = sorted(int(f) for f in own)
        nested = []
        for f in folds:
            d = dict(zip(DESIGN, own[str(f)]))
            rr = comp[(comp.cheap == d["cheap"]) & (comp.expensive == d["expensive"]) & (comp.rule == d["rule"])
                      & (comp.point == d["point"])].iloc[0]
            nested.append(json.loads(rr.hull_fold_diffs)[folds.index(f)])
        nested = np.array(nested)
        from scipy import stats
        n_t, n_w = stats.ttest_1samp(nested, 0.0).pvalue, stats.wilcoxon(nested).pvalue
        rows.append({"task": t, **{k: pick[k] for k in DESIGN}, "f1_iv": pick.f1_iv, "cost_iv": pick.cost_iv,
                     "hull_f1_iv": pick.hull_f1_iv, "gain_iv": pick.gain_iv, "runner_up_gain_iv":
                     sel.sort_values("gain_iv", ascending=False).iloc[1].gain_iv,
                     "f1": r.f1, "cost": r.cost, "rate": r.rate, "hull_f1": r.hull_f1, "hull_lo": r.hull_lo,
                     "hull_hi": r.hull_hi, "lam": r.lam, "diff": r.hull_diff, "fold_diffs": r.hull_fold_diffs,
                     "p_t_fold": r.hull_p_t_fold, "p_w_fold": r.hull_p_w_fold, "p_t_pair": r.hull_p_t_pair,
                     "p_w_pair": r.hull_p_w_pair, "n_folds": int(r.hull_n_folds),
                     "folds_whose_own_iv_picks_the_same": f"{same} of {len(own)}",
                     "own_iv_picks": json.dumps({f: " ".join(map(str, v)) for f, v in own.items()}),
                     "nested_diff": float(nested.mean()), "nested_fold_diffs": json.dumps(nested.round(6).tolist()),
                     "nested_p_t_fold": float(n_t), "nested_p_w_fold": float(n_w)})
    df = pd.DataFrame(rows)
    df["p_t_fold_holm3"] = holm(df.p_t_fold)
    df["p_w_fold_holm3"] = holm(df.p_w_fold)
    df["nested_p_t_fold_holm3"] = holm(df.nested_p_t_fold)
    df["nested_passed"] = (df.nested_diff > 0) & (df.nested_p_t_fold_holm3 < alpha)
    df["passed"] = (df["diff"] > 0) & (df.p_t_fold_holm3 < alpha)
    return df


def fmt_p(p):
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"


def name(r):
    return f"{r.cheap} -> {r.expensive}, {r.rule}, {r.point}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("analysis", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args(argv)
    A = a.analysis
    fam = {v: family(A, v, a.alpha) for v in ("fp32", "int8")}
    for v, df in fam.items():
        df.to_csv(A / f"family_{v}.csv", index=False)

    prim = {v: primary(A, v, a.alpha) for v in ("fp32", "int8")}
    for v, df in prim.items():
        df.to_csv(A / f"primary_{v}.csv", index=False)
    verdict = {"alpha": a.alpha,
               "primary": {v: {"verdict": "GO" if df["passed"].all() else "NO GO", "family_size": len(df),
                               "rows": json.loads(df.to_json(orient="records"))} for v, df in prim.items()},
               "exploratory_family_size": {v: int(len(df)) for v, df in fam.items()}, "views": {}}
    for v, df in fam.items():
        per_task = {}
        for t in TASKS:
            g = df[df.task == t]
            best = g.sort_values("hull_diff", ascending=False).iloc[0]
            per_task[t] = {"n_points": int(len(g)), "pass_fold": int(g.hull_pass_fold.sum()),
                           "pass_pair": int(g.hull_pass_pair.sum()),
                           "best": {k: (best[k].item() if hasattr(best[k], "item") else best[k]) for k in
                                    DESIGN + ["f1", "cost", "rate", "hull_f1", "hull_diff", "hull_lo", "hull_hi",
                                              "lam", "hull_p_t_fold", "hull_p_t_fold_holm", "hull_p_w_fold",
                                              "hull_p_t_pair_holm", "hull_p_w_pair_holm"]}}
        both = df.pivot_table(index=DESIGN, columns="task", values="hull_pass_fold", aggfunc="first")
        strict = both[both.all(axis=1)].reset_index()[DESIGN].to_dict("records")
        lenient = all(per_task[t]["pass_fold"] > 0 for t in TASKS)
        verdict["views"][v] = {"per_task": per_task, "designs_passing_all_tasks": strict,
                               "every_task_has_a_passing_point": lenient,
                               "exploratory_outcome": "one design passes on all tasks" if strict else "none"}
    verdict["verdict"] = verdict["primary"]["fp32"]["verdict"]
    (A / "verdict.json").write_text(json.dumps(verdict, indent=1, default=str))

    L = [f"# Idea 6 stage 1 summary\n",
         f"Verdict (primary analysis, fp32 view): **{verdict['verdict']}**. One design per task picked on inner "
         f"validation only, tested once on the test folds against the time-sharing hull, fold-level paired t-test, "
         f"Holm over the 3 tasks.\n"]
    for v in ("fp32", "int8"):
        P = prim[v]
        L.append(f"\n## Primary analysis, cost view {v}: {verdict['primary'][v]['verdict']}\n")
        L.append("| task | design picked on inner validation | inner-val gain over hull (pts; runner-up) | test F1 "
                 "| bit-ops (T) | escalated | hull F1 (mix) | delta (pts) | fold t p (raw / Holm 3) | fold Wilcoxon p "
                 "(raw / Holm 3) | pair t / Wilcoxon p (raw) | folds whose own inner split picks it | pass |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in P.itertuples():
            mix = f"{r.hull_lo} + {r.lam:.2f} x {r.hull_hi}" if r.lam else r.hull_lo
            L.append(f"| {NAMES[r.task]} | {r.cheap} -> {r.expensive}, {r.rule}, {r.point} | "
                     f"{100 * r.gain_iv:+.2f} ({100 * r.runner_up_gain_iv:+.2f}) | {r.f1:.4f} | {r.cost / 1e12:.2f} | "
                     f"{r.rate:.2f} | {r.hull_f1:.4f} ({mix}) | {100 * r.diff:+.2f} | {fmt_p(r.p_t_fold)} / "
                     f"{fmt_p(r.p_t_fold_holm3)} | {fmt_p(r.p_w_fold)} / {fmt_p(r.p_w_fold_holm3)} | "
                     f"{fmt_p(r.p_t_pair)} / {fmt_p(r.p_w_pair)} | {r.folds_whose_own_iv_picks_the_same} | "
                     f"{'yes' if r.passed else 'no'} |")
        L.append("\nSensitivity (does not decide): nested selection, each test fold using the design its own inner "
                 "split picks, so no selection reads that fold's test clips.\n")
        L.append("| task | per-fold picks | delta (pts) | fold differences (pts) | fold t p (raw / Holm 3) | "
                 "fold Wilcoxon p (raw) | pass |")
        L.append("|---|---|---|---|---|---|---|")
        for r in P.itertuples():
            picks = "; ".join(f"fold {f}: {v}" for f, v in json.loads(r.own_iv_picks).items())
            fd = ", ".join(f"{100 * x:+.1f}" for x in json.loads(r.nested_fold_diffs))
            L.append(f"| {NAMES[r.task]} | {picks} | {100 * r.nested_diff:+.2f} | {fd} | {fmt_p(r.nested_p_t_fold)} / "
                     f"{fmt_p(r.nested_p_t_fold_holm3)} | {fmt_p(r.nested_p_w_fold)} | "
                     f"{'yes' if r.nested_passed else 'no'} |")
    L.append(f"\n# Exploratory (not used for the verdict)\n\nFamily: {verdict['exploratory_family_size']['fp32']} "
             f"cascade points (3 tasks x 4 cheap heads x 3 AST tiers x 2 rules x 6 points) against the time-sharing "
             f"hull, Holm within the family for each p-value series; the int8 view is a second family of "
             f"{verdict['exploratory_family_size']['int8']}. 'Best' below is picked on test results.\n")
    for v in ("fp32", "int8"):
        vv = verdict["views"][v]
        L.append(f"\n## Exploratory family, cost view {v}\n")
        L.append(f"Designs passing on all three tasks (fold-level t, Holm): {len(vv['designs_passing_all_tasks'])}"
                 + (": " + "; ".join(f"{d['cheap']} -> {d['expensive']}, {d['rule']}, {d['point']}"
                                     for d in vv["designs_passing_all_tasks"]) if vv["designs_passing_all_tasks"] else "")
                 + "\n")
        L.append("| task | points passing (fold t, Holm) | points passing (pair t and Wilcoxon, Holm) | best cascade "
                 "| F1 | bit-ops (T) | escalated | hull F1 | hull mix | delta (pts) | fold t p (raw, Holm) | "
                 "fold Wilcoxon p (raw) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for t in TASKS:
            p, b = vv["per_task"][t], vv["per_task"][t]["best"]
            mix = f"{b['hull_lo']} + {b['lam']:.2f} x {b['hull_hi']}" if b["lam"] else b["hull_lo"]
            L.append(f"| {NAMES[t]} | {p['pass_fold']} of {p['n_points']} | {p['pass_pair']} | {b['cheap']} -> "
                     f"{b['expensive']}, {b['rule']}, {b['point']} | {b['f1']:.4f} | {b['cost'] / 1e12:.2f} | "
                     f"{b['rate']:.2f} | {b['hull_f1']:.4f} | {mix} | {100 * b['hull_diff']:+.2f} | "
                     f"{fmt_p(b['hull_p_t_fold'])}, {fmt_p(b['hull_p_t_fold_holm'])} | {fmt_p(b['hull_p_w_fold'])} |")

    # every design on every task, fp32 view, rule ast and avg
    df = fam["fp32"]
    L.append("\n## Every cascade against the hull, fp32 view (delta in F1 points, Holm fold-level t p)\n")
    for t in TASKS:
        g = df[df.task == t]
        L.append(f"\n### {NAMES[t]}\n")
        L.append("| cheap -> AST, rule | " + " | ".join(["b05", "b10", "b20", "b30", "b50", "match"]) + " |")
        L.append("|---|" + "---|" * 6)
        for (ch, ex, ru), h in g.groupby(["cheap", "expensive", "rule"], sort=False):
            h = h.set_index("point")
            cells = [f"{100 * h.loc[p, 'hull_diff']:+.1f} ({fmt_p(h.loc[p, 'hull_p_t_fold_holm'])}; "
                     f"{h.loc[p, 'cost'] / 1e12:.2f} T)" for p in ["b05", "b10", "b20", "b30", "b50", "match"]]
            L.append(f"| {ch} -> {ex}, {ru} | " + " | ".join(cells) + " |")

    # static front
    L.append("\n## Static systems (macro F1, mean over (seed, fold); bit-ops per clip, fp32 view)\n")
    L.append("| system | " + " | ".join(NAMES[t] for t in TASKS) + " | bit-ops (T) |")
    L.append("|---|---|---|---|---|")
    st = {t: pd.read_csv(A / t / "static_points_fp32.csv").set_index("system") for t in TASKS}
    keep = ["eff:esc", "eff:8", "eff:4", "eff:bitnet", "ast:n3g64+esc", "ast:w3g64+esc", "ast:w4_p8+esc",
            "ast:w4g128+esc", "ast:w4g128+4", "ast:w5+esc", "ast:w6+esc", "ast:w8g64+esc", "ast:fp32+esc"]
    for s in keep:
        vals = [f"{st[t].loc[s, 'f1']:.4f}" if s in st[t].index else "-" for t in TASKS]
        cost = next(st[t].loc[s, "cost"] for t in TASKS if s in st[t].index)
        L.append(f"| {s} | " + " | ".join(vals) + f" | {cost / 1e12:.3f} |")

    # gate quality and calibration
    L.append("\n## Gate AUROC (1 - margin) for 'cheap wrong, AST right', pooled over folds, mean over seeds "
             "(min to max)\n")
    L.append("| task | cheap | w8g64 | w4g128 | fp32 | AUROC for 'cheap wrong' |")
    L.append("|---|---|---|---|---|---|")
    for t in TASKS:
        au = pd.read_csv(A / t / "auroc.csv")
        for ch, g in au.groupby("cheap", sort=False):
            cells = []
            for ex in ["w8g64", "w4g128", "fp32"]:
                x = g[g.expensive == ex].auroc_cheap_wrong_ast_right
                cells.append(f"{x.mean():.3f} ({x.min():.3f} to {x.max():.3f})")
            w = g[g.expensive == "w8g64"].auroc_cheap_wrong
            L.append(f"| {NAMES[t]} | {ch} | " + " | ".join(cells) + f" | {w.mean():.3f} |")
    L.append("\n## Calibration of the cheap tier (top-label ECE, 15 bins; confidence minus accuracy), "
             "mean over (seed, fold)\n")
    L.append("| task | " + " | ".join(["eff:esc", "eff:8", "eff:4", "eff:bitnet", "ast:w8g64+esc"]) + " |")
    L.append("|---|---|---|---|---|---|")
    for t in TASKS:
        ca = pd.read_csv(A / t / "calibration.csv").groupby("system")[["ece", "conf_gap"]].mean()
        L.append(f"| {NAMES[t]} | " + " | ".join(f"{ca.loc[s, 'ece']:.3f} ({ca.loc[s, 'conf_gap']:+.3f})"
                                                  for s in ["eff:esc", "eff:8", "eff:4", "eff:bitnet", "ast:w8g64+esc"])
                 + " |")
    # oracle
    L.append("\n## Oracle\n")
    L.append("Same escalation count: the cascade's own count, spent first on clips the cheap head gets wrong and "
             "the escalated decision gets right. Full: escalate exactly those clips. Rule ast, fp32 view.\n")
    L.append("| task | cheap -> AST | point | cascade F1 | oracle F1, same count | gap (pts) | full oracle F1 | "
             "full oracle escalation | full oracle bit-ops (T) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for t in TASKS:
        g = df[(df.task == t) & (df.rule == "ast")]
        of = pd.read_csv(A / t / "oracle_full.csv")
        for ex in ["w4g128", "w8g64"]:
            for ch in ["esc", "4"]:
                for pt in ["b10", "b30"]:
                    r = g[(g.cheap == ch) & (g.expensive == ex) & (g.point == pt)].iloc[0]
                    o = of[(of.cheap == ch) & (of.expensive == ex) & (of.rule == "ast")]
                    L.append(f"| {NAMES[t]} | {ch} -> {ex} | {pt} | {r.f1:.4f} | {r.oracle_f1:.4f} | "
                             f"{100 * (r.oracle_f1 - r.f1):+.2f} | {o.f1.mean():.4f} | {o.rate.mean():.3f} | "
                             f"{o.cost_fp32.mean() / 1e12:.2f} |")
    a.out.write_text("\n".join(L) + "\n")
    print(json.dumps({v: {"primary_verdict": verdict["primary"][v]["verdict"],
                          "primary_picks": {r["task"]: f"{r['cheap']} -> {r['expensive']}, {r['rule']}, {r['point']}"
                                            for r in verdict["primary"][v]["rows"]},
                          "exploratory_designs_all_tasks": len(vv["designs_passing_all_tasks"]),
                          "exploratory_pass_fold": {t: vv["per_task"][t]["pass_fold"] for t in TASKS}}
                      for v, vv in verdict["views"].items()}, indent=1))


if __name__ == "__main__":
    main()
