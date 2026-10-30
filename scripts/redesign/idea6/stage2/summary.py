"""Idea 6 Stage 2 summary: the pre-registered hypotheses with Holm over the
3 tasks (ANALYSIS_PLAN.md, commit 14c8504), the exploratory families with
their sizes, the extractor check, costs and the humility outputs, from
analyze.py's per-task folders.

    python scripts/redesign/idea6/stage2/summary.py --analysis R/analysis --stage1 S1/analysis \
        --out R/analysis
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from idea5b_common import holm  # noqa: E402

TASKS = ["esc50", "esc5", "urban"]
NAMES = {"esc50": "ESC-50, 50 classes", "esc5": "ESC-50, 5 fixed groups", "urban": "UrbanSound8K",
         "esc5off": "ESC-50, official 5 categories"}
EXTRA = ["esc5off"]  # amendment 3, not pre-registered, outside the Holm families
SNR_ORDER = ["clean", "20db", "10db", "5db", "0db"]


def design_str(d):
    bb, k, e, r, b = d
    return f"{bb} EfficientNet + {k} head -> AST {e}, {r}, {b}"


def holm_table(rows, pcol, dcol):
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df[pcol].to_numpy())
    df["pass"] = (df[dcol] > 0) & (df["p_holm"] < 0.05)
    if "p_nb" in df:
        df["p_nb_holm"] = holm(df["p_nb"].to_numpy())
        df["pass_nb"] = (df[dcol] > 0) & (df["p_nb_holm"] < 0.05)
    return df


def fold_paired(a, b):
    """a - b over common (seed, fold) keys, seeds averaged per fold, t-test over folds."""
    m = a.merge(b, on=["seed", "fold"], suffixes=("_a", "_b"))
    d = (m.f1_a - m.f1_b).groupby(m.fold).mean()
    p = stats.ttest_1samp(d, 0.0).pvalue if len(d) > 1 and not np.allclose(d, d.iloc[0]) else np.nan
    return float((m.f1_a - m.f1_b).mean()), float(p), len(d)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--analysis", required=True, type=Path)
    ap.add_argument("--stage1", required=True, type=Path, help="Stage 1 analysis folder (static.csv per task)")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    D = {t: {"sel": pd.read_csv(a.analysis / t / "selected.csv"), "picks": json.loads((a.analysis / t / "picks.json").read_text()),
             "static": pd.read_csv(a.analysis / t / "static.csv"), "e2": pd.read_csv(a.analysis / t / "e2.csv"),
             "esc": pd.read_csv(a.analysis / t / "escalation.csv"), "auroc": pd.read_csv(a.analysis / t / "auroc.csv"),
             "aurocx": pd.read_csv(a.analysis / t / "auroc_cascade_conditions.csv"),
             "calib": pd.read_csv(a.analysis / t / "calibration.csv"),
             "costs": json.loads((a.analysis / t / "costs.json").read_text()),
             "meta": json.loads((a.analysis / t / "meta.json").read_text())}
         for t in TASKS + [x for x in EXTRA if (a.analysis / x / "meta.json").exists()]}
    L = []  # summary.md lines
    out = {}

    def row_of(t, fam, cond):
        s = D[t]["sel"]
        r = s[(s.family == fam) & (s.condition == cond)]
        if len(r) != 1:
            raise SystemExit(f"{t} {fam} {cond}: {len(r)} rows")
        return r.iloc[0]

    # ── pre-registered hypotheses ──────────────────────────────────────────
    hyp = {}
    for name, fam, cond in [("H1", "H1", "field"), ("H2", "H1", "clean"), ("H3", "H3", "field"), ("H4", "H4", "field")]:
        rows = []
        for t in TASKS:
            r = row_of(t, fam, cond)
            rows.append({"task": t, "design": design_str(D[t]["picks"][fam]["design"]), "condition": cond,
                         "f1": r.f1, "cost_T": r.cost / 1e12, "rate": r.rate, "rate_min": r.rate_min,
                         "rate_max": r.rate_max, "hull_f1": r.hull_f1, "hull": f"{r.hull_lo} / {r.hull_hi} ({r.lam:.2f})",
                         "diff_pts": 100 * r.hull_diff, "fold_min_pts": 100 * r.hull_diff_fold_min,
                         "fold_max_pts": 100 * r.hull_diff_fold_max, "p_t_fold": r.hull_p_t_fold,
                         "p_nb": r.hull_p_nb, "p_signflip": r.hull_p_signflip, "floor": r.hull_signflip_floor,
                         "n_folds": r.hull_n_folds})
        hyp[name] = holm_table(rows, "p_t_fold", "diff_pts")
        hyp[name].to_csv(a.out / f"hypothesis_{name}.csv", index=False)
    go = bool(hyp["H1"]["pass"].all())
    # baselines for the H1 / H2 design: random gate and audio gate, Holm over 3 within each comparison
    base = {}
    for cond in ("field", "clean"):
        for comp in ("random", "audio"):
            rows = []
            for t in TASKS:
                r = row_of(t, "H1", cond)
                rows.append({"task": t, "condition": cond, "f1": r.f1, f"{comp}_f1": r[f"{comp}_f1"],
                             "diff_pts": 100 * r[f"{comp}_diff"], "p_t_fold": r[f"{comp}_p_t_fold"],
                             "p_nb": r[f"{comp}_p_nb"], "p_signflip": r[f"{comp}_p_signflip"]})
            base[(cond, comp)] = holm_table(rows, "p_t_fold", "diff_pts")
    # E1: the H1 design on every condition, Holm over all (task, condition) cells per comparison
    e1 = {}
    for comp, pref in (("hull", "hull"), ("random", "random"), ("audio", "audio"), ("best static", "static")):
        rows = []
        for t in TASKS:
            s = D[t]["sel"]
            for _, r in s[s.family == "H1"].iterrows():
                if pd.isna(r.get(f"{pref}_p_t_fold", np.nan)):
                    continue
                rows.append({"task": t, "condition": r.condition, "f1": r.f1, "rate": r.rate, "cost_T": r.cost / 1e12,
                             "diff_pts": 100 * r[f"{pref}_diff"], "p_t_fold": r[f"{pref}_p_t_fold"]})
        e1[comp] = holm_table(rows, "p_t_fold", "diff_pts")
        e1[comp].to_csv(a.out / f"e1_{pref}.csv", index=False)
    # E2: every design on the field mixture against the hull, Holm over all of them
    e2 = pd.concat([D[t]["e2"].assign(task=t) for t in TASKS], ignore_index=True)
    e2["p_holm"] = holm(e2.hull_p_t_fold.to_numpy())
    e2["pass"] = (e2.hull_diff > 0) & (e2.p_holm < 0.05)
    e2.to_csv(a.out / "e2_holm.csv", index=False)
    # E3: unpriced view
    rows = []
    for t in TASKS:
        r = row_of(t, "E3", "field")
        rows.append({"task": t, "design": design_str(D[t]["picks"]["E3"]["design"]), "f1": r.f1, "cost_T": r.cost / 1e12,
                     "rate": r.rate, "hull_f1": r.hull_f1, "diff_pts": 100 * r.hull_diff, "p_t_fold": r.hull_p_t_fold})
    e3 = holm_table(rows, "p_t_fold", "diff_pts")

    verdict = {"GO": go, "rule": "H1 passes on all three tasks (positive fold-level mean, Holm over 3 < 0.05)",
               "picks": {t: D[t]["picks"] for t in TASKS},
               **{h: hyp[h][["task", "design", "diff_pts", "p_t_fold", "p_holm", "pass", "p_nb", "p_nb_holm", "pass_nb",
                             "p_signflip", "floor"]].to_dict("records") for h in hyp},
               "baselines": {f"{c}/{m}": v[["task", "diff_pts", "p_t_fold", "p_holm", "pass", "p_nb", "p_nb_holm",
                                            "p_signflip"]].to_dict("records")
                             for (c, m), v in base.items()},
               "E1_family_sizes": {k: len(v) for k, v in e1.items()}, "E1_passes": {k: int(v["pass"].sum()) for k, v in e1.items()},
               "E2_family_size": len(e2), "E2_passes": int(e2["pass"].sum()),
               "E2_passes_per_task": {t: int(e2[(e2.task == t)]["pass"].sum()) for t in TASKS},
               "E3": e3[["task", "design", "diff_pts", "p_t_fold", "p_holm", "pass"]].to_dict("records")}
    (a.out / "verdict.json").write_text(json.dumps(verdict, indent=1, default=float))

    f = lambda x, n=3: f"{x:.{n}f}"  # noqa: E731
    pfmt = lambda p: f"{p:.2g}"  # noqa: E731
    L.append(f"# Idea 6 Stage 2 summary\n\nVerdict: {'GO' if go else 'NO GO'} ({verdict['rule']}).\n")
    for h, title in [("H1", "H1 primary: field mixture, priced costs, design and thresholds from clean inner validation"),
                     ("H2", "H2: clean replication (same designs and thresholds)"),
                     ("H3", "H3: FP32 cheap backbone only, field mixture"), ("H4", "H4: W8A8 cheap backbone only, field mixture")]:
        L.append(f"## {title}\n")
        L.append("| task | design | F1 | bit-ops (T) | escalated (min to max per pair) | hull (lo / hi, share) | hull F1 | diff (pts) | fold diffs (pts) | fold t p raw / Holm 3 | pass | Nadeau-Bengio p raw / Holm 3 | sign-flip p (floor) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in hyp[h].iterrows():
            L.append(f"| {NAMES[r.task]} | {r.design} | {f(r.f1)} | {f(r.cost_T, 2)} | {f(r.rate, 2)} ({f(r.rate_min, 2)} to {f(r.rate_max, 2)}) | "
                     f"{r.hull} | {f(r.hull_f1)} | {r.diff_pts:+.2f} | {r.fold_min_pts:+.2f} to {r.fold_max_pts:+.2f} | "
                     f"{pfmt(r.p_t_fold)} / {pfmt(r.p_holm)} | {'yes' if r['pass'] else 'no'} | "
                     f"{pfmt(r.p_nb)} / {pfmt(r.p_nb_holm)} | {pfmt(r.p_signflip)} ({pfmt(r.floor)}) |")
        L.append("")
    if "esc5off" in D:
        t = "esc5off"
        L.append("## Secondary task (amendment 3, not pre-registered): ESC-50 with the official five categories\n")
        L.append("Same procedure as the pre-registered tasks; raw p values, outside the Holm families and the verdict.\n")
        L.append("| analysis | condition | design | F1 | bit-ops (T) | escalated | hull F1 | diff (pts) | fold t p | Nadeau-Bengio p | sign-flip p | random gate diff (p) | audio gate diff (p) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name, fam, cond in [("as H1", "H1", "field"), ("as H2", "H1", "clean"), ("as H3", "H3", "field"),
                                ("as H4", "H4", "field"), ("as E3", "E3", "field")]:
            r = row_of(t, fam, cond)
            L.append(f"| {name} | {cond} | {design_str(D[t]['picks'][fam]['design'])} | {r.f1:.3f} | {r.cost / 1e12:.2f} | "
                     f"{r.rate:.2f} | {r.hull_f1:.3f} | {100 * r.hull_diff:+.2f} | {r.hull_p_t_fold:.2g} | {r.hull_p_nb:.2g} | "
                     f"{r.hull_p_signflip:.2g} | {100 * r.random_diff:+.2f} ({r.random_p_t_fold:.2g}) | "
                     + (f"{100 * r.audio_diff:+.2f} ({r.audio_p_t_fold:.2g}) |" if 'audio_diff' in r and not pd.isna(r.audio_diff) else "- |"))
        L.append("")
    L.append("## Baselines for the H1 design (same escalation count; Holm over 3 within each)\n")
    L.append("| condition | comparison | task | cascade F1 | baseline F1 | diff (pts) | fold t p raw / Holm | pass | Nadeau-Bengio p / Holm | sign-flip p |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for (c, m), v in base.items():
        for _, r in v.iterrows():
            L.append(f"| {c} | {m} gate | {NAMES[r.task]} | {f(r.f1)} | {f(r[f'{m}_f1'])} | {r.diff_pts:+.2f} | "
                     f"{pfmt(r.p_t_fold)} / {pfmt(r.p_holm)} | {'yes' if r['pass'] else 'no'} | "
                     f"{pfmt(r.p_nb)} / {pfmt(r.p_nb_holm)} | {pfmt(r.p_signflip)} |")
    L.append("")
    L.append("## E1: the H1 design on every condition (Holm within each comparison over all cells)\n")
    L.append("| task | condition | F1 | escalated | bit-ops (T) | vs hull (pts, Holm p) | vs random gate | vs audio gate | vs best static |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    E1h = e1["hull"].set_index(["task", "condition"])
    for (t, c), r in E1h.iterrows():
        cells = []
        for comp in ("random", "audio", "best static"):
            v = e1[comp].set_index(["task", "condition"])
            cells.append(f"{v.loc[(t, c), 'diff_pts']:+.2f} ({pfmt(v.loc[(t, c), 'p_holm'])})" if (t, c) in v.index else "-")
        L.append(f"| {NAMES[t]} | {c} | {f(r.f1)} | {f(r.rate, 2)} | {f(r.cost_T, 2)} | {r.diff_pts:+.2f} ({pfmt(r.p_holm)}) | "
                 + " | ".join(cells) + " |")
    L.append(f"\nFamily sizes: {verdict['E1_family_sizes']}; passes: {verdict['E1_passes']}.\n")
    L.append(f"## E2: every design on the field mixture against the hull\n\n{len(e2)} comparisons, Holm over all: "
             f"{verdict['E2_passes']} pass ({verdict['E2_passes_per_task']}). Positive mean difference: "
             f"{int((e2.hull_diff > 0).sum())}. Best per task:\n")
    L.append("| task | design | F1 | bit-ops (T) | escalated | diff (pts) | fold t p | Holm p |")
    L.append("|---|---|---|---|---|---|---|---|")
    for t in TASKS:
        g = e2[e2.task == t].sort_values("hull_diff", ascending=False).head(3)
        for _, r in g.iterrows():
            L.append(f"| {NAMES[t]} | {design_str([r.cheap_bb, r.cheap, r.expensive, r.rule, r.budget])} | {f(r.f1)} | "
                     f"{f(r.cost / 1e12, 2)} | {f(r.rate, 2)} | {100 * r.hull_diff:+.2f} | {pfmt(r.hull_p_t_fold)} | {pfmt(r.p_holm)} |")
    L.append("\nE2 by cheap backbone and AST tier (share of designs with a positive mean difference, field mixture):\n")
    e2["pos"] = e2.hull_diff > 0
    L.append(e2.pivot_table(index=["task", "cheap_bb"], columns="expensive", values="pos", aggfunc="mean").round(2).to_markdown())
    L.append("")
    L.append("## E3: unpriced bit-ops (own selection), field mixture\n")
    L.append("| task | design | F1 | bit-ops (T) | escalated | hull F1 | diff (pts) | fold t p raw / Holm 3 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in e3.iterrows():
        L.append(f"| {NAMES[r.task]} | {r.design} | {f(r.f1)} | {f(r.cost_T, 2)} | {f(r.rate, 2)} | {f(r.hull_f1)} | "
                 f"{r.diff_pts:+.2f} | {pfmt(r.p_t_fold)} / {pfmt(r.p_holm)} |")
    L.append("")

    # ── extractor check and static systems ─────────────────────────────────
    L.append("## Extractor check: clean F1 of the rebuilt heads against Stage 1's CSV heads\n")
    L.append("| task | head | Stage 1 (CSV) | rebuilt FP32 | diff (pts, fold t p) | rebuilt W8A8 | W8A8 minus FP32 (pts, fold t p) |")
    L.append("|---|---|---|---|---|---|---|")
    ext = []
    for t in TASKS:
        s1 = pd.read_csv(a.stage1 / t / "static.csv")
        st = D[t]["static"]
        for k in ["esc", "8", "4", "bitnet"]:
            old = s1[s1.system == f"eff:{k}"][["seed", "fold", "f1"]]
            new = st[(st.system == f"eff:fp32:{k}") & (st.condition == "clean")][["seed", "fold", "f1"]]
            q = st[(st.system == f"eff:w8a8:{k}") & (st.condition == "clean")][["seed", "fold", "f1"]]
            d1, p1, _ = fold_paired(new, old)
            d2, p2, _ = fold_paired(q, new)
            ext.append({"task": t, "head": k, "stage1": old.f1.mean(), "fp32": new.f1.mean(), "w8a8": q.f1.mean(),
                        "diff_pts": 100 * d1, "p": p1, "w8a8_minus_fp32_pts": 100 * d2, "p_w8a8": p2})
            L.append(f"| {NAMES[t]} | {k} | {f(old.f1.mean())} | {f(new.f1.mean())} | {100 * d1:+.2f} ({pfmt(p1)}) | "
                     f"{f(q.f1.mean())} | {100 * d2:+.2f} ({pfmt(p2)}) |")
    pd.DataFrame(ext).to_csv(a.out / "extractor_check.csv", index=False)
    L.append("")
    L.append("## Static systems, macro F1 by condition (FP32 head for the cheap tier and the AST)\n")
    for t in TASKS:
        st = D[t]["static"]
        sl = D[t]["meta"]["static_ast_configs"]
        shorts = [f"ast:{x}+esc" for x in sorted({n for v in sl.values() for n in v if "@" in n})]
        sysl = ["eff:fp32:esc", "eff:w8a8:esc", "eff:fp32:4", "eff:w8a8:4", "ast:n4+esc", "ast:w4g128+esc", "ast:w5+esc",
                "ast:w6+esc", "ast:w8g64+esc", "ast:fp32+esc"] + shorts
        pv = st[st.system.isin(sysl)].pivot_table(index="system", columns="condition", values="f1", aggfunc="mean")
        cost = st.groupby("system").cost_priced.mean() / 1e12
        pv.insert(0, "bit-ops priced (T)", cost.reindex(pv.index))
        L.append(f"{NAMES[t]}:\n\n" + pv.reindex([s for s in sysl if s in pv.index]).round(3).to_markdown() + "\n")
    # E5: short-input AST against the same configuration at 1024 frames (FP32 head, own heads at each length)
    L.append("## E5: AST at the short input length minus the same configuration at 1024 frames, F1 points (FP32 head)\n")
    rows = []
    for t in TASKS:
        st = D[t]["static"]
        for sname in sorted({x for x in st.system.unique() if "@" in x and x.endswith("+esc")}):
            base_name = sname.split("@")[0] + "+esc"
            for c, g in st[st.system == sname].groupby("condition"):
                full_ = st[(st.system == base_name) & (st.condition == c)]
                if len(full_):
                    d, p, _ = fold_paired(g[["seed", "fold", "f1"]], full_[["seed", "fold", "f1"]])
                    rows.append({"task": t, "config": sname.replace("ast:", "").replace("+esc", ""), "condition": c,
                                 "f1_short": g.f1.mean(), "f1_1024": full_.f1.mean(), "diff_pts": 100 * d,
                                 "fold_t_p": p, "cost_short_T": g.cost_priced.mean() / 1e12,
                                 "cost_1024_T": full_.cost_priced.mean() / 1e12})
    E5 = pd.DataFrame(rows)
    E5.to_csv(a.out / "e5_short_vs_full.csv", index=False)
    if len(E5):
        L.append(E5.pivot_table(index=["task", "config"], columns="condition", values="diff_pts").round(2).to_markdown())
        L.append("\nClean F1 and priced cost:\n")
        L.append(E5[E5.condition == "clean"][["task", "config", "f1_short", "f1_1024", "diff_pts", "fold_t_p",
                                               "cost_short_T", "cost_1024_T"]].round(4).to_markdown(index=False) + "\n")
    # E4: W8A8 minus FP32 per head and condition
    L.append("## E4: W8A8 minus FP32 cheap backbone, static F1 points (mean over pairs)\n")
    rows = []
    for t in TASKS:
        st = D[t]["static"]
        for k in ["esc", "8", "4", "bitnet"]:
            a8 = st[st.system == f"eff:w8a8:{k}"].groupby("condition").f1.mean()
            a32 = st[st.system == f"eff:fp32:{k}"].groupby("condition").f1.mean()
            rows.append({"task": t, "head": k, **(100 * (a8 - a32)).round(2).to_dict()})
    L.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")

    # ── costs ──────────────────────────────────────────────────────────────
    L.append("## Costs per clip (bit-ops, T)\n")
    c0 = D["esc50"]["costs"]
    L.append("| item | priced | unpriced |\n|---|---|---|")
    for bb in ["fp32", "w8a8"]:
        s = c0["eff_backbone"][bb]
        L.append(f"| EfficientNet-B3 backbone {bb} | {s['backbone_bitops_rescale1024'] / 1e12:.4f} | {s['backbone_bitops'] / 1e12:.4f} |")
    L.append(f"| mel image front end (ESC-50) | {c0['eff_backbone']['fp32']['frontend_bitops_mean'] / 1e12:.4f} | same |")
    L.append(f"| AST fbank front end (ESC-50) | {c0['fbank_bitops_mean'] / 1e12:.4f} | same |")
    for cfg in ["n4", "w4", "w4g128", "w5", "w6", "w8g64", "w3g64", "fp32"] + sorted(
            x for x in c0["ast_backbone"]["priced"] if "@" in x):
        if cfg in c0["ast_backbone"]["priced"]:
            L.append(f"| AST {cfg} backbone (ESC-50) | {c0['ast_backbone']['priced'][cfg] / 1e12:.3f} | "
                     f"{c0['ast_backbone']['unpriced'][cfg] / 1e12:.3f} |")
    cu = D["urban"]["costs"]["ast_backbone"]
    for cfg in sorted(x for x in cu["priced"] if "@" in x):
        L.append(f"| AST {cfg} backbone (UrbanSound8K) | {cu['priced'][cfg] / 1e12:.3f} | {cu['unpriced'][cfg] / 1e12:.3f} |")
    L.append(f"\nEfficientNet storage: FP32 {c0['eff_backbone']['fp32']['storage_bits'] / 8e6:.1f} MB, W8A8 "
             f"{c0['eff_backbone']['w8a8']['storage_bits'] / 8e6:.1f} MB. UrbanSound8K front ends: mel "
             f"{D['urban']['costs']['eff_backbone']['fp32']['frontend_bitops_mean'] / 1e12:.4f} T, fbank "
             f"{D['urban']['costs']['fbank_bitops_mean'] / 1e12:.4f} T.\n")

    # ── humility ───────────────────────────────────────────────────────────
    L.append("## Humility: escalation rate with the clean thresholds\n")
    hum = []
    for t in TASKS:
        bb, k, e, r, b = D[t]["picks"]["H1"]["design"]
        es = D[t]["esc"]
        rows = es[(es.cheap_bb == bb) & (es.cheap == k)]
        for (bud, c), g in rows.groupby(["budget", "condition"]):
            hum.append({"task": t, "cheap": f"{bb}:{k}", "budget": bud, "condition": c, "rate": g.rate.mean()})
        # the H1 design itself (its own budget)
        s = D[t]["sel"]
        for _, rr in s[s.family == "H1"].iterrows():
            hum.append({"task": t, "cheap": f"{bb}:{k}", "budget": f"{b} (H1)", "condition": rr.condition, "rate": rr.rate})
    H = pd.DataFrame(hum)
    allc = H.pivot_table(index=["task", "cheap", "budget"], columns="condition", values="rate", aggfunc="mean")
    order = ["clean"] + [f"{n}__{s}db" for n in ("bg", "pink", "white") for s in (20, 10, 5, 0)] + ["field"]
    allc = allc[[c for c in order if c in allc.columns]]
    L.append(allc.round(3).to_markdown() + "\n")
    # monotonicity over every cheap system and budget
    es = pd.concat([D[t]["esc"].assign(task=t) for t in TASKS])
    m = es.groupby(["task", "cheap_bb", "cheap", "budget", "condition"]).rate.mean().reset_index()
    steps, nonrise = 0, []
    for (t, bb, k, b), g in m.groupby(["task", "cheap_bb", "cheap", "budget"]):
        r = dict(zip(g.condition, g.rate))
        for n in ("bg", "pink", "white"):
            seq = [("clean", r.get("clean"))] + [(f"{n}__{s}db", r.get(f"{n}__{s}db")) for s in (20, 10, 5, 0)]
            seq = [x for x in seq if x[1] is not None]
            for (c1, v1), (c2, v2) in zip(seq, seq[1:]):
                steps += 1
                if v2 <= v1:
                    nonrise.append(f"{t} {bb}:{k} {b} {c1}->{c2} ({v1:.3f} -> {v2:.3f})")
    L.append(f"Steps down in SNR (clean -> 20 -> 10 -> 5 -> 0 dB, per noise kind) over every cheap system and budget: "
             f"{steps}; the rate does not rise in {len(nonrise)}: " + ("; ".join(nonrise[:30]) or "none") + ".\n")
    L.append("## Gate AUROC (1 - margin), H1 cheap system, per condition (mean over seeds)\n")
    rows = []
    for t in TASKS:
        bb, k, e, r, b = D[t]["picks"]["H1"]["design"]
        au = D[t]["auroc"]
        g = au[(au.cheap_bb == bb) & (au.cheap == k)].groupby("condition")[
            [c for c in ("auroc_wrong_margin", "auroc_wrong_noise_lr", "error_rate") if c in au.columns]].mean()
        ax = D[t]["aurocx"]
        gx = ax[(ax.cheap_bb == bb) & (ax.cheap == k) & (ax.expensive == e)].groupby("condition").auroc_fixable.mean()
        g["auroc_cheap_wrong_ast_right"] = gx
        g.insert(0, "task", t)
        rows.append(g.reset_index())
    AU = pd.concat(rows)
    AU.to_csv(a.out / "auroc_h1.csv", index=False)
    L.append(AU.round(3).to_markdown(index=False) + "\n")
    L.append("## Calibration of the H1 cheap system: ECE and confidence minus accuracy, before and after temperature\n")
    rows = []
    for t in TASKS:
        bb, k, e, r, b = D[t]["picks"]["H1"]["design"]
        ca = D[t]["calib"]
        g = ca[ca.system == f"eff:{bb}:{k}"].groupby("condition")[["T", "ece", "conf_gap", "ece_T", "conf_gap_T"]].mean()
        g.insert(0, "task", t)
        rows.append(g.reset_index())
    CB = pd.concat(rows)
    CB.to_csv(a.out / "calibration_h1.csv", index=False)
    L.append(CB.round(3).to_markdown(index=False) + "\n")
    # ── exploratory real shift (amendment 2) and latency (amendment 1) ────
    if (a.out / "e6_cross_dataset.csv").exists():
        e6 = pd.read_csv(a.out / "e6_cross_dataset.csv")
        L.append("## E6: cross-dataset, four shared classes (exploratory)\n")
        L.append("Heads trained on the source dataset's shared-class clips; in = its own test folds, cross = every "
                 "shared-class clip of the other dataset. Accuracy; rates are shares of clips escalated.\n")
        L.append("| source -> target | cheap | AST tier | budget | cheap acc in / cross | AST acc in / cross | cascade acc in / cross | escalated in / cross | AUROC wrong in / cross | AUROC fixable in / cross |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in e6.iterrows():
            L.append(f"| {r.source} -> {r.target} | {r.cheap_bb} | {r.tier} | {r.budget} | {r.cheap_acc_in:.3f} / {r.cheap_acc_cross:.3f} | "
                     f"{r.ast_acc_in:.3f} / {r.ast_acc_cross:.3f} | {r.cascade_acc_in:.3f} / {r.cascade_acc_cross:.3f} | "
                     f"{r.rate_in:.3f} / {r.rate_cross:.3f} | {r.auroc_wrong_in:.3f} / {r.auroc_wrong_cross:.3f} | "
                     f"{r.auroc_fixable_in:.3f} / {r.auroc_fixable_cross:.3f} |")
        t6 = pd.read_csv(a.out / "e6_tests.csv")
        L.append(f"\nCross minus in-domain escalation rate, fold-level t over the source folds, Holm over {len(t6)}: "
                 f"{int(((t6['diff'] > 0) & (t6.p_holm < 0.05)).sum())} pass; positive in {int((t6['diff'] > 0).sum())}.\n")
        L.append(t6.round(4).to_markdown(index=False) + "\n")
    if (a.out / "e7_tests.csv").exists():
        t7 = pd.read_csv(a.out / "e7_tests.csv")
        L.append("## E7: out-of-distribution classes (exploratory)\n")
        L.append("H1 cheap system, clean thresholds; OOD = the other dataset's clips of classes the task lacks. "
                 "Two families of 3 (Holm): OOD minus in-domain escalation at the H1 budget, margin AUROC minus noise_lr AUROC.\n")
        L.append(t7.round(4).to_markdown(index=False) + "\n")
        L.append(pd.read_csv(a.out / "e7_ood.csv").round(3).to_markdown(index=False) + "\n")
    lat = a.out.parent / "latency" / "latency.csv"
    if lat.exists():
        L.append("## CPU latency, batch 1, 2 threads (median ms)\n")
        L.append(pd.read_csv(lat).round(2).to_markdown(index=False) + "\n")
    (a.out / "summary.md").write_text("\n".join(L))
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
