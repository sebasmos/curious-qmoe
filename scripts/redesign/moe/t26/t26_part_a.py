"""T26 Part A (ANALYSIS_PLAN.md C5), POST HOC: S1 and S2 on FSDKaggle2018 (T21's D1 pool design) from the stored score
files of T21 (MN10-Q8, AST-Q4) and T25 (MN04, MN05, MN20, DyMN04 at W8A8), against the widened hull at each system's
own cost. MN20-Q8's FSDKaggle2018 scores were seen in T25, so every number here is post hoc.

Before any S1 or S2 number: every static of the widened set and the T21 cascade are rebuilt on every (condition,
seed, fold) and must equal T25's tables/d1/t25_by_seed_fold.csv to 1e-9 (F1 and cost), and the T21 cascade minus the
widened hull at the cascade's cost must equal T25's H1-wide and H2-wide numbers to 1e-9.

    python t26_part_a.py --r R26
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "t25"))
import t26_core as K6  # noqa: E402
import t25_analyze as T25  # noqa: E402

T, A = K6.T, K6.A
Q = Path.home() / "Documents/qmoe-results"
R21 = Q / "2026-10-26_1408_moe-t21_code-731570e"
R25 = Q / "2026-10-28_2247_moe-t25_code-cc3b048"
NEW = ["mn04", "mn05", "mn20", "dymn04"]


def build_bed(D="d1", seeds=(0, 1, 2, 3, 4)):
    seeds = list(seeds)
    K = int(pd.read_csv(R21 / "csv" / D / "pool.csv").class_id.nunique())
    sc21 = pd.read_csv(R21 / "tables" / D / "primary_scaler_choice.csv").set_index("tier").chosen
    sc25 = pd.read_csv(R25 / "tables" / D / "t25_scaler_choice.csv").set_index("tier").chosen
    V21 = json.loads((R21 / "tables" / D / "verdict.json").read_text())
    V25 = json.loads((R25 / "tables" / D / "verdict.json").read_text())
    scaler = {"mn10": sc21["cheap"], "ast": sc21["ast"], **{m: sc25[m] for m in NEW}}
    Z = {"mn10": T.Scores(R21, D, "pool", "mn10_w8a8", seeds), "ast": T.Scores(R21, D, "pool", "ast_w4g128", seeds),
         **{m: T.Scores(R25, D, "pool", f"{m}_w8a8", seeds) for m in NEW}}
    C = Z["mn10"]
    for m, z in Z.items():
        if not (np.array_equal(z.y, C.y) and np.array_equal(z.fold, C.fold)):
            raise SystemExit(f"{m}: rows or folds differ from T21's MN10")
    c21, _ = T.per_clip_costs(R21, D, K)
    cn = T25.cnn_costs(R25, R21, D, K)
    cost = {"mn10": c21["priced"][0], "ast": c21["priced"][1], **{m: cn[m][0] for m in NEW}, "frontend": cn["frontend"][0]}
    macs = {"mn10": c21["macs"][0], "ast": c21["macs"][1], **{m: cn[m][1] for m in NEW}}
    keys = [(s, f) for s in seeds for f in C.folds]
    y = C.y
    rate = T.BUDGETS[V21["design"]["budget"]]
    cthr = {(s, f): A.thr_rate(A.margin(C.iv(s, f, "clean", scaler["mn10"])[1]), rate) for s, f in keys}
    n = len(y)
    fold_n = {f: int((C.fold == f).sum()) for f in C.folds}
    ratio = float(np.mean([fold_n[f] / (n - fold_n[f] - (n - sum(fold_n.values()))) for f in C.folds]))
    if abs(ratio - V25["nb_ratio"]) > 1e-12:
        raise SystemExit("Nadeau-Bengio ratio differs from T25's")

    def iv(s, f):
        idx = None
        P = {}
        for m, z in Z.items():
            i2, p = z.iv(s, f, "clean", scaler[m])
            if idx is None:
                idx = i2
            elif not np.array_equal(i2, idx):
                raise SystemExit(f"{m} s{s} f{f}: inner-validation rows differ")
            P[m] = p
        if np.intersect1d(idx, np.flatnonzero(C.fold == f)).size:
            raise SystemExit("inner validation overlaps test")
        return y[idx], P, {m: v[idx] for m, v in cost.items()}

    def test(c, s, f):
        te = np.flatnonzero(C.fold == f)
        return y[te], {m: z.test(s, c, scaler[m])[te] for m, z in Z.items()}, {**{m: v[te] for m, v in cost.items()},
                                                                             "_macs": {m: v[te] for m, v in macs.items()}}

    bed = K6.Bed("fsdkaggle2018", K, keys, iv, test, T.CONDS, ratio, boot="perfold", cascade_thr=cthr, boot_seed=T.BOOT_SEED)
    return bed, scaler, V21, V25


def check_t25(TT, CMP, V25, D="d1"):
    ref = pd.read_csv(R25 / "tables" / D / "t25_by_seed_fold.csv").set_index(["cond", "seed", "fold", "system"])
    mine = TT[TT.system.isin(K6.WIDE + ["cascade"])].set_index(["cond", "seed", "fold", "system"])
    j = mine.join(ref[["f1", "cost"]], rsuffix="_t25", how="outer")
    if j[["f1", "f1_t25"]].isna().any().any():
        raise SystemExit("rows differ from T25's t25_by_seed_fold.csv")
    df1 = float((j.f1 - j.f1_t25).abs().max())
    dc = float(((j.cost - j.cost_t25).abs() / j.cost_t25).max())
    if max(df1, dc) > 1e-9:
        raise SystemExit(f"T25 reproduction failed: F1 {df1}, cost {dc}")
    w = {}
    for cond, key in (("field", "H1_wide"), ("clean", "H2_wide")):
        r = CMP[(CMP.cond == cond) & (CMP.system == "cascade")].iloc[0]
        w[cond] = abs(r.d_hull - V25[key]["d"])
        if w[cond] > 1e-9:
            raise SystemExit(f"T25 {key} not reproduced: {r.d_hull} against {V25[key]['d']}")
    return {"rows": len(j), "worst_f1": df1, "worst_cost_rel": dc, "h1_wide_diff": w["field"], "h2_wide_diff": w["clean"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=Path, required=True)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = a.r / "tables" / "a"
    out.mkdir(parents=True, exist_ok=True)
    ap_smoke = a.smoke
    bed, scaler, V21, V25 = build_bed()
    if a.check_only:  # statics, T21 cascade and its widened hull only (T25's published numbers); no S1 or S2
        TT, preds = K6.run_test(bed, {})
        CMP, _ = K6.compare_all(bed, TT, preds, systems=("cascade",))
        print("T25 statics, cascade and widened-hull gains reproduced:", check_t25(TT, CMP, V25), flush=True)
        return
    if ap_smoke:  # code check only: labels permuted, every F1 is chance level; nothing is written
        K6.smoke_permute(bed)
        out = Path(a.r) / "_smoke"
        out.mkdir(parents=True, exist_ok=True)
    SEL, pick, th, sp_iv = K6.select_budgets(bed)
    TT, preds = K6.run_test(bed, th)
    CMP, ST = K6.compare_all(bed, TT, preds)
    chk = {"smoke": True} if ap_smoke else check_t25(TT, CMP, V25)
    print("T25 statics, cascade and widened-hull gains reproduced:", chk, flush=True)
    SEL.to_csv(out / "budget_selection.csv", index=False)
    sp_iv.to_csv(out / "iv_statics.csv", index=False)
    TT.to_csv(out / "by_seed_fold.csv", index=False)
    CMP.to_csv(out / "compare_by_condition.csv", index=False)
    ST.to_csv(out / "statics_by_condition.csv", index=False)
    V = {"part": "A", "label": "POST HOC (MN20-Q8's FSDKaggle2018 scores were seen in T25)", "dataset": bed.name,
         "seeds": sorted({k[0] for k in bed.keys}), "nb_ratio": bed.ratio, "scalers": scaler, "budgets": pick,
         "t21_cascade_budget": V21["design"]["budget"], "t25_reproduction": chk, "family": K6.FAMILY, "results": {}}
    for sysn in ("s1", "s2", "cascade", "mn10+mn20"):
        for cond in ("field", "clean"):
            if sysn == "mn10+mn20":
                g = TT[(TT.cond == cond) & (TT.system == sysn)]
                V["results"][f"{sysn}|{cond}"] = {"f1": float(g.f1.mean()), "cost": float(g.cost.mean())}
                continue
            r = CMP[(CMP.cond == cond) & (CMP.system == sysn)].iloc[0]
            V["results"][f"{sysn}|{cond}"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()
                                              if k.endswith(("_hull", "_vs_mn10+mn20", "_vs_cascade", "_vs_s1")) or k in ("f1", "cost", "rate", "hull", "hull_f1")}
    (out / "verdict.json").write_text(json.dumps(V, indent=1, default=str))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(CMP[CMP.cond.isin(["clean", "field"])][["cond", "system", "f1", "cost", "rate", "hull", "hull_f1", "d_hull", "nb_lo_hull", "nb_hi_hull",
                                                   "p_nb_hull", "folds_pos_hull"]].round(4).to_string())


if __name__ == "__main__":
    main()
