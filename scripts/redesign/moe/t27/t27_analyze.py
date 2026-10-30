"""T27 (ANALYSIS_PLAN.md, POST HOC): T26's S2 against the always-on MN10-Q8 + MN20-Q8 + AST-Q4 average, and against
T25's widened hull extended with that three-model average, on FSDKaggle2018 (d1, T26 Part A's bed) and FSDnoisy18k
(d3, T26 Part C's bed), field mix and clean audio. T26's code path throughout: its beds, scalers, costs, S2 thresholds
(budgets read from T26's verdict files, never reselected), run_test, compare_all and fold_test. The three-model
average enters as one more static of t26_core.statics (mean of the three softmaxes, cost = sum of the three).

    python t27_analyze.py --r R27
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "t26"))
import t26_core as K6  # noqa: E402
import t26_part_a as PA  # noqa: E402
import t26_part_c as PC  # noqa: E402

A = K6.A
R26 = Path.home() / "Documents/qmoe-results/2026-10-29_0912_moe-t26_code-6ecea17"
TRI = "mn10+mn20+ast"
CONDS = ["clean", "field"]
WIDE17 = list(K6.WIDE)
_statics = K6.statics


def statics3(P, c):
    """T26's statics plus the always-on three-model average."""
    pr, cs = _statics(P, c)
    if all(m in P for m in ("mn10", "mn20", "ast")):
        pr[TRI] = ((P["mn10"] + P["mn20"] + P["ast"]) / 3).argmax(1)
        cs[TRI] = float((c["mn10"] + c["mn20"] + c["ast"]).mean())
    return pr, cs


K6.statics = statics3


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def fixed_thresholds(bed, V26):
    """S1 and S2 thresholds at T26's chosen budgets, rebuilt per (seed, fold) exactly as t26_core.select_budgets does."""
    r1 = float(V26["budgets"]["s1"])
    r2, r3 = (float(x) for x in V26["budgets"]["s2"].split("|"))
    th = {"s1": {}, "s2": {}}
    for k in bed.keys:
        _, P, _ = bed.iv(*k)
        th["s1"][k] = A.thr_rate(A.margin(P["mn10"]), r1)
        th["s2"][k] = K6.s2_thr(P["mn10"], P["mn20"], r2, r3)
    return th, {"s1": r1, "s2": [r2, r3]}


def check_t26(TT, CMP17, ref_csv, V26):
    ref = pd.read_csv(ref_csv)
    ref = ref[ref.cond.isin(CONDS)].set_index(["cond", "seed", "fold", "system"])
    mine = TT[TT.system != TRI].set_index(["cond", "seed", "fold", "system"])
    j = mine[["f1", "cost"]].join(ref[["f1", "cost"]], rsuffix="_t26", how="outer")
    if len(j) != len(ref) or j.isna().any().any():
        raise SystemExit("rows differ from T26's by_seed_fold.csv")
    df1 = float((j.f1 - j.f1_t26).abs().max())
    dc = float(((j.cost - j.cost_t26).abs() / j.cost_t26).max())
    dh = {}
    for c in CONDS:
        r = CMP17[(CMP17.cond == c) & (CMP17.system == "s2")].iloc[0]
        dh[c] = abs(float(r.d_hull) - V26["results"][f"s2|{c}"]["d_hull"])
    if max(df1, dc, *dh.values()) > 1e-9:
        raise SystemExit(f"T26 reproduction failed: F1 {df1}, cost {dc}, S2 minus hull {dh}")
    return {"rows": len(j), "worst_f1": df1, "worst_cost_rel": dc, "s2_minus_hull17_diff": dh}


def run(name, bed, V26, ref_csv, out):
    bed.conds, bed.boot = CONDS, None  # field and clean only; no bootstrap in T27
    th, budgets = fixed_thresholds(bed, V26)
    TT, preds = K6.run_test(bed, th)
    K6.WIDE = WIDE17
    CMP17, _ = K6.compare_all(bed, TT, preds, systems=("s2",))
    chk = check_t26(TT, CMP17, ref_csv, V26)
    print(f"[{now()}] {name}: T26 reproduced {chk}", flush=True)
    K6.WIDE = WIDE17 + [TRI]
    CMP18, ST18 = K6.compare_all(bed, TT, preds, systems=("s2",))
    K6.WIDE = WIDE17
    out.mkdir(parents=True, exist_ok=True)
    TT.to_csv(out / "by_seed_fold.csv", index=False)
    ST18.to_csv(out / "statics_by_condition.csv", index=False)
    CMP18.to_csv(out / "s2_vs_extended_hull.csv", index=False)
    rows = []
    for c in CONDS:
        g = TT[TT.cond == c]
        V = lambda nm: g[g.system == nm].set_index(["seed", "fold"]).sort_index()  # noqa: E731
        s2, tri = V("s2"), V(TRI)
        if not s2.index.equals(tri.index):
            raise SystemExit("S2 and the three-model average cover different (seed, fold) cells")
        i = K6.fold_test(s2.f1.to_numpy(), tri.f1.to_numpy(), s2.index, bed.ratio)
        e = CMP18[(CMP18.cond == c) & (CMP18.system == "s2")].iloc[0]
        h = CMP17[(CMP17.cond == c) & (CMP17.system == "s2")].iloc[0]
        r = {"dataset": name, "cond": c, "s2_f1": float(s2.f1.mean()), "s2_cost_T": float(s2.cost.mean()) / 1e12,
             "s2_rate_tier2": float(s2.rate.mean()), "s2_rate_tier3": float(s2.rate_tier3.mean()),
             "tri_f1": float(tri.f1.mean()), "tri_cost_T": float(tri.cost.mean()) / 1e12, "tri_macs_G": float(tri.macs.mean()) / 1e9,
             **{f"i_{k}": v for k, v in i.items()},
             "ii_hull": e.hull, "ii_hull_f1": float(e.hull_f1),
             **{f"ii_{k[:-5]}": (v.item() if hasattr(v, "item") else v) for k, v in e.items() if k.endswith("_hull") and k != "hull_f1"},
             "hull17": h.hull, "hull17_f1": float(h.hull_f1), "hull17_d": float(h.d_hull)}
        for nm in ("s1", "mn10+mn20", "avg"):
            r[f"{nm}_f1"], r[f"{nm}_cost_T"] = float(V(nm).f1.mean()), float(V(nm).cost.mean()) / 1e12
        rows.append(r)
    res = {"dataset": name, "budgets": budgets, "scalers": V26["scalers"], "nb_ratio": bed.ratio, "t26_reproduction": chk,
           "rows": rows}
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"[{now()}] {name}: wrote {out}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=Path, required=True)
    a = ap.parse_args()
    print(f"[{now()}] T27 analysis start (first T27 number computed after this line)", flush=True)
    rows = []
    bed, *_ = PA.build_bed()
    rows += run("fsdkaggle2018", bed, json.loads((R26 / "tables/a/verdict.json").read_text()), R26 / "tables/a/by_seed_fold.csv",
                a.r / "tables" / "d1")
    bed, *_ = PC.build_bed(R26, "d3", [0, 1, 2], PC.NEW)
    rows += run("fsdnoisy18k", bed, json.loads((R26 / "tables/c/d3/verdict.json").read_text()), R26 / "tables/c/d3/by_seed_fold.csv",
                a.r / "tables" / "d3")
    S = pd.DataFrame(rows)
    S.to_csv(a.r / "tables" / "summary.csv", index=False)
    (a.r / "tables" / "summary.json").write_text(json.dumps(rows, indent=1, default=str))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(S[["dataset", "cond", "s2_f1", "s2_cost_T", "tri_f1", "tri_cost_T", "i_d", "i_nb_lo", "i_nb_hi", "i_p_nb", "i_folds_pos",
             "ii_hull", "ii_hull_f1", "ii_d", "ii_nb_lo", "ii_nb_hi", "ii_p_nb", "ii_folds_pos", "hull17_d"]].round(4).to_string())
    print(f"[{now()}] T27 analysis done", flush=True)


if __name__ == "__main__":
    main()
