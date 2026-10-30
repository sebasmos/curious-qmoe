"""T13 step 2 (ANALYSIS_PLAN_T13.md section 5): pre-select one pair per task on CLEAN INNER-VALIDATION clips only, and fix the best single at equal or lower cost
for every combination. Reads y only at inner-validation indices; no test clip, no test fold label. python t13_preselect.py --out DIR"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t13_common import *
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
SHIFT = lambda task: ["clean"] if False else ([] if task == "tau" else NOISE + CORR)
res, rows = {}, []
for task, T in TASKS.items():
    ex = experts_of(task); cost = cost_table(task); seeds = T["seeds"]
    shifted = ["shift"] if task == "tau" else NOISE + CORR
    cells = {e: {c for c in (["clean", "shift"] if task == "tau" else ["clean", "field"] + NOISE + CORR) if available(task, e, c)} for e in ex}
    iv = {}                                          # (expert, seed, fold) -> (logits, y)
    Dy = {}
    for s in seeds:
        for e in ex:
            D = load(task, e, "clean", s)
            for f, (idx, lg) in D["iv"].items(): iv[(e, s, f)] = softmax(lg)
            Dy[s] = {f: D["y"][idx] for f, (idx, lg) in D["iv"].items()}
    folds = sorted({f for (_, _, f) in iv})
    def acc(members, s, f): return float((np.mean([iv[(e, s, f)] for e in members], 0).argmax(1) == Dy[s][f]).mean())
    single = {e: float(np.mean([acc((e,), s, f) for s in seeds for f in folds])) for e in ex}
    sp = pd.DataFrame([dict(system=e, f1=single[e], cost=cost[e]) for e in ex])
    for c in combos(task):
        cc = float(sum(cost[e] for e in c)); m = float(np.mean([acc(c, s, f) for s in seeds for f in folds]))
        v, lo, hi, lam = A.hull_pair(sp, cc)
        pool = sorted(set.intersection(*[cells[e] for e in c]) & set(shifted), key=lambda x: (shifted + ["clean", "field"]).index(x) if x in shifted else 99)
        cand = [e for e in ex if cost[e] <= cc * (1 + 1e-12) and set(pool) <= cells[e]]
        bs = max(cand, key=lambda e: (single[e], -cost[e]))
        rows.append(dict(task=task, combo=cname(c), size=len(c), cost=cc, iv_acc_mix=m * 100, iv_acc_hull=v * 100, gain_over_hull=(m - v) * 100, hull_lo=lo, hull_hi=hi, hull_lam=lam,
                         best_single=bs, best_single_iv_acc=single[bs] * 100, best_single_cost=cost[bs], n_pool_cells=len(pool), pool_cells=" ".join(pool)))
    R = pd.DataFrame(rows); Rt = R[(R.task == task) & (R["size"] == 2)].sort_values(["gain_over_hull", "cost"], ascending=[False, True])
    top = Rt.iloc[0]
    res[task] = dict(pair=top.combo, gain_over_hull=float(top.gain_over_hull), cost=float(top.cost), best_single=top.best_single, pool_cells=top.pool_cells.split(), single_iv_acc={e: single[e] * 100 for e in ex},
                     experts=ex, cost_table=cost, best_single_by_combo={r.combo: r.best_single for r in R[R.task == task].itertuples()})
    print(task, "preselected", top.combo, "gain over hull %.2f" % top.gain_over_hull, "cost %.3g" % top.cost, "best single", top.best_single, "pool", len(res[task]["pool_cells"]), "cells")
    print(Rt[["combo", "cost", "iv_acc_mix", "iv_acc_hull", "gain_over_hull", "best_single"]].to_string(index=False))
pd.DataFrame(rows).to_csv(a.out / "preselect_table.csv", index=False); json.dump(res, open(a.out / "t13_preselected.json", "w"), indent=1)
