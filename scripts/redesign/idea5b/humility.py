"""Idea 5b, step 4: humility check for one cascade (head, A, B, rule, point) (default: the tuple closest
to GO in verdict.json). The threshold is set once per (seed, fold) on the clean inner-validation split and
applied unchanged to every condition's test clips. Reports the test escalation rate, macro F1 and
bit-ops per condition, and the gate's AUROC (score: minus A's margin) for "A wrong, B right"
(and "A wrong, A+B mean right"), per seed pooled over test folds, then averaged over seeds.

    python humility.py --stage2 STAGE2_DIR --out OUT [--tuple HEAD A B RULE POINT | --key theory_tuple --tag theory]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import BUDGETS, POINTS, Run, cascade_cost, cut_points, f1_curve, f1_fast, margin, pick_match, pick_rate

ap = argparse.ArgumentParser()
ap.add_argument("--stage2", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--tuple", nargs=5, default=None, metavar=("HEAD", "A", "B", "RULE", "POINT"))
ap.add_argument("--key", default="closest_tuple", help="verdict.json entry to use when --tuple is not given")
ap.add_argument("--tag", default="", help="output humility<_tag>.csv")
a = ap.parse_args()
if a.tuple is None:
    t = json.load(open(a.out / "verdict.json"))[a.key]
    head, A, B, rule, point = t["head"], t["A"], t["B"], t["rule"], t["point"]
else:
    head, A, B, rule, point = a.tuple
print("tuple", head, A, B, rule, point, flush=True)

rows, auc_rows = [], []
for ds in ["esc50", "esc5", "urban"]:
    meta = json.load(open(a.out / ds / "meta.json"))
    n_cls = meta["n_cls"]
    sc = a.stage2 / "scores" / ds
    for seed, tag in zip(meta["seeds"], meta["tags"]):
        run = Run(sc / tag / head)
        rAc, rBc = run.get(A, "clean"), run.get(B, "clean")
        thr = {}
        for fold in sorted(rAc["iv"]):
            idx, pAi = rAc["iv"][fold]
            pBi = rBc["iv"][fold][1]
            mi = margin(pAi)
            order = np.argsort(mi, kind="stable")
            js, ts = cut_points(mi[order])
            if point == "match":
                yi = rAc["y"][idx]
                fin = pBi.argmax(1) if rule == "B" else (pAi + pBi).argmax(1)
                cur = f1_curve(yi[order], pAi.argmax(1)[order], fin[order], n_cls)
                ci, _ = pick_match(js, cur, f1_fast(yi, pBi.argmax(1), n_cls))
            else:
                ci = pick_rate(js, len(mi), BUDGETS[POINTS.index(point)])
            thr[fold] = ts[ci]
        for cond in meta["conds"]:
            rA, rB = run.get(A, cond), run.get(B, cond)
            y = rA["y"]
            mA = margin(rA["p"])
            aP, bP, vP = rA["p"].argmax(1), rB["p"].argmax(1), (rA["p"] + rB["p"]).argmax(1)
            fin = bP if rule == "B" else vP
            for fold, t_ in thr.items():
                te = rA["tf"] == fold
                e = mA[te] < t_
                pred = np.where(e, fin[te], aP[te])
                rows.append(dict(dataset=ds, cond=cond, seed=seed, fold=fold, rate=e.mean(),
                                 f1=f1_fast(y[te], pred, n_cls),
                                 bitops=float(cascade_cost(A, B, e.mean(), n_cls, "bitops", head))))
            lab_b = (aP != y) & (bP == y)
            lab_v = (aP != y) & (vP == y)
            auc_rows.append(dict(dataset=ds, cond=cond, seed=seed, auroc_A_wrong_B_right=roc_auc_score(lab_b, -mA),
                                 auroc_A_wrong_avg_right=roc_auc_score(lab_v, -mA), frac_A_wrong_B_right=lab_b.mean()))
        del run

H = pd.DataFrame(rows).groupby(["dataset", "cond"]).agg(rate=("rate", "mean"), f1=("f1", "mean"),
                                                        bitops_T=("bitops", lambda x: x.mean() / 1e12)).reset_index()
U = pd.DataFrame(auc_rows).groupby(["dataset", "cond"]).agg(
    auroc_A_wrong_B_right=("auroc_A_wrong_B_right", "mean"), auroc_min_seed=("auroc_A_wrong_B_right", "min"),
    auroc_A_wrong_avg_right=("auroc_A_wrong_avg_right", "mean"),
    frac_A_wrong_B_right=("frac_A_wrong_B_right", "mean")).reset_index()
H = H.merge(U, on=["dataset", "cond"])
H.insert(0, "tuple", f"head {head}: {A}->{B} {rule} {point} (clean-set threshold)")
H.to_csv(a.out / f"humility{'_' + a.tag if a.tag else ''}.csv", index=False)
pd.set_option("display.width", 250)
print(H.to_string(float_format=lambda v: f"{v:.3f}"))
