"""T17 scoring (ANALYSIS_PLAN_T17.md sections 4 and 5). READS TEST LABELS. Run after t17_select.py (selection.json) and the jobs.
Per (bed, condition, seed, fold, arm): accuracy and macro-F1 on all rows of the condition. python t17_score.py --run DIR"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t17_common import *
from common import f1_macro
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True, type=Path); ap.add_argument("--beds", nargs="+", default=None); ap.add_argument("--maxfold", type=int, default=None)
ap.add_argument("--selftest", action="store_true", help="mechanics test only: replaces every test label by a random one, output is meaningless")
a = ap.parse_args()
SEL = json.load(open(a.run / "inputs/selection.json"))
sm = lambda z: (lambda e: e / e.sum(1, keepdims=True))(np.exp(z - z.max(1, keepdims=True)))
SINGLE = ["p10", "p10b", "p20", "b20"] + [f"{t}_T{T}_a{al}" for t in ("kdA", "kdF") for T in (1, 2, 4) for al in (0.25, 0.5, 0.75)]
PAIR = ["indep"] + [f"{m}{l}" for m in ("ncl", "dec") for l in (0.1, 0.3, 0.5)]
rows = []
for bed in (a.beds or BEDS):
    D = load_bed(bed); y = D["y"]
    if a.selftest: y = np.random.default_rng(0).permutation(y)
    K = BEDS[bed]["ncls"]
    for fold in range(1, (a.maxfold or BEDS[bed]["nf"]) + 1):
        for seed in (0, 1, 2):
            R = np.load(a.run / bed / "raw" / f"s{seed}_f{fold}.npz"); S = np.load(T1C / bed / f"raw_strength/s{seed}_f{fold}.npz")
            te, tr = R["te"], R["tr"]; yte = y[te]; cnd = S["cond"] if bed == "tau" else None
            for cn in CONDITIONS[bed] + ["clean"]:
                g = groups(bed, cn, te, D["assign"], cnd); yr = gather_y(yte, g); sel = SEL[bed][str(fold)][cn]
                ratio = (len(g[0][1]) if bed == "tau" else len(te)) / len(tr)
                Z = lambda name, mi: sm(gather({c: R[f"te|{name}|{mi}|{c}"] for c, _ in g}, g).astype(np.float64))
                ast = sm(gather({c: S[f"UA_E2b|{c}"] for c, _ in g}, g).astype(np.float64))
                P = {}
                for nm in SINGLE:
                    if f"te|{nm}|0|{g[0][0]}" in R.files: P[nm] = Z(nm, 0)
                for nm in PAIR: P[nm] = (Z(nm, 0) + Z(nm, 1)) / 2
                P["boost"] = (P["p10"] + P["b20"]) / 2; P["twoseed"] = (P["p10"] + P["p10b"]) / 2; P["sel_ncl"] = P[sel["ncl"]]; P["sel_dec"] = P[sel["dec"]]
                P["sel_kdA"] = P[sel["kdA"]]; P["sel_kdF"] = P[sel["kdF"]]; P["ast_w4"] = ast
                P["sys_plain"] = (P["p10"] + ast) / 2; P["sys_kdA"] = (P["sel_kdA"] + ast) / 2; P["sys_kdF"] = (P["sel_kdF"] + ast) / 2
                chosen = dict(sel_ncl=sel["ncl"], sel_dec=sel["dec"], sel_kdA=sel["kdA"], sel_kdF=sel["kdF"])
                for nm, p in P.items():
                    pr = p.argmax(1); rows.append(dict(bed=bed, cond=cn, seed=seed, fold=fold, arm=nm, acc=100 * (pr == yr).mean(), f1=100 * f1_macro(yr, pr, K), n=len(yr), ratio=ratio, chosen=chosen.get(nm, "")))
        print(bed, "fold", fold, flush=True)
pd.DataFrame(rows).to_csv(a.run / "arm_metrics_per_seed_fold.csv", index=False)
