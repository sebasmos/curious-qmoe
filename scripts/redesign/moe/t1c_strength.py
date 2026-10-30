"""T1c strength gate (ANALYSIS_PLAN_T1C.md section 5): per condition, on the flagged rows, is the uniform AST-FP32 head (E2a, input A) at least 2.0 points above E1?
First script that reads test labels. python t1c_strength.py --run DIR --bed urban|esc50|tau"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1c_common import *
from common import write_json
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True, type=Path); ap.add_argument("--bed", required=True); a = ap.parse_args()
bed = a.bed; D = load_bed(bed); y = D["y"]; NF = BEDS[bed]["nf"]; NEED = 7 if bed == "urban" else 3
TESTED = CONDITIONS[bed]; DESC = DESCRIPTIVE[bed]; EX2 = EXPERTS2[bed]
rows = []
for f in sorted(glob.glob(str(a.run / bed / "raw_strength/s*_f*.npz"))):
    R = np.load(f); te = R["te"]; yte = y[te]; cond = R["cond"] if bed == "tau" else None; tau1 = float(R["tau1"])
    for cn in TESTED + DESC:
        g = groups(bed, cn, te, D["assign"], cond); yr = gather_y(yte, g)
        z1 = gather({c: R[f"z1|{c}"] for c, _ in g}, g).astype(np.float64); fl = pmargin(softmax(z1)) < tau1
        row = dict(seed=int(R["seed"]), fold=int(R["fold"]), cond=cn, rows=len(yr), n_flag=int(fl.sum()), flag_share=fl.mean(), E1=(z1.argmax(1) == yr)[fl].mean(), E1_all=(z1.argmax(1) == yr).mean())
        for e in EX2:
            ze = gather({c: R[f"UA_{e}|{c}"] for c, _ in g}, g); ok = ze.argmax(1) == yr
            row[e] = ok[fl].mean(); row[f"{e}_all"] = ok.mean(); row[f"oracle_{e}"] = (ok | (z1.argmax(1) == yr))[fl].mean()
        rows.append(row)
Dd = pd.DataFrame(rows); Dd.to_csv(a.run / bed / "strength_per_seed_fold.csv", index=False)
Fd = Dd.groupby(["cond", "fold"]).mean(numeric_only=True).drop(columns="seed").reset_index(); Fd.to_csv(a.run / bed / "strength_per_fold.csv", index=False)
out = {"rule": f"E2a uniform AST head minus E1 on flagged rows: mean over folds >= +2.0 points and positive in >= {NEED} of {NF} folds", "need_folds": NEED, "n_folds": NF}
for cn in TESTED + DESC:
    g = Fd[Fd.cond == cn]; o = dict(n_folds=int(len(g)), flag_share_mean=float(g.flag_share.mean()), acc_flag_E1=float(g.E1.mean() * 100), acc_all_E1=float(g.E1_all.mean() * 100))
    for e in EX2:
        d = (g[e] - g.E1) * 100
        o[e] = dict(acc_flag_uniform_AST_head=float(g[e].mean() * 100), diff_vs_E1_pts=float(d.mean()), folds_positive=int((d > 0).sum()), folds_ge_2=int((d >= 2).sum()), oracle_acc_flag=float(g[f"oracle_{e}"].mean() * 100),
                    acc_all=float(g[f"{e}_all"].mean() * 100), all_minus_E1_all_pts=float((g[f"{e}_all"] - g.E1_all).mean() * 100), fold_diffs=[round(float(v), 2) for v in d.values])
    d = o["E2a"]; o["gate_pass"] = bool(d["diff_vs_E1_pts"] >= 2.0 and d["folds_positive"] >= NEED) if cn in TESTED else None
    out[cn] = o
out["tested_conditions"] = TESTED; out["gate_pass"] = {cn: out[cn]["gate_pass"] for cn in TESTED}
write_json(a.run / bed / "strength_verdict.json", out)
for cn in TESTED + DESC:
    o = out[cn]; print(cn, "flag_share %.3f E1 %.2f (all %.2f)" % (o["flag_share_mean"], o["acc_flag_E1"], o["acc_all_E1"]), {e: (round(o[e]["diff_vs_E1_pts"], 2), o[e]["folds_positive"], o[e]["folds_ge_2"]) for e in EX2}, "GATE", o["gate_pass"])
