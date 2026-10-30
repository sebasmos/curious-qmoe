"""T1b strength gate (ANALYSIS_PLAN_T1B.md section 5, step 3): is E2a stronger than E1 on the clips E1 flags?
Reads <run>/<ds>/raw_strength. python t1b_strength.py --run DIR --ds urban|tau"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_table, write_json
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True, type=Path); ap.add_argument("--ds", required=True); a = ap.parse_args()
if a.ds == "urban": y = load_table().label.values.astype(np.int64); NF, CONDS = 10, {"all": None}
else:
    y = pd.read_csv(Path.home() / "Documents/DATASETS/TAU2020Mobile/ast_768/tau2020mobile.csv", usecols=["class_id"]).class_id.values.astype(np.int64); NF, CONDS = 5, {"shift": 1, "clean": 0}
rows = []
for f in sorted(glob.glob(str(a.run / a.ds / "raw_strength/s*_f*.npz"))):
    R = np.load(f); te, flag = R["te"], R["flag"]; yt = y[te]; e1 = R["z1_te"].argmax(1)
    for cn, cv in CONDS.items():
        k = np.ones(len(te), bool) if cv is None else (R["cond"] == cv)
        fl = flag & k
        row = dict(seed=int(R["seed"]), fold=int(R["fold"]), cond=cn, n_flag=int(fl.sum()), E1=(e1[fl] == yt[fl]).mean())
        for e in ("E2a", "E2b", "E2c"):
            if f"{e}_A_U_te" in R.files:
                p = R[f"{e}_A_U_te"].argmax(1); row[e] = (p[fl] == yt[fl]).mean(); row[f"oracle_{e}"] = ((p[fl] == yt[fl]) | (e1[fl] == yt[fl])).mean()
                row[f"{e}_all_cond"] = (p[k] == yt[k]).mean()
        row["E1_all_cond"] = (e1[k] == yt[k]).mean(); rows.append(row)
D = pd.DataFrame(rows); D.to_csv(a.run / a.ds / "strength_per_seed_fold.csv", index=False)
F = D.groupby(["cond", "fold"]).mean(numeric_only=True).drop(columns="seed").reset_index(); F.to_csv(a.run / a.ds / "strength_per_fold.csv", index=False)
out = {"rule": "E2a stronger: mean over folds of (U-A E2a minus E1) on flagged clips >= +2.0 points and positive in >= 7 of 10 folds (TAU: >= 4 of 5), gate condition = all (UrbanSound8K) / shift (TAU)"}
for cn in CONDS:
    g = F[F.cond == cn]; need = 7 if a.ds == "urban" else 4
    o = dict(n_folds=int(len(g)), flagged_share_mean=None, acc_E1=float(g.E1.mean() * 100), acc_all_clips_E1=float(g.E1_all_cond.mean() * 100))
    for e in ("E2a", "E2b", "E2c"):
        if e in g:
            d = (g[e] - g.E1) * 100
            o[e] = dict(acc_flag_uniform_AST_head=float(g[e].mean() * 100), diff_vs_E1_pts=float(d.mean()), folds_positive=int((d > 0).sum()), folds_ge_2=int((d >= 2).sum()),
                        oracle_acc_flag=float(g[f"oracle_{e}"].mean() * 100), acc_all_cond=float(g[f"{e}_all_cond"].mean() * 100), fold_diffs=[round(v, 2) for v in d.values])
    d = o["E2a"]; o["E2a_stronger"] = bool(d["diff_vs_E1_pts"] >= 2.0 and d["folds_positive"] >= need)
    out[cn] = o
gate_cn = "all" if a.ds == "urban" else "shift"
out["gate_condition"] = gate_cn; out["gate_pass"] = out[gate_cn]["E2a_stronger"]
write_json(a.run / a.ds / "strength_verdict.json", out); print(json.dumps(out, indent=1))
