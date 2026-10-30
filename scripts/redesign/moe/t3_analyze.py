"""T3 step 4 (ANALYSIS_PLAN_T3T5.md sections 1 and 3): fold-level contrasts, Holm (family of 15 and within the experiment), pass, stop, regression finding. python t3_analyze.py --run DIR"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t3t5_stats import *
from t3t5_common import T35
import pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True, type=Path); ap.add_argument("--tasks", nargs="+", default=T35); a = ap.parse_args(); R = a.run; (R / "tables").mkdir(exist_ok=True)
S = pd.read_csv(R / "t3_fold.csv")
num = [c for c in S.columns if c not in ("task", "cond", "fit", "seed", "fold")]
F = S.groupby(["task", "cond", "fit", "fold"])[num].mean().reset_index()
MAJ = lambda K: K // 2 + 1
def col(task, cond, fit, c): g = F[(F.task == task) & (F.cond == cond) & (F.fit == fit)].sort_values("fold"); return g[c].to_numpy() if c in g else np.array([])
def contrast(task, cond, fit, c1, c2, thr):
    x, y = col(task, cond, fit, c1), col(task, cond, fit, c2)
    if len(x) == 0 or len(y) == 0 or np.isnan(x - y).all(): return None
    d = x - y; ok = ~np.isnan(d); d = d[ok]; r = nb(d, ratio(task, cond="clean" if cond == "clean" else "shift"))
    return dict(task=task, cond=cond, fit=fit, a=c1, b=c2, mean_a=float(x[ok].mean()), mean_b=float(y[ok].mean()), diff=r["mean"], lo=r["lo"], hi=r["hi"], p=r["p"], folds_ge_thr=int((d >= thr).sum()), folds_lt0=int((d < 0).sum()), n_folds=int(len(d)), thr=thr)
# ---- means table ----
cols = [c for c in F.columns if c.startswith(("A_", "B_", "C_"))]
M = F.groupby(["task", "cond", "fit"])[cols + ["rate_A", "rate_B", "pos_C", "n"]].mean().reset_index(); M.to_csv(R / "tables/t3_mean_auroc.csv", index=False)
# ---- contrasts ----
rows = []
for task in a.tasks:
    for cond, fits in (("pool", ("cf", "transfer")), ("clean", ("cf",))):
        for fit in fits:
            for T_ in "AB":
                for other, thr in (("r1", .05), ("js", .05), ("r2", .05), ("entropy", .05)):
                    r = contrast(task, cond, fit, f"{T_}_margin", f"{T_}_{other}", thr)
                    if r: rows.append({**r, "target": T_, "contrast": f"margin - {other}"})
                r = contrast(task, cond, fit, f"{T_}_reg", f"{T_}_margin", .02)
                if r: rows.append({**r, "target": T_, "contrast": "reg - margin"})
                r = contrast(task, cond, fit, f"{T_}_reg", f"{T_}_r1", .05)
                if r: rows.append({**r, "target": T_, "contrast": "reg - r1"})
            for m_ in ("spearman", "auroc"):
                for other in ("r1", "js", "reg"):
                    r = contrast(task, cond, fit, f"C_{m_}_margin", f"C_{m_}_{other}", .05)
                    if r: rows.append({**r, "target": "C_" + m_, "contrast": f"margin - {other}"})
CT = pd.DataFrame(rows); CT.to_csv(R / "tables/t3_contrasts.csv", index=False)
# ---- primary: target B, pool, cross-fitted on the pool ----
pr = []
for task in a.tasks:
    g = CT[(CT.task == task) & (CT.cond == "pool") & (CT.fit == "cf") & (CT.target == "B")].set_index("contrast")
    r1, js_ = g.loc["margin - r1"], g.loc["margin - js"]
    ok = (r1["diff"] >= .05) and (js_["diff"] >= .05) and (r1.folds_ge_thr >= MAJ(r1.n_folds)) and (js_.folds_ge_thr >= MAJ(js_.n_folds))
    rg = g.loc["reg - margin"]
    pr.append(dict(task=task, n_folds=int(r1.n_folds), auroc_margin=r1.mean_a, auroc_r1=r1.mean_b, auroc_js=js_.mean_b, auroc_r2=g.loc["margin - r2"].mean_b, auroc_reg=rg.mean_a, d_r1=r1["diff"], lo_r1=r1.lo, hi_r1=r1.hi, folds_r1=int(r1.folds_ge_thr), p_r1=r1.p,
                   d_js=js_["diff"], lo_js=js_.lo, hi_js=js_.hi, folds_js=int(js_.folds_ge_thr), p_js=js_.p, d_r2=g.loc["margin - r2"]["diff"], p_r2=g.loc["margin - r2"].p, folds_router_ahead=int(r1.folds_lt0), mean_router_ahead=bool(r1["diff"] < 0),
                   cond_margin_beats_by_005=bool(ok), p_task=max(r1.p, js_.p), reg_minus_margin=rg["diff"], reg_lo=rg.lo, reg_hi=rg.hi, reg_folds_ge_002=int(rg.folds_ge_thr), reg_p=rg.p))
PR = pd.DataFrame(pr); PR["p_F15"] = holm(PR.p_task, 15); PR["p_holm3"] = holm(PR.p_task, 3)
PR["pass_F15"] = PR.cond_margin_beats_by_005 & (PR.p_F15 < .05); PR["pass_holm3"] = PR.cond_margin_beats_by_005 & (PR.p_holm3 < .05)
PR["stop_task"] = PR.mean_router_ahead & (PR.folds_router_ahead >= PR.n_folds.apply(MAJ))
PR["reg_finding"] = (PR.reg_minus_margin >= .02) & (PR.reg_folds_ge_002 >= PR.n_folds.apply(MAJ))
PR.to_csv(R / "tables/t3_primary.csv", index=False)
stop = bool(PR.stop_task.any())
V = dict(stop=stop, stop_tasks=PR[PR.stop_task].task.tolist(), n_pass_F15=int(PR.pass_F15.sum()), n_pass_holm3=int(PR.pass_holm3.sum()), tasks_pass_F15=PR[PR.pass_F15].task.tolist(), tasks_pass_holm3=PR[PR.pass_holm3].task.tolist(),
         verdict_F15="STOP" if stop else ("PASS" if PR.pass_F15.all() else "NO PASS"), verdict_holm3="STOP" if stop else ("PASS" if PR.pass_holm3.all() else "NO PASS"), regression_finding_tasks=PR[PR.reg_finding].task.tolist())
json.dump(V, open(R / "tables/t3_verdict.json", "w"), indent=1, default=float)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
print(PR.round(4).T.to_string()); print(json.dumps(V, indent=1))
