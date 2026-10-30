"""T17 analysis (ANALYSIS_PLAN_T17.md sections 4 to 6): fold-level contrasts, Nadeau-Bengio, hull, Holm over the six primary tests, verdicts. Reads arm_metrics_per_seed_fold.csv (t17_score.py),
selection.json and the T13 per-cell fold accuracies (stored-logit derived). python t17_analyze.py --run DIR --t13 DIR13"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t17_common import *
import t13_common as T13
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True, type=Path); ap.add_argument("--t13", required=True, type=Path); a = ap.parse_args(); R = a.run; (R / "tables").mkdir(exist_ok=True)
M = pd.read_csv(R / "arm_metrics_per_seed_fold.csv"); SEL = json.load(open(R / "inputs/selection.json"))
F = M.groupby(["bed", "cond", "fold", "arm"])[["acc", "f1", "ratio"]].mean().reset_index()
FA = {k: g.sort_values("fold") for k, g in F.groupby(["bed", "cond", "arm"])}
def arr(bed, cond, arm, col="acc"): return FA[(bed, cond, arm)][col].to_numpy()
def ratio(bed, conds): return float(np.mean([FA[(bed, c, "p10")]["ratio"].mean() for c in conds]))
S13 = pd.read_csv(a.t13 / "cell_fold_acc.csv"); S13 = S13[(S13.kind == "single") & S13.seed.isin([0, 1, 2])].groupby(["task", "cell", "system", "fold"])["acc"].mean().reset_index()
A13 = {k: g.sort_values("fold").acc.to_numpy() for k, g in S13.groupby(["task", "cell", "system"])}
def nb(d, ratio_):
    d = np.asarray(d, float); K = len(d); s2 = d.var(ddof=1); se = np.sqrt((1 / K + ratio_) * s2)
    if se == 0: return dict(mean=float(d.mean()), lo=float(d.mean()), hi=float(d.mean()), p=1.0 if d.mean() == 0 else 0.0)
    t = d.mean() / se; h = stats.t.ppf(.975, K - 1) * se
    return dict(mean=float(d.mean()), lo=float(d.mean() - h), hi=float(d.mean() + h), p=float(2 * stats.t.sf(abs(t), K - 1)))
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); out = np.zeros(len(p)); run = 0.0
    for i, ix in enumerate(o): run = max(run, min(1.0, (len(p) - i) * p[ix])); out[ix] = run
    return out
MAJ = lambda K: K // 2 + 1
_h = {}
def hull_cell(task, cell, cost):
    key = (task, cell, round(cost))
    if key not in _h:
        CT = T13.cost_table(task); av = [e for e in T13.experts_of(task) if (task, cell, e) in A13]
        sp = pd.DataFrame([dict(system=e, f1=A13[(task, cell, e)].mean() / 100, cost=CT[e]) for e in av]); v, lo, hi, lam = T13.A.hull_pair(sp, cost)
        _h[key] = ((A13[(task, cell, lo)] if lam == 0 else (1 - lam) * A13[(task, cell, lo)] + lam * A13[(task, cell, hi)]), f"{lo}|{hi}|{lam:.3f}")
    return _h[key]
def hull_cond(bed, cond, cost): return np.mean([hull_cell(TASK_OF_BED[bed], c, cost)[0] for c in T13_CELLS[cond]], 0)
def best_cond(bed, cond, tag):
    """fold values of the best single (T13 stored experts, chosen per fold on the inner validation of the condition family)."""
    out = []
    for i, f in enumerate(range(1, BEDS[bed]["nf"] + 1)):
        e = SEL[bed][str(f)][cond]["best_single"][tag]["choice"]; out.append(np.mean([A13[(TASK_OF_BED[bed], c, e)][i] for c in T13_CELLS[cond]]))
    return np.array(out)
def pooled(bed, arm, conds): return np.mean([arr(bed, c, arm) for c in conds], 0)
COST = {b: costs17(b) for b in BEDS}
rows, prim = [], []
def contrast(bed, conds, name, x, yv, thr=None, **kw):
    K = len(x); d = x - yv; r = nb(d, ratio(bed, conds)); rows.append(dict(bed=bed, conds="+".join(conds), contrast=name, mean=r["mean"], lo=r["lo"], hi=r["hi"], p=r["p"], folds_ge_0_3=int((d >= .3).sum()), folds_ge_0_5=int((d >= .5).sum()),
                 folds_ge_1=int((d >= 1).sum()), folds_pos=int((d > 0).sum()), K=K, fold_d=np.round(d, 2).tolist(), acc_x=float(x.mean()), acc_y=float(yv.mean()), **kw)); return rows[-1]
for bed in BEDS:
    K = BEDS[bed]["nf"]; C = COST[bed]
    for conds in [TASK_CONDS[bed]] + [[c] for c in TASK_CONDS[bed] if len(TASK_CONDS[bed]) > 1] + [["clean"]]:
        P = lambda arm: pooled(bed, arm, conds); Hh = lambda tag: np.mean([hull_cond(bed, c, C[tag]) for c in conds], 0)
        task_level = conds == TASK_CONDS[bed]
        for arm, ctl in (("sel_ncl", "indep"), ("sel_dec", "indep"), ("boost", "indep"), ("sel_ncl", "twoseed"), ("sel_dec", "twoseed"), ("boost", "twoseed"), ("indep", "twoseed"), ("sel_kdA", "p10"), ("sel_kdF", "p10"),
                         ("sys_kdA", "sys_plain"), ("sys_kdF", "sys_plain"), ("sys_plain", "ast_w4"), ("sys_kdA", "ast_w4"), ("p10", "p20")):
            contrast(bed, conds, f"{arm} - {ctl}", P(arm), P(ctl), level="task" if task_level else "cond")
        for arm in ("sel_ncl", "sel_dec", "boost", "indep", "twoseed", "p10", "p20"): contrast(bed, conds, f"{arm} - hull_a", P(arm), Hh("a"), level="task" if task_level else "cond")
        for arm in ("sys_kdA", "sys_kdF", "sys_plain", "ast_w4"): contrast(bed, conds, f"{arm} - hull_b", P(arm), Hh("b"), level="task" if task_level else "cond")
        if conds != ["clean"]:
            contrast(bed, conds, "sel_ncl - best_single_a", P("sel_ncl"), np.mean([best_cond(bed, c, "a") for c in conds], 0), level="task" if task_level else "cond")
            contrast(bed, conds, "sys_kdA - best_single_b", P("sys_kdA"), np.mean([best_cond(bed, c, "b") for c in conds], 0), level="task" if task_level else "cond")
CT = pd.DataFrame(rows); CT.to_csv(R / "tables/contrasts.csv", index=False)
def get(bed, name, conds=None):
    s = CT[(CT.bed == bed) & (CT.contrast == name) & (CT.level == "task")] if conds is None else CT[(CT.bed == bed) & (CT.contrast == name) & (CT.conds == conds)]; return s.iloc[0]
for bed in BEDS:
    K = BEDS[bed]["nf"]; mj = MAJ(K)
    c1, c2, c3 = get(bed, "sel_ncl - indep"), get(bed, "sel_ncl - twoseed"), get(bed, "sel_ncl - hull_a")
    ok_a = bool(c1["mean"] >= .5 and c2["mean"] >= .6 and c3["mean"] >= 1.0 and np.sum(np.array(c1.fold_d) >= .5) >= mj and np.sum(np.array(c2.fold_d) >= .6) >= mj and np.sum(np.array(c3.fold_d) >= 1.0) >= mj)
    prim.append(dict(bed=bed, part="a", clause_means=f"{c1['mean']:+.2f} / {c2['mean']:+.2f} / {c3['mean']:+.2f}", p_raw=max(c1.p, c2.p, c3.p), rule_ok=ok_a, folds=f"{np.sum(np.array(c1.fold_d) >= .5)}/{np.sum(np.array(c2.fold_d) >= .6)}/{np.sum(np.array(c3.fold_d) >= 1.0)} of {K}"))
    g, s_ = get(bed, "sel_kdA - p10"), get(bed, "sys_kdA - hull_b")
    ok_b = bool(g["mean"] >= 1.5 and s_["mean"] >= 1.0 and np.sum(np.array(g.fold_d) >= 1.5) >= mj and np.sum(np.array(s_.fold_d) >= 1.0) >= mj)
    prim.append(dict(bed=bed, part="b", clause_means=f"{g['mean']:+.2f} / {s_['mean']:+.2f}", p_raw=max(g.p, s_.p), rule_ok=ok_b, folds=f"{np.sum(np.array(g.fold_d) >= 1.5)}/{np.sum(np.array(s_.fold_d) >= 1.0)} of {K}"))
PR = pd.DataFrame(prim); PR["p_holm6"] = holm(PR.p_raw.to_numpy()); PR["p_x15"] = np.minimum(1.0, PR.p_holm6 * 15); PR["task_pass"] = PR.rule_ok & (PR.p_holm6 < .05); PR["survives_family15"] = PR.rule_ok & (PR.p_x15 < .05); PR.to_csv(R / "tables/primary.csv", index=False)
verd = {}
for part in "ab":
    sub = PR[PR.part == part]; verd[part] = dict(tasks_passing=int(sub.task_pass.sum()), tasks_rule_only=int(sub.rule_ok.sum()), survives_family15=int(sub.survives_family15.sum()))
# stop rule
stop = {}
for arm, ctl in (("sel_ncl", "indep"), ("sel_dec", "indep"), ("boost", "indep"), ("sel_kdA", "p10"), ("sel_kdF", "p10")):
    stop[arm] = {b: float(get(b, f"{arm} - {ctl}")["mean"]) for b in BEDS}
any_gain = any(v >= .3 for d in stop.values() for v in d.values())
for part, arms in (("a", ["sel_ncl", "sel_dec", "boost"]), ("b", ["sel_kdA", "sel_kdF"])):
    mx = max(v for ar in arms for v in stop[ar].values()); verd[part]["max_pooled_gain_over_control"] = mx
    verd[part]["label"] = "PASS" if verd[part]["tasks_passing"] >= 2 else ("STOP" if mx < .3 else "NO PASS")
verd["stop_table"] = stop; verd["any_arm_gains_0.3_somewhere"] = any_gain
json.dump(verd, open(R / "tables/verdict.json", "w"), indent=1)
# accuracy table of every arm per condition
ACC = F.groupby(["bed", "cond", "arm"])[["acc", "f1"]].mean().reset_index(); ACC.to_csv(R / "tables/arm_accuracy.csv", index=False)
CHO = M[M.chosen != ""].groupby(["bed", "arm", "chosen"]).size().reset_index(name="n"); CHO.to_csv(R / "tables/chosen_hyperparameters.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(PR.to_string()); print(json.dumps(verd, indent=1)); print(CT[CT.level == "task"][["bed", "contrast", "mean", "lo", "hi", "p", "folds_ge_0_5", "K"]].round(3).to_string())
