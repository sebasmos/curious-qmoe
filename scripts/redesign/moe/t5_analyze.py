"""T5 step 4 (ANALYSIS_PLAN_T3T5.md sections 1, 5, 6, 8): fold-level contrasts, Holm (family of 15 and within the experiment), monotonicity exceptions, verdict. python t5_analyze.py --run DIR"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t3t5_stats import *
from t3t5_common import T35, STEPS_NOISE, STEPS_CORR, bed
import pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True, type=Path); a = ap.parse_args(); R = a.run; (R / "tables").mkdir(exist_ok=True)
AU = pd.read_csv(R / "t5_aurc.csv"); RT = pd.read_csv(R / "t5_rate.csv")
AF = AU.groupby(["task", "cond", "system", "conf", "fold"])[["aurc", "excess", "acc80", "acc90", "auroc_err", "err", "n"]].mean().reset_index()
def fv(task, cond, system, conf, col="aurc"):
    g = AF[(AF.task == task) & (AF.cond == cond) & (AF.system == system) & (AF.conf == conf)].sort_values("fold"); return g[col].to_numpy() * (100 if col in ("aurc", "excess") else 1)
MAJ = lambda K: K // 2 + 1
# ---- (b) risk-coverage contrasts: margin vs random and vs noise detector, per system ----
rows = []
for task in T35:
    for cond in ("pool", "clean"):
        rt = ratio(task, cond="clean" if cond == "clean" else "shift")
        for system in ("cheap", "mix", "single"):
            for conf in ("margin", "msp"):
                m = fv(task, cond, system, conf)
                if len(m) == 0: continue
                K = len(m)
                for other in ("random", "noise_lr"):
                    d = m - fv(task, cond, system, other); s = nb(d, rt)
                    for col in ("excess",):
                        pass
                    dx = fv(task, cond, system, conf, "excess") - fv(task, cond, system, other, "excess"); sx = nb(dx, rt)
                    rows.append(dict(task=task, cond=cond, system=system, conf=conf, other=other, aurc_conf=m.mean(), aurc_other=fv(task, cond, system, other).mean(), err=fv(task, cond, system, "random", "err").mean() * 100 if False else fv(task, cond, system, "random").mean(),
                                     diff=s["mean"], lo=s["lo"], hi=s["hi"], p=s["p"], folds_lower=int((d < 0).sum()), n_folds=K, excess_conf=fv(task, cond, system, conf, "excess").mean(), excess_other=fv(task, cond, system, other, "excess").mean(),
                                     excess_diff=sx["mean"], excess_p=sx["p"], acc80_conf=fv(task, cond, system, conf, "acc80").mean(), acc80_other=fv(task, cond, system, other, "acc80").mean(), acc90_conf=fv(task, cond, system, conf, "acc90").mean(), acc90_other=fv(task, cond, system, other, "acc90").mean(),
                                     auroc_err_conf=fv(task, cond, system, conf, "auroc_err").mean(), auroc_err_other=fv(task, cond, system, other, "auroc_err").mean()))
RC = pd.DataFrame(rows); RC.to_csv(R / "tables/risk_coverage_contrasts.csv", index=False)
# primary: cheap, pool, margin; task-level p = weaker contrast
prim = []
for task in T35:
    g = RC[(RC.task == task) & (RC.cond == "pool") & (RC.system == "cheap") & (RC.conf == "margin")]
    prim.append(dict(task=task, diff_random=float(g[g.other == "random"]["diff"].iloc[0]), diff_noise=float(g[g.other == "noise_lr"]["diff"].iloc[0]), p_random=float(g[g.other == "random"].p.iloc[0]), p_noise=float(g[g.other == "noise_lr"].p.iloc[0]),
                     folds_random=int(g[g.other == "random"].folds_lower.iloc[0]), folds_noise=int(g[g.other == "noise_lr"].folds_lower.iloc[0]), n_folds=int(g.n_folds.iloc[0]), aurc_margin=float(g.aurc_conf.iloc[0]), aurc_random=float(g[g.other == "random"].aurc_other.iloc[0]), aurc_noise=float(g[g.other == "noise_lr"].aurc_other.iloc[0])))
PR = pd.DataFrame(prim); PR["p_task"] = PR[["p_random", "p_noise"]].max(1)
PR["p_F15"] = holm(PR.p_task, 15); PR["p_holm3"] = holm(PR.p_task, 3)
ok = lambda r, suffix: (r.diff_random < 0) and (r.diff_noise < 0) and (r.folds_random >= MAJ(r.n_folds)) and (r.folds_noise >= MAJ(r.n_folds))
PR["pass_F15"] = [ok(r, "") and r.p_F15 < .05 for r in PR.itertuples()]; PR["pass_holm3"] = [ok(r, "") and r.p_holm3 < .05 for r in PR.itertuples()]
PR.to_csv(R / "tables/t5_primary_cheap.csv", index=False)
# the same on the mixture (reported)
prm = []
for task in T35:
    g = RC[(RC.task == task) & (RC.cond == "pool") & (RC.system == "mix") & (RC.conf == "margin")]
    prm.append(dict(task=task, diff_random=float(g[g.other == "random"]["diff"].iloc[0]), diff_noise=float(g[g.other == "noise_lr"]["diff"].iloc[0]), p_random=float(g[g.other == "random"].p.iloc[0]), p_noise=float(g[g.other == "noise_lr"].p.iloc[0]),
                    folds_random=int(g[g.other == "random"].folds_lower.iloc[0]), folds_noise=int(g[g.other == "noise_lr"].folds_lower.iloc[0]), n_folds=int(g.n_folds.iloc[0]), aurc_margin=float(g.aurc_conf.iloc[0])))
PM = pd.DataFrame(prm); PM["p_task"] = PM[["p_random", "p_noise"]].max(1); PM["p_F15"] = holm(PM.p_task, 15); PM["p_holm3"] = holm(PM.p_task, 3); PM.to_csv(R / "tables/t5_primary_mixture.csv", index=False)
# ---- (d) mixture margin vs best single's margin ----
mx = []
for task in T35:
    rt = ratio(task, cond="shift"); m, s_ = fv(task, "pool", "mix", "margin"), fv(task, "pool", "single", "margin"); d = m - s_; r = nb(d, rt)
    de = fv(task, "pool", "mix", "margin", "excess") - fv(task, "pool", "single", "margin", "excess"); re = nb(de, rt)
    da = fv(task, "pool", "mix", "margin", "auroc_err") - fv(task, "pool", "single", "margin", "auroc_err"); ra = nb(da, rt)
    mx.append(dict(task=task, aurc_mix=m.mean(), aurc_single=s_.mean(), diff=r["mean"], lo=r["lo"], hi=r["hi"], p=r["p"], folds_mix_lower=int((d < 0).sum()), n_folds=len(d), excess_mix=fv(task, "pool", "mix", "margin", "excess").mean(), excess_single=fv(task, "pool", "single", "margin", "excess").mean(), excess_diff=re["mean"], excess_p=re["p"], auroc_err_diff=ra["mean"], auroc_err_p=ra["p"], err_mix=fv(task, "pool", "mix", "random").mean(), err_single=fv(task, "pool", "single", "random").mean()))
MX = pd.DataFrame(mx); MX["p_holm3"] = holm(MX.p, 3); MX["loses"] = (MX["diff"] > 0) & (MX.p_holm3 < .05); MX["within_tolerance"] = MX["diff"] <= 0.5; MX["does_not_lose"] = MX.within_tolerance & ~MX.loses; MX.to_csv(R / "tables/t5_mixture_vs_single.csv", index=False)
# ---- (c) abstain rate: monotonicity ----
RF = RT.groupby(["task", "system", "q", "cell", "fold"])[["rate", "n", "err"]].mean().reset_index()
def rate(task, system, q, cell): g = RF[(RF.task == task) & (RF.system == system) & (RF.q == q) & (RF.cell == cell)].sort_values("fold"); return g.rate.to_numpy()
def chains(task, system):
    out = []
    if task == "tau": return [("device:" + d, "clean", f"dev:{d}") for d in ["b", "c", "s1", "s2", "s3", "s4", "s5", "s6"]] 
    for t, cl in STEPS_NOISE.items():
        seq = ["clean"] + cl
        if task == "urban" and system != "cheap": seq = ["clean", "bg__10db", "bg__0db"] if t == "bg" else None
        if seq is None: continue
        out += [(f"noise:{t}", seq[i], seq[i + 1]) for i in range(len(seq) - 1)]
    return out
def corr_chains():
    out = []
    for t, cl in STEPS_CORR.items():
        seq = ["clean"] + cl; out += [(f"corr:{t}", seq[i], seq[i + 1]) for i in range(len(seq) - 1)]
    return out
st = []
for task in T35:
    for system in ("cheap", "mix", "single"):
        for q in (30, 10):
            for kind, steps in (("primary", chains(task, system)), ("exploratory_corruption", corr_chains() if (system == "cheap" and task != "tau") else [])):
                res = []
                for fam, c0, c1 in steps:
                    r0, r1 = rate(task, system, q, c0), rate(task, system, q, c1)
                    if len(r0) == 0 or len(r1) == 0: continue
                    rt = ratio(task, cond="clean" if c1 == "clean" else ("shift" if c1 == "shift" else (c1[4:] if c1.startswith("dev:") else None)))
                    if task != "tau": rt = ratio(task)
                    s = nb(r1 - r0, rt); res.append(dict(task=task, system=system, q=q, kind=kind, family=fam, step=f"{c0} -> {c1}", rate_from=r0.mean(), rate_to=r1.mean(), diff=s["mean"], lo=s["lo"], hi=s["hi"], p=s["p"], folds_up=int(((r1 - r0) > 0).sum()), n_folds=len(r0)))
                if res:
                    h = holm([r["p"] for r in res], len(res))
                    for r, hh in zip(res, h): r["p_holm"] = float(hh); r["exception"] = bool((r["diff"] <= 0) or (hh >= .05))
                    st += res
for system in ("cheap", "mix", "single"):          # descriptive: all non-reference devices pooled against device A, outside the Holm family and the exception count
    for q in (30, 10):
        r0, r1 = rate("tau", system, q, "clean"), rate("tau", system, q, "shift")
        if len(r0) and len(r1):
            s_ = nb(r1 - r0, ratio("tau", cond="shift")); st.append(dict(task="tau", system=system, q=q, kind="descriptive_pooled", family="device:shift", step="clean -> shift", rate_from=r0.mean(), rate_to=r1.mean(), diff=s_["mean"], lo=s_["lo"], hi=s_["hi"], p=s_["p"], folds_up=int(((r1 - r0) > 0).sum()), n_folds=len(r0), p_holm=np.nan, exception=False))
ST = pd.DataFrame(st); ST.to_csv(R / "tables/t5_monotonicity_steps.csv", index=False)
RM = RF[(RF.q == 30)].groupby(["task", "system", "cell"])[["rate", "err"]].mean().reset_index(); RM.to_csv(R / "tables/t5_rate_by_cell_q30.csv", index=False)
RM10 = RF[(RF.q == 10)].groupby(["task", "system", "cell"])[["rate", "err"]].mean().reset_index(); RM10.to_csv(R / "tables/t5_rate_by_cell_q10.csv", index=False)
# ---- verdict ----
exc_primary = ST[(ST.system == "cheap") & (ST.q == 30) & (ST.kind == "primary")]
clauseB_F15, clauseB_h3 = bool(PR.pass_F15.all()), bool(PR.pass_holm3.all()); clauseC = not bool(exc_primary.exception.any()); clauseD = bool(MX.does_not_lose.all())
V = dict(clause_b_F15=clauseB_F15, clause_b_holm3=clauseB_h3, clause_c_no_exceptions=clauseC, n_exceptions_cheap_q30=int(exc_primary.exception.sum()), n_steps_cheap_q30=int(len(exc_primary)),
         exceptions_cheap_q30=exc_primary[exc_primary.exception][["task", "family", "step", "diff", "p_holm"]].round(4).to_dict("records"), clause_d_mixture_does_not_lose=clauseD,
         verdict_F15="PASS" if (clauseB_F15 and clauseC and clauseD) else "NO PASS", verdict_holm3="PASS" if (clauseB_h3 and clauseC and clauseD) else "NO PASS",
         failed_clauses_F15=[n for n, ok_ in (("b", clauseB_F15), ("c", clauseC), ("d", clauseD)) if not ok_], failed_clauses_holm3=[n for n, ok_ in (("b", clauseB_h3), ("c", clauseC), ("d", clauseD)) if not ok_])
json.dump(V, open(R / "tables/t5_verdict.json", "w"), indent=1, default=float)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(PR.round(4).to_string()); print(PM.round(4).to_string()); print(MX.round(4).to_string()); print(json.dumps(V, indent=1, default=float))
