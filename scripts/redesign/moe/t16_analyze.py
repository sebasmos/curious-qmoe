"""T16 step 4: fold-level contrasts, depth choice on held-out rows, Holm, verdict (ANALYSIS_PLAN_T16.md sections 4 to 6). Reads the CSVs of t16_score.py.
    python t16_analyze.py --score DIR --out DIR"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/redesign/idea6/stage2"))
from t16_common import BEDS, PRIMARY, EVAL, DEPTHS
from idea5b_common import holm
ap = argparse.ArgumentParser(); ap.add_argument("--score", required=True, type=Path); ap.add_argument("--out", required=True, type=Path); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
VARS = ["fp32", "w4g128"]; DEP = ["d4", "d6", "d8", "d10"]; NTR = {"urban": 7859, "esc50": 1600, "tau": 11520}
cat = lambda pat: pd.concat([pd.read_csv(f) for f in sorted(a.score.glob(pat))], ignore_index=True)
AT, AI, SY, RF = cat("auroc_test_*.csv"), cat("auroc_iv_*.csv"), cat("system_*.csv"), cat("refs_*.csv")
VARS = [v for v in VARS if v in set(AT.variant)]


def nb(d, ratio):
    d = np.asarray(d, float); d = d[np.isfinite(d)]; K = len(d); m = d.mean(); s2 = d.var(ddof=1); se = np.sqrt((1 / K + ratio) * s2)
    if se == 0: return dict(mean=m, lo=m, hi=m, p=1.0 if m == 0 else 0.0)
    t = m / se; h = stats.t.ppf(.975, K - 1) * se; return dict(mean=m, lo=m - h, hi=m + h, p=float(2 * stats.t.sf(abs(t), K - 1)))


def fold_mean(df, col, keys):
    return df.groupby(keys + ["fold"])[col].mean().reset_index()          # seeds averaged inside the fold


def ratio(bed, df):
    n = df[df.fold == 1].n.mean() if "n" in df else np.nan
    return {"urban": 1 / 9, "esc50": 1 / 4}.get(bed, None) or float(n / NTR["tau"])


# ---------------------------------------------------------------- Stage A: AUROC tables ---------------------------------------------------------------
F = fold_mean(AT, "auroc", ["variant", "bed", "cond", "kind"]); N_ = AT.groupby(["variant", "bed", "cond", "kind", "fold"]).n.mean().reset_index()
rows = []
for (v, b, c), g in F.groupby(["variant", "bed", "cond"]):
    mar = g[g.kind == "margin"].set_index("fold").auroc; nn_ = N_[(N_.variant == v) & (N_.bed == b) & (N_.cond == c) & (N_.kind == "margin")]
    r = ratio(b, nn_.assign(n=nn_.n)); K = BEDS[b]["nf"]
    for k, gk in g.groupby("kind"):
        x = gk.set_index("fold").auroc; d = (x - mar).dropna(); s = nb(d.values, r) if len(d) > 2 else dict(mean=np.nan, lo=np.nan, hi=np.nan, p=np.nan)
        rows.append(dict(variant=v, bed=b, cond=c, kind=k, auroc=x.mean(), margin_auroc=mar.mean(), gain=s["mean"], gain_lo=s["lo"], gain_hi=s["hi"], gain_p=s["p"], folds_ge_003=int((d >= .03).sum()), folds_gt0=int((d > 0).sum()), n_folds=len(d)))
AU = pd.DataFrame(rows); AU.to_csv(a.out / "auroc_all.csv", index=False)
# held-out depth choice (primary condition per bed)
FI = fold_mean(AI, "auroc", ["variant", "bed", "cond", "kind"]); DS = {}
for v in VARS:
    for b in BEDS:
        g = FI[(FI.variant == v) & (FI.bed == b) & (FI.cond == PRIMARY[b])]; mar = g[g.kind == "margin"].set_index("fold").auroc
        gains = {k: (gk.set_index("fold").auroc - mar).mean() for k, gk in g.groupby("kind") if k in DEP}; DS[(v, b)] = dict(chosen=max(gains, key=gains.get), iv_gains=gains)
pd.DataFrame([dict(variant=v, bed=b, chosen=d["chosen"], **{f"iv_gain_{k}": x for k, x in d["iv_gains"].items()}) for (v, b), d in DS.items()]).to_csv(a.out / "depth_choice.csv", index=False)
# AUROC pass and stop rule
verd = {}
for v in VARS:
    P = AU[(AU.variant == v)]; ok = 0; det = {}
    for b in BEDS:
        r = P[(P.bed == b) & (P.cond == PRIMARY[b]) & (P.kind == DS[(v, b)]["chosen"])].iloc[0]; need = BEDS[b]["nf"] // 2 + 1
        pas = bool(r.gain >= .03 and r.folds_ge_003 >= need); ok += pas; det[b] = dict(depth=DS[(v, b)]["chosen"], gain=float(r.gain), lo=float(r.gain_lo), hi=float(r.gain_hi), p=float(r.gain_p), folds_ge_003=int(r.folds_ge_003), n_folds=int(r.n_folds), margin=float(r.margin_auroc), router=float(r.auroc), pass_=pas)
    pooled = {k: float(np.mean([P[(P.bed == b) & (P.cond == PRIMARY[b]) & (P.kind == k)].gain.iloc[0] for b in BEDS])) for k in DEP}
    strict = {k: [float(P[(P.bed == b) & (P.cond == PRIMARY[b]) & (P.kind == k)].gain.iloc[0]) for b in BEDS] for k in DEP}
    verd[v] = dict(auroc_pass=ok >= 2, n_tasks_pass=ok, tasks=det, pooled_gain_by_depth=pooled, stop=all(x < .01 for x in pooled.values()), stop_strict=all(max(x) < .01 for x in strict.values()), strict_gains=strict)
# ---------------------------------------------------------------- Stage B ---------------------------------------------------------------------------
SY["key"] = SY.variant + "|" + SY.bed + "|" + SY.cond + "|" + SY.kind
chosen_rows = []
for (v, b, c, k, s, f), g in SY.groupby(["variant", "bed", "cond", "kind", "seed", "fold"]):
    chosen_rows.append(g.loc[g.iv_gain.idxmax()])                 # rate chosen on held-out rows (ties: first = lowest rate)
CH = pd.DataFrame(chosen_rows); CH.to_csv(a.out / "system_at_chosen_rate_seedfold.csv", index=False)
FC = CH.groupby(["variant", "bed", "cond", "kind", "fold"])[["acc", "cost", "esc_rate", "hull", "best_single", "hull_oracle", "f1", "rate"]].mean().reset_index()
sysrows = []
for (v, b, c, k), g in FC.groupby(["variant", "bed", "cond", "kind"]):
    g = g.sort_values("fold"); r = {"urban": 1 / 9, "esc50": 1 / 4}.get(b) or 0.15; K = len(g)
    dh, db, do = g.acc.values - g.hull.values, g.acc.values - g.best_single.values, g.acc.values - g.hull_oracle.values
    sh, sb = nb(dh, r), nb(db, r)
    sysrows.append(dict(variant=v, bed=b, cond=c, kind=k, acc=g.acc.mean(), cost=g.cost.mean(), esc_rate=g.esc_rate.mean(), chosen_rate=g.rate.mean(), hull=g.hull.mean(), best_single=g.best_single.mean(), hull_oracle=g.hull_oracle.mean(), f1=g.f1.mean(),
                        d_hull=sh["mean"], d_hull_lo=sh["lo"], d_hull_hi=sh["hi"], p_hull=sh["p"], folds_hull_ge_05=int((dh >= .5).sum()), folds_hull_ge_10=int((dh >= 1.0).sum()),
                        d_best=sb["mean"], p_best=sb["p"], folds_best_ge_05=int((db >= .5).sum()), d_oracle_hull=do.mean(), n_folds=K))
SYS = pd.DataFrame(sysrows); SYS.to_csv(a.out / "system_all.csv", index=False)
# primary tests: chosen depth, primary condition
prim = []
for v in VARS:
    for b in BEDS:
        r = SYS[(SYS.variant == v) & (SYS.bed == b) & (SYS.cond == PRIMARY[b]) & (SYS.kind == DS[(v, b)]["chosen"])].iloc[0].to_dict(); r["depth"] = DS[(v, b)]["chosen"]; prim.append(r)
PR = pd.DataFrame(prim); PR["p_holm6"] = holm(PR.p_hull.to_numpy()); PR["p_holm6_x15"] = np.minimum(1.0, PR.p_holm6 * 15)
PR["sys_pass"] = (PR.d_hull >= 1.0) & (PR.folds_hull_ge_05 >= PR.n_folds // 2 + 1) & (PR.p_holm6 < .05); PR.to_csv(a.out / "primary_system.csv", index=False)
for v in VARS:
    q = PR[PR.variant == v]; verd[v]["sys_pass"] = bool(q.sys_pass.sum() >= 2); verd[v]["n_sys_pass"] = int(q.sys_pass.sum())
    sp, ap_, st = verd[v]["sys_pass"], verd[v]["auroc_pass"], verd[v]["stop"]
    verd[v]["verdict"] = "STOP" if st else ("PASS" if (sp and ap_) else ("PARTIAL (AUROC pass, system fails)" if ap_ else ("NO PASS (system passes, AUROC does not)" if sp else "NO PASS")))
json.dump(verd, open(a.out / "verdict.json", "w"), indent=1, default=float)
# refs: fixed models and T2b arms, fold-level means
RFm = RF.groupby(["variant", "bed", "cond", "ref", "fold"])[["acc", "cost"]].mean().reset_index().groupby(["variant", "bed", "cond", "ref"])[["acc", "cost"]].mean().reset_index(); RFm.to_csv(a.out / "refs_all.csv", index=False)
print(json.dumps({v: {k: x for k, x in d.items() if k not in ("tasks", "strict_gains")} for v, d in verd.items()}, indent=1, default=float))
print(PR[["variant", "bed", "depth", "acc", "cost", "esc_rate", "hull", "d_hull", "d_hull_lo", "d_hull_hi", "p_hull", "p_holm6", "folds_hull_ge_05", "sys_pass"]].round(3).to_string())
