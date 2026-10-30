"""T20 descriptive rows after the precondition P1 failed (written 2026-10-27 after tables/p1.csv was seen; no rule is tested here).
At the coordinator's request: (1) the cascade rows of amendment 1 (frozen w4g128 strong tier, fine-tuned strong tier, the paired
fine-tuned-minus-frozen cascade contrast) against the rebuilt and the old hull; (2) fine-tuned versus frozen accuracy and macro F1 on the
clips the cheap tier's margin gate flags, to compare with T17 part (c)'s +6.6 (accuracy, fold 1, FP32 twin versus T12's FP32 frozen arm);
(3) the PTQ check (ft-Q4 minus ft-FP32 on clean inner validation). Loading, pick rule and hull are those of t20_analyze.py.
    python t20_descriptive.py --run RESULTS --tasks urban esc50      -> RESULTS/tables/descriptive_after_stop/*.csv"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import t13_common as T13
from t13_common import A, S3C
from t20_common import TASKS, SEEDS, inner_split, bed, test_rows
ap = argparse.ArgumentParser(); ap.add_argument("--run", type=Path, required=True); ap.add_argument("--tasks", nargs="+", required=True); a = ap.parse_args()
OUT = a.run / "tables" / "descriptive_after_stop"; OUT.mkdir(parents=True, exist_ok=True)
BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.50]; sm = A.softmax; marg = A.margin; f1f = A.f1_fast; thr_rate = A.thr_rate; hull_pair = A.hull_pair
FROZEN = {"urban": "astshort", "esc50": "astshort"}; FROZEN_FP32 = {"urban": "astshortfp32", "esc50": "astshortfp32"}


def costs(task):
    c = json.load(open(S3C / f"analysis/{task}/costs.json")); L = TASKS[task]["L"]; t = T13.cost_table(task)
    out = dict(mn10=t["mn10"], mn20=t["mn20"], astfp32=t["astfp32"], astw4=t["astw4"], astshort=t["astshort"], astshortfp32=c["ast"][f"fp32@{L}+esc"],
               ftq4=c["ast"][f"w4g128@{L}+esc"], ftfp32=c["ast"][f"fp32@{L}+esc"])
    out["cheap"] = t["mn10"] if task == "urban" else c["cheap"]["cnn:mn20:w8a8:4"]
    return out


def load_systems(task, seed, cond):
    from analyze import ScoreDir
    S = {"cheap": T13.load(task, "mn10", cond, seed) if task == "urban" else ScoreDir(S3C / f"scores_cnn/esc50/mn20_esc50_s{seed}/4").get("mn20_w8a8", cond)}
    for e in ("mn10", "mn20", "astfp32", "astw4", "astshort"):
        if T13.available(task, e, cond, [seed]): S[e] = T13.load(task, e, cond, seed)
    d, _ = T13.sdir(task, "astshort", seed); sd = ScoreDir(d)
    if sd.has("fp32", cond): S["astshortfp32"] = sd.get("fp32", cond)
    return S


def load_ft(task, seed, cond):
    D = bed(task); N, K = D["N"], TASKS[task]["K"]; out = {}
    for cfg in ("q4", "fp32"):
        z = np.full((N, K), np.nan); iv = {}
        for f in range(1, TASKS[task]["nf"] + 1):
            r = np.load(a.run / "scores" / task / f"f{f}_s{seed}.npz"); z[r["te"]] = r[f"{cfg}_{cond}_te"]; iv[f] = (r["iv"], r[f"{cfg}_{cond}_iv"].astype(np.float64))
        out["ft" + cfg] = {"z": z, "iv": iv}
    return out


def align_iv(sysd, f, rows):
    r, z = sysd["iv"][f]; pos = {int(x): i for i, x in enumerate(r)}; return z[[pos[int(x)] for x in rows]]


def nb(d, ratio):
    d = np.asarray(d, float); k = len(d); se = np.sqrt((1 / k + ratio) * d.var(ddof=1)); m = float(d.mean())
    if se == 0: return dict(mean=m, lo=m, hi=m, p=1.0)
    h = stats.t.ppf(.975, k - 1) * se; return dict(mean=m, lo=m - h, hi=m + h, p=float(2 * stats.t.sf(abs(m / se), k - 1)))


def ratio(task, D):
    n = [(D["fold"] == f).sum() for f in range(1, TASKS[task]["nf"] + 1)]; return float(np.mean(np.array(n) / (D["N"] - np.array(n))))


def static_names(task, S):
    base = [n for n in ("cheap", "mn10", "mn20", "astshort", "astshortfp32", "astw4", "astfp32") if n in S]
    if task == "urban": base.remove("mn10")
    return base + ["ftq4", "ftfp32", "avg_ftq4", "avg_ftfp32", "avg_frozen"]


def probs(name, S, FT, rows, task, f=None, iv=False):
    def one(n):
        src = S[n] if n in S else FT[n]; return sm(align_iv(src, f, rows) if iv else src["z"][rows])
    if name.startswith("avg_"): return (one("cheap") + one({"avg_ftq4": "ftq4", "avg_ftfp32": "ftfp32", "avg_frozen": FROZEN[task]}[name])) / 2
    return one(name)


def cost_of(n, C, task):
    return C["cheap"] + C[{"avg_ftq4": "ftq4", "avg_ftfp32": "ftfp32", "avg_frozen": FROZEN[task]}[n]] if n.startswith("avg_") else C[n]


def acc(y, p): return 100 * float((y == p).mean()) if len(y) else np.nan


rows_c, rows_fl, rows_ptq, rows_pick = [], [], [], []
for task in a.tasks:
    D = bed(task); y = D["y"]; K = TASKS[task]["K"]; nf = TASKS[task]["nf"]; C = costs(task); rt = ratio(task, D); folds = range(1, nf + 1)
    clean = {s: (load_systems(task, s, "clean"), load_ft(task, s, "clean")) for s in SEEDS}; names = static_names(task, clean[0][0])
    STRONG = ("ftq4", "ftfp32", FROZEN[task], FROZEN_FP32[task])
    ivF = {n: [] for n in names}; civ = {(st, b): [] for st in STRONG for b in BUDGETS}; riv = {b: [] for b in BUDGETS}; thr = {}
    for s in SEEDS:
        S, FT = clean[s]
        for f in folds:
            iv = inner_split(task, s, f); yv = y[iv]
            for n in names: ivF[n].append(f1f(yv, probs(n, S, FT, iv, task, f, True).argmax(1), K))
            pc = probs("cheap", S, FT, iv, task, f, True); m = marg(pc)
            for b in BUDGETS:
                th = thr_rate(m, b); thr[(s, f, b)] = th; e = m < th; riv[b].append(e.mean())
                for st in STRONG: civ[(st, b)].append(f1f(yv, np.where(e, ((pc + probs(st, S, FT, iv, task, f, True)) / 2).argmax(1), pc.argmax(1)), K))
    rows_ptq.append(dict(task=task, ftq4_iv_f1=100 * np.mean(ivF["ftq4"]), ftfp32_iv_f1=100 * np.mean(ivF["ftfp32"]), ptq_loss_pts=100 * (np.mean(ivF["ftfp32"]) - np.mean(ivF["ftq4"])),
                         frozen_w4_iv_f1=100 * np.mean(ivF[FROZEN[task]]), frozen_fp32_iv_f1=100 * np.mean(ivF[FROZEN_FP32[task]])))
    spiv = pd.DataFrame([dict(system=n, f1=np.mean(ivF[n]), cost=cost_of(n, C, task)) for n in names]); best = {}
    for st in STRONG:
        for b in BUDGETS:
            c = C["cheap"] + np.mean(riv[b]) * C[st]; g = np.mean(civ[(st, b)]) - hull_pair(spiv, c)[0]
            rows_pick.append(dict(task=task, strong=st, budget=b, iv_rate=np.mean(riv[b]), iv_f1=np.mean(civ[(st, b)]), gain_iv=g))
            if st not in best or g > best[st][1] + 1e-12: best[st] = (b, g)
    for bname in ("field", "clean"):
        SS = clean if bname == "clean" else {s: (load_systems(task, s, "field"), load_ft(task, s, "field")) for s in SEEDS}
        Fs = {n: np.zeros((3, nf)) for n in names}; Fc = {st: np.zeros((3, nf)) for st in STRONG}; Rr = {st: np.zeros((3, nf)) for st in STRONG}
        PAIRS = [("ftq4", FROZEN[task]), ("ftfp32", FROZEN_FP32[task])]
        FL = {(b, sysn, met): np.full((3, nf), np.nan) for b in BUDGETS for sysn in sum(PAIRS, ()) for met in ("acc", "f1")}; NF = {b: np.zeros((3, nf)) for b in BUDGETS}
        for i, s in enumerate(SEEDS):
            S, FT = SS[s]
            for f in folds:
                r = test_rows(task, D, f); yt = y[r]; P = {n: probs(n, S, FT, r, task) for n in names}
                for n in names: Fs[n][i, f - 1] = 100 * f1f(yt, P[n].argmax(1), K)
                m = marg(P["cheap"])
                for st in STRONG:
                    e = m < thr[(s, f, best[st][0])]; Fc[st][i, f - 1] = 100 * f1f(yt, np.where(e, ((P["cheap"] + P[st]) / 2).argmax(1), P["cheap"].argmax(1)), K); Rr[st][i, f - 1] = e.mean()
                for b in BUDGETS:
                    e = m < thr[(s, f, b)]; NF[b][i, f - 1] = e.sum()
                    for sysn in sum(PAIRS, ()):
                        pr = P[sysn][e].argmax(1); FL[(b, sysn, "acc")][i, f - 1] = acc(yt[e], pr); FL[(b, sysn, "f1")][i, f - 1] = 100 * f1f(yt[e], pr, K) if e.any() else np.nan
        fmS = {n: Fs[n].mean(0) for n in names}; sp = pd.DataFrame([dict(system=n, f1=fmS[n].mean(), cost=cost_of(n, C, task)) for n in names])
        sp_old = sp[~sp.system.isin(["ftq4", "ftfp32", "avg_ftq4", "avg_ftfp32"])]
        for st in STRONG:
            fc = Fc[st].mean(0); cost = C["cheap"] + Rr[st].mean() * C[st]
            for hname, spx in (("rebuilt", sp), ("old_frozen", sp_old)):
                hv = hull_pair(spx, cost); lam = hv[3]; hf = fmS[hv[1]] if lam == 0 else (1 - lam) * fmS[hv[1]] + lam * fmS[hv[2]]; d = fc - hf; t = nb(d, rt)
                rows_c.append(dict(task=task, bed=bname, strong=st, rule="avg", budget=best[st][0], rate=Rr[st].mean(), cost_rel_ftq4=cost / C["ftq4"], cascade_f1=fc.mean(), hull=hname, hull_f1=hf.mean(),
                                   hull_lo=hv[1], hull_hi=hv[2], diff=t["mean"], nb_lo=t["lo"], nb_hi=t["hi"], folds_positive=int((d > 0).sum()), n_folds=nf, fold_diffs=np.round(d, 2).tolist()))
        for u, v in (("ftq4", FROZEN[task]), ("ftfp32", FROZEN_FP32[task]), ("ftq4", "ftfp32")):
            d = Fc[u].mean(0) - Fc[v].mean(0); t = nb(d, rt)
            rows_c.append(dict(task=task, bed=bname, strong=f"cascade {u} minus cascade {v}", rule="avg", hull="contrast", diff=t["mean"], nb_lo=t["lo"], nb_hi=t["hi"], folds_positive=int((d > 0).sum()), n_folds=nf, fold_diffs=np.round(d, 2).tolist()))
        for u, v in PAIRS:
            d = fmS[u] - fmS[v]; t = nb(d, rt)
            rows_fl.append(dict(task=task, bed=bname, clips="all", budget=np.nan, metric="f1", ft=u, frozen=v, ft_val=fmS[u].mean(), frozen_val=fmS[v].mean(), diff=t["mean"], nb_lo=t["lo"], nb_hi=t["hi"],
                                folds_positive=int((d > 0).sum()), n_folds=nf, fold1_diff=float(d[0]), fold_diffs=np.round(d, 2).tolist()))
            for b in BUDGETS:
                for met in ("acc", "f1"):
                    fu = np.nanmean(FL[(b, u, met)], 0); fv = np.nanmean(FL[(b, v, met)], 0); d = fu - fv; t = nb(d, rt)
                    rows_fl.append(dict(task=task, bed=bname, clips="flagged", budget=b, flagged_per_fold=float(NF[b].mean()), metric=met, ft=u, frozen=v, ft_val=fu.mean(), frozen_val=fv.mean(), diff=t["mean"],
                                        nb_lo=t["lo"], nb_hi=t["hi"], folds_positive=int((d > 0).sum()), n_folds=nf, fold1_diff=float(d[0]), fold_diffs=np.round(d, 2).tolist()))
for nm, rr in (("cascades", rows_c), ("flagged", rows_fl), ("ptq", rows_ptq), ("pick_iv", rows_pick)): pd.DataFrame(rr).to_csv(OUT / f"{nm}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(pd.DataFrame(rows_ptq).round(2).to_string())
print(pd.DataFrame(rows_c).drop(columns=["fold_diffs", "hull_lo", "hull_hi"]).round(2).to_string())
fl = pd.DataFrame(rows_fl); print(fl[(fl.clips == "all") | (fl.budget.isin([0.1, 0.2]))].drop(columns=["fold_diffs"]).round(2).to_string())
