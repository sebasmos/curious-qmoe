"""T0 for TAU 2020 Mobile (ANALYSIS_PLAN.md Amendment 1): recompute from stored per-clip logits (option B scores) the 24 statics on clean
and shift, the headroom of the best 8-bit CNN over the best AST (BACKLOG -52: 0.6 clean, 7.3 shift with interval 6.1 to 8.5), the
pre-registered cascade pick (mn20 W8A8 head 8, AST fp32 esc, avg, b50) against the time-sharing hull, and the gate AUROC (0.77 clean,
0.69 shift). Compared to option B's static.csv, headroom.csv, selected.csv, auroc.csv and the BACKLOG numbers.
    python s6_t0_tau.py --out DIR"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "idea6/stage2"))
from common import *
from idea5b_common import mix_expected_f1

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
AN = OPTB / "analysis/tau"; K, SEEDS, FOLDS = 10, [0, 1, 2], [1, 2, 3, 4, 5]
costs = json.load(open(AN / "costs.json")); stored = pd.read_csv(AN / "static.csv")
systems = list(stored.system.unique())
def cost_of(sy): return costs["ast"][sy] if sy.startswith("ast:") else costs["cheap"][sy]
_c = {}
def raw(sy, seed):
    if (sy, seed) not in _c:
        if sy.startswith("ast:"):
            cfg, h = sy[4:].split("+"); p = OPTB / f"scores/ast_tau_s{seed}/{h}/{cfg}__all.npz"
        else:
            _, m, bb, k = sy.split(":"); p = OPTB / f"scores/{m}_tau_s{seed}/{k}/{m}_{bb}__all.npz"
        z = np.load(p)
        _c[(sy, seed)] = dict(y=z["y"].astype(np.int64), fold=z["fold"].astype(int), dev=z["device"].astype(str), z=z["test_logits"].astype(np.float64),
                              iv={int(k[10:]): z[k].astype(np.float64) for k in z.files if k.startswith("iv_logits_")})
    return _c[(sy, seed)]
ref = raw(systems[0], 0); y_all, fold_all = ref["y"], ref["fold"]
isA = np.array([d.lower() == "a" for d in ref["dev"]])
check_dev = pd.read_csv(TAU_CSV := Path.home() / "Documents/DATASETS/TAU2020Mobile/ast_768/tau2020mobile.csv", usecols=["folder", "name", "class_id"])
assert (check_dev.folder.values == fold_all).all() and (check_dev.class_id.values == y_all).all()
dev_csv = check_dev.name.str.replace(".wav", "", regex=False).str.split("-").str[-1].values
assert ((dev_csv == "a") == isA).all(), "device column of the score files and the CSV names disagree"
CM = {"clean": isA, "shift": ~isA}
def rows_of(cond, f): return np.flatnonzero((fold_all == f) & CM[cond])
F1 = lambda y, p: f1_macro(y, p, K)
RATIO = {c: float(np.mean([len(rows_of(c, f)) / (isA.sum() - (isA & (fold_all == f)).sum()) for f in FOLDS])) for c in CM}
def tests(x, z, keys, ratio):
    d = np.asarray(x) - np.asarray(z); folds = np.array([k[1] for k in keys])
    fm = np.array([d[folds == f].mean() for f in np.unique(folds)]); k = len(fm); sd = fm.std(ddof=1)
    t_nb = fm.mean() / np.sqrt((1 / k + ratio) * sd ** 2); tc = stats.t.ppf(0.975, k - 1)
    return dict(diff=float(d.mean()), fold_diffs=fm, p_t=float(stats.ttest_1samp(fm, 0.0).pvalue), p_nb=float(2 * stats.t.sf(abs(t_nb), k - 1)),
                ci_lo=float(fm.mean() - tc * fm.std(ddof=1) / np.sqrt(k)), ci_hi=float(fm.mean() + tc * fm.std(ddof=1) / np.sqrt(k)))
# 1. statics
rows = []
for sy in systems:
    for s in SEEDS:
        r = raw(sy, s); pr = r["z"].argmax(1)
        for c in CM:
            for f in FOLDS:
                i = rows_of(c, f); rows.append(dict(system=sy, condition=c, seed=s, fold=f, f1=F1(y_all[i], pr[i]), cost=cost_of(sy)))
ST = pd.DataFrame(rows); ST.to_csv(a.out / "static_recomputed.csv", index=False)
cmp = ST.merge(stored[stored.condition.isin(CM)][["system", "condition", "seed", "fold", "f1", "cost"]], on=["system", "condition", "seed", "fold"], suffixes=("", "_stored"))
check = dict(rows=len(ST), matched=len(cmp), max_abs_f1_diff=float((cmp.f1 - cmp.f1_stored).abs().max()), max_rel_cost_diff=float(((cmp.cost - cmp.cost_stored).abs() / cmp.cost_stored).max()))
print(check)
sp = {c: ST[ST.condition == c].groupby("system").agg(f1=("f1", "mean"), cost=("cost", "mean")).reset_index() for c in CM}
for c in CM:
    df = sp[c]; keep = [not ((df.cost <= r.cost) & (df.f1 >= r.f1) & ((df.cost < r.cost) | (df.f1 > r.f1))).any() for r in df.itertuples()]
    df[keep].sort_values("cost").to_csv(a.out / f"front_{c}.csv", index=False)
# 2. headroom, best w8a8 CNN vs best AST per condition (chosen on mean F1 of that condition), fold-level difference
out, head_rows = [], []
def add(q, src, exp, got, scale=1.0, nd=3):
    e, g = round(float(exp) * scale, nd), round(float(got) * scale, nd); out.append(dict(quantity=q, source=src, expected=e, recomputed=g, match3=bool(e == g)))
hs = pd.read_csv(AN / "headroom.csv")
for c in CM:
    d = sp[c]; cn = d[d.system.str.contains("w8a8")].sort_values("f1").iloc[-1]; ast = d[d.system.str.startswith("ast:")].sort_values("f1").iloc[-1]
    fa = ST[(ST.condition == c) & (ST.system == ast.system)].groupby("fold").f1.mean(); fc = ST[(ST.condition == c) & (ST.system == cn.system)].groupby("fold").f1.mean()
    t = tests((fa - fc).values, np.zeros(5), [(0, f) for f in FOLDS], RATIO[c]); h = hs[(hs.condition == c) & (hs.cheap_kind == "w8a8")].iloc[0]
    head_rows.append(dict(condition=c, best_w8a8_cnn=cn.system, cnn_f1=cn.f1, best_ast=ast.system, ast_f1=ast.f1, headroom_pts=100 * t["diff"], ci_lo_pts=100 * t["ci_lo"], ci_hi_pts=100 * t["ci_hi"]))
    add(f"headroom {c}: best w8a8 CNN F1", "option B headroom.csv", h.cheap_f1, cn.f1); add(f"headroom {c}: best AST F1", "option B headroom.csv", h.ast_f1, ast.f1)
    add(f"headroom {c}: AST minus CNN, points", "option B headroom.csv", h.headroom, t["diff"], 100, 2)
    add(f"headroom {c}: CI low, points", "option B headroom.csv", h.ci_lo, t["ci_lo"], 100, 2); add(f"headroom {c}: CI high, points", "option B headroom.csv", h.ci_hi, t["ci_hi"], 100, 2)
    add(f"static rows {c}: max abs F1 diff", "option B static.csv", 0.0, check["max_abs_f1_diff"])
pd.DataFrame(head_rows).to_csv(a.out / "headroom_recomputed.csv", index=False)
# 3. pre-registered cascade, hull, gate AUROC
CH, AS = "cnn:mn20:w8a8:8", "ast:fp32+esc"; c_ch, c_as = cost_of(CH), cost_of(AS)
def thr_rate(m_iv, rate):
    u = np.unique(m_iv); c = np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]]); rates = (m_iv[None, :] < c[:, None]).mean(1)
    return c[np.lexsort((rates, np.abs(rates - rate)))[0]]
def hull_pair(sp_, c):
    lo_set, hi_set = sp_[sp_.cost <= c * (1 + 1e-12)], sp_[sp_.cost > c * (1 + 1e-12)]; best = None
    for lo in lo_set.itertuples():
        for v in [(lo.f1, lo.system, lo.system, 0.0)] + [((1 - l) * lo.f1 + l * hi.f1, lo.system, hi.system, l) for hi in hi_set.itertuples() for l in [(c - lo.cost) / (hi.cost - lo.cost)]]:
            if best is None or v[0] > best[0] + 1e-15: best = v
    return best
sel = pd.read_csv(AN / "selected.csv"); au = pd.read_csv(AN / "auroc.csv"); au = au[(au.cheap_bb == "mn20") & (au.cheap.astype(str) == "8") & (au.expensive == "fp32")]
res = {}
for c in CM:
    keys, cf, rate, ys, pcs, des, escs = [], [], [], {}, {}, {}, {}
    wrong, score = {s: [] for s in SEEDS}, {s: [] for s in SEEDS}
    for s in SEEDS:
        rc, ra = raw(CH, s), raw(AS, s); pc_all, pa_all = softmax(rc["z"]), softmax(ra["z"])
        for f in FOLDS:
            i = rows_of(c, f); thr = thr_rate(pmargin(softmax(rc["iv"][f])), 0.50)
            m = pmargin(pc_all[i]); esc = m < thr; pc = pc_all[i].argmax(1); de = ((pc_all[i] + pa_all[i]) / 2).argmax(1)
            kk = (s, f); keys.append(kk); ys[kk] = y_all[i]; pcs[kk] = pc; des[kk] = de; escs[kk] = esc
            cf.append(F1(y_all[i], np.where(esc, de, pc))); rate.append(float(esc.mean()))
            wrong[s].append(pc != y_all[i]); score[s].append(1 - m)
    cf, rate = np.array(cf), np.array(rate); cost = c_ch + rate.mean() * c_as
    sp0 = sp[c]; s_rec = {sy: dict(zip(zip(g.seed, g.fold), g.f1)) for sy, g in ST[ST.condition == c].groupby("system")}
    v, lo, hi, lam = hull_pair(sp0, cost)
    def pred_of(sy, kk):
        r = raw(sy, kk[0]); i = rows_of(c, kk[1]); return r["z"][i].argmax(1)
    h = np.array([s_rec[lo][kk] if lam == 0.0 else float(mix_expected_f1(ys[kk], pred_of(lo, kk), pred_of(hi, kk), lam, K)[0]) for kk in keys])
    t = tests(cf, h, keys, RATIO[c]); aus = [roc_auc_score(np.concatenate(wrong[s]), np.concatenate(score[s])) for s in SEEDS]
    srow = sel[sel.condition == c].iloc[0]; arow = au[au.condition == c].auroc_wrong.mean()
    add(f"pick {c}: cascade F1", "option B selected.csv", srow.f1, cf.mean()); add(f"pick {c}: escalation rate", "option B selected.csv", srow.rate, rate.mean())
    add(f"pick {c}: cost, T", "option B selected.csv", srow.cost / 1e12, cost / 1e12); add(f"pick {c}: hull F1", "option B selected.csv", srow.hull_f1, h.mean())
    add(f"pick {c}: cascade minus hull, points", "option B selected.csv / BACKLOG -52", srow.hull_diff, t["diff"], 100, 2)
    add(f"pick {c}: fold t p", "option B selected.csv", srow.hull_p_t_fold, t["p_t"]); add(f"pick {c}: NB p", "option B selected.csv", srow.hull_p_nb, t["p_nb"])
    add(f"gate AUROC {c} (cheap wrong), seed mean", "option B auroc.csv / BACKLOG -52", arow, np.mean(aus))
    res[c] = dict(f1=cf.mean(), rate=rate.mean(), cost=cost, hull=(lo, hi, lam), hull_f1=h.mean(), diff=t["diff"], p_t=t["p_t"], p_nb=t["p_nb"], auroc=float(np.mean(aus)))
add("BACKLOG -52: headroom shift 7.3 points", "BACKLOG -52 text", 0.073, ST.system.size and head_rows[1]["headroom_pts"] / 100, 100, 1)
add("BACKLOG -52: headroom shift CI low 6.1", "BACKLOG -52 text", 0.061, head_rows[1]["ci_lo_pts"] / 100, 100, 1); add("BACKLOG -52: headroom shift CI high 8.5", "BACKLOG -52 text", 0.085, head_rows[1]["ci_hi_pts"] / 100, 100, 1)
add("BACKLOG -52: clean headroom 0.6 points", "BACKLOG -52 text", 0.006, head_rows[0]["headroom_pts"] / 100, 100, 1)
add("BACKLOG -52: pick clean minus hull +1.0 points", "BACKLOG -52 text", 0.010, res["clean"]["diff"], 100, 1); add("BACKLOG -52: pick shift minus hull -1.1 points", "BACKLOG -52 text", -0.011, res["shift"]["diff"], 100, 1)
add("BACKLOG -52: gate AUROC clean 0.77", "BACKLOG -52 text", 0.77, res["clean"]["auroc"], 1, 2); add("BACKLOG -52: gate AUROC shift 0.69", "BACKLOG -52 text", 0.69, res["shift"]["auroc"], 1, 2)
CMP = pd.DataFrame(out); CMP.to_csv(a.out / "t0_comparison.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90); print(CMP.to_string()); print(pd.DataFrame(head_rows).to_string())
write_json(a.out / "t0_summary.json", dict(static_check=check, results=res, headroom=head_rows, all_match=bool(CMP.match3.all()), n_compared=len(CMP), n_mismatch=int((~CMP.match3).sum())))
print("T0 TAU ALL MATCH" if CMP.match3.all() else f"T0 TAU MISMATCHES: {(~CMP.match3).sum()}")
