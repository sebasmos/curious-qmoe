"""T13 step 4 (ANALYSIS_PLAN_T13.md sections 3 to 9): fold-level contrasts, Holm, hull, signal tests, verdict. Reads the CSVs of t13_score.py and the pre-selection.
python t13_analyze.py --run DIR"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t13_common import *
from idea5b_common import holm
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True, type=Path); a = ap.parse_args(); R = a.run; (R / "tables").mkdir(exist_ok=True)
PRE = json.load(open(R / "inputs/t13_preselected.json")); PT = pd.read_csv(R / "inputs/preselect_table.csv")
S = pd.read_csv(R / "cell_fold_acc.csv"); G = pd.read_csv(R / "signal_fold.csv")
FA = S.groupby(["task", "cell", "system", "kind", "fold"])[["acc", "f1", "n"]].mean().reset_index()
AF = {k: g.sort_values("fold") for k, g in FA.groupby(["task", "cell", "system", "kind"])}
SG = {k: g.sort_values("fold") for k, g in G.groupby(["task", "cell", "combo"])}
rng = np.random.default_rng(0)
def arr(task, cell, system, kind="single", col="acc"):
    g = AF.get((task, cell, system, kind)); return None if g is None else g[col].to_numpy()
def ratio_of(task, cells):
    if task == "tau":
        nA = AF[("tau", "clean", "mn10", "single")]["n"].to_numpy(); nc = AF[("tau", cells[0], "mn10", "single")]["n"].to_numpy(); return float(np.mean(nc / (nA.sum() - nA)))
    n = AF[(task, cells[0], experts_of(task)[0], "single")]["n"].to_numpy(); return float(np.mean(n / (n.sum() - n)))
def nb(d, ratio):
    d = np.asarray(d, float); K = len(d); s2 = d.var(ddof=1); se = np.sqrt((1 / K + ratio) * s2)
    if se == 0: return dict(mean=float(d.mean()), lo=float(d.mean()), hi=float(d.mean()), p=1.0 if d.mean() == 0 else 0.0)
    t = d.mean() / se; h = stats.t.ppf(.975, K - 1) * se
    return dict(mean=float(d.mean()), lo=float(d.mean() - h), hi=float(d.mean() + h), p=float(2 * stats.t.sf(abs(t), K - 1)))
_hull = {}
def hull_arr(task, cell, cost, ex):
    """fold accuracies of the time-sharing hull of the single experts stored in `cell` at cost `cost` (Stage 2 hull_pair on cell mean accuracy)."""
    key = (task, cell, round(cost)); 
    if key in _hull: return _hull[key]
    av = [e for e in ex if arr(task, cell, e) is not None]; CT = cost_table(task)
    sp = pd.DataFrame([dict(system=e, f1=arr(task, cell, e).mean() / 100, cost=CT[e]) for e in av]); v, lo, hi, lam = A.hull_pair(sp, cost)
    h = arr(task, cell, lo) if lam == 0 else (1 - lam) * arr(task, cell, lo) + lam * arr(task, cell, hi)
    _hull[key] = (h, lo, hi, lam); return _hull[key]
def pool_mean(task, cells, f): return np.mean([f(c) for c in cells], 0)
MAJ = lambda K: K // 2 + 1
rows_all, prim = [], []
CT = {t: cost_table(t) for t in TASKS}
for task in TASKS:
    ex = experts_of(task); K = TASKS[task]["nf"]
    for c in combos(task):
        nm = cname(c); r = PT[(PT.task == task) & (PT.combo == nm)].iloc[0]; pool_all = r.pool_cells.split() if isinstance(r.pool_cells, str) else []
        if not pool_all: continue
        cost = float(r.cost); bs = r.best_single
        for pname, pool in (("all", pool_all), ("noise", [x for x in pool_all if x in NOISE]), ("corr", [x for x in pool_all if x in CORR])):
            if not pool: continue
            ratio = ratio_of(task, pool)
            for kind in ("mean", "geo"):
                mix = pool_mean(task, pool, lambda cl: arr(task, cl, nm, kind)); best = pool_mean(task, pool, lambda cl: arr(task, cl, bs))
                hl = pool_mean(task, pool, lambda cl: hull_arr(task, cl, cost, ex)[0]); dbest, dhull = mix - best, mix - hl
                mf, bf = pool_mean(task, pool, lambda cl: arr(task, cl, nm, kind, "f1")), pool_mean(task, pool, lambda cl: arr(task, cl, bs, "single", "f1"))
                nbb, nbh, nbf = nb(dbest, ratio), nb(dhull, ratio), nb(mf - bf, ratio)
                orc = pool_mean(task, pool, lambda cl: np.max([arr(task, cl, e) for e in ex if arr(task, cl, e) is not None and CT[task][e] <= cost * (1 + 1e-12)], 0))
                rows_all.append(dict(task=task, pool=pname, combo=nm, size=len(c), kind=kind, cost=cost, n_pool_cells=len(pool), mix_acc=mix.mean(), best_single=bs, best_single_acc=best.mean(), hull_acc=hl.mean(), oracle_best_single_acc=orc.mean(),
                                     d_best=nbb["mean"], d_best_lo=nbb["lo"], d_best_hi=nbb["hi"], p_best=nbb["p"], folds_best_ge_0_5=int((dbest >= .5).sum()), d_hull=nbh["mean"], d_hull_lo=nbh["lo"], d_hull_hi=nbh["hi"], p_hull=nbh["p"],
                                     folds_hull_ge_0_5=int((dhull >= .5).sum()), d_best_f1=nbf["mean"], p_best_f1=nbf["p"], n_folds=K, fold_d_best=np.round(dbest, 2).tolist(), fold_d_hull=np.round(dhull, 2).tolist(),
                                     clean_acc=arr(task, "clean", nm, kind).mean(), field_acc=arr(task, "field", nm, kind).mean() if arr(task, "field", nm, kind) is not None else np.nan, ratio=ratio))
ACV = pd.DataFrame(rows_all); ACV.to_csv(R / "tables/all_combos_pooled_shift.csv", index=False); AC = ACV[ACV.pool == "all"]
# ---------- every system on every cell (accuracy, F1, cost) ----------
LC = FA.groupby(["task", "cell", "system", "kind"])[["acc", "f1"]].mean().reset_index()
LC["cost"] = [sum(CT[t][e] for e in s.split("+")) for t, s in zip(LC.task, LC.system)]; LC.to_csv(R / "tables/all_systems_by_cell.csv", index=False)
# ---------- primary ----------
for task in TASKS:
    pair = PRE[task]["pair"]; ex = experts_of(task); r = AC[(AC.task == task) & (AC.combo == pair) & (AC.kind == "mean")].iloc[0]; K = r.n_folds
    d1, d2 = np.array(r.fold_d_best), np.array(r.fold_d_hull); pmax = max(r.p_best, r.p_hull)
    prim.append(dict(task=task, condition={"esc50": "ESC-50 (50 classes) shifted pooled", "esc5": "ESC-50 (5 groups) shifted pooled", "urban": "UrbanSound8K shifted pooled", "tau": "TAU shift"}[task], pair=pair, cost=r.cost, n_pool_cells=r.n_pool_cells,
                     mix_acc=r.mix_acc, best_single=r.best_single, best_single_acc=r.best_single_acc, best_single_cost=CT[task][r.best_single], hull_acc=r.hull_acc, d_best=r.d_best, d_best_ci=f"{r.d_best_lo:+.2f} to {r.d_best_hi:+.2f}", p_best=r.p_best,
                     folds_best=f"{r.folds_best_ge_0_5}/{K}", d_hull=r.d_hull, d_hull_ci=f"{r.d_hull_lo:+.2f} to {r.d_hull_hi:+.2f}", p_hull=r.p_hull, folds_hull=f"{r.folds_hull_ge_0_5}/{K}", p_max=pmax, d_best_f1=r.d_best_f1,
                     a_mean=bool(r.d_best >= .5 and r.d_hull >= .5), b_folds=bool(r.folds_best_ge_0_5 >= MAJ(K) and r.folds_hull_ge_0_5 >= MAJ(K)), geo_acc=AC[(AC.task == task) & (AC.combo == pair) & (AC.kind == "geo")].iloc[0].mix_acc))
P = pd.DataFrame(prim); P["p_holm"] = holm(P.p_max.to_numpy()); P["p_holm_x6_family"] = np.minimum(1.0, P.p_holm * 6); P["passes"] = P.a_mean & P.b_folds & (P.p_holm < .05)
P.to_csv(R / "tables/primary.csv", index=False)
# ---------- per-cell contrasts of the pre-selected pair ----------
cellrows = []
for task in TASKS:
    pair = PRE[task]["pair"]; ex = experts_of(task); bs = PRE[task]["best_single_by_combo"][pair]; cost = float(PT[(PT.task == task) & (PT.combo == pair)].cost.iloc[0]); mem = pair.split("+")
    cells = ["clean", "shift"] if task == "tau" else ["clean"] + NOISE + ["field"] + CORR
    for cl in cells:
        if arr(task, cl, pair, "mean") is None: continue
        ratio = ratio_of(task, [cl]); mix = arr(task, cl, pair, "mean"); best = arr(task, cl, bs); h, lo, hi, lam = hull_arr(task, cl, cost, ex)
        orc = np.max([arr(task, cl, e) for e in ex if arr(task, cl, e) is not None and CT[task][e] <= cost * (1 + 1e-12)], 0)
        nbb, nbh = nb(mix - best, ratio), nb(mix - h, ratio); geo = arr(task, cl, pair, "geo")
        cellrows.append(dict(task=task, cell=cl, pair=pair, mix_acc=mix.mean(), mix_f1=arr(task, cl, pair, "mean", "f1").mean(), geo_acc=geo.mean(), best_single=bs, best_single_acc=best.mean(), hull_acc=h.mean(), hull=f"{lo}|{hi}|{lam:.3f}",
                             oracle_best_single_acc=orc.mean(), **{f"{e}_acc": arr(task, cl, e).mean() for e in mem}, d_best=nbb["mean"], p_best=nbb["p"], d_hull=nbh["mean"], p_hull=nbh["p"], folds_best_ge_0_5=int(((mix - best) >= .5).sum()),
                             folds_hull_ge_0_5=int(((mix - h) >= .5).sum()), n_folds=len(mix)))
CELL = pd.DataFrame(cellrows); CELL.to_csv(R / "tables/pair_by_cell.csv", index=False)
# ---------- every combination on every cell: contrasts vs the fixed best single and the hull ----------
cc = []
for task in TASKS:
    ex = experts_of(task); cells = ["clean", "shift"] if task == "tau" else ["clean"] + NOISE + ["field"] + CORR
    for c in combos(task):
        nm = cname(c); r = PT[(PT.task == task) & (PT.combo == nm)].iloc[0]; cost = float(r.cost); bs = r.best_single
        for cl in cells:
            mix = arr(task, cl, nm, "mean"); best = arr(task, cl, bs)
            if mix is None or best is None: continue
            ratio = ratio_of(task, [cl]); h, lo, hi, lam = hull_arr(task, cl, cost, ex); nbb, nbh = nb(mix - best, ratio), nb(mix - h, ratio)
            cc.append(dict(task=task, combo=nm, cell=cl, cost=cost, mix_acc=mix.mean(), geo_acc=arr(task, cl, nm, "geo").mean(), mix_f1=arr(task, cl, nm, "mean", "f1").mean(), best_single=bs, best_single_acc=best.mean(), hull_acc=h.mean(), d_best=nbb["mean"], p_best=nbb["p"], d_hull=nbh["mean"], p_hull=nbh["p"]))
pd.DataFrame(cc).to_csv(R / "tables/all_combos_by_cell_contrasts.csv", index=False)
# ---------- stop rule S1 ----------
s1 = {}
for task in TASKS:
    pr = P[P.task == task].iloc[0]; cs = CELL[(CELL.task == task) & (CELL.cell.isin(PRE[task]["pool_cells"]))]
    s1[task] = dict(pooled_d_best=float(pr.d_best), max_cell_d_best=float(cs.d_best.max()), n_cells_ge_0_3=int((cs.d_best >= .3).sum()), n_cells=len(cs), fires=bool(pr.d_best < .3 and (cs.d_best < .3).all()))
S1 = all(v["fires"] for v in s1.values())
# ---------- signal ----------
def sg(task, cl, combo, col):
    g = SG.get((task, cl, combo)); return None if g is None else g.groupby("fold")[col].mean().to_numpy()
def sg_pool(task, pool, combo, col): return np.nanmean([sg(task, c, combo, col) for c in pool if sg(task, c, combo, col) is not None], 0)
srows, drows = [], []
for task in TASKS:
    for c in combos(task):
        nm = cname(c); r = PT[(PT.task == task) & (PT.combo == nm)].iloc[0]; pool = r.pool_cells.split() if isinstance(r.pool_cells, str) else []
        if not pool: continue
        ratio = ratio_of(task, pool); m = sg_pool(task, pool, nm, "auroc_m"); K = len(m)
        row = dict(task=task, combo=nm, size=len(c), n_pool_cells=len(pool), err_rate=sg_pool(task, pool, nm, "err_rate").mean() * 100, auroc_margin=m.mean(), auroc_js=sg_pool(task, pool, nm, "auroc_js").mean(), auroc_maxdiff=sg_pool(task, pool, nm, "auroc_md").mean())
        for vn in ("v1", "v2", "v3", "v1_transfer"):
            v = sg_pool(task, pool, nm, f"auroc_{vn}"); n_ = nb(v - m, ratio); row.update({f"auroc_{vn}": v.mean(), f"added_{vn}": n_["mean"], f"added_{vn}_p": n_["p"], f"added_{vn}_lo": n_["lo"], f"added_{vn}_hi": n_["hi"], f"added_{vn}_folds_pos": int((v - m > 0).sum())})
        row["n_folds"] = K; srows.append(row)
SIG = pd.DataFrame(srows); SIG.to_csv(R / "tables/signal_pooled_all_combos.csv", index=False)
S2 = True; sigp = {}
for task in TASKS:
    r = SIG[(SIG.task == task) & (SIG.combo == PRE[task]["pair"])].iloc[0]; adds = bool(r.added_v1 >= .02 and r.added_v1_folds_pos >= MAJ(r.n_folds)); sigp[task] = dict(added_v1=float(r.added_v1), p=float(r.added_v1_p), folds_pos=int(r.added_v1_folds_pos), adds=adds)
    S2 &= bool(r.added_v1 < .02)
# per-cell signal for the pair + humility + risk-coverage
pc = []
for task in TASKS:
    pair = PRE[task]["pair"]; cells = ["clean", "shift"] if task == "tau" else ["clean"] + NOISE + ["field"] + CORR
    for cl in cells:
        g = SG.get((task, cl, pair))
        if g is None: continue
        mm = g.drop(columns=["task", "cell", "combo"]).groupby("fold").mean().mean()
        pc.append(dict(task=task, cell=cl, **mm.to_dict()))
PC = pd.DataFrame(pc); PC.to_csv(R / "tables/signal_pair_by_cell.csv", index=False)
hum = []
for task in TASKS:
    pair = PRE[task]["pair"]; pool = PRE[task]["pool_cells"]; ratio = ratio_of(task, pool)
    for k in ("js30", "m30", "js10", "m10"):
        cl = sg(task, "clean", pair, f"flag_{k}"); sh = sg_pool(task, pool, pair, f"flag_{k}"); n_ = nb(sh - cl, ratio)
        hum.append(dict(task=task, flag=k, clean_rate=cl.mean(), shifted_pooled_rate=sh.mean(), rise=n_["mean"], lo=n_["lo"], hi=n_["hi"], p=n_["p"], folds_pos=int((sh - cl > 0).sum()), n_folds=len(cl)))
HUM = pd.DataFrame(hum); HUM.to_csv(R / "tables/humility_pooled.csv", index=False)
rc = []
for task in TASKS:
    pair = PRE[task]["pair"]; pool = PRE[task]["pool_cells"]; ratio = ratio_of(task, pool)
    mrow = {c: sg_pool(task, pool, pair, c) for c in ["err_rate", "aurc_js", "aurc_m", "aurc_v1", "acc80_js", "acc80_m", "acc80_v1", "acc90_js", "acc90_m", "acc90_v1", "selacc_js30", "selacc_m30"]}
    dj, dv = nb((mrow["aurc_js"] - mrow["aurc_m"]) * 100, ratio), nb((mrow["aurc_v1"] - mrow["aurc_m"]) * 100, ratio)
    rc.append(dict(task=task, **{k: (v.mean() * 100) for k, v in mrow.items()}, d_aurc_js_minus_m=dj["mean"], p_js=dj["p"], d_aurc_v1_minus_m=dv["mean"], p_v1=dv["p"], d_acc80_js_minus_m=nb((mrow["acc80_js"] - mrow["acc80_m"]) * 100, ratio)["mean"], d_acc90_js_minus_m=nb((mrow["acc90_js"] - mrow["acc90_m"]) * 100, ratio)["mean"]))
RC = pd.DataFrame(rc); RC.to_csv(R / "tables/risk_coverage_pooled.csv", index=False)
# ---------- reproduction check of the single experts against the stored static.csv ----------
rep = []
STAT = {"esc50": (S3C / "analysis/esc50/static.csv", {"mn10": "cnn:mn10:w8a8:esc", "mn20": "cnn:mn20:w8a8:esc", "astfp32": "ast:fp32+esc", "astw4": "ast:w4g128+esc", "astshort": "ast:w4g128@512+esc"}),
        "esc5": (S3C / "analysis/esc5/static.csv", {"mn10": "cnn:mn10:w8a8:esc", "mn20": "cnn:mn20:w8a8:esc", "astfp32": "ast:fp32+esc", "astw4": "ast:w4g128+esc", "astshort": "ast:w4g128@512+esc"}),
        "urban": (UNP / "analysis/urban/static.csv", {"mn10": "cnn:mn10unpr:w8a8:esc", "mn20": "cnn:mn20:w8a8:esc", "astfp32": "ast:fp32+esc", "astw4": "ast:w4g128+esc", "astshort": "ast:w4g128@400+esc"}),
        "tau": (OPTB / "analysis/tau/static.csv", {"mn10": "cnn:mn10:w8a8:esc", "mn20": "cnn:mn20:w8a8:esc", "astfp32": "ast:fp32+esc", "astw4": "ast:w4g128+esc"})}
for task, (p, mp) in STAT.items():
    st = pd.read_csv(p); inv = {v: k for k, v in mp.items()}; st = st[st.system.isin(inv)]; st = st.assign(expert=st.system.map(inv))
    m = S[(S.task == task) & (S.kind == "single")].merge(st, left_on=["cell", "system", "seed", "fold"], right_on=["condition", "expert", "seed", "fold"], suffixes=("", "_st"))
    for (cl, e), g in m.groupby(["cell", "system"]): rep.append(dict(task=task, cell=cl, expert=e, n=len(g), max_abs_f1_diff=float(np.abs(g.f1 / 100 - g.f1_st).max()), max_abs_acc_diff=float(np.abs(g.acc / 100 - g.acc_st).max()) if "acc_st" in g else np.nan))
REP = pd.DataFrame(rep); REP.to_csv(R / "tables/repro_check.csv", index=False)
# ---------- earlier cascade picks (macro-F1) ----------
cas = []
for task in ("esc50", "esc5", "urban"):
    sel = pd.read_csv(S3C / f"analysis/{task}/selected.csv"); sel = sel[sel.family == "P3"]
    for cl in ("clean", "field"):
        r = sel[sel.condition == cl].iloc[0]; pair = PRE[task]["pair"]
        cas.append(dict(task=task, cell=cl, pick=f"{r.cheap_src}:{r.cheap_bb}:{r.cheap}->{r.expensive} {r.rule} {r.budget}", cascade_f1=r.f1 * 100, cascade_cost=r.cost, pair=pair, pair_mixture_f1=float(arr(task, cl, pair, "mean", "f1").mean()),
                        pair_cost=float(PT[(PT.task == task) & (PT.combo == pair)].cost.iloc[0]), hull_f1_stored=r.hull_f1 * 100))
st = pd.read_csv(OPTB / "analysis/tau/selected.csv"); st = st[(st.cheap_bb == "mn20") & (st.cheap.astype(str) == "8") & (st.expensive == "fp32") & (st.rule == "avg") & (st.budget == "b50")]
for cl in ("clean", "shift"):
    r = st[st.condition == cl].iloc[0]; pair = PRE["tau"]["pair"]
    cas.append(dict(task="tau", cell=cl, pick="mn20 (FP32 backbone, head 8) -> AST FP32 avg b50", cascade_f1=r.f1 * 100, cascade_cost=r.cost, pair=pair, pair_mixture_f1=float(arr("tau", cl, pair, "mean", "f1").mean()), pair_cost=float(PT[(PT.task == "tau") & (PT.combo == pair)].cost.iloc[0]), hull_f1_stored=r.hull_f1 * 100))
pd.DataFrame(cas).to_csv(R / "tables/earlier_cascade_picks.csv", index=False)
# ---------- latency (median ms, 2 threads, measured; sequential sum) ----------
L = pd.read_csv(LAT / "tables/steps.csv"); L = L[L.threads == 2].drop_duplicates("step", keep="last") if False else L[L.threads == 2]
def lat(step, ds):
    g = L[(L.step == step) & (L.dataset.isin([ds, "both"]))]; return float(g.median_ms.iloc[-1]) if len(g) else np.nan
lrows = []
for task in ("esc50", "esc5", "urban"):
    ds = "urban8k" if task == "urban" else "esc50"; Ls = TASKS[task]["short"]; hc = "head_cheap_mn10_urban_fp32" if task == "urban" else "head_cheap_esc50_fp32"; ha = "head_ast_urban_fp32" if task == "urban" else "head_ast_esc50_fp32"
    for c in combos(task):
        nm = cname(c); cnn = [m for m in c if m.startswith("mn")]; ast = [m for m in c if m.startswith("ast")]
        out = dict(task=task, combo=nm, latency_basis="measured")
        if any(m in ("astfp32", "astw4") for m in c): out.update(latency_ms_n4=np.nan, latency_ms_fp32=np.nan, latency_basis="AST at 1024 frames not timed")
        else:
            base = (lat("mn_front_end", ds) if cnn else 0) + sum(lat(f"{m}_ort_int8", ds) + lat(hc, ds) for m in cnn) + (lat(f"fbank_{Ls}", ds) if ast else 0) + (lat(ha, ds) if ast else 0)
            out.update(latency_ms_n4=base + (lat(f"ast_{Ls}_n4_kleidiai", ds) if ast else 0), latency_ms_fp32=base + (lat(f"ast_{Ls}_fp32", ds) if ast else 0))
        lrows.append(out)
pd.DataFrame(lrows).to_csv(R / "tables/latency_combos.csv", index=False)
# ---------- verdict ----------
npass = int(P.passes.sum())
verdict = "PASS" if npass >= 1 else ("STOP" if (S1 or S2) else "NO PASS"); epistemic = "STOP (S2 fires)" if S2 else "open"
V = dict(primary=P.drop(columns=[]).to_dict("records"), n_primary_pass=npass, S1_fires=S1, S1_detail=s1, S2_fires=S2, signal_pair_pooled=sigp, verdict=verdict, epistemic_arm=epistemic, family_size=6, holm_over=4)
json.dump(V, open(R / "tables/verdict.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
print(P[["task", "pair", "cost", "n_pool_cells", "mix_acc", "best_single", "best_single_acc", "hull_acc", "d_best", "d_best_ci", "folds_best", "d_hull", "d_hull_ci", "folds_hull", "p_max", "p_holm", "passes", "geo_acc"]].to_string())
print("S1", S1, "S2", S2, "verdict", verdict, "epistemic arm", epistemic); print(json.dumps(s1, indent=1, default=float)); print(json.dumps(sigp, indent=1, default=float))
print(REP.max_abs_f1_diff.describe())
