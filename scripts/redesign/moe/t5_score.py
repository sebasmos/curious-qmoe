"""T5 step 3 (ANALYSIS_PLAN_T3T5.md sections 4 to 6): per (task, system, seed, fold) risk-coverage areas of the margin, max-softmax, the model-free noise detector and random abstention on the shifted pool
and on clean; abstain rates at a threshold fixed on clean inner validation, per cell (noise level, corruption, TAU device). Systems: cheap (mn10), mix (pair, uniform mean), single (pre-selected best single).
Reads test labels.  python t5_score.py --out DIR [--tasks ...] [--shuffle-smoke]"""
import argparse, json, time, warnings
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t3t5_common import *
from t1c_common import TAU_CSV
from scipy.stats import rankdata
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); ap.add_argument("--tasks", nargs="+", default=T35); ap.add_argument("--shuffle-smoke", action="store_true")
ap.add_argument("--max-folds", type=int, default=0); ap.add_argument("--seeds", nargs="*", type=int); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
PRE = json.load(open(Path(__file__).resolve().parent / "t3t5_preselected.json"))
def auroc(pos, score):
    pos = np.asarray(pos).astype(bool); n1 = int(pos.sum()); n0 = len(pos) - n1
    if n1 < 2 or n0 < 2: return np.nan
    r = rankdata(score); return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def curve(err, conf):
    o = np.argsort(-conf, kind="stable"); e = err[o]; return np.cumsum(e) / np.arange(1, len(e) + 1)
def aurc(err, conf): return float(curve(err, conf).mean())
def acc_at(err, conf, c):
    o = np.argsort(-conf, kind="stable"); k = max(1, int(np.ceil(c * len(err)))); return float(1 - err[o][:k].mean())
def msp(p): return p.max(1)
CORR_CELLS = [c for v in STEPS_CORR.values() for c in v]
DEV = ["b", "c", "s1", "s2", "s3", "s4", "s5", "s6"]
loc_tau = pd.read_csv(TAU_CSV, usecols=["location"])["location"].to_numpy()
ar_rows, rate_rows = [], []
for task in a.tasks:
    T = TASKS[task]; K = T["K"]; pre = PRE[task]; single = pre["best_single"]; strong = PAIR[task]; N = bed(task)["N"]; t0 = time.time()
    pool = POOL[task]; rate_cells = ["clean"] + ([] if task == "tau" else NOISE + CORR_CELLS)
    for seed in (a.seeds if a.seeds is not None else T["seeds"]):
        L = {}                                   # (system, cell) -> logits dict
        cells_here = ["clean"] + ([] if task == "tau" else [c for c in NOISE + CORR_CELLS]) + (["shift"] if task == "tau" else [])
        for c in cells_here:
            for ex in {CHEAP, strong, single}:
                if available(task, ex, "clean" if task == "tau" else c): L[(ex, c)] = cellload(task, ex, c, seed)
        Dc0 = L[(CHEAP, "clean")]; y_all = Dc0["y"]; dev = Dc0["device"] if task == "tau" else None; fold_all = Dc0["fold"]
        if a.shuffle_smoke: yperm = np.random.default_rng(1).permutation(len(y_all))
        def probs(sysname, cell, rows=None):
            """softmax of a system in a cell (rows optional); mix = uniform mean of cheap and strong."""
            g = (lambda ex: softmax(L[(ex, cell)]["z"] if rows is None else L[(ex, cell)]["z"][rows]))
            if sysname == "cheap": return g(CHEAP)
            if sysname == "single": return g(single)
            return 0.5 * (g(CHEAP) + g(strong))
        def avail(sysname, cell):
            return {"cheap": [(CHEAP, cell)], "single": [(single, cell)], "mix": [(CHEAP, cell), (strong, cell)]}[sysname] and all(k in L for k in {"cheap": [(CHEAP, cell)], "single": [(single, cell)], "mix": [(CHEAP, cell), (strong, cell)]}[sysname])
        folds = sorted(Dc0["iv"])
        for fi, f in enumerate(folds):
            if a.max_folds and fi >= a.max_folds: break
            # noise detector fitted on the fold's inner validation (idea 7; TAU: option B shift detector)
            iv = Dc0["iv"][f][0]
            if task == "tau":
                dev_rows = np.flatnonzero((fold_all != f) & (dev != "a") & np.isin(loc_tau, list(set(loc_tau[iv]))))
                trn = np.concatenate([iv, dev_rows]); det = A.fit_lr(stats(task, "clean")[trn], (dev[trn] != "a").astype(int))
            else:
                av = A.field_arrays(N)[1]; X = np.empty((len(iv), 10))
                for c in np.unique(av[iv]): m_ = av[iv] == c; X[m_] = stats(task, c)[iv[m_]]
                det = A.fit_lr(X, (av[iv] != "clean").astype(int))
            def rows_in(cell):
                if task == "tau":
                    if cell == "clean": return np.flatnonzero((fold_all == f) & (dev == "a"))
                    if cell == "shift": return np.flatnonzero((fold_all == f) & (dev != "a"))
                    return np.flatnonzero((fold_all == f) & (dev == cell[4:]))
                return np.flatnonzero(fold_all == f)
            # ---- risk-coverage on the shifted pool (stacked) and on clean; per-cell secondary ----
            for cond, cells in (("pool", pool), ("clean", ["clean"])):
                for sysname in ("cheap", "mix", "single"):
                    cl = [c for c in cells if avail(sysname, c)]
                    if len(cl) != len(cells): continue
                    ERR, MAR, MSP, NSE, per = [], [], [], [], {}
                    for c in cl:
                        r = rows_in(c); p = probs(sysname, c, r); y = (y_all[yperm][r] if a.shuffle_smoke else y_all[r])
                        e = (p.argmax(1) != y).astype(float); mg = pmargin(p); ms = msp(p); ns = det(stats(task, c)[r])
                        ERR.append(e); MAR.append(mg); MSP.append(ms); NSE.append(ns); per[c] = (e, mg, ms, ns)
                    def emit(cd, e, mg, ms, ns):
                        r_ = dict(task=task, cond=cd, system=sysname, seed=seed, fold=f, n=len(e), err=float(e.mean()))
                        for cf, conf in (("margin", mg), ("msp", ms), ("noise_lr", -ns)):
                            ar_rows.append({**r_, "conf": cf, "aurc": aurc(e, conf), "excess": aurc(e, conf) - float(e.mean()), "acc80": acc_at(e, conf, .8), "acc90": acc_at(e, conf, .9), "auroc_err": auroc(e, -conf)})
                        ar_rows.append({**r_, "conf": "random", "aurc": float(e.mean()), "excess": 0.0, "acc80": 1 - float(e.mean()), "acc90": 1 - float(e.mean()), "auroc_err": 0.5})
                    emit(cond, *[np.concatenate(x) for x in (ERR, MAR, MSP, NSE)])
                    if cond == "pool" and sysname in ("cheap", "mix"):
                        for c in cl: emit(f"cell:{c}", *per[c])
            # ---- abstain rates at a threshold fixed on clean inner validation ----
            zc_iv, zs_iv = Dc0["iv"][f][1], L[(strong, "clean")]["iv"][f][1]; zb_iv = L[(single, "clean")]["iv"][f][1]
            miv = {"cheap": pmargin(softmax(zc_iv)), "mix": pmargin(0.5 * (softmax(zc_iv) + softmax(zs_iv))), "single": pmargin(softmax(zb_iv))}
            cl_all = rate_cells if task != "tau" else ["clean"] + [f"dev:{d}" for d in DEV] + ["shift"]
            for sysname in ("cheap", "mix", "single"):
                for q in (30, 10):
                    thr = np.quantile(miv[sysname], q / 100)
                    for c in cl_all:
                        cc = c
                        if task == "tau":
                            if not (avail(sysname, "clean")): continue
                        elif not avail(sysname, c): continue
                        r = rows_in(c); base = "clean" if task == "tau" else c; p = probs(sysname, base, r); mg = pmargin(p)
                        y = (y_all[yperm][r] if a.shuffle_smoke else y_all[r])
                        rate_rows.append(dict(task=task, system=sysname, q=q, cell=c, seed=seed, fold=f, n=len(r), rate=float((mg < thr).mean()), err=float((p.argmax(1) != y).mean())))
        print(task, "seed", seed, "%.0fs" % (time.time() - t0), flush=True)
pd.DataFrame(ar_rows).to_csv(a.out / "t5_aurc.csv", index=False); pd.DataFrame(rate_rows).to_csv(a.out / "t5_rate.csv", index=False); print("rows", len(ar_rows), len(rate_rows))
