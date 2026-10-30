"""T13 step 3 (ANALYSIS_PLAN_T13.md sections 2, 3, 7): per (task, cell, seed, fold) accuracy and macro-F1 of every single expert and every 2- and 3-expert uniform mixture
(arithmetic mean of probabilities, and geometric mean as a control), and for the arithmetic mixtures the epistemic-signal quantities (AUROC of margin, JS, maxdiff and
cross-fitted logistic combinations; flag rates at a clean inner-validation threshold; risk-coverage). Reads test labels. Run after t13_preselect.py was committed.
python t13_score.py --out DIR [--tasks esc50 ...]"""
import argparse, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t13_common import *
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); ap.add_argument("--tasks", nargs="+", default=list(TASKS)); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
EPS = 1e-12
def logsm(z): z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))
def ent(p): return -(p * np.log(p + EPS)).sum(1)
def feats(Pl):
    pm = np.mean(Pl, 0); js = ent(pm) - np.mean([ent(p) for p in Pl], 0); mx = np.stack([p.max(1) for p in Pl], 1)
    return pm, np.column_stack([pmargin(pm), js, mx.max(1) - mx.min(1)])
def auroc(err, score):
    n1 = int(err.sum()); n0 = len(err) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(score); return float((r[err.astype(bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def aurc(err, conf):
    o = np.argsort(-conf, kind="stable"); e = err[o]; return float((np.cumsum(e) / np.arange(1, len(e) + 1)).mean())
def acc_at(err, conf, c):
    o = np.argsort(-conf, kind="stable"); k = max(1, int(np.ceil(c * len(err)))); return float(1 - err[o][:k].mean())
def lr_fit(X, e):
    if e.sum() == 0 or e.sum() == len(e): return None
    mu, sd = X.mean(0), X.std(0) + 1e-12; m = LogisticRegression(C=1.0, max_iter=300).fit((X - mu) / sd, e); return lambda Z: m.decision_function((Z - mu) / sd)
VAR = {"v1": [0, 1], "v2": [0, 2], "v3": [0, 1, 2]}
acc_rows, sig_rows = [], []
for task in a.tasks:
    T = TASKS[task]; ex = experts_of(task); seeds, nf, K = T["seeds"], T["nf"], T["K"]; cb = combos(task); folds = list(range(1, nf + 1))
    cells = ["clean", "shift"] if task == "tau" else ["clean"] + NOISE + ["field"] + CORR
    clean = {}                       # (seed, combo) -> clean-row features for the transfer fit
    thr = {}                         # (seed, combo, fold) -> clean inner-validation thresholds
    t0 = time.time()
    for cell in cells:
        av = [e for e in ex if available(task, e, cell if task != "tau" else "clean")]
        if not av: continue
        if task == "tau" and cell == "shift": pass
        for seed in seeds:
            D = {e: load(task, e, "clean" if task == "tau" else cell, seed) for e in av}; D0 = D[av[0]]
            y, fold = D0["y"], D0["fold"]
            mask = (D0["device"] == "a") if (task == "tau" and cell == "clean") else ((D0["device"] != "a") if task == "tau" else np.ones(len(y), bool))
            LP = {e: logsm(D[e]["z"]) for e in av}; P = {e: np.exp(LP[e]) for e in av}
            R = {f: np.flatnonzero(mask & (fold == f)) for f in folds}
            for f in folds:
                te = R[f]
                for e in av:
                    pr = P[e][te].argmax(1); acc_rows.append(dict(task=task, cell=cell, system=e, kind="single", seed=seed, fold=f, n=len(te), acc=100 * (pr == y[te]).mean(), f1=100 * A.f1_fast(y[te], pr, K)))
            for c in [c for c in cb if all(m in av for m in c)]:
                nm = cname(c); pm, F = feats([P[m] for m in c]); pg = softmax(np.mean([LP[m] for m in c], 0))
                err = (pm.argmax(1) != y).astype(float); rows = np.flatnonzero(mask); Fm, em, fm = F[rows], err[rows], fold[rows]
                if cell == "clean":
                    clean[(seed, nm)] = (Fm, em, fm)
                    for f in folds:
                        idx = D[c[0]]["iv"][f][0]; Piv = [softmax(D[m]["iv"][f][1]) for m in c]; _, Fi = feats(Piv)
                        thr[(seed, nm, f)] = dict(js30=np.quantile(Fi[:, 1], .70), m30=np.quantile(Fi[:, 0], .30), js10=np.quantile(Fi[:, 1], .90), m10=np.quantile(Fi[:, 0], .10))
                for f in folds:
                    te = R[f]; prm, prg = pm[te].argmax(1), pg[te].argmax(1)
                    acc_rows.append(dict(task=task, cell=cell, system=nm, kind="mean", seed=seed, fold=f, n=len(te), acc=100 * (prm == y[te]).mean(), f1=100 * A.f1_fast(y[te], prm, K)))
                    acc_rows.append(dict(task=task, cell=cell, system=nm, kind="geo", seed=seed, fold=f, n=len(te), acc=100 * (prg == y[te]).mean(), f1=100 * A.f1_fast(y[te], prg, K)))
                    ti, tr = fm == f, fm != f; Ft, et = Fm[ti], em[ti]
                    r = dict(task=task, cell=cell, combo=nm, seed=seed, fold=f, n=len(et), err_rate=et.mean(), auroc_m=auroc(et, -Ft[:, 0]), auroc_js=auroc(et, Ft[:, 1]), auroc_md=auroc(et, Ft[:, 2]))
                    sc = {}
                    for vn, cols in VAR.items():
                        fit = lr_fit(Fm[tr][:, cols], em[tr]); s = fit(Ft[:, cols]) if fit else None; sc[vn] = s; r[f"auroc_{vn}"] = auroc(et, s) if s is not None else np.nan
                    if cell != "clean" and (seed, nm) in clean:
                        Fc, ec, fc = clean[(seed, nm)]; fit = lr_fit(Fc[fc != f][:, VAR["v1"]], ec[fc != f]); r["auroc_v1_transfer"] = auroc(et, fit(Ft[:, VAR["v1"]])) if fit else np.nan
                    else: r["auroc_v1_transfer"] = r["auroc_v1"]
                    th = thr[(seed, nm, f)]
                    for k, (jt, mt) in {"30": (th["js30"], th["m30"]), "10": (th["js10"], th["m10"])}.items():
                        fj, fmg = Ft[:, 1] > jt, Ft[:, 0] < mt
                        r[f"flag_js{k}"] = fj.mean(); r[f"flag_m{k}"] = fmg.mean()
                        r[f"selacc_js{k}"] = 1 - et[~fj].mean() if (~fj).any() else np.nan; r[f"selacc_m{k}"] = 1 - et[~fmg].mean() if (~fmg).any() else np.nan
                    conf = {"js": -Ft[:, 1], "m": Ft[:, 0], "v1": -sc["v1"] if sc["v1"] is not None else None}
                    for kk, cf in conf.items():
                        r[f"aurc_{kk}"] = aurc(et, cf) if cf is not None else np.nan
                        for cv in (80, 90): r[f"acc{cv}_{kk}"] = acc_at(et, cf, cv / 100) if cf is not None else np.nan
                    sig_rows.append(r)
        print(task, cell, "%.0fs" % (time.time() - t0), flush=True)
pd.DataFrame(acc_rows).to_csv(a.out / "cell_fold_acc.csv", index=False); pd.DataFrame(sig_rows).to_csv(a.out / "signal_fold.csv", index=False)
print("rows", len(acc_rows), len(sig_rows))
