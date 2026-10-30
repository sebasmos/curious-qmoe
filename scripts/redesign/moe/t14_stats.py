"""T14 statistics (ANALYSIS_PLAN_T14.md sections 3, 4, 6). Input: arrays X[seed, fold] (sampling) and F[cell, fold] (shift, seed-averaged)."""
import numpy as np
from scipy import stats
B, SEED = 10000, 20261025
def fold_mean(X): return np.asarray(X, float).mean(0)           # X [S, nf]
def sd_fold(X): return float(np.std(fold_mean(X), ddof=1))
def sd_seed(X): X = np.asarray(X, float); return float(np.std(X.mean(1), ddof=1)) if X.shape[0] > 1 else np.nan
def sd_inter(X):
    X = np.asarray(X, float); S, nf = X.shape
    if S < 2 or nf < 2: return np.nan
    r = X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + X.mean()
    return float(np.sqrt((r ** 2).sum() / ((S - 1) * (nf - 1))))
def _var_ratio(a, b): return np.var(a, ddof=1) / np.var(b, ddof=1) if np.var(b, ddof=1) > 0 else np.nan
def boot_ratio(Xa, Xb, kind="fold", seed=SEED):
    """ratio of variances a over b with a paired percentile bootstrap (folds, or seeds); kind in fold, seed, inter."""
    Xa, Xb = np.asarray(Xa, float), np.asarray(Xb, float); rng = np.random.default_rng(seed)
    S, nf = Xa.shape
    def stat(A_, B_):
        if kind == "fold": return _var_ratio(fold_mean(A_), fold_mean(B_))
        if kind == "seed": return _var_ratio(A_.mean(1), B_.mean(1))
        return (sd_inter(A_) / sd_inter(B_)) ** 2 if sd_inter(B_) > 0 else np.nan
    pt = stat(Xa, Xb); out = []
    for _ in range(B):
        if kind == "seed": idx = rng.integers(0, S, S); out.append(stat(Xa[idx], Xb[idx]))
        else: idx = rng.integers(0, nf, nf); out.append(stat(Xa[:, idx], Xb[:, idx]))
    out = np.array(out, float); out = out[np.isfinite(out)]
    return pt, (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if len(out) else (np.nan, np.nan)
def bf_p(Xa, Xb):
    a, b = fold_mean(Xa), fold_mean(Xb)
    return float(stats.levene(a, b, center="median").pvalue)
def pitman_morgan_p(Xa, Xb):
    a, b = fold_mean(Xa), fold_mean(Xb)
    if len(a) < 4 or np.std(a + b) == 0 or np.std(a - b) == 0: return np.nan
    r = np.corrcoef(a + b, a - b)[0, 1]; n = len(a); t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return float(2 * stats.t.sf(abs(t), n - 2))
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o): run = max(run, (m - i) * p[j]); adj[j] = min(1.0, run)
    return adj
def shift_measures(F, cells, clean="clean"):
    """F [ncell, nf] seed-averaged F1. Returns dict of per-fold arrays: worst, sd_cells, drop, slope (additive SNR cells)."""
    F = np.asarray(F, float); ci = {c: i for i, c in enumerate(cells)}; ic = ci[clean]; oth = [i for c, i in ci.items() if c != clean]
    out = dict(worst=F.min(0), sd_cells=F.std(0, ddof=1), drop=F[ic] - F[oth].mean(0)) if oth else dict(worst=F.min(0))
    sl = []
    for k in ("bg", "pink", "white"):
        names = [f"{k}__{s}db" for s in (0, 5, 10, 20)]
        if all(n in ci for n in names):
            x = np.array([0, 5, 10, 20.0]); sl.append(np.array([np.polyfit(x, F[[ci[n] for n in names], f], 1)[0] for f in range(F.shape[1])]))
    out["slope_per_db"] = np.mean(sl, 0) if sl else np.full(F.shape[1], np.nan)
    return out
def paired(d, nb_ratio=None):
    d = np.asarray(d, float); n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    if d.std(ddof=1) == 0: return dict(mean=m, lo=m, hi=m, p=1.0 if m == 0 else 0.0, p_nb=np.nan, n_ge05=int((d >= 0.5).sum()), n=n)
    t = stats.t.ppf(0.975, n - 1); p = 2 * stats.t.sf(abs(m / se), n - 1)
    pnb = np.nan
    if nb_ratio is not None:
        se2 = d.std(ddof=1) * np.sqrt(1 / n + nb_ratio); pnb = 2 * stats.t.sf(abs(m / se2), n - 1)
    return dict(mean=float(m), lo=float(m - t * se), hi=float(m + t * se), p=float(p), p_nb=float(pnb), n_ge05=int((d >= 0.5).sum()), n=n)
