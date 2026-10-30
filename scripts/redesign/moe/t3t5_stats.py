"""Shared statistics of the T3/T5 analyses (ANALYSIS_PLAN_T3T5.md section 1 and 3): fold-level Nadeau-Bengio test, Holm with a family size, NB ratio."""
import numpy as np
from scipy import stats
from t3t5_common import bed

def nb(d, ratio):
    d = np.asarray(d, float); K = len(d); s2 = d.var(ddof=1); se = np.sqrt((1 / K + ratio) * s2)
    if se == 0: return dict(mean=float(d.mean()), lo=float(d.mean()), hi=float(d.mean()), p=1.0 if d.mean() == 0 else 0.0)
    t = d.mean() / se; h = stats.t.ppf(.975, K - 1) * se
    return dict(mean=float(d.mean()), lo=float(d.mean() - h), hi=float(d.mean() + h), p=float(2 * stats.t.sf(abs(t), K - 1)))

def holm(p, m):
    """Holm step-down with family size m >= len(p); the len(p) tests are assumed to be the smallest of the family only in the sense that the rest sits above them (multipliers m, m-1, ...)."""
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for rank, i in enumerate(o):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj

def ratio(task, n_by_fold=None, cond=None):
    """n_test / n_train of the NB correction. ESC-50, UrbanSound8K: clips of the fold over clips outside it. TAU: rows of the condition in the fold over device-A training rows outside the fold."""
    b = bed(task); fold = b["fold"]; fs = sorted(set(fold.tolist()))
    if task != "tau": return float(np.mean([(fold == f).sum() / (fold != f).sum() for f in fs]))
    dev = b["dev"]; r = []
    for f in fs:
        nA_tr = ((fold != f) & (dev == "a")).sum()
        n = ((fold == f) & (dev == "a")).sum() if cond == "clean" else (((fold == f) & (dev != "a")).sum() if cond in (None, "shift") else ((fold == f) & (dev == cond)).sum())
        r.append(n / nA_tr)
    return float(np.mean(r))
