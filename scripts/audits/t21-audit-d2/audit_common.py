"""Shared helpers for the D2 audit (my own code; nothing imported from the T21 worktree)."""
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import binom

SEEDS = [0, 1, 2]
FOLDS = [1, 2, 3, 4, 5]
K = 113
NOISE = ["bg__10db", "bg__0db", "pink__10db", "pink__0db", "white__10db", "white__0db"]
CONDS = ["clean", *NOISE, "field"]
BUDGETS = {"b05": 0.05, "b10": 0.10, "b20": 0.20, "b30": 0.30, "b50": 0.50}


def softmax(z):
    z = z.astype(np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def margin(p):
    top2 = np.partition(p, -2, axis=1)[:, -2:]
    return top2.max(1) - top2.min(1)


def macro_f1(y, p, k=K, rule="sk"):
    """rule 'sk': classes present in y or p (sklearn average='macro' default labels); 'y': classes present in y only."""
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    nt, npred = cm.sum(1), cm.sum(0)
    den = nt + npred
    keep = den > 0 if rule == "sk" else nt > 0
    return float(np.mean(2 * tp[keep] / np.maximum(den[keep], 1)))


def macro_f1_batch(y, P, k=K, rule="sk"):
    D = P.shape[0]
    idx = (np.arange(D)[:, None] * (k * k) + y[None, :] * k + P).ravel()
    cm = np.bincount(idx, minlength=D * k * k).reshape(D, k, k)
    tp = np.diagonal(cm, axis1=1, axis2=2).astype(float)
    nt, npred = cm.sum(2), cm.sum(1)
    den = nt + npred
    keep = den > 0 if rule == "sk" else nt > 0
    return np.where(keep, 2 * tp / np.maximum(den, 1), 0.0).sum(1) / keep.sum(1)


def thr_for_rate(m, rate):
    """Threshold t so that the share of m strictly below t is as close to rate as possible (midpoint between values)."""
    s = np.sort(m)
    n = len(s)
    j = int(round(rate * n))
    if j <= 0:
        return -np.inf
    if j >= n:
        return np.inf
    return 0.5 * (s[j - 1] + s[j])


def _pmf_sum(n1, q1, n2, q2):
    a = binom.pmf(np.arange(n1 + 1), n1, q1) if n1 else np.ones(1)
    b = binom.pmf(np.arange(n2 + 1), n2, q2) if n2 else np.ones(1)
    return np.convolve(a, b)


def exact_mix_f1_present(y, plo, phi, lam, k=K):
    """Exact E[macro F1 over the classes present in y] when each clip independently takes phi with prob. lam.
    Also returns E[S] (sum over present classes) and the distribution approximation for sklearn's rule."""
    present = np.unique(y)
    diff = plo != phi
    S = 0.0
    for c in present:
        yc = y == c
        n_c = int(yc.sum())
        same_c = (~diff) & (plo == c)
        t0, n0 = int((same_c & yc).sum()), int(same_c.sum())
        lo_c, hi_c = diff & (plo == c), diff & (phi == c)
        pu = _pmf_sum(int((lo_c & yc).sum()), 1 - lam, int((hi_c & yc).sum()), lam)
        pv = _pmf_sum(int((lo_c & ~yc).sum()), 1 - lam, int((hi_c & ~yc).sum()), lam)
        u = np.arange(len(pu))[:, None]
        v = np.arange(len(pv))[None, :]
        S += float(pu @ (2 * (t0 + u) / (n_c + n0 + u + v)) @ pv)
    # sklearn rule: absent classes that are predicted at least once enter the denominator with F1 0
    absent = np.setdiff1d(np.arange(k), present)
    probs = []
    for a in absent:
        sure = ((~diff) & (plo == a)).any()
        if sure:
            probs.append(1.0)
            continue
        n_lo = int((diff & (plo == a)).sum())
        n_hi = int((diff & (phi == a)).sum())
        p_none = (lam ** n_lo) * ((1 - lam) ** n_hi)
        probs.append(1 - p_none)
    dist = np.ones(1)
    for p_ in probs:
        dist = np.convolve(dist, [1 - p_, p_])
    e_inv = float(np.sum(dist / (len(present) + np.arange(len(dist)))))
    return S / len(present), S * e_inv


def mc_mix_f1(y, plo, phi, lam, draws=4000, seed=777, rule="sk", k=K):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws // 500):
        m = rng.random((500, len(y))) < lam
        out.append(macro_f1_batch(y, np.where(m, phi[None, :], plo[None, :]), k, rule))
    out = np.concatenate(out)
    return float(out.mean()), float(out.std(ddof=1) / np.sqrt(len(out)))


def upper_hull_pair(points, c):
    """points: dict name -> (cost, f1). Best static at cost <= c or time-sharing of (lo <= c < hi) at expected cost c."""
    best = None
    for lo, (cl, fl) in points.items():
        if cl > c * (1 + 1e-12):
            continue
        if best is None or fl > best[0]:
            best = (fl, lo, lo, 0.0)
        for hi, (ch, fh) in points.items():
            if ch <= c * (1 + 1e-12):
                continue
            lam = (c - cl) / (ch - cl)
            v = (1 - lam) * fl + lam * fh
            if v > best[0]:
                best = (v, lo, hi, lam)
    return best


def nb_test(fold_d, ratio):
    fd = np.asarray(fold_d, float)
    k = len(fd)
    se = np.sqrt((1 / k + ratio) * fd.var(ddof=1))
    t = fd.mean() / se
    p = 2 * stats.t.sf(abs(t), k - 1)
    h = stats.t.ppf(0.975, k - 1) * se
    return float(p), float(fd.mean() - h), float(fd.mean() + h)


def signflip(fold_d):
    fd = np.asarray(fold_d, float)
    k = len(fd)
    tot = 0
    for b in range(2 ** k):
        s = np.array([1 if (b >> i) & 1 else -1 for i in range(k)])
        tot += abs((s * fd).mean()) >= abs(fd.mean()) - 1e-12
    return tot / 2 ** k


class Store:
    """Lazy reader of the score npz files for one tier."""

    def __init__(self, R: Path, tier: str, design="pool"):
        self.z = {s: np.load(R / "scores" / "d2" / f"{design}_{tier}_s{s}.npz") for s in SEEDS}
        self.cache = {}

    def get(self, s, key):
        if (s, key) not in self.cache:
            self.cache[(s, key)] = self.z[s][key]
        return self.cache[(s, key)]

    def drop(self, prefix):
        for kk in [kk for kk in self.cache if kk[1].startswith(prefix)]:
            del self.cache[kk]
