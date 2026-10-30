"""Auditor's own loaders, metrics, hull and tests for the idea 6 Stage 2 audit.

Nothing here imports the audited analysis code. Inputs are read only:
  R  = idea 6 Stage 2 results folder (cheap-tier and short-AST per-clip logits)
  S2 = idea 5 Stage 2 results folder (1024-frame AST per-clip logits)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

HOME = Path.home()
R = HOME / "Documents/qmoe-results/2026-10-22_1120_redesign-idea6-stage2_code-9f346ba"
S2 = HOME / "Documents/qmoe-results/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38"
OUT = HOME / "Documents/qmoe-results/2026-10-22_1623_redesign-idea6-stage2-audit_code-c892c2f/outputs"

TASKS = {
    "esc50": dict(K=50, seeds=[0, 1, 2, 3, 4], eff="effp_esc50_s{}", ast="astesc50_s{}", short="astshort_esc50_s{}",
                  SL=512, emb="esc50"),
    "esc5": dict(K=5, seeds=[0, 1, 2, 3, 4], eff="effp_esc5_s{}", ast="astesc5_s{}", short="astshort_esc5_s{}",
                 SL=512, emb="esc50"),
    "urban": dict(K=10, seeds=[0, 1, 2], eff="effp_urban_s{}", ast="asturban_s{}", short="astshort_urban_s{}",
                  SL=400, emb="urban8k"),
}
BACKBONES = ["fp32", "w8a8"]
KINDS = ["esc", "8", "4", "bitnet"]
AST_HEADS = ["esc", "4"]
FULL_TIERS = ["w4g128", "n4", "w4", "w5", "w6", "w8g64", "fp32"]
SHORT_CFGS = ["fp32", "w8g64", "w4g128", "n4", "w3g64", "w4"]
RULES = ["ast", "avg"]
BUDGETS = {"b05": 0.05, "b10": 0.10, "b20": 0.20, "b30": 0.30, "b50": 0.50}
CELLS = [f"{k}__{s}db" for k in ("bg", "pink", "white") for s in (10, 5, 0)]


# ── field mixture, written from the plan's text ────────────────────────────
def field_of(n: int, seed: int) -> np.ndarray:
    """ceil(n/2) clips clean, the rest spread over the nine cells in turn, positions from a permutation
    drawn by numpy default_rng(seed)."""
    perm = np.random.default_rng(seed).permutation(n)
    a = np.full(n, "clean", dtype=object)
    rest = perm[(n + 1) // 2:]
    a[rest] = [CELLS[r % 9] for r in range(len(rest))]
    return a


# ── score files ─────────────────────────────────────────────────────────────
def sm(z):
    z = np.asarray(z, np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class Run:
    """One (seed, head) score directory."""

    def __init__(self, d: Path):
        self.d = d
        self.cache = {}

    def has(self, cfg, cond):
        if cond == "field":
            return all((self.d / f"{cfg}__{c}.npz").exists() for c in ["clean"] + CELLS)
        return (self.d / f"{cfg}__{cond}.npz").exists()

    def raw(self, cfg, cond):
        k = (cfg, cond)
        if k not in self.cache:
            z = np.load(self.d / f"{cfg}__{cond}.npz")
            folds = sorted(int(f[7:]) for f in z.files if f.startswith("iv_idx_"))
            self.cache[k] = dict(y=z["y"].astype(np.int64), fold=z["test_fold"].astype(int),
                                 z=z["test_logits"].astype(np.float64),
                                 iv={f: (z[f"iv_idx_{f}"].astype(np.int64), z[f"iv_logits_{f}"].astype(np.float64))
                                     for f in folds},
                                 crc=int(z["names_crc32"]) if "names_crc32" in z.files else None)
        return self.cache[k]

    def get(self, cfg, cond):
        if cond != "field":
            return self.raw(cfg, cond)
        k = (cfg, "field")
        if k not in self.cache:
            b = self.raw(cfg, "clean")
            n = len(b["y"])
            at, av = field_of(n, 5001), field_of(n, 5002)
            z = np.full_like(b["z"], np.nan)
            iv = {f: (idx, np.full_like(v, np.nan)) for f, (idx, v) in b["iv"].items()}
            for c in ["clean"] + CELLS:
                r = self.raw(cfg, c)
                m = at == c
                z[m] = r["z"][m]
                for f, (idx, v) in iv.items():
                    s = av[idx] == c
                    v[s] = r["iv"][f][1][s]
            assert np.isfinite(z).all(), (self.d, cfg)
            assert all(np.isfinite(v).all() for _, v in iv.values()), (self.d, cfg)
            self.cache[k] = dict(y=b["y"], fold=b["fold"], z=z, iv=iv, crc=b["crc"])
        return self.cache[k]

    def clear(self):
        self.cache.clear()


# ── metrics ─────────────────────────────────────────────────────────────────
def f1m(y, p, K):
    """Macro F1 over labels present in y or p (sklearn's average='macro' with zero_division=0)."""
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1)
    ok = den > 0
    return float(np.mean(2 * tp[ok] / den[ok]))


def f1m_many(y, P, K):
    D = P.shape[0]
    idx = (np.arange(D)[:, None] * (K * K) + y[None, :] * K + P).ravel()
    cm = np.bincount(idx, minlength=D * K * K).reshape(D, K, K)
    tp = np.diagonal(cm, axis1=1, axis2=2).astype(float)
    den = cm.sum(1) + cm.sum(2)
    r = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
    return r.sum(1) / (den > 0).sum(1)


def margin(p):
    s = np.sort(p, 1)
    return s[:, -1] - s[:, -2]


def threshold(m, rate):
    """Escalate when margin < t. Cuts only between distinct margins; the cut whose escalation rate is
    closest to `rate`, ties to fewer escalations."""
    ms = np.sort(m)
    n = len(ms)
    js = np.concatenate([[0], np.flatnonzero(ms[1:] > ms[:-1]) + 1, [n]])
    r = js / n
    j = js[np.lexsort((r, np.abs(r - rate)))[0]]
    if j == 0:
        return -np.inf
    if j == n:
        return np.inf
    return (ms[j - 1] + ms[j]) / 2


def mix_f1_exact(y, a, b, lam, K):
    """E[macro F1] when each clip independently takes b with probability lam, else a. Every class must
    be in y. Per class: TP = fixed + Bin(n1, 1-lam) + Bin(n2, lam); false predictions likewise."""
    from scipy.stats import binom
    tot = 0.0
    for c in range(K):
        yc = y == c
        nc = int(yc.sum())
        assert nc > 0
        pa, pb = a == c, b == c
        both = pa & pb
        fixed_tp, fixed_pred = int((both & yc).sum()), int(both.sum())
        ao, bo = pa & ~pb, pb & ~pa
        n_a_tp, n_a_fp = int((ao & yc).sum()), int((ao & ~yc).sum())
        n_b_tp, n_b_fp = int((bo & yc).sum()), int((bo & ~yc).sum())
        # distributions of extra TP and extra FP (independent)
        tp_d = np.convolve(binom.pmf(np.arange(n_a_tp + 1), n_a_tp, 1 - lam), binom.pmf(np.arange(n_b_tp + 1), n_b_tp, lam))
        fp_d = np.convolve(binom.pmf(np.arange(n_a_fp + 1), n_a_fp, 1 - lam), binom.pmf(np.arange(n_b_fp + 1), n_b_fp, lam))
        u = np.arange(len(tp_d))[:, None]
        v = np.arange(len(fp_d))[None, :]
        f = 2 * (fixed_tp + u) / (nc + fixed_pred + u + v)
        tot += float(tp_d @ f @ fp_d)
    return tot / K


def mix_f1_mc(y, a, b, lam, K, U):
    """Monte Carlo E[macro F1] with common uniforms U [draws, n] (classes may be missing from y)."""
    P = np.where(U < lam, b[None, :], a[None, :])
    return float(f1m_many(y, P, K).mean())


def hull_pair(names, f1s, costs, c):
    """Best static at cost <= c, or the best two-system time-sharing (one at or below c, one above) at
    expected cost c, by linear interpolation of mean F1. Returns (value, lo, hi, lam)."""
    best = None
    lo_i = [i for i in range(len(names)) if costs[i] <= c * (1 + 1e-12)]
    hi_i = [i for i in range(len(names)) if costs[i] > c * (1 + 1e-12)]
    for i in lo_i:
        cand = [(f1s[i], names[i], names[i], 0.0)]
        for j in hi_i:
            lam = (c - costs[i]) / (costs[j] - costs[i])
            cand.append(((1 - lam) * f1s[i] + lam * f1s[j], names[i], names[j], lam))
        for v in cand:
            if best is None or v[0] > best[0] + 1e-15:
                best = v
    return best


# ── tests ───────────────────────────────────────────────────────────────────
def fold_means(d, folds):
    d = np.asarray(d, float)
    folds = np.asarray(folds)
    u = np.unique(folds)
    return np.array([d[folds == f].mean() for f in u])


def sign_flip(fm):
    k = len(fm)
    s = 1 - 2 * ((np.arange(2 ** k)[:, None] >> np.arange(k)[None, :]) & 1)
    st = np.abs((s * fm[None, :]).mean(1))
    return float(np.mean(st >= abs(fm.mean()) - 1e-12))


def fold_tests(fm, ratio):
    k = len(fm)
    t = stats.ttest_1samp(fm, 0.0)
    sd = fm.std(ddof=1)
    tnb = fm.mean() / math.sqrt((1 / k + ratio) * sd ** 2)
    return dict(mean=float(fm.mean()), p_t=float(t.pvalue), p_nb=float(2 * stats.t.sf(abs(tnb), k - 1)),
                p_sign=sign_flip(fm), fold_min=float(fm.min()), fold_max=float(fm.max()))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    out = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i]))
        out[i] = run
    return out


def dump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=float))
