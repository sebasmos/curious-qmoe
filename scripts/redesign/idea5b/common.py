"""Idea 5b: shared pieces (costs, score loading, macro F1, thresholds, exact expected F1 of a
random time-sharing of two static systems).

Score loading and the field mixture follow the Stage 3 audit's recompute.py (which reproduced
Stage 2's 780 primary points exactly); costs follow Stage 2's QWave/costs.py, checked against the
extraction sidecars by check_costs() below.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binom

# ---- AST shapes ------------------------------------------------------------------------------
L, D, FF, TOK, PATCH = 12, 768, 3072, 1214, 1212
BLOCKS = [(3 * D, D), (D, D), (FF, D), (D, FF)]          # (out, in) of the 4 Linear blocks per layer
LIN_W = L * sum(o * i for o, i in BLOCKS)                  # 84,934,656 weights
ROWS = L * sum(o for o, _ in BLOCKS)                       # 82,944 output rows (per-channel scales)
LIN_MACS = LIN_W * TOK
ATT_HALF = TOK * TOK * D * L                               # Q K^T MACs; P V is the same
PATCH_MACS = D * 16 * 16 * PATCH
OTHER_BITS = 40_083_456                                     # FP32 parameters outside the 72 Linear layers
RESCALE_PRICE = 32 * 32                                     # one fp32 x int32 rescale multiply-add

# configurations extracted in every condition of every dataset (cascade candidates)
CORE = ["fp32", "w8", "w6", "w5", "w4", "n4", "w8g128", "w4g128", "n4g128", "w8g64", "w3g64", "n3g64"]
BUDGETS = [0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.30]
POINTS = [f"b{int(round(100 * b)):02d}" for b in BUDGETS] + ["match"]
RULES = ["B", "avg"]                                         # B alone on escalated clips; mean of A and B probs
VIEWS = ["bitops", "rescale", "bitsread"]


def parse(cfg: str):
    """-> (weight bits, group or None, P bits, nested?) for a quantized config."""
    base, _, suf = cfg.partition("_")
    pbits = 8 if suf == "p8" else 16
    wb = int(base[1])
    g = int(base.split("g")[1]) if "g" in base else None
    return wb, g, pbits, base.startswith("n")


def rescale_macs(g) -> int:
    """fp32 rescale multiply-adds per clip: one per token, output row and weight group
    (per-channel: one group per row)."""
    return TOK * L * sum(o * (i // g if g else 1) for o, i in BLOCKS)


def bitops(cfg: str, rescale_price: float = 0.0) -> float:
    """Backbone bit-ops per clip (MACs x weight bits x activation bits). Stage 2 rule when
    rescale_price is 0; cost view (ii) charges every weight-scale rescale at rescale_price."""
    if cfg == "fp32":
        return float((LIN_MACS + 2 * ATT_HALF + PATCH_MACS) * 32 * 32)
    wb, g, pbits, _ = parse(cfg)
    b = LIN_MACS * wb * 8 + ATT_HALF * 8 * 8 + ATT_HALF * pbits * 8 + PATCH_MACS * 32 * 32
    return float(b + rescale_macs(g) * rescale_price)


def bits_read(cfg: str) -> float:
    """Weight bits read per clip by a static model (memory-bound view): Linear codes, one fp32
    scale per row or group, every other FP32 parameter. Heads are left out (same head everywhere)."""
    if cfg == "fp32":
        return float(32 * LIN_W + OTHER_BITS)
    wb, g, _, _ = parse(cfg)
    return float(wb * LIN_W + 32 * (LIN_W // g if g else ROWS) + OTHER_BITS)


def head_bitops(head: str, n_cls: int) -> float:
    macs = D * 640 + 640 * 320 + 320 * n_cls
    return float(macs * (32 * 32 if head == "esc" else 4 * 8))


def gsuffix(cfg: str) -> str:
    return cfg[cfg.index("g"):] if "g" in cfg else ""


def is_nested(a: str, b: str) -> bool:
    """A is the nested child of B: n<k>[g] -> w8[g] (one stored copy)."""
    return a.startswith("n") and b == "w8" + gsuffix(a)


def static_cost(cfg: str, head: str, n_cls: int, view: str) -> float:
    if view == "bitops":
        return bitops(cfg) + head_bitops(head, n_cls)
    if view == "rescale":
        return bitops(cfg, RESCALE_PRICE) + head_bitops(head, n_cls)
    return bits_read(cfg)


def cascade_cost(a: str, b: str, rate, n_cls: int, view: str, head: str = "esc"):
    """Mean cost per clip of A on every clip plus B on the escalated fraction `rate`
    (same head on both passes). Memory view: a nested child reads k + 1 bit-planes (rounding
    needs the next plane) and an escalated clip the remaining 8 - k - 1."""
    rate = np.asarray(rate, float)
    if view in ("bitops", "rescale"):
        return static_cost(a, head, n_cls, view) + rate * static_cost(b, head, n_cls, view)
    if is_nested(a, b):
        k = parse(a)[0]
        return bits_read(a) + LIN_W + rate * (8 - k - 1) * LIN_W
    return bits_read(a) + rate * bits_read(b)


def check_costs(emb_dir: Path) -> dict:
    """Compare bitops() with the extraction sidecars; returns {config: (mine, sidecar)}."""
    out = {}
    for j in sorted(emb_dir.glob("*__clean.json")):
        s = json.load(open(j))
        cfg = j.name.split("__")[0]
        sc = s.get("backbone_cost_per_clip", {}).get("bitops")
        if sc is not None:
            out[cfg] = (bitops(cfg), float(sc))
    return out


# ---- field mixture and score files (as in the Stage 3 audit) ---------------------------------
CELLS = [f"{k}__{s}db" for k in ("bg", "pink", "white") for s in (10, 5, 0)]


def field_assign(n: int, seed: int) -> np.ndarray:
    perm = np.random.default_rng(seed).permutation(n)
    a = np.array(["clean"] * n, dtype=object)
    for r, i in enumerate(perm[(n + 1) // 2:]):
        a[i] = CELLS[r % 9]
    return a


def softmax(z):
    z = z.astype(np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class Run:
    """Scores of one (seed, head): test-fold and inner-validation probabilities per config and condition."""

    def __init__(self, d: Path):
        self.d, self.c = d, {}

    def exists(self, cfg, cond):
        if cond == "field":
            return all((self.d / f"{cfg}__{c}.npz").exists() for c in ["clean"] + CELLS)
        return (self.d / f"{cfg}__{cond}.npz").exists()

    def raw(self, cfg, cond):
        k = (cfg, cond)
        if k not in self.c:
            z = np.load(self.d / f"{cfg}__{cond}.npz")
            folds = sorted(int(f[7:]) for f in z.files if f.startswith("iv_idx_"))
            self.c[k] = dict(y=z["y"].astype(int), tf=z["test_fold"].astype(int), p=softmax(z["test_logits"]),
                             iv={f: (z[f"iv_idx_{f}"].astype(int), softmax(z[f"iv_logits_{f}"])) for f in folds})
        return self.c[k]

    def get(self, cfg, cond):
        if cond != "field":
            return self.raw(cfg, cond)
        k = (cfg, "field")
        if k not in self.c:
            base = self.raw(cfg, "clean")
            n = len(base["y"])
            at, av = field_assign(n, 5001), field_assign(n, 5002)
            p = np.full_like(base["p"], np.nan)
            iv = {f: (idx, np.full_like(pr, np.nan)) for f, (idx, pr) in base["iv"].items()}
            for c in ["clean"] + CELLS:
                r = self.raw(cfg, c)
                p[at == c] = r["p"][at == c]
                for f, (idx, pr) in iv.items():
                    s = av[idx] == c
                    pr[s] = r["iv"][f][1][s]
            self.c[k] = dict(y=base["y"], tf=base["tf"], p=p, iv=iv)
        return self.c[k]

    def complete(self, cfg, cond):
        if not self.exists(cfg, cond):
            return False
        r = self.get(cfg, cond)
        return bool(np.isfinite(r["p"]).all() and all(np.isfinite(pr).all() for _, pr in r["iv"].values()))


# ---- metrics and thresholds -----------------------------------------------------------------
def f1_fast(y, p, k):
    """Macro F1 over classes present in y or p (sklearn's rule), from a bincount confusion matrix."""
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1)
    present = den > 0
    return float(np.mean(2 * tp[present] / den[present]))


def margin(p):
    s = np.sort(p, 1)
    return s[:, -1] - s[:, -2]


def cut_points(m_sorted):
    """Tie-free escalation counts j (escalate the j lowest margins) and their thresholds
    (escalate when margin < t)."""
    n = len(m_sorted)
    js = [0] + [j for j in range(1, n) if m_sorted[j - 1] < m_sorted[j]] + [n]
    ts = [-np.inf] + [(m_sorted[j - 1] + m_sorted[j]) / 2 for j in js[1:-1]] + [np.inf]
    return np.array(js), np.array(ts)


def pick_rate(js, n, rate):
    """Index of the cut whose rate is closest to `rate` (ties: fewer escalations)."""
    r = js / n
    return int(np.lexsort((r, np.abs(r - rate)))[0])


def f1_curve(y, pa, pb, k):
    """Macro F1 when the first j clips (in the given order) take pb and the rest pa, j = 0..n."""
    n = len(y)
    oa, ob, oy = np.eye(k)[pa], np.eye(k)[pb], np.eye(k)[y]
    ca = np.vstack([np.zeros(k), np.cumsum(oa, 0)])
    cb = np.vstack([np.zeros(k), np.cumsum(ob, 0)])
    ta = np.vstack([np.zeros(k), np.cumsum(oa * oy, 0)])
    tb = np.vstack([np.zeros(k), np.cumsum(ob * oy, 0)])
    tp = tb + (ta[-1] - ta)
    npred = cb + (ca[-1] - ca)
    den = oy.sum(0)[None, :] + npred
    present = den > 0
    f = np.where(present, 2 * tp / np.where(present, den, 1), 0.0)
    return f.sum(1) / present.sum(1)


def pick_match(js, curve, target):
    """Smallest tie-free escalation whose F1 reaches target; else the best-F1 cut (ties: fewer)."""
    f = curve[js]
    ok = np.flatnonzero(f >= target - 1e-12)
    if len(ok):
        return int(ok[0]), True
    return int(np.lexsort((js, -f))[0]), False


# ---- exact expected macro F1 of a random time-sharing ---------------------------------------
def _pmf(n, p):
    return binom.pmf(np.arange(n + 1)[None, :], n, p[:, None])


def _conv(a, b):
    out = np.zeros((a.shape[0], a.shape[1] + b.shape[1] - 1))
    for j in range(b.shape[1]):
        out[:, j:j + a.shape[1]] += a * b[:, j:j + 1]
    return out


def mix_expected_f1(y, plo, phi, lams, k):
    """Exact E[macro F1] when every clip independently takes phi with probability lam and plo
    otherwise. Every class must be present in y (true for every test fold here), so the macro
    average always runs over all k classes."""
    lams = np.atleast_1d(np.asarray(lams, float))
    out = np.zeros(len(lams))
    agree = plo == phi
    for c in range(k):
        yc = y == c
        n_c = int(yc.sum())
        assert n_c > 0
        both = agree & (plo == c)
        t0, n0 = int((both & yc).sum()), int(both.sum())
        lo_only = (plo == c) & ~agree
        hi_only = (phi == c) & ~agree
        a1 = int((lo_only & yc).sum()); a2 = int(lo_only.sum()) - a1
        b1 = int((hi_only & yc).sum()); b2 = int(hi_only.sum()) - b1
        pu = _conv(_pmf(a1, 1 - lams), _pmf(b1, lams))
        pv = _conv(_pmf(a2, 1 - lams), _pmf(b2, lams))
        u = np.arange(pu.shape[1])[:, None]
        v = np.arange(pv.shape[1])[None, :]
        w = 2 * (t0 + u) / (n_c + n0 + u + v)
        out += np.einsum("gu,gv,uv->g", pu, pv, w)
    return out / k


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj
