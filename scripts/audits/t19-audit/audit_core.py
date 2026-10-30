"""T19 audit: independent loaders, F1, costs, cascade, hull and tests from the raw score files.
Written by the auditor; imports nothing from the T19 code base. CPU only."""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

Q = Path.home() / "Documents/qmoe-results"
R = Q / "2026-10-26_1409_moe-t19_code-ae334eb"
S2 = Q / "2026-10-22_1120_redesign-idea6-stage2_code-9f346ba"
S3B = Q / "2026-10-22_1652_redesign-idea6-stage3b_code-07dc1db"
S3C = Q / "2026-10-23_1227_redesign-idea6-stage3c_code-9c9cbf1"
UNP = Q / "2026-10-23_1918_redesign-idea6-stage3c-unpadded_code-cbf6be6"
DS = Path.home() / "Documents/DATASETS"
OUT = Path(os.environ.get("AUDIT_OUT", Path(__file__).resolve().parent))

TASK = {
    "esc50": dict(K=50, seeds=[0, 1, 2, 3, 4], nf=5, frames=512, fbank=4267888640.0, csv=DS / "ESC-50/ast_768/esc-50.csv",
                  label="label", rec_token=1),
    "urban": dict(K=10, seeds=[0, 1, 2], nf=10, frames=400, fbank=3112228956.408612, csv=DS / "Urban8k/ast_768/urbansound8k.csv",
                  label="class_id", rec_token=0),
}
NEW = ["w8a8", "n4a4", "n4a6", "w4a6", "w4a4", "w8a6", "w8a4", "w6a6", "w4a8", "n4a8", "mp6", "mp8"]
LEG = ["fp32", "w8g64", "w4g128", "n4", "w4", "w3g64"]
FIELD_CELLS = [(k, s) for k in ("bg", "pink", "white") for s in (10, 5, 0)]


# ── field mix (written from the stage 2 definition: ceil(n/2) clean, the rest over 9 cells in turn) ──
def field_assign(n, seed):
    order = np.random.default_rng(seed).permutation(n)
    out = np.array(["clean"] * n, dtype=object)
    for r, i in enumerate(order[(n + 1) // 2:]):
        k, s = FIELD_CELLS[r % 9]
        out[i] = f"{k}__{s}db"
    return out


_npz = {}


def npz(p: Path):
    if p not in _npz:
        z = np.load(p)
        iv = {int(k[7:]): (z[k].astype(int), z[f"iv_logits_{k[7:]}"].astype(np.float64)) for k in z.files if k.startswith("iv_idx_")}
        _npz[p] = dict(y=z["y"].astype(int), fold=z["test_fold"].astype(int), z=z["test_logits"].astype(np.float64), iv=iv,
                       crc=int(z["names_crc32"]) if "names_crc32" in z.files else None)
    return _npz[p]


def load(d: Path, stem: str, bed: str):
    """d: folder holding <stem>__<cond>.npz for one seed. bed clean or field (assembled: test seed 5001, iv seed 5002)."""
    if bed != "field":
        f = d / f"{stem}__{bed}.npz"
        return npz(f) if f.exists() else None
    base = d / f"{stem}__clean.npz"
    if not base.exists():
        return None
    b = npz(base)
    n = len(b["y"])
    at, av = field_assign(n, 5001), field_assign(n, 5002)
    z = np.full_like(b["z"], np.nan)
    iv = {f: (idx, np.full_like(v, np.nan)) for f, (idx, v) in b["iv"].items()}
    for c in ["clean"] + [f"{k}__{s}db" for k, s in FIELD_CELLS]:
        p = d / f"{stem}__{c}.npz"
        if not p.exists():
            return None
        r = npz(p)
        assert np.array_equal(r["y"], b["y"]) and np.array_equal(r["fold"], b["fold"])
        m = at == c
        z[m] = r["z"][m]
        for f, (idx, v) in iv.items():
            assert np.array_equal(r["iv"][f][0], idx)
            s = av[idx] == c
            v[s] = r["iv"][f][1][s]
    if np.isnan(z).any() or any(np.isnan(v).any() for _, v in iv.values()):
        return None
    return dict(y=b["y"], fold=b["fold"], z=z, iv=iv, crc=b["crc"])


def sys_dirs(task):
    """name -> (function seed -> folder, stem)."""
    S = {}
    sc = R / "scores" / task
    for c in NEW:
        S[f"{c}:own"] = (lambda s, c=c: sc / c / f"s{s}", c)
        if c != "w8a8":
            S[f"{c}:w8head"] = (lambda s, c=c: sc / f"{c}@w8a8" / f"s{s}", c)
    for c in LEG:
        S[f"L:{c}:own"] = (lambda s, c=c: sc / f"L_{c}" / f"s{s}", c)
        S[f"L:{c}:stored"] = (lambda s, c=c: S2 / f"scores_ast_short/{task}/astshort_{task}_s{s}/esc", c)
    if task == "urban":
        S["mn10"] = (lambda s: UNP / f"scores_unpr/urban/mn10unp_urban_s{s}/esc", "mn10_w8a8")
    else:
        S["mn10"] = (lambda s: S3B / f"scores_cnn/{task}/mn10_{task}_s{s}/esc", "mn10_w8a8")
    S["mn20"] = (lambda s: S3C / f"scores_cnn/{task}/mn20_{task}_s{s}/esc", "mn20_w8a8")
    return S


def get(task, name, seed, bed):
    fn, stem = sys_dirs(task)[name]
    return load(fn(seed), stem, bed)


# ── metric ──
def mf1(y, p, K, absent_zero=False):
    if absent_zero:
        return 100 * f1_score(y, p, labels=np.arange(K), average="macro", zero_division=0)
    return 100 * f1_score(y, p, average="macro")   # sklearn default: classes present in y or pred


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def margin(p):
    s = np.sort(p, 1)
    return s[:, -1] - s[:, -2]


# ── costs, written from scratch (AST base: 12 blocks, D 768, FF 3072, patch 16 stride 10, 128 mel bins, 2 extra tokens) ──
def ast_tokens(frames):
    return 12 * ((frames - 16) // 10 + 1) + 2


def ast_bitops(frames, wb, ab, pb, group=128, asym=True, price=1024.0, fp32=False):
    T = ast_tokens(frames)
    P = T - 2
    D, FF = 768, 3072
    lins = [(3 * D, D), (D, D), (FF, D), (D, FF)]
    lin_macs = sum(o * i for o, i in lins) * T
    att = T * T * D
    ab = [ab] * 12 if isinstance(ab, int) else ab
    lin = qk = pv = resc = zp = 0.0
    for a in ab:
        lin += lin_macs * wb * a
        qk += att * a * a
        pv += att * pb * a
        if not fp32:
            resc += T * sum(o * (i // group if group else 1) for o, i in lins) * price
            if asym:
                zp += T * sum(o for o, _ in lins) * price
    patch = D * 256 * P * 32 * 32
    unp = lin + qk + pv + patch
    return dict(unpriced=unp, priced=unp + resc + zp, linear=lin, qk=qk, pv=pv, patch=patch, rescale=resc, zp=zp)


def head_cost(K):
    return (768 * 640 + 640 * 320 + 320 * K) * 1024.0


MP = {"esc50": {"mp6": [1, 2, 3, 5, 7, 8], "mp8": [1, 2, 3, 4, 5, 6, 7, 8]},
      "urban": {"mp6": [1, 2, 3, 5, 7, 8], "mp8": [1, 2, 3, 4, 5, 7, 8, 9]}}
CFG = {"w8a8": (8, 8), "n4a4": (4, 4), "n4a6": (4, 6), "w4a6": (4, 6), "w4a4": (4, 4), "w8a6": (8, 6), "w8a4": (8, 4),
       "w6a6": (6, 6), "w4a8": (4, 8), "n4a8": (4, 8), "mp6": (4, "mp6"), "mp8": (4, "mp8")}
LEGW = {"fp32": (32, None), "w8g64": (8, 64), "w4g128": (4, 128), "n4": (4, None), "w4": (4, None), "w3g64": (3, 64)}
CNN_COST = {"esc50": {"mn10": "cnn:mn10:w8a8:esc", "mn20": "cnn:mn20:w8a8:esc"},
            "urban": {"mn10": "cnn:mn10unp:w8a8:esc", "mn20": "cnn:mn20:w8a8:esc"}}


def cost(task, name, priced=True):
    t = TASK[task]
    F = t["frames"]
    if name in ("mn10", "mn20"):
        return json.load(open(S3C / f"analysis/{task}/costs.json"))["cheap"][CNN_COST[task][name]]
    key = "priced" if priced else "unpriced"
    if name.startswith("L:"):
        c = name.split(":")[1]
        wb, g = LEGW[c]
        if wb == 32:
            bb = ast_bitops(F, 32, 32, 32, fp32=True)["unpriced"]
        else:
            bb = ast_bitops(F, wb, 8, 16, group=g, asym=False)[key]
    else:
        c = name.split(":")[0]
        wb, a = CFG[c]
        if isinstance(a, str):
            a = [4 if i in MP[task][a] else 8 for i in range(12)]
        bb = ast_bitops(F, wb, a, 8, group=128, asym=True)[key]
    return t["fbank"] + bb + head_cost(t["K"])


# ── statistics ──
def nb(d, ratio):
    d = np.asarray(d, float)
    k = len(d)
    se = np.sqrt((1 / k + ratio) * d.var(ddof=1))
    t = d.mean() / se
    h = stats.t.ppf(0.975, k - 1) * se
    return dict(mean=float(d.mean()), lo=float(d.mean() - h), hi=float(d.mean() + h), p=float(2 * stats.t.sf(abs(t), k - 1)))


def holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def hull(points, c):
    """points: list of (name, cost, mean f1). Upper concave envelope at cost c: best single at <= c or a pair lo <= c < hi."""
    best = None
    lo = [p for p in points if p[1] <= c * (1 + 1e-12)]
    hi = [p for p in points if p[1] > c * (1 + 1e-12)]
    for a in lo:
        if best is None or a[2] > best[0]:
            best = (a[2], a[0], None, 0.0)
        for b in hi:
            lam = (c - a[1]) / (b[1] - a[1])
            v = (1 - lam) * a[2] + lam * b[2]
            if v > best[0]:
                best = (v, a[0], b[0], lam)
    return best


def thr_closest(m, b):
    """Threshold so that the share of rows with margin < thr is closest to b (ties: lower share)."""
    u = np.unique(m)
    cand = np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]])
    rate = np.array([(m < c).mean() for c in cand])
    err = np.abs(rate - b)
    i = np.flatnonzero(err == err.min())
    return cand[i[np.argmin(rate[i])]]


def thr_quantile(m, b):
    return np.quantile(m, b)
