"""Audit of idea 6 stage 1, checks 1, 4, 5 and 6: an independent recomputation from the per-clip
logits, for one task.

Written from scratch (no import of the idea 6 scripts). Inputs, read only:
  cheap tier   idea 6 scores/<task>/<tag>/<kind>.npz (EfficientNet heads esc, 8, 4, bitnet)
  AST tier     idea 5 Stage 2 scores/<task>/<tag>/<head>/<config>__clean.npz (heads esc and 4)
Costs are recomputed here from shapes (AST) and from the EfficientNet count of check5_costs.py.

What it does:
  * softmax, margin p1 - p2, thresholds from inner validation only (tie-free cuts; rate closest to
    the budget, ties to fewer; `match` = smallest cut whose inner F1 reaches the AST tier's inner F1,
    else the best-F1 cut), decisions `ast` and `avg`, macro F1 with sklearn.
  * cost views: i (claim: EfficientNet at 32 x 32), int8 (EfficientNet 8 x 8), ii (group and
    channel scales priced at 32 x 32 per rescale multiply-add, AST as idea 5b view ii),
    int8_ii (int8 EfficientNet plus one rescale per conv output), ii_fe (view ii plus front ends:
    mel image for EfficientNet at 44.1 kHz, the upper of the two counts, and the AST fbank).
  * hull at the design's mean cost: best static system or two-system random mix, pair picked on
    mean F1; expected F1 per (seed, fold) computed exactly (binomial) when every class is in the
    rows, otherwise by 4000 random draws.
  * selection on inner validation: gain_iv = mean over pairs of inner F1 minus the inner-validation
    hull at the design's mean inner cost. Also per-fold gains (nested) with the hull pair chosen on
    all folds' inner means (claim's way) and on the fold's own inner means (strict).
  * test comparison of every design: per-pair differences, fold means, t and Wilcoxon at fold and
    pair level.
  * static-set variants for check 6: `claim` (every cheap head, every clean AST config x 2 heads)
    and `plus_ens` (adds cheap-head ensembles, EfficientNet + AST ensembles, AST two-head ensembles).

    python recompute.py --task esc50 --out OUT
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binom
from sklearn.metrics import f1_score

H = Path.home() / "Documents"
import os; RD = Path(os.environ.get("RESULTS_DIR", str(H / "results")))  # results root (archive edit: path only)
I6 = RD / "2026-10-22_0953_redesign-idea6-stage1_code-ec70260"
S2 = RD / "2026-10-20_2359_redesign-idea5-stage2_code-3c18c38"
TASK = {"esc50": ("effesc50_s{}", "astesc50_s{}", range(5), 50, "esc"),
        "esc5": ("esc_singles_s{}", "astesc5_s{}", range(5), 5, "esc"),
        "urban": ("urban_singles_s{}", "asturban_s{}", range(3), 10, "urban")}
EFF = ["esc", "8", "4", "bitnet"]
HEADS = ["esc", "4"]
TIERS_CLAIM = ["w8g64", "w4g128", "fp32"]
TIERS_EXTRA = ["w4", "n4", "w3g64", "w4_p8"]          # exploratory tiers (per-channel 4-bit, 3-bit, uint8 P)
RULES = ["ast", "avg"]
BUDGETS = {"b05": .05, "b10": .10, "b20": .20, "b30": .30, "b50": .50}
POINTS = list(BUDGETS) + ["match"]
VIEWS = ["i", "int8", "ii", "int8_ii", "ii_fe", "i_fe32", "ii_fe32", "i_fe32s", "ii_fe32s"]

# ---- costs ------------------------------------------------------------------------------------
EFF_MACS = 1827964688.0                # check5: torch FlopCounterMode, equal to torchvision's 1.827 G + Linear
EFF_OUTPUTS = 23831832.0               # check5: conv and Linear output elements (per-channel rescales)
MEL_FE = {"esc": 73374302.0, "urban": 58733490.0}      # check5: MACs, 44.1 kHz, 5 s / 4 s
FBANK_FE = {"esc": 23726260.0, "urban": 18990460.0}
MEL_FE32 = {"esc": 6891 * 170242.0, "urban": 5513 * 170242.0}  # check5: hop 32 at 44.1 kHz (c26abba script)
# lower bound for the same front end with a sparse filterbank (each FFT bin feeds at most 2 triangles):
# window 2048 + FFT 34816 + power 2050 + mel 2 x 1025 + log 128 = 41092 MACs per frame
MEL_FE32S = {"esc": 6891 * 41092.0, "urban": 5513 * 41092.0}
L, D, FF, TOK, PATCH = 12, 768, 3072, 1214, 1212
BLOCKS = [(3 * D, D), (D, D), (FF, D), (D, FF)]
LIN_MACS = L * sum(o * i for o, i in BLOCKS) * TOK
ATT_HALF = TOK * TOK * D * L
PATCH_MACS = D * 256 * PATCH


def ast_bb(cfg, price=0.0):
    if cfg == "fp32":
        return (LIN_MACS + 2 * ATT_HALF + PATCH_MACS) * 1024.0
    base, _, suf = cfg.partition("_")
    pb = 8 if suf == "p8" else 16
    wb = int(base[1])
    g = int(base.split("g")[1]) if "g" in base else None
    resc = TOK * L * sum(o * (i // g if g else 1) for o, i in BLOCKS)
    return LIN_MACS * wb * 8 + ATT_HALF * 64 + ATT_HALF * pb * 8 + PATCH_MACS * 1024.0 + resc * price


def head_bitops(kind, in_dim, k):
    hidden, last = in_dim * 640 + 640 * 320, 320 * k
    per = {"esc": 1024, "8": 64, "4": 32, "bitnet": 4}[kind]
    return hidden * per + last * 1024.0


def costs(view, k_cls, fe_key):
    rs = view in ("ii", "int8_ii", "ii_fe", "ii_fe32", "ii_fe32s")
    eff = EFF_MACS * (64 if view.startswith("int8") else 1024)
    if view == "int8_ii":
        eff += EFF_OUTPUTS * 1024
    if view == "ii_fe":
        eff += MEL_FE[fe_key] * 1024
    if view.endswith("fe32"):
        eff += MEL_FE32[fe_key] * 1024
    if view.endswith("fe32s"):
        eff += MEL_FE32S[fe_key] * 1024
    ast_extra = FBANK_FE[fe_key] * 1024 if view in ("ii_fe", "i_fe32", "ii_fe32", "i_fe32s", "ii_fe32s") else 0.0
    cheap = {k: eff + head_bitops(k, 1536, k_cls) for k in EFF}
    ast = lambda cfg, h: ast_bb(cfg, 1024.0 if rs else 0.0) + ast_extra + head_bitops(h, 768, k_cls)
    return cheap, ast


# ---- scores -----------------------------------------------------------------------------------
def softmax(z):
    z = z.astype(np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def load(p):
    z = np.load(p)
    folds = sorted(int(f.split("_")[-1]) for f in z.files if f.startswith("iv_idx_"))
    return {"y": z["y"].astype(int), "tf": z["test_fold"].astype(int), "p": softmax(z["test_logits"]),
            "iv": {f: (z[f"iv_idx_{f}"].astype(int), softmax(z[f"iv_logits_{f}"])) for f in folds}}


def mf1(y, p):
    return float(f1_score(y, p, average="macro", zero_division=0))


def margin(p):
    s = np.sort(p, 1)
    return s[:, -1] - s[:, -2]


def cuts(m):
    ms = np.sort(m)
    n = len(ms)
    js = [0] + [j for j in range(1, n) if ms[j - 1] < ms[j]] + [n]
    ts = [-np.inf] + [(ms[j - 1] + ms[j]) / 2 for j in js[1:-1]] + [np.inf]
    return np.array(js), np.array(ts)


def f1_prefix_curve(y, pa, pb, k, order):
    """Macro F1 (sklearn rule: classes in y or predicted) when the first j clips of `order` take pb."""
    y, pa, pb = y[order], pa[order], pb[order]
    oa, ob, oy = np.eye(k)[pa], np.eye(k)[pb], np.eye(k)[y]
    z = np.zeros((1, k))
    ca, cb = np.vstack([z, np.cumsum(oa, 0)]), np.vstack([z, np.cumsum(ob, 0)])
    ta, tb = np.vstack([z, np.cumsum(oa * oy, 0)]), np.vstack([z, np.cumsum(ob * oy, 0)])
    tp = tb + (ta[-1] - ta)
    den = oy.sum(0)[None, :] + cb + (ca[-1] - ca)
    ok = den > 0
    return np.where(ok, 2 * tp / np.where(ok, den, 1), 0).sum(1) / ok.sum(1)


# ---- hull -------------------------------------------------------------------------------------
def _pmf(n, p):
    return binom.pmf(np.arange(n + 1), n, p)


GRID = np.linspace(0.0, 1.0, 201)


def exact_curve(y, plo, phi, k):
    """Exact E[macro F1] over GRID when each clip takes phi with probability lam (every class in y)."""
    out = np.zeros(len(GRID))
    agree = plo == phi
    for c in range(k):
        yc = y == c
        n_c = int(yc.sum())
        both = agree & (plo == c)
        t0, n0 = int((both & yc).sum()), int(both.sum())
        lo_only, hi_only = (plo == c) & ~agree, (phi == c) & ~agree
        a1 = int((lo_only & yc).sum()); a2 = int(lo_only.sum()) - a1
        b1 = int((hi_only & yc).sum()); b2 = int(hi_only.sum()) - b1
        pa1 = binom.pmf(np.arange(a1 + 1)[None, :], a1, 1 - GRID[:, None])
        pb1 = binom.pmf(np.arange(b1 + 1)[None, :], b1, GRID[:, None])
        pa2 = binom.pmf(np.arange(a2 + 1)[None, :], a2, 1 - GRID[:, None])
        pb2 = binom.pmf(np.arange(b2 + 1)[None, :], b2, GRID[:, None])
        pu = np.stack([np.convolve(pa1[g], pb1[g]) for g in range(len(GRID))])
        pv = np.stack([np.convolve(pa2[g], pb2[g]) for g in range(len(GRID))])
        u = np.arange(pu.shape[1])[:, None]
        v = np.arange(pv.shape[1])[None, :]
        out += np.einsum("gu,uv,gv->g", pu, 2 * (t0 + u) / (n_c + n0 + u + v), pv)
    return out / k


def mc_curve(y, plo, phi, k, seed, draws=1000):
    """Monte Carlo E[macro F1] over GRID[::4] (sklearn rule), common random numbers, then GRID."""
    g = GRID[::4]
    U = np.random.default_rng(seed).random((draws, len(y)))
    vals = []
    for lam in g:
        P = np.where(U < lam, phi[None, :], plo[None, :])
        idx = (np.arange(draws)[:, None] * k * k + y[None, :] * k + P).ravel()
        cm = np.bincount(idx, minlength=draws * k * k).reshape(draws, k, k)
        tp = np.diagonal(cm, axis1=1, axis2=2)
        den = cm.sum(1) + cm.sum(2)
        ok = den > 0
        vals.append((np.where(ok, 2 * tp / np.maximum(den, 1), 0).sum(1) / ok.sum(1)).mean())
    vals = np.array(vals)
    vals[0], vals[-1] = mf1(y, plo), mf1(y, phi)
    return np.interp(GRID, g, vals)


_CURVES = {}


def mix_f1(y, plo, phi, lam, k, key):
    """E[macro F1] of the random mix at share lam, from a cached curve per (split, lo, hi, seed, fold)."""
    if lam == 0.0:
        return mf1(y, plo)
    if key not in _CURVES:
        if len(np.unique(y)) == k:
            _CURVES[key] = exact_curve(y, plo, phi, k)
        else:
            _CURVES[key] = mc_curve(y, plo, phi, k, seed=zlib.crc32(repr(key).encode()))
    return float(np.interp(lam, GRID, _CURVES[key]))


def hull_pair(names, f1s, cs, c):
    best = None
    lo = [i for i in range(len(names)) if cs[i] <= c * (1 + 1e-12)]
    hi = [i for i in range(len(names)) if cs[i] > c * (1 + 1e-12)]
    for i in lo:
        cand = [(f1s[i], i, i, 0.0)]
        for j in hi:
            lam = (c - cs[i]) / (cs[j] - cs[i])
            cand.append(((1 - lam) * f1s[i] + lam * f1s[j], i, j, lam))
        for v in cand:
            if best is None or v[0] > best[0] + 1e-15:
                best = v
    return best


def tests(d, folds):
    d = np.asarray(d, float)
    fm = np.array([d[folds == f].mean() for f in np.unique(folds)])
    out = {"diff": float(d.mean()), "fold_diffs": json.dumps(np.round(fm, 6).tolist()),
           "n_folds": len(fm), "n_pos_folds": int((fm > 0).sum())}
    out["p_t_fold"] = float(stats.ttest_1samp(fm, 0).pvalue)
    out["p_w_fold"] = float(stats.wilcoxon(fm).pvalue) if np.any(fm != 0) else 1.0
    out["p_t_pair"] = float(stats.ttest_1samp(d, 0).pvalue)
    out["p_w_pair"] = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return out


# ---- main -------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK))
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    t0 = time.time()
    etag, atag, seeds, K, fe = TASK[a.task]
    out = a.out / a.task
    out.mkdir(parents=True, exist_ok=True)
    ast_cfgs = sorted(p.name.split("__")[0] for p in (S2 / "scores" / a.task / atag.format(0) / "esc").glob("*__clean.npz"))
    tiers = TIERS_CLAIM + [t for t in TIERS_EXTRA if t in ast_cfgs]

    # static systems and cascades per (seed, fold)
    S, SV, CA = [], [], []
    preds, preds_iv, Y, YV = {}, {}, {}, {}
    sys_kind = {}
    for s in seeds:
        ch = {k: load(I6 / "scores" / a.task / etag.format(s) / f"{k}.npz") for k in EFF}
        at = {(c, h): load(S2 / "scores" / a.task / atag.format(s) / h / f"{c}__clean.npz") for c in ast_cfgs for h in HEADS}
        ref = ch["esc"]
        for f in sorted(ref["iv"]):
            te = np.flatnonzero(ref["tf"] == f)
            ivx = ref["iv"][f][0]
            y, yv = ref["y"][te], ref["y"][ivx]
            for r in list(ch.values()) + list(at.values()):
                assert np.array_equal(r["y"], ref["y"]) and np.array_equal(r["iv"][f][0], ivx)
            Y[(s, f)], YV[(s, f)] = y, yv
            # probabilities of every static system on test and inner validation
            P = {f"eff:{k}": (ch[k]["p"][te], ch[k]["iv"][f][1]) for k in EFF}
            P.update({f"ast:{c}+{h}": (at[(c, h)]["p"][te], at[(c, h)]["iv"][f][1]) for c in ast_cfgs for h in HEADS})
            # ensembles (check 6), mean of probabilities
            ens = {"ens:esc+4": ["eff:esc", "eff:4"], "ens:esc+8+4": ["eff:esc", "eff:8", "eff:4"],
                   "ens:eff+w4g128": ["eff:esc", "ast:w4g128+esc"], "ens:eff+w3g64": ["eff:esc", "ast:w3g64+esc"],
                   "ens:w4g128x2": ["ast:w4g128+esc", "ast:w4g128+4"]}
            for nm, parts in ens.items():
                P[nm] = (np.mean([P[q][0] for q in parts], 0), np.mean([P[q][1] for q in parts], 0))
                sys_kind[nm] = parts
            for nm, (pt, pv) in P.items():
                pr, prv = pt.argmax(1), pv.argmax(1)
                preds[(nm, s, f)], preds_iv[(nm, s, f)] = pr, prv
                S.append({"system": nm, "seed": s, "fold": f, "f1": mf1(y, pr)})
                SV.append({"system": nm, "seed": s, "fold": f, "f1": mf1(yv, prv)})
            for k in EFF:
                pc, pcv = P[f"eff:{k}"]
                mt, mv = margin(pc), margin(pcv)
                ac, acv = pc.argmax(1), pcv.argmax(1)
                js, ts = cuts(mv)
                order = np.argsort(mv, kind="stable")
                for e in tiers:
                    pe, pev = P[f"ast:{e}+esc"]
                    target = mf1(yv, pev.argmax(1))
                    for rule in RULES:
                        if rule == "ast":
                            de, dev = pe.argmax(1), pev.argmax(1)
                        else:
                            de, dev = ((pc + pe) / 2).argmax(1), ((pcv + pev) / 2).argmax(1)
                        curve = None
                        for pt_name in POINTS:
                            if pt_name in BUDGETS:
                                r = js / len(mv)
                                i = int(np.lexsort((r, np.abs(r - BUDGETS[pt_name])))[0])
                            else:
                                if curve is None:
                                    curve = f1_prefix_curve(yv, acv, dev, K, order)[js]
                                ok = np.flatnonzero(curve >= target - 1e-12)
                                i = int(ok[0]) if len(ok) else int(np.lexsort((js, -curve))[0])
                            t = ts[i]
                            esc, escv = mt < t, mv < t
                            pred, predv = np.where(esc, de, ac), np.where(escv, dev, acv)
                            CA.append({"cheap": k, "tier": e, "rule": rule, "point": pt_name, "seed": s, "fold": f,
                                       "f1": mf1(y, pred), "f1_iv": mf1(yv, predv), "rate": float(esc.mean()),
                                       "iv_rate": float(escv.mean()), "thr": float(t), "n_esc": int(esc.sum()),
                                       "claim_space": e in TIERS_CLAIM})
        print(f"{a.task} seed {s} done {time.time() - t0:.0f}s", flush=True)
    S, SV, CA = pd.DataFrame(S), pd.DataFrame(SV), pd.DataFrame(CA)
    S.to_csv(out / "static.csv", index=False)
    SV.to_csv(out / "static_iv.csv", index=False)
    CA.to_csv(out / "cascades.csv", index=False)

    keys = sorted(Y)
    fold_of = np.array([k[1] for k in keys])
    designs = CA.groupby(["cheap", "tier", "rule", "point"], sort=False)
    names_all = sorted(S.system.unique())
    sets = {"claim": [n for n in names_all if not n.startswith("ens:")], "plus_ens": names_all}
    srec = {n: g.set_index(["seed", "fold"]).f1 for n, g in S.groupby("system")}
    svrec = {n: g.set_index(["seed", "fold"]).f1 for n, g in SV.groupby("system")}
    rows = []
    for view in VIEWS:
        cheap_c, ast_c = costs(view, K, fe)

        def cost_of(n):
            if n.startswith("eff:"):
                return cheap_c[n[4:]]
            if n.startswith("ast:"):
                c, h = n[4:].split("+")
                return ast_c(c, h)
            parts = sys_kind[n]
            tot = 0.0
            if any(p.startswith("eff:") for p in parts):
                tot += cheap_c["esc"] + sum(head_bitops(p[4:], 1536, K) for p in parts[1:] if p.startswith("eff:"))
            asts = [p for p in parts if p.startswith("ast:")]
            if asts:
                c0 = asts[0][4:].split("+")[0]
                tot += ast_c(c0, "esc") + sum(head_bitops(p.split("+")[1], 768, K) for p in asts[1:])
            return tot
        for setname, names in sets.items():
            if view not in ("i", "ii") and setname != "claim":
                continue
            cs = np.array([cost_of(n) for n in names])
            f_te = np.array([srec[n].mean() for n in names])
            f_iv = np.array([svrec[n].mean() for n in names])
            # per-fold inner means (strict nested hull)
            f_iv_fold = {f: np.array([svrec[n].xs(f, level="fold").mean() for n in names]) for f in np.unique(fold_of)}
            for (k, e, rule, pt_name), g in designs:
                g = g.set_index(["seed", "fold"]).loc[keys]
                ec = ast_c(e, "esc")
                c_te = cheap_c[k] + g.rate.to_numpy() * ec
                c_iv = cheap_c[k] + g.iv_rate.to_numpy() * ec
                c, civ = float(c_te.mean()), float(c_iv.mean())
                v, i, j, lam = hull_pair(names, f_te, cs, c)
                hull = np.array([mix_f1(Y[kk], preds[(names[i], *kk)], preds[(names[j], *kk)], lam, K, ("te", names[i], names[j], *kk))
                                 for kk in keys])
                vi, ii_, jj, lami = hull_pair(names, f_iv, cs, civ)
                hull_iv = np.array([mix_f1(YV[kk], preds_iv[(names[ii_], *kk)], preds_iv[(names[jj], *kk)], lami, K,
                                           ("iv", names[ii_], names[jj], *kk)) for kk in keys])
                giv = g.f1_iv.to_numpy() - hull_iv
                # strict nested: hull pair chosen on the fold's own inner means
                strict = {}
                for f in np.unique(fold_of):
                    m = fold_of == f
                    cf = float(c_iv[m].mean())
                    _, i2, j2, l2 = hull_pair(names, f_iv_fold[f], cs, cf)
                    hv = [mix_f1(YV[kk], preds_iv[(names[i2], *kk)], preds_iv[(names[j2], *kk)], l2, K, ("iv", names[i2], names[j2], *kk))
                          for kk, mm in zip(keys, m) if mm]
                    strict[int(f)] = float((g.f1_iv.to_numpy()[m] - np.array(hv)).mean())
                row = {"view": view, "set": setname, "cheap": k, "tier": e, "rule": rule, "point": pt_name,
                       "claim_space": e in TIERS_CLAIM, "f1": float(g.f1.mean()), "cost": c, "rate": float(g.rate.mean()),
                       "hull_lo": names[i], "hull_hi": names[j], "lam": lam, "hull_f1": float(hull.mean()),
                       "hull_f1_interp": v, "f1_iv": float(g.f1_iv.mean()), "cost_iv": civ,
                       "iv_rate": float(g.iv_rate.mean()), "hull_iv_lo": names[ii_], "hull_iv_hi": names[jj],
                       "lam_iv": lami, "hull_f1_iv": float(hull_iv.mean()), "gain_iv": float(giv.mean()),
                       "gain_iv_fold": json.dumps({int(f): float(giv[fold_of == f].mean()) for f in np.unique(fold_of)}),
                       "gain_iv_fold_strict": json.dumps(strict),
                       "pair_diffs": json.dumps(np.round(g.f1.to_numpy() - hull, 6).tolist())}
                row.update(tests(g.f1.to_numpy() - hull, fold_of))
                rows.append(row)
        print(f"{a.task} view {view} done {time.time() - t0:.0f}s", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(out / "designs.csv", index=False)
    stat = []
    for view in VIEWS:
        cheap_c, ast_c = costs(view, K, fe)
        for n in names_all:
            if n.startswith("ens:"):
                continue
            c = cheap_c[n[4:]] if n.startswith("eff:") else ast_c(*n[4:].split("+"))
            stat.append({"view": view, "system": n, "cost": c, "f1": srec[n].mean(), "f1_iv": svrec[n].mean()})
    for n in names_all:
        if n.startswith("ens:"):
            stat.append({"view": "any", "system": n, "cost": np.nan, "f1": srec[n].mean(), "f1_iv": svrec[n].mean()})
    pd.DataFrame(stat).to_csv(out / "static_points.csv", index=False)
    json.dump({"task": a.task, "seeds": list(seeds), "ast_configs": ast_cfgs, "tiers": tiers, "views": VIEWS,
               "seconds": time.time() - t0}, open(out / "meta.json", "w"), indent=1)
    print(f"{a.task} finished {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
