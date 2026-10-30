"""T26 core (ANALYSIS_PLAN.md C1 to C4, C8): the two systems, their thresholds, the budget choice on clean inner
validation, the widened equal-cost hull at each system's own cost, and the fold-level tests. Every dataset adapter
(Part A, B, C) hands this module the same structure (a Bed), so the three parts run one code path.

S1: MN10-Q8, then MN20-Q8 on the clips whose MN10 margin is below the threshold; escalated clips take the mean of the
    two softmaxes (T21's average rule with MN20-Q8 in place of AST-Q4).
S2: MN10-Q8; tier 2 MN20-Q8 on rate r2 of clips (MN10 margin), output the pair mean; tier 3 AST-Q4 on the tier-2 clips
    with the lowest margin of the pair mean so that r3 of all clips reach it (capped at the tier-2 clips), output the
    mean of the three softmaxes (T23's thr_b and system_b with MN20-Q8 as the middle tier).
Hull: T25's 17 systems (MN10-Q8, AST-Q4, always-average of the two, MN04/MN05/MN20/DyMN04-Q8, the ten two-CNN
    averages), stage 2's hull_pair over mean test F1 and mean priced cost at the system's own mean cost, the hull's
    per-(seed, fold) F1 the exact expected macro F1 of random time-sharing (T21's exp_f1).
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "t21"))
import t21_analyze as T  # noqa: E402  stage 2 helpers (T.A), exp_f1, nb_ci, f1_rows, holm

A, holm = T.A, T.holm
CNN5 = ["mn04", "mn05", "mn10", "mn20", "dymn04"]
PAIRS = [f"{a}+{b}" for a, b in itertools.combinations(CNN5, 2)]
WIDE = ["mn10", "ast", "avg", "mn04", "mn05", "mn20", "dymn04", *PAIRS]
BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.50]
S2_GRID = [(r2, r3) for r2 in BUDGETS for r3 in BUDGETS]
FAMILY = 22


# ── systems ──
def s1_pred(p10, p20, t):
    esc = A.margin(p10) < t
    return np.where(esc, ((p10 + p20) / 2).argmax(1), p10.argmax(1)), esc


def s2_thr(p10, p20, r2, r3):
    m1 = A.margin(p10)
    t2 = A.thr_rate(m1, r2)
    e2 = m1 < t2
    if e2.sum() == 0:
        return t2, -np.inf
    t3 = A.thr_rate(A.margin((p10 + p20) / 2)[e2], min(1.0, r3 * len(p10) / e2.sum()))
    return t2, t3


def s2_pred(p10, p20, pa, t2, t3):
    e2 = A.margin(p10) < t2
    q = (p10 + p20) / 2
    e3 = e2 & (A.margin(q) < t3)
    return np.where(e3, ((p10 + p20 + pa) / 3).argmax(1), np.where(e2, q.argmax(1), p10.argmax(1))), e2, e3


def cascade_pred(p10, pa, t):
    esc = A.margin(p10) < t
    return np.where(esc, ((p10 + pa) / 2).argmax(1), p10.argmax(1)), esc


def statics(P, c):
    """Predictions and mean per-clip cost of every static of the widened set that the tiers in P allow."""
    pr, cs = {}, {}
    for m, p in P.items():
        pr[m], cs[m] = p.argmax(1), float(c[m].mean())
    if "ast" in P:
        pr["avg"], cs["avg"] = ((P["mn10"] + P["ast"]) / 2).argmax(1), float((c["mn10"] + c["ast"]).mean())
    for pn in PAIRS:
        a, b = pn.split("+")
        if a in P and b in P:
            pr[pn], cs[pn] = ((P[a] + P[b]) / 2).argmax(1), float((c[a] + c[b]).mean())
    return pr, cs


class Bed:
    """One dataset. keys: [(seed, fold)]; iv(s, f) -> (y, {tier: P}, {tier: per-clip cost}) on clean inner validation;
    test(c, s, f) -> (y, {tier: P}, {tier: per-clip cost}); conds; ratio (Nadeau-Bengio n_test/n_train);
    boot: None, "perfold" (T21) or "shared" (T24: the folds share one test set); cascade_thr: {(s, f): t} or None."""

    def __init__(self, name, K, keys, iv, test, conds, ratio, boot=None, cascade_thr=None, boot_seed=None):
        self.name, self.K, self.keys, self.iv, self.test, self.conds = name, K, keys, iv, test, conds
        self.ratio, self.boot, self.cascade_thr, self.boot_seed = ratio, boot, cascade_thr, boot_seed


# ── budget choice on clean inner validation (C2) ──
def select_budgets(bed):
    K, keys = bed.K, bed.keys
    IV = {k: bed.iv(*k) for k in keys}
    st = {}
    for k, (y, P, c) in IV.items():
        pr, cs = statics(P, c)
        for nm in pr:
            st[(nm, k)] = (A.f1_fast(y, pr[nm], K), pr[nm], cs[nm])
    names = sorted({nm for nm, _ in st}, key=WIDE.index)
    sp = pd.DataFrame([{"system": nm, "f1": np.mean([st[(nm, k)][0] for k in keys]),
                        "cost": np.mean([st[(nm, k)][2] for k in keys])} for nm in names])

    def iv_hull(c_iv, rng):  # T21's inner-validation hull: Monte Carlo time-sharing, A.DRAWS draws
        v, lo, hi, lam = A.hull_pair(sp, c_iv)
        h = []
        for k in keys:
            y = IV[k][0]
            plo = st[(lo, k)][1]
            if lam == 0.0:
                h.append(st[(lo, k)][0])
                continue
            dr = rng.random((A.DRAWS, len(plo))) < lam
            h.append(float(A.f1_batch(y, np.where(dr, st[(hi, k)][1][None, :], plo[None, :]), K).mean()))
        return float(np.mean(h)), f"{lo}|{hi}|{lam:.3f}"

    rows, thr = [], {}
    rng = np.random.default_rng(A.RNG_IV)
    for r in BUDGETS:
        f1s, cs = [], []
        for k in keys:
            y, P, c = IV[k]
            t = A.thr_rate(A.margin(P["mn10"]), r)
            thr[("s1", r, k)] = t
            pr, esc = s1_pred(P["mn10"], P["mn20"], t)
            f1s.append(A.f1_fast(y, pr, K))
            cs.append(float(np.mean(c["mn10"] + esc * c["mn20"])))
        h, hs = iv_hull(float(np.mean(cs)), rng)
        rows.append({"system": "s1", "budget": str(r), "f1_iv": 100 * np.mean(f1s), "cost_iv": np.mean(cs), "hull_iv": 100 * h,
                     "gain_iv": 100 * (np.mean(f1s) - h), "hull": hs})
    if "ast" in IV[keys[0]][1]:
        rng = np.random.default_rng(A.RNG_IV)
        for r2, r3 in S2_GRID:
            f1s, cs = [], []
            for k in keys:
                y, P, c = IV[k]
                t2, t3 = s2_thr(P["mn10"], P["mn20"], r2, r3)
                thr[("s2", (r2, r3), k)] = (t2, t3)
                pr, e2, e3 = s2_pred(P["mn10"], P["mn20"], P["ast"], t2, t3)
                f1s.append(A.f1_fast(y, pr, K))
                cs.append(float(np.mean(c["mn10"] + e2 * c["mn20"] + e3 * c["ast"])))
            h, hs = iv_hull(float(np.mean(cs)), rng)
            rows.append({"system": "s2", "budget": f"{r2}|{r3}", "f1_iv": 100 * np.mean(f1s), "cost_iv": np.mean(cs), "hull_iv": 100 * h,
                         "gain_iv": 100 * (np.mean(f1s) - h), "hull": hs})
    SEL = pd.DataFrame(rows)
    pick, th = {}, {}
    for sysn in SEL.system.unique():
        g = SEL[SEL.system == sysn].sort_values(["gain_iv", "cost_iv"], ascending=[False, True])
        b = g.iloc[0].budget
        pick[sysn] = b
        key = float(b) if sysn == "s1" else tuple(float(x) for x in b.split("|"))
        th[sysn] = {k: thr[(sysn, key, k)] for k in keys}
    return SEL, pick, th, sp


# ── test (C3, C8) ──
def run_test(bed, th):
    K, keys = bed.K, bed.keys
    rows, preds = [], {}
    for c in bed.conds:
        for k in keys:
            y, P, cc = bed.test(c, *k)
            for nm, p in P.items():
                if not np.isfinite(p).all():
                    raise SystemExit(f"{bed.name} {nm} {c} {k}: missing test logits")
            pr, cs = statics(P, cc)
            rate = {}
            if "s1" in th:
                pr["s1"], e = s1_pred(P["mn10"], P["mn20"], th["s1"][k])
                cs["s1"], rate["s1"] = float(np.mean(cc["mn10"] + e * cc["mn20"])), float(e.mean())
                # descriptive: S1 with the two CNNs sharing one front end (escalated clips do not pay MN20's front end)
                if "frontend" in cc:
                    cs["s1_sharedfe"] = float(np.mean(cc["mn10"] + e * (cc["mn20"] - cc["frontend"])))
            if "ast" in P:
                if "s2" in th:
                    pr["s2"], e2, e3 = s2_pred(P["mn10"], P["mn20"], P["ast"], *th["s2"][k])
                    cs["s2"] = float(np.mean(cc["mn10"] + e2 * cc["mn20"] + e3 * cc["ast"]))
                    rate["s2"], rate["s2_tier3"] = float(e2.mean()), float(e3.mean())
                if bed.cascade_thr is not None:
                    pr["cascade"], ec = cascade_pred(P["mn10"], P["ast"], bed.cascade_thr[k])
                    cs["cascade"], rate["cascade"] = float(np.mean(cc["mn10"] + ec * cc["ast"])), float(ec.mean())
            mc = cc.get("_macs")  # unpriced view (MACs per clip), secondary
            ms = {}
            if mc is not None:
                _, ms = statics({m: P[m] for m in P if m in mc}, mc)
                if "s1" in pr:
                    ms["s1"] = float(np.mean(mc["mn10"] + e * mc["mn20"]))
                if "s2" in pr:
                    ms["s2"] = float(np.mean(mc["mn10"] + e2 * mc["mn20"] + e3 * mc["ast"]))
                if "cascade" in pr:
                    ms["cascade"] = float(np.mean(mc["mn10"] + ec * mc["ast"]))
            for nm in pr:
                rows.append({"cond": c, "seed": k[0], "fold": k[1], "system": nm, "f1": A.f1_fast(y, pr[nm], K) * 100,
                             "cost": cs[nm], "macs": ms.get(nm, np.nan), "rate": rate.get(nm, np.nan),
                             "rate_tier3": rate.get("s2_tier3", np.nan) if nm == "s2" else np.nan})
            if "s1_sharedfe" in cs:
                rows.append({"cond": c, "seed": k[0], "fold": k[1], "system": "s1_sharedfe", "f1": A.f1_fast(y, pr["s1"], K) * 100,
                             "cost": cs["s1_sharedfe"], "rate": rate["s1"], "rate_tier3": np.nan})
            preds[(c, k)] = (y, pr)
    return pd.DataFrame(rows), preds


def fold_test(x, z, idx, ratio):
    d = x - z
    fm = pd.Series(d, index=idx).groupby(level="fold").mean().to_numpy()
    t = A.tests(x, z, list(idx), ratio)
    lo, hi = T.nb_ci(fm, ratio)
    return {"d": float(fm.mean()), "nb_lo": lo, "nb_hi": hi, "p_nb": float(t["p_nb"]), "p_signflip": float(t["p_signflip"]),
            "signflip_floor": float(t["signflip_floor"]), "folds_pos": int((fm > 0).sum()), "n_folds": len(fm),
            "fold_d": json.dumps(np.round(fm, 3).tolist())}


def compare_all(bed, TT, preds, systems=("s1", "s2", "cascade")):
    """Each system against the widened hull at its own mean cost, against the MN10+MN20 average and against the T21
    cascade (both at their own costs), per condition; clip-level bootstrap of system minus hull on clean and field."""
    K, keys = bed.K, bed.keys
    out, hull_rows = [], []
    for c in bed.conds:
        g = TT[TT.cond == c]
        sp = g[g.system.isin(WIDE)].groupby("system").agg(f1=("f1", "mean"), cost=("cost", "mean"), macs=("macs", "mean")).reset_index()
        sp["f1"] /= 100
        V = lambda nm: g[g.system == nm].set_index(["seed", "fold"]).sort_index().f1  # noqa: E731
        for sysn in systems:
            if sysn not in g.system.values:
                continue
            xs = V(sysn)
            Cx = float(g[g.system == sysn].cost.mean())
            v, lo, hi, lam = A.hull_pair(sp[["system", "f1", "cost"]], Cx)
            hv = np.array([(A.f1_fast(preds[(c, k)][0], preds[(c, k)][1][lo], K) if lam == 0 else
                            T.exp_f1(preds[(c, k)][0], preds[(c, k)][1][lo], preds[(c, k)][1][hi], lam, K)) * 100 for k in xs.index])
            r = {"dataset": bed.name, "cond": c, "system": sysn, "f1": float(xs.mean()), "cost": Cx,
                 "macs": float(g[g.system == sysn].macs.mean()) if "macs" in g else np.nan,
                 "rate": float(g[g.system == sysn].rate.mean()), "hull": f"{lo}|{hi}|{lam:.4f}", "hull_lo": lo, "hull_hi": hi,
                 "hull_lam": lam, "hull_f1": float(hv.mean())}
            r.update({f"{kk}_hull": vv for kk, vv in fold_test(xs.to_numpy(), hv, xs.index, bed.ratio).items()})
            for other in ("mn10+mn20", "cascade", "s1"):
                if other != sysn and other in g.system.values:
                    zo = V(other)
                    r.update({f"{kk}_vs_{other}": vv for kk, vv in fold_test(xs.to_numpy(), zo.to_numpy(), xs.index, bed.ratio).items()})
                    r[f"f1_{other}"], r[f"cost_{other}"] = float(zo.mean()), float(g[g.system == other].cost.mean())
            if bed.boot and c in ("clean", "field"):
                r["boot_lo_hull"], r["boot_hi_hull"] = boot_ci(bed, preds, c, sysn, lo, hi, lam)
            out.append(r)
        for s_ in sp.itertuples():
            hull_rows.append({"dataset": bed.name, "cond": c, "system": s_.system, "f1": s_.f1 * 100, "cost": s_.cost, "macs": s_.macs})
    return pd.DataFrame(out), pd.DataFrame(hull_rows)


def boot_ci(bed, preds, c, sysn, lo, hi, lam):
    """T21's clip-level bootstrap (2,000 draws) of system minus hull; "perfold": each fold's own test clips resampled
    (T21, T25); "shared": the same resampled test clips for every fold and seed (T24)."""
    K = bed.K
    seeds = sorted({k[0] for k in bed.keys})
    folds = sorted({k[1] for k in bed.keys})
    rng = np.random.default_rng(bed.boot_seed)
    acc = np.zeros(T.NB_BOOT)
    if bed.boot == "shared":
        n = len(preds[(c, bed.keys[0])][0])
        I_shared = rng.integers(0, n, (T.NB_BOOT, n))
    for f in folds:
        n_f = len(preds[(c, (seeds[0], f))][0])
        I = I_shared if bed.boot == "shared" else rng.integers(0, n_f, (T.NB_BOOT, n_f))
        dsum = np.zeros(T.NB_BOOT)
        for s in seeds:
            y, pr = preds[(c, (s, f))]
            Y = y[I]
            fc = T.f1_rows(Y, pr[sysn][I], K)
            fh = (1 - lam) * T.f1_rows(Y, pr[lo][I], K) + (lam * T.f1_rows(Y, pr[hi][I], K) if lam > 0 else 0)
            dsum += fc - fh
        acc += dsum / len(seeds)
    acc = acc / len(folds) * 100
    return float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))


def smoke_permute(bed, seed=99):
    """Code check only: every label the bed returns is replaced by a fixed random permutation of it (chance-level F1)."""
    cache = {}

    def perm(y, tag):
        if tag not in cache:
            cache[tag] = np.random.default_rng(seed + len(cache)).permutation(len(y))
        return y[cache[tag]]
    iv0, te0 = bed.iv, bed.test

    def iv(s, f):
        y, P, c = iv0(s, f)
        return perm(y, ("iv", s, f)), P, c

    def test(c_, s, f):
        y, P, c = te0(c_, s, f)
        return perm(y, ("te", c_, f)), P, c
    bed.iv, bed.test = iv, test


def scaler_rule(Z_iv_f1):
    """T21's rule: refit only if its mean clean inner-validation macro F1 is strictly higher than fp32fit."""
    return "refit" if Z_iv_f1["refit"] > Z_iv_f1["fp32fit"] else "fp32fit"
