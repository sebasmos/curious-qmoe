"""T24 analysis (ANALYSIS_PLAN.md C1, C5, C6): scaler and budget choice on clean inner validation, statics, the fixed
cascade, the equal-cost hull, random gate, best single, tests, humility, verdict, Holm with T21's D1.

Same rules and code paths as t21_analyze.py (stage 2 helpers, stage 3c hull_pair, mix_expected_f1), adapted to the
fixed-test design: five fold models scored on one official test set. Changes from t21_analyze.py:
- test probabilities are per fold (t24_score.py layout); every fold's test rows are the official test set;
- Nadeau-Bengio ratio per fold = n_test / (n_fit + n_iv) of that fold's model (the rows the model used), averaged;
- clip-level bootstrap: the same resampled test clips for every fold and seed in a draw (the folds share the test set);
- Holm over the datasets where H1 is tested: T21's D1 p (read from its verdict.json) and this dataset's p;
  family-of-20 number min(1, 20 p) for transparency;
- precondition reported with every fold's value and the count of folds at or above 2.0 (the rule is the fold mean);
- optional --drop FILE (one `name` per line, all.csv names): the same analysis on the test rows not listed
  (overlap sensitivity, section C9), written under the label `overlapdrop`.
No int8 and no official-split secondary (ANALYSIS_PLAN.md C1).

    python t24_analyze.py --r R --d d3 --seeds 0 1 2 [--drop R/tables/d3/overlap_drop.txt]
    python t24_analyze.py --r R --combine d3 [d4]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import stats
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in (ROOT, ROOT / "scripts/redesign/idea6/stage2"):
    sys.path.insert(0, str(p))
import analyze as A  # noqa: E402  stage 2 helpers
from idea5b_common import holm, mix_expected_f1  # noqa: E402
from QWave import efficientat_mn as M  # noqa: E402
from QWave.costs import cost as net_cost  # noqa: E402
from scripts.benchmark import build_model  # noqa: E402

BUDGETS = {"b05": 0.05, "b10": 0.10, "b20": 0.20, "b30": 0.30, "b50": 0.50}
NOISE = ["bg__10db", "bg__0db", "pink__10db", "pink__0db", "white__10db", "white__0db"]
CONDS = ["clean", *NOISE, "field"]
NB_BOOT, BOOT_SEED, FAMILY = 2000, 20261028, 20
MC_DRAWS, MC_SEED = 4000, 20261027
T21_D1_VERDICT = Path.home() / "Documents/qmoe-results/2026-10-26_1408_moe-t21_code-731570e/tables/d1/verdict.json"
T21_D1_P = 0.00171982442287984  # H1 p_nb of T21 D1 (FSDKaggle2018), field mix, +2.83


def d1_p():
    v = json.loads(T21_D1_VERDICT.read_text())["H1"]
    if abs(v["p_nb"] - T21_D1_P) > 1e-15:
        raise SystemExit("T21 D1 verdict.json differs from the p fixed in the plan")
    return T21_D1_P, float(v["d"])


def per_clip_costs(R, D, n_cls):
    """Priced bit-ops and MACs per clip of all.csv (each clip at its own length), as t21_analyze.per_clip_costs."""
    cfg = OmegaConf.load(ROOT / "config/esc50.yaml")
    with contextlib.redirect_stdout(io.StringIO()):
        h_mn = net_cost(build_model("esc", 960, n_cls, cfg))
        h_ast = net_cost(build_model("esc", 768, n_cls, cfg))
    Lm = np.load(R / "emb" / D / "mn_all" / "lengths__clean.npy")
    base = M.load_mn("mn10", Path.home() / "Documents/qmoe-weights/efficientat_release-v0.0.1/mn10_as_mAP_471.pt")
    cache = {}
    mn_b, mn_m = np.zeros(len(Lm)), np.zeros(len(Lm))
    for i, L in enumerate(Lm):
        fr = M.frames_of(M.n32_of(int(L)))
        if fr not in cache:
            cache[fr] = M.feature_macs(base, fr)
        fm = M.frontend_macs(int(L))
        mn_b[i] = fm * 1024 + M.backbone_bitops({**cache[fr], "aggregate_macs": 0.0}, "w8a8", 1024.0) + h_mn[1]
        mn_m[i] = fm + cache[fr]["macs"] + h_mn[0]
    tp = np.load(R / "emb" / D / "ast_all" / "time_patches.npy")
    La = np.load(R / "emb" / D / "ast_all" / "lengths16k.npy")
    if not np.array_equal(La, Lm):
        raise SystemExit("MN10 and AST read different clip lengths")
    fb = np.array([A.ast_fbank_frontend_macs(int(x)) for x in La])
    ast_b = np.array([A.ast_bitops_tok("w4g128", 1024.0, 12 * t + 2, 12 * t) for t in tp]) + fb * 1024 + h_ast[1]
    ast_m = np.array([A.ast_bitops_tok("fp32", 0.0, 12 * t + 2, 12 * t) / 1024 for t in tp]) + fb + h_ast[0]
    return {"priced": (mn_b, ast_b), "macs": (mn_m, ast_m)}, {"head_mn_bitops": h_mn[1], "head_ast_bitops": h_ast[1]}


def exp_f1(y, plo, phi, lam, k):
    """Expected macro F1 of random time-sharing (exact when every class is in y, else Monte Carlo), as T21."""
    if len(np.unique(y)) == k:
        return float(mix_expected_f1(y, plo, phi, lam, k)[0])
    rng = np.random.default_rng(MC_SEED)
    acc = []
    for _ in range(MC_DRAWS // 500):
        dr = rng.random((500, len(y))) < lam
        acc.append(A.f1_batch(y, np.where(dr, phi[None, :], plo[None, :]), k))
    return float(np.concatenate(acc).mean())


def nb_ci(fm, ratio):
    fm = np.asarray(fm, float)
    k = len(fm)
    se = np.sqrt((1 / k + ratio) * fm.var(ddof=1))
    h = stats.t.ppf(0.975, k - 1) * se
    return float(fm.mean() - h), float(fm.mean() + h)


def f1_rows(Y, P, k):
    """Macro F1 per row of Y, P [D, n] (classes present in y or p, sklearn's rule)."""
    Dn = Y.shape[0]
    idx = (np.arange(Dn)[:, None] * (k * k) + Y * k + P).ravel()
    cm = np.bincount(idx, minlength=Dn * k * k).reshape(Dn, k, k)
    tp = np.diagonal(cm, axis1=1, axis2=2).astype(float)
    den = cm.sum(1) + cm.sum(2)
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0).sum(1) / (den > 0).sum(1)


class Scores:
    def __init__(self, R, D, tier, seeds):
        self.z = {s: dict(np.load(R / "scores" / D / f"{tier}_s{s}.npz")) for s in seeds}
        z0 = self.z[seeds[0]]
        self.y, self.test_idx = z0["y"].astype(np.int64), z0["test_idx"].astype(int)
        self.folds = sorted(int(k[7:]) for k in z0 if k.startswith("iv_idx_"))
        self.n_fit = {f: int(z0[f"n_fit_{f}"]) for f in self.folds}

    def test(self, s, f, c, k):
        return A.softmax(self.z[s][f"test__{f}__{c}__{k}"].astype(np.float64))

    def iv(self, s, f, c, k):
        key = f"iv__{f}__{c}__{k}"
        return self.z[s][f"iv_idx_{f}"].astype(int), (A.softmax(self.z[s][key].astype(np.float64)) if key in self.z[s] else None)


def analyze_design(R, D, seeds, costs, K, out, label, keep=None, conds=CONDS):
    C = Scores(R, D, "mn10_w8a8", seeds)
    S = Scores(R, D, "ast_w4g128", seeds)
    if not (np.array_equal(C.y, S.y) and np.array_equal(C.test_idx, S.test_idx) and C.folds == S.folds):
        raise SystemExit("rows differ between tiers")
    y = C.y
    keep = np.ones(len(C.test_idx), bool) if keep is None else keep
    te = C.test_idx[keep]
    cc, ca = costs["priced"]
    keys = [(s, f) for s in seeds for f in C.folds]
    for s, f in keys:  # leakage checks
        iv, _ = C.iv(s, f, "clean", "fp32fit")
        iv2, _ = S.iv(s, f, "clean", "fp32fit")
        assert not (set(iv) & set(C.test_idx)) and not (set(iv2) & set(C.test_idx)), "inner validation overlaps test"
    # ── scaler choice on clean inner validation ──
    sc_rows, scaler = [], {}
    for name, T in (("cheap", C), ("ast", S)):
        m = {}
        for k in ("fp32fit", "refit"):
            v = []
            for s, f in keys:
                iv, p = T.iv(s, f, "clean", k)
                v.append(A.f1_fast(y[iv], p.argmax(1), K))
            m[k] = float(np.mean(v)) * 100
        scaler[name] = "refit" if m["refit"] > m["fp32fit"] else "fp32fit"
        sc_rows.append({"tier": name, "iv_f1_fp32fit": m["fp32fit"], "iv_f1_refit": m["refit"], "chosen": scaler[name]})
    ks, ka = scaler["cheap"], scaler["ast"]
    # ── budget choice on clean inner validation (T21 code path) ──
    rng = np.random.default_rng(A.RNG_IV)
    ivd = {}
    for s, f in keys:
        iv, pc = C.iv(s, f, "clean", ks)
        iv2, pa = S.iv(s, f, "clean", ka)
        if not np.array_equal(iv, iv2):
            raise SystemExit(f"s{s} f{f}: the two tiers' inner-validation rows differ")
        ivd[(s, f)] = (iv, pc, pa)
    st_iv = {}
    for (s, f), (iv, pc, pa) in ivd.items():
        for nm, p, c_ in (("mn", pc, cc[iv]), ("ast", pa, ca[iv]), ("avg", (pc + pa) / 2, cc[iv] + ca[iv])):
            st_iv[(nm, s, f)] = (A.f1_fast(y[iv], p.argmax(1), K), p.argmax(1), float(c_.mean()))
    sp_iv = pd.DataFrame([{"system": nm, "f1": np.mean([st_iv[(nm, s, f)][0] for s, f in keys]),
                           "cost": np.mean([st_iv[(nm, s, f)][2] for s, f in keys])} for nm in ("mn", "ast", "avg")])
    thr, b_rows = {}, []
    for b, rate in BUDGETS.items():
        f1s, cs, hull = [], [], []
        for s, f in keys:
            iv, pc, pa = ivd[(s, f)]
            t_ = A.thr_rate(A.margin(pc), rate)
            thr[(b, s, f)] = t_
            esc = A.margin(pc) < t_
            pred = np.where(esc, ((pc + pa) / 2).argmax(1), pc.argmax(1))
            f1s.append(A.f1_fast(y[iv], pred, K))
            cs.append(float(np.mean(cc[iv] + esc * ca[iv])))
        c_iv = float(np.mean(cs))
        v, lo, hi, lam = A.hull_pair(sp_iv, c_iv)
        for s, f in keys:
            iv = ivd[(s, f)][0]
            plo = st_iv[(lo, s, f)][1]
            if lam == 0.0:
                hull.append(st_iv[(lo, s, f)][0])
                continue
            phi = st_iv[(hi, s, f)][1]
            dr = rng.random((A.DRAWS, len(plo))) < lam
            hull.append(float(A.f1_batch(y[iv], np.where(dr, phi[None, :], plo[None, :]), K).mean()))
        b_rows.append({"budget": b, "f1_iv": np.mean(f1s) * 100, "cost_iv": c_iv, "hull_iv": np.mean(hull) * 100,
                       "gain_iv": (np.mean(f1s) - np.mean(hull)) * 100, "hull": f"{lo}|{hi}|{lam:.3f}"})
    BS = pd.DataFrame(b_rows).sort_values(["gain_iv", "cost_iv"], ascending=[False, True])
    bstar = BS.iloc[0].budget
    # best single on inner validation of the condition family
    best = {}
    for fam in ("clean", "field"):
        v = {}
        for nm, T, k in (("mn", C, ks), ("ast", S, ka)):
            vals = []
            for s, f in keys:
                iv, p = T.iv(s, f, fam, k)
                if p is None:
                    break
                ok = np.isfinite(p).all(1)
                vals.append(A.f1_fast(y[iv][ok], p[ok].argmax(1), K))
            v[nm] = np.mean(vals) if len(vals) == len(keys) else np.nan
        best[fam] = max(v, key=lambda x: -np.inf if np.isnan(v[x]) else v[x])
    # ── test (every fold model on the same test rows) ──
    rows, au, preds = [], [], {}
    yt = y[te]
    mc, ma = costs["macs"]
    for c in conds:
        for s, f in keys:
            pc, pa = C.test(s, f, c, ks)[keep], S.test(s, f, c, ka)[keep]
            if not (np.isfinite(pc).all() and np.isfinite(pa).all()):
                raise SystemExit(f"{c} s{s} f{f}: missing test logits")
            m = A.margin(pc)
            esc = m < thr[(bstar, s, f)]
            pr = {"mn": pc.argmax(1), "ast": pa.argmax(1), "avg": ((pc + pa) / 2).argmax(1)}
            pr["cascade"] = np.where(esc, pr["avg"], pr["mn"])
            cost = {"mn": cc[te].mean(), "ast": ca[te].mean(), "avg": (cc[te] + ca[te]).mean(),
                    "cascade": float(np.mean(cc[te] + esc * ca[te]))}
            macs = {"mn": mc[te].mean(), "ast": ma[te].mean(), "avg": (mc[te] + ma[te]).mean(), "cascade": float(np.mean(mc[te] + esc * ma[te]))}
            for nm, p in pr.items():
                rows.append({"cond": c, "seed": s, "fold": f, "system": nm, "f1": A.f1_fast(yt, p, K) * 100,
                             "acc": float((p == yt).mean()) * 100, "cost": cost[nm], "macs": macs[nm], "rate": float(esc.mean()) if nm == "cascade" else np.nan})
            rows.append({"cond": c, "seed": s, "fold": f, "system": "random_gate", "f1": exp_f1(yt, pr["mn"], pr["avg"], esc.mean(), K) * 100,
                         "cost": cost["cascade"], "rate": float(esc.mean())})
            preds[(c, s, f)] = pr
            fix = (pr["mn"] != yt) & (pr["ast"] == yt)
            wrong = pr["mn"] != yt
            au.append({"cond": c, "seed": s, "fold": f, "auroc_fixable": roc_auc_score(fix, 1 - m) if 0 < fix.sum() < len(fix) else np.nan,
                       "auroc_wrong": roc_auc_score(wrong, 1 - m) if 0 < wrong.sum() < len(wrong) else np.nan,
                       "rate": float(esc.mean()), "cheap_error": float(wrong.mean()), "fixable": float(fix.mean())})
    T_ = pd.DataFrame(rows)
    AU = pd.DataFrame(au)
    n_te = len(te)
    n_iv = {f: len(C.iv(seeds[0], f, "clean", "fp32fit")[0]) for f in C.folds}
    ratio = float(np.mean([n_te / (C.n_fit[f] + n_iv[f]) for f in C.folds]))
    res = []
    bs_I = np.random.default_rng(BOOT_SEED).integers(0, n_te, (NB_BOOT, n_te))  # same clips for every fold and seed
    for c in conds:
        g = T_[T_.cond == c]
        sp = g[g.system.isin(["mn", "ast", "avg"])].groupby("system").agg(f1=("f1", "mean"), cost=("cost", "mean")).reset_index()
        sp["f1"] /= 100
        casc = g[g.system == "cascade"].set_index(["seed", "fold"]).sort_index()
        Cc = float(casc.cost.mean())
        v, lo, hi, lam = A.hull_pair(sp, Cc)
        hull = {}
        for s, f in keys:
            pr = preds[(c, s, f)]
            hull[(s, f)] = (A.f1_fast(yt, pr[lo], K) if lam == 0 else exp_f1(yt, pr[lo], pr[hi], lam, K)) * 100
        fam = "field" if c == "field" else "clean"
        bsn = best[fam]
        vals = lambda nm: g[g.system == nm].set_index(["seed", "fold"]).sort_index().f1  # noqa: E731
        xc = vals("cascade")
        r = {"label": label, "cond": c, "budget": bstar, "scaler_cheap": ks, "scaler_ast": ka, "n_test": n_te,
             "cascade_f1": xc.mean(), "cascade_cost": Cc, "rate": casc.rate.mean(), "rate_min": casc.rate.min(), "rate_max": casc.rate.max(),
             "mn_f1": vals("mn").mean(), "ast_f1": vals("ast").mean(), "avg_f1": vals("avg").mean(),
             "mn_cost": sp.set_index("system").cost["mn"], "ast_cost": sp.set_index("system").cost["ast"],
             "hull": f"{lo}|{hi}|{lam:.4f}", "hull_f1": np.mean(list(hull.values())), "random_gate_f1": vals("random_gate").mean(),
             "best_single": bsn, "best_single_f1": vals(bsn).mean(), "n_folds": len(C.folds), "n_seeds": len(seeds)}
        hv = np.array([hull[k_] for k_ in xc.index])
        for nm, z in (("hull", hv), ("random", vals("random_gate").to_numpy()), ("best", vals(bsn).to_numpy()), ("avg", vals("avg").to_numpy())):
            d = xc.to_numpy() - z
            fm = pd.Series(d, index=xc.index).groupby(level="fold").mean().to_numpy()
            r[f"d_{nm}"] = float(fm.mean())
            r[f"folds_pos_{nm}"] = int((fm > 0).sum())
            t = A.tests(xc.to_numpy(), z, list(xc.index), ratio)
            r[f"p_nb_{nm}"], r[f"p_signflip_{nm}"] = t["p_nb"], t["p_signflip"]
            r[f"nb_lo_{nm}"], r[f"nb_hi_{nm}"] = nb_ci(fm, ratio)
            r[f"fold_d_{nm}"] = json.dumps(np.round(fm, 3).tolist())
        am = (vals("ast") - vals("mn")).groupby(level="fold").mean()
        r["d_ast_minus_mn"] = float(am.mean())
        r["fold_ast_minus_mn"] = json.dumps(np.round(am.to_numpy(), 3).tolist())
        r["folds_ast_minus_mn_ge2"] = int((am.to_numpy() >= 2.0).sum())
        if c in ("field", "clean"):
            acc = np.zeros(NB_BOOT)
            Y = yt[bs_I]
            for f in C.folds:
                dsum = np.zeros(NB_BOOT)
                for s in seeds:
                    pr = preds[(c, s, f)]
                    fc = f1_rows(Y, pr["cascade"][bs_I], K)
                    fh = (1 - lam) * f1_rows(Y, pr[lo][bs_I], K) + (lam * f1_rows(Y, pr[hi][bs_I], K) if lam > 0 else 0)
                    dsum += fc - fh
                acc += dsum / len(seeds)
            acc = acc / len(C.folds) * 100
            r["boot_lo_hull"], r["boot_hi_hull"] = float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))
        res.append(r)
    RES = pd.DataFrame(res)
    T_.to_csv(out / f"{label}_by_seed_fold.csv", index=False)
    AU.to_csv(out / f"{label}_gate_by_seed_fold.csv", index=False)
    RES.to_csv(out / f"{label}_by_condition.csv", index=False)
    BS.to_csv(out / f"{label}_budget_selection.csv", index=False)
    pd.DataFrame(sc_rows).to_csv(out / f"{label}_scaler_choice.csv", index=False)
    return RES, AU, BS, best, ratio


def humility(AU):
    rate = AU.groupby(["cond", "fold"]).rate.mean().unstack()  # cond x fold
    fm = rate.mean(1)
    out, exc = [], []
    for kind in ("bg", "pink", "white"):
        for a_, b_ in (("clean", f"{kind}__10db"), (f"{kind}__10db", f"{kind}__0db")):
            ok = bool(fm[b_] > fm[a_])
            out.append({"step": f"{a_} -> {b_}", "rate_from": fm[a_] * 100, "rate_to": fm[b_] * 100, "rises": ok,
                        "folds_rising": int((rate.loc[b_] > rate.loc[a_]).sum())})
            if not ok:
                exc.append(f"{a_} -> {b_}")
    ok = bool(fm["field"] > fm["clean"])
    out.append({"step": "clean -> field", "rate_from": fm["clean"] * 100, "rate_to": fm["field"] * 100, "rises": ok,
                "folds_rising": int((rate.loc["field"] > rate.loc["clean"]).sum())})
    if not ok:
        exc.append("clean -> field")
    auc = AU.groupby("cond")[["auroc_fixable", "auroc_wrong", "cheap_error", "fixable", "rate"]].mean()
    return pd.DataFrame(out), exc, auc


def summary_row(r):
    return {"d": float(r.d_hull), "nb_ci95": [float(r.nb_lo_hull), float(r.nb_hi_hull)], "boot_ci95": [float(r.boot_lo_hull), float(r.boot_hi_hull)],
            "p_nb": float(r.p_nb_hull), "folds_positive": int(r.folds_pos_hull), "fold_d": r.fold_d_hull, "n_test": int(r.n_test),
            "cascade_f1": float(r.cascade_f1), "hull_f1": float(r.hull_f1), "ast_minus_mn": float(r.d_ast_minus_mn)}


def combine(R, ds):
    """Holm over D1 (T21) and the T24 datasets whose precondition passed; S1 over D1 and those datasets."""
    p1, d1 = d1_p()
    V = {d: json.loads((R / "tables" / d / "verdict.json").read_text()) for d in ds}
    tested = [d for d in ds if V[d]["precondition"]["passes"]]
    ps = np.array([p1] + [V[d]["H1"]["p_nb"] for d in tested])
    adj = holm(ps)
    out = {"candidates_tried": ds, "precondition": {d: {"value": V[d]["precondition"]["value"], "passes": V[d]["precondition"]["passes"]} for d in ds},
           "holm_over": ["D1_FSDKaggle2018_T21"] + tested,
           "H1": {"D1_FSDKaggle2018_T21": {"d": d1, "p_nb": p1, "p_holm": float(adj[0]), "passes": bool(d1 >= 1.0 and adj[0] < 0.05)}}}
    for d, p in zip(tested, adj[1:]):
        out["H1"][d] = {"d": V[d]["H1"]["d"], "p_nb": V[d]["H1"]["p_nb"], "p_holm": float(p), "p_family20": min(1.0, FAMILY * V[d]["H1"]["p_nb"]),
                        "nb_ci95": V[d]["H1"]["nb_ci95"], "boot_ci95": V[d]["H1"]["boot_ci95"], "folds_positive": V[d]["H1"]["folds_positive"],
                        "passes": bool(V[d]["H1"]["d"] >= 1.0 and p < 0.05)}
    out["S1"] = {"rule": "|cascade - hull| < 0.3 on D1 and on every T24 dataset where H1 is tested",
                 "per_dataset": {"D1_FSDKaggle2018_T21": bool(abs(d1) < 0.3), **{d: V[d]["S1"]["fires_here"] for d in tested}},
                 "fires": bool(abs(d1) < 0.3 and all(V[d]["S1"]["fires_here"] for d in tested))}
    for d in ds:
        out.setdefault("H2", {})[d] = V[d]["H2"]["passes"]
        out.setdefault("H3", {})[d] = V[d]["H3"]["passes"]
    (R / "tables" / "verdict_combined.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=Path, required=True)
    ap.add_argument("--d", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--drop", type=Path, default=None)
    ap.add_argument("--combine", nargs="*", default=None)
    a = ap.parse_args()
    R, D = a.r, a.d
    if a.combine is not None:
        return combine(R, a.combine)
    seeds = a.seeds or {"d3": [0, 1, 2], "d4": [0, 1, 2, 3, 4]}[D]
    out = R / "tables" / D
    out.mkdir(parents=True, exist_ok=True)
    al = pd.read_csv(R / "csv" / D / "all.csv")
    K = int(al.class_id.nunique())
    costs, hc = per_clip_costs(R, D, K)
    pd.DataFrame({"mn10_w8a8_bitops": costs["priced"][0], "ast_w4g128_bitops": costs["priced"][1], "mn10_macs": costs["macs"][0],
                  "ast_macs": costs["macs"][1]}).describe().to_csv(out / "costs_per_clip_summary.csv")
    prim, AU, BS, best, ratio = analyze_design(R, D, seeds, costs, K, out, "primary")
    hum, exc, auc = humility(AU)
    hum.to_csv(out / "humility.csv", index=False)
    auc.to_csv(out / "gate_auroc_by_condition.csv")
    fr = prim[prim.cond == "field"].iloc[0]
    cl = prim[prim.cond == "clean"].iloc[0]
    pre_ok = bool(fr.d_ast_minus_mn >= 2.0)
    p_h1 = float(fr.p_nb_hull)
    p1, d1 = d1_p()
    adj = holm(np.array([p1, p_h1]))
    h1 = bool(pre_ok and fr.d_hull >= 1.0 and adj[1] < 0.05)
    h2 = bool(cl.d_hull >= 1.0 and cl.p_nb_hull < 0.05)
    V = {"dataset": D, "seeds": seeds,
         "precondition": {"rule": "AST-Q4 minus MN10-Q8 on the field mix >= 2.0 points (mean over all five folds, as T21)",
                          "value": float(fr.d_ast_minus_mn), "folds": fr.fold_ast_minus_mn, "folds_ge_2": int(fr.folds_ast_minus_mn_ge2), "passes": pre_ok},
         "H1": {"rule": "field mix: cascade minus equal-cost hull >= 1.0 point, Nadeau-Bengio two-sided, Holm over D1 (T21) and this dataset < 0.05",
                "d": float(fr.d_hull), "nb_ci95": [fr.nb_lo_hull, fr.nb_hi_hull], "boot_ci95": [fr.boot_lo_hull, fr.boot_hi_hull],
                "p_nb": p_h1, "p_holm_with_d1": float(adj[1]) if pre_ok else None, "d1_p_holm_with_this": float(adj[0]) if pre_ok else p1,
                "p_family20": min(1.0, FAMILY * p_h1), "p_signflip": float(fr.p_signflip_hull),
                "signflip_floor": 2 / 2 ** int(fr.n_folds), "folds_positive": int(fr.folds_pos_hull), "n_folds": int(fr.n_folds),
                "fold_d": fr.fold_d_hull, "passes": h1 if pre_ok else "not tested (precondition failed)"},
         "H2": {"rule": "same on clean audio (raw NB p < 0.05)", "d": float(cl.d_hull), "nb_ci95": [cl.nb_lo_hull, cl.nb_hi_hull],
                "boot_ci95": [cl.boot_lo_hull, cl.boot_hi_hull], "p_nb": float(cl.p_nb_hull), "folds_positive": int(cl.folds_pos_hull), "passes": h2},
         "H3": {"rule": "escalation (clean threshold) rises at every noise step; field above clean", "exceptions": exc, "passes": not exc,
                "gate_auroc_fixable_field": float(auc.loc["field", "auroc_fixable"]), "gate_auroc_fixable_clean": float(auc.loc["clean", "auroc_fixable"])},
         "S1": {"rule": "|cascade - hull| < 0.3 on D1 and on this dataset", "fires_here": bool(abs(fr.d_hull) < 0.3),
                "fires": bool(abs(fr.d_hull) < 0.3 and abs(d1) < 0.3)},
         "design": {"budget": fr.budget, "scaler_cheap": fr.scaler_cheap, "scaler_ast": fr.scaler_ast, "best_single_field": best["field"],
                    "best_single_clean": best["clean"]}, "nb_ratio": ratio, "head_costs": hc}
    if a.drop is not None:
        names = set(Path(a.drop).read_text().split())
        keep = ~al.name.iloc[np.flatnonzero(al.folder.to_numpy() == 0)].isin(names).to_numpy()
        od = analyze_design(R, D, seeds, costs, K, out, "overlapdrop", keep=keep, conds=["clean", "field"])[0]
        V["sensitivity_overlap_dropped"] = {"n_dropped": int((~keep).sum()), "field": summary_row(od[od.cond == "field"].iloc[0]),
                                            "clean": summary_row(od[od.cond == "clean"].iloc[0])}
    (out / "verdict.json").write_text(json.dumps(V, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(prim[["cond", "cascade_f1", "mn_f1", "ast_f1", "avg_f1", "hull_f1", "d_hull", "p_nb_hull", "folds_pos_hull", "d_random", "d_best", "rate", "cascade_cost", "ast_cost"]].round(3).to_string())
    print(json.dumps(V, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))[:3500])


if __name__ == "__main__":
    main()
