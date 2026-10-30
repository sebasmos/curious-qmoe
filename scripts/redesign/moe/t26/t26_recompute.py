"""T26 own recompute from the raw score files (ANALYSIS_PLAN.md C9). Imports none of the analysis helpers (no stage 2
analyze, no t21/t24/t26 analysis code): own softmax, margin, threshold, scaler rule, macro F1 (sklearn), budget
choice with a linear inner-validation hull, own hull over the widened statics with Monte Carlo time-sharing, own
Nadeau-Bengio test. Shared with the analysis: only the per-clip cost arrays (T21's audited MN10 and AST costs, T24's,
and the CNN MAC hooks), as T25's recompute did.

    python t26_recompute.py --r R26 --part a
    python t26_recompute.py --r R26 --part c --d d3 --seeds 0 1 2 [--precondition-only]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax
from sklearn.metrics import f1_score

Q = Path.home() / "Documents/qmoe-results"
R21 = Q / "2026-10-26_1408_moe-t21_code-731570e"
R25 = Q / "2026-10-28_2247_moe-t25_code-cc3b048"
R24 = Q / "2026-10-28_1827_moe-t24_code-9a51cf5"
CNN5 = ["mn04", "mn05", "mn10", "mn20", "dymn04"]
PAIRS = list(itertools.combinations(CNN5, 2))
RATES = [0.05, 0.10, 0.20, 0.30, 0.50]
MC = 2000


def marg(p):
    s = np.partition(p, -2, axis=1)
    return s[:, -1] - s[:, -2]


def cut_at(m, r):
    """Threshold t: the share of m strictly below t is closest to r (ties: the smaller share)."""
    u = np.unique(m)
    cuts = np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]])
    share = np.searchsorted(np.sort(m), cuts, side="left") / len(m)
    best = min(range(len(cuts)), key=lambda i: (abs(share[i] - r), share[i]))
    return cuts[best]


def f1(y, p):
    return 100 * f1_score(y, p, average="macro")


def f1_many(y, P, k):
    """Macro F1 (classes present in y or the prediction) for each row of P [D, n]."""
    Y = np.eye(k, dtype=np.int32)[y]
    out = np.empty(len(P))
    for i, p in enumerate(P):
        Pp = np.eye(k, dtype=np.int32)[p]
        tp = (Y * Pp).sum(0)
        den = Y.sum(0) + Pp.sum(0)
        ok = den > 0
        out[i] = 100 * np.mean(2 * tp[ok] / den[ok])
    return out


def envelope(points, c):
    """Best (F1, lo, hi, lam) at cost c over static points {name: (cost, f1)}: one system at or below c, or the straight
    line between one at or below and one above."""
    best = None
    for a, (ca, fa) in points.items():
        if ca > c * (1 + 1e-12):
            continue
        cand = [(fa, a, a, 0.0)]
        for b, (cb, fb) in points.items():
            if cb > c * (1 + 1e-12):
                lam = (c - ca) / (cb - ca)
                cand.append(((1 - lam) * fa + lam * fb, a, b, lam))
        for v in cand:
            if best is None or v[0] > best[0] + 1e-12:
                best = v
    return best


def nb(fm, ratio):
    fm = np.asarray(fm, float)
    k = len(fm)
    se = np.sqrt((1 / k + ratio) * fm.var(ddof=1))
    t = fm.mean() / se
    h = stats.t.ppf(0.975, k - 1) * se
    return {"d": float(fm.mean()), "lo": float(fm.mean() - h), "hi": float(fm.mean() + h), "p": float(2 * stats.t.sf(abs(t), k - 1)),
            "folds_pos": int((fm > 0).sum())}


def static_preds(P):
    pr = {m: p.argmax(1) for m, p in P.items()}
    if "ast" in P:
        pr["avg"] = (P["mn10"] + P["ast"]).argmax(1)
    for a, b in PAIRS:
        if a in P and b in P:
            pr[f"{a}+{b}"] = (P[a] + P[b]).argmax(1)
    return pr


def static_cost(c):
    out = {m: v for m, v in c.items() if m != "frontend"}
    if "ast" in c:
        out["avg"] = c["mn10"] + c["ast"]
    for a, b in PAIRS:
        if a in c and b in c:
            out[f"{a}+{b}"] = c[a] + c[b]
    return out


def systems(P, c, th):
    """S1, S2 and their per-clip costs."""
    m1 = marg(P["mn10"])
    out = {}
    if "s1" in th:
        e = m1 < th["s1"]
        out["s1"] = (np.where(e, (P["mn10"] + P["mn20"]).argmax(1), P["mn10"].argmax(1)), c["mn10"] + e * c["mn20"])
    if "s2" in th:
        t2, t3 = th["s2"]
        e2 = m1 < t2
        q = (P["mn10"] + P["mn20"]) / 2
        e3 = e2 & (marg(q) < t3)
        pred = np.where(e3, (P["mn10"] + P["mn20"] + P["ast"]).argmax(1), np.where(e2, q.argmax(1), P["mn10"].argmax(1)))
        out["s2"] = (pred, c["mn10"] + e2 * c["mn20"] + e3 * c["ast"])
    return out


def s2_cuts(P, r2, r3):
    m1 = marg(P["mn10"])
    t2 = cut_at(m1, r2)
    e2 = m1 < t2
    if e2.sum() == 0:
        return t2, -np.inf
    return t2, cut_at(marg((P["mn10"] + P["mn20"]) / 2)[e2], min(1.0, r3 * len(m1) / e2.sum()))


def choose(IV, keys):
    """Budgets on clean inner validation: largest gain over the linear widened hull, ties to the lower cost."""
    pts = {}
    for k in keys:
        y, P, c = IV[k]
        pr, cs = static_preds(P), static_cost(c)
        for nm in pr:
            a = pts.setdefault(nm, [[], []])
            a[0].append(cs[nm].mean())
            a[1].append(f1(y, pr[nm]))
    pts = {nm: (np.mean(v[0]), np.mean(v[1])) for nm, v in pts.items()}
    res, pick, cuts = [], {}, {}
    grid = {"s1": [(r,) for r in RATES]}
    if "ast" in IV[keys[0]][1]:
        grid["s2"] = list(itertools.product(RATES, RATES))
    for sysn, G in grid.items():
        rows = []
        for g in G:
            th = {k: (cut_at(marg(IV[k][1]["mn10"]), g[0]) if sysn == "s1" else s2_cuts(IV[k][1], *g)) for k in keys}
            f, cs = [], []
            for k in keys:
                y, P, c = IV[k]
                pred, cc = systems(P, c, {sysn: th[k]})[sysn]
                f.append(f1(y, pred))
                cs.append(cc.mean())
            h = envelope(pts, float(np.mean(cs)))[0]
            rows.append((np.mean(f) - h, np.mean(cs), g, th))
        rows.sort(key=lambda r: (-r[0], r[1]))
        pick[sysn], cuts[sysn] = rows[0][2], rows[0][3]
        res += [{"system": sysn, "budget": "|".join(str(x) for x in r[2]), "gain_iv": r[0], "cost_iv": r[1]} for r in rows]
    return pick, cuts, pd.DataFrame(res)


def evaluate(TE, keys, cuts, K, ratio, conds, rng_seed=7):
    out = {}
    rng = np.random.default_rng(rng_seed)
    for cond in conds:
        rec, pts = {}, {}
        for k in keys:
            y, P, c = TE[(cond, k)]
            pr, cs = static_preds(P), static_cost(c)
            sysd = systems(P, c, {s: cuts[s][k] for s in cuts})
            rec[k] = (y, pr, sysd)
            for nm in pr:
                a = pts.setdefault(nm, [[], []])
                a[0].append(cs[nm].mean())
                a[1].append(f1(y, pr[nm]))
        pts = {nm: (np.mean(v[0]), np.mean(v[1])) for nm, v in pts.items()}
        for sysn in cuts:
            Cx = float(np.mean([rec[k][2][sysn][1].mean() for k in keys]))
            _, lo, hi, lam = envelope(pts, Cx)
            xs, hs = [], []
            for k in keys:
                y, pr, sysd = rec[k]
                xs.append(f1(y, sysd[sysn][0]))
                if lam == 0:
                    hs.append(f1(y, pr[lo]))
                else:
                    dr = rng.random((MC, len(y))) < lam
                    hs.append(float(f1_many(y, np.where(dr, pr[hi][None, :], pr[lo][None, :]), K).mean()))
            d = pd.Series(np.array(xs) - np.array(hs), index=pd.MultiIndex.from_tuples(keys, names=["seed", "fold"]))
            fm = d.groupby(level="fold").mean().to_numpy()
            out[f"{sysn}|{cond}"] = {"f1": float(np.mean(xs)), "cost": Cx, "hull": f"{lo}|{hi}|{lam:.4f}", "hull_f1": float(np.mean(hs)), **nb(fm, ratio)}
        if "mn20" in pts:
            g = pd.Series([f1(rec[k][0], rec[k][1]["mn20"]) - f1(rec[k][0], rec[k][1]["mn10"]) for k in keys],
                          index=pd.MultiIndex.from_tuples(keys, names=["seed", "fold"])).groupby(level="fold").mean()
            out[f"mn20_minus_mn10|{cond}"] = {"d": float(g.mean()), "fold_values": np.round(g.to_numpy(), 3).tolist()}
    return out


def scaler_choice(z, folds, seeds, y):
    v = {}
    for sc in ("fp32fit", "refit"):
        v[sc] = np.mean([f1(y[z[s][f"iv_idx_{f}"]], z[s][f"iv__{f}__clean__{sc}"].argmax(1)) for s in seeds for f in folds])
    return "refit" if v["refit"] > v["fp32fit"] else "fp32fit"


def part_a(R):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "t21"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "t25"))
    import t21_analyze as T  # cost arrays only
    import t25_analyze as T25  # cost arrays only
    D, seeds = "d1", [0, 1, 2, 3, 4]
    pool = pd.read_csv(R21 / "csv" / D / "pool.csv")
    K = pool.class_id.nunique()
    src = {"mn10": (R21, "mn10_w8a8"), "ast": (R21, "ast_w4g128"), **{m: (R25, f"{m}_w8a8") for m in ("mn04", "mn05", "mn20", "dymn04")}}
    z = {m: {s: dict(np.load(r / "scores" / D / f"pool_{t}_s{s}.npz")) for s in seeds} for m, (r, t) in src.items()}
    y, fold = z["mn10"][0]["y"].astype(int), z["mn10"][0]["fold"].astype(int)
    folds = sorted(np.unique(fold).tolist())
    sc = {m: scaler_choice(z[m], folds, seeds, y) for m in z}
    c21, _ = T.per_clip_costs(R21, D, K)
    cn = T25.cnn_costs(R25, R21, D, K)
    cost = {"mn10": c21["priced"][0], "ast": c21["priced"][1], **{m: cn[m][0] for m in ("mn04", "mn05", "mn20", "dymn04")}}
    keys = [(s, f) for s in seeds for f in folds]
    IV, TE = {}, {}
    for s, f in keys:
        idx = z["mn10"][s][f"iv_idx_{f}"].astype(int)
        IV[(s, f)] = (y[idx], {m: softmax(z[m][s][f"iv__{f}__clean__{sc[m]}"].astype(np.float64), axis=1) for m in z}, {m: v[idx] for m, v in cost.items()})
        te = np.flatnonzero(fold == f)
        for c in ("clean", "field"):
            TE[(c, (s, f))] = (y[te], {m: softmax(z[m][s][f"test__{c}__{sc[m]}"].astype(np.float64), axis=1)[te] for m in z}, {m: v[te] for m, v in cost.items()})
    n = len(y)
    ratio = float(np.mean([(fold == f).sum() / (n - (fold == f).sum()) for f in folds]))
    pick, cuts, SEL = choose(IV, keys)
    return {"scalers": sc, "budgets": {k: list(v) for k, v in pick.items()}, "nb_ratio": ratio, **evaluate(TE, keys, cuts, K, ratio, ["clean", "field"])}, SEL


def part_c(R, D, seeds, pre_only):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "t24"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import t24_analyze as T24  # cost arrays only
    import t26_part_c as PC  # cnn_costs only
    al = pd.read_csv(R24 / "csv" / D / "all.csv")
    K = al.class_id.nunique()
    models = ["mn20"] if pre_only else ["mn04", "mn05", "mn20", "dymn04"]
    src = {"mn10": (R24, "mn10_w8a8"), "ast": (R24, "ast_w4g128"), **{m: (R, f"{m}_w8a8") for m in models}}
    z = {m: {s: dict(np.load(r / "scores" / D / f"{t}_s{s}.npz")) for s in seeds} for m, (r, t) in src.items()}
    y, te = z["mn10"][seeds[0]]["y"].astype(int), z["mn10"][seeds[0]]["test_idx"].astype(int)
    folds = sorted(int(k[7:]) for k in z["mn10"][seeds[0]] if k.startswith("iv_idx_"))
    sc = {m: scaler_choice(z[m], folds, seeds, y) for m in z}
    c24, _ = T24.per_clip_costs(R24, D, K)
    cn = PC.cnn_costs(R, D, K, models)
    cost = {"mn10": c24["priced"][0], "ast": c24["priced"][1], **{m: cn[m] for m in models}}
    keys = [(s, f) for s in seeds for f in folds]
    IV, TE = {}, {}
    for s, f in keys:
        idx = z["mn10"][s][f"iv_idx_{f}"].astype(int)
        IV[(s, f)] = (y[idx], {m: softmax(z[m][s][f"iv__{f}__clean__{sc[m]}"].astype(np.float64), axis=1) for m in z}, {m: v[idx] for m, v in cost.items()})
        for c in ("clean", "field"):
            TE[(c, (s, f))] = (y[te], {m: softmax(z[m][s][f"test__{f}__{c}__{sc[m]}"].astype(np.float64), axis=1) for m in z}, {m: v[te] for m, v in cost.items()})
    z0 = z["mn10"][seeds[0]]
    ratio = float(np.mean([len(te) / (int(z0[f"n_fit_{f}"]) + len(z0[f"iv_idx_{f}"])) for f in folds]))
    if pre_only:
        return {"scalers": sc, "nb_ratio": ratio, **evaluate(TE, keys, {}, K, ratio, ["clean", "field"])}, None
    pick, cuts, SEL = choose(IV, keys)
    return {"scalers": sc, "budgets": {k: list(v) for k, v in pick.items()}, "nb_ratio": ratio, **evaluate(TE, keys, cuts, K, ratio, ["clean", "field"])}, SEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=Path, required=True)
    ap.add_argument("--part", choices=["a", "c"], required=True)
    ap.add_argument("--d")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--precondition-only", action="store_true")
    a = ap.parse_args()
    if a.part == "a":
        res, SEL = part_a(a.r)
        out = a.r / "tables" / "a"
    else:
        res, SEL = part_c(a.r, a.d, a.seeds, a.precondition_only)
        out = a.r / "tables" / "c" / a.d
    out.mkdir(parents=True, exist_ok=True)
    tag = "recompute_precondition" if a.precondition_only else "recompute"
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1, default=str))
    if SEL is not None:
        SEL.to_csv(out / f"{tag}_budget_selection.csv", index=False)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
