"""T24 own recompute (ANALYSIS_PLAN.md C11), written apart from t24_analyze.py: it imports none of the analysis or
stage 2 helpers. From the saved score files only, it redoes the scaler choice, the budget choice (its own threshold
rule and a linearly interpolated hull), the test F1 of every system per fold model and seed, the precondition, the
cascade minus the equal-cost hull on the field mix and on clean audio, and the Nadeau-Bengio test, then prints them
beside t24_analyze.py's verdict.json.

Own cost model (secondary to the analysis's priced costs): AST w4g128 per clip from its token count N = 12 t + 2
(12 blocks; Linear MACs 12 x 768^2 per token at 4 x 8 bits plus one 32 x 32 rescale per 128 MACs; QK^T at 8 x 8 and PV
at 16 x 8 bits, N^2 x 768 MACs each); MN10 per clip as its sidecar mean bit-ops scaled by the clip's length. Front
ends and heads are left out (under 0.2% of an AST pass).

    python t24_recompute.py --r R --d d3 --seeds 0 1 2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

BUD = {"b05": 0.05, "b10": 0.10, "b20": 0.20, "b30": 0.30, "b50": 0.50}


def sm(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def f1(y, p):
    return 100 * f1_score(y, p, average="macro", labels=np.union1d(np.unique(y), np.unique(p)), zero_division=0)


def mg(p):
    s = np.sort(p, 1)
    return s[:, -1] - s[:, -2]


def thr_for(m, rate):
    k = int(round(rate * len(m)))
    s = np.sort(m)
    if k <= 0:
        return -np.inf
    if k >= len(s):
        return np.inf
    return (s[k - 1] + s[k]) / 2


def hull(points, c):
    """Upper envelope of (cost, f1) points at cost c by linear interpolation; returns f1, lo, hi, lam."""
    best = (-np.inf, None, None, 0.0)
    for a, (ca, fa) in points.items():
        if ca <= c and fa > best[0]:
            best = (fa, a, a, 0.0)
        for b, (cb, fb) in points.items():
            if ca <= c < cb:
                lam = (c - ca) / (cb - ca)
                v = (1 - lam) * fa + lam * fb
                if v > best[0]:
                    best = (v, a, b, lam)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=Path, required=True)
    ap.add_argument("--d", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    a = ap.parse_args()
    R, D = a.r, a.d
    zc = {s: np.load(R / "scores" / D / f"mn10_w8a8_s{s}.npz") for s in a.seeds}
    za = {s: np.load(R / "scores" / D / f"ast_w4g128_s{s}.npz") for s in a.seeds}
    z0 = zc[a.seeds[0]]
    y, te = z0["y"].astype(int), z0["test_idx"].astype(int)
    folds = sorted(int(k[7:]) for k in z0.files if k.startswith("iv_idx_"))
    tp = np.load(R / "emb" / D / "ast_all" / "time_patches.npy").astype(float)
    L = np.load(R / "emb" / D / "mn_all" / "lengths__clean.npy").astype(float)
    N = 12 * tp + 2
    lin = 12 * N * 12 * 768 ** 2
    att = 12 * N ** 2 * 768
    ca = lin * 32 + lin / 128 * 1024 + att * 64 + att * 128
    side = json.loads((R / "emb" / D / "mn_all" / "mn10_w8a8__clean.json").read_text())
    cc = (side["frontend_bitops_mean"] + side["backbone_bitops_rescale1024_mean"]) * L / L.mean()
    keys = [(s, f) for s in a.seeds for f in folds]
    out = {}
    # scaler choice
    sc = {}
    for nm, Z in (("cheap", zc), ("ast", za)):
        m = {k: np.mean([f1(y[Z[s][f"iv_idx_{f}"]], Z[s][f"iv__{f}__clean__{k}"].argmax(1)) for s, f in keys]) for k in ("fp32fit", "refit")}
        sc[nm] = "refit" if m["refit"] > m["fp32fit"] else "fp32fit"
        out[f"scaler_{nm}"] = {**m, "chosen": sc[nm]}
    # budget choice on clean inner validation
    ivd = {(s, f): (zc[s][f"iv_idx_{f}"].astype(int), sm(zc[s][f"iv__{f}__clean__{sc['cheap']}"].astype(float)),
                    sm(za[s][f"iv__{f}__clean__{sc['ast']}"].astype(float))) for s, f in keys}
    pts = {}
    for nm in ("mn", "ast", "avg"):
        fs, cs = [], []
        for (s, f), (iv, pc, pa) in ivd.items():
            p = {"mn": pc, "ast": pa, "avg": pc + pa}[nm]
            fs.append(f1(y[iv], p.argmax(1)))
            cs.append({"mn": cc[iv], "ast": ca[iv], "avg": cc[iv] + ca[iv]}[nm].mean())
        pts[nm] = (np.mean(cs), np.mean(fs))
    gains = {}
    for b, rate in BUD.items():
        fs, cs = [], []
        for (s, f), (iv, pc, pa) in ivd.items():
            e = mg(pc) < thr_for(mg(pc), rate)
            fs.append(f1(y[iv], np.where(e, (pc + pa).argmax(1), pc.argmax(1))))
            cs.append(np.mean(cc[iv] + e * ca[iv]))
        gains[b] = np.mean(fs) - hull(pts, np.mean(cs))[0]
    bstar = sorted(gains, key=lambda b: (-round(gains[b], 9), BUD[b]))[0]
    out["budget_gains_iv"], out["budget"] = gains, bstar
    meta = json.loads((R / "tables" / D / "verdict.json").read_text())
    b_use = meta["design"]["budget"]
    out["budget_used_for_test"] = b_use
    nfit = {f: int(z0[f"n_fit_{f}"]) + len(z0[f"iv_idx_{f}"]) for f in folds}
    ratio = np.mean([len(te) / nfit[f] for f in folds])
    yt = y[te]
    for cond in ("field", "clean"):
        F = {k: [] for k in ("mn", "ast", "avg", "cas")}
        C = {k: [] for k in ("mn", "ast", "avg", "cas")}
        rate = []
        for s, f in keys:
            iv, pci, _ = ivd[(s, f)]
            t = thr_for(mg(pci), BUD[b_use])
            pc = sm(zc[s][f"test__{f}__{cond}__{sc['cheap']}"].astype(float))
            pa = sm(za[s][f"test__{f}__{cond}__{sc['ast']}"].astype(float))
            e = mg(pc) < t
            pr = {"mn": pc.argmax(1), "ast": pa.argmax(1), "avg": (pc + pa).argmax(1)}
            pr["cas"] = np.where(e, pr["avg"], pr["mn"])
            for k in pr:
                F[k].append(f1(yt, pr[k]))
            C["mn"].append(cc[te].mean()); C["ast"].append(ca[te].mean()); C["avg"].append((cc[te] + ca[te]).mean())
            C["cas"].append(np.mean(cc[te] + e * ca[te]))
            rate.append(e.mean())
        F = {k: np.array(v).reshape(len(a.seeds), len(folds)) for k, v in F.items()}
        P = {k: (np.mean(C[k]), F[k].mean()) for k in ("mn", "ast", "avg")}
        h, lo, hi, lam = hull(P, np.mean(C["cas"]))
        H = (1 - lam) * F[lo] + lam * F[hi]
        dfold = (F["cas"] - H).mean(0)
        k = len(folds)
        tnb = dfold.mean() / np.sqrt((1 / k + ratio) * dfold.var(ddof=1))
        out[cond] = {"mn": F["mn"].mean(), "ast": F["ast"].mean(), "avg": F["avg"].mean(), "cascade": F["cas"].mean(), "rate": float(np.mean(rate)),
                     "ast_minus_mn_folds": (F["ast"] - F["mn"]).mean(0).round(3).tolist(), "ast_minus_mn": float((F["ast"] - F["mn"]).mean()),
                     "hull": f"{lo}|{hi}|{lam:.4f}", "d_hull_linear": float(dfold.mean()), "fold_d": dfold.round(3).tolist(),
                     "p_nb": float(2 * stats.t.sf(abs(tnb), k - 1))}
    out["analysis_verdict"] = {"precondition": meta["precondition"]["value"], "H1_d": meta["H1"]["d"], "H1_p": meta["H1"]["p_nb"],
                               "H2_d": meta["H2"]["d"], "H2_p": meta["H2"]["p_nb"], "budget": meta["design"]["budget"]}
    (R / "tables" / D / "recompute.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
