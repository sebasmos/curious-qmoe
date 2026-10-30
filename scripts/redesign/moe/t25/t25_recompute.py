"""T25 independent recompute of H1-wide and H2-wide from the raw score files (plan C7). Own code for every step
except the AST and MN10 per-clip costs (T21's audited per_clip_costs) and the CNN MAC hooks (the model definitions):
own softmax, own scaler rule, own threshold at the budget rate, sklearn macro F1, own hull over all 17 systems,
the hull's expected F1 by Monte Carlo time-sharing (2,000 draws), own Nadeau-Bengio test. Compares with
tables/<d>/verdict.json and writes tables/<d>/recompute.json.

    python t25_recompute.py --r R25 --base R21 --d d1
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "t21"))
import t21_analyze as T  # noqa: E402  only for T21's audited per-clip costs and the head network builder
from QWave import efficientat_dymn as DY  # noqa: E402
from QWave import efficientat_mn as M  # noqa: E402

CNN5 = ["mn04", "mn05", "mn10", "mn20", "dymn04"]
NEW = ["mn04", "mn05", "mn20", "dymn04"]
WEIGHTS = Path.home() / "Documents/qmoe-weights/efficientat_release-v0.0.1"
WFILE = {"mn04": "mn04_as_mAP_432.pt", "mn05": "mn05_as_mAP_443.pt", "mn20": "mn20_as_mAP_478.pt", "dymn04": "dymn04_as.pt"}
BUD = {"b05": 0.05, "b10": 0.10, "b20": 0.20, "b30": 0.30, "b50": 0.50}


def sm(z):
    z = z.astype(np.float64)
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def mf1(y, p):
    return 100 * f1_score(y, p, average="macro", zero_division=0)


def marg(p):
    s = np.sort(p, 1)
    return s[:, -1] - s[:, -2]


def thr_at(m, rate):
    """Threshold t (escalate when margin < t) whose inner-validation rate is closest to `rate` (ties: lower rate)."""
    u = np.unique(m)
    cand = np.concatenate([[-np.inf], (u[:-1] + u[1:]) / 2, [np.inf]])
    r = np.array([(m < t).mean() for t in cand])
    best = min(range(len(cand)), key=lambda i: (abs(r[i] - rate), r[i]))
    return cand[best]


def head_bitops(dim, K):
    from omegaconf import OmegaConf
    net = T.build_model("esc", dim, K, OmegaConf.load(T.ROOT / "config/esc50.yaml"))
    return sum(m.in_features * m.out_features for m in net.modules() if isinstance(m, torch.nn.Linear)) * 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=Path, required=True)
    ap.add_argument("--base", type=Path, default=Path.home() / "Documents/qmoe-results/2026-10-26_1408_moe-t21_code-731570e")
    ap.add_argument("--d", default="d1")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    R, B, D, seeds = a.r, a.base, a.d, a.seeds
    V = json.loads((R / "tables" / D / "verdict.json").read_text())
    V21 = json.loads((B / "tables" / D / "verdict.json").read_text())
    budget = BUD[V21["design"]["budget"]]
    files = {"mn10": (B, "pool_mn10_w8a8"), "ast": (B, "pool_ast_w4g128"), **{m: (R, f"pool_{m}_w8a8") for m in NEW}}
    Z = {t: {s: np.load(r / "scores" / D / f"{stem}_s{s}.npz") for s in seeds} for t, (r, stem) in files.items()}
    y = Z["mn10"][seeds[0]]["y"].astype(int)
    fold = Z["mn10"][seeds[0]]["fold"].astype(int)
    folds = sorted(set(fold) - {-1})
    K = len(np.unique(y))
    # scaler: T21's for its two tiers (fixed), own rule for the new ones
    sc21 = pd.read_csv(B / "tables" / D / "primary_scaler_choice.csv").set_index("tier").chosen
    scaler = {"mn10": sc21["cheap"], "ast": sc21["ast"]}
    for t in NEW:
        v = {k: np.mean([mf1(y[Z[t][s][f"iv_idx_{f}"]], Z[t][s][f"iv__{f}__clean__{k}"].argmax(1)) for s in seeds for f in folds])
             for k in ("fp32fit", "refit")}
        scaler[t] = "refit" if v["refit"] > v["fp32fit"] else "fp32fit"
    # per-clip costs
    c21, _ = T.per_clip_costs(B, D, K)
    cost = {"mn10": c21["priced"][0], "ast": c21["priced"][1]}
    L = np.load(B / "emb" / D / "mn_pool" / "lengths__clean.npy")
    fr = np.array([M.frames_of(M.n32_of(int(x))) for x in L])
    fe = np.array([M.frontend_macs(int(x)) for x in L]) * 1024.0
    for t in NEW:
        lib = DY if t.startswith("dymn") else M
        b = DY.load_dymn(t, WEIGHTS / WFILE[t]) if t.startswith("dymn") else M.load_mn(t, WEIGHTS / WFILE[t])
        with contextlib.redirect_stdout(io.StringIO()):
            hb = head_bitops(b.emb_dim, K)
        bb = {}
        for f_ in np.unique(fr):
            q = lib.feature_macs(b, int(f_))
            bb[f_] = q["macs"] * 64 + q["out_elems"] * 1024 + q.get("aggregate_macs", 0.0) * 1024
        cost[t] = fe + np.array([bb[f_] for f_ in fr]) + hb
    pairs = [f"{x}+{z}" for x, z in itertools.combinations(CNN5, 2)]
    systems = ["mn10", "ast", "avg", *NEW, *pairs]
    rng = np.random.default_rng(7)
    out = {"scalers": scaler}
    ratio = None
    for bed in ("field", "clean"):
        F, Cst, PR = {}, {}, {}
        for s in seeds:
            for f in folds:
                te = np.flatnonzero(fold == f)
                P = {t: sm(Z[t][s][f"test__{bed}__{scaler[t]}"][te]) for t in files}
                iv = Z["mn10"][s][f"iv_idx_{f}"]
                t_ = thr_at(marg(sm(Z["mn10"][s][f"iv__{f}__clean__{scaler['mn10']}"])), budget)
                esc = marg(P["mn10"]) < t_
                pr = {t: P[t].argmax(1) for t in P}
                pr["avg"] = (P["mn10"] + P["ast"]).argmax(1)
                pr["cascade"] = np.where(esc, pr["avg"], pr["mn10"])
                cs = {t: cost[t][te].mean() for t in files}
                cs["avg"] = cs["mn10"] + cs["ast"]
                cs["cascade"] = float((cost["mn10"][te] + esc * cost["ast"][te]).mean())
                for p_ in pairs:
                    x, z = p_.split("+")
                    pr[p_] = (P[x] + P[z]).argmax(1)
                    cs[p_] = cs[x] + cs[z]
                PR[(s, f)] = (y[te], pr)
                for k_, v_ in pr.items():
                    F[(k_, s, f)] = mf1(y[te], v_)
                    Cst[(k_, s, f)] = cs[k_]
        keys = [(s, f) for s in seeds for f in folds]
        mean = {k_: np.mean([F[(k_, s, f)] for s, f in keys]) for k_ in systems + ["cascade"]}
        mc = {k_: np.mean([Cst[(k_, s, f)] for s, f in keys]) for k_ in systems + ["cascade"]}
        c = mc["cascade"]
        best = (-1, None, None, 0.0)
        for lo in systems:  # own hull: best single at or below c, or a line to any system above c
            if mc[lo] > c * (1 + 1e-12):
                continue
            if mean[lo] > best[0]:
                best = (mean[lo], lo, lo, 0.0)
            for hi in systems:
                if mc[hi] > c * (1 + 1e-12):
                    lam = (c - mc[lo]) / (mc[hi] - mc[lo])
                    v = (1 - lam) * mean[lo] + lam * mean[hi]
                    if v > best[0]:
                        best = (v, lo, hi, lam)
        _, lo, hi, lam = best
        d = {}
        for s, f in keys:
            yt, pr = PR[(s, f)]
            if lam == 0:
                h = mf1(yt, pr[lo])
            else:
                draws = rng.random((2000, len(yt))) < lam
                h = np.mean([mf1(yt, np.where(dr, pr[hi], pr[lo])) for dr in draws])
            d[(s, f)] = F[("cascade", s, f)] - h
        fm = np.array([np.mean([d[(s, f)] for s in seeds]) for f in folds])
        n_te = np.array([(fold == f).sum() for f in folds])
        ratio = float(np.mean(n_te / (len(y) - n_te)))
        k = len(fm)
        se = np.sqrt((1 / k + ratio) * fm.var(ddof=1))
        tval = fm.mean() / se
        p = float(2 * stats.t.sf(abs(tval), k - 1))
        h_ = stats.t.ppf(0.975, k - 1) * se
        out[bed] = {"cascade_f1": mean["cascade"], "cascade_cost": c, "hull": f"{lo}|{hi}|{lam:.4f}", "d": float(fm.mean()),
                    "nb_ci95": [float(fm.mean() - h_), float(fm.mean() + h_)], "p_nb": p, "folds_positive": int((fm > 0).sum()),
                    "fold_d": np.round(fm, 3).tolist(),
                    "statics": {k_: [round(mean[k_], 3), float(mc[k_])] for k_ in systems}}
    key = {"field": "H1_wide", "clean": "H2_wide"}
    out["compare"] = {bed: {"analysis_d": V[key[bed]]["d"], "recompute_d": out[bed]["d"], "abs_diff": abs(V[key[bed]]["d"] - out[bed]["d"]),
                            "analysis_hull": V[key[bed]]["hull"], "recompute_hull": out[bed]["hull"],
                            "analysis_p": V[key[bed]]["p_nb"], "recompute_p": out[bed]["p_nb"]} for bed in ("field", "clean")}
    out["scalers_match"] = scaler == V["scalers"]
    (R / "tables" / D / "recompute.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k_: out[k_] for k_ in ("compare", "scalers_match")}, indent=1, default=float))


if __name__ == "__main__":
    main()
