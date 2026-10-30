"""Idea 5b, step 2: compare every cascade point with the time-sharing hull of all static systems,
in three cost views.

For a point with mean cost c (over its (seed, fold) pairs), the comparator is the best random mix
of two static systems (any config scored in that condition, either head) at mean cost c, or one
static system at cost <= c: the pair (lo, hi) and share lam on hi that maximise the linear
interpolation of the systems' mean F1. Its F1 per (seed, fold) is the exact expected macro F1 of
the mix (every clip independently on hi with probability lam), from that seed's two systems.

Cost views: bitops (Stage 2), rescale (bit-ops plus every weight-scale rescale, per group or per
row, at 32 x 32), bitsread (weight bits read per clip; nested child reads k + 1 planes).

Writes OUT/<dataset>/points.csv: one row per (cond, head, A, B, rule, point, view).

    python hull.py --dataset esc50 --out OUT
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import VIEWS, cascade_cost, mix_expected_f1, static_cost

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
d = a.out / a.dataset
meta = json.load(open(d / "meta.json"))
n_cls = meta["n_cls"]
S = pd.read_csv(d / "static_pairs.csv", dtype={"head": str})
C = pd.read_csv(d / "cascade_pairs.csv.gz", dtype={"head": str})
P = np.load(d / "static_preds.npz")
y, tf = P["y"], P["tf"]
keys_sf = sorted({(int(s), int(f)) for s, f in zip(S.seed, S.fold)})
t0 = time.time()

for v in VIEWS:
    S[f"cost_{v}"] = [static_cost(c, h, n_cls, v) for c, h in zip(S.config, S["head"])]
    cc = np.empty(len(C))
    for (h, A, B), g in C.groupby(["head", "A", "B"]).groups.items():
        cc[g] = cascade_cost(A, B, C.rate.to_numpy()[g], n_cls, v, h)
    C[f"cost_{v}"] = cc

rows = []
for cond in meta["conds"]:
    s = S[S.cond == cond]
    sysm = s.groupby(["config", "head"]).agg(f1=("f1", "mean"), n=("f1", "size"),
                                             **{f"cost_{v}": (f"cost_{v}", "first") for v in VIEWS}).reset_index()
    assert (sysm.n == len(keys_sf)).all()
    sys_f1 = {(r.config, r["head"]): s[(s.config == r.config) & (s["head"] == r["head"])]
              .set_index(["seed", "fold"]).f1 for _, r in sysm.iterrows()}
    cg = C[C.cond == cond]
    grp = cg.groupby(["head", "A", "B", "rule", "point"])
    for v in VIEWS:
        cost = sysm[f"cost_{v}"].to_numpy()
        f1m = sysm.f1.to_numpy()
        pts = grp.agg(f1=("f1", "mean"), cost=(f"cost_{v}", "mean")).reset_index()
        # hull selection per point (vectorised over points)
        c = pts.cost.to_numpy()[:, None, None]
        lo_ok = cost[None, :, None] <= c * (1 + 1e-12)
        hi_ok = cost[None, None, :] > c
        lam = np.where(hi_ok, (c - cost[None, :, None]) / np.where(hi_ok, cost[None, None, :] - cost[None, :, None], 1), 0)
        val = (1 - lam) * f1m[None, :, None] + lam * f1m[None, None, :]
        single = np.where(lo_ok[:, :, 0], f1m[None, :], -np.inf)          # lo alone
        val = np.where(lo_ok & hi_ok, val, -np.inf)
        best_pair = val.reshape(len(pts), -1).argmax(1)
        best_pair_v = val.reshape(len(pts), -1).max(1)
        best_single = single.argmax(1)
        use_single = single.max(1) >= best_pair_v
        lo_i = np.where(use_single, best_single, best_pair // len(sysm))
        hi_i = np.where(use_single, best_single, best_pair % len(sysm))
        lam_v = np.where(use_single, 0.0, lam.reshape(len(pts), -1)[np.arange(len(pts)), best_pair])
        lin_v = np.where(use_single, single.max(1), best_pair_v)
        # exact expected F1 of each mix per (seed, fold), batched over points sharing (lo, hi)
        hull = np.zeros((len(pts), len(keys_sf)))
        for (li, hi_), idxs in pd.Series(range(len(pts))).groupby([lo_i, hi_i]).groups.items():
            idxs = np.asarray(idxs)
            lo_sys = (sysm.config[li], sysm["head"][li])
            hi_sys = (sysm.config[hi_], sysm["head"][hi_])
            for j, (seed, fold) in enumerate(keys_sf):
                if li == hi_:
                    hull[idxs, j] = sys_f1[lo_sys].loc[(seed, fold)]
                    continue
                m = tf == fold
                plo = P[f"{seed}|{lo_sys[1]}|{lo_sys[0]}|{cond}"][m].astype(int)
                phi = P[f"{seed}|{hi_sys[1]}|{hi_sys[0]}|{cond}"][m].astype(int)
                hull[idxs, j] = mix_expected_f1(y[m], plo, phi, lam_v[idxs], n_cls)
        # paired statistics
        casc = np.zeros_like(hull)
        for i, ((h, A, B, rule, pt), g) in enumerate(grp):
            assert (h, A, B, rule, pt) == tuple(pts.loc[i, ["head", "A", "B", "rule", "point"]])
            casc[i] = g.set_index(["seed", "fold"]).f1.loc[keys_sf].to_numpy()
        diff = casc - hull
        folds = np.array([k[1] for k in keys_sf])
        uf = np.unique(folds)
        fd = np.stack([diff[:, folds == f].mean(1) for f in uf], 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            p_pair = stats.ttest_1samp(diff, 0, axis=1).pvalue
            p_fold = stats.ttest_1samp(fd, 0, axis=1).pvalue
        for i in range(len(pts)):
            r = pts.iloc[i]
            rows.append(dict(dataset=a.dataset, cond=cond, head=r["head"], A=r.A, B=r.B, rule=r.rule, point=r.point, view=v,
                             f1=r.f1, cost=r.cost, hull_lo="+".join(sysm.loc[lo_i[i], ["config", "head"]]),
                             hull_hi="+".join(sysm.loc[hi_i[i], ["config", "head"]]), lam=lam_v[i],
                             hull_f1_lin=lin_v[i], hull_f1=hull[i].mean(), diff_pts=100 * diff[i].mean(),
                             n_pairs=diff.shape[1], n_pos_pairs=int((diff[i] > 0).sum()), p_pair=p_pair[i],
                             n_folds=len(uf), n_pos_folds=int((fd[i] > 0).sum()), p_fold=p_fold[i],
                             fold_diffs_pts=json.dumps((100 * fd[i]).round(3).tolist())))
    print(f"{a.dataset} {cond} done ({time.time() - t0:.0f}s)", flush=True)

R = pd.DataFrame(rows)
R.to_csv(d / "points.csv", index=False)
print("points", len(R), "max |hull exact - linear| (F1 pts):", 100 * (R.hull_f1 - R.hull_f1_lin).abs().max())
