"""Idea 5b, step 3: Holm over the full family, pass per cost view, the GO rule, best points and
the fix rate g each needed against the one it achieved.

Family: every (dataset, condition, head, A, B, rule, point) computed by hull.py. Holm is applied within
each cost view over the whole family to the fold-level t-test p value (seeds averaged within each
fold). A point passes a view when its mean difference against the hull is positive and its Holm p is
below 0.05; it counts when it passes in bitops and in rescale (an intersection of two tests, each
at level 0.05, needs no further correction). bitsread is reported only.

GO: some single (head, A, B, rule, point) counts on clean and on the field mixture of ESC-50 (50 classes),
ESC-50 (5 classes) and UrbanSound8K.

g (F1 units): achieved = (F1 cascade - F1 of A alone) / f; needed against the hull = (hull F1 -
F1 of A) / f; needed against time-sharing of A and B = (F1 B - F1 A) * b / (b - a), with f the
test escalation rate and a, b the per-clip bit-ops of the A and B passes with the head. net_fix_acc
is (clips fixed - clips broken) / clips escalated, pooled over (seed, fold).

    python verdict.py --out OUT
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import VIEWS, holm, static_cost

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
DS = ["esc50", "esc5", "urban"]
KEY = ["dataset", "cond", "head", "A", "B", "rule", "point"]
TUP = ["head", "A", "B", "rule", "point"]
GO_CELLS = [(d, c) for d in DS for c in ("clean", "field")]

R = pd.concat([pd.read_csv(a.out / d / "points.csv", dtype={"head": str}) for d in DS], ignore_index=True)
R["p_fold"] = R.p_fold.fillna(1.0)
R["p_pair"] = R.p_pair.fillna(1.0)
family = {v: int((R.view == v).sum()) for v in VIEWS}
assert len(set(family.values())) == 1
N = family["bitops"]
for v in VIEWS:
    m = R.view == v
    R.loc[m, "holm_fold"] = holm(R.loc[m, "p_fold"].to_numpy())
    R.loc[m, "holm_pair"] = holm(R.loc[m, "p_pair"].to_numpy())
R["pass"] = (R.diff_pts > 0) & (R.holm_fold < 0.05)
R["raw_pass"] = (R.diff_pts > 0) & (R.p_fold < 0.05)
R.to_csv(a.out / "points_all.csv", index=False)

W = R.set_index(KEY + ["view"])[["diff_pts", "p_fold", "holm_fold", "pass", "raw_pass", "cost", "hull_f1", "f1",
                                 "hull_lo", "hull_hi", "lam"]].unstack("view")
W.columns = [f"{x}_{v}" for x, v in W.columns]
W = W.reset_index()
W["f1"] = W.f1_bitops
W["counts"] = W.pass_bitops.astype(bool) & W.pass_rescale.astype(bool)
W["raw_counts"] = W.raw_pass_bitops.astype(bool) & W.raw_pass_rescale.astype(bool)
W["min_diff_i_ii"] = W[["diff_pts_bitops", "diff_pts_rescale"]].min(1)
W.to_csv(a.out / "points_wide.csv", index=False)

# ---- counts per dataset and condition ---------------------------------------------------------
cnt = W.groupby(["dataset", "cond"]).agg(points=("f1", "size"),
                                         pos_bitops=("diff_pts_bitops", lambda x: int((x > 0).sum())),
                                         pos_rescale=("diff_pts_rescale", lambda x: int((x > 0).sum())),
                                         pos_both=("min_diff_i_ii", lambda x: int((x > 0).sum())),
                                         raw_both=("raw_counts", "sum"),
                                         pass_bitops=("pass_bitops", "sum"), pass_rescale=("pass_rescale", "sum"),
                                         pass_bitsread=("pass_bitsread", "sum"), counts=("counts", "sum"),
                                         best_diff_bitops=("diff_pts_bitops", "max"),
                                         best_diff_rescale=("diff_pts_rescale", "max"),
                                         best_min_diff=("min_diff_i_ii", "max")).reset_index()
cnt.to_csv(a.out / "counts.csv", index=False)

# ---- GO rule ------------------------------------------------------------------------------------
G = W[[(d, c) in GO_CELLS for d, c in zip(W.dataset, W.cond)]].copy()
G["cell"] = G.dataset + "/" + G.cond
go = G.set_index(TUP + ["cell"])[["counts", "raw_counts", "min_diff_i_ii", "diff_pts_bitops",
                                  "diff_pts_rescale"]].unstack("cell")
go.columns = [f"{x}|{c}" for x, c in go.columns]
go = go.reset_index()
cells = [f"{d}/{c}" for d, c in GO_CELLS]
go["n_cells_counting"] = sum(go[f"counts|{c}"].astype(bool) for c in cells)
go["n_cells_raw"] = sum(go[f"raw_counts|{c}"].astype(bool) for c in cells)
go["n_cells_positive"] = sum((go[f"min_diff_i_ii|{c}"] > 0) for c in cells)
go["worst_min_diff"] = go[[f"min_diff_i_ii|{c}" for c in cells]].min(1)
go["worst_diff_bitops"] = go[[f"diff_pts_bitops|{c}" for c in cells]].min(1)
go = go.sort_values(["n_cells_counting", "worst_min_diff"], ascending=False)
go.to_csv(a.out / "go_tuples.csv", index=False)
GO = bool((go.n_cells_counting == len(cells)).any())

# ---- best points and g ----------------------------------------------------------------------
stat, casc = {}, {}
for d in DS:
    s = pd.read_csv(a.out / d / "static_pairs.csv", dtype={"head": str})
    stat[d] = s.groupby(["cond", "head", "config"]).agg(f1=("f1", "mean"), acc=("acc", "mean"))
    c = pd.read_csv(a.out / d / "cascade_pairs.csv.gz", dtype={"head": str})
    casc[d] = c.groupby(["cond", "head", "A", "B", "rule", "point"]).agg(
        rate=("rate", "mean"), acc=("acc", "mean"), n_esc=("n_esc", "sum"), n_fixed=("n_fixed", "sum"),
        n_broken=("n_broken", "sum"), match_reached=("match_reached", "mean"))
n_cls = {"esc50": 50, "esc5": 5, "urban": 10}


def g_row(r):
    d = r.dataset
    sa, sb = stat[d].loc[(r.cond, r["head"], r.A)], stat[d].loc[(r.cond, r["head"], r.B)]
    cc = casc[d].loc[(r.cond, r["head"], r.A, r.B, r.rule, r.point)]
    av, bv = static_cost(r.A, r["head"], n_cls[d], "bitops"), static_cost(r.B, r["head"], n_cls[d], "bitops")
    f = cc.rate
    return dict(rate=f, f1_A=sa.f1, f1_B=sb.f1, acc_A=sa.acc, acc_B=sb.acc, acc=cc.acc,
                g_achieved=(r.f1 - sa.f1) / f if f > 0 else np.nan,
                g_needed_hull=(r.hull_f1_bitops - sa.f1) / f if f > 0 else np.nan,
                g_needed_hull_rescale=(r.hull_f1_rescale - sa.f1) / f if f > 0 else np.nan,
                g_needed_AB=(sb.f1 - sa.f1) * bv / (bv - av),
                net_fix_acc=(cc.n_fixed - cc.n_broken) / cc.n_esc if cc.n_esc else np.nan,
                net_fix_needed_AB_acc=(sb.acc - sa.acc) * bv / (bv - av),
                fixed=int(cc.n_fixed), broken=int(cc.n_broken), escalated=int(cc.n_esc))


best = []
for (d, cnd), g in W.groupby(["dataset", "cond"]):
    for crit in ("diff_pts_bitops", "min_diff_i_ii"):
        r = g.sort_values(crit, ascending=False).iloc[0]
        best.append(dict(dataset=d, cond=cnd, chosen_by=crit, **{k: r[k] for k in TUP}, f1=r.f1,
                         cost_T=r.cost_bitops / 1e12, hull_f1=r.hull_f1_bitops,
                         hull=f"{r.hull_lo_bitops} / {r.hull_hi_bitops} ({r.lam_bitops:.2f})",
                         diff_bitops=r.diff_pts_bitops, p_fold_bitops=r.p_fold_bitops, holm_bitops=r.holm_fold_bitops,
                         diff_rescale=r.diff_pts_rescale, p_fold_rescale=r.p_fold_rescale,
                         holm_rescale=r.holm_fold_rescale, diff_bitsread=r.diff_pts_bitsread,
                         p_fold_bitsread=r.p_fold_bitsread, **g_row(r)))
B = pd.DataFrame(best)
B.to_csv(a.out / "best_points.csv", index=False)

def tuple_table(t, label):
    tr = []
    for d, cnd in GO_CELLS:
        r = W[(W.dataset == d) & (W.cond == cnd) & (W["head"] == t[0]) & (W.A == t[1]) & (W.B == t[2])
              & (W.rule == t[3]) & (W.point == t[4])].iloc[0]
        tr.append(dict(which=label, dataset=d, cond=cnd, **{k: r[k] for k in TUP}, f1=r.f1,
                       cost_T=r.cost_bitops / 1e12, hull_f1=r.hull_f1_bitops,
                       hull=f"{r.hull_lo_bitops} / {r.hull_hi_bitops} ({r.lam_bitops:.2f})",
                       diff_bitops=r.diff_pts_bitops, p_fold_bitops=r.p_fold_bitops,
                       holm_bitops=r.holm_fold_bitops, diff_rescale=r.diff_pts_rescale,
                       p_fold_rescale=r.p_fold_rescale, holm_rescale=r.holm_fold_rescale,
                       diff_bitsread=r.diff_pts_bitsread, **g_row(r)))
    return pd.DataFrame(tr)


# the single tuple closest to GO (largest worst-cell difference over the 6 GO cells, both views)
top = go.sort_values("worst_min_diff", ascending=False).iloc[0]
top_t = tuple(top[k] for k in TUP)
T = tuple_table(top_t, "closest to GO")
T.to_csv(a.out / "best_tuple.csv", index=False)

# the simple theory's candidate: a 3-bit A escalating to the next hull point, a 4-bit B
TH_A, TH_B = ["w3g64", "n3g64"], ["w4", "n4", "w4g128", "n4g128"]
th = go[go.A.isin(TH_A) & go.B.isin(TH_B)].sort_values("worst_diff_bitops", ascending=False).iloc[0]
th_t = tuple(th[k] for k in TUP)
TT = tuple_table(th_t, "best 3-bit -> 4-bit tuple")
cellbest = []
WT = W[W.A.isin(TH_A) & W.B.isin(TH_B)]
for d, cnd in GO_CELLS:
    r = WT[(WT.dataset == d) & (WT.cond == cnd)].sort_values("diff_pts_bitops", ascending=False).iloc[0]
    cellbest.append(tuple_table(tuple(r[k] for k in TUP), "best 3-bit -> 4-bit point in this cell").pipe(
        lambda x: x[(x.dataset == d) & (x.cond == cnd)]))
pd.concat([TT] + cellbest, ignore_index=True).to_csv(a.out / "theory_pairs.csv", index=False)

# smallest Holm step: the first rejection needs p < 0.05 / N; the t a fold-level test needs for it
need = {nf: float(stats.t.isf(0.025 / N, nf - 1)) for nf in (5, 10)}
summary = dict(GO=GO, family_size_per_view=N, views=VIEWS, n_pairs_AB=int(W.groupby(["A", "B"]).ngroups),
               rules=sorted(W.rule.unique()), points=sorted(W.point.unique()),
               conditions={d: sorted(W[W.dataset == d].cond.unique()) for d in DS},
               holm_first_step_alpha=0.05 / N, t_needed_first_step={"5 folds": need[5], "10 folds": need[10]},
               n_counting=int(W.counts.sum()), n_pass_bitops=int(W.pass_bitops.sum()),
               n_pass_rescale=int(W.pass_rescale.sum()), n_pass_bitsread=int(W.pass_bitsread.sum()),
               n_raw_counting=int(W.raw_counts.sum()),
               tuples_positive_in_all_6=int((go.n_cells_positive == 6).sum()),
               tuples_raw_in_all_6=int((go.n_cells_raw == 6).sum()),
               max_cells_positive=int(go.n_cells_positive.max()), max_cells_raw=int(go.n_cells_raw.max()),
               closest_tuple=dict(zip(TUP, top_t)), closest_worst_min_diff=float(top.worst_min_diff),
               theory_tuple=dict(zip(TUP, th_t)), theory_worst_diff_bitops=float(th.worst_diff_bitops),
               smallest_p_fold_positive={v: float(W.loc[W[f"diff_pts_{v}"] > 0, f"p_fold_{v}"].min()) for v in VIEWS},
               smallest_p_fold_positive_go_cells={v: float(G.loc[G[f"diff_pts_{v}"] > 0, f"p_fold_{v}"].min())
                                                  for v in ("bitops", "rescale")})
json.dump(summary, open(a.out / "verdict.json", "w"), indent=1, default=str)
pd.set_option("display.width", 250)
print(json.dumps(summary, indent=1, default=str))
print(cnt.to_string())
print(B.to_string(float_format=lambda v: f"{v:.4g}"))
print(T.to_string(float_format=lambda v: f"{v:.4g}"))
print(go.head(15)[TUP + ["n_cells_counting", "n_cells_raw", "n_cells_positive", "worst_min_diff",
                         "worst_diff_bitops"]].to_string())
