"""Checks 1, 4, 5, 6 and 7 from recompute.py's tables.

1. Match against the claim: every cascade cell and static cell, then the primary analysis (pick on
   inner validation, test against the hull, fold-level t, Holm over 3) and the nested variants.
4. Selection-rule sensitivity (view i): runner-up, best inner F1 under a fixed inner cost, one design
   for all tasks, W8 tier only, and the spread of test results over every design with a positive
   inner-validation gain.
5. The primary analysis in every cost view, with the claim's picks held fixed and with re-selection
   (claim's design space, and with per-channel and 3-bit AST tiers added).
6. The claim's picks against the hull built with ensembles added.
7. Statistics for the primary designs: Shapiro-Wilk on fold deltas, exact sign-flip permutation p,
   bootstrap and t intervals over folds, a Nadeau-Bengio corrected t, pair-level tests.

    python verdict.py OUT
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

H = Path.home() / "Documents"
import os; RD = Path(os.environ.get("RESULTS_DIR", str(H / "results")))  # results root (archive edit: path only)
I6 = RD / "2026-10-22_0953_redesign-idea6-stage1_code-ec70260/analysis"
TASKS = ["esc50", "esc5", "urban"]
DES = ["cheap", "tier", "rule", "point"]
N_TEST_TRAIN = {"esc50": (400, 1436), "esc5": (400, 1436), "urban": (873, 7055)}  # mean test and fit sizes
OUT = Path(sys.argv[1])
R = OUT / "recompute"


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p, kind="stable")
    adj, run = np.empty(len(p)), 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


def load(task):
    d = pd.read_csv(R / task / "designs.csv", dtype={"cheap": str})
    return d


def pick(d, key="gain_iv"):
    return d.sort_values([key, "cost_iv"], ascending=[False, True]).iloc[0]


def name(r):
    return f"{r.cheap} -> {r.tier}, {r.rule}, {r.point}"


def nested(d, col):
    """Each fold uses the design with the largest per-fold inner gain (col); returns fold diffs."""
    g = {}
    for _, r in d.iterrows():
        for f, v in json.loads(r[col]).items():
            if f not in g or v > g[f][0]:
                g[f] = (v, r)
    folds = sorted(g, key=int)
    diffs = [json.loads(g[f][1].fold_diffs)[i] for i, f in enumerate(folds)]
    return np.array(diffs), {f: name(g[f][1]) for f in folds}


def primary(view, set_="claim", space="claim", fixed=None):
    rows = []
    for t in TASKS:
        d = load(t)
        d = d[(d.view == view) & (d.set == set_)]
        if space == "claim":
            d = d[d.claim_space]
        if fixed is not None:
            f = fixed[t]
            r = d[(d.cheap == f[0]) & (d.tier == f[1]) & (d.rule == f[2]) & (d.point == f[3])].iloc[0]
        else:
            r = pick(d)
        nd, npk = nested(d, "gain_iv_fold")
        sd, spk = nested(d, "gain_iv_fold_strict")
        rows.append({"task": t, "design": name(r), "f1": r.f1, "cost_T": r.cost / 1e12, "rate": r.rate,
                     "hull_f1": r.hull_f1, "hull": f"{r.hull_lo} + {r.lam:.2f} x {r.hull_hi}" if r.lam else r.hull_lo,
                     "gain_iv": r.gain_iv, "diff": r["diff"], "fold_diffs": r.fold_diffs, "n_pos_folds": r.n_pos_folds,
                     "p_t_fold": r.p_t_fold, "p_w_fold": r.p_w_fold, "p_t_pair": r.p_t_pair, "p_w_pair": r.p_w_pair,
                     "nested_diff": nd.mean(), "nested_p_t": stats.ttest_1samp(nd, 0).pvalue,
                     "nested_fold_diffs": json.dumps(np.round(nd, 6).tolist()), "nested_picks": json.dumps(npk),
                     "strict_diff": sd.mean(), "strict_p_t": stats.ttest_1samp(sd, 0).pvalue,
                     "strict_picks": json.dumps(spk)})
    P = pd.DataFrame(rows)
    P["p_t_fold_holm3"] = holm(P.p_t_fold)
    P["nested_p_holm3"] = holm(P.nested_p_t)
    P["strict_p_holm3"] = holm(P.strict_p_t)
    P["pass"] = (P["diff"] > 0) & (P.p_t_fold_holm3 < 0.05)
    P["view"], P["set"], P["space"] = view, set_, space
    return P


def main():
    rep = {}
    # ---------------- check 1a: cell-level match
    cell = {}
    for t in TASKS:
        mine = pd.read_csv(R / t / "cascades.csv", dtype={"cheap": str})
        mine = mine[mine.claim_space].rename(columns={"tier": "expensive"})
        theirs = pd.read_csv(I6 / t / "cascades.csv", dtype={"cheap": str})
        m = theirs.merge(mine, on=["cheap", "expensive", "rule", "point", "seed", "fold"], suffixes=("_c", "_a"))
        st_m = pd.read_csv(R / t / "static.csv")
        st_t = pd.read_csv(I6 / t / "static.csv")
        s = st_t.merge(st_m, on=["system", "seed", "fold"], suffixes=("_c", "_a"))
        cell[t] = {"cascade_cells_claim": len(theirs), "matched": len(m),
                   "max_abs_f1": float((m.f1_c - m.f1_a).abs().max()),
                   "max_abs_f1_iv": float((m.f1_iv_c - m.f1_iv_a).abs().max()),
                   "max_abs_rate": float((m.rate_c - m.rate_a).abs().max()),
                   "max_abs_iv_rate": float((m.iv_rate_c - m.iv_rate_a).abs().max()),
                   "static_cells": len(st_t), "static_matched": len(s),
                   "static_max_abs_f1": float((s.f1_c - s.f1_a).abs().max())}
    rep["check1_cells"] = cell
    print(json.dumps(cell, indent=1))

    # ---------------- check 1b: primary, view i and int8
    claim = {v: pd.read_csv(I6 / f"primary_{v}.csv", dtype={"cheap": str}) for v in ("fp32", "int8")}
    prim = {}
    for v, cv in (("i", "fp32"), ("int8", "int8")):
        P = primary(v)
        C = claim[cv]
        P["claim_design"] = [f"{r.cheap} -> {r.expensive}, {r.rule}, {r.point}" for r in C.itertuples()]
        P["claim_diff"], P["claim_p_t_fold"], P["claim_hull_f1"] = C["diff"].values, C.p_t_fold.values, C.hull_f1.values
        P["claim_f1"], P["claim_cost_T"] = C.f1.values, C.cost.values / 1e12
        P["claim_nested_diff"], P["claim_nested_p"] = C.nested_diff.values, C.nested_p_t_fold.values
        prim[v] = P
    rep["check1_primary"] = {v: json.loads(P.to_json(orient="records")) for v, P in prim.items()}
    for v, P in prim.items():
        print(f"\nprimary view {v}")
        print(P[["task", "design", "claim_design", "f1", "claim_f1", "cost_T", "claim_cost_T", "hull_f1", "claim_hull_f1",
                 "diff", "claim_diff", "p_t_fold", "claim_p_t_fold", "p_t_fold_holm3", "nested_diff", "claim_nested_diff",
                 "nested_p_holm3", "strict_diff", "strict_p_holm3", "pass"]].to_string())
    fixed = {r.task: tuple(r.design.replace(" -> ", ",").replace(", ", ",").split(",")) for r in prim["i"].itertuples()}

    # ---------------- check 5: every cost view, fixed picks and re-selection
    views = []
    for v in ["i", "int8", "ii", "int8_ii", "ii_fe", "i_fe32", "ii_fe32", "i_fe32s", "ii_fe32s"]:
        for mode, kw in [("fixed_claim_pick", {"fixed": fixed}), ("reselect_claim_space", {}),
                         ("reselect_with_extra_tiers", {"space": "all"})]:
            P = primary(v, **kw)
            P["mode"] = mode
            views.append(P)
    V = pd.concat(views, ignore_index=True)
    V.to_csv(OUT / "check5_views.csv", index=False)
    print("\ncheck 5 views")
    print(V[["view", "mode", "task", "design", "f1", "cost_T", "hull_f1", "hull", "diff", "p_t_fold", "p_t_fold_holm3",
             "nested_diff", "nested_p_holm3", "pass"]].to_string())

    # ---------------- check 6: hull with ensembles (views i and ii)
    ens = []
    for v in ["i", "ii"]:
        P = primary(v, set_="plus_ens", fixed=fixed)
        ens.append(P)
        P2 = primary(v, set_="plus_ens")
        P2["mode"] = "reselect"
        ens.append(P2)
    E = pd.concat(ens, ignore_index=True)
    E.to_csv(OUT / "check6_ensembles.csv", index=False)
    print("\ncheck 6 ensembles")
    print(E[["view", "task", "design", "f1", "cost_T", "hull_f1", "hull", "diff", "p_t_fold", "p_t_fold_holm3", "pass"]].to_string())

    # ---------------- check 4: selection sensitivity, view i, claim space
    sens = []
    allpos = []
    for t in TASKS:
        d = load(t)
        d = d[(d.view == "i") & (d.set == "claim") & d.claim_space].copy()
        ds = d.sort_values(["gain_iv", "cost_iv"], ascending=[False, True])
        rules = {"claim: max inner gain over hull": ds.iloc[0], "runner-up inner gain": ds.iloc[1],
                 "third inner gain": ds.iloc[2]}
        for cap in (3.0, 3.5, 4.0, 5.0):
            dd = d[d.cost_iv <= cap * 1e12]
            rules[f"max inner F1 at inner cost <= {cap} T"] = pick(dd, "f1_iv")
        w8 = d[d.tier == "w8g64"]
        rules["W8 tier only, max inner gain"] = pick(w8)
        rules["FP32 head, w4g128, ast, b20 (fixed)"] = d[(d.cheap == "esc") & (d.tier == "w4g128") & (d.rule == "ast") & (d.point == "b20")].iloc[0]
        rules["FP32 head, w4g128, ast, b10 (fixed)"] = d[(d.cheap == "esc") & (d.tier == "w4g128") & (d.rule == "ast") & (d.point == "b10")].iloc[0]
        rules["FP32 head, w4g128, ast, b50 (fixed)"] = d[(d.cheap == "esc") & (d.tier == "w4g128") & (d.rule == "ast") & (d.point == "b50")].iloc[0]
        for rn, r in rules.items():
            sens.append({"task": t, "rule_name": rn, "design": name(r), "gain_iv": r.gain_iv, "f1_iv": r.f1_iv,
                         "cost_iv_T": r.cost_iv / 1e12, "f1": r.f1, "cost_T": r.cost / 1e12, "diff": r["diff"],
                         "p_t_fold": r.p_t_fold, "n_pos_folds": r.n_pos_folds, "fold_diffs": r.fold_diffs})
        pos = d[d.gain_iv > 0]
        allpos.append({"task": t, "n_designs": len(d), "n_positive_gain_iv": len(pos),
                       "of_which_positive_test_diff": int((pos["diff"] > 0).sum()),
                       "p_t_fold_below_0.05": int((pos.p_t_fold < 0.05).sum()),
                       "p_t_fold_below_0.0167": int((pos.p_t_fold < 0.05 / 3).sum()),
                       "rank_corr_gain_iv_vs_test_diff": float(stats.spearmanr(d.gain_iv, d["diff"]).statistic)})
    # one design for all three tasks
    dd = {t: load(t) for t in TASKS}
    dd = {t: d[(d.view == "i") & (d.set == "claim") & d.claim_space].set_index(DES) for t, d in dd.items()}
    mean_gain = sum(dd[t].gain_iv for t in TASKS) / 3
    best = mean_gain.idxmax()
    for t in TASKS:
        r = dd[t].loc[best]
        sens.append({"task": t, "rule_name": "one design for all tasks (max mean inner gain)", "design": " ".join(best),
                     "gain_iv": r.gain_iv, "f1_iv": r.f1_iv, "cost_iv_T": r.cost_iv / 1e12, "f1": r.f1,
                     "cost_T": r.cost / 1e12, "diff": r["diff"], "p_t_fold": r.p_t_fold, "n_pos_folds": r.n_pos_folds,
                     "fold_diffs": r.fold_diffs})
    S = pd.DataFrame(sens)
    # Holm over the 3 tasks within each rule
    S["p_t_fold_holm3"] = S.groupby("rule_name").p_t_fold.transform(lambda p: holm(p.values))
    S["pass"] = (S["diff"] > 0) & (S.p_t_fold_holm3 < 0.05)
    S.to_csv(OUT / "check4_selection_rules.csv", index=False)
    rep["check4_all_positive"] = allpos
    print("\ncheck 4")
    print(S[["task", "rule_name", "design", "gain_iv", "cost_T", "diff", "p_t_fold", "p_t_fold_holm3", "pass"]].to_string())
    print(json.dumps(allpos, indent=1))

    # ---------------- check 7: statistics for the primary designs (view i)
    st = []
    for r in prim["i"].itertuples():
        fd = np.array(json.loads(r.fold_diffs))
        n = len(fd)
        sw = stats.shapiro(fd)
        signs = np.array(list(itertools.product([-1, 1], repeat=n)))
        null = (signs * fd).mean(1)
        p_perm = float((np.abs(null) >= abs(fd.mean()) - 1e-15).mean())
        rng = np.random.default_rng(7)
        boot = fd[rng.integers(0, n, (20000, n))].mean(1)
        tci = stats.t.interval(0.95, n - 1, loc=fd.mean(), scale=fd.std(ddof=1) / np.sqrt(n))
        nt, ntr = N_TEST_TRAIN[r.task]
        t_nb = fd.mean() / np.sqrt((1 / n + nt / ntr) * fd.var(ddof=1))
        p_nb = 2 * stats.t.sf(abs(t_nb), n - 1)
        st.append({"task": r.task, "design": r.design, "fold_diffs_pts": np.round(100 * fd, 2).tolist(),
                   "mean_pts": 100 * fd.mean(), "sd_pts": 100 * fd.std(ddof=1), "shapiro_p": float(sw.pvalue),
                   "t_p": float(r.p_t_fold), "t_ci95_pts": [100 * tci[0], 100 * tci[1]],
                   "signflip_exact_p": p_perm, "min_possible_signflip_p": 2 / 2 ** n,
                   "wilcoxon_p": float(r.p_w_fold), "boot_ci95_pts": [100 * np.quantile(boot, .025), 100 * np.quantile(boot, .975)],
                   "nadeau_bengio_p": float(p_nb), "pair_t_p": float(r.p_t_pair), "pair_w_p": float(r.p_w_pair)})
    ST = pd.DataFrame(st)
    ST["signflip_holm3"] = holm(ST.signflip_exact_p)
    ST["nb_holm3"] = holm(ST.nadeau_bengio_p)
    ST.to_csv(OUT / "check7_stats.csv", index=False)
    print("\ncheck 7")
    print(ST.to_string())
    rep["check7"] = json.loads(ST.to_json(orient="records"))
    (OUT / "verdict_audit.json").write_text(json.dumps(rep, indent=1, default=float))
    for v, P in prim.items():
        P.to_csv(OUT / f"check1_primary_{v}.csv", index=False)


if __name__ == "__main__":
    main()
