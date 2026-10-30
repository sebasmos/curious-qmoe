"""Checks 1, 6, 7, 10 across tasks: Holm over the three tasks for every variant, normality and bootstrap of
the fold differences, one design for all tasks. Reads outputs/recompute_<task>/. Writes outputs/combined.csv
and outputs/combined.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, R, holm  # noqa: E402

TASKS = ["esc50", "esc5", "urban"]
res = {t: json.loads((OUT / f"recompute_{t}/tests.json").read_text()) for t in TASKS}
fd = {t: dict(np.load(OUT / f"recompute_{t}/fold_diffs.npz")) for t in TASKS}
claim = {t: r for t, r in zip(TASKS, pd.read_csv(R / "analysis/hypothesis_H1.csv").to_dict("records"))}
rows = []


def add(variant, getter):
    ps, recs = [], []
    for t in TASKS:
        o = getter(t)
        if o is None:
            return
        recs.append((t, o))
        ps.append(o["p_t"])
    ph = holm(ps)
    for (t, o), p in zip(recs, ph):
        rows.append(dict(variant=variant, task=t, design=o.get("design", ""), f1=o.get("f1"), cost_T=o.get("cost_T"),
                         hull_f1=o.get("hull_f1"), diff_pts=o["diff_pts"], p_t=o["p_t"], p_holm3=p,
                         p_nb=o.get("p_nb"), p_sign=o.get("p_sign"), passes=bool(o["diff_pts"] > 0 and p < 0.05),
                         fold_diffs=json.dumps(o.get("fold_diffs_pts"))))


add("H1 (mine)", lambda t: res[t]["H1"])
add("H2 clean (mine)", lambda t: res[t]["H2"])
add("H3 FP32 cheap tier", lambda t: res[t]["H3"])
add("H4 W8A8 cheap tier", lambda t: res[t]["H4"])
add("E3 unpriced", lambda t: res[t]["E3_unpriced"])
add("H1, hull with ensembles", lambda t: res[t]["H1_enlarged_hull"])
add("H2, hull with ensembles", lambda t: res[t]["H2_enlarged_hull"])
add("H1, every elementwise op and quantize priced on both tiers", lambda t: res[t]["H1_cost_all_elementwise"])
add("nested strict (design and hull pair from the fold's own inner rows)",
    lambda t: {**res[t]["nested"]["strict"], "design": json.dumps(res[t]["nested"]["picks"])})
add("nested, design per fold, hull pair as claim", lambda t: {**res[t]["nested"]["design_only"], "design": ""})
for k in ("rank2", "rank3", "fixed_b05", "fixed_b10", "fixed_b20", "fixed_b30", "fixed_b50", "same_tier_other_rule"):
    add(f"selection: {k}", lambda t, k=k: res[t]["sensitivity"][k])

# one design for all tasks: largest mean inner-validation gain over the three tasks
sel = []
for t in TASKS:
    s = pd.read_csv(OUT / f"recompute_{t}/selection.csv")
    s["tier"] = s.expensive.str.replace("@512", "@short").str.replace("@400", "@short")
    s["task"] = t
    sel.append(s)
S = pd.concat(sel)
key = ["cheap_bb", "cheap", "tier", "rule", "budget"]
G = S.groupby(key).gain_iv.agg(["mean", "count"]).reset_index()
G = G[G["count"] == 3].sort_values("mean", ascending=False)
one = G.iloc[0]
one_rows = []
for t in TASKS:
    e2 = pd.read_csv(OUT / f"recompute_{t}/e2_field.csv")
    e2["tier"] = e2.expensive.str.replace("@512", "@short").str.replace("@400", "@short")
    r = e2[(e2.cheap_bb == one.cheap_bb) & (e2.cheap == one.cheap) & (e2.tier == one.tier) & (e2.rule == one.rule)
           & (e2.budget == one.budget)].iloc[0]
    one_rows.append(dict(design=" ".join(map(str, one[key])), f1=r.f1, cost_T=r.cost_T, diff_pts=r.diff_pts, p_t=r.p_t,
                         p_nb=r.p_nb, fold_diffs_pts=json.loads(r.fold_diffs)))
add("selection: one design for all tasks", lambda t: one_rows[TASKS.index(t)])

C = pd.DataFrame(rows)
C.to_csv(OUT / "combined.csv", index=False)

# statistics of the H1 fold differences
st = {}
for t in TASKS:
    x = fd[t]["H1"] * 100
    k = len(x)
    sh = stats.shapiro(x)
    ci_t = stats.t.interval(0.95, k - 1, loc=x.mean(), scale=x.std(ddof=1) / np.sqrt(k))
    rng = np.random.default_rng(1)
    bs = rng.choice(x, (20000, k), replace=True).mean(1)
    st[t] = dict(fold_diffs_pts=np.round(x, 3).tolist(), mean=float(x.mean()), shapiro_p=float(sh.pvalue),
                 t_ci95=[float(ci_t[0]), float(ci_t[1])], bootstrap_ci95=np.percentile(bs, [2.5, 97.5]).tolist(),
                 wilcoxon_p=float(stats.wilcoxon(x).pvalue), n_folds=k,
                 claim_diff_pts=claim[t]["diff_pts"], claim_p_t=claim[t]["p_t_fold"])
# E2-style rank check
sp = {t: res[t]["spearman_gain_iv_vs_test_diff"] for t in TASKS}
e2pos = {}
for t in TASKS:
    e2 = pd.read_csv(OUT / f"recompute_{t}/e2_field.csv")
    e2pos[t] = dict(n=len(e2), positive=int((e2.diff_pts > 0).sum()),
                    positive_without_w8a8_and_short=int(((e2.diff_pts > 0) & ~((e2.cheap_bb == "w8a8") & e2.expensive.str.contains("@"))).sum()),
                    best=e2.sort_values("diff_pts").iloc[-1][["cheap_bb", "cheap", "expensive", "rule", "budget", "diff_pts"]].tolist())
out = dict(stats_h1=st, spearman_gain_iv_vs_test_diff=sp, one_design_for_all=dict(design=" ".join(map(str, one[key])),
           mean_gain_iv=float(one["mean"])), e2=e2pos,
           picks={t: dict(mine=res[t]["pick_mine"], claim=res[t]["pick_claim"], gain=res[t]["pick_gain_iv"],
                          runner_up=res[t]["runner_up_gain_iv"]) for t in TASKS},
           delta_cheap_cost={t: res[t]["delta_cheap_cost"] for t in TASKS},
           extra_budgets={t: res[t]["extra_budgets_h1_tiers"] for t in TASKS})
(OUT / "combined.json").write_text(json.dumps(out, indent=1, default=float))
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
print(C[["variant", "task", "design", "f1", "cost_T", "hull_f1", "diff_pts", "p_t", "p_holm3", "p_nb", "passes"]]
      .round(4).to_string())
print(json.dumps(out, indent=1, default=float))
