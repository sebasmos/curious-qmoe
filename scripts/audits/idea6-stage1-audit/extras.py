"""Extras for checks 1, 5 and 6: strict nested fold differences, a measured-latency view, ensemble
static points.   python extras.py OUT"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
OUT = Path(sys.argv[1]); R = OUT / "recompute"
TASKS = ["esc50", "esc5", "urban"]
res = {"strict_nested": {}, "latency": {}, "ensembles": {}}
for v in ["i", "ii"]:
    for t in TASKS:
        d = pd.read_csv(R / t / "designs.csv", dtype={"cheap": str})
        d = d[(d.view == v) & (d.set == "claim") & d.claim_space]
        best = {}
        for _, r in d.iterrows():
            for f, g in json.loads(r.gain_iv_fold_strict).items():
                if f not in best or g > best[f][0]:
                    best[f] = (g, r)
        folds = sorted(best, key=int)
        fd = [json.loads(best[f][1].fold_diffs)[i] for i, f in enumerate(folds)]
        res["strict_nested"][f"{v}/{t}"] = {"picks": {f: f"{best[f][1].cheap} {best[f][1].tier} {best[f][1].rule} {best[f][1].point}" for f in folds},
                                            "fold_diffs_pts": [round(100 * x, 2) for x in fd], "mean_pts": 100 * float(np.mean(fd)),
                                            "p_t": float(stats.ttest_1samp(fd, 0).pvalue)}
# latency view: measured FP32 CPU wall clock (check5), batch 1, 2 threads; no low-bit kernels, so every AST config
# takes the AST FP32 time and the best AST F1 is the hull's upper end; linear interpolation of mean F1
wc = json.load(open(OUT / "check5_costs.json"))["wall_clock_fp32_cpu_2threads"]
te, ta = float(np.median(wc["efficientnet_b3_300_s"])), float(np.median(wc["ast_1024_s"]))
picks = {"esc50": ("esc", "w4g128", "ast", "b30"), "esc5": ("4", "w4g128", "ast", "b30"), "urban": ("esc", "w4g128", "avg", "b30")}
for t in TASKS:
    sp = pd.read_csv(R / t / "static_points.csv")
    sp = sp[sp.view == "i"]
    eff = sp[sp.system.str.startswith("eff:")].f1.max()
    ast_best = sp[sp.system.str.startswith("ast:")].sort_values("f1").iloc[-1]
    d = pd.read_csv(R / t / "designs.csv", dtype={"cheap": str})
    r = d[(d.view == "i") & (d.set == "claim") & (d.cheap == picks[t][0]) & (d.tier == picks[t][1]) & (d.rule == picks[t][2]) & (d.point == picks[t][3])].iloc[0]
    mean_t = te + r.rate * ta
    lam = (mean_t - te) / (ta - te)
    hull = eff + lam * (ast_best.f1 - eff)
    res["latency"][t] = {"t_eff_s": te, "t_ast_s": ta, "cascade_mean_s": mean_t, "cascade_escalated_clip_s": te + ta,
                         "cascade_f1": r.f1, "hull_f1_at_mean_time": hull, "hull_upper": ast_best.system, "ast_best_f1": ast_best.f1,
                         "diff_pts_mean_time": 100 * (r.f1 - hull),
                         "under_deadline_between_ast_and_cascade_tail": f"only static systems fit; best is {ast_best.system} at {ast_best.f1:.4f}"}
    e = sp = pd.read_csv(R / t / "static_points.csv")
    res["ensembles"][t] = {n: float(e[(e.system == n)].f1.iloc[0]) for n in ["ens:esc+4", "ens:esc+8+4", "ens:eff+w4g128", "ens:eff+w3g64", "ens:w4g128x2", "eff:esc", "eff:4"] if (e.system == n).any()}
(OUT / "extras.json").write_text(json.dumps(res, indent=1, default=float))
print(json.dumps(res, indent=1, default=float))
