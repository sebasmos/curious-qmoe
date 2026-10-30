"""Applies the preconditions of ANALYSIS_PLAN_T15.md section 5 and ANALYSIS_PLAN_T18.md section 2 to <R>/precond/precond_views.csv and the waveform JSONs; writes
<R>/precond/survivors.json and precond_table.md.   python t15_survivors.py"""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t15_common import R, LABEL
P = pd.read_csv(R / "precond/precond_views.csv"); cost = json.load(open(R / "inputs/view_costs.json"))
ok = {}  # view -> {dataset: bool}
for v in ["v1", "v2", "v3", "v4", "v6", "lra", "lrb"]:
    ok[v] = {r.dataset: bool(r.within_5_points) for r in P[P.view == v].itertuples()}
for ds in ["urban", "esc50"]:
    f = R / f"precond/wave_precond_{ds}.json"
    if f.exists(): ok.setdefault("v5", {})[ds] = bool(json.load(open(f))["reached_0_60_within_10"])
views = {"urban": [], "esc50": [], "tau": []}
for v in ["v1", "v2", "v3", "v4", "v5", "v6"]:
    o = ok.get(v, {})
    for ds in ("urban", "esc50"):
        if o.get(ds): views[ds].append(v)
    if o.get("urban") and o.get("esc50"): views["tau"].append(v)
qual = [v for v in ("lra", "lrb") if ok.get(v, {}).get("urban") and ok.get(v, {}).get("esc50")]
lr = min(qual, key=lambda v: cost["esc50"][v]["total_bitops"]) if qual else None
res = {"views": views, "lr": lr, "lr_qualifying": qual, "per_view_dataset": ok}
(R / "precond/survivors.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
lines = ["| Dataset | View | inner-validation macro-F1 | MN10-Q8 (same clips) | gap, points | MN10-Q8 padded, gap | passes |", "|---|---|---|---|---|---|---|"]
for r in P[P.view != "mn10_q8"].itertuples():
    lines.append(f"| {r.dataset} | {LABEL.get(r.view, r.view)} | {100 * r.f1:.2f} | {100 * r.mn10_q8:.2f} | {r.gap_to_mn10_q8_points:+.2f} | {r.gap_to_padded_mn10_points:+.2f} | {'yes' if r.within_5_points else 'no'} |")
for ds in ("urban", "esc50"):
    f = R / f"precond/wave_precond_{ds}.json"
    if f.exists():
        w = json.load(open(f)); lines.append(f"| {ds} | V5 waveform CNN (seed 0, fold 1, FP32 per epoch, best of {w['epochs_run']} epochs) | {100 * w['best_iv_f1_fp32']:.2f} (W8A8 {100 * w['best_epoch_iv_f1_w8a8']:.2f}) | {100 * w['mn10_q8_iv_f1_same_clips']:.2f} | | | {'yes' if w['reached_0_60_within_10'] else 'no'} (bar 0.60) |")
(R / "precond/precond_table.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
