"""T22 learning-rate choice (ANALYSIS_PLAN.md C2): the body rate with the highest best-checkpoint inner-validation macro F1 on fold 1,
seed 0, ties to the lower rate.   python t22_pick_lr.py --runs RUNS --task esc50"""
import argparse, json
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--runs", type=Path, required=True); ap.add_argument("--task", required=True); a = ap.parse_args()
GRID = ["1e-4", "3e-4", "1e-3"]
r = {g: json.load(open(a.runs / f"{a.task}_f1_s0_lr{g}.json"))["best_iv_f1_fp32"] for g in GRID}
best = max(GRID, key=lambda g: (r[g], -float(g)))
out = {"task": a.task, "grid": r, "chosen": best, "rule": "highest best-checkpoint FP32 inner-validation macro F1, fold 1 seed 0, ties to the lower rate"}
(a.runs / f"lr_choice_{a.task}.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out))
