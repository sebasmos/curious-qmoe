"""Reviewer extras, task C: summary of c_scaler_curve_raw.csv. For each (model, task, condition) and n: mean macro F1 over all
(seed, fold, draw), sd over the 20 draws inside a (seed, fold) pair (averaged over pairs), and sd over pairs of the pair's
draw-mean. The three references are the FP32 backbone with its scaler, the real-INT8 embeddings with the FP32 scaler (the
transplant) and the real-INT8 embeddings with the scaler refit on all fit rows.

    python c_summary.py RAW_CSV OUT_DIR
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

raw = pd.read_csv(sys.argv[1]); out = Path(sys.argv[2])
G = ["model", "task", "head", "condition"]
ref = raw[raw.variant != "int8_subset_refit"].groupby(G + ["variant"]).f1.mean().unstack("variant") * 100
sub = raw[raw.variant == "int8_subset_refit"]
pair = sub.groupby(G + ["n", "seed", "fold"]).f1.agg(["mean", "std"]).reset_index()
s = pair.groupby(G + ["n"]).agg(f1=("mean", "mean"), sd_draws=("std", "mean"), sd_pairs=("mean", "std"), n_pairs=("mean", "size")).reset_index()
for c in ("f1", "sd_draws", "sd_pairs"):
    s[c] *= 100
s = s.join(ref, on=G)
s["gap_to_full_refit_pts"] = s.f1 - s.int8_full_refit
s["gap_to_fp32_backbone_pts"] = s.f1 - s.fp32_backbone_fp32_scaler
s.to_csv(out / "c_scaler_curve_summary.csv", index=False)
pd.set_option("display.width", 220)
print(ref.round(2).to_string())
print(s.round(2).to_string())
