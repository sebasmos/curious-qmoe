"""Reviewer extras, task B: head width (Q1, Q2, Q4, Q8, BitNet) against the FP32 head on the clean MN10 and MN20 embeddings,
ESC-50 (50 classes), protocol 3 recipe, official folds, seeds 0 to 4 (25 seed-fold pairs).

Macro F1 per head from summary.json of each run (Stage 3b / 3c folders for esc, 8, 4, bitnet; this folder for 1 and 2).
Differences to the FP32 head (kind esc) are paired by (seed, fold). Two intervals are given: pair level (25 pairs, df 24,
optimistic because the pairs share folds) and fold level (seeds averaged inside each fold, 5 folds, df 4; the one the
draft's Table 1 re-check uses). TOST equivalence at +-1.0 point is reported at both levels; the fold-level one is the
conservative reading.

    python b_heads.py --s3b S3B --s3c S3C --r R --out OUT
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
for k in ("s3b", "s3c", "r", "out"):
    ap.add_argument("--" + k, required=True, type=Path)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
KINDS = ["esc", "8", "4", "2", "1", "bitnet"]
NAME = {"esc": "FP32", "8": "Q8", "4": "Q4", "2": "Q2", "1": "Q1", "bitnet": "BitNet"}
BOUND = 1.0

def runs(bb, k, s):
    if k in ("1", "2"):
        return a.r / "outputs" / f"{bb}_esc50_s{s}" / k
    return (a.s3b if bb == "mn10" else a.s3c) / "outputs" / f"{bb}_esc50_s{s}" / k

def f1s(bb, k):
    M = np.zeros((5, 5))  # seed x fold
    for s in range(5):
        S = json.loads((runs(bb, k, s) / "summary.json").read_text())["folds"]
        assert sorted(f["fold"] for f in S) == [1, 2, 3, 4, 5]
        for f in S:
            M[s, f["fold"] - 1] = f["f1_macro"]
    return M

def tost(d, bound):
    n = len(d); se = d.std(ddof=1) / np.sqrt(n)
    return max(stats.t.sf((d.mean() + bound) / se, n - 1), stats.t.cdf((d.mean() - bound) / se, n - 1))

rows = []
for bb in ("mn10", "mn20"):
    M = {k: f1s(bb, k) for k in KINDS}
    for k in KINDS:
        row = dict(backbone=bb, head=NAME[k], macro_f1_pairs=M[k].mean() * 100, macro_f1_sd_over_folds=M[k].mean(0).std(ddof=1) * 100)
        if k != "esc":
            dp = (M[k] - M["esc"]).ravel() * 100            # 25 pairs
            df_ = (M[k] - M["esc"]).mean(0) * 100           # 5 fold means
            for tag, d in (("pair", dp), ("fold", df_)):
                n = len(d); se = d.std(ddof=1) / np.sqrt(n); h = stats.t.ppf(0.975, n - 1) * se
                row.update({f"diff_{tag}": d.mean(), f"ci_lo_{tag}": d.mean() - h, f"ci_hi_{tag}": d.mean() + h,
                            f"p_t_{tag}": 2 * stats.t.sf(abs(d.mean() / se), n - 1) if se > 0 else np.nan,
                            f"tost_p_{tag}": tost(d, BOUND)})
        rows.append(row)
R = pd.DataFrame(rows)
R.to_csv(a.out / "b_heads.csv", index=False)
print(R.round(3).to_string())
