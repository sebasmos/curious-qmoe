"""T22 own recompute of P1 from raw score files only: the coarse expert's saved inner-validation logits (scores_p1/*.npz) and the stored
MN10-Q8 npz files listed (with sha256) in T13's file_list.csv; macro F1 by sklearn.   python t22_p1_recompute.py --r R --tasks esc50 urban"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
ap = argparse.ArgumentParser(); ap.add_argument("--r", type=Path, required=True); ap.add_argument("--tasks", nargs="+", required=True); a = ap.parse_args()
FL = pd.read_csv(Path.home() / "Documents/qmoe-results/2026-10-25_0145_moe-t13_code-eaaedc9/inputs/file_list.csv")
out = {}
for t in a.tasks:
    rows = []
    for s in (0, 1, 2):
        c = np.load(a.r / "scores_p1" / f"{t}_f1_s{s}_iv.npz")
        r = FL[(FL.task == t) & (FL.expert == "mn10") & (FL.cond == "clean") & (FL.seed == s)].iloc[0]
        assert hashlib.sha256(open(r.path, "rb").read()).hexdigest() == r.sha256
        z = np.load(r.path); idx = z["iv_idx_1"]; o = np.argsort(idx)
        assert np.array_equal(idx[o], c["iv_idx"]) and np.array_equal(z["y"][c["iv_idx"]], c["y"])
        f = lambda yy, p: 100 * f1_score(yy, p, average="macro", labels=np.unique(np.concatenate([yy, p])))  # noqa: E731
        rows.append({"seed": s, "coarse_w8a8": f(c["y"], c["logits_w8a8"].argmax(1)), "mn10_q8": f(c["y"], z["iv_logits_1"][o].argmax(1))})
    D = pd.DataFrame(rows); out[t] = {"coarse_w8a8": D.coarse_w8a8.mean(), "mn10_q8": D.mn10_q8.mean(), "diff": D.coarse_w8a8.mean() - D.mn10_q8.mean(),
                                      "per_seed_diff": (D.coarse_w8a8 - D.mn10_q8).round(3).tolist()}
rep = json.loads((a.r / "tables/p1.json").read_text())
for t in out: out[t]["abs_diff_vs_reported"] = abs(out[t]["diff"] - rep["tasks"][t]["diff_w8a8_minus_mn10"])
(a.r / "tables/p1_recompute.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
