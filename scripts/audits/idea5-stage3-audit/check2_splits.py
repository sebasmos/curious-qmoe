"""Check 2: splits. For every head run (dataset, seed, head) and fold:
  - test_idx is exactly the rows of the official fold (CSV 'folder' column)
  - inner_val_idx sits inside the training folds, disjoint from test
  - no recording id (ESC-50 name token 1, UrbanSound8K token 0) is on both sides of the
    inner split (fit vs inner validation), nor on both sides of the outer split (test vs train)
  - the inner validation rows stored in every scores npz equal split.json's
  - summary.json n_inner_val / n_test agree
Also: how much the inner splits differ between seeds (Jaccard of the iv sets).

    python check2_splits.py STAGE2_DIR OUT_JSON
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

s2, out = Path(sys.argv[1]), Path(sys.argv[2])
D = Path.home() / "Documents/DATASETS"
csvs = {"esc50": (D / "ESC-50/ast_768/esc-50.csv", 1), "esc5": (D / "ESC-50/ast_768/esc-50.csv", 1),
        "urban": (D / "Urban8k/ast_768/urbansound8k.csv", 0)}
tagpre = {"esc50": "astesc50_s", "esc5": "astesc5_s", "urban": "asturban_s"}
res, problems = {}, []
for ds, (csv, tok) in csvs.items():
    meta = pd.read_csv(csv, usecols=["folder", "name"])
    rec = meta["name"].str.split("-").str[tok].to_numpy()
    folder = meta["folder"].to_numpy()
    # outer: does any recording span two official folds?
    span = pd.Series(folder).groupby(rec).nunique()
    res[f"{ds}_recordings_spanning_folds"] = int((span > 1).sum())
    ivsets = {}
    nchk = 0
    for run in sorted((s2 / "outputs").glob(f"{tagpre[ds]}*")):
        if "headw4" in run.name:
            continue
        seed = int(run.name.split("_s")[1])
        for head in ("esc", "4"):
            summ = {f["fold"]: f for f in json.loads((run / head / "summary.json").read_text())["folds"]}
            for fd in sorted((run / head).glob("fold_*")):
                f = int(fd.name.split("_")[1])
                sp = json.loads((fd / "split.json").read_text())
                te, iv = np.array(sp["test_idx"]), np.array(sp["inner_val_idx"])
                tr = np.setdiff1d(np.arange(len(meta)), te)
                fit = np.setdiff1d(tr, iv)
                ok = {
                    "test_is_official_fold": np.array_equal(np.sort(te), np.flatnonzero(folder == f)),
                    "iv_in_train": np.isin(iv, tr).all(),
                    "iv_test_disjoint": len(np.intersect1d(iv, te)) == 0,
                    "iv_fit_recordings_disjoint": len(np.intersect1d(rec[iv], rec[fit])) == 0,
                    "test_train_recordings_disjoint": len(np.intersect1d(rec[te], rec[tr])) == 0,
                    "summary_n_inner_val": summ[f].get("n_inner_val", len(iv)) == len(iv),
                }
                z = np.load(s2 / "scores" / ds / run.name / head / "n3g64__clean.npz")
                ok["npz_iv_equals_split"] = np.array_equal(z[f"iv_idx_{f}"], iv)
                ok["npz_test_fold_equals_split"] = np.array_equal(np.flatnonzero(z["test_fold"] == f), np.sort(te))
                for k, v in ok.items():
                    if not v:
                        problems.append(f"{ds} {run.name} {head} fold {f}: {k}")
                nchk += 1
                if head == "esc":
                    ivsets[(seed, f)] = set(iv.tolist())
    jac = [len(ivsets[(a, f)] & ivsets[(b, f)]) / len(ivsets[(a, f)] | ivsets[(b, f)])
           for (a, f) in ivsets for (b, g) in ivsets if g == f and a < b]
    res[f"{ds}_runs_folds_checked"] = nchk
    res[f"{ds}_iv_jaccard_between_seeds_mean"] = float(np.mean(jac)) if jac else None
    res[f"{ds}_iv_size_range"] = [min(len(v) for v in ivsets.values()), max(len(v) for v in ivsets.values())]
res["problems"] = problems
Path(out).write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
