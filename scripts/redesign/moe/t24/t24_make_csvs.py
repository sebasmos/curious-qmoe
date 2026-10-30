"""T24 training CSVs for scripts/benchmark.py (cross_validation.fixed_test=true) from the clean FP32 embeddings of
every clip of all.csv (ANALYSIS_PLAN.md C2).

heads/<tier>.csv  all.csv rows in all.csv order; columns folder (0 = official test, 1..5 = train folds), name, label,
                  category, class_id, role (v = clean label, usable as inner validation; n = noisy label, fit only;
                  test), then the embedding columns 0..D-1.

    python t24_make_csvs.py --r R --d d3
"""
import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

TIERS = {"mn10": ("mn_all", "mn10_fp32__clean.npy"), "ast": ("ast_all", "fp32__clean.npy")}
META = ["folder", "name", "label", "category", "class_id", "role"]


def load(r, d, sub, fname, meta):
    X = np.load(r / "emb" / d / sub / fname)
    side = json.loads((r / "emb" / d / sub / fname.replace(".npy", ".json")).read_text())
    crc = zlib.crc32("\n".join(meta["name"].map(lambda s: Path(s).stem)).encode())
    if crc != side["names_crc32"] or len(X) != len(meta) or not np.isfinite(X).all():
        raise SystemExit(f"{sub}/{fname}: rows do not match")
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=Path, required=True)
    ap.add_argument("--d", required=True)
    ap.add_argument("--tiers", nargs="+", default=list(TIERS))
    a = ap.parse_args()
    al = pd.read_csv(a.r / "csv" / a.d / "all.csv")
    out = a.r / "csv" / a.d / "heads"
    out.mkdir(parents=True, exist_ok=True)
    for t in a.tiers:
        sub, fn = TIERS[t]
        X = load(a.r, a.d, sub, fn, al)
        cols = [str(i) for i in range(X.shape[1])]
        df = pd.concat([al[META].reset_index(drop=True), pd.DataFrame(X, columns=cols)], axis=1)
        df.to_csv(out / f"{t}.csv", index=False, float_format="%.8g")
        print(t, df.shape, flush=True)


if __name__ == "__main__":
    main()
