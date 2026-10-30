"""Write an embedding .npy as a training CSV: the clean AST CSV's metadata
columns and row order, the .npy's 768 features (for a head trained on a
quantized backbone's clean embeddings).

    python scripts/redesign/idea5/emb_to_csv.py CLEAN_AST_CSV EMB.npy OUT.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(clean_csv, emb, out):
    header = pd.read_csv(clean_csv, nrows=0).columns
    meta_cols = [c for c in header if not c.isdigit()]
    meta = pd.read_csv(clean_csv, usecols=meta_cols)[meta_cols]
    X = np.load(emb)
    if X.shape != (len(meta), 768) or not np.isfinite(X).all():
        raise SystemExit(f"{emb}: shape {X.shape} or non-finite values")
    df = pd.concat([meta, pd.DataFrame(X, columns=[str(i) for i in range(768)])], axis=1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.8g")
    print(f"wrote {out} {df.shape}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
