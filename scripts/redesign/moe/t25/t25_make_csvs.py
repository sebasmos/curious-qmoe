"""T25 head-training CSVs for scripts/benchmark.py from the clean FP32 embeddings of each new CNN (T21's
t21_make_csvs.py, pool design only). Rows in T21's pool.csv order, folder = T21's outer fold 1..5.
Columns: folder, name, label, category, class_id, then the embedding columns 0..D-1.

    python t25_make_csvs.py --r R25 --models mn04 mn05 mn20 dymn04
"""
import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

META = ["folder", "name", "label", "category", "class_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=Path, required=True)
    ap.add_argument("--d", default="d1")
    ap.add_argument("--models", nargs="+", default=["mn04", "mn05", "mn20", "dymn04"])
    a = ap.parse_args()
    pool = pd.read_csv(a.r / "csv" / a.d / "pool.csv")
    crc = zlib.crc32("\n".join(pool["name"].map(lambda s: Path(s).stem)).encode())
    out = a.r / "csv" / a.d / "heads"
    out.mkdir(parents=True, exist_ok=True)
    for m in a.models:
        f = a.r / "emb" / a.d / "cnn_pool" / f"{m}_fp32__clean.npy"
        X = np.load(f)
        side = json.loads(f.with_suffix(".json").read_text())
        if side["names_crc32"] != crc or len(X) != len(pool) or not np.isfinite(X).all():
            raise SystemExit(f"{f}: rows do not match")
        df = pd.concat([pool[META].copy(), pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])], axis=1)
        df.to_csv(out / f"pool_{m}.csv", index=False, float_format="%.8g")
        print(m, df.shape, flush=True)


if __name__ == "__main__":
    main()
