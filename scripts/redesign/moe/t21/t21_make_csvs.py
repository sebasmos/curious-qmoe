"""T21 training CSVs for scripts/benchmark.py from the clean FP32 embeddings (ANALYSIS_PLAN.md C1, C2).

pool_<tier>.csv      pool rows in pool.csv order, folder = outer fold 1..5.
official_<tier>.csv  official split: pool rows (official test -> folder 1, verified train -> folder 2) in pool order,
                     then extra rows (folder 2) in extra.csv order. official_rows.csv maps each row to (source, index).
                     D2 (no `official` column): pool rows are the dev clips (folder 2), extra rows the eval clips (folder 1).
                     When the metadata has `head_name` (D2: "u<uploader>-<fname>"), it is the name column, so the
                     protocol 3 inner split (recording_token=0) groups by uploader.
Columns: folder, name, label, category, class_id, then the embedding columns 0..D-1.

    python t21_make_csvs.py --r R --d d1
"""
import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

TIERS = {"mn10": ("mn_pool", "mn_extra", "mn10_fp32__clean.npy"), "ast": ("ast_pool", "ast_extra", "fp32__clean.npy")}
META = ["folder", "name", "label", "category", "class_id"]


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
    ap.add_argument("--d", default="d1")
    ap.add_argument("--tiers", nargs="+", default=list(TIERS))
    a = ap.parse_args()
    pool = pd.read_csv(a.r / "csv" / a.d / "pool.csv")
    extra = pd.read_csv(a.r / "csv" / a.d / "extra.csv")
    out = a.r / "csv" / a.d / "heads"
    out.mkdir(parents=True, exist_ok=True)
    for t in a.tiers:
        ps, es, fn = TIERS[t]
        Xp, Xe = load(a.r, a.d, ps, fn, pool), load(a.r, a.d, es, fn, extra)
        cols = [str(i) for i in range(Xp.shape[1])]
        mp, me = pool[META].copy(), extra[META].copy()
        if "head_name" in pool.columns:
            mp["name"], me["name"] = pool.head_name, extra.head_name
        dfp = pd.concat([mp, pd.DataFrame(Xp, columns=cols)], axis=1)
        dfp.to_csv(out / f"pool_{t}.csv", index=False, float_format="%.8g")
        po, eo = mp.copy(), me.copy()
        d1 = "official" in pool.columns
        po["folder"] = np.where(pool.official == "test", 1, 2) if d1 else 2
        eo["folder"] = 2 if d1 else 1
        dfo = pd.concat([pd.concat([po, pd.DataFrame(Xp, columns=cols)], axis=1),
                         pd.concat([eo, pd.DataFrame(Xe, columns=cols)], axis=1)], ignore_index=True)
        dfo.to_csv(out / f"official_{t}.csv", index=False, float_format="%.8g")
        print(t, dfp.shape, dfo.shape, flush=True)
    pd.DataFrame({"source": ["pool"] * len(pool) + ["extra"] * len(extra),
                  "index": list(range(len(pool))) + list(range(len(extra)))}).to_csv(out / "official_rows.csv", index=False)


if __name__ == "__main__":
    main()
