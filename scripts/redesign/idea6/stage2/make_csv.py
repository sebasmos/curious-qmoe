"""Training CSV for scripts/benchmark.py from an extracted embedding array:
the metadata columns (folder, name, label, category, class_id) of the CSV the
rows were extracted in (same order, checked by the names checksum in the
sidecar), then the 1536 feature columns 0..1535.

    python scripts/redesign/idea6/stage2/make_csv.py --emb EMB/esc50/fp32__clean.npy \
        --meta ~/Documents/DATASETS/ESC-50/ast_768/esc-50.csv --out CSV/esc-50_effb3pool_fp32.csv
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    X = np.load(a.emb)
    side = json.loads(a.emb.with_suffix(".json").read_text())
    meta = pd.read_csv(a.meta, usecols=["folder", "name", "label", "category", "class_id"])
    crc = zlib.crc32("\n".join(meta["name"].map(lambda s: Path(s).stem)).encode())
    if crc != side["names_crc32"] or len(meta) != len(X) or not np.isfinite(X).all():
        raise SystemExit(f"{a.emb}: rows do not match {a.meta}")
    df = pd.concat([meta, pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])], axis=1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False, float_format="%.8g")
    print(f"{a.out}: {df.shape}, names_crc32 {crc}")


if __name__ == "__main__":
    main()
