"""Training CSV for scripts/benchmark.py from a universe embedding array (idea 6 option B): the device-A rows of
the TAU metadata CSV (the first block of rows, checked) with the standard columns (folder, name, label, category,
class_id) and the feature columns 0..D-1 of the clean FP32 embedding of those rows. The sidecar's names checksum
must equal the universe CSV's.

    python scripts/redesign/idea6/optionB/make_train_csv.py --emb R/emb_cnn/tau/mn10_fp32__clean.npy \
        --meta ~/Documents/DATASETS/TAU2020Mobile/ast_768/tau2020mobile.csv --out R/csv/tau_mn10_fp32.csv
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
    meta = pd.read_csv(a.meta)
    crc = zlib.crc32("\n".join(meta["name"].map(lambda s: Path(s).stem)).encode())
    if crc != side["names_crc32"] or len(meta) != len(X) or not np.isfinite(X).all():
        raise SystemExit(f"{a.emb}: rows do not match {a.meta}")
    isa = (meta.device == "a").to_numpy()
    n_a = int(isa.sum())
    if not isa[:n_a].all() or isa[n_a:].any():
        raise SystemExit("device-A rows are not the first block")
    df = pd.concat([meta.loc[:n_a - 1, ["folder", "name", "label", "category", "class_id"]],
                    pd.DataFrame(X[:n_a], columns=[str(i) for i in range(X.shape[1])])], axis=1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False, float_format="%.8g")
    print(f"{a.out}: {df.shape}, universe names_crc32 {crc}")


if __name__ == "__main__":
    main()
