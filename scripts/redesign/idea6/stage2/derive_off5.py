"""Derived training CSV for amendment 3: the same rows and columns, with
class_id (scripts/benchmark.py's label column) set to the official ESC-50
category, target // 10, and category to its name. The source CSV is only read.

    python scripts/redesign/idea6/stage2/derive_off5.py SRC.csv OUT.csv
"""
from __future__ import annotations

import sys

import pandas as pd

NAMES = ["Animals", "Natural soundscapes and water sounds", "Human non-speech", "Interior/domestic",
         "Exterior/urban"]


def main(src, out):
    df = pd.read_csv(src)
    if sorted(df["label"].unique()) != list(range(50)):
        raise SystemExit(f"{src}: label is not the 50 ESC-50 targets")
    df["class_id"] = df["label"] // 10
    df["category"] = df["class_id"].map(dict(enumerate(NAMES)))
    df.to_csv(out, index=False, float_format="%.8g")
    print(f"{out}: {df.shape}, per class {df.class_id.value_counts().sort_index().tolist()}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
