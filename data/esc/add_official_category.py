#!/usr/bin/env python
"""Append the official ESC-50 major category to a copy of an embedding CSV.

The ``class_id`` column that every experiment trains on is five fixed groups of
ten classes (blocks of five targets repeating every 25, see
``esc-50-data_processing.py``), not the dataset's official grouping. This script
adds ``official_category`` (``label // 10``: 0 animals, 1 natural soundscapes
and water sounds, 2 human non-speech sounds, 3 interior/domestic sounds,
4 exterior/urban noises) and ``official_category_name`` next to it. The input
file is never rewritten: experiments read the CSVs in place, so the result goes
to a new path.

Usage:
    python data/esc/add_official_category.py in.csv out.csv
"""
import argparse
import sys

import pandas as pd

OFFICIAL_CATEGORY_NAMES = {0: "Animals", 1: "Natural soundscapes & water sounds",
                           2: "Human, non-speech sounds", 3: "Interior/domestic sounds",
                           4: "Exterior/urban noises"}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("src", help="embedding CSV with a 'label' column (ESC-50 target, 0-49)")
    p.add_argument("dst", help="output path; must differ from src")
    a = p.parse_args()
    if a.src == a.dst:
        sys.exit("dst must differ from src: the input CSV is never rewritten")
    df = pd.read_csv(a.src)
    if "label" not in df.columns:
        sys.exit("no 'label' column (the ESC-50 target)")
    if not df["label"].between(0, 49).all():
        sys.exit("'label' must be an ESC-50 target in 0..49")
    cat = (df["label"] // 10).astype("int8")
    pos = df.columns.get_loc("class_id") + 1 if "class_id" in df.columns else df.columns.get_loc("label") + 1
    df.insert(pos, "official_category", cat)
    df.insert(pos + 1, "official_category_name", cat.map(OFFICIAL_CATEGORY_NAMES))
    df.to_csv(a.dst, index=False)
    print(f"{a.dst}: {len(df)} rows, official_category counts "
          f"{df['official_category'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
