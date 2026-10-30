"""Metadata CSV with location-grouped folds for TAU Urban Acoustic Scenes 2020 Mobile (idea 6 option B).

Reads the dataset's meta.csv (23,035 segments: 14,400 of device A, 1,078 B, 1,077 C, 1,080 each S1 to S6; every
B, C and S segment is a parallel version of a device-A segment at the same location). Five folds are built once
over the device-A segments with StratifiedGroupKFold(5, shuffle, random_state=0) on the scene label with the
location id (token 2 of the file name, unique across scenes and cities) as the group; every other device's segment
takes the fold of its location, so no location is on both sides of any split. Row order: device A first, then the
other devices, each block sorted by file stem; the device-A block is the training CSV's row order.

Columns: folder (fold 1 to 5), name (<stem>.wav), label and category (scene), class_id (0 to 9, scenes sorted),
device, city, location, segment, official (train, test or none in the DCASE fold1 split, recorded, not used).

    python scripts/redesign/idea6/optionB/build_csv.py --meta .../meta.csv --eval-setup .../evaluation_setup --out .../tau2020mobile.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

N_FOLDS, SEED = 5, 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--eval-setup", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    m = pd.read_csv(a.meta, sep="\t")
    stem = m.filename.str.replace("audio/", "", regex=False).str.replace(".wav", "", regex=False)
    p = stem.str.split("-")
    df = pd.DataFrame({"name": stem + ".wav", "label": p.str[0], "city": p.str[1], "location": p.str[2].astype(int),
                       "segment": p.str[3].astype(int), "device": p.str[4]})
    if not (df.label == m.scene_label).all():
        raise SystemExit("scene label differs from the file name")
    scenes = sorted(df.label.unique())
    df["category"] = df.label
    df["class_id"] = df.label.map({s: i for i, s in enumerate(scenes)})
    loc_scene = df.groupby("location")[["label", "city"]].nunique()
    if (loc_scene > 1).any().any():
        raise SystemExit("a location id spans more than one scene or city")
    A = df[df.device == "a"].sort_values("name").reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_of_loc = {}
    for k, (_, te) in enumerate(sgkf.split(np.zeros(len(A)), A.class_id, A.location), 1):
        for loc in A.location.iloc[te].unique():
            if loc in fold_of_loc:
                raise SystemExit(f"location {loc} in two folds")
            fold_of_loc[int(loc)] = k
    if set(df.location) - set(fold_of_loc):
        raise SystemExit("a device segment's location has no device-A segment")
    df["folder"] = df.location.map(fold_of_loc).astype(int)
    off = {}
    for part, fn in (("train", "fold1_train.csv"), ("test", "fold1_evaluate.csv")):
        s = pd.read_csv(a.eval_setup / fn, sep="\t").filename.str.replace("audio/", "", regex=False)
        off.update({x: part for x in s})
    df["official"] = df.name.map(off).fillna("none")
    df["_dev"] = (df.device != "a").astype(int)
    df = df.sort_values(["_dev", "name"]).drop(columns="_dev").reset_index(drop=True)
    df = df[["folder", "name", "label", "category", "class_id", "device", "city", "location", "segment", "official"]]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    stats = {"n": int(len(df)), "n_a": int((df.device == "a").sum()), "scenes": scenes, "n_folds": N_FOLDS, "seed": SEED,
             "locations": int(df.location.nunique()),
             "fold_counts_by_device": df.groupby(["device", "folder"]).size().unstack().to_dict(),
             "fold_counts_by_scene_a": A.assign(folder=A.location.map(fold_of_loc)).groupby(["label", "folder"]).size().unstack().to_dict(),
             "fold_locations": df.groupby("folder").location.nunique().to_dict(),
             "official": df.groupby(["official", "device"]).size().unstack().fillna(0).astype(int).to_dict()}
    a.out.with_suffix(".folds.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: stats[k] for k in ("n", "n_a", "locations", "fold_locations")}))
    print(pd.DataFrame(stats["fold_counts_by_device"]).to_string())


if __name__ == "__main__":
    main()
