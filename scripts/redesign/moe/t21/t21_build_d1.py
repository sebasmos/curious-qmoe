"""T21 D1 (FSDKaggle2018) metadata CSVs (ANALYSIS_PLAN.md C2). Reads only the two official metadata CSVs.

pool.csv   the 5,310 manually verified clips (verified train plus official test), sorted by file name; folder = outer
           fold 1..5 of StratifiedKFold(5, shuffle=True, random_state=42) by class; columns folder, name, label,
           category, class_id, official (train|test), manually_verified, freesound_id, license.
extra.csv  the 5,763 non-verified official train clips (clean extraction only, secondary design).
classes.json  class names in class_id order (sorted names), per-class counts.

    python -I t21_build_d1.py META_DIR OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

meta, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
tr = pd.read_csv(meta / "train_post_competition.csv")
te = pd.read_csv(meta / "test_post_competition_scoring_clips.csv")
assert tr.freesound_id.is_unique and te.freesound_id.is_unique and not (set(tr.freesound_id) & set(te.freesound_id))
assert not (set(tr.fname) & set(te.fname))
classes = sorted(tr.label.unique())
assert classes == sorted(te.label.unique()) and len(classes) == 41
cid = {c: i for i, c in enumerate(classes)}
tr["official"], te["official"] = "train", "test"
te["manually_verified"] = 1
pool = pd.concat([tr[tr.manually_verified == 1], te], ignore_index=True).sort_values("fname").reset_index(drop=True)
extra = tr[tr.manually_verified == 0].sort_values("fname").reset_index(drop=True)
for d in (pool, extra):
    d["name"], d["category"], d["class_id"] = d.fname, d.label, d.label.map(cid)
    d["label"] = d["class_id"]
skf = StratifiedKFold(5, shuffle=True, random_state=42)
fold = np.zeros(len(pool), int)
for k, (_, tei) in enumerate(skf.split(np.zeros(len(pool)), pool.class_id), 1):
    fold[tei] = k
pool["folder"] = fold
extra["folder"] = 0
cols = ["folder", "name", "label", "category", "class_id", "official", "manually_verified", "freesound_id", "license"]
pool[cols].to_csv(out / "pool.csv", index=False)
extra[cols].to_csv(out / "extra.csv", index=False)
ct = pd.crosstab(pool.category, pool.folder)
assert (ct > 0).all().all(), "a fold misses a class"
info = {"classes": classes, "n_pool": len(pool), "n_extra": len(extra), "n_official_test": int((pool.official == "test").sum()),
        "pool_per_class_min": int(pool.class_id.value_counts().min()), "pool_per_class_max": int(pool.class_id.value_counts().max()),
        "fold_sizes": pool.folder.value_counts().sort_index().tolist(), "per_class_per_fold_min": int(ct.values.min()),
        "licences_pool": pool.license.value_counts().to_dict(), "licences_extra": extra.license.value_counts().to_dict()}
(out / "classes.json").write_text(json.dumps(info, indent=1))
ct.to_csv(out / "class_by_fold.csv")
print(json.dumps({k: v for k, v in info.items() if k != "classes"}, indent=1))
