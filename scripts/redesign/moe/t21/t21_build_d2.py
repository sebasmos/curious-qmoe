"""T21 D2 (FSD50K single-label subset) metadata CSVs (ANALYSIS_PLAN.md C2). Reads only the ground truth and clip info.

A class B is an ancestor of A when every dev clip labelled A is also labelled B (A != B); a leaf is no class's ancestor.
A clip qualifies when exactly one of its labels is a leaf and every other label is an ancestor of that leaf. Classes kept:
at least 60 qualifying dev clips.
pool.csv   qualifying dev clips, sorted by name; folder = outer fold 1..5 of StratifiedGroupKFold(5, shuffle=True,
           random_state=42) by class, grouped by uploader; head_name = "u<uploader index>-<fname>" so the protocol 3
           inner split (recording_token=0) groups by uploader.
extra.csv  qualifying eval clips (official test of the secondary design), folder 0.

    python -I t21_build_d2.py META_X_DIR OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

meta, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
gt = meta / "FSD50K.ground_truth"
dev, ev = pd.read_csv(gt / "dev.csv"), pd.read_csv(gt / "eval.csv")
info_dev = json.loads((meta / "FSD50K.metadata/dev_clips_info_FSD50K.json").read_text())
info_ev = json.loads((meta / "FSD50K.metadata/eval_clips_info_FSD50K.json").read_text())
for d in (dev, ev):
    d["labset"] = d["labels"].str.split(",").map(frozenset)
vocab = sorted(set().union(*dev.labset))
clips_of = {c: set(np.flatnonzero(dev.labset.map(lambda s: c in s))) for c in vocab}
anc = {a: {b for b in vocab if b != a and clips_of[a] <= clips_of[b]} for a in vocab}
is_anc = set().union(*anc.values())
leaves = sorted(c for c in vocab if c not in is_anc)


def leaf_of(s):
    lv = [c for c in s if c in leaves]
    if len(lv) != 1:
        return None
    return lv[0] if all(c in anc[lv[0]] for c in s if c != lv[0]) else None


for d in (dev, ev):
    d["leaf"] = d.labset.map(leaf_of)
cnt = dev.leaf.value_counts()
classes = sorted(cnt[cnt >= 60].index)
cid = {c: i for i, c in enumerate(classes)}
pool = dev[dev.leaf.isin(classes)].copy()
extra = ev[ev.leaf.isin(classes)].copy()
for d, inf in ((pool, info_dev), (extra, info_ev)):
    d["name"] = d.fname.astype(str) + ".wav"
    d["category"], d["class_id"] = d.leaf, d.leaf.map(cid)
    d["label"] = d.class_id
    d["uploader"] = d.fname.astype(str).map(lambda f: inf[f]["uploader"])
    d["license"] = d.fname.astype(str).map(lambda f: inf[f]["license"])
pool = pool.sort_values("name").reset_index(drop=True)
extra = extra.sort_values("name").reset_index(drop=True)
upl = {u: i for i, u in enumerate(sorted(set(pool.uploader) | set(extra.uploader)))}
for d in (pool, extra):
    d["head_name"] = d.uploader.map(lambda u: f"u{upl[u]}") + "-" + d.name
sgk = StratifiedGroupKFold(5, shuffle=True, random_state=42)
fold = np.zeros(len(pool), int)
for k, (_, tei) in enumerate(sgk.split(np.zeros(len(pool)), pool.class_id, pool.uploader), 1):
    fold[tei] = k
pool["folder"], extra["folder"] = fold, 0
cols = ["folder", "name", "label", "category", "class_id", "head_name", "uploader", "license"]
pool[cols].to_csv(out / "pool.csv", index=False)
extra[cols].to_csv(out / "extra.csv", index=False)
ct = pd.crosstab(pool.category, pool.folder)
assert pool.groupby("uploader").folder.nunique().max() == 1
info = {"classes": classes, "n_vocab": len(vocab), "n_leaves": len(leaves), "n_classes": len(classes), "n_pool": len(pool),
        "n_extra": len(extra), "n_uploaders_pool": int(pool.uploader.nunique()), "fold_sizes": pool.folder.value_counts().sort_index().tolist(),
        "per_class_min": int(pool.class_id.value_counts().min()), "per_class_per_fold_min": int(ct.values.min()),
        "classes_missing_in_a_fold": int((ct == 0).any(axis=1).sum()), "extra_per_class_min": int(extra.class_id.value_counts().min()),
        "licences_pool": pool.license.value_counts().to_dict()}
(out / "classes.json").write_text(json.dumps(info, indent=1))
ct.to_csv(out / "class_by_fold.csv")
print(json.dumps({k: v for k, v in info.items() if k not in ("classes", "licences_pool")}, indent=1))
