"""T24 D3 (FSDnoisy18k, Zenodo 2529934) metadata CSVs (ANALYSIS_PLAN.md C2). Reads only train.csv and test.csv of
FSDnoisy18k.meta and, for uploader groups, FSD50K's clip-info JSONs (already on disk from T21).

all.csv   train rows (sorted by name) then test rows (sorted by name). folder: 1..5 for train (StratifiedGroupKFold(5,
          shuffle=True, random_state=42) by class, grouped by `group`), 0 for test. role: "v" for manually verified
          train clips (inner-validation rows of the model of their fold), "n" for the noisy train clips, "test".
          group: "u:<uploader>" where FSD50K's clip info gives the uploader of that Freesound id, else "c:<id>" (own
          group). part/src: which zip and which member the clip comes from.
eval.csv  the rows of all.csv with role "v" or "test", same order: the clips extracted under noise (field mix
          assignment over these rows).

    python -I t24_build_d3.py FSDNOISY_META_DIR FSD50K_META_X_DIR OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

meta, f50, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
tr, te = pd.read_csv(meta / "train.csv"), pd.read_csv(meta / "test.csv")
assert tr.fname.is_unique and te.fname.is_unique
# Amendment 1: three noisy train rows share their Freesound id (file name) with a test clip; they are dropped from the
# train set so no test sound is trained on. The official test set is kept whole.
shared = sorted(set(tr.fname) & set(te.fname))
tr = tr[~tr.fname.isin(shared)].reset_index(drop=True)
classes = sorted(tr.label.unique())
assert classes == sorted(te.label.unique()), "train and test classes differ"
cid = {c: i for i, c in enumerate(classes)}
up = {}
for f in ("dev_clips_info_FSD50K.json", "eval_clips_info_FSD50K.json"):
    up.update({str(k): v["uploader"] for k, v in json.loads((f50 / "FSD50K.metadata" / f).read_text()).items()})
tr["part"], te["part"] = "train", "test"
tr["role"] = np.where(tr.manually_verified == 1, "v", "n")
te["role"], te["manually_verified"] = "test", 1
for d in (tr, te):
    d["src"] = d.fname
    d["stem"] = d.fname.map(lambda s: Path(s).stem)
    d["freesound_id"] = d.stem.where(d.stem.str.fullmatch(r"\d+"), "")
    d["group"] = [f"u:{up[s]}" if s in up else f"c:{s}" for s in d.stem]
    d["name"] = d.stem + ".wav"
    d["category"], d["class_id"] = d.label, d.label.map(cid)
    d["label"] = d.class_id
tr = tr.sort_values("name").reset_index(drop=True)
te = te.sort_values("name").reset_index(drop=True)
fold = np.zeros(len(tr), int)
sgk = StratifiedGroupKFold(5, shuffle=True, random_state=42)
for k, (_, tei) in enumerate(sgk.split(np.zeros(len(tr)), tr.class_id, tr.group), 1):
    fold[tei] = k
tr["folder"], te["folder"] = fold, 0
cols = ["folder", "name", "label", "category", "class_id", "role", "group", "freesound_id", "manually_verified", "part", "src"]
al = pd.concat([tr[cols], te[cols]], ignore_index=True)
assert al.name.is_unique
al.to_csv(out / "all.csv", index=False)
ev = al[al.role.isin(["v", "test"])]
ev.to_csv(out / "eval.csv", index=False)
trn = al[al.folder > 0]
assert trn.groupby("group").folder.nunique().max() == 1
ct_v = pd.crosstab(trn[trn.role == "v"].category, trn[trn.role == "v"].folder)
info = {"dataset": "FSDnoisy18k", "train_rows_dropped_shared_with_test": shared, "classes": classes, "n_classes": len(classes), "n_train": len(tr), "n_train_verified": int((tr.role == "v").sum()),
        "n_test": len(te), "n_eval_rows": len(ev), "freesound_id_parsed": int((al.freesound_id != "").sum()),
        "train_with_uploader": int(tr.group.str.startswith("u:").sum()), "n_groups_train": int(tr.group.nunique()),
        "fold_sizes": trn.folder.value_counts().sort_index().tolist(),
        "verified_fold_sizes": trn[trn.role == "v"].folder.value_counts().sort_index().tolist(),
        "verified_per_class_per_fold_min": int(ct_v.values.min()), "test_per_class_min": int(te.class_id.value_counts().min())}
(out / "classes.json").write_text(json.dumps(info, indent=1))
ct_v.to_csv(out / "verified_class_by_fold.csv")
print(json.dumps({k: v for k, v in info.items() if k != "classes"}, indent=1))
