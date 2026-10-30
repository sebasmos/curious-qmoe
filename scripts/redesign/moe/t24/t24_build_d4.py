"""T24 D4 (TUT Acoustic Scenes 2017, Zenodo 400515 development and 1040168 evaluation) metadata CSVs
(ANALYSIS_PLAN.md C2). Reads only the two meta.txt files (tab separated: file, scene label, recording location id;
the evaluation file may lack the third column).

all.csv   development rows (sorted) then evaluation rows (sorted). folder 1..5 for development
          (StratifiedGroupKFold(5, shuffle=True, random_state=42) by scene, grouped by recording location), 0 for
          evaluation. role "v" for every development clip (all labels are reliable), "test" for evaluation clips.
          name "dev-<stem>.wav" or "eval-<stem>.wav" so the two parts cannot collide.
eval.csv  every row (all clips are inner-validation or test rows).

    python -I t24_build_d4.py DEV_META_TXT EVAL_META_TXT OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def read(p, part):
    rows = [ln.rstrip("\r\n").split("\t") for ln in Path(p).read_text().splitlines() if ln.strip()]
    d = pd.DataFrame({"file": [r[0] for r in rows], "scene": [r[1] for r in rows],
                      "loc": [r[2] if len(r) > 2 else "" for r in rows]})
    d["part"], d["src"] = part, d.file.map(lambda s: Path(s).name)
    d["name"] = ("dev-" if part == "dev" else "eval-") + d.src.map(lambda s: Path(s).stem) + ".wav"
    return d


dv, ev = read(sys.argv[1], "dev"), read(sys.argv[2], "eval")
out = Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
assert dv.src.is_unique and ev.src.is_unique and (dv["loc"] != "").all()
classes = sorted(dv.scene.unique())
assert classes == sorted(ev.scene.unique())
cid = {c: i for i, c in enumerate(classes)}
for d in (dv, ev):
    d["category"], d["class_id"] = d.scene, d.scene.map(cid)
    d["label"] = d.class_id
    d["freesound_id"], d["manually_verified"] = "", 1
dv["group"], ev["group"] = "l:" + dv["loc"], "e:" + ev.src
dv["role"], ev["role"] = "v", "test"
dv = dv.sort_values("name").reset_index(drop=True)
ev = ev.sort_values("name").reset_index(drop=True)
fold = np.zeros(len(dv), int)
for k, (_, tei) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(np.zeros(len(dv)), dv.class_id, dv.group), 1):
    fold[tei] = k
dv["folder"], ev["folder"] = fold, 0
cols = ["folder", "name", "label", "category", "class_id", "role", "group", "freesound_id", "manually_verified", "part", "src"]
al = pd.concat([dv[cols], ev[cols]], ignore_index=True)
assert al.name.is_unique and dv.groupby("group").folder.nunique().max() == 1
al.to_csv(out / "all.csv", index=False)
al.to_csv(out / "eval.csv", index=False)
ct = pd.crosstab(dv.category, dv.folder)
info = {"dataset": "TUT Acoustic Scenes 2017", "classes": classes, "n_classes": len(classes), "n_dev": len(dv), "n_eval": len(ev),
        "n_locations": int(dv.group.nunique()), "fold_sizes": dv.folder.value_counts().sort_index().tolist(),
        "per_class_per_fold_min": int(ct.values.min()), "eval_per_class_min": int(ev.class_id.value_counts().min())}
(out / "classes.json").write_text(json.dumps(info, indent=1))
ct.to_csv(out / "class_by_fold.csv")
print(json.dumps({k: v for k, v in info.items() if k != "classes"}, indent=1))
