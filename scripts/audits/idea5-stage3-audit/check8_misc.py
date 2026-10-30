"""Check 8: data hygiene.
  - silent clips left clean (every extraction sidecar)
  - non-finite rows in every full-condition embedding file; partial (field-only) files listed
  - identical embedding rows inside a file (duplicate clips), and noisy files equal to clean
  - background plan: how many distinct background clips, reuse counts
  - pair counts per dataset in Stage 2's static.csv / cascades.csv, and duplicated rows
  - seeds: share of test clips on which two seeds' FP32 heads disagree (clean, fp32 backbone)
  - Stage 2 README placeholders

    python check8_misc.py STAGE2_DIR OUT_JSON
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

s2, out = Path(sys.argv[1]), Path(sys.argv[2])
res = {}
for ds in ["esc50", "urban8k"]:
    d = s2 / "emb" / ds
    sil, partial, nonfinite, dup, same_as_clean = {}, [], [], {}, []
    for j in sorted(d.glob("*.json")):
        if j.name == "bg_plan.json":
            continue
        side = json.loads(j.read_text())
        if side.get("silent_clips_left_clean"):
            sil[j.stem] = side["silent_clips_left_clean"]
    for f in sorted(d.glob("*.npy")):
        x = np.load(f, mmap_mode="r")
        fin = np.isfinite(x).all(1)
        if not fin.all():
            if fin.any() and (~fin).sum() < len(x):
                partial.append(f"{f.stem}: {int(fin.sum())} rows")
            else:
                nonfinite.append(f.stem)
            continue
        u = np.unique(np.asarray(x), axis=0)
        if len(u) < len(x):
            dup[f.stem] = int(len(x) - len(u))
        cfg, cond = f.stem.split("__", 1)
        if cond != "clean" and (d / f"{cfg}__clean.npy").exists():
            c = np.load(d / f"{cfg}__clean.npy", mmap_mode="r")
            if np.allclose(np.asarray(x), np.asarray(c)):
                same_as_clean.append(f.stem)
    plan = json.loads((d / "bg_plan.json").read_text())
    cnt = Counter(v["bg_clip"] for v in plan.values())
    res[ds] = {"silent_clips_left_clean": sil, "partial_files": len(partial), "partial_examples": partial[:6],
               "files_with_nonfinite_rows_not_partial": nonfinite, "files_with_duplicate_rows": dup,
               "noisy_files_equal_to_clean": same_as_clean, "bg_plan_clips": len(plan),
               "bg_distinct_sources": len(cnt), "bg_max_reuse": max(cnt.values()),
               "bg_mean_reuse": float(np.mean(list(cnt.values())))}
for ds in ["esc50", "esc5", "urban"]:
    S = pd.read_csv(s2 / "analysis" / ds / "static.csv")
    C = pd.read_csv(s2 / "analysis" / ds / "cascades.csv")
    pairs = S.groupby(["condition", "config", "head"]).size()
    res[f"{ds}_pairs_per_static_system"] = sorted(set(pairs.tolist()))
    res[f"{ds}_static_duplicated_rows"] = int(S.duplicated(["condition", "config", "head", "seed", "fold"]).sum())
    res[f"{ds}_cascade_duplicated_rows"] = int(C.duplicated(["condition", "cheap", "expensive", "head", "signal",
                                                              "point", "seed", "fold"]).sum())
    res[f"{ds}_nan_f1"] = int(S.f1.isna().sum() + C.f1.isna().sum())
    res[f"{ds}_seeds"] = sorted(S.seed.unique().tolist())
    res[f"{ds}_folds"] = sorted(S.fold.unique().tolist())
    sc = s2 / "scores" / ds
    tags = sorted(p.name for p in sc.iterdir() if p.is_dir() and "headw4" not in p.name)
    preds = [np.load(sc / t / "esc" / "fp32__clean.npz")["test_logits"].argmax(1) for t in tags]
    dis = [float((preds[i] != preds[j]).mean()) for i in range(len(preds)) for j in range(i + 1, len(preds))]
    res[f"{ds}_seed_pairwise_disagreement_fp32_clean"] = [round(min(dis), 4), round(max(dis), 4)]
txt = (s2 / "README.md").read_text()
res["stage2_readme_placeholders"] = [p for p in ("UPDATE_VERDICT", "RESULTS") if p in txt.split()]
Path(out).write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
