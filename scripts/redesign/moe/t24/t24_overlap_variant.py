"""T24 descriptive overlap variant, NOT pre-registered (written after the D3 scores, because the pre-registered C9 drop
removes 940 of 947 D3 test clips, almost all through FSD50K): the field and clean comparisons on the test clips left
after dropping only those shared with FSDKaggle2018 or with the UrbanSound8K noise bank, or mixed with bg noise from a
recording that is itself a D3 clip. FSD50K overlap is kept (no model here trains on FSD50K). Writes
tables/<d>/overlapdrop_keepfsd50k_*.csv and overlap_variant.json.   python t24_overlap_variant.py --r R --d d3 --seeds 0 1 2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t24_analyze as T  # noqa: E402

DATA = Path.home() / "Documents/DATASETS"
ap = argparse.ArgumentParser()
ap.add_argument("--r", type=Path, required=True)
ap.add_argument("--d", required=True)
ap.add_argument("--seeds", nargs="+", type=int, required=True)
a = ap.parse_args()
al = pd.read_csv(a.r / "csv" / a.d / "all.csv", dtype={"freesound_id": str})
k18 = set()
for f in ("train_post_competition.csv", "test_post_competition_scoring_clips.csv"):
    k18 |= set(pd.read_csv(DATA / "FSDKaggle2018/meta_x/FSDKaggle2018.meta" / f).freesound_id.astype(str))
u8 = {Path(n).stem.split("-")[0] for n in pd.read_csv(DATA / "Urban8k/ast_768/urbansound8k.csv", usecols=["name"])["name"]}
plan = json.loads((a.r / "emb" / a.d / "ast_eval" / "bg_plan.json").read_text())
own = set(al.freesound_id.dropna())
bg_own = {k for k, v in plan.items() if str(v.get("bg_clip", "")).split("-")[0] in own}
te = al[al.folder == 0]
drop = te.freesound_id.isin(k18) | te.freesound_id.isin(u8) | te.name.map(lambda s: Path(s).stem).isin(bg_own)
keep = ~drop.to_numpy()
costs, _ = T.per_clip_costs(a.r, a.d, int(al.class_id.nunique()))
od = T.analyze_design(a.r, a.d, a.seeds, costs, int(al.class_id.nunique()), a.r / "tables" / a.d, "overlapdrop_keepfsd50k",
                      keep=keep, conds=["clean", "field"])[0]
res = {"note": "descriptive, not pre-registered", "n_dropped": int(drop.sum()), "n_test_left": int(keep.sum()),
       "field": T.summary_row(od[od.cond == "field"].iloc[0]), "clean": T.summary_row(od[od.cond == "clean"].iloc[0])}
(a.r / "tables" / a.d / "overlap_variant.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
