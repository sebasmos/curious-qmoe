"""T19 step 19.6 checks (ANALYSIS_PLAN_T19.md B9): row counts equal the CSVs, names checksums equal across configurations,
no NaN inside extracted rows, and which rows each file holds. Writes <results>/tables/emb_check.csv; exits 1 on a failed check.

    python t19_check_emb.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t19_common import TASKS, results_dir  # noqa: E402

R = results_dir()
rows, bad = [], []
for task in ("esc50", "urban", "tau"):
    T = TASKS[task]
    n_csv = len(pd.read_csv(T["csv"], usecols=[0]))
    d = R / "emb" / T["emb"]
    crcs = set()
    for f in sorted(d.glob("*__*.npy")):
        if f.name.endswith(".tmp.npy"):
            continue
        if not f.with_suffix(".json").exists():
            rows.append(dict(task=task, file=f.name, note="being written, no sidecar yet"))
            continue
        E = np.load(f, mmap_mode="r")
        meta = json.loads(f.with_suffix(".json").read_text())
        fin = np.isfinite(E).all(1)
        anynan_in_rows = bool((np.isnan(E).any(1) & ~np.isnan(E).all(1)).any())
        crcs.add(meta["names_crc32"])
        r = dict(task=task, file=f.name, rows=len(E), rows_csv=n_csv, rows_complete=int(fin.sum()),
                 rows_meta=meta["rows_extracted_total"], partial_nan_rows=anynan_in_rows, crc=meta["names_crc32"])
        rows.append(r)
        if len(E) != n_csv or anynan_in_rows or (f.stem.endswith("__clean") and not fin.all()):
            bad.append(r)
    if len(crcs) > 1:
        bad.append(dict(task=task, file="*", crc=sorted(crcs)))
out = R / "tables" / "emb_check.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"{len(rows)} files checked, {len(bad)} problems -> {out}")
for b in bad:
    print("PROBLEM", b)
sys.exit(1 if bad else 0)
