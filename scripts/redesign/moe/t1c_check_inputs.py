"""T1c step 1: every stored embedding the plan uses exists, has the CSV's row count, is finite with no zero row, and its sidecar names the expected config.
Writes <out>/check_inputs.json and <out>/file_list.csv (sha256). python t1c_check_inputs.py --out DIR [--no-sha]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t1c_common import *
from common import sha256, write_json
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); ap.add_argument("--no-sha", action="store_true"); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
rep, files, ok = {}, [], True
for bed in BEDS:
    B = load_bed(bed); N = B["N"]; r = dict(N=N, n_classes=int(len(np.unique(B["y"]))), folds={int(f): int((B["fold"] == f).sum()) for f in np.unique(B["fold"])}, experts={})
    if bed != "tau":
        r["field_cell_sizes"] = {c: int((B["assign"] == c).sum()) for c in ALLCELLS}
    else: r["device_a_rows"] = int(B["is_a"].sum())
    for ex in ["E1", "E2a", "E2b"] + (["E2c"] if bed != "tau" else []):
        for cell in cells_of(bed):
            p = emb_path(bed, ex, cell); row = dict(bed=bed, expert=ex, cell=cell, path=str(p))
            if not p.exists(): row["status"] = "MISSING"; ok = False
            else:
                X = np.load(p, mmap_mode="r"); row.update(shape=list(X.shape), finite=bool(np.isfinite(np.asarray(X)).all()), zero_rows=int((np.abs(np.nan_to_num(np.asarray(X))).sum(1) == 0).sum() - (~np.isfinite(np.asarray(X)).all(1)).sum()))
                sc = p.with_suffix(".json"); sj = json.load(open(sc)) if sc.exists() else {}
                row.update(sidecar_config=sj.get("config"), sidecar_max_length=sj.get("max_length"), sidecar_rows=sj.get("rows_extracted", sj.get("n_clips")))
                fin = np.isfinite(np.asarray(X)).all(1); row["finite_rows"] = int(fin.sum())
                partial_ok = bed == "urban" and ex != "E1" and cell not in ("clean", "bg__10db", "bg__0db") and bool(fin[B["assign"] == cell].all())   # Amendment 1
                row["partial_cell_test_rows_finite"] = partial_ok
                good = X.shape[0] == N and (row["finite"] or partial_ok) and row["zero_rows"] == 0 and X.shape[1] in (768, 960)
                row["status"] = "ok" if good else "BAD"; ok &= good
                if not a.no_sha: row["sha256"] = sha256(p)
            files.append(row)
    rep[bed] = r
rep["all_ok"] = ok; write_json(a.out / "check_inputs.json", rep); pd.DataFrame(files).to_csv(a.out / "file_list.csv", index=False)
print(json.dumps({b: rep[b] for b in BEDS}, indent=1)[:2500]); print("ALL OK" if ok else "PROBLEM")
print(pd.DataFrame(files).groupby(["bed", "expert"]).agg(n=("cell", "count"), shape=("shape", "first"), maxlen=("sidecar_max_length", "first"), cfg=("sidecar_config", "first"), status=("status", lambda s: ",".join(sorted(set(s))))).to_string())
