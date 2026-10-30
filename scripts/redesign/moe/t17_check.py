"""T17 input and reproduction check (no accuracy, no test label). python t17_check.py --out DIR
1. every MN10 and MN20 embedding file used: shape (rows, dim), finite, no all-zero row; row count equals the bed; 2. T1c split files exist for all seed-folds and `tr`/`te` follow the fold column;
3. one-member group equals heads.fit_head(metric='acc') to 0.0 on UrbanSound8K seed 0 fold 1 (training and inner-validation clips only)."""
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from t17_common import *
from heads import fit_head
from t17_heads import train_group
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
rep = {}; files = []
for bed in BEDS:
    D = load_bed(bed); N = D["N"]; rep[bed] = {}
    for ex in ("E1", "M20"):
        for cell in cells_of(bed):
            p = emb17(bed, ex, cell); X = np.load(p, mmap_mode="r"); ok = X.shape[0] == N and np.isfinite(X).all() and (np.abs(X).sum(1) > 0).all()
            rep[bed][f"{ex}|{cell}"] = dict(shape=list(X.shape), ok=bool(ok)); files.append(p); assert ok, (bed, ex, cell, X.shape)
    for seed in (0, 1, 2):
        for fold in range(1, BEDS[bed]["nf"] + 1):
            S = np.load(T1C / bed / f"raw_strength/s{seed}_f{fold}.npz"); O = np.load(T1C / bed / f"raw_oof/s{seed}_f{fold}.npz")
            tr = np.flatnonzero((D["fold"] != fold) & D["is_a"]) if bed == "tau" else np.flatnonzero(D["fold"] != fold)
            assert (S["tr"] == tr).all() and (S["te"] == np.flatnonzero(D["fold"] == fold)).all() and (O["te"] == S["te"]).all() and O["E2a_oof"].shape == (len(tr), BEDS[bed]["ncls"])
            files += [T1C / bed / f"raw_strength/s{seed}_f{fold}.npz", T1C / bed / f"raw_oof/s{seed}_f{fold}.npz"]
    print(bed, "ok", flush=True)
D = load_bed("urban"); y = D["y"]; S = np.load(T1C / "urban/raw_strength/s0_f1.npz"); tr = S["tr"]; fg, ig = tr[S["fit_p"]], tr[S["iv_p"]]; X = np.load(emb17("urban", "E1", "clean")).astype(np.float32)
h = fit_head(X[fg], y[fg], np.ones(len(fg)), 0, X[ig], y[ig], metric="acc", ncls=10); G = train_group([X[fg]], y[fg], [np.ones(len(fg))], 0, [X[ig]], y[ig], 10)
rep["R1_group_equals_fit_head_max_abs_diff"] = float(np.abs(h.logits(X[ig]) - G.logits([X[ig]])[0]).max()); assert rep["R1_group_equals_fit_head_max_abs_diff"] == 0.0
json.dump(rep, open(a.out / "check_inputs.json", "w"), indent=1)
open(a.out / "file_list.csv", "w").write("path,sha256\n" + "\n".join(f"{p},{hashlib.sha256(open(p, 'rb').read()).hexdigest()}" if p.stat().st_size < 3e8 else f"{p},skipped_large" for p in files))
print("check ok, R1 diff", rep["R1_group_equals_fit_head_max_abs_diff"])
