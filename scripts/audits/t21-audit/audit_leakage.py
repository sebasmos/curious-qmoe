"""Leakage and fold checks for T21 D1. Own code. Usage: python -I audit_leakage.py <results_dir> <dataset_dir>"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
R, DS = Path(sys.argv[1]), Path(sys.argv[2])
M = DS / "meta_x/FSDKaggle2018.meta"
tr = pd.read_csv(M / "train_post_competition.csv"); te = pd.read_csv(M / "test_post_competition_scoring_clips.csv")
print("meta rows train/test:", len(tr), len(te), "; verified train:", int(tr.manually_verified.sum()))
pool_m = pd.concat([tr[tr.manually_verified == 1].assign(official="train"), te.assign(official="test")]).sort_values("fname").reset_index(drop=True)
print("verified pool size:", len(pool_m), "; freesound ids shared train/test:", len(set(tr.freesound_id) & set(te.freesound_id)))
skf = StratifiedKFold(5, shuffle=True, random_state=42)
fold = np.zeros(len(pool_m), int)
for k, (_, t) in enumerate(skf.split(np.zeros(len(pool_m)), pool_m.label), 1):
    fold[t] = k
pool = pd.read_csv(R / "csv/d1/pool.csv")
print("pool.csv names equal my sorted verified pool:", list(pool.name) == list(pool_m.fname))
print("my StratifiedKFold(5, shuffle, 42) folds equal pool.csv folder:", np.array_equal(fold, pool.folder.to_numpy()))
print("labels equal:", list(pool.label) == list(pool_m.label))
# head CSV row alignment and split files
for head in ("mn10", "ast"):
    h = pd.read_csv(R / f"csv/d1/heads/pool_{head}.csv", usecols=lambda c: not c.isdigit())
    print(head, "head CSV columns:", list(h.columns)[:8], "; rows", len(h))
    nm = h[[c for c in h.columns if "name" in c.lower() or "file" in c.lower()][0]].astype(str).str.replace(".wav", "", regex=False)
    print(head, "head CSV row order equals pool.csv:", list(nm) == list(pool.name.str.replace(".wav", "", regex=False)))
    for s in range(5):
        for f in range(1, 6):
            sp = json.loads((R / f"outputs/d1_pool_{head}_s{s}/esc/fold_{f}/split.json").read_text())
            keys = list(sp)
            T = set(sp["test_idx"]); IV = set(sp["inner_val_idx"]); TR = set(sp.get("train_idx", sp.get("inner_train_idx", [])))
            assert T == set(np.flatnonzero(pool.folder.to_numpy() == f)), (head, s, f)
            assert not (T & IV) and not (T & TR) and not (IV & TR), (head, s, f)
            if TR: assert len(T | IV | TR) == len(pool), (head, s, f)
    print(head, "split.json keys:", keys, "; test = fold rows, train/iv/test disjoint and covering, 25 runs: True")
# int8 calibration lists (secondary path) and duplicates across folds by FP32 embedding
cal = list((R / "models/d1").rglob("*calib*"))
print("int8 calibration files:", [str(c.relative_to(R)) for c in cal][:6])
X = np.load(R / "emb/d1/mn_pool/mn10_fp32__clean.npy").astype(np.float64)
Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
F = pool.folder.to_numpy()
best = np.full(len(X), -1.0); arg = np.zeros(len(X), int)
for i0 in range(0, len(X), 500):
    S = Xn[i0:i0 + 500] @ Xn.T
    for j in range(S.shape[0]):
        S[j, F == F[i0 + j]] = -1
    arg[i0:i0 + 500] = S.argmax(1); best[i0:i0 + 500] = S.max(1)
for th in (0.9999, 0.999, 0.99):
    m = best > th
    same = (pool.label.to_numpy()[m] == pool.label.to_numpy()[arg[m]]).mean() if m.any() else np.nan
    print(f"clips with a cross-fold neighbour of cosine > {th}: {int(m.sum())} (same label: {same:.2f})")
Xa = np.load(R / "emb/d1/ast_pool/fp32__clean.npy").astype(np.float64); Xa /= np.linalg.norm(Xa, axis=1, keepdims=True)
cand = np.flatnonzero(best > 0.999)
print("of those > 0.999 on MN10, AST FP32 cosine > 0.999 too:", int(sum(Xa[i] @ Xa[arg[i]] > 0.999 for i in cand)))
pd.DataFrame({"name": pool.name[cand], "fold": F[cand], "nn_name": pool.name.to_numpy()[arg[cand]], "nn_fold": F[arg[cand]],
              "cos_mn": best[cand], "label": pool.label[cand], "nn_label": pool.label.to_numpy()[arg[cand]]}).to_csv(Path(sys.argv[3]) / "crossfold_near_duplicates.csv", index=False)
