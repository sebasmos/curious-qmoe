"""Exact and near duplicates across D2 folds: md5 of each wav file, md5 of its PCM data chunk (own RIFF parser,
files read as bytes only), and MN10 FP32 clean embedding cosine across folds.
Usage: python -I audit_dups_d2.py <T21 results> <FSD50K audio_raw/dev> <audit out>
"""
import hashlib
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R, AUD, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
pool = pd.read_csv(R / "csv/d2/pool.csv", dtype={"uploader": str})
log = open(OUT / "audit_dups.log", "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")


def pcm_md5(b):
    if b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        return None
    i = 12
    while i + 8 <= len(b):
        cid, size = b[i:i + 4], struct.unpack("<I", b[i + 4:i + 8])[0]
        if cid == b"data":
            return hashlib.md5(b[i + 8:i + 8 + size]).hexdigest()
        i += 8 + size + (size & 1)
    return None


fm, pm = [], []
for n in pool.name:
    b = (AUD / n).read_bytes()
    fm.append(hashlib.md5(b).hexdigest())
    pm.append(pcm_md5(b))
pool["md5"], pool["pcm_md5"] = fm, pm
P("files hashed", len(pool), "PCM chunk found", int(pool.pcm_md5.notna().sum()))
for col in ("md5", "pcm_md5"):
    g = pool[pool.duplicated(col, keep=False) & pool[col].notna()].groupby(col)
    groups = [d for _, d in g]
    cross = [d for d in groups if d.folder.nunique() > 1]
    P(f"{col}: duplicate groups {len(groups)} (clips {sum(len(d) for d in groups)}), spanning folds {len(cross)} (clips {sum(len(d) for d in cross)}), "
      f"with different labels {sum(d.class_id.nunique() > 1 for d in groups)}")
    for d in cross[:10]:
        P("   ", d[["name", "folder", "category", "uploader"]].to_dict("records"))
pool[["name", "folder", "category", "uploader", "md5", "pcm_md5"]].to_csv(OUT / "pool_hashes.csv.gz", index=False)
# embedding near duplicates across folds
X = np.load(R / "emb/d2/mn_pool/mn10_fp32__clean.npy").astype(np.float32)
X /= np.linalg.norm(X, axis=1, keepdims=True)
fold = pool.folder.to_numpy()
y = pool.class_id.to_numpy()
best = np.full(len(X), -1.0, np.float32)
arg = np.full(len(X), -1)
for i in range(0, len(X), 2048):
    S = X[i:i + 2048] @ X.T
    S[fold[i:i + 2048][:, None] == fold[None, :]] = -2
    j = S.argmax(1)
    best[i:i + 2048], arg[i:i + 2048] = S[np.arange(len(j)), j], j
rows = pd.DataFrame({"name": pool.name, "folder": fold, "category": pool.category, "uploader": pool.uploader, "best_cross_cos": best,
                     "partner": pool.name.to_numpy()[arg], "partner_folder": fold[arg], "partner_category": pool.category.to_numpy()[arg],
                     "partner_uploader": pool.uploader.to_numpy()[arg]})
rows.to_csv(OUT / "crossfold_nearest.csv.gz", index=False)
for t in (0.9999, 0.999, 0.995):
    sel = rows[rows.best_cross_cos > t]
    P(f"clips with a cross-fold partner above cosine {t}: {len(sel)}, same label {int((sel.category == sel.partner_category).sum())}")
sel = rows[rows.best_cross_cos > 0.999].sort_values("best_cross_cos", ascending=False)
sel.head(40).to_csv(OUT / "crossfold_near_duplicates_top.csv", index=False)
P(sel.head(15).to_string())
log.close()
