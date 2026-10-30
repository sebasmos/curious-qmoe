"""Step 1: every UrbanSound8K clip has an embedding in both backbones, the folds are the 10 official folds, the score files
agree with the CSV. Writes check_inputs.json and file_list.csv (path, bytes, sha256) to --out. Read only on stored results."""
import argparse, glob, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
d = load_table()
res, files, ok = {}, [], True
def check(name, cond, detail=""):
    global ok
    res[name] = {"pass": bool(cond), "detail": detail}
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name, detail)

check("csv rows", len(d) == 8732, f"{len(d)} rows")
check("csv names unique", d.stem.is_unique)
m = pd.read_csv(META)
m["stem"] = m["slice_file_name"].str.replace(".wav", "", regex=False)
j = d.merge(m, on="stem", how="outer", indicator=True)
check("every csv clip is in UrbanSound8K metadata", (j["_merge"] == "both").all(), f"{(j['_merge']=='both').sum()} matched")
check("csv folder == official metadata fold", (j["folder"] == j["fold"]).all())
check("csv label == official classID", (j["label"] == j["classID"]).all() and (d.label == d.class_id).all())
check("ten folds, sizes", sorted(d.folder.unique()) == list(range(1, 11)), str(d.folder.value_counts().sort_index().to_dict()))
# protocol 3: no recording on both sides of the inner split is by construction (grouped by recording id); official folds are
# grouped by source recording already
rec_folds = d.groupby("recording").folder.nunique()
# informational: 5 Freesound ids (152 clips) appear in two official folds; the folds themselves are the official ones, so nothing is changed
res["recordings_in_more_than_one_official_fold"] = {"n_ids": int((rec_folds > 1).sum()), "n_clips": int(d.recording.isin(rec_folds[rec_folds > 1].index).sum()), "n_ids_total": int(len(rec_folds))}
print("info", res["recordings_in_more_than_one_official_fold"])
crc = names_crc(d)
res["names_crc32_csv"] = crc
for k, p in EMB.items():
    x = np.load(p, mmap_mode="r")
    meta = json.loads(p.with_suffix(".json").read_text())
    check(f"embedding {k}", x.shape[0] == 8732 and np.isfinite(np.asarray(x)).all() and (np.abs(np.asarray(x)).sum(1) > 0).all()
          and meta.get("names_crc32") == crc, f"shape {x.shape}, crc in json {meta.get('names_crc32')} vs csv {crc}")
    files += [p, p.with_suffix(".json")]
# score files: rows, folds, labels
score_dirs = {
    "mn10 w8a8 head esc": [S3B / f"scores_cnn/urban/mn10_urban_s{s}/esc/mn10_w8a8__clean.npz" for s in range(3)],
    "ast256 w4g128 head esc": [S3B / f"scores_ast256/urban/ast256_urban_s{s}/esc/w4g128__clean.npz" for s in range(3)],
    "ast384 w4g128 head esc": [S3B / f"scores_ast384/urban/ast384_urban_s{s}/esc/w4g128__clean.npz" for s in range(3)],
    "ast fp32 1024 head esc (P3 tier)": [I5 / f"scores/urban/asturban_s{s}/esc/fp32__clean.npz" for s in range(3)],
}
for k, ps in score_dirs.items():
    good = True
    for p in ps:
        z = np.load(p)
        good &= (z["y"] == d.label.values).all() and (z["test_fold"] == d.folder.values).all() and ("names_crc32" not in z.files or int(z["names_crc32"]) == crc)
        files.append(p)
    check(f"scores {k}: y and fold equal the CSV", good, f"{len(ps)} seeds, names checksum {'present' if 'names_crc32' in z.files else 'absent in this file'}")
files.append(CSV)
pd.DataFrame([{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]).to_csv(a.out / "file_list.csv", index=False)
res["all_pass"] = ok
write_json(a.out / "check_inputs.json", res)
print("ALL PASS" if ok else "SOME FAIL")
sys.exit(0 if ok else 1)
