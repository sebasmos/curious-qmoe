"""Completeness check of every Stage 3b embedding file: rows equal the dataset's clip count, the names checksum
equals the dataset CSV's, finite rows equal the rows the sidecar says were extracted (UrbanSound8K field cells hold
NaN outside the clips the field assignments need), and the expected set of files per folder exists.

    python scripts/redesign/idea6/stage3b/check_embeddings.py R [--out R/logs/check_embeddings.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

N = {"esc50": 2000, "urban8k": 8732}
CRC = {"esc50": 1212562368, "urban8k": 4194399870}
NOISY = [f"{k}__{s}db" for k in ("bg", "pink", "white") for s in (10, 5, 0, 20)]
FIELD_ONLY = ["bg__5db", "pink__10db", "pink__5db", "pink__0db", "white__10db", "white__5db", "white__0db"]


def expected(folder):
    ds = "esc50" if folder.startswith("esc50") else "urban8k"
    if folder.startswith("emb_cnn"):
        return [f"{m}_{c}__{k}" for m in ("mn04", "mn10") for c in ("fp32", "w8a8") for k in ["clean"] + NOISY]
    conds = ["clean"] + (NOISY if ds == "esc50" else ["bg__10db", "bg__0db"] + FIELD_ONLY)
    return ["fp32__clean"] + [f"{c}__{k}" for c in ("n4", "w4g128") for k in conds]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res, bad = {}, []
    folders = [f"emb_ast/{d}" for d in ("esc50_256", "esc50_384", "esc50_256o256", "esc50_256x2", "urban8k_256",
                                        "urban8k_384")] + ["emb_cnn/esc50", "emb_cnn/urban8k"]
    for fo in folders:
        ds = "esc50" if "esc50" in fo else "urban8k"
        key = fo.split("/")[1] if fo.startswith("emb_ast") else "emb_cnn_" + ds
        want = expected(key if fo.startswith("emb_ast") else "emb_cnn")
        have = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(a.results, fo, "*__*.npy"))
                if not os.path.basename(p).startswith(("lengths", "_"))}
        missing = sorted(set(want) - have)
        rows = {}
        for name in sorted(have):
            p = os.path.join(a.results, fo, name + ".npy")
            x = np.load(p, mmap_mode="r")
            s = json.load(open(p[:-4] + ".json"))
            fin = int(np.isfinite(np.asarray(x)).all(1).sum())
            ok = (x.shape[0] == N[ds] and s["names_crc32"] == CRC[ds]
                  and fin == s.get("rows_extracted", s["n_clips"]))
            rows[name] = {"shape": list(x.shape), "finite_rows": fin, "ok": bool(ok)}
            if not ok:
                bad.append(f"{fo}/{name}")
        res[fo] = {"n_files": len(have), "missing": missing, "files": rows}
        print(f"{fo}: {len(have)} files, missing {missing}, bad {[n for n, r in rows.items() if not r['ok']]}")
    print("bad files:", bad)
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
