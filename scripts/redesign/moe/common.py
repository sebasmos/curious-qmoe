"""Shared paths and helpers for the difficulty mixture-of-experts pilot (T0, T1a, T1). Stored results are read only."""
from __future__ import annotations

import hashlib
import json
import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

Q = Path(os.path.expanduser("~/Documents/qmoe-results"))
FINAL = Q / "2026-10-24_1507_redesign-idea6-final_code-ed82400"
S2 = Q / "2026-10-22_1120_redesign-idea6-stage2_code-9f346ba"
S3B = Q / "2026-10-22_1652_redesign-idea6-stage3b_code-07dc1db"
S3C = Q / "2026-10-23_1227_redesign-idea6-stage3c_code-9c9cbf1"
I5 = Q / "2026-10-20_2359_redesign-idea5-stage2_code-3c18c38"
OPTB = Q / "2026-10-23_2133_redesign-idea6-optionB_code-f320295"
DS = Path(os.path.expanduser("~/Documents/DATASETS/Urban8k"))
CSV = DS / "ast_768" / "urbansound8k.csv"
META = DS / "audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/metadata/UrbanSound8K.csv"

# stored clean embeddings used by T1 (row order = CSV order)
EMB = {
    "mn10_w8a8": S3B / "emb_cnn/urban8k/mn10_w8a8__clean.npy",
    "mn10_fp32": S3B / "emb_cnn/urban8k/mn10_fp32__clean.npy",
    "ast256_w4g128": S3B / "emb_ast/urban8k_256/w4g128__clean.npy",
    "ast384_w4g128": S3B / "emb_ast/urban8k_384/w4g128__clean.npy",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_table() -> pd.DataFrame:
    d = pd.read_csv(CSV, usecols=["folder", "name", "label", "category", "class_id"])
    d["stem"] = d["name"].str.replace(".png", "", regex=False)
    d["recording"] = d["name"].str.split("-").str[0].astype(int)  # source recording id (token 0)
    return d


def names_crc(d: pd.DataFrame) -> int:
    return zlib.crc32("\n".join(d["stem"]).encode())


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def pmargin(p):
    """Probability margin top1 - top2 (the repo's gate, analyze.py margin)."""
    s = np.sort(p, 1)
    return s[:, -1] - s[:, -2]


def f1_macro(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1)
    ok = den > 0
    return float(np.mean(2 * tp[ok] / den[ok]))


def write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
