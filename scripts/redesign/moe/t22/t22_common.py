"""T22 shared definitions (ANALYSIS_PLAN.md): tasks, audio paths, the stored tiers' arrays (read only), rows and inner splits."""
from __future__ import annotations

import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in (ROOT, HERE.parent, ROOT / "scripts/redesign/idea6/stage2", ROOT / "scripts/redesign/idea6/final", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

E = Path.home() / "Documents/DATASETS"
URB_CSV, URB_AUD = E / "Urban8k/ast_768/urbansound8k.csv", E / "Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio"
ESC_CSV, ESC_AUD = E / "ESC-50/ast_768/esc-50.csv", E / "ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio"
TAU_CSV = E / "TAU2020Mobile/ast_768/tau2020mobile.csv"
TAU_AUD = E / "TAU2020Mobile/audio_raw/tau2020mobile_zenodo_3819968/TAU-urban-acoustic-scenes-2020-mobile-development/audio"
WEIGHTS = Path.home() / "Documents/qmoe-weights/efficientat_release-v0.0.1/mn10_as_mAP_471.pt"
CACHE = Path(os.environ.get("T22_CACHE", Path.home() / "Documents/qmoe-work/t22-cache"))
TASKS = {
    "esc50": dict(csv=ESC_CSV, aud=ESC_AUD, bg_csv=URB_CSV, bg_aud=URB_AUD, K=50, nf=5, epochs=60),
    "urban": dict(csv=URB_CSV, aud=URB_AUD, bg_csv=ESC_CSV, bg_aud=ESC_AUD, K=10, nf=10, epochs=30),
    "tau": dict(csv=TAU_CSV, aud=TAU_AUD, bg_csv=None, bg_aud=None, K=10, nf=5, epochs=20),
}
SEEDS = [0, 1, 2]
SNRS = (20, 10, 5, 0)
NOISE = [f"{k}__{s}db" for k in ("bg", "pink", "white") for s in SNRS]
CORR = ["reverb_rt300ms", "reverb_rt600ms", "reverb_rt1200ms", "clip_p50", "clip_p25", "clip_p10", "lowpass_4000hz", "lowpass_2000hz",
        "lowpass_1000hz"]
CORR_ARG = {"reverb_rt300ms": ("reverb", 0.3), "reverb_rt600ms": ("reverb", 0.6), "reverb_rt1200ms": ("reverb", 1.2), "clip_p50": ("clip", 0.5),
            "clip_p25": ("clip", 0.25), "clip_p10": ("clip", 0.1), "lowpass_4000hz": ("lowpass", 4000.0), "lowpass_2000hz": ("lowpass", 2000.0),
            "lowpass_1000hz": ("lowpass", 1000.0)}


def cond_arg(c: str):
    """Condition name -> (kind, level) for the mixer / corruption code."""
    if c == "clean":
        return "clean", None
    if c in CORR_ARG:
        return CORR_ARG[c]
    k, s = c.split("__")
    return k, float(s[:-2])


def stems(task):
    return [Path(s).stem for s in pd.read_csv(TASKS[task]["csv"], usecols=["name"])["name"]]


def names_crc(task):
    return zlib.crc32("\n".join(stems(task)).encode())


def bed(task):
    """y, fold (and TAU device) from the stored MN10-Q8 seed-0 clean array; checked against every other stored file in the alignment step."""
    import t13_common as T13

    D = T13.load(task, "mn10", "clean", 0)
    out = {"y": D["y"], "fold": D["fold"], "N": len(D["y"]), "crc": D["crc"]}
    out["is_a"] = (D["device"] == "a") if task == "tau" else np.ones(len(D["y"]), bool)
    if task == "tau":
        out["device"] = D["device"]
    return out


def inner_split(task, seed, fold):
    """Stored grouped inner-validation rows (global, sorted) of (seed, fold): MN10-Q8's clean array of that seed."""
    import t13_common as T13

    D = T13.load(task, "mn10", "clean", seed)
    return np.sort(D["iv"][fold][0].astype(np.int64))


def train_rows(task, B, fold):
    m = B["fold"] != fold
    if task == "tau":
        m &= B["is_a"]
    return np.flatnonzero(m)


def cache_rows(task, B):
    return np.flatnonzero(B["is_a"])


def macro_f1(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1)
    ok = den > 0
    return float(np.mean(2 * tp[ok] / den[ok]))
