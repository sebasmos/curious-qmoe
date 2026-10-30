"""T20 shared definitions (ANALYSIS_PLAN_T20.md): tasks, audio, the stored inner split, the cross-fitted teacher, the filter-bank cache.
Stored results are read only."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t1c_common import load_bed  # noqa: E402
from common import Q, S2, S3C, OPTB  # noqa: E402

E = Path.home() / "Documents/DATASETS"
URB_CSV, URB_AUD = E / "Urban8k/ast_768/urbansound8k.csv", E / "Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio"
ESC_CSV, ESC_AUD = E / "ESC-50/ast_768/esc-50.csv", E / "ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio"
TAU_CSV = E / "TAU2020Mobile/ast_768/tau2020mobile.csv"
TAU_AUD = E / "TAU2020Mobile/audio_raw/tau2020mobile_zenodo_3819968/TAU-urban-acoustic-scenes-2020-mobile-development/audio"
UNP = Q / "2026-10-23_1918_redesign-idea6-stage3c-unpadded_code-cbf6be6"
T1C = Q / "2026-10-25_0052_moe-t1c_code-7d7429e"
CACHE = Path.home() / "Documents/qmoe-work/t20-cache"
# csv, audio, background csv, background audio (stage 2 / option C pairing), frames, classes, folds
TASKS = {
    "urban": dict(csv=URB_CSV, aud=URB_AUD, bg_csv=ESC_CSV, bg_aud=ESC_AUD, L=400, K=10, nf=10),
    "esc50": dict(csv=ESC_CSV, aud=ESC_AUD, bg_csv=URB_CSV, bg_aud=URB_AUD, L=512, K=50, nf=5),
    "tau": dict(csv=TAU_CSV, aud=TAU_AUD, bg_csv=None, bg_aud=None, L=1024, K=10, nf=5),
}
SEEDS = [0, 1, 2]
BLOCK = 10


def stems(task):
    return [Path(s).stem for s in pd.read_csv(TASKS[task]["csv"], usecols=["name"])["name"]]


def cheap_file(task, seed, cell="clean") -> Path:
    """Stored logits of the cheap tier (ANALYSIS_PLAN_T20.md 3.2)."""
    if task == "urban": return UNP / f"scores_unpr/urban/mn10unp_urban_s{seed}/esc/mn10_w8a8__{cell}.npz"
    if task == "esc50": return S3C / f"scores_cnn/esc50/mn20_esc50_s{seed}/4/mn20_w8a8__{cell}.npz"
    return OPTB / f"scores/mn20_tau_s{seed}/8/mn20_w8a8__all.npz"


def inner_split(task, seed, fold):
    """Stored grouped inner-validation rows (global) of (seed, fold), the same for every stored expert."""
    z = np.load(cheap_file(task, seed)); return np.sort(z[f"iv_idx_{fold}"].astype(np.int64))


def train_rows(task, D, fold):
    m = D["fold"] != fold
    if task == "tau": m &= D["is_a"]
    return np.flatnonzero(m)


def test_rows(task, D, fold): return np.flatnonzero(D["fold"] == fold)


def teacher(task, seed, fold, tr):
    O = np.load(T1C / task / "raw_oof" / f"s{seed}_f{fold}.npz"); assert np.array_equal(O["tr"], tr), "teacher rows differ"
    return O["E2a_oof"].astype(np.float32)


def bed(task):
    D = load_bed(task)
    return D


def macro_f1(y, p, k):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k); tp = np.diag(cm).astype(float); den = cm.sum(0) + cm.sum(1); ok = den > 0
    return float(np.mean(2 * tp[ok] / den[ok]))


def cache_rows(task, D):
    return np.flatnonzero(D["is_a"]) if task == "tau" else np.arange(D["N"])
