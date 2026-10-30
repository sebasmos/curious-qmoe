"""T3/T5 shared definitions (ANALYSIS_PLAN_T3T5.md): cells, stored embeddings and audio statistics, loading of cheap and strong logits. Stored results are read only."""
from __future__ import annotations
import json, os, sys, zlib
from pathlib import Path
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ.setdefault(_v, "2")
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t13_common import *            # noqa: F401,F403  (Q, S3B, S3C, FINAL, UNP, OPTB, TASKS, NOISE, CORR, load, available, cost_table, tau_rows, softmax, pmargin, A)
from t1c_common import load_bed     # noqa: E402  row tables (stems, devices)

I7 = Q / "2026-10-22_1105_redesign-idea7-stage1_code-e2af723"
T13RUN = Q / "2026-10-25_0145_moe-t13_code-eaaedc9"
PAIR = {"esc50": "astw4", "urban": "astw4", "tau": "astfp32"}          # strong expert of T13's pre-selected pair (checked against t13_preselected.json by the check script)
CHEAP = "mn10"
T35 = ["esc50", "urban", "tau"]
POOL = {"esc50": list(NOISE), "urban": ["bg__10db", "bg__0db"], "tau": ["shift"]}
BEDNAME = {"esc50": "esc50", "urban": "urban", "tau": "tau"}
STEPS_NOISE = {"bg": ["bg__20db", "bg__10db", "bg__5db", "bg__0db"], "pink": ["pink__20db", "pink__10db", "pink__5db", "pink__0db"], "white": ["white__20db", "white__10db", "white__5db", "white__0db"]}
STEPS_CORR = {"reverb": ["reverb_rt300ms", "reverb_rt600ms", "reverb_rt1200ms"], "clip": ["clip_p50", "clip_p25", "clip_p10"], "lowpass": ["lowpass_4000hz", "lowpass_2000hz", "lowpass_1000hz"]}


def emb_file(task, cell):
    c = "clean" if task == "tau" else cell
    if task == "esc50": return S3B / f"emb_cnn/esc50/mn10_w8a8__{c}.npy"
    if task == "urban": return Q / "2026-10-23_1227_redesign-idea6-stage3c_code-9c9cbf1" / f"emb_cnn/urban8k_unpadded/mn10_w8a8__{c}.npy"
    return OPTB / "emb_cnn/tau/mn10_w8a8__clean.npy"


def stats_file(task, cell):
    c = "clean" if task == "tau" else cell
    if task == "tau": return OPTB / "audio_stats/tau/clean.npz"
    d = "esc50" if task == "esc50" else "urban"
    p = I7 / f"audio_stats/{d}/{c}.npz"
    return p if p.exists() else FINAL / f"audio_stats/{d}/{c}.npz"


_BED = {}
def bed(task):
    if task not in _BED: _BED[task] = load_bed(task)
    return _BED[task]


def names(task): return [str(s).replace(".wav", "") for s in bed(task)["stem"]]
def names_crc(task): return zlib.crc32("\n".join(names(task)).encode())


_EMB, _ST = {}, {}
def emb(task, cell):
    k = (task, "clean" if task == "tau" else cell)
    if k not in _EMB: _EMB[k] = np.load(emb_file(task, cell)).astype(np.float32)
    return _EMB[k]
def stats(task, cell):
    k = (task, "clean" if task == "tau" else cell)
    if k not in _ST: _ST[k] = np.load(stats_file(task, cell), allow_pickle=True)["stats"].astype(np.float64)
    return _ST[k]
STAT_NAMES = ["level_db", "snr_est_db", "floor_db", "flatness_db", "ltas_flat_db", "zcr", "centroid_khz", "rolloff_khz", "hf_ratio_db", "crest_db"]


def cellload(task, expert, cell, seed):
    """Logits of one expert in one cell: TAU 'clean'/'shift' read the same arrays (rows chosen by tau_rows)."""
    return load(task, expert, "clean" if task == "tau" else cell, seed)


def rows_of(task, D, cell, fold):
    """Test rows of `fold` in `cell`."""
    if task == "tau": return tau_rows(D, fold, "clean" if cell == "clean" else "shift")
    return np.flatnonzero(D["fold"] == fold)


def js(pa, pb, eps=1e-12):
    m = 0.5 * (pa + pb)
    kl = lambda p, q: (p * (np.log(p + eps) - np.log(q + eps))).sum(1)
    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)
