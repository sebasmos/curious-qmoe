"""T13 shared definitions (ANALYSIS_PLAN_T13.md): the stored score folders, experts, conditions, costs. Stored results are read only.
Scores are the per-clip logits of earlier rounds (heads trained there); nothing is trained or extracted in T13."""
from __future__ import annotations
import json, sys
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/redesign/idea6/stage2"))
from common import Q, S2, S3B, S3C, I5, OPTB  # noqa: E402
import analyze as A  # noqa: E402   Stage 2 helpers: ScoreDir (field mixture assembled), softmax, hull_pair, tests
from audio_shift import FIELD_CELLS, cond_name  # noqa: E402

FINAL = Q / "2026-10-24_1507_redesign-idea6-final_code-ed82400"
UNP = Q / "2026-10-23_1918_redesign-idea6-stage3c-unpadded_code-cbf6be6"
LAT = Q / "2026-10-23_1919_redesign-idea6-latency-full_code-cbf6be6"
SNRS = (20, 10, 5, 0)
NOISE = [cond_name(k, s) for k in ("bg", "pink", "white") for s in SNRS]          # 12 additive cells
CORR = ["reverb_rt300ms", "reverb_rt600ms", "reverb_rt1200ms", "clip_p50", "clip_p25", "clip_p10", "lowpass_4000hz", "lowpass_2000hz", "lowpass_1000hz"]
TASKS = {
    "esc50": dict(seeds=[0, 1, 2, 3, 4], nf=5, K=50, short=512),
    "esc5": dict(seeds=[0, 1, 2, 3, 4], nf=5, K=5, short=512),
    "urban": dict(seeds=[0, 1, 2], nf=10, K=10, short=400),
    "tau": dict(seeds=[0, 1, 2], nf=5, K=10, short=None),
}
EXPERTS = ["mn10", "mn20", "astfp32", "astw4", "astshort"]       # cheap CNN W8A8 x2, AST FP32, AST w4g128 (1024 frames), short AST w4g128
LABEL = {"mn10": "MN10-Q8", "mn20": "MN20-Q8", "astfp32": "AST-FP32", "astw4": "AST-w4g128", "astshort": "AST-short-w4g128"}


def experts_of(task): return [e for e in EXPERTS if not (task == "tau" and e == "astshort")]
def combos(task, sizes=(2, 3)): return [c for n in sizes for c in combinations(experts_of(task), n)]
def cname(c): return "+".join(c)


def sdir(task, expert, seed, corrupt=False) -> tuple[Path, str]:
    """(score folder of head `esc`, config stem) of one expert, one seed. corrupt=True: the final round's non-additive folders."""
    t = task
    if expert == "mn10":
        if t == "urban": d = (FINAL / f"scores/urban/mn10unp_urban_s{seed}") if corrupt else (UNP / f"scores_unpr/urban/mn10unp_urban_s{seed}")
        else: d = (FINAL / f"scores/{t}/mn10_{t}_s{seed}") if corrupt else (S3B / f"scores_cnn/{t}/mn10_{t}_s{seed}")
        return d / "esc", "mn10_w8a8"
    if expert == "mn20":
        d = (FINAL / f"scores/{t}/mn20_{t}_s{seed}") if corrupt else (S3C / f"scores_cnn/{t}/mn20_{t}_s{seed}")
        return d / "esc", "mn20_w8a8"
    if expert in ("astfp32", "astw4"):
        assert not corrupt
        return I5 / f"scores/{t}/ast{t}_s{seed}/esc", "fp32" if expert == "astfp32" else "w4g128"
    d = (FINAL / f"scores/{t}/astshort_{t}_s{seed}") if corrupt else (S2 / f"scores_ast_short/{t}/astshort_{t}_s{seed}")
    return d / "esc", "w4g128"


def tau_file(expert, seed) -> Path:
    if expert == "mn10": return OPTB / f"scores/mn10_tau_s{seed}/esc/mn10_w8a8__all.npz"
    if expert == "mn20": return OPTB / f"scores/mn20_tau_s{seed}/esc/mn20_w8a8__all.npz"
    return OPTB / f"scores/ast_tau_s{seed}/esc/{'fp32' if expert == 'astfp32' else 'w4g128'}__all.npz"


_SD = {}
def _scoredir(d: Path):
    if d not in _SD: _SD[d] = A.ScoreDir(d)
    return _SD[d]


def tau_load(expert, seed):
    z = np.load(tau_file(expert, seed)); iv = {int(k[7:]): z[k] for k in z.files if k.startswith("iv_idx_")}
    return {"y": z["y"].astype(np.int64), "fold": z["fold"].astype(int), "device": z["device"].astype(str), "z": z["test_logits"].astype(np.float64),
            "iv": {f: (iv[f], z[f"iv_logits_{f}"].astype(np.float64)) for f in sorted(iv)}, "crc": int(z["names_crc32"]), "n_a": int(z["n_a"])}


def available(task, expert, cond, seeds=None) -> bool:
    """Is the (expert, condition) cell complete (finite test logits and inner-validation logits) for every seed of the task?"""
    seeds = seeds or TASKS[task]["seeds"]
    if task == "tau": return cond in ("clean", "shift") and all(tau_file(expert, s).exists() for s in seeds)
    corrupt = cond in CORR
    if corrupt and expert in ("astfp32", "astw4"): return False
    for s in seeds:
        d, cfg = sdir(task, expert, s, corrupt)
        if not d.exists(): return False
        sd = _scoredir(d)
        if not sd.has(cfg, cond): return False
        if cond != "field" and not sd.complete(cfg, cond): return False
    return True


def load(task, expert, cond, seed) -> dict:
    """{'y','fold','z' [N,K], 'iv': {fold: (idx, logits)}} for condition `cond` (clean, noise cell, field, corruption); TAU: cond is clean or shift
    (same arrays, the row mask is in tau_rows)."""
    if task == "tau": return tau_load(expert, seed)
    d, cfg = sdir(task, expert, seed, cond in CORR)
    return _scoredir(d).get(cfg, cond)


def cost_table(task) -> dict:
    """Priced bit-ops per clip (backbone + head, repo cost module) of each expert."""
    if task == "tau":
        c = json.load(open(OPTB / "analysis/tau/costs.json"))
        return dict(mn10=c["cheap"]["cnn:mn10:w8a8:esc"], mn20=c["cheap"]["cnn:mn20:w8a8:esc"], astfp32=c["ast"]["ast:fp32+esc"], astw4=c["ast"]["ast:w4g128+esc"])
    c = json.load(open(S3C / f"analysis/{task}/costs.json")); L = TASKS[task]["short"]
    return dict(mn10=c["cheap"]["cnn:mn10unp:w8a8:esc" if task == "urban" else "cnn:mn10:w8a8:esc"], mn20=c["cheap"]["cnn:mn20:w8a8:esc"],
                astfp32=c["ast"]["fp32+esc"], astw4=c["ast"]["w4g128+esc"], astshort=c["ast"][f"w4g128@{L}+esc"])


def tau_rows(D, fold, which):
    """Test rows of TAU fold `fold`: clean = device A, shift = every other device."""
    m = D["fold"] == fold
    return np.flatnonzero(m & (D["device"] == "a")) if which == "clean" else np.flatnonzero(m & (D["device"] != "a"))


def softmax(z): return A.softmax(z)
def pmargin(p): return A.margin(p)
