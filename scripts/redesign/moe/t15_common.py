"""Shared definitions of T15 and T18 (ANALYSIS_PLAN_T15.md, ANALYSIS_PLAN_T18.md): tasks, pools, experts (stored and new views), score loading, costs, tests.
Stored scores are read only (t13_common); the new views' scores are written by score_eff.py (urban, esc50) and score_optb.py (tau) into <R>/scores."""
from __future__ import annotations
import json, os, sys
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t13_common import (A, cost_table as stored_costs, load as stored_load, available as stored_available, tau_load, tau_rows, TASKS as T13TASKS, OPTB,  # noqa: F401
                        sdir, softmax, pmargin)
from audio_shift import FIELD_CELLS, cond_name  # noqa: E402,F401

R = Path(os.environ.get("T15_R", Path(open(Path.home() / "Documents/qmoe-work/t15-R.txt").read().strip())))
TASKS = {"urban": dict(seeds=[0, 1, 2], nf=10, K=10, ds="urban", short=400), "esc50": dict(seeds=[0, 1, 2, 3, 4], nf=5, K=50, ds="esc50", short=512),
         "tau": dict(seeds=[0, 1, 2], nf=5, K=10, ds="tau", short=None)}
POOL = {"urban": ["field", "bg__0db"], "esc50": ["field", "bg__0db", "pink__0db", "white__0db"], "tau": ["shift"]}
CELLS = {t: ["clean"] + POOL[t] if t != "tau" else ["clean", "shift"] for t in TASKS}
STORED = ["mn10", "mn20", "astfp32", "astw4", "astshort"]
NEW_T15 = ["v1", "v2", "v3", "v4", "v5", "v6"]          # v5 = waveform CNN
LR = ["lra", "lrb"]
LABEL = {"mn10": "MN10-Q8", "mn20": "MN20-Q8", "astfp32": "AST-FP32", "astw4": "AST-w4g128", "astshort": "AST-short-w4g128", "v1": "V1 low band", "v2": "V2 high band",
         "v3": "V3 short window", "v4": "V4 long window", "v5": "V5 waveform CNN", "v6": "V6 CNN14", "lra": "LR-A", "lrb": "LR-B"}
E1, E2 = "mn10", "astw4"
FAMILY = 15


def is_new(e): return e not in STORED


def sd_new(task, view, seed):
    """Score folder of a new view (urban, esc50): <R>/scores/<ds>/<view>_<ds>_s<seed>/esc ."""
    return R / f"scores/{task}/{view}_{task}_s{seed}/esc"


def tau_new_file(view, seed): return R / f"scores/tau/{view}_tau_s{seed}/esc/{view}__all.npz"


_SD = {}
def _sd(d):
    if d not in _SD: _SD[d] = A.ScoreDir(d)
    return _SD[d]


def view_available(task, view, seeds=None):
    seeds = seeds or TASKS[task]["seeds"]
    if task == "tau": return all(tau_new_file(view, s).exists() for s in seeds)
    ok = True
    for s in seeds:
        d = sd_new(task, view, s)
        if not d.exists(): return False
        sd = _sd(d)
        ok &= all(sd.has(view, c) for c in ["clean"] + [c for c in POOL[task]])
    return ok


def available(task, expert, cell):
    if is_new(expert): return view_available(task, expert)
    return stored_available(task, expert, "clean" if task == "tau" else cell, None) if True else False


_TAUC = {}
def load(task, expert, cell, seed):
    """{'y','fold','z' [N,K], 'iv' {fold: (idx, logits)}, + TAU: 'device'} ; TAU cells clean/shift use the same arrays (row masks via rows())."""
    if task == "tau":
        key = (expert, seed)
        if key not in _TAUC:
            if is_new(expert):
                z = np.load(tau_new_file(expert, seed)); iv = {int(k[7:]): z[k] for k in z.files if k.startswith("iv_idx_")}
                _TAUC[key] = {"y": z["y"].astype(np.int64), "fold": z["fold"].astype(int), "device": z["device"].astype(str), "z": z["test_logits"].astype(np.float64),
                              "iv": {f: (iv[f], z[f"iv_logits_{f}"].astype(np.float64)) for f in sorted(iv)}}
            else: _TAUC[key] = stored_load("tau", expert, "clean", seed)
        return _TAUC[key]
    if is_new(expert): return _sd(sd_new(task, expert, seed)).get(expert, cell)
    return stored_load(task, expert, cell, seed)


def rows(task, D, fold, cell):
    """Test rows of one fold in a cell."""
    if task == "tau": return tau_rows(D, fold, cell)
    return np.flatnonzero(D["fold"] == fold)


def new_costs(task):
    c = json.load(open(R / "inputs/view_costs.json"))[task]
    return {v: c[v]["total_bitops"] for v in c}


def costs(task):
    out = dict(stored_costs(task)); out.update(new_costs(task) if (R / "inputs/view_costs.json").exists() else {})
    if (R / "inputs/wave_costs.json").exists():
        w = json.load(open(R / "inputs/wave_costs.json"))
        if task in w: out["v5"] = w[task]
    return out


def acc(y, p): return 100.0 * float((p == y).mean())
def f1(y, p, K): return 100.0 * A.f1_fast(y, p, K)


def nb(d, ratio):
    d = np.asarray(d, float); K = len(d); s2 = d.var(ddof=1); se = np.sqrt((1 / K + ratio) * s2)
    if se == 0: return dict(mean=float(d.mean()), lo=float(d.mean()), hi=float(d.mean()), p=1.0 if d.mean() == 0 else 0.0)
    t = d.mean() / se; h = stats.t.ppf(.975, K - 1) * se
    return dict(mean=float(d.mean()), lo=float(d.mean() - h), hi=float(d.mean() + h), p=float(2 * stats.t.sf(abs(t), K - 1)))


def holm_family(p, m=FAMILY):
    """Holm step-down for the len(p) tests of one experiment inside a family of m: the other m - len(p) tests enter as p = 1, so the sorted p-values are
    compared with 0.05/m, 0.05/(m-1), ...; returns adjusted p."""
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def hull_value(sp, c):
    """Stage 2 hull_pair on a table with columns system, f1 (any accuracy scale), cost; returns (value, lo, hi, lam)."""
    return A.hull_pair(sp, c)


def subsets(pool, sizes=(2, 3, 4)): return [c for n in sizes for c in combinations(pool, n)]
def cname(c): return "+".join(c)
def mean_cost(task, c): return sum(costs(task)[e] for e in c)
