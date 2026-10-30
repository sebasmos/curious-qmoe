"""T16 shared definitions (ANALYSIS_PLAN_T16.md): beds, feature files, conditions, costs. Stored results are read only."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t1c_common import *  # noqa: F401,F403  (beds, emb_path, load_bed, groups, gather, cost_table, CONDITIONS, T1C paths)
from audio_shift import FIELD_SEEDS, FIELD_CELLS, cond_name, field_assignment  # noqa: F401

T1C = Q / "2026-10-25_0052_moe-t1c_code-7d7429e"
WORK = Path.home() / "Documents/qmoe-work/t16-data"
FEAT = WORK / "feat"                      # <bed>__<cell>.npy [N,5,2,768] and <bed>__<cell>_emb.npy [N,768]
DEPTHS = [4, 6, 8, 10, 12]               # blocks run before the features are read; 12 = final block (after the LayerNorm)
FRAMES = {"urban": 400, "esc50": 512, "tau": 1024}
SEEDS = [0, 1, 2]
# cells extracted for every clip, per bed; the other field cells only for the rows the test or validation assignment gives them
FULL_CELLS = {"urban": ["clean", "bg__10db", "bg__0db"], "esc50": ["clean", "bg__0db", "pink__0db", "white__0db"], "tau": ["clean"]}
ALL_CELLS = {"urban": ["clean"] + [cond_name(k, s) for k, s in FIELD_CELLS], "esc50": ["clean"] + [cond_name(k, s) for k, s in FIELD_CELLS], "tau": ["clean"]}
# evaluation conditions: name -> (bed, kind, spec). kind 'cells': list of cells, every test row in each (rows pooled); 'field': test assignment; 'shift': TAU non-A rows
EVAL = {
    "urban": {"U-field": ("field", None), "U-0dB": ("cells", ["bg__0db"]), "U-bg10": ("cells", ["bg__10db"]), "U-clean": ("cells", ["clean"])},
    "esc50": {"E-field": ("field", None), "E-0dB": ("cells", ["bg__0db", "pink__0db", "white__0db"]), "E-bg0": ("cells", ["bg__0db"]), "E-pink0": ("cells", ["pink__0db"]),
              "E-white0": ("cells", ["white__0db"]), "E-clean": ("cells", ["clean"])},
    "tau": {"T-shift": ("shift", None), "T-clean": ("clean", None)},
}
PRIMARY = {"urban": "U-field", "esc50": "E-field", "tau": "T-shift"}
T2B_NAME = {"U-field": "U-field", "U-0dB": "U-0dB", "E-field": "E-field", "E-0dB": "E-0dB", "T-shift": "T-shift"}


def feat_path(bed, cell): return FEAT / f"{bed}__{cell}.npy"
def emb16_path(bed, cell): return FEAT / f"{bed}__{cell}_emb.npy"


def csv_and_audio(bed):
    E = Path.home() / "Documents/DATASETS"
    if bed == "urban":
        return E / "Urban8k/ast_768/urbansound8k.csv", E / "Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio", E / "ESC-50/ast_768/esc-50.csv", E / "ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio"
    if bed == "esc50":
        return E / "ESC-50/ast_768/esc-50.csv", E / "ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio", E / "Urban8k/ast_768/urbansound8k.csv", E / "Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio"
    return E / "TAU2020Mobile/ast_768/tau2020mobile.csv", E / "TAU2020Mobile/audio_raw/tau2020mobile_zenodo_3819968/TAU-urban-acoustic-scenes-2020-mobile-development/audio", None, None


def cell_rows(bed, N):
    """cell -> sorted row indices to extract: every clip for FULL_CELLS, else the rows the test or validation field assignment gives the cell."""
    out = {}
    at, av = np.array(field_assignment(N, FIELD_SEEDS["test"])), np.array(field_assignment(N, FIELD_SEEDS["val"]))
    for c in ALL_CELLS[bed]:
        out[c] = np.arange(N) if c in FULL_CELLS[bed] else np.flatnonzero((at == c) | (av == c))
    return out


MODEL, REV = "MIT/ast-finetuned-audioset-10-10-0.4593", "f826b80d28226b62986cc218e5cec390b1096902"
sys.path.insert(0, str(HERE.parents[2]))


def load_ast(L: int, device="cpu"):
    """Feature extractor padded to L frames and the FP32 AST cut to L frames (QWave.ast_frames.shorten_ast, as T12/stage 2)."""
    from transformers import ASTFeatureExtractor, ASTModel
    from QWave.ast_frames import shorten_ast
    ext = ASTFeatureExtractor.from_pretrained(MODEL, revision=REV, max_length=L)
    m = shorten_ast(ASTModel.from_pretrained(MODEL, revision=REV), L).to(device).eval()
    for p in m.parameters(): p.requires_grad_(False)
    return ext, m


def layer_fwd(layer, h):
    o = layer(h)
    return o[0] if isinstance(o, tuple) else o


# ---------------------------------------------------------------- items: the rows of a condition, in T1c's order -----------------------------------------------------
_MM = {}
def mmf(path):
    if path not in _MM: _MM[path] = np.load(path, mmap_mode="r")
    return _MM[path]


def items(bed, cn, rows, assign, cond=None):
    """(global rows, cells) of condition `cn` for the rows `rows`; same order as t1c_common.groups (so T1c's per-condition arrays line up).
    assign: field assignment vector over all N rows (test seed for test rows, validation seed for held-out rows). cond: 0/1 shift mask per entry of `rows` (TAU)."""
    if cn == "U-bg10": g = [("bg__10db", np.arange(len(rows)))]
    elif cn == "E-clean" or cn == "U-clean" or cn == "T-clean": g = groups(bed, "clean", rows, assign, cond)
    elif cn in ("E-bg0", "E-pink0", "E-white0"): g = groups(bed, cn[2:].lower(), rows, assign, cond)
    else: g = groups(bed, cn, rows, assign, cond)
    r = np.concatenate([rows[i] for _, i in g]); c = np.concatenate([[cell] * len(i) for cell, i in g]) if g else np.array([], str)
    return r, np.asarray(c)


def take(path_fn, rows, cells, tail=None):
    """Gather per-item arrays from per-cell files: path_fn(cell) -> .npy path; tail indexes the trailing dims (e.g. (slice(None),) )."""
    out = None
    for cell in dict.fromkeys(cells.tolist()):
        m = cells == cell; idx = rows[m]; srt = np.argsort(idx, kind='stable'); a = np.asarray(mmf(path_fn(cell))[idx[srt]]); res = np.empty_like(a); res[srt] = a
        if out is None: out = np.full((len(rows),) + a.shape[1:], np.nan, np.float32)
        out[np.flatnonzero(m)] = res
    return out
