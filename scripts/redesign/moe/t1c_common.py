"""Shared definitions for T1c (ANALYSIS_PLAN_T1C.md): beds, stored embeddings per cell, conditions, costs. Stored results are read only."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[2] / "scripts/redesign/idea6/stage2"))
from common import Q, S2, S3B, S3C, I5, OPTB, load_table, softmax, pmargin  # noqa: F401
from audio_shift import FIELD_CELLS, FIELD_SEEDS, cond_name, field_assignment  # noqa: E402

PILOT = Q / "2026-10-24_2013_moe-difficulty-pilot_code-b77b98e"
T1B = Q / "2026-10-24_2208_moe-t1b_code-4e3e036"
T2B = Q / "2026-10-24_2224_moe-t2b_code-f5e4935"
DS = Path.home() / "Documents/DATASETS"
ESC_CSV = DS / "ESC-50/ast_768/esc-50.csv"
TAU_CSV = DS / "TAU2020Mobile/ast_768/tau2020mobile.csv"
CELLS9 = [cond_name(k, s) for k, s in FIELD_CELLS]            # bg__10db ... white__0db
ALLCELLS = ["clean"] + CELLS9
ZERO_DB = ["bg__0db", "pink__0db", "white__0db"]
BEDS = {"urban": dict(nf=10, ncls=10, nb_ratio_unit="clips"), "esc50": dict(nf=5, ncls=50, nb_ratio_unit="clips"), "tau": dict(nf=5, ncls=10, nb_ratio_unit="rows")}
# experts of the second tier, in cost order (cheapest first), per bed
EXPERTS2 = {"urban": ["E2c", "E2b", "E2a"], "esc50": ["E2c", "E2b", "E2a"], "tau": ["E2b", "E2a"]}


def emb_path(bed: str, expert: str, cell: str) -> Path:
    if bed == "urban":
        d = {"E1": S3B / "emb_cnn/urban8k/mn10_w8a8", "E2a": I5 / "emb/urban8k/fp32", "E2b": I5 / "emb/urban8k/w4g128",
             "E2c": S2 / "emb_ast_short/urban8k/w4g128"}[expert]
    elif bed == "esc50":
        d = {"E1": S3B / "emb_cnn/esc50/mn10_w8a8", "E2a": I5 / "emb/esc50/fp32", "E2b": I5 / "emb/esc50/w4g128",
             "E2c": S2 / "emb_ast_short/esc50/w4g128"}[expert]
    else:
        assert cell == "clean"
        d = {"E1": OPTB / "emb_cnn/tau/mn10_w8a8", "E2a": OPTB / "emb_ast/tau/fp32", "E2b": OPTB / "emb_ast/tau/w4g128"}[expert]
    return Path(f"{d}__{cell}.npy")


def cells_of(bed): return ["clean"] if bed == "tau" else ALLCELLS


def load_bed(bed: str) -> dict:
    """Row table of a bed: y (true labels), rec (source recording / location, the inner-split group), fold, N, and for TAU the device and the shift mask."""
    if bed == "urban":
        d = load_table(); out = dict(y=d.label.values.astype(np.int64), rec=d.recording.values, fold=d.folder.values, stem=d.stem.values)
    elif bed == "esc50":
        d = pd.read_csv(ESC_CSV, usecols=["folder", "name", "label"]); stem = d["name"].str.replace(".png", "", regex=False)
        out = dict(y=d.label.values.astype(np.int64), rec=d["name"].str.split("-").str[1].astype(int).values, fold=d.folder.values, stem=stem.values)
    else:
        d = pd.read_csv(TAU_CSV, usecols=["folder", "name", "class_id"])
        dev = d.name.str.replace(".wav", "", regex=False).str.split("-").str[-1].values
        out = dict(y=d.class_id.values.astype(np.int64), rec=d.name.str.split("-").str[2].astype(int).values, fold=d.folder.values, stem=d["name"].values, dev=dev, is_a=(dev == "a"))
    out["N"] = len(out["y"]); out["assign"] = None if bed == "tau" else np.array(field_assignment(out["N"], FIELD_SEEDS["test"]))
    return out


def cost_table(bed: str) -> dict:
    """Per-clip bit-ops of the experts from the repo cost module (E1, E2c, E2b, E2a)."""
    if bed == "urban":
        c = json.load(open(S3C / "analysis/urban/costs.json")); return dict(E1=c["cheap"]["cnn:mn10:w8a8:esc"], E2c=c["ast"]["w4g128@400+esc"], E2b=c["ast"]["w4g128+esc"], E2a=c["ast"]["fp32+esc"])
    if bed == "esc50":
        c = json.load(open(S3C / "analysis/esc50/costs.json")); return dict(E1=c["cheap"]["cnn:mn10:w8a8:esc"], E2c=c["ast"]["w4g128@512+esc"], E2b=c["ast"]["w4g128+esc"], E2a=c["ast"]["fp32+esc"])
    c = json.load(open(OPTB / "analysis/tau/costs.json")); return dict(E1=c["cheap"]["cnn:mn10:w8a8:esc"], E2b=c["ast"]["ast:w4g128+esc"], E2a=c["ast"]["ast:fp32+esc"])


CONDITIONS = {"urban": ["U-field", "U-0dB"], "esc50": ["E-field", "E-0dB"], "tau": ["T-shift"]}
BED_OF = {c: b for b, cs in CONDITIONS.items() for c in cs}
DESCRIPTIVE = {"urban": ["clean", "bg0"], "esc50": ["clean", "bg0", "pink0", "white0"], "tau": ["clean"]}


def groups(bed: str, cname: str, te: np.ndarray, assign, cond=None):
    """Rows of a condition for the test rows `te` of one fold: list of (cell, positions in te). Order is fixed so labels can be stacked the same way."""
    if cname in ("U-field", "E-field"):
        a = assign[te]; return [(c, np.flatnonzero(a == c)) for c in ALLCELLS if (a == c).any()]
    if cname == "U-0dB": return [("bg__0db", np.arange(len(te)))]      # Amendment 1: only bg 0 dB is stored for every clip on UrbanSound8K
    if cname == "E-0dB":
        return [(c, np.arange(len(te))) for c in ZERO_DB]
    if cname == "T-shift": return [("clean", np.flatnonzero(cond == 1))]
    if cname == "clean": return [("clean", np.arange(len(te)) if bed != "tau" else np.flatnonzero(cond == 0))]
    if cname in ("bg0", "pink0", "white0"): return [(cname[:-1] + "__0db", np.arange(len(te)))]
    raise KeyError(cname)


def gather(Z: dict, grp):
    """Z: cell -> array [n_te, ...] ; returns the stacked rows of the condition in the order of `grp`."""
    return np.concatenate([Z[c][i] for c, i in grp], 0)


def gather_y(yte, grp): return np.concatenate([yte[i] for _, i in grp], 0)
