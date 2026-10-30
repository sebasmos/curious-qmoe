"""T17 shared definitions (ANALYSIS_PLAN_T17.md). Builds on t1c_common (beds, cells, conditions, groups) and adds MN20 paths and the inner-validation families."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t1c_common import *  # noqa: F401,F403
from t1c_common import S3C, OPTB, emb_path as _emb_path, ALLCELLS, ZERO_DB, load_bed, FIELD_SEEDS, field_assignment  # noqa: F401

T1C = Q / "2026-10-25_0052_moe-t1c_code-7d7429e"
VAL_SEED = FIELD_SEEDS["val"]


def emb17(bed: str, ex: str, cell: str) -> Path:
    """ex in {E1 (mn10 W8A8), M20 (mn20 W8A8)}."""
    if ex == "E1": return _emb_path(bed, "E1", cell)
    d = {"urban": S3C / "emb_cnn/urban8k/mn20_w8a8", "esc50": S3C / "emb_cnn/esc50/mn20_w8a8", "tau": OPTB / "emb_cnn/tau/mn20_w8a8"}[bed]
    return Path(f"{d}__{cell}.npy")


def families(bed: str, D: dict, iv_g: np.ndarray, fold: int) -> dict:
    """Inner-validation families: name -> list of (cell, global row indices). Labels are y[rows]. No test row appears."""
    fam = {"clean": [("clean", iv_g)]}
    if bed == "tau":
        rows = np.flatnonzero((D["fold"] != fold) & (~D["is_a"]))
        fam["shift"] = [("clean", rows)]
    else:
        av = np.array(field_assignment(D["N"], VAL_SEED))[iv_g]
        fam["field"] = [(c, iv_g[av == c]) for c in ALLCELLS if (av == c).any()]
        fam["zero"] = [("bg__0db", iv_g)] if bed == "urban" else [(c, iv_g) for c in ZERO_DB]
    return fam


FAMILY_OF = {"U-field": "field", "U-0dB": "zero", "E-field": "field", "E-0dB": "zero", "T-shift": "shift", "clean": "clean"}
T13_CELLS = {"U-field": ["field"], "U-0dB": ["bg__0db"], "E-field": ["field"], "E-0dB": ZERO_DB, "T-shift": ["shift"], "clean": ["clean"]}
TASK_OF_BED = {"urban": "urban", "esc50": "esc50", "tau": "tau"}
TASK_CONDS = {"urban": ["U-field", "U-0dB"], "esc50": ["E-field", "E-0dB"], "tau": ["T-shift"]}


def costs17(bed: str) -> dict:
    """Priced bit-ops: pair of (a) = E1 + mn20; system of (b) = E1 + AST w4g128 (1024 frames); the singles' costs are T13's."""
    import json
    import t13_common as T13
    task = TASK_OF_BED[bed]; ct = cost_table(bed); t13 = T13.cost_table(task)          # cost_table from t1c_common
    c20 = json.load(open((OPTB / "analysis/tau/costs.json") if bed == "tau" else (S3C / f"analysis/{task if task != 'urban' else 'urban'}/costs.json")))["cheap"]["cnn:mn20:w8a8:esc"]
    return dict(a=ct["E1"] + c20, b=ct["E1"] + ct["E2b"], mn10=ct["E1"], mn20=c20, singles=t13)
