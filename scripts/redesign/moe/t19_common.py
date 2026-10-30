"""T19 shared definitions (ANALYSIS_PLAN_T19.md, Part B): tasks, configurations, costs, paths."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for p in (ROOT, HERE, ROOT / "scripts/redesign/idea6/stage2", ROOT / "scripts/redesign/idea5"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from QWave.costs import ast_t19_bitops  # noqa: E402

HOME = Path.home()
DS = HOME / "Documents/DATASETS"
Q = HOME / "Documents/qmoe-results"
S2 = Q / "2026-10-22_1120_redesign-idea6-stage2_code-9f346ba"
S3C = Q / "2026-10-23_1227_redesign-idea6-stage3c_code-9c9cbf1"
OPTB = Q / "2026-10-23_2133_redesign-idea6-optionB_code-f320295"
FINAL = Q / "2026-10-24_1507_redesign-idea6-final_code-ed82400"
MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
REVISION = "f826b80d28226b62986cc218e5cec390b1096902"

TASKS = {
    "esc50": dict(csv=DS / "ESC-50/ast_768/esc-50.csv", audio=DS / "ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio",
                  length=512, seeds=[0, 1, 2, 3, 4], nf=5, label_col="label", K=50, rec_token=1, emb="esc50",
                  fbank=4267888640.0),
    "esc5": dict(csv=DS / "ESC-50/ast_768/esc-50.csv", audio=None, length=512, seeds=[0, 1, 2, 3, 4], nf=5,
                 label_col="class_id", K=5, rec_token=1, emb="esc50", fbank=4267888640.0),
    "urban": dict(csv=DS / "Urban8k/ast_768/urbansound8k.csv",
                  audio=DS / "Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio",
                  length=400, seeds=[0, 1, 2], nf=10, label_col="class_id", K=10, rec_token=0, emb="urban8k",
                  fbank=3112228956.408612),
    "tau": dict(csv=DS / "TAU2020Mobile/ast_768/tau2020mobile.csv",
                audio=DS / "TAU2020Mobile/audio_raw/tau2020mobile_zenodo_3819968",
                length=1024, seeds=[0, 1, 2], nf=5, label_col="class_id", K=10, rec_token=2, emb="tau",
                fbank=8418160640.0),
}
# calibration sources (D1): task -> list of (source task, number of inputs)
CALIB_SOURCES = {"esc50": [("urban", 256)], "urban": [("esc50", 256)], "tau": [("esc50", 128), ("urban", 128)]}

# configurations of the T19 mode (B3): name -> (weight spec name, activation bits or "mp6"/"mp8", sr seed)
CONFIGS = {
    "w8a8": ("w8g128", 8, None), "n4a4": ("n4g128", 4, None), "n4a6": ("n4g128", 6, None),
    "w4a6": ("w4g128", 6, None), "w4a4": ("w4g128", 4, None), "w8a6": ("w8g128", 6, None),
    "w8a4": ("w8g128", 4, None), "w6a6": ("w6g128", 6, None), "w4a8": ("w4g128", 8, None),
    "n4a8": ("n4g128", 8, None), "mp6": ("n4g128", "mp6", None), "mp8": ("n4g128", "mp8", None),
    "n4a4sr1": ("n4g128", 4, 1), "n4a4sr2": ("n4g128", 4, 2), "n4a6sr1": ("n4g128", 6, 1), "n4a6sr2": ("n4g128", 6, 2),
}
STATICS_NEW = ["w8a8", "n4a4", "n4a6", "w4a6", "w4a4", "w8a6", "w8a4", "w6a6", "w4a8", "n4a8", "mp6", "mp8"]
LEGACY_SHORT = ["fp32", "w8g64", "w4g128", "n4", "w4", "w3g64"]   # stored, idea 6 stage 2 (ESC-50 512, UrbanSound8K 400)
LEGACY_TAU = ["fp32", "w8g64", "w4g128", "n4"]                     # stored, option B (1024)
N_A4_BLOCKS = {"mp6": 6, "mp8": 8}
P_BITS = 8


def head_bitops(K: int) -> float:
    """FP32 head 768-640-320-K at 32 x 32 (the stage 3c head cost)."""
    return float((768 * 640 + 640 * 320 + 320 * K) * 1024)


def layer_bits(cfg: str, calib: dict | None = None) -> tuple:
    """Activation bits per block; mp6/mp8 need the scan's order (least sensitive first) from calib json."""
    a = CONFIGS[cfg][1]
    if isinstance(a, int):
        return (a,) * 12
    order = calib["sensitivity_order_least_first"]
    low = set(order[:N_A4_BLOCKS[a]])
    return tuple(4 if i in low else 8 for i in range(12))


def parse_w(name: str):
    """(bits, group, nested) of a weight spec name such as w8g128, n4g128, n4, w3g64, fp32."""
    if name == "fp32":
        return 32, None, False
    nested = name.startswith("n")
    b = int(name[1])
    g = int(name.split("g")[1]) if "g" in name else None
    return b, g, nested


def backbone_bitops(task: str, cfg: str, priced: bool = True) -> float:
    """Backbone bit-ops of a T19 configuration (B4) or of a stored legacy one (prefix 'L:')."""
    L = TASKS[task]["length"]
    if cfg.startswith("L:"):
        name = cfg[2:]
        wb, g, _ = parse_w(name)
        if wb == 32:
            c = ast_t19_bitops(L, 32, 32, 32, group=None, asym=False, price=0.0)
            return c["unpriced"]
        c = ast_t19_bitops(L, wb, 8, 16, group=g, asym=False)
        return c["total"] if priced else c["unpriced"]
    wname, a, _ = CONFIGS[cfg]
    wb, g, _ = parse_w(wname)
    ab = [4] * N_A4_BLOCKS[a] + [8] * (12 - N_A4_BLOCKS[a]) if isinstance(a, str) else a
    c = ast_t19_bitops(L, wb, ab, P_BITS, group=g, asym=True)
    return c["total"] if priced else c["unpriced"]


def system_cost(task: str, cfg: str, priced: bool = True) -> float:
    """Front end + backbone + FP32 head, per clip."""
    t = TASKS[task]
    return t["fbank"] + backbone_bitops(task, cfg, priced) + head_bitops(t["K"])


def results_dir() -> Path:
    return Path(os.environ.get("T19_R") or (ROOT / ".results_dir").read_text().strip())
