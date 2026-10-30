"""T21 shared pieces: the 10.24 s length rule on top of stage 2's audio reader, paths, and the clip-level cost formulas.

Audio: stage 2's audio_shift.load_clip (mono, mean removed, 16 kHz) followed by keeping the first MAX_SAMPLES samples
(ANALYSIS_PLAN.md C3). Every condition is applied to the truncated waveform by stage 2's unchanged mixer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
for p in (ROOT, ROOT / "scripts/redesign/idea6/stage2", ROOT / "scripts/redesign/idea6/final", ROOT / "scripts/redesign/idea6"):
    sys.path.insert(0, str(p))

import audio_shift as AS  # noqa: E402

MAX_SAMPLES = 163840  # 10.24 s at 16 kHz
_orig_load = AS.load_clip


def load_clip_t21(path):
    return _orig_load(path)[:MAX_SAMPLES]


def patch_loader():
    """Make every later `from audio_shift import load_clip` (and the module attribute) use the truncating reader."""
    AS.load_clip = load_clip_t21


HOME = Path.home()
DATA = HOME / "Documents/DATASETS"
URB_CSV = DATA / "Urban8k/ast_768/urbansound8k.csv"
URB_AUDIO = DATA / "Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio"
WEIGHTS = HOME / "Documents/qmoe-weights/efficientat_release-v0.0.1"
AST_MODEL, AST_REV = "MIT/ast-finetuned-audioset-10-10-0.4593", "f826b80d28226b62986cc218e5cec390b1096902"


def fbank_frames(n16: int) -> int:
    """Frames of the AST fbank (25 ms window, 10 ms shift, kaldi snip_edges) of n 16 kHz samples."""
    return 1 + (n16 - 400) // 160 if n16 >= 400 else 0


def time_patches(frames: int) -> int:
    return (min(frames, 1024) - 16) // 10 + 1
