"""T24 shared pieces on top of T21's (t21_common): the reader for the prepared audio.

Disk rule (round plan section 12): only what the models read is kept. Each clip is stored once as the exact float32
array T21's reader returns for the original file (stage 2's load_clip: mono, mean removed, resampled to 16 kHz, then
the first 163,840 samples), written as a 16 kHz float32 WAV by t24_prep_audio.py. Reading such a file back returns it
unchanged, so every tier sees bit-identical input to what T21's pipeline would compute from the original file. Any
other file (the UrbanSound8K noise bank) goes through T21's reader as before.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
T21 = HERE.parent / "t21"
sys.path.insert(0, str(T21))
import t21_common as C21  # noqa: E402

import audio_shift as AS  # noqa: E402

PREP_DIR_NAME = "audio16k_t24"  # every prepared file lives under a directory of this name
MAX_SAMPLES = C21.MAX_SAMPLES
_orig_load = C21._orig_load  # stage 2's load_clip, untouched


def load_clip_t24(path):
    p = Path(path)
    if PREP_DIR_NAME in p.parts:
        import soundfile as sf

        x, sr = sf.read(str(p), dtype="float32", always_2d=False)
        if sr != 16000 or x.ndim != 1:
            raise SystemExit(f"{p}: prepared file is not 16 kHz mono")
        return x[:MAX_SAMPLES]
    return _orig_load(path)[:MAX_SAMPLES]


def patch_loader():
    AS.load_clip = load_clip_t24
    C21.load_clip_t21 = load_clip_t24
