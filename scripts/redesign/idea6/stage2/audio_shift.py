"""Audio reading and seeded corruption for idea 5, importable.

The reading, noise and mixing functions are Stage 1's
(scripts/redesign/idea5/extract_conditions.py), copied unchanged so Stage 2
corrupts every clip exactly as Stage 1 did (that script parses its arguments
at import and cannot be imported). Stage 2 checks the copy by re-extracting
Stage 1 conditions and comparing embeddings.

Field mixture (Stage 2): every clip gets one condition. A seeded permutation
of the clips puts the first half (rounded up) clean and spreads the rest over
the nine cells {bg, pink, white} x {10, 5, 0} dB in turn, so the cells differ
in size by at most one clip. Test clips use one seed and inner-validation
clips another, so a threshold never sees the conditions of the clips it is
applied to.
"""
from __future__ import annotations

import zlib

import numpy as np

TARGET_SR = 16000
FIELD_CELLS = tuple((kind, snr) for kind in ("bg", "pink", "white") for snr in (10, 5, 0))
FIELD_SEEDS = {"test": 5001, "val": 5002}


# ── Stage 1's audio and noise (unchanged) ──────────────────────────────────

def index_wavs(audio_dir):
    index = {}
    for path in sorted(audio_dir.rglob("*.wav")):
        if path.stem in index:
            raise SystemExit(f"duplicate clip name {path.stem}: {index[path.stem]} and {path}")
        index[path.stem] = path
    return index


def load_clip(path):
    import soundfile as sf
    import torch
    import torchaudio.functional as AF

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio.mean(axis=1))
    wav = wav - wav.mean()
    if sr != TARGET_SR:
        wav = AF.resample(wav, sr, TARGET_SR)
    return wav.numpy()


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def clip_seed(name, kind):
    return zlib.crc32(f"{name}|{kind}".encode())


def pink(rng, n):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = f[1] if len(f) > 1 else 1.0
    return np.fft.irfft(spec / np.sqrt(f), n=n)


class NoiseMaker:
    def __init__(self, bg_names, bg_index, cache):
        self.bg_names, self.bg_index, self.cache = bg_names, bg_index, cache
        self.bg_cache, self.plan = {}, {}

    def bg_wave(self, j):
        if j not in self.bg_cache:
            w = load_clip(self.bg_index[self.bg_names[j]])
            if not self.cache:
                return w
            self.bg_cache[j] = w
        return self.bg_cache[j]

    def __call__(self, name, kind, n):
        rng = np.random.default_rng(clip_seed(name, kind))
        if kind == "white":
            noise = rng.standard_normal(n)
        elif kind == "pink":
            noise = pink(rng, n)
        elif kind == "bg":
            for _ in range(100):
                j = int(rng.integers(len(self.bg_names)))
                w = self.bg_wave(j)
                if len(w) < n:
                    w = np.tile(w, int(np.ceil(n / len(w))))
                off = int(rng.integers(len(w) - n + 1))
                noise = w[off:off + n].astype(np.float64)
                if rms(noise - noise.mean()) > 0:
                    break
            else:
                raise RuntimeError(f"{name}: no non-silent background window in 100 draws")
            self.plan[name] = {"bg_clip": self.bg_names[j], "offset_16k": off}
        else:
            raise ValueError(kind)
        return noise - noise.mean()


def mix(x, noise, snr):
    rx = rms(x)
    if rx == 0:
        return x, False
    g = rx / (rms(noise) * 10 ** (snr / 20))
    return (x.astype(np.float64) + g * noise).astype(np.float32), True


# ── Stage 2: conditions and the field mixture ──────────────────────────────

def cond_name(kind: str, snr) -> str:
    """'clean', or '<kind>__<snr>db' (Stage 1's file suffix)."""
    return "clean" if kind == "clean" else f"{kind}__{int(snr)}db"


def field_assignment(n: int, seed: int) -> list[str]:
    """Condition name of each of n clips: ceil(n/2) clean, the rest over the
    nine FIELD_CELLS in turn, positions from a permutation seeded by seed."""
    order = np.random.default_rng(seed).permutation(n)
    out = ["clean"] * n
    for r, i in enumerate(order[(n + 1) // 2:]):
        out[i] = cond_name(*FIELD_CELLS[r % len(FIELD_CELLS)])
    return out
