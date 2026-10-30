"""Non-additive corruptions of a 16 kHz waveform (idea 6 final round, ANALYSIS_PLAN.md).

  reverb:RT60   convolution with a synthetic exponential-decay impulse response
  clip:Q        hard clip at Q times the clip's own peak
  lowpass:HZ    zero-phase 8th-order Butterworth low-pass

No level correction is applied after any of them. Nothing is downloaded and nothing is random per clip: the impulse
responses come from a fixed seed, one per RT60.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, fftconvolve, sosfiltfilt

SR = 16000
IR_SEED = 20241024
DRR_DB = 0.0
KINDS = ("reverb", "clip", "lowpass")
LEVELS = {"reverb": (0.3, 0.6, 1.2), "clip": (0.5, 0.25, 0.1), "lowpass": (4000.0, 2000.0, 1000.0)}
ORDER = [f"{k}:{lv:g}" for k in KINDS for lv in LEVELS[k]]  # command-line spelling, mild to severe within a kind

_IR = {}


def cond_label(kind: str, level: float) -> str:
    """File-name suffix of a condition (one underscore only, so '__' still separates config and condition)."""
    if kind == "reverb":
        return f"reverb_rt{int(round(level * 1000))}ms"
    if kind == "clip":
        return f"clip_p{int(round(level * 100))}"
    if kind == "lowpass":
        return f"lowpass_{int(round(level))}hz"
    raise ValueError(kind)


def impulse_response(rt60: float) -> np.ndarray:
    key = int(round(rt60 * 1000))
    if key not in _IR:
        rng = np.random.default_rng([IR_SEED, key])
        n = int(round(rt60 * SR))
        t = np.arange(n) / SR
        tail = rng.standard_normal(n) * np.exp(-np.log(1000.0) * t / rt60)
        tail[0] = 0.0
        tail *= np.sqrt(10 ** (-DRR_DB / 10)) / np.sqrt(np.sum(tail ** 2))
        h = tail.copy()
        h[0] = 1.0
        _IR[key] = h / np.sqrt(np.sum(h ** 2))
    return _IR[key]


def reverb(x: np.ndarray, rt60: float) -> np.ndarray:
    return fftconvolve(x.astype(np.float64), impulse_response(rt60))[:len(x)].astype(np.float32)


def hard_clip(x: np.ndarray, frac: float) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak == 0.0:
        return x.astype(np.float32)
    c = frac * peak
    return np.clip(x, -c, c).astype(np.float32)


def lowpass(x: np.ndarray, hz: float) -> np.ndarray:
    sos = butter(8, hz, btype="low", fs=SR, output="sos")
    pad = min(3 * (2 * len(sos) + 1), len(x) - 1)
    return sosfiltfilt(sos, x.astype(np.float64), padlen=pad).astype(np.float32)


def corrupt(x: np.ndarray, kind: str, level: float) -> np.ndarray:
    if kind == "reverb":
        return reverb(x, level)
    if kind == "clip":
        return hard_clip(x, level)
    if kind == "lowpass":
        return lowpass(x, level)
    raise ValueError(kind)


def level_change_db(x: np.ndarray, y: np.ndarray) -> float | None:
    rx = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    ry = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    return None if rx == 0 or ry == 0 else 20 * float(np.log10(ry / rx))
