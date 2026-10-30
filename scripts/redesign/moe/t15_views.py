"""T15/T18 views (ANALYSIS_PLAN_T15.md section 3, ANALYSIS_PLAN_T18.md section 2): front ends that turn 16 kHz waveforms into the input of a frozen backbone, and the
priced costs of each view. Mel views feed mn10 W8A8 (QWave.efficientat_mn); v6 feeds CNN14 W8A8 (QWave.panns_cnn14)."""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from QWave import efficientat_mn as M
from QWave import panns_cnn14 as P
from QWave.effnet_quant import rfft_macs

FLOOR = (math.log(1e-5) + 4.5) / 5.0
MEL_VIEWS = ["v0", "v1", "v2", "v3", "v4", "lra", "lrb"]
ALL_VIEWS = MEL_VIEWS + ["v6"]
DIM = {v: 960 for v in MEL_VIEWS} | {"v6": 2048}
HEAD_DIM = DIM
# (n_fft, win, hop) at 32 kHz
STFT = {"v0": (1024, 800, 320), "v3": (512, 512, 128), "v4": (2048, 2048, 512)}
LR_RATE, LR_NFFT, LR_WIN, LR_HOP = 8000, 256, 200, 160
LR_COLS = LR_NFFT // 2 + 1  # 129 columns, 0 to 4 kHz


@lru_cache(maxsize=None)
def bank(n_fft: int) -> torch.Tensor:
    """Kaldi mel filters of the native front end for an n_fft at 32 kHz: [128, n_fft/2 + 1]."""
    from torchaudio.compliance.kaldi import get_mel_banks

    banks, _ = get_mel_banks(M.N_MELS, n_fft, float(M.SR), M.F_MIN, M.F_MAX, 100.0, -500.0, 1.0)
    return F.pad(banks, (0, 1)).float()


@lru_cache(maxsize=None)
def centroid_hz() -> np.ndarray:
    b = bank(1024).numpy().astype(np.float64)
    f = np.arange(b.shape[1]) * M.SR / 1024
    return (b * f).sum(1) / np.maximum(b.sum(1), 1e-30)


@lru_cache(maxsize=None)
def lr_rows() -> np.ndarray:
    """Rows of the native bank whose weights all lie in columns 0..128 (below 4 kHz)."""
    b = bank(1024).numpy()
    return np.flatnonzero((b[:, LR_COLS:] == 0).all(1) & (b.sum(1) > 0))


def band_mask(view: str) -> np.ndarray:
    c = centroid_hz()
    return (c > 2000.0) if view == "v1" else ((c < 2000.0) | (c > 8000.0))


def _stft_power(x, n_fft, win, hop):
    w = torch.hann_window(win, periodic=False, device=x.device)
    s = torch.stft(x, n_fft, hop_length=hop, win_length=win, window=w, center=True, pad_mode="reflect", normalized=False, return_complex=True)
    return s.real ** 2 + s.imag ** 2


def _logmel(power, rows_bank):
    return ((torch.log(torch.matmul(rows_bank, power) + 1e-5) + 4.5) / 5.0).unsqueeze(1)


def mel_inputs(wave16: torch.Tensor, views) -> dict:
    """wave16 [B, n] (16 kHz) -> {view: input tensor [B, ...]} for the requested views (CPU)."""
    out = {}
    need32 = [v for v in views if v in ("v0", "v1", "v2", "v3", "v4", "v6")]
    if need32:
        w32 = M.resample_to_32k(wave16)
        x = w32[:, 1:] - 0.97 * w32[:, :-1]
        if any(v in need32 for v in ("v0", "v1", "v2")):
            m0 = _logmel(_stft_power(x, *STFT["v0"]), bank(1024))
            if "v0" in views:
                out["v0"] = m0
            for v in ("v1", "v2"):
                if v in views:
                    m = m0.clone()
                    m[:, :, torch.from_numpy(band_mask(v)), :] = FLOOR
                    out[v] = m
        for v in ("v3", "v4"):
            if v in views:
                n_fft, win, hop = STFT[v]
                out[v] = _logmel(_stft_power(x, n_fft, win, hop), bank(n_fft))
        if "v6" in views:
            out["v6"] = P.log_mel(w32)
    if "lra" in views or "lrb" in views:
        w8 = AF.resample(wave16, M.SR_IN, LR_RATE)
        x8 = w8[:, 1:] - 0.97 * w8[:, :-1]
        power = _stft_power(x8, LR_NFFT, LR_WIN, LR_HOP)
        rows = torch.from_numpy(lr_rows())
        mb = bank(1024)[rows][:, :LR_COLS]
        m = _logmel(power, mb)  # [B, 1, nrows, T]
        if "lrb" in views:
            out["lrb"] = m
        if "lra" in views:
            full = torch.full((m.shape[0], 1, M.N_MELS, m.shape[3]), FLOOR)
            full[:, :, rows, :] = m
            out["lra"] = full
    return out


# ── costs ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _taps(orig_sr: int, new_sr: int) -> int:
    from torchaudio.functional.functional import _get_sinc_resample_kernel

    g = math.gcd(orig_sr, new_sr)
    k, _ = _get_sinc_resample_kernel(orig_sr, new_sr, g, 6, 0.99, "sinc_interp_hann", None)
    return int(k.shape[-1])


@lru_cache(maxsize=None)
def _nnz(n_fft: int, rows_key: str) -> int:
    b = bank(n_fft)
    if rows_key == "lr":
        return int((b[torch.from_numpy(lr_rows())][:, :LR_COLS] != 0).sum())
    return int((b != 0).sum())


def frames_of(view: str, n16: int) -> int:
    """STFT frames of a clip of n16 samples (16 kHz) in a view."""
    if view in ("lra", "lrb"):
        n8 = math.ceil(n16 * LR_RATE / M.SR_IN)
        return 1 + (n8 - 1) // LR_HOP
    n32 = M.n32_of(n16)
    if view == "v6":
        return 1 + n32 // P.HOP
    return 1 + (n32 - 1) // STFT.get(view, STFT["v0"])[2]


def n_rows(view: str) -> int:
    return len(lr_rows()) if view == "lrb" else M.N_MELS


def frontend_macs(view: str, n16: int) -> float:
    """Multiply-add equivalents of the view's front end for one clip, counted as efficientat_mn.frontend_macs (resampler taps per output sample,
    pre-emphasis, per frame the window, a real FFT, the power, the nonzero mel weights, a log and a scale per mel value)."""
    fr = frames_of(view, n16)
    if view in ("lra", "lrb"):
        n8 = math.ceil(n16 * LR_RATE / M.SR_IN)
        per = LR_WIN + rfft_macs(LR_NFFT) + 2 * LR_COLS + _nnz(1024, "lr") + 2 * len(lr_rows())
        return float(_taps(M.SR_IN, LR_RATE) * n8 + (n8 - 1) + fr * per)
    n32 = M.n32_of(n16)
    resample = _taps(M.SR_IN, M.SR) * n32 if True else 0
    if view == "v6":
        return float(resample + P.frontend_macs(n32))
    n_fft, win, hop = STFT.get(view, STFT["v0"])
    per = win + rfft_macs(n_fft) + 2 * (n_fft // 2 + 1) + _nnz(n_fft, "all") + 2 * M.N_MELS
    return float(resample + (n32 - 1) + fr * per)
