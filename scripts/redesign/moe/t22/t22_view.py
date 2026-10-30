"""T22 coarse view and coarse expert (ANALYSIS_PLAN.md section C1): the 16 kHz waveform every tier reads, resampled to 8 kHz,
a 64-band log-mel front end with a 20 ms hop (twice MN10's 10 ms), and an EfficientAT MN fine-tuned on it.

Front end at 8 kHz: torchaudio's default sinc resampler 16 -> 8 kHz; pre-emphasis 0.97; symmetric Hann window of 200 samples (25 ms),
n_fft 256, hop 160 (20 ms), centred with reflect padding; power; 64 Kaldi mel filters from 0 to 4 kHz (get_mel_banks, the same
routine MN10's front end uses, one zero column padded); (log(mel + 1e-5) + 4.5) / 5, MN10's normalisation.

Costs per clip (priced bit-ops, the repo convention of QWave/efficientat_mn.py and t15_costs.py): front end multiply-add
equivalents x 32 x 32; backbone MACs x 8 x 8 (W8A8) plus one rescale per output element at 32 x 32; classifier Linear priced by
QWave.costs.cost (FP32, 32 x 32).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torch import nn

from QWave import efficientat_mn as M
from QWave.effnet_quant import rfft_macs

SR_IN, SR_LR = 16000, 8000
N_FFT, WIN, HOP, N_MELS = 256, 200, 160, 64
F_MIN, F_MAX = 0.0, 4000.0
FLOOR = (math.log(1e-5) + 4.5) / 5.0
PAD16 = {"esc50": 80000, "urban": 64587, "tau": 160000}  # one input shape per task: zero-pad the 16 kHz waveform at the end


@lru_cache(maxsize=None)
def mel_bank() -> torch.Tensor:
    from torchaudio.compliance.kaldi import get_mel_banks

    banks, _ = get_mel_banks(N_MELS, N_FFT, float(SR_LR), F_MIN, F_MAX, 100.0, -500.0, 1.0)
    b = F.pad(banks, (0, 1)).float()  # [64, 129]
    assert (b.sum(1) > 0).all(), "empty mel row"
    return b


def logmel(wave16: torch.Tensor) -> torch.Tensor:
    """[B, n] 16 kHz -> [B, 1, 64, frames]."""
    w8 = AF.resample(wave16.float(), SR_IN, SR_LR)
    x = w8[:, 1:] - 0.97 * w8[:, :-1]
    win = torch.hann_window(WIN, periodic=False, device=x.device)
    s = torch.stft(x, N_FFT, hop_length=HOP, win_length=WIN, window=win, center=True, pad_mode="reflect", normalized=False,
                   return_complex=True)
    p = s.real ** 2 + s.imag ** 2
    return ((torch.log(torch.matmul(mel_bank().to(x.device), p) + 1e-5) + 4.5) / 5.0).unsqueeze(1)


def pad16(w: np.ndarray, task: str) -> np.ndarray:
    n = PAD16[task]
    if len(w) > n:
        raise ValueError(f"clip longer than the task's padded length ({len(w)} > {n})")
    return np.pad(w, (0, n - len(w)))


def n8_of(n16: int) -> int:
    return math.ceil(n16 * SR_LR / SR_IN)


def frames_of(n16: int) -> int:
    return 1 + (n8_of(n16) - 1) // HOP  # pre-emphasis leaves n8 - 1 samples; centred STFT


@lru_cache(maxsize=None)
def resample_taps() -> int:
    from torchaudio.functional.functional import _get_sinc_resample_kernel

    g = math.gcd(SR_IN, SR_LR)
    k, _ = _get_sinc_resample_kernel(SR_IN, SR_LR, g, 6, 0.99, "sinc_interp_hann", None)
    return int(k.shape[-1])


def frontend_macs(n16: int) -> float:
    """Resampler taps per output sample, pre-emphasis, per frame the window, a 256-point real FFT, two per power bin, the nonzero
    mel weights, a log and a scale per mel value (the counting rule of QWave.efficientat_mn.frontend_macs)."""
    n8 = n8_of(n16)
    per = WIN + rfft_macs(N_FFT) + 2 * (N_FFT // 2 + 1) + int((mel_bank() != 0).sum()) + 2 * N_MELS
    return float(resample_taps() * n8 + (n8 - 1) + frames_of(n16) * per)


def feature_macs(model: nn.Module, rows: int, frames: int) -> tuple[float, float]:
    """(MACs, output elements) of every Conv2d and Linear in model.features on one [1, 1, rows, frames] input."""
    parts = [0.0, 0.0]

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            kh, kw = mod.kernel_size
            n_out = out.shape[1] * out.shape[2] * out.shape[3]
            parts[0] += n_out * (mod.in_channels // mod.groups) * kh * kw
        else:
            n_out = mod.out_features
            parts[0] += mod.in_features * mod.out_features
        parts[1] += n_out

    hs = [m.register_forward_hook(hook) for m in model.features.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    try:
        with torch.no_grad():
            model.features(torch.zeros(1, 1, rows, frames))
    finally:
        for h in hs:
            h.remove()
    return parts[0], parts[1]


class Coarse(nn.Module):
    """EfficientAT MN features (AudioSet start) + dropout + a new Linear classifier on the pooled embedding."""

    def __init__(self, base: M.MN, n_cls: int, dropout: float = 0.2):
        super().__init__()
        self.features = base.features
        self.emb_dim = base.emb_dim
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(base.emb_dim, n_cls)
        for m in self.features.modules():  # amendment 1: BN momentum 0.1 while fine-tuning (EfficientAT's 0.01 keeps AudioSet statistics)
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.1

    def embed(self, x):
        return F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)

    def forward(self, x):
        return self.fc(self.drop(self.embed(x)))


class _FeatWrap(nn.Module):
    """Gives QuantCNN the .features / forward(pooled) interface it expects."""

    def __init__(self, coarse: Coarse):
        super().__init__()
        self.features = coarse.features

    def forward(self, x):
        return F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)


def quantize_w8a8(model: Coarse, device="cpu"):
    """Stage 3c PTQ recipe (QWave.efficientat_mn.QuantCNN): per-output-channel 8-bit weights, 8-bit dynamic activations on every
    Conv2d and squeeze-excitation Linear inside .features; batch norm, activations, pool and the FP32 classifier unchanged.
    Returns f(x) -> logits."""
    q = M.QuantCNN(_FeatWrap(model), 8, 8, device).eval()
    fc = nn.Linear(model.fc.in_features, model.fc.out_features)
    fc.load_state_dict(model.fc.state_dict())
    fc = fc.to(device).eval()

    @torch.no_grad()
    def f(x):
        return fc(q(x))

    return f
