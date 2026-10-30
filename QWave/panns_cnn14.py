"""PANNs CNN14 (Kong et al., TASLP 2020; github.com/qiuqiangkong/audioset_tagging_cnn, MIT licence), reimplemented for T15 as a second
pretrained audio model, with its documented 32 kHz front end and a W8A8 variant using the conventions of QWave/effnet_quant.py.

Architecture and module names follow the authors' models.py (ConvBlock, Cnn14) so that the released state dict loads with
torch.load(weights_only=True). The state dict also holds the buffers of torchlibrosa's STFT and mel filter bank; they are not modules here
(torch.stft and torchaudio's slaney mel bank replace them), and check_buffers() compares them with ours. No downloaded code is imported or run.

Front end (the Cnn14 defaults of the authors' config): 32 kHz mono, n_fft 1024, window 1024 (periodic Hann), hop 320, centred with reflect
padding, power spectrum, 64 mel rows 50 to 14000 Hz (slaney scale and norm, as librosa.filters.mel), 10 log10(max(mel, 1e-10)), then bn0 over
the 64 rows. Embedding: ReLU(fc1(max + mean over frequency-pooled time)) of the authors' forward, 2048-d.
"""
from __future__ import annotations

import copy
import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from QWave.effnet_quant import act_scale_dims, fake_quant_sym, rfft_macs, rtn_per_channel

SR, N_FFT, HOP, N_MELS, FMIN, FMAX, AMIN = 32000, 1024, 320, 64, 50.0, 14000.0, 1e-10


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn1, self.bn2 = nn.BatchNorm2d(cout), nn.BatchNorm2d(cout)

    def forward(self, x, pool=(2, 2)):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, pool)


class Cnn14(nn.Module):
    """Backbone on the log-mel map [B, 1, frames, 64]; forward returns the 2048-d embedding."""

    def __init__(self, classes_num=527):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(N_MELS)
        chans = [1, 64, 128, 256, 512, 1024, 2048]
        for i in range(6):
            setattr(self, f"conv_block{i + 1}", ConvBlock(chans[i], chans[i + 1]))
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, classes_num)

    def forward(self, x):
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for i in range(1, 6):
            x = getattr(self, f"conv_block{i}")(x)
        x = self.conv_block6(x, (1, 1))
        x = x.mean(dim=3)
        x = x.max(dim=2)[0] + x.mean(dim=2)
        return F.relu(self.fc1(x))

    def audioset_logits(self, x):
        return torch.sigmoid(self.fc_audioset(self.forward(x)))


@lru_cache(maxsize=None)
def mel_matrix() -> torch.Tensor:
    import torchaudio.functional as AF

    return AF.melscale_fbanks(N_FFT // 2 + 1, FMIN, FMAX, N_MELS, SR, norm="slaney", mel_scale="slaney").float()  # [513, 64]


def log_mel(wave32: torch.Tensor) -> torch.Tensor:
    """[B, n] at 32 kHz -> [B, 1, frames, 64]."""
    x = wave32.float()
    spec = torch.stft(x, N_FFT, hop_length=HOP, win_length=N_FFT, window=torch.hann_window(N_FFT, periodic=True, device=x.device),
                      center=True, pad_mode="reflect", return_complex=True)
    power = (spec.real ** 2 + spec.imag ** 2).transpose(1, 2)  # [B, frames, 513]
    mel = torch.matmul(power, mel_matrix().to(x.device))
    return (10.0 * torch.log10(torch.clamp(mel, min=AMIN))).unsqueeze(1)


def load_cnn14(path) -> Cnn14:
    """The released checkpoint (weights_only), without the torchlibrosa buffers; strict for everything else."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck["model"] if "model" in ck else ck
    keep = {k: v for k, v in sd.items() if not k.startswith(("spectrogram_extractor.", "logmel_extractor.", "spec_augmenter."))}
    m = Cnn14()
    m.load_state_dict(keep, strict=True)
    return m.eval()


def check_buffers(path) -> dict:
    """Max abs difference between the checkpoint's own mel filter bank and window/DFT buffers and ours."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck["model"] if "model" in ck else ck
    out = {"keys": [k for k in sd if k.startswith(("spectrogram_extractor.", "logmel_extractor."))]}
    melW = sd["logmel_extractor.melW"].float()
    out["mel_max_abs_diff"] = float((melW - mel_matrix()).abs().max())
    k = "spectrogram_extractor.stft.conv_real.weight"
    if k in sd:
        w = sd[k].float().squeeze(1)  # [513, 1024]: window * cos
        n = torch.arange(N_FFT).float()
        win = torch.hann_window(N_FFT, periodic=True)
        f = torch.arange(N_FFT // 2 + 1).float()[:, None]
        ours = win[None, :] * torch.cos(2 * math.pi * f * n[None, :] / N_FFT)
        out["stft_real_max_abs_diff"] = float((w - ours).abs().max())
    return out


class QuantCnn14(nn.Module):
    """Copy of the backbone with per-output-channel 8-bit weights on every Conv2d and Linear and 8-bit dynamic inputs
    (scales of effnet_quant.act_scale_dims for convolutions, one per row for Linear); BatchNorm, ReLU, pooling in FP32."""

    def __init__(self, base: Cnn14, wbits=8, abits=8, device="cpu"):
        super().__init__()
        m = copy.deepcopy(base).cpu().eval()
        self.abits = abits
        with torch.no_grad():
            for mod in m.modules():
                if isinstance(mod, nn.Conv2d):
                    dims = act_scale_dims(mod)
                elif isinstance(mod, nn.Linear) and mod is not m.fc_audioset:
                    dims = (-1,)
                else:
                    continue
                codes, s = rtn_per_channel(mod.weight.data, wbits)
                mod.weight.data.copy_(codes * s)
                mod.register_forward_pre_hook(self._hook(dims))
        self.net = m.to(device)

    def _hook(self, dims):
        def hook(mod, inp):
            return (fake_quant_sym(inp[0], self.abits, dims),) + tuple(inp[1:])
        return hook

    @torch.no_grad()
    def forward(self, x):
        return self.net(x)


def macs_at(model: Cnn14, frames: int) -> dict:
    """MACs and output elements of every Conv2d and of fc1 for one clip of `frames` mel frames (hooks on a zero input)."""
    tot = {"macs": 0.0, "out": 0.0}

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            kh, kw = mod.kernel_size
            n_out = out.shape[1] * out.shape[2] * out.shape[3]
            tot["macs"] += n_out * mod.in_channels * kh * kw
        else:
            n_out = mod.out_features
            tot["macs"] += mod.in_features * mod.out_features
        tot["out"] += n_out

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d) or m is model.fc1]
    try:
        with torch.no_grad():
            model(torch.zeros(1, 1, frames, N_MELS))
    finally:
        for h in hs:
            h.remove()
    return {"macs": tot["macs"], "out_elems": tot["out"]}


def mel_nnz() -> int:
    return int((mel_matrix() != 0).sum())


def frontend_macs(n_32k: int) -> float:
    """Multiply-add equivalents of the front end for a clip of n samples at 32 kHz (after resampling, counted separately): per frame a
    window, a real FFT, the power spectrum, the nonzero mel weights, a log and a scale per mel value."""
    frames = 1 + n_32k // HOP
    return float(frames * (N_FFT + rfft_macs(N_FFT) + 2 * (N_FFT // 2 + 1) + mel_nnz() + 2 * N_MELS))
