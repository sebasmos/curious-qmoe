"""EfficientAT MobileNetV3 (Schmid, Koutini and Widmer, ICASSP 2023), an
AudioSet-pretrained small CNN, as a static system and a cheap tier for idea 6
Stage 3b: its documented front end, pooled clip embeddings, a W8A8 variant
with the conventions of QWave/effnet_quant.py, and per-clip costs.

Source. Architecture and front end follow github.com/fschmid56/EfficientAT
(MIT licence, Copyright (c) 2022 Florian Schmid) at commit a425fdc:
models/mn/model.py, models/mn/block_types.py, models/mn/utils.py and
models/preprocess.py, rewritten here for the configuration the AudioSet
checkpoints use (get_model defaults: head 'mlp', squeeze-excitation on the
channel axis only, se_r 4, strides 2 2 2 2). Module names match the
released state dicts, so load_mn() loads them with strict=True through
torch.load(weights_only=True): no pickled code runs. Weights: release
v0.0.1, mn04_as_mAP_432.pt (width 0.4) and mn10_as_mAP_471.pt (width 1.0);
Stage 3c adds mn05_as_mAP_443.pt (0.5) and mn20_as_mAP_478.pt (2.0).

Front end (AugmentMelSTFT in eval mode, its defaults): 32 kHz mono;
pre-emphasis y[t] = x[t+1] - 0.97 x[t]; STFT with n_fft 1024, a symmetric
Hann window of 800 samples, hop 320, centred with reflect padding; power;
128 Kaldi mel filters from 0 to 15 kHz (get_mel_banks with vtln_low 100,
vtln_high -500, padded with one zero column); log(mel + 1e-5); (x + 4.5) / 5.
The waveforms here are the 16 kHz mono audio both other tiers read
(audio_shift.load_clip and mix), resampled to 32 kHz by
torchaudio.functional.resample with its defaults, so the model sees nothing
above 8 kHz. Clips run at their own length (no padding): the network is
convolutional and ends in a global average pool.

Embedding: the global average pool of the last 1 x 1 convolution's output
(the 'features' MN.forward returns): 384-d for mn04, 960-d for mn10. The
AudioSet classifier is kept only for the sanity check of audioset_logits().

Quantization ("w8a8"), fake quantization in float with the rules of
QWave/effnet_quant.py: every Conv2d and every squeeze-excitation Linear has
symmetric per-output-channel 8-bit weights (quantized once on the CPU), and
its input is 8-bit symmetric dynamic with a scale an integer kernel can
factor out: per pixel over the channels for 1 x 1 convolutions, per channel
over the pixels for depthwise ones, per clip for the dense 3 x 3 stem, and
per row (per clip) for the Linear layers. Batch norm, hardswish, ReLU, the
squeeze-excitation mean, sigmoid and gate, residuals and the pool stay FP32.

Costs per clip (bit-ops = MACs x weight bits x activation bits): MACs and
output elements of every Conv2d and Linear in the feature extractor from
forward hooks at the clip's frame count; FP32 at 32 x 32, w8a8 at 8 x 8
plus one rescale per output element at rescale_price (32 x 32 in the priced
view). Front end in multiply-add equivalents at 32 x 32, counted as in
QWave/effnet_quant.py: the resampler's taps, one per pre-emphasis sample,
per frame the 800 window samples, a real FFT of 1024 points, two per power
bin, the nonzero mel weights, a log and a normalization per mel value.
"""
from __future__ import annotations

import copy
import math
from functools import lru_cache, partial

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.misc import Conv2dNormActivation

from QWave.effnet_quant import act_scale_dims, fake_quant_sym, rfft_macs, rtn_per_channel

SR_IN = 16000
SR = 32000
N_FFT, WIN, HOP, N_MELS = 1024, 800, 320, 128
F_MIN, F_MAX = 0.0, 15000.0  # sr // 2 - fmax_aug_range // 2 with the defaults
WIDTHS = {"mn04": 0.4, "mn05": 0.5, "mn10": 1.0, "mn20": 2.0}  # Stage 3c adds mn05 and mn20
CONFIGS = ("fp32", "w8a8")


# ── architecture (EfficientAT models/mn, AudioSet configuration) ───────────

def make_divisible(v: float, divisor: int = 8, min_value: int | None = None) -> int:
    min_value = divisor if min_value is None else min_value
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# (input, kernel, expanded, output, squeeze-excitation, hardswish, stride) of the 15 blocks
BNECK = [(16, 3, 16, 16, False, False, 1), (16, 3, 64, 24, False, False, 2), (24, 3, 72, 24, False, False, 1),
         (24, 5, 72, 40, True, False, 2), (40, 5, 120, 40, True, False, 1), (40, 5, 120, 40, True, False, 1),
         (40, 3, 240, 80, False, True, 2), (80, 3, 200, 80, False, True, 1), (80, 3, 184, 80, False, True, 1),
         (80, 3, 184, 80, False, True, 1), (80, 3, 480, 112, True, True, 1), (112, 3, 672, 112, True, True, 1),
         (112, 5, 672, 160, True, True, 2), (160, 5, 960, 160, True, True, 1), (160, 5, 960, 160, True, True, 1)]


class SqueezeExcitation(nn.Module):
    """Channel squeeze-excitation with Linear layers (block_types.SqueezeExcitation, se_dim 1)."""

    def __init__(self, channels: int, squeeze: int):
        super().__init__()
        self.fc1 = nn.Linear(channels, squeeze)
        self.fc2 = nn.Linear(squeeze, channels)
        self.activation = nn.ReLU()
        self.scale_activation = nn.Sigmoid()

    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        shape = s.shape
        s = self.fc2(self.activation(self.fc1(s.flatten(1))))
        return self.scale_activation(s).view(shape) * x


class ConcurrentSE(nn.Module):
    """block_types.ConcurrentSEBlock with se_dims [1] and se_agg 'max': one layer, so max over it is the identity."""

    def __init__(self, channels: int, se_r: int = 4):
        super().__init__()
        self.conc_se_layers = nn.ModuleList([SqueezeExcitation(channels, make_divisible(channels // se_r, 8))])

    def forward(self, x):
        return self.conc_se_layers[0](x)


class InvertedResidual(nn.Module):
    def __init__(self, cin, k, cexp, cout, se, hs, stride, norm):
        super().__init__()
        act = nn.Hardswish if hs else nn.ReLU
        layers = []
        if cexp != cin:
            layers.append(Conv2dNormActivation(cin, cexp, kernel_size=1, norm_layer=norm, activation_layer=act))
        layers.append(Conv2dNormActivation(cexp, cexp, kernel_size=k, stride=stride, groups=cexp, norm_layer=norm,
                                           activation_layer=act))
        if se:
            layers.append(ConcurrentSE(cexp))
        layers.append(Conv2dNormActivation(cexp, cout, kernel_size=1, norm_layer=norm, activation_layer=None))
        self.block = nn.Sequential(*layers)
        self.use_res_connect = stride == 1 and cin == cout

    def forward(self, x):
        y = self.block(x)
        return y + x if self.use_res_connect else y


class MN(nn.Module):
    """EfficientAT's MN at a width multiplier: features (stem, 15 blocks, last 1 x 1 convolution) and the 'mlp'
    AudioSet classifier. forward() returns the pooled embedding [B, C]."""

    def __init__(self, width_mult: float, num_classes: int = 527, dropout: float = 0.2):
        super().__init__()
        adj = partial(lambda c, w: make_divisible(c * w, 8), w=width_mult)
        norm = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        cfg = [(adj(a), k, adj(e), adj(o), se, hs, s) for a, k, e, o, se, hs, s in BNECK]
        layers = [Conv2dNormActivation(1, cfg[0][0], kernel_size=3, stride=2, norm_layer=norm,
                                       activation_layer=nn.Hardswish)]
        layers += [InvertedResidual(*c, norm) for c in cfg]
        last_in = cfg[-1][3]
        self.emb_dim = 6 * last_in
        layers.append(Conv2dNormActivation(last_in, self.emb_dim, kernel_size=1, norm_layer=norm,
                                           activation_layer=nn.Hardswish))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(self.emb_dim, adj(1280)),
                                        nn.Hardswish(), nn.Dropout(dropout), nn.Linear(adj(1280), num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)

    def audioset_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_mn(name: str) -> MN:
    return MN(WIDTHS[name]).eval()


def load_mn(name: str, path) -> MN:
    """An MN with a released AudioSet state dict loaded strictly (torch.load with weights_only=True)."""
    m = build_mn(name)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    m.load_state_dict(sd, strict=True)
    return m.eval()


# ── front end ──────────────────────────────────────────────────────────────

def resample_to_32k(wave: torch.Tensor) -> torch.Tensor:
    import torchaudio.functional as AF

    return AF.resample(wave, SR_IN, SR)


@lru_cache(maxsize=None)
def _mel_basis() -> torch.Tensor:
    from torchaudio.compliance.kaldi import get_mel_banks

    banks, _ = get_mel_banks(N_MELS, N_FFT, float(SR), F_MIN, F_MAX, 100.0, -500.0, 1.0)
    return F.pad(banks, (0, 1)).float()  # [128, 513]


def frames_of(n_32k: int) -> int:
    """STFT frames of a 32 kHz clip of n samples after pre-emphasis (n - 1 samples, centred)."""
    return 1 + (n_32k - 1) // HOP


def log_mel(wave32: torch.Tensor) -> torch.Tensor:
    """[B, n] at 32 kHz -> [B, 1, 128, frames], AugmentMelSTFT.forward in eval mode."""
    x = wave32.float()
    x = x[:, 1:] - 0.97 * x[:, :-1]
    win = torch.hann_window(WIN, periodic=False, device=x.device)
    spec = torch.stft(x, N_FFT, hop_length=HOP, win_length=WIN, window=win, center=True, pad_mode="reflect",
                      normalized=False, return_complex=True)
    power = spec.real ** 2 + spec.imag ** 2
    mel = torch.matmul(_mel_basis().to(x.device), power)
    return ((torch.log(mel + 1e-5) + 4.5) / 5.0).unsqueeze(1)


def model_input(waves16) -> list[tuple[list[int], torch.Tensor]]:
    """List of 1-D 16 kHz waveforms -> [(indices, [b, 1, 128, frames])], one batch per distinct length."""
    by_len: dict[int, list[int]] = {}
    for i, w in enumerate(waves16):
        by_len.setdefault(len(w), []).append(i)
    out = []
    for _, idx in sorted(by_len.items()):
        x = torch.stack([torch.as_tensor(waves16[i], dtype=torch.float32) for i in idx])
        out.append((idx, log_mel(resample_to_32k(x))))
    return out


# ── W8A8 ───────────────────────────────────────────────────────────────────

class QuantCNN(nn.Module):
    """A copy of `base` whose Conv2d and Linear layers inside `.features` have per-output-channel weights at
    weight_bits and inputs fake-quantized at act_bits with the scales of QWave.effnet_quant.act_scale_dims (Conv2d)
    or one scale per row (Linear). Weights are quantized on the CPU, then the copy moves to `device`."""

    def __init__(self, base: nn.Module, weight_bits: int = 8, act_bits: int = 8, device="cpu"):
        super().__init__()
        self.weight_bits, self.act_bits = weight_bits, act_bits
        m = copy.deepcopy(base).cpu().eval()
        self.n_conv = self.n_linear = 0
        with torch.no_grad():
            for mod in m.features.modules():
                if isinstance(mod, nn.Conv2d):
                    dims = act_scale_dims(mod)
                    self.n_conv += 1
                elif isinstance(mod, nn.Linear):
                    dims = (-1,)
                    self.n_linear += 1
                else:
                    continue
                codes, s = rtn_per_channel(mod.weight.data, weight_bits)
                mod.weight.data.copy_(codes * s)
                mod.register_forward_pre_hook(self._hook(dims))
        self.net = m.to(device)

    def _hook(self, dims):
        def hook(mod, inp):
            return (fake_quant_sym(inp[0], self.act_bits, dims),) + tuple(inp[1:])
        return hook

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_backbone(config: str, base: nn.Module, device="cpu") -> nn.Module:
    if config == "fp32":
        return copy.deepcopy(base).to(device).eval()
    if config == "w8a8":
        return QuantCNN(base, 8, 8, device).eval()
    raise ValueError(f"bad configuration name '{config}'")


# ── costs ──────────────────────────────────────────────────────────────────

def feature_macs(model: MN, frames: int) -> dict:
    """MACs and output elements of every Conv2d and Linear in model.features on one [1, 1, 128, frames] input,
    from forward hooks, split into 'conv' (dense and 1 x 1), 'depthwise' and 'squeeze_excitation' (Linear)."""
    parts = {"conv": [0, 0], "depthwise": [0, 0], "squeeze_excitation": [0, 0]}
    hooks = []

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            kh, kw = mod.kernel_size
            n_out = out.shape[1] * out.shape[2] * out.shape[3]
            key = "depthwise" if mod.groups > 1 else "conv"
            macs = n_out * (mod.in_channels // mod.groups) * kh * kw
        else:
            n_out, macs, key = mod.out_features, mod.in_features * mod.out_features, "squeeze_excitation"
        parts[key][0] += macs
        parts[key][1] += n_out

    for mm in model.features.modules():
        if isinstance(mm, (nn.Conv2d, nn.Linear)):
            hooks.append(mm.register_forward_hook(hook))
    try:
        with torch.no_grad():
            p = next(model.parameters())
            model.features(torch.zeros(1, 1, N_MELS, frames, device=p.device, dtype=p.dtype))
    finally:
        for h in hooks:
            h.remove()
    return {"frames": frames, "macs": float(sum(v[0] for v in parts.values())),
            "out_elems": float(sum(v[1] for v in parts.values())),
            "parts_macs": {k: float(v[0]) for k, v in parts.items()}}


def backbone_bitops(macs: dict, config: str, rescale_price: float = 0.0) -> float:
    if config == "fp32":
        return macs["macs"] * 32 * 32
    if config == "w8a8":
        return macs["macs"] * 8 * 8 + macs["out_elems"] * rescale_price
    raise ValueError(config)


@lru_cache(maxsize=None)
def resample_taps() -> tuple[int, int, int]:
    """(orig, new, taps) of torchaudio's 16 kHz -> 32 kHz sinc kernel with its defaults."""
    from torchaudio.functional.functional import _get_sinc_resample_kernel

    g = math.gcd(SR_IN, SR)
    k, _ = _get_sinc_resample_kernel(SR_IN, SR, g, 6, 0.99, "sinc_interp_hann", None)
    return SR_IN // g, SR // g, int(k.shape[-1])


def mel_nnz() -> int:
    return int((_mel_basis() != 0).sum())


def frontend_macs(n_16k: int) -> float:
    """Multiply-add equivalents of the front end for one clip of n samples at 16 kHz: resampling (taps per output
    sample), pre-emphasis, then per frame the window, a 1024-point real FFT, the power spectrum, the nonzero mel
    weights, a log and a normalization per mel value."""
    orig, new, taps = resample_taps()
    n32 = math.ceil(new * n_16k / orig)
    frames = frames_of(n32)
    per_frame = WIN + rfft_macs(N_FFT) + 2 * (N_FFT // 2 + 1) + mel_nnz() + 2 * N_MELS
    return float(taps * n32 + (n32 - 1) + frames * per_frame)


def n32_of(n_16k: int) -> int:
    orig, new, _ = resample_taps()
    return math.ceil(new * n_16k / orig)


def weight_bits_stored(model: MN, config: str) -> float:
    """Bits to store the feature extractor: Conv2d and Linear weights at the configuration's bits plus one fp32
    scale per output channel (FP32: 32 bits), every other parameter in fp32. The AudioSet classifier is left out."""
    bits = 0.0
    for mod in model.features.modules():
        for name, p in mod.named_parameters(recurse=False):
            if isinstance(mod, (nn.Conv2d, nn.Linear)) and name == "weight" and config == "w8a8":
                bits += 8 * p.numel() + 32 * p.shape[0]
            else:
                bits += 32 * p.numel()
    return bits
