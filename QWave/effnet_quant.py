"""EfficientNet-B3 clip embeddings from mel-spectrogram images rendered in
memory, an 8-bit (W8A8) fake-quantized variant, and per-clip costs of both
tiers' front ends and of the convolutional backbone.

Why it exists. The repo's extractors (data/esc/esc-50-data_processing.py,
data/urban8k/urban8k-data_processing.py) read mel-spectrogram PNGs whose
rendering code is not in the repo, resize them to 300 x 300, run torchvision's
efficientnet_b3 (IMAGENET1K_V1) and store the output of a freshly initialized,
unseeded nn.Linear(1536, 1536) put in place of the classifier, so the stored
embeddings cannot be regenerated. This module renders the image from the
waveform in memory and stores the pooled 1536-d features (the input of that
Linear), so nothing in the extractor is random.

Front end (MelImageSpec, every field recorded with the embeddings):
  waveform    mono, 16 kHz, mean removed: the waveform the AST tier sees
              (idea 5 Stage 2's audio_shift.load_clip and noise mixing), so
              both tiers read identical audio
  STFT        window 2048 (periodic Hann), hop 512 (the paper), centred with
              zero padding, power spectrogram
  mel         128 triangular filters on the HTK scale mel(f) = 1125 ln(1 + f /
              700) (the paper's formula), 0 Hz to 8 kHz, Slaney area
              normalization (librosa's default filter norm)
  dB          10 log10(S / max S), floored at -80 dB (librosa's
              power_to_db(ref=np.max, top_db=80))
  image       min-max scaled per clip, through a 256-entry matplotlib colour
              map as uint8 (what a saved PNG holds), low frequencies at the
              bottom, one pixel per (mel band, frame)
  resize      bilinear with antialiasing to 300 x 300, rounded back to uint8
              and divided by 255 (transforms.Resize((300, 300)) on a PIL image
              followed by ToTensor), no ImageNet mean and std normalization
              (as in the repo's extractors)
The colour map, the filter normalization and the frame-per-pixel layout are
not documented anywhere for the original PNGs; they are choices, recorded in
every sidecar.

Backbone: torchvision efficientnet_b3, IMAGENET1K_V1 weights, classifier
replaced by nn.Identity: features -> global average pool -> 1536-d.

Quantization ("w8a8"), simulated in float (fake quantization), with the
conventions of the AST library (QWave/backbone_quant.py on
redesign/idea5-stage2):
  weights      every Conv2d (stem, expansion and projection 1 x 1, depthwise,
               squeeze-excitation reduce and expand, head 1 x 1): symmetric
               round to nearest per output channel, qmax = 2^(b-1) - 1,
               quantized once on the CPU (ties to even) and then moved to the
               device. Batch norm stays a separate FP32 layer; with one
               scale per output channel, folding it into the weights first
               would give the same codes up to float rounding, because the
               fold multiplies each output channel by one constant.
  activations  the input of every Conv2d, 8-bit symmetric dynamic,
               s = max|x| / 127 over the slice an integer kernel can take
               the scale out of (the sum of the convolution must not run
               across two scales):
                 1 x 1 convolutions (expansion, projection, the head
                   convolution, squeeze-excitation reduce and expand): one
                   scale per pixel over the channels, the AST's per-token
                   rule (a 1 x 1 convolution is a Linear on every pixel)
                 depthwise convolutions: one scale per channel over the
                   pixels (the sum runs over one channel's kernel window)
                 the dense 3 x 3 stem: one scale per clip over (C, H, W)
               One scale per clip and tensor everywhere was tried first and
               moved the embedding of 32 probe clips (numpy seed 0) to a mean
               relative error of 0.64 from FP32 (8-bit weights alone: 0.080);
               the rule above gives 0.099. Chosen on that error, no labels.
  kept FP32    batch norm, SiLU, the squeeze-excitation sigmoid and gate
               multiply, residual additions, the global pool and the stored
               embedding (the AST keeps LayerNorm, GELU, softmax arithmetic
               and residuals in FP32 the same way).
Activations are rounded on the device (MPS rounds exact ties away from zero,
the CPU to even; continuous activations almost never meet an exact tie).

Costs per clip (bit-ops = MACs x weight bits x activation bits):
  backbone     MACs of every Conv2d at 300 x 300 from forward hooks, split
               into convolutions, depthwise and squeeze-excitation; FP32 at
               32 x 32, w<b>a8 at b x 8. With rescale_price > 0, one rescale
               multiply-add per output element of every quantized
               convolution (per-channel weight scale times the activation
               scale applied to the integer accumulator) at that price
               (idea 5b's cost view (ii) charges 32 x 32 per rescale).
               Elementwise work is left out, as on the AST.
  front ends   STFT, mel and image steps of this module, and the AST's Kaldi
               fbank, counted in multiply-add equivalents at FP32 (32 x 32):
               a real FFT of n points as 2.5 n log2(n) flops = 1.25 n log2(n)
               multiply-adds, one per window sample, two per power bin, one
               per nonzero mel-filter weight, one per log, per normalized
               value and per interpolation tap.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

EMB_DIM = 1536
CONFIGS = ("fp32", "w8a8")


# ── front end: waveform -> mel-spectrogram image ───────────────────────────

@dataclass(frozen=True)
class MelImageSpec:
    sr: int = 16000
    n_fft: int = 2048
    hop: int = 512
    n_mels: int = 128
    f_min: float = 0.0
    mel_scale: str = "htk"
    mel_norm: str | None = "slaney"
    top_db: float = 80.0
    cmap: str = "magma"
    size: int = 300

    def as_dict(self) -> dict:
        return asdict(self)


_LUT: dict[str, torch.Tensor] = {}
_MEL: dict[MelImageSpec, nn.Module] = {}


def colormap_lut(name: str) -> torch.Tensor:
    """[256, 3] uint8 RGB entries of a matplotlib colour map, as
    Colormap(i, bytes=True) gives them (what a saved PNG holds)."""
    if name not in _LUT:
        import matplotlib

        cm = matplotlib.colormaps[name].resampled(256)
        _LUT[name] = torch.from_numpy(np.ascontiguousarray(cm(np.arange(256), bytes=True)[:, :3]))
    return _LUT[name]


def _mel_transform(spec: MelImageSpec) -> nn.Module:
    if spec not in _MEL:
        import torchaudio

        _MEL[spec] = torchaudio.transforms.MelSpectrogram(
            sample_rate=spec.sr, n_fft=spec.n_fft, win_length=spec.n_fft, hop_length=spec.hop,
            f_min=spec.f_min, f_max=None, n_mels=spec.n_mels, window_fn=torch.hann_window, power=2.0,
            center=True, pad_mode="constant", norm=spec.mel_norm, mel_scale=spec.mel_scale)
    return _MEL[spec]


def mel_power(wave: torch.Tensor, spec: MelImageSpec) -> torch.Tensor:
    """[..., n_mels, frames] power mel spectrogram."""
    return _mel_transform(spec)(wave.float())


def power_to_db(S: torch.Tensor, spec: MelImageSpec) -> torch.Tensor:
    """dB relative to each clip's maximum, floored at -top_db."""
    db = 10.0 * torch.log10(S.clamp_min(1e-10))
    ref = db.amax(dim=(-2, -1), keepdim=True)
    return torch.maximum(db - ref, torch.full_like(db, -spec.top_db))


def mel_db(wave: torch.Tensor, spec: MelImageSpec) -> torch.Tensor:
    return power_to_db(mel_power(wave, spec), spec)


def flatness_of(S: torch.Tensor) -> np.ndarray:
    """[B, n_mels, frames] power -> [B] mean over frames of the spectral
    flatness across mel bands (geometric over arithmetic mean). Frames that
    are zero in every band are skipped; a silent clip gets 0."""
    S = S.double()
    out = np.zeros(S.shape[0])
    for i in range(S.shape[0]):
        live = S[i].sum(0) > 0
        if live.any():
            x = S[i][:, live].clamp_min(1e-20)
            out[i] = float((x.log().mean(0).exp() / x.mean(0)).mean())
    return out


def db_to_image(db: torch.Tensor, spec: MelImageSpec) -> torch.Tensor:
    """[B, n_mels, frames] dB -> [B, 3, size, size] float in [0, 1]: min-max
    scaled per clip, colour map as uint8, flipped so low frequencies are at
    the bottom, resized (bilinear, antialiased), rounded to uint8, / 255."""
    lo = db.amin(dim=(-2, -1), keepdim=True)
    hi = db.amax(dim=(-2, -1), keepdim=True)
    rng = hi - lo
    x = torch.where(rng > 0, (db - lo) / torch.where(rng > 0, rng, torch.ones_like(rng)), torch.zeros_like(db))
    idx = torch.clamp((x * 256).long(), 0, 255)  # matplotlib: floor(x N), x == 1 -> N - 1
    rgb = colormap_lut(spec.cmap)[idx]  # [B, M, T, 3] uint8
    img = rgb.flip(1).permute(0, 3, 1, 2).float()
    img = F.interpolate(img, size=(spec.size, spec.size), mode="bilinear", align_corners=False, antialias=True)
    return img.round_().clamp_(0, 255).div_(255.0)


def mel_images(waves, spec: MelImageSpec = MelImageSpec(), with_flatness: bool = False):
    """List of 1-D float waveforms at spec.sr -> [B, 3, size, size] (and,
    with_flatness, the [B] spectral flatness of flatness_of from the same mel
    power). Clips of equal length are rendered together; the result is the
    same either way."""
    out = torch.empty(len(waves), 3, spec.size, spec.size)
    flat = np.zeros(len(waves))
    by_len: dict[int, list[int]] = {}
    for i, w in enumerate(waves):
        by_len.setdefault(len(w), []).append(i)
    for _, idx in by_len.items():
        batch = torch.stack([torch.as_tensor(np.asarray(waves[i]), dtype=torch.float32) for i in idx])
        S = mel_power(batch, spec)
        out[idx] = db_to_image(power_to_db(S, spec), spec)
        if with_flatness:
            flat[idx] = flatness_of(S)
    return (out, flat) if with_flatness else out


# ── backbone ───────────────────────────────────────────────────────────────

def efficientnet_b3_pooled(pretrained: bool = True) -> nn.Module:
    """torchvision efficientnet_b3 (IMAGENET1K_V1 when pretrained) whose
    classifier is nn.Identity: output = global average pool of the last
    feature map, 1536-d. In eval mode."""
    from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

    m = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None)
    m.classifier = nn.Identity()
    return m.eval()


def rtn_per_channel(w: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric round-to-nearest codes and scales, one scale per output
    channel (dim 0). Returns (codes shaped like w, scales [out, 1, ...])."""
    if not 2 <= bits <= 8:
        raise ValueError("bits must be in [2, 8]")
    qmax = 2 ** (bits - 1) - 1
    s = w.abs().amax(dim=tuple(range(1, w.dim())), keepdim=True).clamp_min(1e-12) / qmax
    return torch.clamp(torch.round(w / s), -qmax, qmax), s


def fake_quant_sym(x: torch.Tensor, bits: int, dims) -> torch.Tensor:
    """Symmetric dynamic fake quantization with one scale per slice reduced
    over `dims`: s = max|x| / (2^(bits-1) - 1). An all-zero slice stays zero."""
    qmax = 2 ** (bits - 1) - 1
    s = x.abs().amax(dim=dims, keepdim=True).clamp_min(1e-30) / qmax
    return torch.round(x / s) * s


def act_scale_dims(conv: nn.Conv2d) -> tuple[int, ...]:
    """Axes of a [B, C, H, W] Conv2d input that one activation scale spans:
    channels for a 1 x 1 convolution (per pixel), pixels for a depthwise one
    (per channel), everything but the batch otherwise (per clip)."""
    if conv.groups > 1 and conv.groups == conv.in_channels:
        return (2, 3)
    if conv.kernel_size == (1, 1):
        return (1,)
    return (1, 2, 3)


def parse_config(name: str) -> tuple[int, int | None]:
    """(weight bits, activation bits) of 'fp32' (32, None) or 'w<b>a8'."""
    if name == "fp32":
        return 32, None
    if len(name) == 4 and name[0] == "w" and name[2:] == "a8" and name[1].isdigit():
        return int(name[1]), 8
    raise ValueError(f"bad configuration name '{name}'")


class QuantEffNet(nn.Module):
    """An efficientnet_b3_pooled copy whose Conv2d weights are fake-quantized
    per output channel and whose Conv2d inputs are fake-quantized with the
    scales of act_scale_dims (see the module docstring). Weights are
    quantized on the CPU, then the copy is moved to `device`."""

    def __init__(self, base: nn.Module, weight_bits: int = 8, act_bits: int = 8, device="cpu"):
        super().__init__()
        import copy

        self.weight_bits, self.act_bits = weight_bits, act_bits
        m = copy.deepcopy(base).cpu().eval()
        self.n_conv = 0
        with torch.no_grad():
            for mod in m.modules():
                if isinstance(mod, nn.Conv2d):
                    codes, s = rtn_per_channel(mod.weight.data, weight_bits)
                    mod.weight.data.copy_(codes * s)
                    mod.register_forward_pre_hook(self._hook(act_scale_dims(mod)))
                    self.n_conv += 1
        self.net = m.to(device)

    def _hook(self, dims):
        def hook(mod, inp):
            return (fake_quant_sym(inp[0], self.act_bits, dims),) + tuple(inp[1:])
        return hook

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_backbone(config: str, base: nn.Module, device="cpu") -> nn.Module:
    wb, ab = parse_config(config)
    if wb == 32:
        import copy

        return copy.deepcopy(base).to(device).eval()
    return QuantEffNet(base, wb, ab, device).eval()


# ── costs ──────────────────────────────────────────────────────────────────

def conv_macs(model: nn.Module, size: int = 300) -> dict:
    """MACs and output elements of every Conv2d (and Linear) of a backbone on
    one [3, size, size] image, from forward hooks on a zero input, split into
    'conv', 'depthwise' (groups > 1) and 'squeeze_excitation' (1 x 1 output
    map). Also the parameter count."""
    parts = {"conv": [0, 0], "depthwise": [0, 0], "squeeze_excitation": [0, 0], "linear": [0, 0]}
    hooks = []

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            kh, kw = mod.kernel_size
            n_out = out.shape[1] * out.shape[2] * out.shape[3]
            macs = n_out * (mod.in_channels // mod.groups) * kh * kw
            key = "depthwise" if mod.groups > 1 else ("squeeze_excitation" if out.shape[2] == 1 else "conv")
        elif isinstance(mod, nn.Linear):
            n_out, macs, key = mod.out_features, mod.in_features * mod.out_features, "linear"
        else:
            return
        parts[key][0] += macs
        parts[key][1] += n_out

    for mm in model.modules():
        if isinstance(mm, (nn.Conv2d, nn.Linear)):
            hooks.append(mm.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            p = next(model.parameters())
            model(torch.zeros(1, 3, size, size, device=p.device, dtype=p.dtype))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    macs = sum(v[0] for v in parts.values())
    return {"input": [3, size, size], "macs": float(macs), "out_elems": float(sum(v[1] for v in parts.values())),
            "parts_macs": {k: float(v[0]) for k, v in parts.items()},
            "parts_out_elems": {k: float(v[1]) for k, v in parts.items()},
            "params": int(sum(p.numel() for p in model.parameters()))}


def backbone_bitops(macs: dict, config: str, rescale_price: float = 0.0) -> float:
    """Bit-ops per clip of a configuration from conv_macs output: MACs x
    weight bits x activation bits, plus (quantized only) one rescale per
    output element at rescale_price."""
    wb, ab = parse_config(config)
    if wb == 32:
        return macs["macs"] * 32 * 32
    return macs["macs"] * wb * ab + macs["out_elems"] * rescale_price


def weight_bits_stored(model: nn.Module, config: str) -> float:
    """Bits to store the backbone: Conv2d weights at the configuration's
    bits plus one fp32 scale per output channel (FP32: 32 bits), every other
    parameter (batch norm, biases) in fp32."""
    wb, _ = parse_config(config)
    bits = 0.0
    for mod in model.modules():
        for name, p in mod.named_parameters(recurse=False):
            if isinstance(mod, nn.Conv2d) and name == "weight" and wb < 32:
                bits += wb * p.numel() + 32 * p.shape[0]
            else:
                bits += 32 * p.numel()
    return bits


def rfft_macs(n: int) -> float:
    """Multiply-add equivalents of a real FFT of n points: 2.5 n log2 n
    flops, two flops per multiply-add."""
    return 1.25 * n * math.log2(n)


def mel_filter_nnz(spec: MelImageSpec) -> int:
    fb = _mel_transform(spec).mel_scale.fb  # [n_freqs, n_mels]
    return int((fb != 0).sum())


def mel_image_frontend_macs(n_samples: int, spec: MelImageSpec = MelImageSpec()) -> float:
    """Multiply-add equivalents of this module's front end for one clip of
    n_samples at spec.sr: per frame a window, a real FFT, the power
    spectrum, the nonzero mel weights and a log; then the min-max scaling of
    every value and a bilinear resize with 2 taps per axis on 3 channels
    (both axes are upsampled to 300 for every clip here)."""
    frames = 1 + n_samples // spec.hop
    bins = spec.n_fft // 2 + 1
    per_frame = spec.n_fft + rfft_macs(spec.n_fft) + 2 * bins + spec.n_mels
    stft = frames * per_frame + frames * mel_filter_nnz(spec)
    scale = spec.n_mels * frames
    resize = 3 * (spec.n_mels * spec.size * 2 + spec.size * spec.size * 2)
    return float(stft + scale + resize)


def ast_fbank_frontend_macs(n_samples: int, sr: int = 16000, num_mel_bins: int = 128,
                            max_length: int = 1024) -> float:
    """Multiply-add equivalents of the AST feature extractor's Kaldi fbank on
    one clip: 25 ms frames (400 samples) every 10 ms (160), each with DC
    removal, pre-emphasis and a Hann window (one per sample), a 512-point
    real FFT, the power spectrum, the nonzero mel weights and a log; then the
    mean and std normalization of the padded [max_length, num_mel_bins]
    input."""
    from torchaudio.compliance.kaldi import get_mel_banks

    win, shift, n_fft = int(0.025 * sr), int(0.010 * sr), 512
    frames = 0 if n_samples < win else 1 + (n_samples - win) // shift
    banks, _ = get_mel_banks(num_mel_bins, n_fft, float(sr), 20.0, 0.0, 100.0, -500.0, 1.0)
    nnz = int((banks != 0).sum())
    per_frame = 3 * win + rfft_macs(n_fft) + 2 * (n_fft // 2 + 1) + nnz + num_mel_bins
    return float(frames * per_frame + max_length * num_mel_bins)


FP32_BITOPS_PER_MAC = 32 * 32
