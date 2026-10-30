"""EfficientAT dynamic MobileNet (DyMN; Schmid, Masoudian, Koutini and Widmer,
2023), the dymn04 AudioSet checkpoint, as a static system and a cheap tier for
idea 6 Stage 3c: the same front end and pooled embedding as efficientat_mn,
a W8A8 variant and per-clip costs.

Source. Architecture follows github.com/fschmid56/EfficientAT (MIT licence,
Copyright (c) 2022 Florian Schmid) at commit a425fdc: models/dymn/model.py,
models/dymn/dy_block.py and models/dymn/utils.py, rewritten here for the
configuration get_model() builds for an AudioSet checkpoint (head 'mlp',
use_dy_blocks 'all', context_ratio 4, context size 32 to 128, dyrelu_k 2,
dyconv_k 4, strides 2 2 2 2, temperature 1 since pretrain_final_temp is 1
for AudioSet weights). Module names match the released state dict, so
load_dymn() loads dymn04_as.pt with strict=True through
torch.load(weights_only=True).

Each block (DY_Block) computes a context from its input (ContextGen: the
frequency and time averages of the input, one shared 1 x 1 convolution,
batch norm, hardswish; the mean over the sequence is the context vector
h_c, two more 1 x 1 convolutions give the coordinate-attention sequences).
h_c parameterizes the dynamic convolutions (DynamicConv: softmax attention
over k = 4 kernels, one aggregated kernel per clip) and the dynamic ReLU
(DyReLUB: the maximum of two per-channel affine maps whose coefficients come
from h_c). Coordinate attention multiplies the depthwise output by the
sigmoids of the two sequences. Expand (1 x 1), depthwise (k x k) and project
(1 x 1) are all dynamic convolutions.

Embedding: the global average pool of out_c (the last 1 x 1 convolution):
384-d for dymn04 (6 x 64).

Quantization ("w8a8"), fake quantization in float with the rules of
QWave/effnet_quant.py: every static Conv2d (stem, out_c, the three
ContextGen convolutions) and every Linear (kernel attention, DyReLU
coefficients) has symmetric per-output-channel 8-bit weights quantized
once; every dynamic convolution's aggregated kernel is quantized per output
channel to 8 bits per clip at run time (its kernel is input dependent, so it
cannot be stored quantized); inputs are 8-bit symmetric dynamic with the
scales an integer kernel can factor out (per pixel for 1 x 1, per channel
for depthwise, per clip for the dense 3 x 3 stem, per row for Linear).
Batch norm, the activations, sigmoids, the DyReLU maximum, coordinate
attention products, residuals and pools stay FP32.

Costs per clip: MACs and output elements of every static Conv2d and Linear
(forward hooks) and of every DynamicConv (output elements x in_channels /
groups x k x k), at the clip's frame count; the kernel aggregation
(k x kernel elements per clip, the attention @ weight product) is counted
separately at 32 x 32 in both configurations, and in W8A8 the aggregated
kernel's per-clip quantization adds nothing to the bit-op count (elementwise
work is left out everywhere). Front end: efficientat_mn.frontend_macs.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.misc import Conv2dNormActivation

from QWave.effnet_quant import act_scale_dims, fake_quant_sym, rtn_per_channel
from QWave.efficientat_mn import BNECK, make_divisible

WIDTHS = {"dymn04": 0.4}
WEIGHT_FILES = {"dymn04": "dymn04_as.pt"}
CONFIGS = ("fp32", "w8a8")
K_CONV, M_RELU, CONTEXT_RATIO, MIN_CTX, MAX_CTX = 4, 2, 4, 32, 128
TEMPERATURE = 1.0  # pretrain_final_temp for AudioSet checkpoints


def _norm(c):
    return nn.BatchNorm2d(c, eps=0.001, momentum=0.01)


class DynamicConv(nn.Module):
    """k kernels mixed by a softmax attention computed from the context vector; one aggregated kernel per clip."""

    def __init__(self, cin, cout, ctx, kernel_size, stride=1, padding=0, groups=1, k=K_CONV):
        super().__init__()
        self.in_channels, self.out_channels, self.kernel_size = cin, cout, kernel_size
        self.stride, self.padding, self.groups, self.k = stride, padding, groups, k
        self.residuals = nn.Sequential(nn.Linear(ctx, k))
        self.weight = nn.Parameter(torch.zeros(1, 1, k, cout * (cin // groups) * kernel_size * kernel_size))
        self.quant_bits = None  # set by QuantDyMN: per-clip, per-output-channel quantization of the aggregate
        self.act_dims = (2, 3) if (groups > 1 and groups == cin) else ((1,) if kernel_size == 1 else (1, 2, 3))

    def aggregate(self, h_c):
        b = h_c.shape[0]
        att = F.softmax(self.residuals(h_c.view(b, -1)).view(b, 1, 1, -1) / TEMPERATURE, dim=-1)
        w = (att @ self.weight).transpose(1, 2).reshape(b, self.out_channels, self.in_channels // self.groups,
                                                        self.kernel_size, self.kernel_size)
        return w

    def forward(self, x, g):
        b, c, f, t = x.shape
        w = self.aggregate(g[0])
        if self.quant_bits is not None:
            codes, s = rtn_per_channel(w.reshape(b * self.out_channels, -1), self.quant_bits)
            w = (codes * s).view_as(w)
            x = fake_quant_sym(x, self.quant_bits, self.act_dims)
        w = w.view(b * self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size)
        y = F.conv2d(x.reshape(1, -1, f, t), w, None, self.stride, self.padding, 1, self.groups * b)
        return y.view(b, self.out_channels, y.shape[-2], y.shape[-1])


class DyReLUB(nn.Module):
    def __init__(self, channels, ctx, M=M_RELU):
        super().__init__()
        self.channels, self.M = channels, M
        self.coef_net = nn.Sequential(nn.Linear(ctx, 2 * M * channels))
        self.sigmoid = nn.Sigmoid()
        self.register_buffer("lambdas", torch.tensor([1.0] * M + [0.5] * M))
        self.register_buffer("init_v", torch.tensor([1.0] + [0.0] * (2 * M - 1)))

    def forward(self, x, g):
        b = x.shape[0]
        theta = 2 * self.sigmoid(self.coef_net(g[0].view(b, -1))) - 1
        coefs = theta.view(-1, self.channels, 1, 1, 2 * self.M) * self.lambdas + self.init_v
        xm = x.unsqueeze(-1) * coefs[..., :self.M] + coefs[..., self.M:]
        return torch.maximum(xm[..., 0], xm[..., 1]) if self.M == 2 else xm.max(-1)[0]


class CoordAtt(nn.Module):
    def forward(self, x, g):
        return x * g[1].sigmoid() * g[2].sigmoid()


class DynamicWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, g):
        return self.module(x)


class ContextGen(nn.Module):
    def __init__(self, ctx, cin, cexp, stride=1):
        super().__init__()
        self.joint_conv = nn.Conv2d(cin, ctx, 1, bias=False)
        self.joint_norm = _norm(ctx)
        self.joint_act = nn.Hardswish()
        self.conv_f = nn.Conv2d(ctx, cexp, 1)
        self.conv_t = nn.Conv2d(ctx, cexp, 1)
        if stride > 1:
            self.pool_f = nn.AvgPool2d((3, 1), (stride, 1), (1, 0))
            self.pool_t = nn.AvgPool2d((1, 3), (1, stride), (0, 1))
        else:
            self.pool_f, self.pool_t = nn.Sequential(), nn.Sequential()

    def forward(self, x):
        cf = F.adaptive_avg_pool2d(x, (None, 1))
        ct = F.adaptive_avg_pool2d(x, (1, None)).permute(0, 1, 3, 2)
        f, t = cf.shape[2], ct.shape[2]
        g = self.joint_act(self.joint_norm(self.joint_conv(torch.cat([cf, ct], 2))))
        h_cf, h_ct = torch.split(g, [f, t], 2)
        h_ct = h_ct.permute(0, 1, 3, 2)
        h_c = g.mean(2, keepdim=True)
        return h_c, self.conv_f(self.pool_f(h_cf)), self.conv_t(self.pool_t(h_ct))


class DY_Block(nn.Module):
    def __init__(self, cin, k, cexp, cout, hs, stride, width):
        super().__init__()
        self.use_res_connect = stride == 1 and cin == cout
        ctx = int(np.clip(make_divisible(cexp // CONTEXT_RATIO, 8), make_divisible(MIN_CTX * width, 8),
                          make_divisible(MAX_CTX * width, 8)))
        self.context_dim = ctx
        act = nn.Hardswish if hs else nn.ReLU
        if cexp != cin:
            self.exp_conv = DynamicConv(cin, cexp, ctx, 1)
            self.exp_norm = _norm(cexp)
            self.exp_act = DynamicWrapper(act())
        else:
            self.exp_conv, self.exp_norm, self.exp_act = DynamicWrapper(nn.Identity()), nn.Identity(), \
                DynamicWrapper(nn.Identity())
        self.depth_conv = DynamicConv(cexp, cexp, ctx, k, stride, (k - 1) // 2, groups=cexp)
        self.depth_norm = _norm(cexp)
        self.depth_act = DyReLUB(cexp, ctx)
        self.ca = CoordAtt()
        self.proj_conv = DynamicConv(cexp, cout, ctx, 1)
        self.proj_norm = _norm(cout)
        self.context_gen = ContextGen(ctx, cin, cexp, stride)

    def forward(self, x):
        inp = x
        g = self.context_gen(x)
        x = self.exp_act(self.exp_norm(self.exp_conv(x, g)), g)
        x = self.ca(self.depth_act(self.depth_norm(self.depth_conv(x, g)), g), g)
        x = self.proj_norm(self.proj_conv(x, g))
        return x + inp if self.use_res_connect else x


class DyMN(nn.Module):
    """EfficientAT's DyMN at a width multiplier: in_c, 15 dynamic blocks, out_c and the 'mlp' AudioSet classifier.
    forward() returns the pooled embedding [B, C]."""

    def __init__(self, width: float, num_classes: int = 527, dropout: float = 0.2):
        super().__init__()
        adj = lambda c: make_divisible(c * width, 8)  # noqa: E731
        cfg = [(adj(a), k, adj(e), adj(o), hs, s) for a, k, e, o, _se, hs, s in BNECK]
        self.in_c = Conv2dNormActivation(1, cfg[0][0], kernel_size=3, stride=2, norm_layer=_norm,
                                         activation_layer=nn.Hardswish)
        self.layers = nn.ModuleList([DY_Block(*c, width) for c in cfg])
        self.emb_dim = 6 * cfg[-1][3]
        self.out_c = Conv2dNormActivation(cfg[-1][3], self.emb_dim, kernel_size=1, norm_layer=_norm,
                                          activation_layer=nn.Hardswish)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(self.emb_dim, adj(1280)),
                                        nn.Hardswish(), nn.Dropout(dropout), nn.Linear(adj(1280), num_classes))

    def features(self, x):
        x = self.in_c(x)
        for layer in self.layers:
            x = layer(x)
        return self.out_c(x)

    def forward(self, x):
        return F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)

    def audioset_logits(self, x):
        return self.classifier(self.features(x))


def build_dymn(name: str) -> DyMN:
    return DyMN(WIDTHS[name]).eval()


def load_dymn(name: str, path) -> DyMN:
    m = build_dymn(name)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    m.load_state_dict(sd, strict=True)
    return m.eval()


# ── W8A8 ───────────────────────────────────────────────────────────────────

def _feature_modules(m: DyMN):
    for name, mod in m.named_modules():
        if name.startswith("classifier"):
            continue
        yield name, mod


class QuantDyMN(nn.Module):
    """A copy of `base` whose static Conv2d and Linear layers outside the classifier have per-output-channel
    weights at weight_bits and inputs fake-quantized at act_bits (effnet_quant's scale axes, per row for Linear),
    and whose DynamicConv layers quantize their aggregated kernel per clip and output channel and their input the
    same way. Weights are quantized on the CPU, then the copy moves to `device`."""

    def __init__(self, base: DyMN, weight_bits: int = 8, act_bits: int = 8, device="cpu"):
        super().__init__()
        self.weight_bits, self.act_bits = weight_bits, act_bits
        m = copy.deepcopy(base).cpu().eval()
        self.n_conv = self.n_linear = self.n_dynamic = 0
        with torch.no_grad():
            for _, mod in _feature_modules(m):
                if isinstance(mod, DynamicConv):
                    mod.quant_bits = weight_bits if weight_bits == act_bits else None
                    if mod.quant_bits is None:
                        raise ValueError("DynamicConv quantization uses one bit width for kernel and input")
                    self.n_dynamic += 1
                    continue
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
    def forward(self, x):
        return self.net(x)


def build_backbone(config: str, base: DyMN, device="cpu") -> nn.Module:
    if config == "fp32":
        return copy.deepcopy(base).to(device).eval()
    if config == "w8a8":
        return QuantDyMN(base, 8, 8, device).eval()
    raise ValueError(f"bad configuration name '{config}'")


# ── costs ──────────────────────────────────────────────────────────────────

def feature_macs(model: DyMN, frames: int) -> dict:
    """MACs and output elements of the feature extractor on one [1, 1, 128, frames] input: static Conv2d and
    Linear from hooks ('conv', 'depthwise', 'linear'), DynamicConv from hooks on the module ('dyn_conv',
    'dyn_depthwise'), and the kernel aggregation ('aggregate', k x kernel elements per DynamicConv, one clip)."""
    parts = {k: [0, 0] for k in ("conv", "depthwise", "linear", "dyn_conv", "dyn_depthwise", "aggregate")}
    hooks = []

    def hook(mod, inp, out):
        if isinstance(mod, DynamicConv):
            n_out = out.shape[1] * out.shape[2] * out.shape[3]
            key = "dyn_depthwise" if mod.groups > 1 else "dyn_conv"
            parts[key][0] += n_out * (mod.in_channels // mod.groups) * mod.kernel_size ** 2
            parts[key][1] += n_out
            parts["aggregate"][0] += mod.k * mod.weight.shape[-1]
            parts["aggregate"][1] += mod.weight.shape[-1]
        elif isinstance(mod, nn.Conv2d):
            kh, kw = mod.kernel_size
            n_out = out.shape[1] * out.shape[2] * out.shape[3]
            key = "depthwise" if mod.groups > 1 else "conv"
            parts[key][0] += n_out * (mod.in_channels // mod.groups) * kh * kw
            parts[key][1] += n_out
        else:
            parts["linear"][0] += mod.in_features * mod.out_features
            parts["linear"][1] += mod.out_features

    for _, mm in _feature_modules(model):
        if isinstance(mm, (nn.Conv2d, nn.Linear, DynamicConv)):
            hooks.append(mm.register_forward_hook(hook))
    try:
        with torch.no_grad():
            p = next(model.parameters())
            model.features(torch.zeros(1, 1, 128, frames, device=p.device, dtype=p.dtype))
    finally:
        for h in hooks:
            h.remove()
    quant = {k: v for k, v in parts.items() if k != "aggregate"}
    return {"frames": frames, "macs": float(sum(v[0] for v in quant.values())),
            "out_elems": float(sum(v[1] for v in quant.values())),
            "aggregate_macs": float(parts["aggregate"][0]),
            "parts_macs": {k: float(v[0]) for k, v in parts.items()}}


def backbone_bitops(macs: dict, config: str, rescale_price: float = 0.0) -> float:
    """Quantizable MACs at the configuration's bits plus the kernel aggregation at 32 x 32 in both."""
    agg = macs.get("aggregate_macs", 0.0) * 32 * 32
    if config == "fp32":
        return macs["macs"] * 32 * 32 + agg
    if config == "w8a8":
        return macs["macs"] * 8 * 8 + macs["out_elems"] * rescale_price + agg
    raise ValueError(config)


def weight_bits_stored(model: DyMN, config: str) -> float:
    """Bits to store the feature extractor: static Conv2d and Linear weights and the k kernels of every
    DynamicConv at the configuration's bits plus one fp32 scale per output channel (per kernel), everything else
    fp32. The classifier is left out."""
    bits = 0.0
    for _, mod in _feature_modules(model):
        for name, p in mod.named_parameters(recurse=False):
            if config == "w8a8" and name == "weight" and isinstance(mod, (nn.Conv2d, nn.Linear)):
                bits += 8 * p.numel() + 32 * p.shape[0]
            elif config == "w8a8" and name == "weight" and isinstance(mod, DynamicConv):
                bits += 8 * p.numel() + 32 * mod.k * mod.out_channels
            else:
                bits += 32 * p.numel()
    return bits
