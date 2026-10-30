"""Inference cost per clip: multiply-accumulates (MACs), bit operations (MACs
x weight bits x activation bits, ternary counted as 2 bits) and weight bytes
at deployed precision.

layer_cost, cost and moe_cost price the heads and MoEs built from
QWave.qmoe_layers on one embedding (one clip). transformer_cost prices a
transformer backbone such as the AST of data/audio_embeddings at a chosen
precision, so a method that changes the backbone's bits can be put on the
same axis.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from torch import nn


def layer_cost(m) -> np.ndarray | None:
    """[MACs, bit operations] of one linear layer per sample, else None.

    BitLinear "bitnet": ternary weights and activations, 2 bits each; k < 16:
    k-bit weights, 8-bit activations; 16: fp16 weights, fp32 activations;
    32: fp32 both. nn.Linear: fp32 both. Dynamic int8 Linear (PTQ): 8-bit
    weights and 8-bit dynamically quantized activations."""
    from torch.ao.nn.quantized.dynamic import Linear as DynamicInt8Linear
    from QWave.qmoe_layers import BitLinear

    if isinstance(m, BitLinear):
        nb = m.num_bits
        w, a = (2, 2) if nb == "bitnet" else (nb, 8 if nb < 16 else 32)
    elif isinstance(m, nn.Linear):
        w, a = 32, 32
    elif isinstance(m, DynamicInt8Linear):
        w, a = 8, 8
    else:
        return None
    macs = m.in_features * m.out_features
    return np.array([macs, macs * w * a], dtype=float)


def cost(module) -> np.ndarray:
    """[MACs, bit operations] per sample, summed over every linear layer."""
    return sum((c for c in map(layer_cost, module.modules()) if c is not None), np.zeros(2))


def moe_cost(model, shares) -> np.ndarray:
    """Expected per-sample cost of an MoE: every router pass (M MC passes plus
    the logits pass for a Bayesian router), the shared trunk if any, and each
    expert weighted by its routing share."""
    passes = model.router.mc_samples + 1 if model.use_curiosity else 1
    c = passes * cost(model.router)
    if model.shared is not None:
        return c + cost(model.shared.trunk) + sum(s * cost(h) for s, h in zip(shares, model.shared.heads))
    return c + sum(s * cost(e) for s, e in zip(shares, model.experts))


def weight_bytes(model) -> float:
    """Bytes to store `model` at its deployed precision (every parameter, see
    QWave.memory.storage_bits for the per-layer rules)."""
    from QWave.memory import storage_bits

    return storage_bits(model) / 8


# ── transformer backbones ──────────────────────────────────────────────────

def _bits(b) -> int:
    return 2 if b == "bitnet" else int(b)


def _quantized_tensor_bits(n_weights: int, rows: int, b) -> int:
    """Storage of one weight tensor, as QWave.memory.storage_bits counts a
    BitLinear: below 16 bits (ternary included) the codes plus one fp32
    scale per output row; 16 and 32 bits plain."""
    nb = _bits(b)
    return nb * n_weights + (32 * rows if nb < 16 else 0)


def ast_token_count(config) -> tuple[int, int]:
    """(patches, tokens) of a Hugging Face ASTConfig: the patch grid of
    ASTEmbeddings.get_shape, plus the [CLS] and [DIST] tokens. The default
    config (128 mel bins, 1024 frames, 16x16 patches, stride 10) gives
    12 x 101 = 1212 patches and 1214 tokens."""
    f = (config.num_mel_bins - config.patch_size) // config.frequency_stride + 1
    t = (config.max_length - config.patch_size) // config.time_stride + 1
    return f * t, f * t + 2


def transformer_cost(model, tokens: int, patches: int, weight_bits, act_bits: int, patch_bits) -> dict:
    """Per-clip MACs, bit operations and weight bytes of a transformer encoder
    such as transformers' ASTModel.

    linear       every nn.Linear runs once per token: in x out x tokens MACs,
                 at weight_bits (one int for every layer, or a mapping from
                 module name to bits) times act_bits. Pass the encoder, not a
                 classification wrapper, whose head runs once per clip.
    patch_embed  every nn.Conv2d (the patch embedding) at its own precision,
                 patch_bits = (weight bits, activation bits): out channels x
                 in channels / groups x kernel area MACs per patch.
    attention    the two activation-by-activation products of each
                 self-attention module (scores Q K^T and the weighted sum with
                 V), 2 x tokens^2 x all_head_size MACs at act_bits x act_bits.
                 Recognized by the all_head_size attribute of transformers'
                 self-attention modules.
    total        the sum of the three.

    weight_bytes stores Linear and Conv2d weights at their bit-width (below
    16 bits plus one fp32 scale per output row or channel) and every other
    parameter (biases, LayerNorms, position embeddings, special tokens) in
    fp32. Bit-widths are ints, or "bitnet" for ternary (2 bits).
    """
    pw, pa = patch_bits
    a = _bits(act_bits)
    parts = {k: np.zeros(2) for k in ("linear", "patch_embed", "attention")}
    store = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if isinstance(weight_bits, Mapping):
                if name not in weight_bits:
                    raise KeyError(f"weight_bits has no entry for Linear '{name}'")
                wb = weight_bits[name]
            else:
                wb = weight_bits
            macs = m.in_features * m.out_features * tokens
            parts["linear"] += [macs, macs * _bits(wb) * a]
            store += _quantized_tensor_bits(m.weight.numel(), m.out_features, wb)
            store += 32 * (m.bias.numel() if m.bias is not None else 0)
        elif isinstance(m, nn.Conv2d):
            kh, kw = m.kernel_size
            macs = m.out_channels * (m.in_channels // m.groups) * kh * kw * patches
            parts["patch_embed"] += [macs, macs * _bits(pw) * _bits(pa)]
            store += _quantized_tensor_bits(m.weight.numel(), m.out_channels, pw)
            store += 32 * (m.bias.numel() if m.bias is not None else 0)
        else:
            if hasattr(m, "all_head_size"):
                macs = 2 * tokens * tokens * m.all_head_size
                parts["attention"] += [macs, macs * a * a]
            store += 32 * sum(p.numel() for p in m.parameters(recurse=False))
    out = {k: {"macs": float(v[0]), "bitops": float(v[1])} for k, v in parts.items()}
    total = sum(parts.values())
    out["total"] = {"macs": float(total[0]), "bitops": float(total[1])}
    out["weight_bytes"] = store / 8
    return out


# ── T19: AST bit-ops with per-block activation bits ────────────────────────

AST_SHAPE = dict(L=12, D=768, FF=3072, patch=16)


def ast_tokens(max_length: int) -> tuple[int, int]:
    """(patches, tokens) of the AST at max_length frames (128 mel bins,
    16 x 16 patches, stride 10): 12 x ((max_length - 16) // 10 + 1) patches
    plus the [CLS] and [DIST] tokens."""
    nt = (max_length - 16) // 10 + 1
    return 12 * nt, 12 * nt + 2


def ast_t19_bitops(max_length: int, wbits, abits, p_bits: int, group: int | None = 128, asym: bool = True,
                   price: float = 32 * 32) -> dict:
    """Per-clip bit-ops of the AST encoder (ANALYSIS_PLAN_T19.md B4).

    wbits, abits   an int, or one int per block (12); wbits 32 = FP32
    Linear         MACs x weight bits x activation bits
    Q K^T          tokens^2 x D MACs per block at a x a
    P V            tokens^2 x D MACs per block at p_bits x a
    patch conv     FP32, 32 x 32
    rescale        `price` per token, output row and weight group (per
                   channel: one group per row); 0 for FP32 weights
    zero point     `price` per token and output row of every Linear whose
                   input is asymmetric (asym)
    Returns the parts and "total" (priced) and "unpriced" (no rescale and
    zero-point terms)."""
    L, D, FF = AST_SHAPE["L"], AST_SHAPE["D"], AST_SHAPE["FF"]
    patches, T = ast_tokens(max_length)
    wb = [wbits] * L if isinstance(wbits, int) else list(wbits)
    ab = [abits] * L if isinstance(abits, int) else list(abits)
    blocks = [(3 * D, D), (D, D), (FF, D), (D, FF)]
    lin = sum(o * i for o, i in blocks) * T
    att = T * T * D
    out = {"linear": 0.0, "qk": 0.0, "pv": 0.0, "rescale": 0.0, "zero_point": 0.0}
    for w, a in zip(wb, ab):
        out["linear"] += lin * w * a
        out["qk"] += att * a * a
        out["pv"] += att * p_bits * a
        if w < 32:
            out["rescale"] += T * sum(o * (i // group if group else 1) for o, i in blocks) * price
            if asym:
                out["zero_point"] += T * sum(o for o, _ in blocks) * price
    out["patch"] = float(D * AST_SHAPE["patch"] ** 2 * patches * 32 * 32)
    out["unpriced"] = out["linear"] + out["qk"] + out["pv"] + out["patch"]
    out["total"] = out["unpriced"] + out["rescale"] + out["zero_point"]
    out.update(tokens=T, patches=patches)
    return {k: float(v) for k, v in out.items()}
