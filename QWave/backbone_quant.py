"""Post-training quantization of an Audio Spectrogram Transformer (AST)
encoder, simulated in float (fake quantization), with per-clip costs.

Weights: the 72 encoder nn.Linear layers (query, key, value, attention
output, intermediate and output dense of the 12 layers). Symmetric, round to
nearest with torch.round, codes held as integer-valued floats. torch.round
rounds exact .5 ties to even on CPU but away from zero on MPS (torch 2.7).
Direct RTN almost never meets an exact tie; the nested slice c_8 / 2^(8-k)
meets one whenever c_8 is an odd multiple of 2^(7-k), so nested children
depend on the device. The Stage 2 embeddings were made on MPS (ties away from
zero); the CPU unit tests check ties to even.

  per channel  one scale per output row: s = max|W_row| / qmax with
               qmax = 2^(b-1) - 1, codes clamp(round(W / s), -qmax, qmax).
               b = 2 gives ternary codes {-1, 0, 1}.
  group        one scale per output row and run of `group` consecutive input
               columns, same rule inside each run (the input width must be a
               multiple of `group`).
  nested       an 8-bit parent (per channel or group, qmax 127) and a k-bit
               child sliced from the parent's top bits with rounding, as in
               Matryoshka Quantization and any-precision storage:
                   c_k = clamp(round(c_8 / 2^(8-k)), -2^(k-1), 2^(k-1) - 1)
                   W_k = c_k * s_8 * 2^(8-k)
               The child reuses the parent's scales, so one stored 8-bit
               parent serves every child. The child range is the full two's
               complement range of k bits (2^k levels), one more level than
               direct RTN at k bits. The child is a rounded slice, not the
               literal top k bit-planes: rounding makes c_k depend on parent
               bit 8-k-1 as well (the carry), so a bit-plane reader needs
               k + 1 planes for the cheap pass.

Activations, in every configuration except the FP32 reference
(act_bits=None): 8-bit dynamic fake quantization of every input of an
integer product.
  Linear inputs   per token (row), symmetric, s = max|x_token| / 127. The
                  query, key and value projections share one quantized input.
  Q, K of QK^T    per token and head, symmetric; the 1/sqrt(head dim)
                  scale multiplies the scores inside the softmax.
  P of P V        the softmax probabilities, per query row, unsigned at
                  p_bits (default 16), s = max_j P_ij / (2^p_bits - 1).
                  Computed without forming P: with e = exp(S - max_j S),
                  max_j P = 1 / sum_j e, so the quantized row is
                  round(qmax e) / (qmax sum_j e). P is the one input kept
                  above 8 bits: AST's attention rows over 1214 tokens are
                  diffuse, and at 8 bits every probability below
                  max_j P / 510 rounds to zero (about 5% of each row's mass on
                  ESC-50 clips), which alone moves the W8 embedding about ten
                  times further from FP32 than 8-bit weights do. 12 bits
                  already recover it; 16 is the integer width hardware has.
  V of P V        per head channel over the tokens, symmetric: the scale has
                  to be constant along the summed axis (tokens) for an integer
                  kernel to factor it out, so V cannot take per-token scales.
Kept FP32: the patch-embedding convolution, the [CLS] and [DIST] tokens,
position embeddings, every LayerNorm, every bias, softmax (exponent and
normalizer), GELU, residual additions, the final LayerNorm and the pooling.

Costs per clip: bit operations from QWave.costs.transformer_cost (Linear
layers at weight bits x activation bits, Q K^T at activation bits squared,
P V at p_bits x activation bits, the patch convolution at 32 x 32), weight storage
(codes, one fp32 scale per output row or per group, every other parameter in
fp32) and the weight bits read for one clip (the same count: every weight is
read once). cascade_cost combines a cheap and an expensive configuration.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

PARENT_BITS = 8
P_BITS = 16


# ── weight quantizers ──────────────────────────────────────────────────────

def _grouped(w: torch.Tensor, group: int | None) -> torch.Tensor:
    """[out, in] -> [out, n_groups, group] (a view); per channel is one group."""
    out_f, in_f = w.shape
    if group is None:
        return w.reshape(out_f, 1, in_f)
    if in_f % group:
        raise ValueError(f"input width {in_f} is not a multiple of group size {group}")
    return w.reshape(out_f, in_f // group, group)


def rtn_codes(w: torch.Tensor, bits: int, group: int | None = None):
    """Symmetric round-to-nearest codes and scales of a 2-D weight.

    Returns (codes [out, n_groups, group], scales [out, n_groups, 1]); per
    channel has n_groups = 1. W is approximated by codes * scales."""
    if bits < 2:
        raise ValueError("bits must be at least 2")
    qmax = 2 ** (bits - 1) - 1
    wg = _grouped(w, group)
    s = wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    return torch.clamp(torch.round(wg / s), -qmax, qmax), s


def nested_codes(parent: torch.Tensor, k: int, parent_bits: int = PARENT_BITS) -> torch.Tensor:
    """k-bit child codes of parent codes: clamp(round(c / 2^(pb-k)), -2^(k-1),
    2^(k-1) - 1). Dequantize with the parent scale times 2^(pb-k)."""
    if not 2 <= k <= parent_bits:
        raise ValueError(f"child bits {k} outside [2, {parent_bits}]")
    if k == parent_bits:
        return parent.clone()
    step = 2 ** (parent_bits - k)
    return torch.clamp(torch.round(parent / step), -(2 ** (k - 1)), 2 ** (k - 1) - 1)


@dataclass(frozen=True)
class WeightSpec:
    """How the encoder Linear weights are stored and computed.

    bits         32 = FP32 weights, else the bits the product runs at
    group        None = one scale per output channel, else the group size
    nested       True: a child of the PARENT_BITS parent with the same
                 granularity, sliced from its top bits (bits < PARENT_BITS)
    """
    bits: int
    group: int | None = None
    nested: bool = False

    def __post_init__(self):
        if self.bits == 32 and (self.group is not None or self.nested):
            raise ValueError("FP32 weights take no group or nesting")
        if self.nested and self.bits >= PARENT_BITS:
            raise ValueError("a nested child has fewer bits than its parent")
        if self.bits != 32 and not 2 <= self.bits <= 8:
            raise ValueError("bits must be 32 or in [2, 8]")

    @property
    def name(self) -> str:
        if self.bits == 32:
            return "fp32"
        return f"{'n' if self.nested else 'w'}{self.bits}" + (f"g{self.group}" if self.group else "")

    @property
    def parent(self) -> "WeightSpec | None":
        """The stored 8-bit parent of a nested child, else None."""
        return WeightSpec(PARENT_BITS, self.group) if self.nested else None


def parse_config(name: str) -> tuple[WeightSpec, int | None]:
    """(WeightSpec, activation bits) of a configuration name: 'fp32' is the
    FP32 reference (no activation quantization); 'w<b>' direct RTN and
    'n<b>' a nested child of the 8-bit parent, either with 'g<G>' for group
    size G; every name but 'fp32' runs with 8-bit activations."""
    m = re.fullmatch(r"(fp32)|([wn])(\d)(?:g(\d+))?", name)
    if not m:
        raise ValueError(f"bad configuration name '{name}'")
    if m.group(1):
        return WeightSpec(32), None
    spec = WeightSpec(int(m.group(3)), int(m.group(4)) if m.group(4) else None, nested=m.group(2) == "n")
    return spec, 8


def quantize_weight(w: torch.Tensor, spec: WeightSpec) -> torch.Tensor:
    """Dequantized weight of `spec` (a float tensor shaped like w)."""
    if spec.bits == 32:
        return w.clone()
    if spec.nested:
        codes, s = rtn_codes(w, PARENT_BITS, spec.group)
        codes, s = nested_codes(codes, spec.bits), s * 2 ** (PARENT_BITS - spec.bits)
    else:
        codes, s = rtn_codes(w, spec.bits, spec.group)
    return (codes * s).reshape(w.shape)


# ── activation fake quantization ───────────────────────────────────────────
#
# Reference implementations in torch ops, and on Apple GPUs (MPS) fused Metal
# kernels with the same arithmetic: one threadgroup per row reduces the row
# and writes the quantized row in place. They agree with the torch ops up to
# float rounding (a division or exponent one ulp apart can move a value that
# sits on a rounding boundary by one step: a few elements in ten million).

USE_METAL = True  # set False to force the torch ops on MPS (tests compare the two)

_METAL_SRC = r"""
#include <metal_stdlib>
using namespace metal;

// symmetric per-row fake quantization in place; row r starts at
// (r / inner) * outer + (r % inner) * cols
kernel void fq_rows(device float* x [[buffer(0)]],
                    constant uint& cols [[buffer(1)]],
                    constant float& qmax [[buffer(2)]],
                    constant uint& inner [[buffer(3)]],
                    constant uint& outer [[buffer(4)]],
                    uint row [[threadgroup_position_in_grid]],
                    uint tid [[thread_position_in_threadgroup]],
                    uint nt [[threads_per_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[32];
    device float* r = x + (ulong)(row / inner) * outer + (ulong)(row % inner) * cols;
    float m = 0.0f;
    for (uint j = tid; j < cols; j += nt) m = max(m, fabs(r[j]));
    m = simd_max(m);
    if (lane == 0) red[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = lane < (nt + 31) / 32 ? red[lane] : 0.0f;
        v = simd_max(v);
        if (lane == 0) red[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = max(red[0], 1e-30f) / qmax;
    for (uint j = tid; j < cols; j += nt) r[j] = rint(precise::divide(r[j], s)) * s;
}

// attention scores in place: e = exp((s - max) * scale), row <- rint(qmax e),
// z[row] = sum of e
kernel void softmax_q_rows(device float* x [[buffer(0)]],
                           device float* z [[buffer(1)]],
                           constant uint& cols [[buffer(2)]],
                           constant float& qmax [[buffer(3)]],
                           constant float& scale [[buffer(4)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint nt [[threads_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[32];
    device float* r = x + (ulong)row * cols;
    float m = -INFINITY;
    for (uint j = tid; j < cols; j += nt) m = max(m, r[j]);
    m = simd_max(m);
    if (lane == 0) red[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = lane < (nt + 31) / 32 ? red[lane] : -INFINITY;
        v = simd_max(v);
        if (lane == 0) red[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mm = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    for (uint j = tid; j < cols; j += nt) {
        float e = precise::exp((r[j] - mm) * scale);
        acc += e;
        r[j] = rint(qmax * e);
    }
    acc = simd_sum(acc);
    if (lane == 0) red[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = lane < (nt + 31) / 32 ? red[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) z[row] = v;
    }
}
"""
_METAL = {}


def _metal(x: torch.Tensor):
    """The compiled kernels when x lives on MPS and USE_METAL is set, else None."""
    if not (USE_METAL and x.device.type == "mps" and x.dtype == torch.float32):
        return None
    if "lib" not in _METAL:
        _METAL["lib"] = torch.mps.compile_shader(_METAL_SRC)
    return _METAL["lib"]


def _threads(cols: int) -> int:
    return min(256, max(32, -(-cols // 32) * 32))


def fake_quant_sym(x: torch.Tensor, bits: int, dim: int = -1) -> torch.Tensor:
    """Symmetric dynamic fake quantization with one scale per slice along
    `dim` (dim=-1: per token): s = max|x| / (2^(bits-1) - 1). An all-zero
    slice stays zero. Torch ops; returns a new tensor."""
    qmax = 2 ** (bits - 1) - 1
    lo, hi = torch.aminmax(x, dim=dim, keepdim=True)
    s = torch.maximum(hi, -lo).clamp_min(1e-30) / qmax
    return (x / s).round_().mul_(s)


def fake_quant_rows_(x: torch.Tensor, bits: int) -> torch.Tensor:
    """fake_quant_sym over the last dimension, in place on a contiguous x."""
    lib = _metal(x)
    if lib is None:
        return x.copy_(fake_quant_sym(x, bits))
    if not x.is_contiguous():
        raise ValueError("fake_quant_rows_ needs a contiguous tensor")
    cols = x.shape[-1]
    nt = _threads(cols)
    lib.fq_rows(x, cols, float(2 ** (bits - 1) - 1), 1, cols, threads=x.numel() // cols * nt, group_size=nt)
    return x


def fake_quant_qk_(qkv: torch.Tensor, bits: int, heads: int, head_dim: int) -> torch.Tensor:
    """In place on a contiguous [B, T, 3 * heads * head_dim] query-key-value
    output: the query and key parts quantized per token and head (rows of
    head_dim), the value part untouched."""
    lib = _metal(qkv)
    B, T, _ = qkv.shape
    if lib is None:
        qk = qkv.view(B, T, 3, heads, head_dim)[:, :, :2]
        qk.copy_(fake_quant_sym(qk, bits))
        return qkv
    if not qkv.is_contiguous():
        raise ValueError("fake_quant_qk_ needs a contiguous tensor")
    nt = _threads(head_dim)
    lib.fq_rows(qkv, head_dim, float(2 ** (bits - 1) - 1), 2 * heads, 3 * heads * head_dim,
                threads=B * T * 2 * heads * nt, group_size=nt)
    return qkv


def quant_softmax_av(scores: torch.Tensor, v: torch.Tensor, bits: int = 8, scale: float = 1.0) -> torch.Tensor:
    """softmax(scale * scores) @ v with the probabilities fake-quantized per
    row, unsigned at `bits` (scale max_j P_ij / (2^bits - 1)), computed as
    round(qmax e) @ v / (qmax sum_j e) with e = exp(scale (s - max_j s)).
    `scores` (contiguous) is overwritten."""
    qmax = 2 ** bits - 1
    lib = _metal(scores)
    if lib is None:
        scores.sub_(scores.amax(dim=-1, keepdim=True)).mul_(scale).exp_()
        z = scores.sum(dim=-1, keepdim=True)
        scores.mul_(qmax).round_()
    else:
        if not scores.is_contiguous():
            raise ValueError("quant_softmax_av needs contiguous scores")
        cols = scores.shape[-1]
        z = torch.empty(scores.shape[:-1] + (1,), device=scores.device)
        nt = _threads(cols)
        lib.softmax_q_rows(scores, z, cols, float(qmax), float(scale), threads=scores.numel() // cols * nt,
                           group_size=nt)
    return torch.matmul(scores, v).div_(z * qmax)


# ── the encoder forward ────────────────────────────────────────────────────

LINEAR_SLOTS = ("qkv", "attn_out", "fc1", "fc2")


def _layer_linears(layer):
    """The four (fused) weight matrices of an ASTLayer and their biases."""
    att = layer.attention.attention
    return {
        "qkv": (torch.cat([att.query.weight, att.key.weight, att.value.weight]),
                torch.cat([att.query.bias, att.key.bias, att.value.bias])),
        "attn_out": (layer.attention.output.dense.weight, layer.attention.output.dense.bias),
        "fc1": (layer.intermediate.dense.weight, layer.intermediate.dense.bias),
        "fc2": (layer.output.dense.weight, layer.output.dense.bias),
    }


class QuantAST(nn.Module):
    """pooler_output of a transformers ASTModel under a WeightSpec and an
    activation bit-width, with the query, key and value projections fused.

    Quantizing the fused [3D, D] query-key-value matrix per output row (or
    per row and input group) is the same as quantizing the three matrices
    separately. set_config re-quantizes from the FP32 master weights, so one
    instance serves every configuration. Build it from a model already on
    its device: the fused copies are made where the model's weights live."""

    def __init__(self, ast_model, p_bits: int = P_BITS):
        super().__init__()
        self.ast, self.p_bits = ast_model.eval(), p_bits
        cfg = ast_model.config
        self.heads, self.head_dim = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
        self.scaling = self.head_dim ** -0.5
        with torch.no_grad():
            lin = [_layer_linears(layer) for layer in ast_model.encoder.layer]
            self.master = [{k: w.detach().clone() for k, (w, _) in d.items()} for d in lin]
            self.bias = [{k: b.detach().clone() for k, (_, b) in d.items()} for d in lin]
        self.weights = [{k: w.clone() for k, w in m.items()} for m in self.master]
        self.spec, self.act_bits = WeightSpec(32), None

    @torch.no_grad()
    def set_config(self, spec: WeightSpec, act_bits: int | None) -> None:
        for m, w in zip(self.master, self.weights):
            for k in LINEAR_SLOTS:
                w[k].copy_(quantize_weight(m[k], spec))
        self.spec, self.act_bits = spec, act_bits

    def _aq_(self, x):
        return x if self.act_bits is None else fake_quant_rows_(x, self.act_bits)

    def _layer(self, h, layer, w, b):
        B, T, D = h.shape
        H, dh, a = self.heads, self.head_dim, self.act_bits
        qkv = F.linear(self._aq_(layer.layernorm_before(h)), w["qkv"], b["qkv"])
        if a is not None:
            fake_quant_qk_(qkv, a, H, dh)
        q, k, v = qkv.view(B, T, 3, H, dh).permute(2, 0, 3, 1, 4).unbind(0)
        scores = torch.matmul(q, k.transpose(-1, -2))
        if a is None:
            ctx = torch.matmul(torch.softmax(scores.mul_(self.scaling), dim=-1), v)
        else:
            ctx = quant_softmax_av(scores, fake_quant_sym(v, a, dim=-2), self.p_bits, self.scaling)
        ctx = ctx.transpose(1, 2).reshape(B, T, D).contiguous()
        h = h + F.linear(self._aq_(ctx), w["attn_out"], b["attn_out"])
        x = F.gelu(F.linear(self._aq_(layer.layernorm_after(h)), w["fc1"], b["fc1"]))
        return h + F.linear(self._aq_(x), w["fc2"], b["fc2"])

    @torch.no_grad()
    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        h = self.ast.embeddings(input_values)
        for layer, w, b in zip(self.ast.encoder.layer, self.weights, self.bias):
            h = self._layer(h, layer, w, b)
        h = self.ast.layernorm(h)
        return (h[:, 0] + h[:, 1]) / 2


# ── costs ──────────────────────────────────────────────────────────────────

def backbone_cost(ast_model, spec: WeightSpec, act_bits: int | None, p_bits: int = P_BITS) -> dict:
    """Per-clip bit-ops (transformer_cost; patch convolution at 32 x 32, and
    the P V half of the attention MACs re-priced at p_bits x act_bits) and
    weight bits stored and read for one clip. Group-wise scales add one fp32
    scale per output row and group beyond the per-row one transformer_cost
    counts."""
    from QWave.costs import ast_token_count, transformer_cost

    patches, tokens = ast_token_count(ast_model.config)
    a = 32 if act_bits is None else act_bits
    c = transformer_cost(ast_model, tokens, patches, spec.bits, a, (32, 32))
    if act_bits is not None:
        pv = c["attention"]["macs"] / 2  # Q K^T and P V have the same MAC count
        extra = pv * (p_bits - a) * a
        c["attention"]["bitops"] += extra
        c["total"]["bitops"] += extra
    bits = 8 * c["weight_bytes"]
    if spec.group is not None:
        for m in ast_model.encoder.modules():
            if isinstance(m, nn.Linear):
                bits += 32 * m.out_features * (m.in_features // spec.group - 1)
    linear_codes = sum(m.weight.numel() for m in ast_model.encoder.modules() if isinstance(m, nn.Linear))
    return {"config": spec.name, "weight_bits": spec.bits, "act_bits": a,
            "p_bits": 32 if act_bits is None else p_bits, "group": spec.group,
            "nested": spec.nested, "bitops": c["total"]["bitops"], "macs": c["total"]["macs"],
            "parts": {k: c[k] for k in ("linear", "patch_embed", "attention")},
            "storage_bits": float(bits), "bits_read": float(bits), "linear_weights": int(linear_codes)}


def cascade_cost(cheap: dict, expensive: dict, rate: float, head_cheap: float = 0.0,
                 head_expensive: float = 0.0) -> dict:
    """Expected per-clip cost of a two-stage cascade: every clip runs `cheap`
    (backbone_cost dicts) and its head; a fraction `rate` re-runs `expensive`
    and its head. Bit-ops add. Nested (cheap is a child of expensive, its
    8-bit parent): only the parent is stored, and an escalated clip reads the
    parent's remaining 8-k bit-planes of the Linear codes, so a clip reads at
    most the parent's bits. Otherwise (a nested child escalating anywhere
    else is then stored as its own k-bit codes) both models are stored and an
    escalated clip reads the whole expensive model."""
    bitops = cheap["bitops"] + head_cheap + rate * (expensive["bitops"] + head_expensive)
    nested = bool(cheap["nested"]) and expensive["weight_bits"] == PARENT_BITS \
        and not expensive["nested"] and cheap["group"] == expensive["group"]
    if nested:
        remaining = (PARENT_BITS - cheap["weight_bits"]) * cheap["linear_weights"]
        storage, read = expensive["storage_bits"], cheap["bits_read"] + rate * remaining
    else:
        storage = cheap["storage_bits"] + expensive["storage_bits"]
        read = cheap["bits_read"] + rate * expensive["bits_read"]
    return {"bitops": bitops, "storage_bits": storage, "bits_read": read, "nested": nested, "rate": rate}


# ── T19: low-bit activations (A4, A6, A8) with SmoothQuant and clipping ────
#
# ANALYSIS_PLAN_T19.md, Part B (B2, D3). A second forward mode of QuantAST,
# switched on by set_t19(); set_config() returns to the legacy mode above.
#   Linear inputs  per token, asymmetric: the token's values are clamped to
#                  the site's calibrated bounds [lo, hi], the range
#                  [min, max] of the clamped token is widened to contain 0,
#                  s = (max - min) / (2^b - 1), zero point round(-min / s),
#                  codes clamp(round(x / s) + zp, 0, 2^b - 1).
#   Q, K           per token and head, symmetric, after clamping to +-c.
#   V              per head channel over the tokens, symmetric, after +-c.
#   P              per query row, unsigned at p_bits (quant_softmax_av).
#   SmoothQuant    the four Linear inputs of a block are divided by s_j per
#                  input channel j and the weight columns multiplied by s_j
#                  before weight quantization (s from calibration).
#   Per-block activation bits (mixed precision): layer_bits, one per block.
#   Stochastic rounding (sr_seed): floor(x / s + u), u ~ U[0, 1), on every
#                  activation input (Linear inputs, Q, K, V, P).

LIN_SITES = ("qkv", "attn_out", "fc1", "fc2")
ATT_SITES = ("q", "k", "v")


def _round(x: torch.Tensor, gen) -> torch.Tensor:
    """Round to nearest (gen None) or, for any other gen, stochastic
    floor(x + u) with u from the global generator of x's device, in place."""
    if gen is None:
        return x.round_()
    return x.add_(torch.rand_like(x)).floor_()


def fake_quant_asym(x: torch.Tensor, bits: int, lo: float | None = None, hi: float | None = None,
                    gen=None) -> torch.Tensor:
    """Per-row (last dim) asymmetric dynamic fake quantization with the row's
    range widened to contain 0, after clamping x to [lo, hi] (None: no
    clamp). Returns a new tensor."""
    qmax = 2 ** bits - 1
    if lo is not None:
        x = x.clamp(lo, hi)
    mn, mx = torch.aminmax(x, dim=-1, keepdim=True)
    mn, mx = mn.clamp(max=0.0), mx.clamp(min=0.0)
    s = ((mx - mn) / qmax).clamp_min(1e-30)
    zp = torch.round(-mn / s)
    q = _round(x / s, gen).add_(zp).clamp_(0, qmax)
    return q.sub_(zp).mul_(s)


def fake_quant_sym_clip(x: torch.Tensor, bits: int, dim: int = -1, c: float | None = None, gen=None) -> torch.Tensor:
    """fake_quant_sym after clamping to +-c (None: no clamp), with optional
    stochastic rounding. Returns a new tensor."""
    if c is not None:
        x = x.clamp(-c, c)
    if gen is None:
        return fake_quant_sym(x, bits, dim)
    qmax = 2 ** (bits - 1) - 1
    lo, hi = torch.aminmax(x, dim=dim, keepdim=True)
    s = torch.maximum(hi, -lo).clamp_min(1e-30) / qmax
    return _round(x / s, gen).clamp_(-qmax, qmax).mul_(s)


def quant_softmax_av_sr(scores: torch.Tensor, v: torch.Tensor, bits: int, scale: float, gen) -> torch.Tensor:
    """quant_softmax_av with stochastic rounding of the probability codes."""
    qmax = 2 ** bits - 1
    scores = scores.sub(scores.amax(dim=-1, keepdim=True)).mul_(scale).exp_()
    z = scores.sum(dim=-1, keepdim=True)
    scores = _round(scores.mul_(qmax), gen).clamp_(0, qmax)
    return torch.matmul(scores, v).div_(z * qmax)


@dataclass
class T19Act:
    """Activation setting of the T19 mode.

    layer_bits   activation bits of each of the 12 blocks
    p_bits       bits of the attention probabilities
    smooth       {block: {site: s [in]}} SmoothQuant scales, or None
    clip         per activation bit-width, a list over blocks of {site:
                 (lo, hi)} for LIN_SITES and {site: c} for ATT_SITES:
                 clip[bits][block]; None, or a missing or None entry for a
                 width, means no clamp
    asym         Linear inputs asymmetric (True) or symmetric (legacy rule)
    sr_seed      stochastic rounding seed, None = round to nearest"""
    layer_bits: tuple
    p_bits: int = 8
    smooth: dict | None = None
    clip: dict | None = None
    asym: bool = True
    sr_seed: int | None = None


def smooth_scales(act_absmax: torch.Tensor, w: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """SmoothQuant s_j = max|X_j|^alpha / max_i |W_ij|^(1 - alpha) for a
    weight [out, in] and the calibration absmax per input channel [in]."""
    wmax = w.abs().amax(dim=0).clamp_min(1e-8)
    s = act_absmax.clamp_min(1e-8).pow(alpha) / wmax.pow(1 - alpha)
    return s.clamp_min(1e-5)


def _t19_set(self, spec: WeightSpec, act: T19Act) -> None:
    """Switch the QuantAST to the T19 mode: weights quantized from the master
    multiplied by the smoothing scales (input columns), activations by
    `act`."""
    with torch.no_grad():
        for i, (m, w) in enumerate(zip(self.master, self.weights)):
            for k in LINEAR_SLOTS:
                wm = m[k]
                if act.smooth is not None:
                    wm = wm * act.smooth[i][k].to(wm.device)[None, :]
                w[k].copy_(quantize_weight(wm, spec))
    if len(act.layer_bits) != len(self.weights):
        raise ValueError("layer_bits needs one entry per block")
    self.spec, self.act_bits, self.t19 = spec, None, act
    self._sm = None if act.smooth is None else [{k: act.smooth[i][k].to(self.weights[i][k].device)
                                                 for k in LINEAR_SLOTS} for i in range(len(self.weights))]
    # stochastic rounding draws from the device's global generator, reseeded
    # at every forward call with (sr_seed, call count): reproducible for the
    # same batches in the same order
    self._gen = None if act.sr_seed is None else True
    self._calls = 0


def _t19_clip(self, a, i):
    """Clip bounds of block i at activation width a, or None (no clamp)."""
    if self.t19.clip is None or self.t19.clip.get(a) is None:
        return None
    return self.t19.clip[a][i]


def _t19_lin(self, x, i, site):
    a = self.t19.layer_bits[i]
    if self._sm is not None:
        x = x / self._sm[i][site]
    cb = _t19_clip(self, a, i)
    cl = None if cb is None else cb[site]
    if self.t19.asym:
        lo, hi = (None, None) if cl is None else cl
        return fake_quant_asym(x, a, lo, hi, self._gen)
    c = None if cl is None else max(-cl[0], cl[1])
    if self._gen is None:
        # the legacy kernel (fused Metal on MPS): an 8-bit quantized AST is chaotic in the last bit (one rounding-boundary
        # difference in layer 0 moves the pooled embedding by about 4%), so the legacy setting must use the legacy kernel
        x = x.contiguous() if c is None else x.clamp(-c, c).contiguous()
        return fake_quant_rows_(x, a)  # in place: x is always a fresh tensor here
    return fake_quant_sym_clip(x, a, -1, c, self._gen)


def _t19_layer(self, h, layer, w, b, i):
    B, T, D = h.shape
    H, dh = self.heads, self.head_dim
    a = self.t19.layer_bits[i]
    cl = _t19_clip(self, a, i)
    qkv = F.linear(_t19_lin(self, layer.layernorm_before(h), i, "qkv"), w["qkv"], b["qkv"]).contiguous()
    if self._gen is None:
        # Q and K with the legacy kernel (per token and head, in place on the fused output), after the clamp
        if cl is not None:
            parts = qkv.view(B, T, 3, H * dh)
            parts[:, :, 0].clamp_(-cl["q"], cl["q"])
            parts[:, :, 1].clamp_(-cl["k"], cl["k"])
        fake_quant_qk_(qkv, a, H, dh)
        q, k, v = qkv.view(B, T, 3, H, dh).permute(2, 0, 3, 1, 4).unbind(0)
    else:
        q, k, v = qkv.view(B, T, 3, H, dh).permute(2, 0, 3, 1, 4).unbind(0)
        q = fake_quant_sym_clip(q, a, -1, None if cl is None else cl["q"], self._gen)
        k = fake_quant_sym_clip(k, a, -1, None if cl is None else cl["k"], self._gen)
    v = fake_quant_sym_clip(v, a, -2, None if cl is None else cl["v"], self._gen)
    scores = torch.matmul(q, k.transpose(-1, -2))
    if self._gen is None:
        ctx = quant_softmax_av(scores, v, self.t19.p_bits, self.scaling)
    else:
        ctx = quant_softmax_av_sr(scores, v, self.t19.p_bits, self.scaling, self._gen)
    ctx = ctx.transpose(1, 2).reshape(B, T, D).contiguous()
    h = h + F.linear(_t19_lin(self, ctx, i, "attn_out"), w["attn_out"], b["attn_out"])
    x = F.gelu(F.linear(_t19_lin(self, layer.layernorm_after(h), i, "fc1"), w["fc1"], b["fc1"]))
    return h + F.linear(_t19_lin(self, x, i, "fc2"), w["fc2"], b["fc2"])


@torch.no_grad()
def _t19_forward(self, input_values: torch.Tensor) -> torch.Tensor:
    if getattr(self, "t19", None) is None:
        return QuantAST._legacy_forward(self, input_values)
    if self._gen is not None:
        torch.manual_seed(1_000_003 * int(self.t19.sr_seed) + self._calls)
        self._calls += 1
    h = self.ast.embeddings(input_values)
    for i, (layer, w, b) in enumerate(zip(self.ast.encoder.layer, self.weights, self.bias)):
        h = _t19_layer(self, h, layer, w, b, i)
    h = self.ast.layernorm(h)
    return (h[:, 0] + h[:, 1]) / 2


_legacy_set_config = QuantAST.set_config


def _set_config_legacy(self, spec: WeightSpec, act_bits: int | None) -> None:
    self.t19 = None
    _legacy_set_config(self, spec, act_bits)


QuantAST._legacy_forward = QuantAST.forward
QuantAST.forward = _t19_forward
QuantAST.set_config = _set_config_legacy
QuantAST.set_t19 = _t19_set


@torch.no_grad()
def collect_calibration(qast: QuantAST, batches, tokens_per_clip: int = 8, seed: int = 19) -> dict:
    """FP32 forward of the master weights over `batches` (input_values
    tensors) with hooks: per block, the absmax per input channel of the four
    Linear inputs, and a sample of `tokens_per_clip` random token rows per
    clip of each Linear input and of Q, K, V (absolute values for Q, K, V),
    on the CPU. Returns {"absmax": [12 x {site: [in]}], "rows": [12 x
    {site: [n, in]}]}."""
    qast.set_config(WeightSpec(32), None)
    L = len(qast.weights)
    absmax = [{} for _ in range(L)]
    rows = [{s: [] for s in LIN_SITES + ATT_SITES} for _ in range(L)]
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    H, dh = qast.heads, qast.head_dim
    for x in batches:
        B = x.shape[0]
        h = qast.ast.embeddings(x)
        T = h.shape[1]
        pick = torch.stack([torch.randperm(T, generator=gen)[:tokens_per_clip] for _ in range(B)])  # [B, n]
        bi = torch.arange(B)[:, None].expand_as(pick)

        def take(t, site, i):
            r = t.detach()[bi, pick.to(t.device)].reshape(-1, t.shape[-1]).float().cpu()
            rows[i][site].append(r)

        def amax(t, site, i):
            m = t.detach().abs().amax(dim=(0, 1)).float().cpu()
            absmax[i][site] = m if site not in absmax[i] else torch.maximum(absmax[i][site], m)

        for i, (layer, w, b) in enumerate(zip(qast.ast.encoder.layer, qast.weights, qast.bias)):
            x1 = layer.layernorm_before(h)
            amax(x1, "qkv", i); take(x1, "qkv", i)
            qkv = F.linear(x1, w["qkv"], b["qkv"])
            q, k, v = qkv.view(B, T, 3, H * dh).unbind(2)
            take(q.abs(), "q", i); take(k.abs(), "k", i); take(v.abs(), "v", i)
            qh, kh, vh = qkv.view(B, T, 3, H, dh).permute(2, 0, 3, 1, 4).unbind(0)
            ctx = torch.matmul(torch.softmax(torch.matmul(qh, kh.transpose(-1, -2)) * qast.scaling, dim=-1), vh)
            ctx = ctx.transpose(1, 2).reshape(B, T, H * dh)
            amax(ctx, "attn_out", i); take(ctx, "attn_out", i)
            h = h + F.linear(ctx, w["attn_out"], b["attn_out"])
            x2 = layer.layernorm_after(h)
            amax(x2, "fc1", i); take(x2, "fc1", i)
            x3 = F.gelu(F.linear(x2, w["fc1"], b["fc1"]))
            amax(x3, "fc2", i); take(x3, "fc2", i)
            h = h + F.linear(x3, w["fc2"], b["fc2"])
    return {"absmax": absmax, "rows": [{s: torch.cat(v) for s, v in r.items()} for r in rows]}


def calibration_tables(qast: QuantAST, calib: dict, percentiles=(99.9, 99.99), alpha: float = 0.5) -> dict:
    """SmoothQuant scales from the calibration absmax and the master weights,
    and clip bounds per percentile from the sampled rows after smoothing:
    {"smooth": [12 x {site: s}], "clip": {p: [12 x {site: (lo, hi) or c}]}}."""
    import numpy as np

    smooth = []
    for i, m in enumerate(qast.master):
        smooth.append({k: smooth_scales(calib["absmax"][i][k], m[k].float().cpu(), alpha) for k in LINEAR_SLOTS})
    clip = {}
    for p in percentiles:
        per = []
        for i in range(len(qast.master)):
            d = {}
            for k in LIN_SITES:
                v = (calib["rows"][i][k] / smooth[i][k][None, :]).numpy().ravel()
                d[k] = (float(np.percentile(v, 100 - p)), float(np.percentile(v, p)))
            for k in ATT_SITES:
                d[k] = float(np.percentile(calib["rows"][i][k].numpy().ravel(), p))
            per.append(d)
        clip[p] = per
    return {"smooth": smooth, "clip": clip}
