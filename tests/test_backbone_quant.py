"""AST backbone quantization (QWave/backbone_quant.py): weight quantizers,
nested children, activation fake quantization, the fused forward against
transformers' ASTModel on a tiny random model, and the cost accounting.

Run: PYTHONPATH=. pytest tests/test_backbone_quant.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import QWave.backbone_quant as bq  # noqa: E402
from QWave.backbone_quant import (PARENT_BITS, QuantAST, WeightSpec, backbone_cost, cascade_cost,  # noqa: E402
                                  fake_quant_qk_, fake_quant_rows_, fake_quant_sym, nested_codes, parse_config,
                                  quant_softmax_av, quantize_weight, rtn_codes)

MPS = torch.backends.mps.is_available()
from QWave.costs import ast_token_count, transformer_cost  # noqa: E402


def tiny_ast(seed=0):
    from transformers import ASTConfig, ASTModel

    torch.manual_seed(seed)
    cfg = ASTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64,
                    num_mel_bins=32, max_length=64, patch_size=16, frequency_stride=16, time_stride=16,
                    attn_implementation="eager")
    m = ASTModel(cfg).eval()
    with torch.no_grad():  # non-trivial biases, LayerNorms and tokens
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return m


# ── weight quantizers ──────────────────────────────────────────────────────

@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("group", [None, 16])
def test_rtn_codes_on_grid_and_within_half_step(bits, group):
    torch.manual_seed(0)
    w = torch.randn(6, 64)
    codes, s = rtn_codes(w, bits, group)
    qmax = 2 ** (bits - 1) - 1
    assert codes.shape == (6, 1 if group is None else 64 // group, 64 if group is None else group)
    assert torch.equal(codes, codes.round()) and codes.abs().max() == qmax
    assert torch.equal(codes.abs().amax(-1, keepdim=True), torch.full_like(s, qmax))  # every group hits qmax
    err = (codes * s).reshape(6, 64) - w
    assert (err.abs().reshape(codes.shape) <= s / 2 + 1e-7).all()


def test_ternary_and_group_equals_channel_when_one_group():
    w = torch.tensor([[1.0, -0.4, 0.2, -0.9]])
    codes, s = rtn_codes(w, 2)
    assert codes.flatten().tolist() == [1, 0, 0, -1] and float(s) == pytest.approx(1.0)
    torch.manual_seed(1)
    w = torch.randn(5, 32)
    assert torch.equal(quantize_weight(w, WeightSpec(4)), quantize_weight(w, WeightSpec(4, group=32)))


def test_group_size_must_divide():
    with pytest.raises(ValueError):
        rtn_codes(torch.randn(2, 30), 4, 16)


def test_nested_codes_by_hand():
    c = torch.tensor([127., -127., 8., 24., -24., 7., -9., 0.])
    # /16: 7.94 -> 8 clamped to 7; -7.94 -> -8; 0.5 -> 0 (ties to even); 1.5 -> 2; -1.5 -> -2; 0.44 -> 0
    assert nested_codes(c, 4).tolist() == [7, -8, 0, 2, -2, 0, -1, 0]
    # /32: 3.97 -> 4 clamped to 3; -3.97 -> -4
    assert nested_codes(c, 3).tolist()[:2] == [3, -4]
    assert torch.equal(nested_codes(c, 8), c)
    with pytest.raises(ValueError):
        nested_codes(c, 1)


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("group", [None, 16])
def test_nested_weight_is_child_of_parent(k, group):
    torch.manual_seed(2)
    w = torch.randn(4, 32)
    pc, ps = rtn_codes(w, PARENT_BITS, group)
    child = quantize_weight(w, WeightSpec(k, group, nested=True))
    expect = (nested_codes(pc, k) * ps * 2 ** (PARENT_BITS - k)).reshape(w.shape)
    assert torch.allclose(child, expect)
    # the child sits on the parent's grid scaled by 2^(8-k), with 2^k levels at most
    codes = (child.reshape(pc.shape) / (ps * 2 ** (PARENT_BITS - k))).round()
    assert codes.min() >= -(2 ** (k - 1)) and codes.max() <= 2 ** (k - 1) - 1
    # escalating to the parent: child codes times 2^(8-k) plus a residual in [-2^(7-k), 2^(7-k)]
    # wherever the child was not clamped
    resid = pc - nested_codes(pc, k) * 2 ** (PARENT_BITS - k)
    unclamped = nested_codes(pc, k).abs() < 2 ** (k - 1) - 1
    assert resid[unclamped].abs().max() <= 2 ** (PARENT_BITS - k - 1)


def test_weight_spec_names_and_rules():
    assert [WeightSpec(32).name, WeightSpec(4).name, WeightSpec(3, 128).name, WeightSpec(4, nested=True).name,
            WeightSpec(3, 128, nested=True).name] == ["fp32", "w4", "w3g128", "n4", "n3g128"]
    assert WeightSpec(3, 64, nested=True).parent == WeightSpec(8, 64)
    for bad in (dict(bits=32, group=64), dict(bits=8, nested=True), dict(bits=1), dict(bits=16)):
        with pytest.raises(ValueError):
            WeightSpec(**bad)


def test_parse_config_round_trips():
    for name in ("fp32", "w8", "w5", "w4g128", "n4", "n3g64", "w2g128"):
        spec, act = parse_config(name)
        assert spec.name == name and act == (None if name == "fp32" else 8)
    for bad in ("w9", "x4", "n8", "fp16", "w4g"):
        with pytest.raises(ValueError):
            parse_config(bad)


# ── activations ────────────────────────────────────────────────────────────

def test_fake_quant_sym_per_token():
    torch.manual_seed(3)
    x = torch.randn(2, 5, 12)
    x[0, 1] = 0
    y = fake_quant_sym(x, 8)
    s = x.abs().amax(-1, keepdim=True) / 127
    assert torch.equal(y[0, 1], torch.zeros(12))
    nz = s.squeeze(-1) > 0
    assert torch.allclose((y / s.clamp_min(1e-30))[nz], (y / s.clamp_min(1e-30))[nz].round(), atol=1e-4)
    assert ((y - x).abs() <= s / 2 + 1e-6).all()
    assert torch.allclose(y.abs().amax(-1), x.abs().amax(-1))  # the extreme value is exact
    assert torch.allclose(fake_quant_sym(y, 8), y, atol=1e-6)  # idempotent
    # per channel over tokens (dim=-2) is the transpose of per token
    assert torch.allclose(fake_quant_sym(x, 8, dim=-2), fake_quant_sym(x.transpose(-1, -2), 8).transpose(-1, -2))


@pytest.mark.parametrize("scale", [1.0, 0.125])
def test_quant_softmax_av_equals_explicit_quantized_probabilities(scale):
    torch.manual_seed(4)
    S, v = 3 * torch.randn(2, 3, 7, 7), torch.randn(2, 3, 7, 5)
    P = torch.softmax(S * scale, -1)
    s = P.amax(-1, keepdim=True) / 255
    expect = torch.round(P / s) * s @ v
    got = quant_softmax_av(S.clone(), v, 8, scale)
    assert torch.allclose(got, expect, atol=1e-5)
    assert not torch.allclose(got, P @ v, atol=1e-6)  # quantization did something


def test_in_place_helpers_match_reference_on_cpu():
    torch.manual_seed(5)
    x = torch.randn(3, 4, 24)
    assert torch.equal(fake_quant_rows_(x.clone(), 8), fake_quant_sym(x, 8))
    qkv = torch.randn(2, 5, 3 * 2 * 4)
    got = fake_quant_qk_(qkv.clone(), 8, heads=2, head_dim=4).view(2, 5, 3, 2, 4)
    ref = qkv.view(2, 5, 3, 2, 4)
    assert torch.equal(got[:, :, :2], fake_quant_sym(ref[:, :, :2], 8)) and torch.equal(got[:, :, 2], ref[:, :, 2])


def _off_by_step_share(a, b):
    return (a != b).float().mean().item()


@pytest.mark.skipif(not MPS, reason="needs an Apple GPU")
def test_metal_kernels_match_torch_ops():
    torch.manual_seed(6)
    dev = torch.device("mps")
    for shape in [(2, 50, 768), (2, 50, 3072), (3, 7, 64), (2, 3, 40)]:
        x = 3 * torch.randn(*shape, device=dev)
        got, ref = fake_quant_rows_(x.clone(), 8), fake_quant_sym(x, 8)
        assert _off_by_step_share(got, ref) < 1e-4
        assert (got - ref).abs().max() <= (x.abs().amax(-1, keepdim=True) / 127).max() * 1.0001
    qkv = torch.randn(2, 50, 3 * 12 * 64, device=dev)
    got = fake_quant_qk_(qkv.clone(), 8, 12, 64).view(2, 50, 3, 12, 64).cpu()
    ref = qkv.view(2, 50, 3, 12, 64).cpu()
    assert _off_by_step_share(got[:, :, :2], fake_quant_sym(ref[:, :, :2], 8)) < 1e-4
    assert torch.equal(got[:, :, 2], ref[:, :, 2])
    S, v = 4 * torch.randn(2, 3, 300, 300, device=dev), torch.randn(2, 3, 300, 64, device=dev)
    for bits in (8, 16):
        got = quant_softmax_av(S.clone(), v, bits, 0.125)
        bq.USE_METAL = False
        try:
            ref = quant_softmax_av(S.clone(), v, bits, 0.125)
        finally:
            bq.USE_METAL = True
        assert ((got - ref).norm() / ref.norm()).item() < 1e-4


@pytest.mark.skipif(not MPS, reason="needs an Apple GPU")
@pytest.mark.parametrize("name", ["fp32", "w8", "n4g16"])
def test_metal_forward_matches_torch_forward(name):
    dev = torch.device("mps")
    m = tiny_ast().to(dev)
    x = torch.randn(3, 64, 32, device=dev)
    q = QuantAST(m)
    q.set_config(*parse_config(name))
    fast = q(x)
    bq.USE_METAL = False
    try:
        slow = q(x)
    finally:
        bq.USE_METAL = True
    assert torch.allclose(fast, slow, atol=2e-4)
    if name == "fp32":
        with torch.no_grad():
            assert torch.allclose(fast, m(input_values=x).pooler_output, atol=1e-5)


# ── the forward ────────────────────────────────────────────────────────────

def test_fp32_forward_matches_transformers():
    m = tiny_ast()
    x = torch.randn(3, 64, 32)
    q = QuantAST(m)
    with torch.no_grad():
        ref = m(input_values=x).pooler_output
    assert torch.allclose(q(x), ref, atol=1e-5)


def _quantize_hf_in_place(m, spec):
    with torch.no_grad():
        for mod in m.encoder.modules():
            if isinstance(mod, nn.Linear):
                mod.weight.copy_(quantize_weight(mod.weight, spec))


@pytest.mark.parametrize("spec", [WeightSpec(8), WeightSpec(4, 16), WeightSpec(3, 16, nested=True)])
def test_fused_weight_quantization_matches_per_matrix(spec):
    m = tiny_ast()
    x = torch.randn(2, 64, 32)
    q = QuantAST(m)
    q.set_config(spec, None)
    _quantize_hf_in_place(m, spec)
    with torch.no_grad():
        ref = m(input_values=x).pooler_output
    assert torch.allclose(q(x), ref, atol=1e-5)


def _reference_a8(m, x, bits=8, p_bits=8):
    """Unfused reference: transformers' modules with every product's inputs
    fake-quantized explicitly, probabilities quantized after the softmax."""
    def aq(t, dim=-1):
        qmax = 2 ** (bits - 1) - 1
        s = t.abs().amax(dim, keepdim=True).clamp_min(1e-30) / qmax
        return torch.round(t / s) * s

    with torch.no_grad():
        h = m.embeddings(x)
        for layer in m.encoder.layer:
            att = layer.attention.attention
            B, T, D = h.shape
            xn = aq(layer.layernorm_before(h))
            sh = lambda t: t.view(B, T, att.num_attention_heads, att.attention_head_size).transpose(1, 2)  # noqa: E731
            qh, kh, vh = sh(att.query(xn)), sh(att.key(xn)), sh(att.value(xn))
            scores = aq(qh) @ aq(kh).transpose(-1, -2) * att.scaling
            P = torch.softmax(scores, -1)
            sp = P.amax(-1, keepdim=True) / (2 ** p_bits - 1)
            ctx = (torch.round(P / sp) * sp) @ aq(vh, dim=-2)
            ctx = ctx.transpose(1, 2).reshape(B, T, D)
            h = h + layer.attention.output.dense(aq(ctx))
            inter = layer.intermediate(aq(layer.layernorm_after(h)))
            h = h + layer.output.dense(aq(inter))
        h = m.layernorm(h)
        return (h[:, 0] + h[:, 1]) / 2


@pytest.mark.parametrize("spec,p_bits", [(WeightSpec(8), 8), (WeightSpec(4, 16, nested=True), 8),
                                         (WeightSpec(4), 16)])
def test_a8_forward_matches_unfused_reference(spec, p_bits):
    m = tiny_ast()
    x = torch.randn(2, 64, 32)
    q = QuantAST(m, p_bits=p_bits)
    q.set_config(spec, 8)
    out = q(x)
    _quantize_hf_in_place(m, spec)
    ref = _reference_a8(m, x, p_bits=p_bits)
    # scaling Q before or after its quantization differs only by float rounding, which can flip a
    # rounding tie of a downstream activation; allow a small tolerance
    assert torch.allclose(out, ref, atol=2e-4), (out - ref).abs().max()
    q.set_config(WeightSpec(32), None)  # back to the reference
    with torch.no_grad():
        assert not torch.allclose(q(x), out, atol=1e-6)


def test_set_config_requantizes_from_master():
    m = tiny_ast()
    x = torch.randn(1, 64, 32)
    q = QuantAST(m)
    q.set_config(WeightSpec(4), 8)
    a = q(x)
    q.set_config(WeightSpec(2, 16), 8)
    q.set_config(WeightSpec(4), 8)
    assert torch.equal(q(x), a)


# ── costs ──────────────────────────────────────────────────────────────────

def test_backbone_cost_by_hand():
    m = tiny_ast()
    patches, tokens = ast_token_count(m.config)
    assert (patches, tokens) == (8, 10)
    lin_macs = 2 * (4 * 32 * 32 + 32 * 64 * 2) * tokens  # per layer: q, k, v, out; fc1, fc2
    attn_macs = 2 * 2 * tokens * tokens * 32
    patch_macs = 32 * 16 * 16 * patches
    c = backbone_cost(m, WeightSpec(4), 8)
    assert c["bitops"] == lin_macs * 4 * 8 + attn_macs / 2 * (8 * 8 + 16 * 8) + patch_macs * 32 * 32
    assert backbone_cost(m, WeightSpec(4), 8, p_bits=8)["bitops"] == \
        lin_macs * 4 * 8 + attn_macs * 8 * 8 + patch_macs * 32 * 32
    f = backbone_cost(m, WeightSpec(32), None)
    assert f["bitops"] == (lin_macs + attn_macs + patch_macs) * 32 * 32
    assert f["storage_bits"] == 8 * transformer_cost(m, tokens, patches, 32, 32, (32, 32))["weight_bytes"]
    n_lin = 2 * (4 * 32 * 32 + 2 * 32 * 64)
    assert c["linear_weights"] == n_lin
    # group 16: one extra fp32 scale per row per extra group
    g = backbone_cost(m, WeightSpec(4, 16), 8)
    rows_groups = 2 * (4 * 32 * (32 // 16 - 1) + 64 * (32 // 16 - 1) + 32 * (64 // 16 - 1))
    assert g["storage_bits"] - c["storage_bits"] == 32 * rows_groups
    assert c["storage_bits"] - backbone_cost(m, WeightSpec(3), 8)["storage_bits"] == n_lin


def test_cascade_cost_nested_and_not():
    m = tiny_ast()
    n4, w8, w4, f = (backbone_cost(m, s, a) for s, a in
                     ((WeightSpec(4, nested=True), 8), (WeightSpec(8), 8), (WeightSpec(4), 8), (WeightSpec(32), None)))
    c = cascade_cost(n4, w8, 0.25, head_cheap=10, head_expensive=20)
    assert c["nested"] and c["bitops"] == n4["bitops"] + 10 + 0.25 * (w8["bitops"] + 20)
    assert c["storage_bits"] == w8["storage_bits"]
    full = cascade_cost(n4, w8, 1.0)
    assert full["bits_read"] == pytest.approx(w8["bits_read"])  # every plane read once, no more
    d = cascade_cost(w4, f, 0.1)
    assert not d["nested"] and d["storage_bits"] == w4["storage_bits"] + f["storage_bits"]
    assert d["bits_read"] == pytest.approx(w4["bits_read"] + 0.1 * f["bits_read"])
    e = cascade_cost(n4, f, 0.1)  # a child escalating past its parent is stored on its own
    assert not e["nested"] and e["storage_bits"] == n4["storage_bits"] + f["storage_bits"]
    np.testing.assert_allclose(cascade_cost(w4, w8, 0.0)["bitops"], w4["bitops"])
