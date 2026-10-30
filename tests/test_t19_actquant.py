"""T19 low-bit activation mode of QuantAST (QWave/backbone_quant.py, set_t19):
the asymmetric and clipped quantizers, exact reproduction of the legacy A8
path, SmoothQuant's identity, stochastic rounding, per-block bits and the
calibration tables. Tiny random AST on the CPU.

Run: PYTHONPATH=. pytest tests/test_t19_actquant.py -q
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from QWave.backbone_quant import (QuantAST, T19Act, WeightSpec, calibration_tables, collect_calibration,  # noqa: E402
                                  fake_quant_asym, fake_quant_sym, fake_quant_sym_clip, smooth_scales)
from tests.test_backbone_quant import tiny_ast  # noqa: E402


def inputs(m, n=3, seed=1):
    torch.manual_seed(seed)
    return torch.randn(n, m.config.max_length, m.config.num_mel_bins)


@pytest.mark.parametrize("bits", [4, 6, 8])
def test_asym_on_grid_zero_exact_and_half_step(bits):
    torch.manual_seed(0)
    x = torch.randn(5, 40) * 3 + 1.0
    x[0] = x[0].abs() + 0.5            # all positive row: range widened to contain 0
    y = fake_quant_asym(x, bits)
    qmax = 2 ** bits - 1
    mn = x.amin(-1, keepdim=True).clamp(max=0)
    mx = x.amax(-1, keepdim=True).clamp(min=0)
    s = (mx - mn) / qmax
    zp = torch.round(-mn / s)
    codes = y / s + zp
    assert torch.allclose(codes, codes.round(), atol=1e-3)
    assert (codes.round() >= 0).all() and (codes.round() <= qmax).all()
    assert ((y - x).abs() <= s * 0.5 + 1e-5 + s * 0.51 * (zp != -mn / s)).all()  # rounding the zero point moves the grid by < 1 step
    assert torch.equal(fake_quant_asym(torch.zeros(2, 8), bits), torch.zeros(2, 8))


def test_asym_clamp_bounds_values():
    x = torch.tensor([[-10.0, -1.0, 0.0, 0.5, 1.0, 50.0]])
    y = fake_quant_asym(x, 8, lo=-2.0, hi=2.0)
    assert y.min() >= -2.0 - 1e-6 and y.max() <= 2.0 + 1e-6


def test_sym_clip_none_equals_legacy_and_clamps():
    torch.manual_seed(0)
    x = torch.randn(4, 3, 7, 8)
    for dim in (-1, -2):
        assert torch.equal(fake_quant_sym_clip(x, 4, dim), fake_quant_sym(x, 4, dim))
    assert fake_quant_sym_clip(x, 4, -1, c=0.5).abs().max() <= 0.5 + 1e-6


def test_legacy_a8_reproduced_exactly_by_t19_mode():
    m = tiny_ast()
    qa = QuantAST(m, p_bits=16)
    x = inputs(m)
    refs = {}
    for spec in (WeightSpec(8, 16), WeightSpec(4, 16, nested=True), WeightSpec(4, 16)):
        qa.set_config(spec, 8)
        refs[spec] = qa(x)
        qa.set_t19(spec, T19Act(layer_bits=(8,) * 2, p_bits=16, smooth=None, clip=None, asym=False))
        out = qa(x)
        assert torch.equal(out, refs[spec]), (spec, (out - refs[spec]).abs().max())
    qa.set_config(WeightSpec(8, 16), 8)  # back to the legacy mode
    assert torch.equal(qa(x), refs[WeightSpec(8, 16)])


def test_smoothing_is_exact_in_float():
    torch.manual_seed(0)
    x, w = torch.randn(10, 16) * torch.linspace(0.1, 20, 16), torch.randn(8, 16)
    s = smooth_scales(x.abs().amax(0), w)
    assert torch.allclose((x / s) @ (w * s).T, x @ w.T, atol=1e-4, rtol=1e-5)
    # migration balances the per-channel maxima: max|X/s| == max|W s| per channel (alpha 0.5)
    assert torch.allclose((x / s).abs().amax(0), (w * s).abs().amax(0), rtol=1e-4)


def calib_of(qa, m):
    c = collect_calibration(qa, [inputs(m, 4, 7), inputs(m, 4, 8)], tokens_per_clip=3)
    return c, calibration_tables(qa, c, percentiles=(99.9, 99.99))


def test_calibration_tables_shapes_and_order():
    m = tiny_ast()
    qa = QuantAST(m, p_bits=8)
    c, t = calib_of(qa, m)
    D, F = m.config.hidden_size, m.config.intermediate_size
    assert t["smooth"][0]["qkv"].shape == (D,) and t["smooth"][1]["fc2"].shape == (F,)
    assert c["rows"][0]["qkv"].shape == (8 * 3, D)
    for i in range(2):
        for k in ("qkv", "attn_out", "fc1", "fc2"):
            lo9, hi9 = t["clip"][99.9][i][k]
            lo99, hi99 = t["clip"][99.99][i][k]
            assert lo99 <= lo9 < hi9 <= hi99
        for k in ("q", "k", "v"):
            assert 0 < t["clip"][99.9][i][k] <= t["clip"][99.99][i][k]


def act(t, bits, clip_p=99.9, sr=None, asym=True):
    clip = None if clip_p is None else {b: t["clip"][clip_p] for b in set(bits)}
    return T19Act(layer_bits=tuple(bits), p_bits=8, smooth=t["smooth"], clip=clip, asym=asym, sr_seed=sr)


def test_smoothed_a8_close_to_unsmoothed_and_fp32():
    m = tiny_ast()
    qa = QuantAST(m, p_bits=8)
    _, t = calib_of(qa, m)
    x = inputs(m)
    qa.set_config(WeightSpec(32), None)
    ref = qa(x)
    qa.set_t19(WeightSpec(8, 16), act(t, (8, 8), clip_p=None))
    a8 = qa(x)
    cos = torch.nn.functional.cosine_similarity(a8, ref).min()
    assert cos > 0.99
    qa.set_t19(WeightSpec(4, 16, nested=True), act(t, (4, 4)))
    a4 = qa(x)
    assert torch.isfinite(a4).all() and (a4 - ref).norm() > (a8 - ref).norm()


def test_stochastic_rounding_seeds_and_mixed_bits():
    m = tiny_ast()
    qa = QuantAST(m, p_bits=8)
    _, t = calib_of(qa, m)
    x = inputs(m)
    outs = []
    for sr in (1, 1, 2):
        qa.set_t19(WeightSpec(4, 16, nested=True), act(t, (4, 4), sr=sr))
        outs.append(qa(x))
    assert torch.equal(outs[0], outs[1]) and not torch.equal(outs[0], outs[2])
    qa.set_t19(WeightSpec(4, 16, nested=True), act(t, (4, 4)))
    det = qa(x)
    assert not torch.equal(det, outs[0])
    qa.set_t19(WeightSpec(4, 16, nested=True), act(t, (4, 8)))
    mixed = qa(x)
    qa.set_t19(WeightSpec(4, 16, nested=True), act(t, (8, 8)))
    a8 = qa(x)
    assert not torch.equal(mixed, det) and not torch.equal(mixed, a8)


def test_sr_is_unbiased():
    torch.manual_seed(0)
    x = torch.full((20000,), 0.3)
    from QWave.backbone_quant import _round
    assert abs(_round(x.clone(), True).mean().item() - 0.3) < 0.02
