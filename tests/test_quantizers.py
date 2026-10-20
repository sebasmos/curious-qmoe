"""Weight quantizers: k-bit levels, fp16 for 16-bit, the FP32 baseline on the
same architecture, and dynamic int8 PTQ.

Run: PYTHONPATH=. pytest tests/test_quantizers.py -q
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from QWave.qmoe_layers import BitLinear, BitNetExpert, quantize_weight_kbit, ptq_dynamic_int8


def gaussian_weight(rows=64, cols=512, seed=0):
    return torch.randn(rows, cols, generator=torch.Generator().manual_seed(seed))


def levels_per_row(wq):
    return [len(torch.unique(r)) for r in wq]


def integer_codes(wq):
    """Integer level of every weight: level 1 (the row's smallest nonzero
    magnitude) is always occupied for Gaussian rows."""
    step = torch.where(wq == 0, torch.full_like(wq, float("inf")), wq.abs()).amin(dim=1, keepdim=True)
    return (wq / step).round()


def test_two_bit_is_ternary_with_minority_zeros():
    wq = quantize_weight_kbit(gaussian_weight(), 2)
    assert set(levels_per_row(wq)) == {3}
    zeros = (wq == 0).float().mean().item()
    assert zeros < 0.5, f"2-bit leaves {zeros:.0%} zeros"


def test_four_bit_reaches_all_fifteen_levels():
    codes = integer_codes(quantize_weight_kbit(gaussian_weight(), 4))
    assert torch.unique(codes).tolist() == list(range(-7, 8))
    full = sum(n == 15 for n in levels_per_row(codes)) / codes.shape[0]
    assert full > 0.8, f"only {full:.0%} of rows use all 15 levels"


def test_levels_are_symmetric_integers_times_a_row_step():
    w = gaussian_weight()
    for bits in (2, 4, 8):
        qmax = 2 ** (bits - 1) - 1
        wq = quantize_weight_kbit(w, bits)
        step = wq.abs().amax(dim=1, keepdim=True) / qmax
        codes = wq / step
        assert torch.allclose(codes, codes.round(), atol=1e-4)
        assert codes.abs().max() <= qmax + 1e-4


def test_one_bit_is_binary_with_mean_abs_scale():
    w = gaussian_weight()
    wq = quantize_weight_kbit(w, 1)
    assert set(levels_per_row(wq)) == {2}
    assert torch.allclose(wq.abs()[:, 0], w.abs().mean(dim=1))


def test_mse_clipping_beats_per_layer_max_scaling():
    w = gaussian_weight()
    for bits in (2, 4):
        qmax = 2 ** (bits - 1) - 1
        step = w.abs().max() / qmax
        old = (w / step).round().clamp(-qmax - 1, qmax) * step
        assert (quantize_weight_kbit(w, bits) - w).pow(2).sum() < (old - w).pow(2).sum()


def test_kbit_forward_uses_quantized_weights_with_ste_gradient():
    torch.manual_seed(0)
    layer = BitLinear(32, 8, num_bits=4)
    x = torch.randn(5, 32)
    layer(x).sum().backward()
    assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0


def test_quantized_weight_cache_follows_updates():
    torch.manual_seed(0)
    layer = BitLinear(32, 8, num_bits=4)
    x = torch.randn(5, 32)
    before = layer(x).detach()
    with torch.no_grad():
        layer.weight.mul_(2.0)
    after = layer(x).detach()
    assert not torch.allclose(before, after), "cached quantized weights must refresh after an update"


def test_sixteen_bit_rounds_weights_to_fp16():
    torch.manual_seed(0)
    layer = BitLinear(32, 8, num_bits=16)
    x = torch.randn(5, 32)
    expected = nn.functional.linear(layer.act_ln(x), layer.weight.detach().half().float(), layer.bias)
    assert torch.allclose(layer(x), expected)
    layer(x).sum().backward()
    assert layer.weight.grad.abs().sum() > 0


def test_fp32_baseline_has_the_k_bit_architecture():
    fp32 = BitNetExpert(64, 5, [16, 8], 0.1, num_bits=32)
    q4 = BitNetExpert(64, 5, [16, 8], 0.1, num_bits=4)
    assert [type(m) for m in fp32.modules()] == [type(m) for m in q4.modules()]
    assert sum(p.numel() for p in fp32.parameters()) == sum(p.numel() for p in q4.parameters())


def test_ptq_quantizes_every_linear_and_stays_close():
    torch.manual_seed(0)
    model = BitNetExpert(64, 5, [32, 16], 0.1, num_bits=32).eval()
    q = ptq_dynamic_int8(model)
    qlinears = [m for m in q.modules() if type(m).__name__ == "Linear" and "quantized" in type(m).__module__]
    assert len(qlinears) == 3, "two hidden layers and the classifier"
    assert not any(isinstance(m, BitLinear) for m in q.modules())
    x = torch.randn(16, 64)
    with torch.no_grad():
        ref, out = model(x), q(x)
    assert (ref - out).abs().max() < 0.1 * ref.abs().max()
    assert any(isinstance(m, BitLinear) for m in model.modules()), "the FP32 model is left untouched"
