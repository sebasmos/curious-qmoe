"""Tests for QWave/effnet_quant.py: the in-memory mel image, the quantizers,
the W8A8 EfficientNet-B3 and the cost counters. CPU only, random weights
(pretrained=False) so no download is needed."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from QWave.effnet_quant import (MelImageSpec, act_scale_dims, ast_fbank_frontend_macs, backbone_bitops,
                                build_backbone, colormap_lut, conv_macs, efficientnet_b3_pooled,
                                fake_quant_sym, flatness_of, mel_image_frontend_macs, mel_images, mel_power,
                                parse_config, rfft_macs, rtn_per_channel, weight_bits_stored)


@pytest.fixture(scope="module")
def base():
    torch.manual_seed(0)
    return efficientnet_b3_pooled(pretrained=False)


def _waves(seed=0, lengths=(16000, 16000, 3000)):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(n).astype(np.float32) * 0.1 for n in lengths]


# ── front end ──────────────────────────────────────────────────────────────

def test_mel_image_shape_range_and_uint8_levels():
    imgs = mel_images(_waves())
    assert imgs.shape == (3, 3, 300, 300)
    assert imgs.min() >= 0 and imgs.max() <= 1
    assert torch.allclose(imgs * 255, (imgs * 255).round(), atol=1e-4)


def test_mel_image_deterministic_and_batch_invariant():
    w = _waves()
    a = mel_images(w)
    b = torch.cat([mel_images([x]) for x in w])
    assert torch.equal(a, b)
    assert torch.equal(a, mel_images(w))


def test_mel_image_handles_short_and_silent_clips():
    imgs, flat = mel_images([np.zeros(16000, np.float32), np.ones(100, np.float32) * 0.01], with_flatness=True)
    assert torch.isfinite(imgs).all()
    assert flat[0] == 0.0
    # a silent clip is one colour: the lowest entry of the map
    lut = colormap_lut("magma")[0].float() / 255
    assert torch.allclose(imgs[0, :, 150, 150], lut, atol=1e-6)


def test_low_frequencies_at_the_bottom():
    t = np.arange(32000) / 16000
    tone = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    img = mel_images([tone])[0].mean(0)
    assert img[-30:].mean() > img[:30].mean()


def test_colormap_lut_matches_matplotlib_bytes():
    import matplotlib

    lut = colormap_lut("magma")
    ref = matplotlib.colormaps["magma"](np.array([0.0, 0.5, 0.999]), bytes=True)[:, :3]
    idx = torch.tensor([0, 128, 255])
    assert np.array_equal(lut[idx].numpy(), ref)


def test_flatness_white_above_tone():
    spec = MelImageSpec()
    t = np.arange(32000) / 16000
    tone = torch.tensor(np.sin(2 * np.pi * 1000 * t), dtype=torch.float32)
    noise = torch.tensor(np.random.default_rng(1).standard_normal(32000), dtype=torch.float32)
    f = flatness_of(torch.stack([mel_power(tone, spec), mel_power(noise, spec)]))
    assert 0 < f[0] < f[1] <= 1


# ── quantizers ─────────────────────────────────────────────────────────────

def test_rtn_per_channel_range_and_error():
    w = torch.randn(16, 8, 3, 3)
    codes, s = rtn_per_channel(w, 8)
    assert codes.abs().max() <= 127
    assert torch.equal(codes, codes.round())
    assert ((codes * s - w).abs() <= s / 2 + 1e-7).all()
    # the largest weight of each channel lands on the top code
    assert torch.equal(codes.abs().flatten(1).amax(1), torch.full((16,), 127.0))


def test_fake_quant_levels_and_zero_slices():
    x = torch.randn(2, 4, 5, 5)
    x[1] = 0
    q = fake_quant_sym(x, 8, (1, 2, 3))
    assert torch.equal(q[1], torch.zeros_like(q[1]))
    s = x[0].abs().max() / 127
    assert len(torch.unique((q[0] / s).round())) <= 255
    assert (q[0] - x[0]).abs().max() <= s / 2 + 1e-6


def test_act_scale_dims_by_layer_kind():
    assert act_scale_dims(nn.Conv2d(8, 16, 1)) == (1,)
    assert act_scale_dims(nn.Conv2d(8, 8, 3, groups=8)) == (2, 3)
    assert act_scale_dims(nn.Conv2d(3, 40, 3, stride=2)) == (1, 2, 3)


def test_parse_config():
    assert parse_config("fp32") == (32, None)
    assert parse_config("w8a8") == (8, 8)
    assert parse_config("w4a8") == (4, 8)
    with pytest.raises(ValueError):
        parse_config("w8")


def test_w8a8_close_to_fp32_and_batch_invariant(base):
    torch.manual_seed(1)
    x = torch.rand(3, 3, 96, 96)
    fp = build_backbone("fp32", base)
    q = build_backbone("w8a8", base)
    assert q.n_conv == sum(isinstance(m, nn.Conv2d) for m in base.modules())
    with torch.no_grad():
        e, eq = fp(x), q(x)
        alone = torch.cat([q(x[i:i + 1]) for i in range(3)])
    assert e.shape == (3, 1536)
    with torch.no_grad():
        assert torch.allclose(e, base(x), atol=1e-5)
    rel = ((eq - e).norm(dim=1) / e.norm(dim=1)).max().item()
    assert 0 < rel < 0.5
    assert torch.allclose(alone, eq, atol=1e-5)
    # the base model is left untouched
    assert all(not m._forward_pre_hooks for m in base.modules() if isinstance(m, nn.Conv2d))


# ── costs ──────────────────────────────────────────────────────────────────

def test_conv_macs_match_stage1_count(base):
    c = conv_macs(base, 300)
    # idea 6 Stage 1 costs.json: conv 1.690487 G, depthwise 0.133259 G, squeeze-excitation 1.859648 M
    assert c["parts_macs"]["conv"] == 1690487136
    assert c["parts_macs"]["depthwise"] == 133258608
    assert c["parts_macs"]["squeeze_excitation"] == 1859648
    assert c["parts_macs"]["linear"] == 0
    assert c["macs"] == sum(c["parts_macs"].values())


def test_backbone_bitops_views():
    m = {"macs": 10.0, "out_elems": 3.0}
    assert backbone_bitops(m, "fp32") == 10 * 1024
    assert backbone_bitops(m, "fp32", 1024) == 10 * 1024
    assert backbone_bitops(m, "w8a8") == 10 * 64
    assert backbone_bitops(m, "w8a8", 1024) == 10 * 64 + 3 * 1024


def test_weight_bits_stored(base):
    n_conv = sum(m.weight.numel() for m in base.modules() if isinstance(m, nn.Conv2d))
    rows = sum(m.weight.shape[0] for m in base.modules() if isinstance(m, nn.Conv2d))
    n_all = sum(p.numel() for p in base.parameters())
    assert weight_bits_stored(base, "fp32") == 32 * n_all
    assert weight_bits_stored(base, "w8a8") == 32 * n_all - 24 * n_conv + 32 * rows


def test_frontend_costs():
    assert rfft_macs(1024) == 1.25 * 1024 * 10
    a, b = mel_image_frontend_macs(16000), mel_image_frontend_macs(32000)
    assert 0 < a < b
    f = ast_fbank_frontend_macs(80000)
    assert ast_fbank_frontend_macs(100) == 1024 * 128  # shorter than one frame: normalization only
    assert f > ast_fbank_frontend_macs(40000)
    assert math.isfinite(f)
