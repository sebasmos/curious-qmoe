"""Tests for QWave/efficientat_mn.py: the EfficientAT MobileNetV3 rebuild, its front end, the W8A8 copy and the
cost counters. CPU only, random weights, no download (the released weights were checked against EfficientAT's own
code separately, see the Stage 3b README)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from QWave import efficientat_mn as M


@pytest.fixture(scope="module")
def mn04():
    torch.manual_seed(0)
    m = M.build_mn("mn04")
    # give batch norm non-trivial statistics so quantization errors show
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.1, 0.1)
            mod.running_var.uniform_(0.5, 1.5)
    return m.eval()


def test_state_dict_layout_matches_the_release():
    # 312 entries and these shapes in mn04_as_mAP_432.pt / mn10_as_mAP_471.pt
    for name, emb, se_shape, n_par in (("mn04", 384, (8, 32), 993485), ("mn10", 960, (24, 72), 4901277)):
        sd = M.build_mn(name).state_dict()
        assert len(sd) == 312
        assert tuple(sd["features.4.block.2.conc_se_layers.0.fc1.weight"].shape) == se_shape
        assert sum(v.numel() for v in sd.values()) == n_par
        assert M.build_mn(name).emb_dim == emb


def test_embedding_shape_any_length(mn04):
    for frames in (32, 400, 500):
        with torch.no_grad():
            e = mn04(torch.randn(2, 1, 128, frames))
        assert e.shape == (2, 384) and torch.isfinite(e).all()


def test_make_divisible_matches_mobilenet_rule():
    assert M.make_divisible(16 * 0.4) == 8
    assert M.make_divisible(24 * 0.4) == 16  # 8 would lose more than 10%
    assert M.make_divisible(960) == 960


def test_front_end_frames_and_silence():
    n16 = 16000
    w = torch.zeros(1, n16)
    x = M.log_mel(M.resample_to_32k(w))
    assert x.shape == (1, 1, 128, M.frames_of(M.n32_of(n16)))
    assert M.frames_of(160000) == 500
    assert torch.allclose(x, torch.full_like(x, (math.log(1e-5) + 4.5) / 5), atol=1e-5)


def test_front_end_tone_lands_in_the_right_band():
    t = torch.arange(32000) / 16000
    tone = torch.sin(2 * math.pi * 1000 * t)[None]
    x = M.log_mel(M.resample_to_32k(tone))[0, 0].mean(1)
    fb = M._mel_basis()
    bin_1k = round(1000 / (M.SR / M.N_FFT))
    assert int(x.argmax()) == int(fb[:, bin_1k].argmax())


def test_model_input_groups_by_length():
    waves = [np.zeros(8000, np.float32), np.ones(16000, np.float32) * 0.01, np.zeros(8000, np.float32)]
    groups = M.model_input(waves)
    assert sorted(i for g, _ in groups for i in g) == [0, 1, 2]
    for idx, x in groups:
        assert x.shape[0] == len(idx) and x.shape[1:3] == (1, 128)


def test_w8a8_weights_on_a_per_channel_grid_and_inputs_quantized(mn04):
    q = M.build_backbone("w8a8", mn04)
    assert q.n_conv > 0 and q.n_linear > 0
    for mod in q.net.features.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            w = mod.weight.detach()
            s = w.abs().reshape(w.shape[0], -1).amax(1).clamp_min(1e-12) / 127
            codes = w.reshape(w.shape[0], -1) / s[:, None]
            assert torch.allclose(codes, codes.round(), atol=1e-3)
    # every Linear input reaching the layer lies on a per-row 8-bit grid
    seen = []

    def grab(mod, inp):
        seen.append(inp[0].detach())

    hs = [m.register_forward_pre_hook(grab) for m in q.net.features.modules() if isinstance(m, nn.Linear)]
    with torch.no_grad():
        q(torch.randn(2, 1, 128, 64))
    for h in hs:
        h.remove()
    assert seen
    for x in seen:
        s = x.abs().amax(-1, keepdim=True) / 127
        assert torch.allclose(x / s, (x / s).round(), atol=1e-3)


def test_w8a8_close_to_fp32_and_classifier_untouched(mn04):
    q = M.build_backbone("w8a8", mn04)
    x = torch.randn(3, 1, 128, 100)
    with torch.no_grad():
        a, b = mn04(x), q(x)
    rel = (a - b).norm(dim=1) / a.norm(dim=1)
    assert float(rel.mean()) < 0.2
    assert torch.equal(q.net.classifier[2].weight, mn04.classifier[2].weight)
    with pytest.raises(ValueError):
        M.build_backbone("w4a4", mn04)


def test_feature_macs_by_hand_and_linear_in_time(mn04):
    c = M.feature_macs(mn04, 64)
    stem = mn04.features[0][0]
    # stem: 1 -> 8 channels, 3 x 3, stride 2 on 128 x 64: 64 x 32 outputs per channel
    assert stem.out_channels == 8
    assert c["parts_macs"]["conv"] > 8 * 64 * 32 * 9
    assert c["parts_macs"]["squeeze_excitation"] > 0
    c2 = M.feature_macs(mn04, 128)
    assert 1.8 < c2["macs"] / c["macs"] < 2.1  # convolutions scale with time; the SE Linear does not
    assert M.backbone_bitops(c, "fp32") == c["macs"] * 1024
    assert M.backbone_bitops(c, "w8a8", 1024.0) == c["macs"] * 64 + c["out_elems"] * 1024


def test_frontend_macs_counts():
    orig, new, taps = M.resample_taps()
    assert (orig, new) == (1, 2) and taps == 15
    n16 = 80000
    n32 = M.n32_of(n16)
    assert n32 == 160000
    per_frame = M.WIN + M.rfft_macs(M.N_FFT) + 2 * 513 + M.mel_nnz() + 2 * 128
    assert M.frontend_macs(n16) == pytest.approx(taps * n32 + n32 - 1 + 500 * per_frame)
    assert 0 < M.mel_nnz() < 2 * 513


def test_storage_bits(mn04):
    fp = M.weight_bits_stored(mn04, "fp32")
    q = M.weight_bits_stored(mn04, "w8a8")
    n_feat = sum(p.numel() for p in mn04.features.parameters())
    assert fp == 32 * n_feat
    assert q < fp / 3
