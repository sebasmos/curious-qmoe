"""Tests for QWave/ast_frames.py on a small random AST (no download)."""
from __future__ import annotations

import copy

import pytest
import torch

transformers = pytest.importorskip("transformers")

from QWave.ast_frames import kept_token_index, shorten_ast, time_patches  # noqa: E402
from QWave.costs import ast_token_count  # noqa: E402


def _model(max_length=128):
    from transformers import ASTConfig, ASTModel

    torch.manual_seed(0)
    cfg = ASTConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
                    max_length=max_length, num_mel_bins=32, patch_size=16, frequency_stride=10, time_stride=10)
    return ASTModel(cfg).eval()


def test_shortened_embeddings_equal_the_kept_tokens_of_the_full_model():
    full = _model(128)
    short = shorten_ast(copy.deepcopy(full), 64)
    x = torch.randn(2, 64, 32)
    pad = torch.full((2, 64, 32), -0.7)  # any constant padding value
    with torch.no_grad():
        e_full = full.embeddings(torch.cat([x, pad], dim=1))
        e_short = short.embeddings(x)
    idx = kept_token_index(full.config, 64)
    assert e_short.shape[1] == len(idx) == 2 + 2 * time_patches(full.config, 64)
    assert torch.allclose(e_short, e_full[:, idx], atol=1e-6)


def test_shortened_model_runs_and_prices_fewer_tokens():
    full = _model(128)
    p_full, t_full = ast_token_count(full.config)
    short = shorten_ast(copy.deepcopy(full), 64)
    p_short, t_short = ast_token_count(short.config)
    assert (t_full, t_short) == (2 + 2 * 12, 2 + 2 * 5)  # 2 frequency rows of 12 and 5 time patches
    with torch.no_grad():
        out = short(torch.randn(1, 64, 32)).pooler_output
    assert out.shape == (1, 32)


def test_pretrained_grid_sizes():
    from transformers import ASTConfig

    cfg = ASTConfig()  # 128 mel bins, 1024 frames
    assert time_patches(cfg) == 101
    assert time_patches(cfg, 512) == 50 and time_patches(cfg, 400) == 39


def test_cannot_lengthen():
    with pytest.raises(ValueError):
        shorten_ast(_model(64), 128)
