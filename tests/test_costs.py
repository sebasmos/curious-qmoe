"""Bit-operation and weight-byte accounting (QWave/costs.py) against hand
calculations.

Run: PYTHONPATH=. pytest tests/test_costs.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from QWave.costs import ast_token_count, cost, layer_cost, moe_cost, transformer_cost, weight_bytes
from QWave.memory import storage_bits
from QWave.qmoe_layers import BitLinear, BitNetExpert, ptq_dynamic_int8


# ── heads ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bits,w,a", [(1, 1, 8), (4, 4, 8), (8, 8, 8), (16, 16, 32), (32, 32, 32), ("bitnet", 2, 2)])
def test_bitlinear_cost(bits, w, a):
    assert layer_cost(BitLinear(10, 6, num_bits=bits)).tolist() == [60, 60 * w * a]


def test_plain_and_int8_linear_cost():
    assert layer_cost(nn.Linear(10, 6)).tolist() == [60, 60 * 32 * 32]
    assert layer_cost(nn.LayerNorm(10)) is None
    q = ptq_dynamic_int8(BitNetExpert(10, 3, [6], 0.0, num_bits=32))
    assert cost(q).tolist() == [10 * 6 + 6 * 3, (10 * 6 + 6 * 3) * 8 * 8]


def test_q4_head_cost_by_hand():
    # 1536 -> 640 -> 320 at 4-bit weights, 8-bit activations; fp32 classifier
    m = BitNetExpert(1536, 10, [640, 320], 0.1, num_bits=4)
    hidden = 1536 * 640 + 640 * 320
    assert cost(m).tolist() == [hidden + 320 * 10, hidden * 4 * 8 + 320 * 10 * 32 * 32]


def test_moe_cost_weights_experts_by_share():
    from omegaconf import OmegaConf
    from QWave.moe import qMoEModelBatched

    cfg = OmegaConf.load(ROOT / "config" / "esc50.yaml")
    r = cfg.experiment.router
    r.expert_quantizations, r.num_experts, r.use_curiosity = ["4", "8"], 2, False
    model = qMoEModelBatched(cfg, 16, 3, 2, 1)
    router = cost(model.router)
    e4, e8 = cost(model.experts[0]), cost(model.experts[1])
    assert e8[1] == 2 * e4[1] - 320 * 3 * 1024, "only the hidden layers scale with the bit-width"
    np.testing.assert_allclose(moe_cost(model, [0.25, 0.75]), router + 0.25 * e4 + 0.75 * e8)


def test_weight_bytes_by_hand():
    m = BitNetExpert(4, 2, [3], 0.0, num_bits=4)
    # BitLinear 4->3: 12 codes x 4 bits, 3 row scales and 3 biases in fp32;
    # its pre-LN (8 params), the LayerNorm(3) (6) and the fp32 classifier 3->2 (8)
    bits = 12 * 4 + 3 * 32 + 3 * 32 + 8 * 32 + 6 * 32 + 8 * 32
    assert weight_bytes(m) == bits / 8 == storage_bits(m) / 8


# ── transformer backbone ───────────────────────────────────────────────────

def tiny_ast():
    from transformers import ASTConfig, ASTModel

    cfg = ASTConfig(hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
                    patch_size=4, num_mel_bins=12, max_length=12, frequency_stride=4, time_stride=4)
    return ASTModel(cfg), cfg


def test_ast_token_count():
    from transformers import ASTConfig

    assert ast_token_count(ASTConfig()) == (1212, 1214)
    model, cfg = tiny_ast()
    patches, tokens = ast_token_count(cfg)
    assert (patches, tokens) == (9, 11)
    out = model(input_values=torch.randn(1, cfg.max_length, cfg.num_mel_bins)).last_hidden_state
    assert out.shape[1] == tokens


def test_transformer_cost_by_hand():
    model, cfg = tiny_ast()
    c = transformer_cost(model, tokens=11, patches=9, weight_bits=4, act_bits=8, patch_bits=(8, 8))
    # per token: q, k, v, attention output 8x8 each, intermediate 8x16, output 16x8
    linear = (4 * 64 + 128 + 128) * 11
    patch = 8 * 1 * 4 * 4 * 9  # out channels x in channels x kernel x patches
    attn = 2 * 11 * 11 * 8  # Q K^T and the weighted sum of V
    assert c["linear"] == {"macs": linear, "bitops": linear * 4 * 8}
    assert c["patch_embed"] == {"macs": patch, "bitops": patch * 8 * 8}
    assert c["attention"] == {"macs": attn, "bitops": attn * 8 * 8}
    assert c["total"] == {"macs": linear + patch + attn, "bitops": linear * 32 + patch * 64 + attn * 64}
    # 4-bit Linear weights plus fp32 row scales and biases, the 8-bit conv the
    # same way, and in fp32: [CLS], [DIST], 11 x 8 position embeddings, 3 LayerNorms
    bits = (4 * (64 * 4 + 8 * 32 * 2)          # q, k, v, attention output
            + 128 * 4 + 16 * 32 * 2            # intermediate
            + 128 * 4 + 8 * 32 * 2             # output
            + 128 * 8 + 8 * 32 * 2             # patch embedding
            + 32 * (8 + 8 + 88 + 3 * 16))
    assert c["weight_bytes"] == bits / 8


def test_transformer_cost_fp32_and_per_layer_bits():
    model, _ = tiny_ast()
    fp32 = transformer_cost(model, 11, 9, 32, 32, (32, 32))
    assert fp32["weight_bytes"] == 4 * sum(p.numel() for p in model.parameters())
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    bits = {n: (2 if n.endswith("query") else 8) for n in names}
    c = transformer_cost(model, 11, 9, bits, 8, (32, 32))
    assert c["linear"]["bitops"] == 11 * (64 * 2 * 8 + (64 * 3 + 128 + 128) * 8 * 8)
    with pytest.raises(KeyError):
        transformer_cost(model, 11, 9, {names[0]: 4}, 8, (32, 32))
