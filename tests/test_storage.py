"""Model storage from per-layer bit-widths (singles, PTQ, MoE, shared trunk).

Run: PYTHONPATH=. pytest tests/test_storage.py -q
"""
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from QWave.memory import storage_bits, print_size_of_model
from QWave.moe import qMoEModelBatched
from QWave.qmoe_layers import BitNetExpert, ptq_dynamic_int8

IN, H0, H1, C = 1536, 640, 320, 5


def expected_bits(bits):
    """Two BitLinear layers (weights, per-row scale, bias, pre-LN) plus the
    trailing LayerNorms and the fp32 classifier."""
    def layer(i, o):
        if bits == "bitnet":
            w = 2 * i * o + 32 * o
        elif bits >= 32:
            w = 32 * i * o
        elif bits == 16:
            w = 16 * i * o
        else:
            w = bits * i * o + 32 * o
        return w + 32 * o + 32 * 2 * i  # bias, pre-LayerNorm
    norms = 32 * 2 * (H0 + H1)
    head = 32 * (H1 * C + C)
    return layer(IN, H0) + layer(H0, H1) + norms + head


def test_single_model_storage_follows_bit_width():
    sizes = {}
    for bits in (1, 2, 4, 8, 16, 32, "bitnet"):
        model = BitNetExpert(IN, C, [H0, H1], 0.1, num_bits=bits)
        sizes[bits] = storage_bits(model)
        assert sizes[bits] == expected_bits(bits), bits
    assert sizes[1] < sizes[2] < sizes[4] < sizes[8] < sizes[16] < sizes[32]
    assert sizes["bitnet"] == sizes[2], "ternary BitNet is stored as 2-bit codes with a per-row scale"


def test_print_size_of_model_reports_bytes_per_bit_width():
    q4 = print_size_of_model(BitNetExpert(IN, C, [H0, H1], 0.1, num_bits=4))
    q16 = print_size_of_model(BitNetExpert(IN, C, [H0, H1], 0.1, num_bits=16))
    assert q4 == expected_bits(4) / 8 and q16 == expected_bits(16) / 8


def test_ptq_storage_is_int8_weights():
    fp32 = BitNetExpert(IN, C, [H0, H1], 0.1, num_bits=32)
    q = ptq_dynamic_int8(fp32)
    linear_w = IN * H0 + H0 * H1 + H1 * C
    assert storage_bits(fp32) - storage_bits(q) == 24 * linear_w - 3 * 64


def _moe(experts, shared_trunk=False):
    cfg = OmegaConf.create({"experiment": {
        "router": {"hidden_dim": 128, "expert_quantizations": experts, "num_experts": len(experts),
                   "top_k": 1, "use_curiosity": True, "curiosity_strategy": "precision_prior",
                   "curiosity_alpha": 1.0, "mc_samples": 2, "load_balancing_alpha": 1e-3,
                   "shared_trunk": shared_trunk, "trunk_bits": 8},
        "model": {"hidden_sizes": [H0, H1], "dropout_prob": 0.1}}})
    return qMoEModelBatched(cfg, IN, C, len(experts), 1)


def test_moe_storage_is_router_plus_experts():
    moe = _moe(["bitnet", "4", "8"])
    router = 32 * sum(p.numel() for p in moe.router.parameters())
    assert storage_bits(moe) == router + expected_bits("bitnet") + expected_bits(4) + expected_bits(8)


def test_shared_trunk_is_smaller_than_full_experts():
    shared, full = _moe(["bitnet", "4", "8"], shared_trunk=True), _moe(["bitnet", "4", "8"])
    assert storage_bits(shared) < storage_bits(full)
    assert sum(p.numel() for p in shared.parameters()) < sum(p.numel() for p in full.parameters()) / 2
