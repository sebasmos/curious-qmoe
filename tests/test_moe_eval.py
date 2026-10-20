"""MoE evaluation: PTQ experts are quantized in the moe path, held-out routing
fractions, and the popcount expert is refused.

Run: PYTHONPATH=. pytest tests/test_moe_eval.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.benchmark as bench
from QWave.moe import qMoEModelBatched
from QWave.qmoe_layers import BitNetExpert


def moe_cfg(experts, alpha=1.0):
    return OmegaConf.create({"experiment": {
        "router": {"hidden_dim": 16, "expert_quantizations": experts, "num_experts": len(experts),
                   "top_k": 1, "use_curiosity": True, "curiosity_strategy": "precision_prior",
                   "curiosity_alpha": alpha, "mc_samples": 4, "load_balancing_alpha": 1e-3},
        "model": {"hidden_sizes": [16, 8], "dropout_prob": 0.2}}})


def is_int8_linear(m):
    return type(m).__name__ == "Linear" and "quantized" in type(m).__module__


def test_ptq_experts_are_quantized_in_the_moe_path():
    cfg = moe_cfg(["bitnet", "qesc"])
    torch.manual_seed(0)
    model = qMoEModelBatched(cfg, 32, 4, 2, 1)
    eval_model = bench.prepare_eval_model("moe", model, cfg)
    assert any(is_int8_linear(m) for m in eval_model.experts[1].modules()), "qesc expert must be int8"
    assert not any(is_int8_linear(m) for m in eval_model.experts[0].modules())
    out, router_p, _, _ = eval_model(torch.randn(6, 32))
    assert out.shape == (6, 4) and router_p.shape == (6, 2)


def test_ptq_single_model_is_quantized():
    cfg = moe_cfg(["bitnet"])
    model = BitNetExpert(32, 4, [16, 8], 0.1, num_bits=32)
    q = bench.prepare_eval_model("qesc", model, cfg)
    assert sum(is_int8_linear(m) for m in q.modules()) == 3


def test_routing_fractions_overall_and_by_quartile():
    rng = np.random.default_rng(0)
    experts = rng.integers(0, 3, 400)
    unc = rng.random(400)
    block = bench.routing_fractions(experts, unc, 3)
    assert np.isclose(sum(block["overall"]), 1.0)
    assert len(block["by_uncertainty_quartile"]) == 4
    assert all(np.isclose(sum(q), 1.0) for q in block["by_uncertainty_quartile"])
    assert sum(block["quartile_counts"]) == 400
    assert "by_uncertainty_quartile" not in bench.routing_fractions(experts, None, 3)


def test_route_test_fold_is_repeatable_under_a_seed():
    torch.manual_seed(0)
    model = qMoEModelBatched(moe_cfg(["bitnet", "4", "8"]), 32, 4, 3, 1)
    data = torch.utils.data.TensorDataset(torch.randn(40, 32), torch.zeros(40, dtype=torch.long))
    loader = torch.utils.data.DataLoader(data, batch_size=16)
    a = bench.route_test_fold(model, loader, "cpu", seed=3)
    b = bench.route_test_fold(model, loader, "cpu", seed=3)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_popcount_expert_is_refused():
    with pytest.raises(ValueError, match="popcount"):
        qMoEModelBatched(moe_cfg(["bitnet", "popcount"]), 32, 4, 2, 1)
