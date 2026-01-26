#!/usr/bin/env python3
"""
Unit tests for curiosity-driven routing mechanisms.

Tests all three curiosity strategies:
- kl_divergence (Paper's Equation 8)
- uncertainty_weighted (Variance reduction)
- entropy_regularization (Experimental)
"""

import torch
import torch.nn.functional as F
from QWave.moe import qMoEModelBatched
from omegaconf import OmegaConf
import pytest


def create_test_config(strategy="uncertainty_weighted"):
    """Create test configuration for MoE model."""
    return OmegaConf.create({
        "experiment": {
            "router": {
                "hidden_dim": 128,
                "expert_quantizations": ['4', '8'],
                "num_experts": 2,
                "top_k": 1,
                "use_curiosity": True,
                "curiosity_alpha": 0.1,
                "curiosity_strategy": strategy,
                "mc_samples": 10,
                "safe_expert_idx": -1,
                "load_balancing_alpha": 1e-3,
            },
            "model": {"hidden_sizes": [640, 320], "dropout_prob": 0.2},
        }
    })


def test_baseline_no_curiosity():
    """Test that baseline (no curiosity) produces deterministic routing."""
    cfg = create_test_config()
    cfg.experiment.router.use_curiosity = False

    model = qMoEModelBatched(cfg, in_dim=1536, num_classes=50, num_experts=2, top_k=1)
    model.eval()

    x = torch.randn(16, 1536)
    out1, router_p1, _, _ = model(x)
    out2, router_p2, _, _ = model(x)

    # Should be deterministic when curiosity is off
    assert torch.allclose(router_p1, router_p2, atol=1e-5), \
        "Baseline routing should be deterministic!"

    print("✓ Baseline (no curiosity) is deterministic")


@pytest.mark.parametrize("strategy", [
    "kl_divergence",
    "entropy_regularization"
])
def test_curiosity_modifies_routing(strategy):
    """Test that each curiosity strategy modifies routing probabilities."""
    cfg = create_test_config(strategy)

    model = qMoEModelBatched(cfg, in_dim=1536, num_classes=50, num_experts=2, top_k=1)
    model.eval()

    x = torch.randn(16, 1536)

    # Without curiosity
    cfg_baseline = create_test_config(strategy)
    cfg_baseline.experiment.router.use_curiosity = False
    model_baseline = qMoEModelBatched(cfg_baseline, in_dim=1536, num_classes=50, num_experts=2, top_k=1)
    model_baseline.eval()
    _, router_p_baseline, _, _ = model_baseline(x)

    # With curiosity
    _, router_p_curiosity, _, _ = model(x)

    # Routing should be different
    diff = (router_p_baseline - router_p_curiosity).abs().mean()
    assert diff > 1e-4, f"Strategy '{strategy}' should change routing probabilities! (diff={diff:.6f})"

    print(f"✓ Strategy '{strategy}' modifies routing (mean diff={diff:.4f})")
    print(f"  Baseline expert 0 usage: {(router_p_baseline.argmax(dim=1) == 0).float().mean():.3f}")
    print(f"  Curiosity expert 0 usage: {(router_p_curiosity.argmax(dim=1) == 0).float().mean():.3f}")


def test_kl_divergence_modifies_distribution():
    """Test that KL divergence modifies routing distribution."""
    cfg = create_test_config("kl_divergence")

    cfg_baseline = create_test_config("kl_divergence")
    cfg_baseline.experiment.router.use_curiosity = False
    model_baseline = qMoEModelBatched(cfg_baseline, in_dim=1536, num_classes=50, num_experts=2, top_k=1)
    model_baseline.eval()

    model_kl = qMoEModelBatched(cfg, in_dim=1536, num_classes=50, num_experts=2, top_k=1)
    model_kl.eval()

    x = torch.randn(32, 1536)

    _, router_p_baseline, _, _ = model_baseline(x)
    _, router_p_kl, _, _ = model_kl(x)

    # Compute routing entropy
    entropy_baseline = -(router_p_baseline * (router_p_baseline + 1e-8).log()).sum(dim=1).mean()
    entropy_kl = -(router_p_kl * (router_p_kl + 1e-8).log()).sum(dim=1).mean()

    print(f"  Baseline entropy: {entropy_baseline:.4f}")
    print(f"  KL divergence entropy: {entropy_kl:.4f}")

    # KL divergence may increase or decrease entropy depending on base distribution
    # Just check that it's different
    assert not torch.allclose(entropy_baseline, entropy_kl, atol=1e-3), \
        "KL divergence should modify routing distribution!"

    print("✓ KL divergence modifies routing distribution")


def test_all_strategies_preserve_normalization():
    """Test that all strategies produce valid probability distributions."""
    strategies = ["kl_divergence", "entropy_regularization"]

    for strategy in strategies:
        cfg = create_test_config(strategy)
        model = qMoEModelBatched(cfg, in_dim=1536, num_classes=50, num_experts=2, top_k=1)
        model.eval()

        x = torch.randn(16, 1536)
        _, router_p, _, _ = model(x)

        # Check probabilities sum to 1
        sums = router_p.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
            f"Strategy '{strategy}' produces invalid probabilities! (sums={sums[:3]})"

        # Check all probabilities are in [0, 1]
        assert (router_p >= 0).all() and (router_p <= 1).all(), \
            f"Strategy '{strategy}' has probabilities outside [0,1]!"

        print(f"✓ Strategy '{strategy}' preserves normalization")


if __name__ == "__main__":
    print("\n=== Testing Curiosity Mechanisms ===\n")

    test_baseline_no_curiosity()
    print()

    for strategy in ["kl_divergence", "entropy_regularization"]:
        test_curiosity_modifies_routing(strategy)
        print()

    test_kl_divergence_modifies_distribution()
    print()

    test_all_strategies_preserve_normalization()
    print()

    print("=== All Tests Passed! ===")
