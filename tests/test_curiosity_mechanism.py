#!/usr/bin/env python3
"""
Unit tests for curiosity-driven routing mechanisms.

Tests the two curiosity strategies:
- kl_divergence (Paper's Equation 8)
- entropy_regularization

REGRESSION HISTORY: the original version of this suite compared two
independently initialized models, so routing always differed and every test
passed even while _apply_curiosity was an exact identity map (the per-expert
KL term was summed to one scalar per sample, which renormalizes away).
These tests now hold the model fixed and vary only alpha / uncertainty on the
SAME router logits, which is the comparison that catches that bug.

Run: PYTHONPATH=. python tests/test_curiosity_mechanism.py
(no pytest dependency)
"""

import torch
import torch.nn.functional as F
from QWave.moe import qMoEModelBatched
from omegaconf import OmegaConf


def create_test_config(strategy="kl_divergence", alpha=0.1):
    """Create test configuration for MoE model."""
    return OmegaConf.create({
        "experiment": {
            "router": {
                "hidden_dim": 128,
                "expert_quantizations": ['4', '8'],
                "num_experts": 2,
                "top_k": 1,
                "use_curiosity": True,
                "curiosity_alpha": alpha,
                "curiosity_strategy": strategy,
                "mc_samples": 10,
                "safe_expert_idx": -1,
                "load_balancing_alpha": 1e-3,
            },
            "model": {"hidden_sizes": [640, 320], "dropout_prob": 0.2},
        }
    })


def make_model(strategy="kl_divergence", alpha=0.1, seed=0):
    torch.manual_seed(seed)
    cfg = create_test_config(strategy, alpha)
    model = qMoEModelBatched(cfg, in_dim=1536, num_classes=50, num_experts=2, top_k=1)
    model.eval()
    return model


def test_baseline_no_curiosity():
    """Baseline (no curiosity) produces deterministic routing."""
    cfg = create_test_config()
    cfg.experiment.router.use_curiosity = False
    torch.manual_seed(0)
    model = qMoEModelBatched(cfg, in_dim=1536, num_classes=50, num_experts=2, top_k=1)
    model.eval()

    x = torch.randn(16, 1536)
    _, router_p1, _, _ = model(x)
    _, router_p2, _, _ = model(x)

    assert torch.allclose(router_p1, router_p2, atol=1e-5), \
        "Baseline routing should be deterministic!"
    print("✓ Baseline (no curiosity) is deterministic")


def test_alpha_zero_is_identity():
    """alpha=0 must leave the routing distribution exactly unchanged, both strategies."""
    torch.manual_seed(1)
    logits = torch.randn(64, 2) * 1.5
    uncertainty = torch.rand(64)
    base = F.softmax(logits, dim=1)

    for strategy in ["kl_divergence", "entropy_regularization"]:
        model = make_model(strategy, alpha=0.0)
        out = model._apply_curiosity(logits, uncertainty)
        d = (out - base).abs().max().item()
        assert d < 1e-6, f"alpha=0 must be identity for '{strategy}' (max diff {d})"
        print(f"✓ alpha=0 is identity for '{strategy}'")


def test_alpha_changes_routing_same_model():
    """THE regression test for the identity-map bug: the SAME model on the SAME
    logits must produce genuinely different routing when only alpha changes."""
    torch.manual_seed(2)
    logits = torch.randn(256, 2) * 1.5
    uncertainty = torch.rand(256)
    base = F.softmax(logits, dim=1)

    for strategy in ["kl_divergence", "entropy_regularization"]:
        model = make_model(strategy)
        prev_delta = 0.0
        for alpha in [0.1, 0.3, 1.0]:
            model.curiosity_alpha = alpha
            out = model._apply_curiosity(logits, uncertainty)
            delta = (out - base).abs().max().item()
            assert delta > 1e-3, (
                f"Strategy '{strategy}' at alpha={alpha} changed routing by only "
                f"{delta:.2e}; the identity-map bug is back")
            assert delta > prev_delta, (
                f"Strategy '{strategy}': larger alpha must move routing further "
                f"(alpha={alpha}: {delta:.4f} <= previous {prev_delta:.4f})")
            prev_delta = delta
        print(f"✓ Strategy '{strategy}': alpha has a real, monotone effect "
              f"(max|Δp| at alpha=1.0: {prev_delta:.4f})")


def test_kl_sharpens_toward_one_hot():
    """The paper's stability argument: Eq. 8 pushes routing toward one-hot,
    so top-1 probability must rise and routing entropy must fall with alpha."""
    torch.manual_seed(3)
    logits = torch.randn(512, 2) * 1.5
    uncertainty = torch.rand(512)
    model = make_model("kl_divergence")

    prev_top1, prev_entropy = 0.0, float("inf")
    for alpha in [0.0, 0.2, 0.3, 1.0]:
        model.curiosity_alpha = alpha
        out = model._apply_curiosity(logits, uncertainty)
        top1 = out.max(dim=1).values.mean().item()
        entropy = -(out * (out + 1e-12).log()).sum(dim=1).mean().item()
        assert top1 >= prev_top1, f"top-1 prob must be non-decreasing in alpha ({top1} < {prev_top1})"
        assert entropy <= prev_entropy, f"routing entropy must be non-increasing in alpha"
        prev_top1, prev_entropy = top1, entropy
    print(f"✓ KL strategy sharpens toward one-hot (top-1 {prev_top1:.4f}, entropy {prev_entropy:.4f} at alpha=1.0)")


def test_kl_matches_power_sharpening():
    """Fixed Eq. 8 has the closed form p^curious ∝ p^(1+alpha)."""
    torch.manual_seed(4)
    logits = torch.randn(128, 2) * 1.5
    uncertainty = torch.rand(128)
    model = make_model("kl_divergence")
    alpha = 0.3
    model.curiosity_alpha = alpha

    out = model._apply_curiosity(logits, uncertainty)
    base = F.softmax(logits, dim=1)
    power = base.pow(1 + alpha)
    power = power / power.sum(dim=1, keepdim=True)
    d = (out - power).abs().max().item()
    assert d < 1e-4, f"KL strategy must equal power sharpening p^(1+alpha) (max diff {d})"
    print(f"✓ KL strategy matches closed form p^(1+alpha) (max diff {d:.2e})")


def test_entropy_strategy_uses_uncertainty():
    """The entropy strategy must actually consume epistemic uncertainty:
    high-uncertainty samples move more than low-uncertainty samples."""
    torch.manual_seed(5)
    B = 200
    logits = torch.randn(B, 2) * 1.5
    base = F.softmax(logits, dim=1)
    model = make_model("entropy_regularization")
    model.curiosity_alpha = 1.0

    u = torch.linspace(0.0, 1.0, B)
    out = model._apply_curiosity(logits, u)
    low_move = (out[:50] - base[:50]).abs().max().item()
    high_move = (out[-50:] - base[-50:]).abs().max().item()
    assert high_move > low_move, (
        f"High-uncertainty samples must move more than low-uncertainty ones "
        f"({high_move:.4f} vs {low_move:.4f})")

    zero = model._apply_curiosity(logits, torch.zeros(B))
    dz = (zero - base).abs().max().item()
    assert dz < 1e-6, f"Degenerate (all-zero) uncertainty must fall back to base routing (diff {dz})"
    print(f"✓ Entropy strategy consumes uncertainty (high-u move {high_move:.4f} > low-u move {low_move:.4f})")


def test_precision_prior_routes_uncertain_to_high_precision():
    """precision_prior must shift uncertain samples toward the highest-precision
    expert BY CONSTRUCTION, while confident samples keep their base routing."""
    torch.manual_seed(7)
    cfg = create_test_config("precision_prior", alpha=1.0)
    cfg.experiment.router.expert_quantizations = ['bitnet', '4', '8']
    cfg.experiment.router.num_experts = 3
    model = qMoEModelBatched(cfg, in_dim=1536, num_classes=50, num_experts=3, top_k=1)
    model.eval()

    # beta must order bitnet < 4 < 8
    beta = model.precision_beta.tolist()
    assert beta[0] < beta[1] < beta[2], f"precision ordering wrong: {beta}"
    assert abs(beta[0]) < 1e-8 and abs(beta[2] - 1.0) < 1e-8

    B = 400
    logits = torch.randn(B, 3) * 1.5
    base = F.softmax(logits, dim=1)
    u = torch.linspace(0.0, 1.0, B)
    out = model._apply_curiosity(logits, u)

    # High-uncertainty rows gain mass on the highest-precision expert
    hi_gain = (out[-100:, 2] - base[-100:, 2]).mean().item()
    lo_gain = (out[:100, 2] - base[:100, 2]).mean().item()
    assert hi_gain > 0, f"uncertain samples must gain Q8 mass (got {hi_gain})"
    assert hi_gain > lo_gain, "Q8 gain must grow with uncertainty"
    # Low-uncertainty rows stay near base routing
    assert (out[:20] - base[:20]).abs().max().item() < 0.05, "confident samples should keep base routing"

    # alpha=0 identity
    model.curiosity_alpha = 0.0
    z = model._apply_curiosity(logits, u)
    assert (z - base).abs().max().item() < 1e-6, "alpha=0 must be identity for precision_prior"
    print(f"✓ precision_prior routes uncertainty to high precision (Q8 gain {hi_gain:.4f} at high u, {lo_gain:.4f} at low u)")


def test_new_strategies_route_uncertain_to_high_precision():
    """precision_sharp, escalation and soft_escalation must all shift the
    most-uncertain samples toward the highest-precision expert, and be the
    identity at alpha=0."""
    torch.manual_seed(8)
    B = 400
    logits = torch.randn(B, 3) * 1.5
    u = torch.linspace(0.0, 1.0, B)

    for strategy in ["precision_sharp", "escalation", "soft_escalation"]:
        cfg = create_test_config(strategy, alpha=0.3)
        cfg.experiment.router.expert_quantizations = ['bitnet', '4', '8']
        cfg.experiment.router.num_experts = 3
        torch.manual_seed(8)
        model = qMoEModelBatched(cfg, in_dim=1536, num_classes=50, num_experts=3, top_k=1)
        model.eval()
        base = F.softmax(logits, dim=1)

        out = model._apply_curiosity(logits, u)
        sums = out.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), f"{strategy}: not normalized"
        hi_gain = (out[-50:, 2] - base[-50:, 2]).mean().item()
        assert hi_gain > 0, f"{strategy}: most-uncertain rows must gain Q8 mass (got {hi_gain})"

        model.curiosity_alpha = 0.0
        z = model._apply_curiosity(logits, u)
        assert (z - base).abs().max().item() < 1e-6, f"{strategy}: alpha=0 must be identity"
        print(f"✓ {strategy}: uncertain rows gain Q8 mass ({hi_gain:.4f}), alpha=0 identity")


def test_all_strategies_preserve_normalization():
    """All strategies produce valid probability distributions."""
    torch.manual_seed(6)
    logits = torch.randn(64, 2) * 2
    uncertainty = torch.rand(64)

    for strategy in ["kl_divergence", "entropy_regularization", "precision_prior"]:
        model = make_model(strategy)
        for alpha in [0.02, 0.2, 1.0, 5.0]:
            model.curiosity_alpha = alpha
            router_p = model._apply_curiosity(logits, uncertainty)
            sums = router_p.sum(dim=1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
                f"Strategy '{strategy}' alpha={alpha}: probabilities do not sum to 1"
            assert (router_p >= 0).all() and (router_p <= 1).all(), \
                f"Strategy '{strategy}' alpha={alpha}: probabilities outside [0,1]"
        print(f"✓ Strategy '{strategy}' preserves normalization across alphas")


def test_forward_pass_end_to_end():
    """Full forward pass with curiosity on returns the 4-tuple and valid shapes."""
    model = make_model("kl_divergence", alpha=0.3)
    x = torch.randn(16, 1536)
    out, router_p, lb_loss, uncertainty = model(x)
    assert out.shape == (16, 50)
    assert router_p.shape == (16, 2)
    assert uncertainty is not None and uncertainty.shape == (16,)
    print("✓ End-to-end forward pass with curiosity returns valid outputs")


if __name__ == "__main__":
    print("\n=== Testing Curiosity Mechanisms (same-model, vary-alpha) ===\n")
    test_baseline_no_curiosity()
    test_alpha_zero_is_identity()
    test_alpha_changes_routing_same_model()
    test_kl_sharpens_toward_one_hot()
    test_kl_matches_power_sharpening()
    test_entropy_strategy_uses_uncertainty()
    test_precision_prior_routes_uncertain_to_high_precision()
    test_new_strategies_route_uncertain_to_high_precision()
    test_all_strategies_preserve_normalization()
    test_forward_pass_end_to_end()
    print("\n=== All Tests Passed! ===")
