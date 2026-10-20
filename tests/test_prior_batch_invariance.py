"""Eval-mode routing must not depend on batch composition.

The uncertainty-consuming strategies used to min-max normalize uncertainty
within each batch, so at batch size 1 (u_max == u_min) the precision prior
returned base routing and did nothing in streaming inference. Eval mode now
normalizes with running statistics tracked in training.

Run: PYTHONPATH=. pytest tests/test_prior_batch_invariance.py -q
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from QWave.moe import qMoEModelBatched

STRATEGIES = ["precision_prior", "entropy_regularization", "precision_sharp", "escalation", "soft_escalation"]
IN_DIM = 32


def make_model(strategy, alpha=1.0):
    torch.manual_seed(0)
    cfg = OmegaConf.create({"experiment": {
        "router": {"hidden_dim": 32, "expert_quantizations": ["bitnet", "4", "8"], "num_experts": 3,
                   "top_k": 1, "use_curiosity": True, "curiosity_strategy": strategy,
                   "curiosity_alpha": alpha, "mc_samples": 8, "load_balancing_alpha": 1e-3},
        "model": {"hidden_sizes": [16, 8], "dropout_prob": 0.3}}})
    model = qMoEModelBatched(cfg, IN_DIM, 5, 3, 1)
    model.train()
    for _ in range(5):  # what training does to the running statistics
        model(torch.randn(64, IN_DIM) * 2)
    assert model.unc_num_batches_tracked > 0
    return model.eval()


def seed_mc_dropout_per_sample(model, seed=1234):
    """Make MC dropout draw the same masks for a sample alone or in a batch:
    the real router runs row by row, reseeded before every row."""
    real_forward = model.router.forward

    def forward(x, compute_uncertainty=False):
        outs = []
        for row in x:
            torch.manual_seed(seed)
            outs.append(real_forward(row.unsqueeze(0), compute_uncertainty))
        logits = torch.cat([o[0] for o in outs])
        unc = torch.cat([o[1] for o in outs]) if compute_uncertainty else None
        return logits, unc

    model.router.forward = forward
    return model


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_single_sample_routing_equals_its_routing_in_a_batch(strategy):
    model = seed_mc_dropout_per_sample(make_model(strategy, alpha=0.3 if strategy == "escalation" else 1.0))
    x = torch.randn(24, IN_DIM) * 2
    with torch.no_grad():
        out_b, p_b, _, u_b = model(x)
        for i in range(len(x)):
            out_1, p_1, _, u_1 = model(x[i:i + 1])
            assert torch.allclose(u_1, u_b[i:i + 1], atol=1e-6)
            assert torch.allclose(p_1, p_b[i:i + 1], atol=1e-6), f"{strategy}: routing of sample {i} depends on its batch"
            assert torch.allclose(out_1, out_b[i:i + 1], atol=1e-5), f"{strategy}: output of sample {i} depends on its batch"


def test_precision_prior_acts_at_batch_size_one():
    model = seed_mc_dropout_per_sample(make_model("precision_prior"))
    x = torch.randn(24, IN_DIM) * 2
    moved = 0
    with torch.no_grad():
        for i in range(len(x)):
            _, p, _, _ = model(x[i:i + 1])
            base = F.softmax(model.router.net(x[i:i + 1]), dim=1)
            moved += int(not torch.allclose(p, base, atol=1e-6))
    assert moved > len(x) // 2, f"the prior changed routing for only {moved} of {len(x)} single samples"


def test_eval_normalization_is_clamped_to_unit_interval():
    model = make_model("precision_prior")
    lo, hi = model.unc_running_min.item(), model.unc_running_max.item()
    u = torch.tensor([lo - 1.0, lo, (lo + hi) / 2, hi, hi + 1.0])
    u_norm = model._normalized_uncertainty(u).squeeze(1)
    assert u_norm.min() >= 0 and u_norm.max() <= 1
    assert u_norm[0] == 0 and u_norm[-1] == 1


def test_running_statistics_are_saved_with_the_weights():
    model = make_model("precision_prior")
    state = model.state_dict()
    for name in ("unc_running_min", "unc_running_max", "unc_running_threshold", "unc_num_batches_tracked"):
        assert name in state
    fresh = qMoEModelBatched(OmegaConf.create({"experiment": {
        "router": {"hidden_dim": 32, "expert_quantizations": ["bitnet", "4", "8"], "num_experts": 3,
                   "top_k": 1, "use_curiosity": True, "curiosity_strategy": "precision_prior",
                   "curiosity_alpha": 1.0, "mc_samples": 8, "load_balancing_alpha": 1e-3},
        "model": {"hidden_sizes": [16, 8], "dropout_prob": 0.3}}}), IN_DIM, 5, 3, 1)
    fresh.load_state_dict(state)
    assert fresh.unc_running_max == model.unc_running_max


def test_train_mode_updates_running_range_with_momentum():
    model = make_model("precision_prior")
    model.train()
    before = model.unc_running_max.clone()
    u = torch.tensor([0.0, 10.0])
    model._normalized_uncertainty(u)
    assert torch.isclose(model.unc_running_max, 0.9 * before + 0.1 * 10.0)
