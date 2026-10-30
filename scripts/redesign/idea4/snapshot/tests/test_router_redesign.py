"""Router redesign options (redesign idea 4): dense experts with an
expected-loss router, entropy bonus, error head, hard escalation with a tau
chosen on the inner validation split, and the held-out diagnostics.

Run: PYTHONPATH=. pytest tests/test_router_redesign.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.benchmark as bench
from QWave.moe import qMoEModelBatched, expected_loss_router_loss, select_escalation_tau
from scripts.analysis import router_diagnostics
from test_protocol import synthetic_csv, tiny_cfg

IN, C = 12, 3


def cfg(**router):
    r = {"hidden_dim": 16, "expert_quantizations": ["bitnet", "4", "8"], "num_experts": 3, "top_k": 1,
         "use_curiosity": True, "curiosity_strategy": "precision_prior", "curiosity_alpha": 1.0,
         "mc_samples": 4, "load_balancing_alpha": 1e-3}
    r.update(router)
    return OmegaConf.create({"experiment": {"router": r, "model": {"hidden_sizes": [16, 8], "dropout_prob": 0.2}}})


def build(seed=0, **router):
    torch.manual_seed(seed)
    return qMoEModelBatched(cfg(**router), IN, C, 3, 1)


DENSE_C = dict(dense_experts=True, router_entropy_bonus=0.1, error_head=True, uncertainty_source="error_head")


def test_defaults_keep_the_original_model():
    old, new = build(), build(dense_experts=False, router_entropy_bonus=0.0, error_head=False,
                         uncertainty_source="mc_variance", escalation_rule="none")
    assert old.state_dict().keys() == new.state_dict().keys()
    assert "escalation_tau" not in new.state_dict()
    assert new.router.net[-1].out_features == 3
    x = torch.randn(10, IN)
    old.eval(), new.eval()
    torch.manual_seed(1)
    a = old(x)
    torch.manual_seed(1)
    b = new(x)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and torch.equal(a[3], b[3])


@pytest.mark.parametrize("bad", [dict(error_head=True), dict(router_entropy_bonus=0.1),
                                 dict(uncertainty_source="error_head", dense_experts=True),
                                 dict(escalation_rule="threshold", use_curiosity=False),
                                 dict(dense_experts=True, shared_trunk=True), dict(uncertainty_source="x")])
def test_invalid_combinations_are_refused(bad):
    with pytest.raises(ValueError):
        build(**bad)


def test_router_loss_optimum_is_the_gibbs_distribution():
    ce = torch.tensor([[0.2, 0.5, 1.0], [2.0, 0.1, 0.3]])
    lam = 0.3
    logits = (-ce / lam).clone().requires_grad_(True)
    expected_loss_router_loss(logits, ce, lam).backward()
    assert logits.grad.abs().max() < 1e-6
    # without the bonus the loss keeps falling as the router saturates
    sharp = (-ce / 0.01)
    assert expected_loss_router_loss(sharp, ce) < expected_loss_router_loss(-ce / lam, ce)


def test_dense_training_trains_every_expert_on_every_clip():
    m = build(dense_experts=True)
    m.train()
    x, y = torch.randn(16, IN), torch.randint(0, C, (16,))
    out, router_p, _, _ = m(x)
    logits = m._dense["expert_logits"]
    sel = router_p.argmax(1)
    assert torch.allclose(out, logits[sel, torch.arange(16)])
    crit = torch.nn.CrossEntropyLoss()
    loss = m.dense_loss(y, crit)
    expert_params = [p for e in m.experts for p in e.parameters()]
    g_dense = torch.autograd.grad(loss, expert_params, retain_graph=True, allow_unused=True)
    g_ref = torch.autograd.grad(sum(crit(l, y) for l in logits), expert_params, allow_unused=True)
    for a, b in zip(g_dense, g_ref):
        # the router loss sends nothing into the experts (expert losses detached)
        assert (a is None and b is None) or torch.allclose(a, b, atol=1e-7)
    g_router = torch.autograd.grad(m.dense_loss(y, crit), list(m.router.parameters()))
    assert any(g.abs().sum() > 0 for g in g_router)


def test_dense_eval_output_is_the_selected_experts_unscaled_logits():
    m = build(dense_experts=True)
    m.eval()
    x = torch.randn(20, IN)
    with torch.no_grad():
        out, router_p, _, _ = m(x)
        sel = router_p.argmax(1)
        ref = torch.stack([m.experts[int(s)](x[i:i + 1])[0] for i, s in enumerate(sel)])
    assert torch.allclose(out, ref, atol=1e-6)
    assert router_p.max(1).values.lt(1).any()  # so scaling would have shown


def test_error_head_is_the_uncertainty_and_its_loss_targets_the_cheap_expert():
    m = build(**DENSE_C)
    assert m.router.net[-1].out_features == 4
    m.eval()
    with torch.no_grad():
        _, _, _, u = m(torch.randn(8, IN))
    assert torch.equal(u, m.router.aux["error_prob"]) and ((u >= 0) & (u <= 1)).all()
    m.train()
    x, y = torch.randn(16, IN), torch.randint(0, C, (16,))
    m(x)
    crit = torch.nn.CrossEntropyLoss()
    with_head = m.dense_loss(y, crit)
    m.error_head = False
    without = m.dense_loss(y, crit)
    target = (m._dense["expert_logits"][0].argmax(1) != y).float()  # expert 0 is BitNet, the cheapest
    bce = F.binary_cross_entropy_with_logits(m.router.aux["error_logit"], target)
    assert torch.allclose(with_head - without, bce)


def test_escalation_routes_above_tau_to_the_highest_precision_expert():
    m = build(escalation_rule="threshold", **DENSE_C)
    m.eval()
    x = torch.randn(30, IN)
    torch.manual_seed(2)
    with torch.no_grad():
        _, p_off, _, u = m(x)
    tau = float(u.median())
    m.escalation_tau.fill_(tau)
    torch.manual_seed(2)
    with torch.no_grad():
        _, p_on, _, u2 = m(x)
    assert torch.equal(u, u2)
    esc = u > tau
    assert (p_on[esc].argmax(1) == 2).all()
    assert torch.equal(p_on[~esc], p_off[~esc])
    m.train()
    _, p_train, _, _ = m(x)
    assert p_train.max(1).values.lt(1).all()  # no escalation in training


def test_select_tau_uses_the_given_split_and_hits_the_target_share():
    m = build(escalation_rule="threshold", **DENSE_C)
    data = torch.utils.data.TensorDataset(torch.randn(50, IN), torch.randint(0, C, (50,)))
    loader = torch.utils.data.DataLoader(data, batch_size=16)
    tau, info = select_escalation_tau(m, loader, "cpu", seed=0, rule="target_share", target_share=0.3)
    assert float(m.escalation_tau) == tau
    assert abs(info["inner_val_escalation_share"] - 0.3) <= 0.02 + 1e-9
    tau2, info2 = select_escalation_tau(m, loader, "cpu", seed=0, rule="max_f1")
    assert info2["inner_val_f1_macro"] >= info2["inner_val_f1_macro_rule_off"]


@pytest.fixture(scope="module")
def dense_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("idea4")
    csv = tmp / "data.csv"
    synthetic_csv(csv)
    c = tiny_cfg(csv, tmp / "run", ["moe"])
    r = c.experiment.router
    r.dense_experts, r.router_entropy_bonus, r.error_head = True, 0.1, True
    r.uncertainty_source, r.escalation_rule, r.curiosity_alpha = "error_head", "threshold", 1.0
    bench.run_cv(str(csv), c)
    return tmp / "run" / "moe"


def test_benchmark_stores_tau_and_diagnostics_replay_the_scored_pass(dense_run, tmp_path):
    summary = json.loads((dense_run / "summary.json").read_text())
    for f in summary["folds"]:
        assert f["escalation"]["tau_rule"] == "target_share"
        ckpt = torch.load(dense_run / f"fold_{f['fold']}" / "best.pth")
        assert float(ckpt["escalation_tau"]) == pytest.approx(f["escalation"]["tau"])
    out = tmp_path / "diag.json"
    router_diagnostics.main(["--run", f"D={dense_run}", "--out", str(out)])
    d = json.loads(out.read_text())["D"]
    for row, f in zip(d["folds"], summary["folds"]):
        assert row["f1_macro"] == pytest.approx(f["f1_macro"])
        assert np.isclose(sum(row["shares"]), 1.0)
        assert len(row["hi_share_by_difficulty_decile"]) == 10
        assert {"auroc_rule_u_model_err", "auroc_mc_variance_cheap_err", "bops_rel_q4"} <= set(row)
        assert 0 < row["expert_bops_rel_q4"] < row["bops_rel_q4"]
        assert {"auroc_ref_model_conf_model_err", "auroc_ref_cheap_conf_cheap_err"} <= set(row)


def test_router_holdout_needs_dense_experts():
    with pytest.raises(ValueError):
        build(router_holdout=0.25)


def test_dense_loss_parts_split_expert_and_router_terms():
    m = build(**DENSE_C)
    m.train()
    x, y = torch.randn(16, IN), torch.randint(0, C, (16,))
    m(x)
    crit = torch.nn.CrossEntropyLoss()
    experts = m.dense_loss(y, crit, "experts")
    assert torch.allclose(experts, sum(crit(l, y) for l in m._dense["expert_logits"]))
    assert torch.allclose(experts + m.dense_loss(y, crit, "router"), m.dense_loss(y, crit))
    expert_params = [p for e in m.experts for p in e.parameters()]
    grads = torch.autograd.grad(m.dense_loss(y, crit, "router"), expert_params, allow_unused=True)
    assert all(g is None or g.abs().max() == 0 for g in grads)


def test_router_phase_trains_only_the_router():
    from QWave.moe import train_moe_local
    c = cfg(**DENSE_C)
    c.experiment.train = {"optimizer": "adamw", "lr": 1e-2, "weight_decay": 0.01, "label_smoothing": 0.0,
                          "class_weighted_loss": False, "batch_size": 8, "max_epochs": 1, "patience": 5,
                          "min_delta": 0.0, "lr_factor": 0.5, "lr_patience": 2, "grad_clip_norm": 1.0}
    torch.manual_seed(0)
    m = qMoEModelBatched(c, IN, C, 3, 1)
    data = torch.utils.data.TensorDataset(torch.randn(24, IN), torch.randint(0, C, (24,)))
    loader = torch.utils.data.DataLoader(data, batch_size=8)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    train_moe_local(c, True, m, [], loader, None, IN, "cpu", ".", False, "/dev/null", router_loader=loader)
    after = m.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before if k.startswith("experts."))
    assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith("router."))


@pytest.fixture(scope="module")
def holdout_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("idea4h")
    csv = tmp / "data.csv"
    synthetic_csv(csv)
    c = tiny_cfg(csv, tmp / "run", ["moe"])
    r = c.experiment.router
    r.dense_experts, r.router_entropy_bonus, r.error_head, r.router_holdout = True, 0.1, True, 0.25
    r.uncertainty_source, r.escalation_rule = "error_head", "threshold"
    bench.run_cv(str(csv), c)
    return tmp / "run" / "moe"


def test_router_split_is_carved_from_the_fit_split_only(holdout_run, tmp_path):
    summary = json.loads((holdout_run / "summary.json").read_text())
    for f in summary["folds"]:
        split = json.loads((holdout_run / f"fold_{f['fold']}" / "split.json").read_text())
        router, test, iv = (set(split[k]) for k in ("router_idx", "test_idx", "inner_val_idx"))
        assert router and router.isdisjoint(test) and router.isdisjoint(iv)
        assert f["n_fit"] * 0.2 <= len(router) <= f["n_fit"] * 0.3
    out = tmp_path / "diag.json"
    router_diagnostics.main(["--run", f"D={holdout_run}", "--run", f"C={holdout_run}#no_escalation",
                             "--out", str(out)])
    d = json.loads(out.read_text())
    assert d["C"]["view"] == "no_escalation" and d["C"]["folds"][0]["f1_macro_scored"] is None
    for row, f in zip(d["D"]["folds"], summary["folds"]):
        assert row["f1_macro"] == pytest.approx(f["f1_macro"])


def test_deciles_split_by_rank():
    d = router_diagnostics.deciles(np.arange(100)[::-1].astype(float))
    assert d[0] == 9 and d[-1] == 0 and np.all(np.bincount(d) == 10)
