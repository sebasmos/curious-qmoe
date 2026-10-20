"""Reviewer controls: the mixed-precision single model, parameter-matched
single models, and the input dimension and label column taken from the CSV.

Run: PYTHONPATH=. pytest tests/test_extra_models.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.benchmark as bench
from QWave.memory import storage_bits
from QWave.moe import qMoEModelBatched
from QWave.qmoe_layers import BitLinear, BitNetExpert, SharedTrunkMoE
from QWave.utils import get_num_parameters
from scripts.analysis.run_loader import load_trained_fold
from test_protocol import tiny_cfg


def cfg_e3(hidden=(640, 320)):
    cfg = OmegaConf.load(ROOT / "config" / "esc50.yaml")
    cfg.experiment.model.hidden_sizes = list(hidden)
    r = cfg.experiment.router
    r.expert_quantizations, r.num_experts, r.top_k, r.use_curiosity = ["bitnet", "4", "8"], 3, 1, True
    return cfg


def bitlinears(model):
    return [m for m in model.modules() if isinstance(m, BitLinear)]


# ── mixed precision ────────────────────────────────────────────────────────

@pytest.mark.parametrize("hidden, bits", [((640, 320), [8, 4]), ((64, 32, 16), [8, 4, 4])])
def test_mp8_4_first_layer_8_bit_rest_4_bit(hidden, bits):
    model = bench.build_model("mp8_4", 1536, 5, cfg_e3(hidden))
    assert [m.num_bits for m in bitlinears(model)] == bits
    assert isinstance(model.net[-1], torch.nn.Linear), "fp32 classifier, as in the k-bit models"
    rows = [(m.weight.shape[0], m.weight.numel(), m.num_bits) for m in bitlinears(model)]
    weights = sum(nb * n + 32 * r for r, n, nb in rows)
    other = sum(p.numel() for p in model.parameters()) - sum(m.weight.numel() for m in bitlinears(model))
    assert storage_bits(model) == weights + 32 * other


def test_mp8_4_matches_the_shared_trunk_path_of_the_4_bit_expert():
    single = bench.build_model("mp8_4", 1536, 5, cfg_e3())
    st = SharedTrunkMoE(1536, 5, [640, 320], 0.1, expert_bits=["bitnet", "4", "8"], trunk_bits=8)
    path = bitlinears(st.trunk) + bitlinears(st.heads[1])
    assert [(m.in_features, m.out_features, m.num_bits) for m in bitlinears(single)] == \
        [(m.in_features, m.out_features, m.num_bits) for m in path]


def test_layer_bits_must_cover_every_hidden_layer():
    with pytest.raises(ValueError):
        BitNetExpert(16, 3, [8, 8], 0.1, num_bits=4, layer_bits=[8])
    plain = BitNetExpert(16, 3, [8, 8], 0.1, num_bits=4)
    assert [m.num_bits for m in bitlinears(plain)] == [4, 4], "default unchanged"


# ── parameter-matched singles ──────────────────────────────────────────────

@pytest.mark.parametrize("in_dim, n_cls", [(1536, 5), (1536, 10), (768, 5), (768, 50)])
@pytest.mark.parametrize("kind", sorted(bench.PARAM_MATCHED))
def test_param_matched_count_within_2_percent_of_the_moe(kind, in_dim, n_cls):
    cfg = cfg_e3()
    bits, shared = bench.PARAM_MATCHED[kind]
    ref_cfg = cfg_e3()
    ref_cfg.experiment.router.shared_trunk = shared
    target = get_num_parameters(qMoEModelBatched(ref_cfg, in_dim, n_cls, 3, 1))
    model = bench.build_model(kind, in_dim, n_cls, cfg)
    n = get_num_parameters(model)
    assert abs(n - target) / target < 0.02
    m = model.param_match
    assert m["reference_parameter_count"] == target and m["parameter_count"] == n
    assert m["reference"] == ("shared_trunk" if shared else "moe")
    assert {mod.num_bits for mod in bitlinears(model)} == {bits}
    h0, h1 = m["hidden_sizes"]
    assert h0 > 640 and abs(h0 - 2 * h1) <= 1, "widened, 2:1 ratio of the base widths kept"


def test_width_search_draws_nothing_from_the_rng():
    state = torch.random.get_rng_state()
    bench.param_matched_widths(cfg_e3(), 1536, 5, 8, True)
    assert torch.equal(state, torch.random.get_rng_state())


def test_full_and_shared_trunk_references_differ():
    full, _ = bench.param_matched_widths(cfg_e3(), 1536, 5, 8, False)
    shared, _ = bench.param_matched_widths(cfg_e3(), 1536, 5, 8, True)
    assert full[0] > shared[0] > 640


# ── input dimension and label column from the CSV ──────────────────────────

N_FEAT, N_ROWS, N_FINE, N_FOLDS = 24, 180, 6, 3


@pytest.fixture(scope="module")
def run768(tmp_path_factory):
    """ESC-50-shaped CSV with 24 feature columns and 6 fine labels over 3
    super-categories, trained on the fine labels (label_col=label)."""
    tmp = tmp_path_factory.mktemp("indim")
    rng = np.random.default_rng(0)
    fine = np.arange(N_ROWS) % N_FINE
    X = rng.normal(size=(N_ROWS, N_FEAT)) + fine[:, None] * 0.5
    df = pd.DataFrame(X, columns=[str(i) for i in range(N_FEAT)])
    df.insert(0, "folder", (np.arange(N_ROWS) // N_FINE) % N_FOLDS + 1)
    df.insert(1, "name", [f"{i}.png" for i in range(N_ROWS)])
    df.insert(2, "label", fine)
    df.insert(3, "category", "x")
    df.insert(4, "class_id", fine // 2)
    csv = tmp / "data.csv"
    df.to_csv(csv, index=False)
    cfg = tiny_cfg(csv, tmp / "run", ["mp8_4", "pm8_st", "moe"])
    cfg.experiment.datasets.esc.label_col = "label"
    cfg.experiment.router.shared_trunk = True
    cfg.experiment.train.max_epochs = 2
    bench.run_cv(str(csv), cfg)
    return tmp / "run"


@pytest.mark.parametrize("kind", ["mp8_4", "pm8_st"])
def test_single_input_and_output_dims_come_from_the_csv(run768, kind):
    # net: BitLinear, LayerNorm, ReLU, Dropout, BitLinear, ..., classifier at net.8
    state = torch.load(run768 / kind / "fold_1" / "best_model.pth", map_location="cpu")
    assert state["net.0.weight"].shape[1] == N_FEAT
    assert state["net.0.act_ln.weight"].shape == (N_FEAT,)
    assert state["net.8.weight"].shape[0] == N_FINE, "label_col=label trains on the fine classes"


def test_moe_router_and_trunk_dims_come_from_the_csv(run768):
    state = torch.load(run768 / "moe" / "fold_1" / "best.pth", map_location="cpu")
    assert state["router.net.0.weight"].shape[1] == N_FEAT
    assert state["shared.trunk.0.weight"].shape[1] == N_FEAT
    # each head: BitLinear, LayerNorm, ReLU, Dropout, classifier at index 4
    assert all(state[f"shared.heads.{i}.4.weight"].shape[0] == N_FINE for i in range(3))


def test_param_match_is_recorded_and_the_run_reloads(run768):
    s = json.loads((run768 / "pm8_st" / "summary.json").read_text())
    m = s["param_match"]
    assert m["parameter_count"] == s["parameter_count"] and m["reference"] == "shared_trunk"
    assert abs(m["relative_gap"]) < 0.02
    assert "param_match" not in json.loads((run768 / "mp8_4" / "summary.json").read_text())
    for kind in ("mp8_4", "pm8_st", "moe"):
        tf = load_trained_fold(run768 / kind, 1)
        assert tf.X_test.shape[1] == N_FEAT
        out = tf.model(tf.X_test[:4])
        assert (out[0] if kind == "moe" else out).shape == (4, N_FINE)
