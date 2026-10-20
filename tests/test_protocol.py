"""Evaluation protocol: predefined folds, inner validation split, seeding, and
evaluation of the inner-validation-selected checkpoint.

Run: PYTHONPATH=. pytest tests/test_protocol.py -q
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

from QWave.datasets import make_outer_folds, inner_split
from QWave.utils import set_seed
from QWave import train_utils
import scripts.benchmark as bench

N_FEAT, N_ROWS, N_CLASSES, N_FOLDS = 16, 150, 3, 3


def synthetic_csv(path):
    """Separable 3-class data with 3 balanced predefined folds. Feature "0"
    holds the row id, so loaders can be traced back to CSV rows."""
    rng = np.random.default_rng(0)
    y = np.arange(N_ROWS) % N_CLASSES
    X = rng.normal(size=(N_ROWS, N_FEAT)) + y[:, None] * 0.8
    X[:, 0] = np.arange(N_ROWS)
    df = pd.DataFrame(X, columns=[str(i) for i in range(N_FEAT)])
    df.insert(0, "folder", (np.arange(N_ROWS) // N_CLASSES) % N_FOLDS + 1)
    df["class_id"] = y
    df.to_csv(path, index=False)
    return df


def tiny_cfg(csv, tag, models, seed=0):
    cfg = OmegaConf.load(ROOT / "config" / "esc50.yaml")
    e = cfg.experiment
    e.datasets.esc.csv = str(csv)
    e.device = "cpu"
    e.models_to_run = models
    e.seed = seed
    e.cross_validation.n_splits = N_FOLDS
    e.cross_validation.fold_column = "folder"
    e.model.hidden_sizes = [8, 8]
    e.model.epochs = 3
    e.router.expert_quantizations = ["bitnet", "4", "8"]
    e.router.num_experts = 3
    e.router.top_k = 1
    e.router.hidden_dim = 8
    e.router.use_curiosity = True
    e.router.mc_samples = 3
    e.metadata.tag = str(tag)
    return cfg


# ── folds and inner split ──────────────────────────────────────────────────

def test_predefined_folds_are_leave_one_fold_out():
    df = pd.DataFrame({"folder": [1, 2, 3, 1, 2, 3, 1, 2, 3], "x": range(9)})
    cv = OmegaConf.create({"n_splits": 3, "fold_column": "folder"})
    folds = make_outer_folds(df, np.zeros(9), cv)
    assert [f for f, _, _ in folds] == [1, 2, 3]
    for f, tr, te in folds:
        assert set(te) == set(np.flatnonzero(df.folder == f))
        assert set(tr) == set(range(9)) - set(te)


def test_predefined_fold_count_must_match_n_splits():
    df = pd.DataFrame({"folder": [1, 2, 1, 2]})
    with pytest.raises(ValueError):
        make_outer_folds(df, np.zeros(4), OmegaConf.create({"n_splits": 5, "fold_column": "folder"}))


def test_stratified_folds_do_not_depend_on_run_seed():
    labels = np.arange(100) % 4
    cv = OmegaConf.create({"n_splits": 5, "fold_column": None, "random_seed": 42})
    set_seed(0)
    a = make_outer_folds(pd.DataFrame(index=range(100)), labels, cv)
    set_seed(7)
    np.random.rand(10)
    b = make_outer_folds(pd.DataFrame(index=range(100)), labels, cv)
    for (fa, tra, tea), (fb, trb, teb) in zip(a, b):
        assert fa == fb and np.array_equal(tea, teb) and np.array_equal(tra, trb)


def test_inner_split_partitions_training_folds():
    labels = np.arange(200) % 4
    train_idx = np.arange(40, 200)
    fit, val = inner_split(train_idx, labels, 0.1, seed=0)
    assert set(fit).isdisjoint(val)
    assert set(fit) | set(val) == set(train_idx)
    assert len(val) == 16
    assert np.bincount(labels[val]).tolist() == [4, 4, 4, 4]
    fit2, val2 = inner_split(train_idx, labels, 0.1, seed=0)
    assert np.array_equal(val, val2)
    _, val3 = inner_split(train_idx, labels, 0.1, seed=1)
    assert not np.array_equal(val, val3)


# ── the trainer hands back the selected epoch, not the last one ─────────────

def test_train_pytorch_local_returns_selected_epoch(tmp_path, monkeypatch):
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    X, y = torch.randn(32, 4), torch.arange(32) % 2
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=8)
    states = []
    scores = iter([0.9, 0.1, 0.2, 0.3, 0.4])

    def fake_validate(m, loader, criterion, device, average="macro"):
        states.append({k: v.clone() for k, v in m.state_dict().items()})
        return 0.0, next(scores), [0], [0], [[1.0, 0.0]]

    monkeypatch.setattr(train_utils, "_validate_single_epoch", fake_validate)
    args = tiny_cfg(tmp_path / "x.csv", tmp_path / "x", ["4"]).experiment
    args.model.epochs = 5
    returned, *_ = train_utils.train_pytorch_local(args, model, loader, loader, torch.ones(2), 4,
                                                   "cpu", str(tmp_path))
    for k, v in returned.state_dict().items():
        assert torch.equal(v, states[0][k]), "must return the epoch-1 (best) weights"
    assert any(not torch.equal(states[-1][k], states[0][k]) for k in states[0])


# ── end to end through run_cv ──────────────────────────────────────────────

class _Recorder:
    def __init__(self):
        self.datasets, self.train_calls, self.evaluated = [], [], []


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """run_cv three times (seed 0 twice, seed 1 once) on the synthetic CSV,
    recording which rows reach training, selection and the final test pass."""
    tmp = tmp_path_factory.mktemp("protocol")
    csv = tmp / "data.csv"
    df = synthetic_csv(csv)
    mp = pytest.MonkeyPatch()
    rec = _Recorder()

    class TracingDataset(bench.EmbeddingAdaptDataset):
        def __init__(self, frame, *a, **k):
            self.row_ids = frame["0"].to_numpy().astype(int)
            super().__init__(frame, *a, **k)
            rec.datasets.append(self)

    def wrap_trainer(fn, tr_pos, val_pos):
        def wrapped(*args):
            rec.train_calls.append((args[tr_pos].dataset.row_ids, args[val_pos].dataset.row_ids))
            return fn(*args)
        return wrapped

    def wrap_eval(fn):
        def wrapped(model, loader, *args):
            rec.evaluated.append(({k: v.clone() for k, v in model.state_dict().items()},
                                  loader.dataset.row_ids))
            return fn(model, loader, *args)
        return wrapped

    mp.setattr(bench, "EmbeddingAdaptDataset", TracingDataset)
    mp.setattr(bench, "train_pytorch_local", wrap_trainer(bench.train_pytorch_local, 2, 3))
    mp.setattr(bench, "train_moe_local", wrap_trainer(bench.train_moe_local, 3, 4))
    mp.setattr(bench, "_validate_single_epoch", wrap_eval(bench._validate_single_epoch))
    mp.setattr(bench, "_validate_moe_epoch", wrap_eval(bench._validate_moe_epoch))

    out = {}
    for name, seed in (("a", 0), ("b", 0), ("c", 1)):
        start = len(rec.evaluated)
        bench.run_cv(str(csv), tiny_cfg(csv, tmp / name, ["4", "moe"], seed=seed))
        out[name] = (tmp / name, rec.evaluated[start:])
    mp.undo()
    return df, rec, out


def test_test_fold_never_reaches_training_or_selection(runs):
    df, rec, out = runs
    run_dir, _ = out["a"]
    calls = rec.train_calls[: 2 * N_FOLDS]  # run "a": model '4' then 'moe'
    for i, (fit_ids, val_ids) in enumerate(calls):
        fold = i % N_FOLDS + 1
        kind = "4" if i < N_FOLDS else "moe"
        split = json.loads((run_dir / kind / f"fold_{fold}" / "split.json").read_text())
        test = set(split["test_idx"])
        assert test == set(np.flatnonzero(df.folder == fold)), "test fold must be the predefined fold"
        assert set(val_ids) == set(split["inner_val_idx"])
        assert test.isdisjoint(fit_ids) and test.isdisjoint(val_ids) and set(fit_ids).isdisjoint(val_ids)
        assert set(fit_ids) | set(val_ids) | test == set(range(N_ROWS))


def test_scaler_is_fit_on_training_portion_only(runs):
    df, rec, _ = runs
    feats = df.drop(columns=["folder", "class_id"]).to_numpy(dtype=np.float32)
    # each fold builds (fit, inner val, test) in that order; run "a" has 2 models x 3 folds
    triples = [rec.datasets[i:i + 3] for i in range(0, 3 * 2 * N_FOLDS, 3)]
    for (fit_ds, iv_ds, te_ds), (fit_ids, _) in zip(triples, rec.train_calls):
        assert np.array_equal(np.sort(fit_ds.row_ids), np.sort(fit_ids))
        np.testing.assert_allclose(fit_ds.scaler.mean_, feats[fit_ds.row_ids].mean(axis=0), rtol=1e-5)
        assert iv_ds.scaler is fit_ds.scaler and te_ds.scaler is fit_ds.scaler


def test_evaluated_weights_are_the_selected_checkpoint(runs):
    _, _, out = runs
    run_dir, evaluated = out["a"]
    for i, (state, rows) in enumerate(evaluated):
        fold = i % N_FOLDS + 1
        kind = "4" if i < N_FOLDS else "moe"
        ckpt = torch.load(run_dir / kind / f"fold_{fold}" / bench.checkpoint_name(kind))
        for k, v in ckpt.items():
            assert torch.equal(state[k], v), f"{kind} fold {fold}: evaluated weights are not the saved selection"
        split = json.loads((run_dir / kind / f"fold_{fold}" / "split.json").read_text())
        assert set(rows) == set(split["test_idx"])


def test_same_seed_gives_identical_fold_metrics(runs):
    _, _, out = runs
    for kind in ("4", "moe"):
        a = json.loads((out["a"][0] / kind / "summary.json").read_text())
        b = json.loads((out["b"][0] / kind / "summary.json").read_text())
        c = json.loads((out["c"][0] / kind / "summary.json").read_text())
        assert a["seed"] == 0 and c["seed"] == 1
        for fa, fb in zip(a["folds"], b["folds"]):
            assert (fa["fold"], fa["seed"]) == (fb["fold"], fb["seed"])
            for key in ("f1_macro", "f1_weighted", "accuracy", "inner_val_f1_macro"):
                assert fa[key] == fb[key], f"{kind} fold {fa['fold']} {key} differs under the same seed"
        sa = json.loads((out["a"][0] / kind / "fold_1" / "split.json").read_text())
        sc = json.loads((out["c"][0] / kind / "fold_1" / "split.json").read_text())
        assert sa["test_idx"] == sc["test_idx"], "test folds must not move with the seed"
        assert sa["inner_val_idx"] != sc["inner_val_idx"], "the inner split follows the seed"


def test_summary_carries_per_fold_pairing_keys(runs):
    _, _, out = runs
    s = json.loads((out["a"][0] / "moe" / "summary.json").read_text())
    assert s["protocol_version"] == bench.PROTOCOL_VERSION
    assert s["primary_metric"] == "f1_macro"
    assert s["fold_scheme"] == {"type": "predefined", "column": "folder"}
    assert [f["fold"] for f in s["folds"]] == [1, 2, 3]
    assert all({"seed", "f1_macro", "f1_weighted", "accuracy"} <= set(f) for f in s["folds"])
