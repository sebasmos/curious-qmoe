"""Protocol 3: the inner validation split grouped by source recording
(cross_validation.inner_split_groups=recording), and protocol 2 unchanged
with the flag off.

Run: PYTHONPATH=. pytest tests/test_protocol_v3.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from QWave.datasets import inner_split, recording_groups
import scripts.benchmark as bench
from test_protocol import N_FOLDS, synthetic_csv, tiny_cfg


def grouped_data(n_rec=60, seed=0):
    """Recordings of 1 to 8 rows, one class each, 4 classes."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 9, size=n_rec)
    groups = np.repeat(np.arange(n_rec), sizes)
    labels = np.repeat(np.arange(n_rec) % 4, sizes)
    return labels, groups


# ── recording ids ──────────────────────────────────────────────────────────

def test_recording_ids_from_names():
    esc = ["1-100038-A-14.png", "1-100038-B-14.png", "2-9032-A-0.png"]
    assert recording_groups(esc, 1).tolist() == ["100038", "100038", "9032"]
    urban = ["100032-3-0-0.png", "100263-2-0-117.png", "100263-2-0-121.png"]
    assert recording_groups(urban, 0).tolist() == ["100032", "100263", "100263"]
    assert recording_groups(["0_AAT-FS_0.png"], None) is None
    with pytest.raises(ValueError):
        recording_groups(["0_AATFS_0.png"], 1)


def test_dataset_configs_default_to_protocol_2():
    tokens = {}
    for name in ("esc50", "urban8k", "quinn"):
        cv = OmegaConf.load(ROOT / "config" / f"{name}.yaml").experiment.cross_validation
        assert cv.inner_split_groups == "none"
        tokens[name] = cv.recording_token
    assert tokens == {"esc50": 1, "urban8k": 0, "quinn": None}


# ── the grouped split ──────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", [0, 1, 7])
def test_no_recording_on_both_sides_and_every_class_fit(seed):
    labels, groups = grouped_data()
    train_idx = np.flatnonzero(groups >= 5)  # rows of recordings 0-4 play the test fold
    fit, val = inner_split(train_idx, labels, 0.1, seed, groups)
    assert set(fit).isdisjoint(val) and set(fit) | set(val) == set(train_idx)
    assert set(groups[fit]).isdisjoint(groups[val])
    assert set(labels[fit]) == set(labels[train_idx])
    assert 0 < len(val) < len(train_idx) / 4


def test_grouped_split_is_deterministic_per_seed():
    labels, groups = grouped_data()
    train_idx = np.arange(len(labels))
    a = inner_split(train_idx, labels, 0.1, 3, groups)
    b = inner_split(train_idx, labels, 0.1, 3, groups)
    c = inner_split(train_idx, labels, 0.1, 4, groups)
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[0], b[0])
    assert not np.array_equal(a[1], c[1])


def test_first_split_that_keeps_every_class_in_fit():
    # class 3 lives in one recording only: the split whose validation part
    # holds it must be skipped for the next one
    labels, groups = grouped_data(n_rec=40)
    groups = groups.copy()
    labels = labels.copy()
    lone = groups == 3
    labels[labels == 3] = 2
    labels[lone] = 3
    train_idx = np.arange(len(labels))
    for seed in range(20):
        sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
        splits = list(sgkf.split(np.zeros(len(labels)), labels, groups))
        if 3 in labels[splits[0][1]]:
            break
    else:
        pytest.fail("no seed puts the lone recording in the first validation fold")
    expected = next(v for _, v in splits if 3 not in labels[v])
    fit, val = inner_split(train_idx, labels, 0.1, seed, groups)
    assert np.array_equal(val, np.sort(expected))
    assert 3 in labels[fit]


def test_groups_none_is_the_protocol_2_split():
    labels = np.arange(200) % 4
    train_idx = np.arange(40, 200)
    fit, val = inner_split(train_idx, labels, 0.1, 5)
    ref_fit, ref_val = train_test_split(train_idx, test_size=0.1, stratify=labels[train_idx], random_state=5)
    assert np.array_equal(fit, np.sort(ref_fit)) and np.array_equal(val, np.sort(ref_val))
    fit2, val2 = inner_split(train_idx, labels, 0.1, 5, groups=None)
    assert np.array_equal(fit, fit2) and np.array_equal(val, val2)


# ── end to end through run_cv ──────────────────────────────────────────────

def named_csv(path):
    """synthetic_csv plus ESC-50 style names: recordings of 3 rows, one class
    and one fold each."""
    df = synthetic_csv(path)
    i = np.arange(len(df))
    rec = [f"{b}{f}{c}" for b, f, c in zip(i // 27, df.folder, df.class_id)]
    df.insert(1, "name", [f"{f}-{r}-A-{c}.png" for f, r, c in zip(df.folder, rec, df.class_id)])
    df.to_csv(path, index=False)
    return df


@pytest.fixture(scope="module")
def v3_runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("protocol_v3")
    csv = tmp / "data.csv"
    df = named_csv(csv)
    out = {}
    for name, groups, token in (("off", "none", 1), ("v3", "recording", 1), ("nogroups", "recording", None)):
        cfg = tiny_cfg(csv, tmp / name, ["4"])
        cfg.experiment.train.max_epochs = 1
        cfg.experiment.cross_validation.inner_split_groups = groups
        cfg.experiment.cross_validation.recording_token = token
        bench.run_cv(str(csv), cfg)
        out[name] = tmp / name / "4"
    return df, out


def splits(run_dir):
    return [json.loads((run_dir / f"fold_{f}" / "split.json").read_text()) for f in range(1, N_FOLDS + 1)]


def test_run_cv_v3_keeps_recordings_on_one_side(v3_runs):
    df, out = v3_runs
    rec = recording_groups(df["name"], 1)
    s = json.loads((out["v3"] / "summary.json").read_text())
    assert s["protocol_version"] == 3
    assert s["inner_split"] == {"groups": "recording", "recording_token": 1, "grouped": True}
    assert all(f["protocol_version"] == 3 for f in s["folds"])
    for sp in splits(out["v3"]):
        test, val = np.asarray(sp["test_idx"]), np.asarray(sp["inner_val_idx"])
        fit = np.setdiff1d(np.arange(len(df)), np.concatenate([test, val]))
        assert set(rec[fit]).isdisjoint(rec[val])
        assert set(df.class_id[fit]) == set(df.class_id)
        assert sp["inner_split"]["grouped"]


def test_run_cv_flag_off_is_protocol_2(v3_runs):
    df, out = v3_runs
    s = json.loads((out["off"] / "summary.json").read_text())
    assert s["protocol_version"] == 2 and "inner_split" not in s
    assert all("protocol_version" not in f and "inner_split" not in f for f in s["folds"])
    labels = df.class_id.to_numpy()
    for sp in splits(out["off"]):
        assert set(sp) == {"fold", "seed", "fold_scheme", "test_idx", "inner_val_idx"}
        train_idx = np.setdiff1d(np.arange(len(df)), sp["test_idx"])
        _, val = inner_split(train_idx, labels, 0.1, 0)
        assert sp["inner_val_idx"] == val.tolist()


def test_run_cv_without_recording_ids_falls_back_to_stratified(v3_runs):
    _, out = v3_runs
    s = json.loads((out["nogroups"] / "summary.json").read_text())
    assert s["protocol_version"] == 3
    assert s["inner_split"] == {"groups": "recording", "recording_token": None, "grouped": False}
    for a, b in zip(splits(out["nogroups"]), splits(out["off"])):
        assert a["inner_val_idx"] == b["inner_val_idx"], "the fallback is the protocol 2 split"


def test_unknown_grouping_is_rejected(tmp_path):
    csv = tmp_path / "data.csv"
    named_csv(csv)
    cfg = tiny_cfg(csv, tmp_path / "bad", ["4"])
    cfg.experiment.cross_validation.inner_split_groups = "site"
    with pytest.raises(ValueError):
        bench.run_cv(str(csv), cfg)
