"""The int8 scaler rule (QWave/deploy.py): a head whose StandardScaler was fit on
FP32 features collapses on real int8 features because near-constant dimensions
shift by many FP32 standard deviations, and refitting only the scaler on the
deployed features recovers it with the head weights untouched.

Run: PYTHONPATH=. pytest tests/test_deploy_scaler.py -q
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from QWave.deploy import predict, refit_scaler, shifted_dims  # noqa: E402

N, D, C = 900, 24, 3
NEAR_CONSTANT_DIM, NC_MEAN, NC_STD = 23, -0.2300, 1e-4
INT8_SCALE = 0.0098  # per-tensor grid step of the deployed backbone's output


def synthetic_fp32(seed=0):
    """FP32 embeddings: eight informative dims, fifteen noise dims, and one dim whose whole
    range (std 1e-4) is far below the int8 grid step but carries a little class information."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, C, N)
    X = rng.normal(0.0, 1.0, (N, D)).astype(np.float32)
    X[:, 8:23] *= 0.5
    means = rng.normal(0.0, 1.5, (C, 8))
    X[:, :8] += means[y]
    X[:, NEAR_CONSTANT_DIM] = NC_MEAN + NC_STD * (rng.normal(0.0, 1.0, N) + 0.5 * (y - 1))
    return X, y


def int8_grid(X):
    """Static per-tensor int8: every value snaps to the nearest grid point."""
    return (np.round(X / INT8_SCALE) * INT8_SCALE).astype(np.float32)


def train_head(Z, y, seed=0):
    torch.manual_seed(seed)
    head = nn.Sequential(nn.Linear(D, 32), nn.ReLU(), nn.Linear(32, C))
    opt = torch.optim.Adam(head.parameters(), lr=0.02)
    Zt, yt = torch.tensor(Z, dtype=torch.float32), torch.tensor(y)
    for _ in range(300):
        opt.zero_grad()
        nn.functional.cross_entropy(head(Zt), yt).backward()
        opt.step()
    return head.eval()


def test_fp32_scaler_collapses_on_int8_and_refit_recovers():
    X, y = synthetic_fp32()
    fit, test = np.arange(N) < 600, np.arange(N) >= 600
    scaler_fp32 = StandardScaler().fit(X[fit])
    head = train_head(scaler_fp32.transform(X[fit]), y[fit])
    weights_before = [p.detach().clone() for p in head.parameters()]

    f1 = lambda scaler, feats: f1_score(y[test], predict(head, scaler, feats[test]), average="macro")
    f1_fp32 = f1(scaler_fp32, X)
    assert f1_fp32 > 0.9

    X8 = int8_grid(X)
    # the near-constant dimension is the only one the FP32 scaler blows up
    assert np.abs(X8[:, NEAR_CONSTANT_DIM] - X[:, NEAR_CONSTANT_DIM]).max() < INT8_SCALE
    dims = shifted_dims(scaler_fp32, X8, threshold_sd=5.0)
    assert dims.tolist() == [NEAR_CONSTANT_DIM]
    shift_sd = abs(X8[:, NEAR_CONSTANT_DIM].mean() - scaler_fp32.mean_[NEAR_CONSTANT_DIM]) / scaler_fp32.scale_[NEAR_CONSTANT_DIM]
    assert shift_sd > 20

    # collapse: the FP32 scaler on int8 features
    f1_int8_fp32_scaler = f1(scaler_fp32, X8)
    assert f1_int8_fp32_scaler < 0.5 * f1_fp32

    # recovery: refit only the scaler on the deployed fit rows, head unchanged
    scaler_int8, head_out = refit_scaler(X8[fit], head)
    assert head_out is head
    assert all(torch.equal(a, b) for a, b in zip(weights_before, head.parameters()))
    assert shifted_dims(scaler_int8, X8, threshold_sd=1.0).size == 0
    f1_int8_refit = f1(scaler_int8, X8)
    assert f1_int8_refit > f1_fp32 - 0.05
    assert f1_int8_refit > 1.5 * f1_int8_fp32_scaler


def test_refit_scaler_rejects_bad_input():
    import pytest
    head = nn.Linear(D, C)
    with pytest.raises(ValueError):
        refit_scaler(np.zeros(D), head)
    with pytest.raises(ValueError):
        refit_scaler(np.zeros((1, D)), head)
