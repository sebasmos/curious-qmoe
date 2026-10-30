"""Tests for idea 6 Stage 3c: the DyMN rebuild (QWave/efficientat_dymn.py), its W8A8 copy and cost counters, the
new MN widths, the test-only inner-validation score directory and the fold-level CI of the analysis, and
benchmark.py's only_folds option. CPU only, random weights, no download."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from QWave import efficientat_dymn as D
from QWave import efficientat_mn as M

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "redesign" / "idea6" / "stage3c"))


@pytest.fixture(scope="module")
def dymn04():
    torch.manual_seed(0)
    m = D.build_dymn("dymn04")
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.1, 0.1)
            mod.running_var.uniform_(0.5, 1.5)
        if isinstance(mod, D.DynamicConv):
            nn.init.normal_(mod.weight, 0, 0.05)
    return m.eval()


def test_mn_widths_and_state_dict_sizes():
    # mn05_as_mAP_443.pt and mn20_as_mAP_478.pt: 312 entries, these parameter counts (checked against the release)
    for name, emb, n_par in (("mn05", 480, 1443637), ("mn20", 1920, 17958133)):
        m = M.build_mn(name)
        assert len(m.state_dict()) == 312 and m.emb_dim == emb
        assert sum(v.numel() for v in m.state_dict().values()) == n_par


def test_dymn_state_dict_matches_the_release(dymn04):
    # dymn04_as.pt: 578 entries, 1,972,124 parameters, 384-d embedding
    sd = dymn04.state_dict()
    assert len(sd) == 578 and dymn04.emb_dim == 384
    assert sum(v.numel() for v in sd.values()) == 1972124
    assert tuple(sd["layers.1.exp_conv.weight"].shape) == (1, 1, 4, 24 * 8)
    assert "layers.0.depth_act.lambdas" in sd and "layers.0.exp_conv.weight" not in sd  # block 0 has no expansion


def test_dymn_embedding_shape_any_length(dymn04):
    for frames in (32, 400, 501):
        with torch.no_grad():
            e = dymn04(torch.randn(2, 1, 128, frames))
        assert e.shape == (2, 384) and torch.isfinite(e).all()


def test_dymn_batch_independent(dymn04):
    """Each clip gets its own aggregated kernels: a clip's embedding does not depend on its batch mates."""
    x = torch.randn(3, 1, 128, 64)
    with torch.no_grad():
        a = dymn04(x)
        b = torch.cat([dymn04(x[i:i + 1]) for i in range(3)])
    assert torch.allclose(a, b, atol=1e-4)


def test_dymn_w8a8_quantizes_static_and_dynamic_kernels(dymn04):
    q = D.build_backbone("w8a8", dymn04)
    assert q.n_conv > 0 and q.n_linear > 0 and q.n_dynamic == 15 * 3 - 1  # block 0 has no expand conv
    for _, mod in D._feature_modules(q.net):
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            w = mod.weight.detach()
            s = w.abs().reshape(w.shape[0], -1).amax(1).clamp_min(1e-12) / 127
            codes = w.reshape(w.shape[0], -1) / s[:, None]
            assert torch.allclose(codes, codes.round(), atol=1e-3)
        if isinstance(mod, D.DynamicConv):
            assert mod.quant_bits == 8
    x = torch.randn(2, 1, 128, 100)
    with torch.no_grad():
        a, b = dymn04(x), q(x)
    rel = (a - b).norm(dim=1) / a.norm(dim=1)
    assert float(rel.mean()) < 0.3
    with pytest.raises(ValueError):
        D.build_backbone("w4a4", dymn04)


def test_dymn_costs(dymn04):
    c = D.feature_macs(dymn04, 64)
    assert c["aggregate_macs"] > 0 and c["parts_macs"]["dyn_conv"] > 0 and c["parts_macs"]["dyn_depthwise"] > 0
    c2 = D.feature_macs(dymn04, 128)
    assert 1.7 < c2["macs"] / c["macs"] < 2.1  # convolutions scale with time; the linear context layers do not
    assert c2["aggregate_macs"] == c["aggregate_macs"]  # one aggregation per clip, whatever its length
    assert D.backbone_bitops(c, "fp32") == c["macs"] * 1024 + c["aggregate_macs"] * 1024
    assert D.backbone_bitops(c, "w8a8", 1024.0) == c["macs"] * 64 + c["out_elems"] * 1024 + c["aggregate_macs"] * 1024
    assert D.weight_bits_stored(dymn04, "w8a8") < D.weight_bits_stored(dymn04, "fp32") / 3


def _write_scores(d: Path, cfg, cond, n=12, k=3, iv_nan=False):
    import json
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    y = np.arange(n) % k
    fold = np.where(np.arange(n) < n // 2, 1, 2).astype(np.int16)
    z = rng.normal(size=(n, k)).astype(np.float32)
    res = {"y": y, "test_fold": fold, "test_logits": z, "names_crc32": np.int64(7)}
    for f in (1, 2):
        idx = np.flatnonzero(fold != f)[:3]
        res[f"iv_idx_{f}"] = idx
        res[f"iv_logits_{f}"] = (np.full((3, k), np.nan, np.float32) if iv_nan else rng.normal(size=(3, k)).astype(np.float32))
    np.savez_compressed(d / f"{cfg}__{cond}.npz", **res)
    return json


def test_test_only_iv_scoredir_accepts_nan_inner_validation_on_noisy(tmp_path):
    import analyze_3c
    import analyze as A

    d = tmp_path / "run"
    _write_scores(d, "mn10_int8", "clean")
    for c in ["bg__10db", "bg__5db", "bg__0db", "pink__10db", "pink__5db", "pink__0db", "white__10db", "white__5db", "white__0db"]:
        _write_scores(d, "mn10_int8", c, iv_nan=True)
    sd = analyze_3c.TestOnlyIVScoreDir(d)
    assert sd.has("mn10_int8", "clean") and sd.has("mn10_int8", "bg__10db") and sd.has("mn10_int8", "field")
    r = sd.get("mn10_int8", "field")
    assert np.isfinite(r["z"]).all() and all(np.isnan(v).all() for _, v in r["iv"].values())
    # Stage 2's ScoreDir refuses the same files (its field inner-validation rows would be NaN)
    plain = A.ScoreDir(d)
    assert not plain.has("mn10_int8", "bg__10db")


def test_ci95_matches_a_t_interval():
    import analyze_3c
    from scipy import stats

    fm = np.array([0.01, 0.02, 0.015, 0.0, 0.03])
    lo, hi = analyze_3c.ci95(fm)
    h = stats.t.ppf(0.975, 4) * fm.std(ddof=1) / np.sqrt(5)
    assert lo == pytest.approx(fm.mean() - h) and hi == pytest.approx(fm.mean() + h)
    assert analyze_3c.ci95(np.array([0.2, 0.2])) == (pytest.approx(0.2), pytest.approx(0.2))


def test_only_folds_filters_outer_folds():
    """benchmark.py's cross_validation.only_folds keeps the named folds, computed as without the option."""
    import pandas as pd
    from omegaconf import OmegaConf

    from QWave.datasets import make_outer_folds

    df = pd.DataFrame({"folder": np.repeat([1, 2, 3], 4), "x": np.arange(12)})
    cv = OmegaConf.create({"fold_column": "folder", "n_splits": 3, "random_seed": 42, "only_folds": [2]})
    folds = make_outer_folds(df, np.zeros(12, int), cv)
    keep = {int(f) for f in cv.get("only_folds")}
    kept = [f for f in folds if int(f[0]) in keep]
    assert [f[0] for f in kept] == [2]
    full = [f for f in make_outer_folds(df, np.zeros(12, int), cv) if f[0] == 2][0]
    assert np.array_equal(kept[0][1], full[1]) and np.array_equal(kept[0][2], full[2])
