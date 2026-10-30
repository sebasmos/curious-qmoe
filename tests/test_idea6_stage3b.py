"""Tests for idea 6 Stage 3b's analysis pieces that Stage 2 did not have: the one-sided non-inferiority test and the
two-window mean builder."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "redesign" / "idea6" / "stage3b"))

import analyze_3b  # noqa: E402
import make_mean2  # noqa: E402


def _keys(n_folds=5, seeds=(0, 1)):
    return [(s, f) for s in seeds for f in range(1, n_folds + 1)]


def test_ni_test_matches_a_shifted_one_sided_t():
    rng = np.random.default_rng(0)
    keys = _keys()
    z = rng.uniform(0.7, 0.8, len(keys))
    x = z - 0.005 + rng.normal(0, 0.002, len(keys))
    r = analyze_3b.ni_test(x, z, keys, ratio=0.25)
    d = x - z
    fm = np.array([d[[k[1] == f for k in keys]].mean() for f in range(1, 6)])
    t = (fm.mean() + 0.01) / (fm.std(ddof=1) / np.sqrt(5))
    assert r["ni_p"] == pytest.approx(stats.t.sf(t, 4))
    assert r["ni_diff"] == pytest.approx(d.mean())
    assert r["ni_p_nb"] >= r["ni_p"]  # the corrected variance is larger


def test_ni_test_rejects_a_clear_loss():
    keys = _keys()
    z = np.full(len(keys), 0.8)
    x = z - 0.05 + np.linspace(-0.002, 0.002, len(keys))
    assert analyze_3b.ni_test(x, z, keys, ratio=0.25)["ni_p"] > 0.5


def _write(d: Path, name, x, offset, cfg="n4", crc=7):
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{name}.npy", x)
    side = {"config": cfg, "max_length": 256, "frame_offset": offset, "names_crc32": crc,
            "backbone_cost_per_clip": {"bitops": 10.0, "macs": 3.0, "weight_bits": 4}}
    (d / f"{name}.json").write_text(json.dumps(side))


def test_make_mean2_averages_windows_and_doubles_cost(tmp_path):
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = a + 2
    b[1] = np.nan
    a[1] = np.nan
    _write(tmp_path / "w1", "n4__clean", a, 0)
    _write(tmp_path / "w2", "n4__clean", b, 256)
    make_mean2.main(["--first", str(tmp_path / "w1"), "--second", str(tmp_path / "w2"), "--out", str(tmp_path / "m")])
    m = np.load(tmp_path / "m" / "n4__clean.npy")
    assert np.allclose(m[[0, 2]], (a[[0, 2]] + b[[0, 2]]) / 2)
    assert np.isnan(m[1]).all()
    side = json.loads((tmp_path / "m" / "n4__clean.json").read_text())
    assert side["backbone_cost_per_clip"]["bitops"] == 20.0 and side["backbone_cost_per_clip"]["macs"] == 6.0
    assert side["backbone_cost_per_clip"]["weight_bits"] == 4


def test_make_mean2_refuses_mismatched_windows(tmp_path):
    a = np.zeros((2, 4), np.float32)
    _write(tmp_path / "w1", "n4__clean", a, 0, crc=1)
    _write(tmp_path / "w2", "n4__clean", a, 256, crc=2)
    with pytest.raises(SystemExit):
        make_mean2.main(["--first", str(tmp_path / "w1"), "--second", str(tmp_path / "w2"), "--out", str(tmp_path / "m")])
