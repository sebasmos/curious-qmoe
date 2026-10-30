"""Tests for idea 6 final round: the non-additive corruptions (scripts/redesign/idea6/final/corrupt.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "redesign" / "idea6" / "final"))

import corrupt as C  # noqa: E402

SR = C.SR
RNG = np.random.default_rng(0)
X = (RNG.standard_normal(5 * SR) * 0.1).astype(np.float32)


def test_impulse_response_is_deterministic_and_unit_energy():
    for rt in C.LEVELS["reverb"]:
        h = C.impulse_response(rt)
        assert len(h) == int(round(rt * SR))
        assert abs(np.sum(h ** 2) - 1.0) < 1e-9
        C._IR.clear()
        assert np.array_equal(h, C.impulse_response(rt))


def test_impulse_response_decay_matches_rt60():
    for rt in C.LEVELS["reverb"]:
        h = C.impulse_response(rt)[1:]  # tail only
        edc = np.cumsum((h ** 2)[::-1])[::-1]
        edc_db = 10 * np.log10(edc / edc[0])
        t = np.arange(len(h)) / SR
        sel = (edc_db < -5) & (edc_db > -35)  # fit the -5 to -35 dB range, extrapolate to -60
        slope = np.polyfit(t[sel], edc_db[sel], 1)[0]
        rt_est = -60.0 / slope
        assert abs(rt_est - rt) / rt < 0.15, (rt, rt_est)


def test_reverb_keeps_length_dtype_and_direct_path():
    y = C.reverb(X, 0.6)
    assert y.shape == X.shape and y.dtype == np.float32
    d = np.zeros(SR, np.float32)
    d[0] = 1.0
    out = C.reverb(d, 0.3)
    assert abs(out[0] - C.impulse_response(0.3)[0]) < 1e-6


def test_clip_peak_and_silence():
    for q in C.LEVELS["clip"]:
        y = C.hard_clip(X, q)
        assert abs(np.max(np.abs(y)) - q * np.max(np.abs(X))) < 1e-6
        inside = np.abs(X) <= q * np.max(np.abs(X))
        assert np.array_equal(y[inside], X[inside])
    z = np.zeros(100, np.float32)
    assert np.array_equal(C.hard_clip(z, 0.1), z)


def test_lowpass_attenuates_above_cutoff_and_is_zero_phase():
    t = np.arange(2 * SR) / SR
    for fc in C.LEVELS["lowpass"]:
        lo = np.sin(2 * np.pi * 0.25 * fc * t).astype(np.float32)
        hi = np.sin(2 * np.pi * 2.0 * fc * t).astype(np.float32) if 2.0 * fc < SR / 2 else None
        y = C.lowpass(lo, fc)
        mid = slice(SR // 2, -SR // 2)
        assert np.max(np.abs(y[mid] - lo[mid])) < 0.02  # passband kept, no delay
        if hi is not None:
            assert np.sqrt(np.mean(C.lowpass(hi, fc)[mid] ** 2)) < 0.02 * np.sqrt(np.mean(hi[mid] ** 2)) + 1e-3


def test_lowpass_short_clip_does_not_fail():
    assert len(C.lowpass(X[:50], 2000.0)) == 50


def test_labels_are_unique_and_have_one_double_underscore_free_form():
    labels = [C.cond_label(k, lv) for k in C.KINDS for lv in C.LEVELS[k]]
    assert len(set(labels)) == 9 and all("__" not in s for s in labels)
    assert labels[0] == "reverb_rt300ms" and labels[-1] == "lowpass_1000hz"


def test_level_change_db():
    assert abs(C.level_change_db(X, 2 * X) - 6.0206) < 1e-3
    assert C.level_change_db(np.zeros(10, np.float32), X[:10]) is None
