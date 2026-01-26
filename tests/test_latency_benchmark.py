#!/usr/bin/env python3
"""Unit tests for latency_benchmark.py"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analysis.latency_benchmark import (
    LatencyStats,
    VarianceTestResult,
    RouterOverhead,
    measure_latency,
    compare_variance,
    measure_router_overhead,
    build_q4_model,
    build_moe_model,
    build_router,
)


class TestLatencyStats:
    def test_to_dict_excludes_raw_latencies(self):
        stats = LatencyStats(
            mean_ms=10.0,
            std_ms=1.0,
            min_ms=8.0,
            max_ms=12.0,
            variance_ms=1.0,
            num_passes=100,
            raw_latencies=[1.0, 2.0, 3.0],
        )
        d = stats.to_dict()
        assert "raw_latencies" not in d
        assert d["mean_ms"] == 10.0
        assert d["num_passes"] == 100


class TestMeasureLatency:
    def test_measure_latency_basic(self):
        model = build_q4_model(in_dim=64, num_classes=10, device="cpu")
        data = torch.randn(32, 64)
        stats = measure_latency(model, data, num_passes=5, warmup=2, batch_size=16)

        assert stats.num_passes == 5
        assert stats.mean_ms > 0
        assert stats.std_ms >= 0
        assert stats.min_ms <= stats.mean_ms <= stats.max_ms
        assert len(stats.raw_latencies) == 5

    def test_measure_latency_moe_model(self):
        model = build_moe_model(in_dim=64, num_classes=10, device="cpu", mc_samples=3)
        data = torch.randn(32, 64)
        stats = measure_latency(model, data, num_passes=3, warmup=1, batch_size=16)

        assert stats.num_passes == 3
        assert stats.mean_ms > 0


class TestCompareVariance:
    def test_levene_test_identical_distributions(self):
        np.random.seed(42)
        lat1 = list(np.random.normal(10, 2, 100))
        lat2 = list(np.random.normal(10, 2, 100))
        result = compare_variance(lat1, lat2, ("A", "B"))

        # Similar variances should have high p-value (not significant)
        assert result.p_value > 0.01
        assert 0.5 < result.variance_ratio < 2.0

    def test_levene_test_different_distributions(self):
        np.random.seed(42)
        lat1 = list(np.random.normal(10, 5, 100))  # high variance
        lat2 = list(np.random.normal(10, 1, 100))  # low variance
        result = compare_variance(lat1, lat2, ("High", "Low"))

        # Very different variances should have low p-value
        assert result.p_value < 0.05
        assert result.significant
        assert result.variance_ratio > 10  # 25/1 = 25x difference expected

    def test_summary_format(self):
        lat1 = [1.0, 2.0, 3.0]
        lat2 = [1.0, 1.5, 2.0]
        result = compare_variance(lat1, lat2, ("Q4", "MoE"))
        assert "Q4" in result.summary
        assert "MoE" in result.summary
        assert "var=" in result.summary


class TestRouterOverhead:
    def test_router_overhead_measurement(self):
        router = build_router(in_dim=64, num_experts=4, mc_samples=3, device="cpu")
        moe = build_moe_model(in_dim=64, num_classes=10, device="cpu", mc_samples=3)
        data = torch.randn(32, 64)

        overhead = measure_router_overhead(router, moe, data, num_trials=3, batch_size=16)

        assert overhead.mc_samples == 3
        assert overhead.router_ms > 0
        assert overhead.total_ms > 0
        assert 0 <= overhead.overhead_pct <= 100


class TestModelBuilders:
    def test_build_q4_model(self):
        model = build_q4_model(in_dim=64, num_classes=10, device="cpu")
        x = torch.randn(4, 64)
        out = model(x)
        assert out.shape == (4, 10)

    def test_build_moe_model(self):
        model = build_moe_model(in_dim=64, num_classes=10, device="cpu", mc_samples=3)
        x = torch.randn(4, 64)
        out = model(x)
        assert len(out) == 4  # (logits, router_p, lb_loss, curiosity)
        assert out[0].shape == (4, 10)

    def test_build_router(self):
        router = build_router(in_dim=64, num_experts=4, mc_samples=5, device="cpu")
        x = torch.randn(4, 64)
        logits, uncertainty = router(x, compute_uncertainty=True)
        assert logits.shape == (4, 4)
        assert uncertainty.shape == (4,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
