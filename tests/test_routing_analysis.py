#!/usr/bin/env python3
"""Unit tests for routing_analysis.py"""

import sys
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analysis.routing_analysis import (
    RoutingInfo,
    ExpertSummary,
    extract_routing,
    build_moe,
)


class TestRoutingInfo:
    def test_dataclass_fields(self):
        info = RoutingInfo(
            idx=0,
            expert=2,
            expert_name="4-bit",
            confidence=0.85,
            uncertainty=0.02,
            label=5,
            pred=5,
        )
        assert info.expert == 2
        assert info.expert_name == "4-bit"
        assert info.confidence == 0.85
        assert info.uncertainty == 0.02

    def test_to_dict(self):
        info = RoutingInfo(
            idx=0,
            expert=1,
            expert_name="2-bit",
            confidence=0.9,
            uncertainty=None,
            label=3,
            pred=3,
        )
        d = asdict(info)
        assert "expert" in d
        assert d["uncertainty"] is None


class TestExpertSummary:
    def test_summary_fields(self):
        summary = ExpertSummary(
            expert=0,
            name="1-bit",
            count=150,
            avg_conf=0.75,
            avg_unc=0.03,
        )
        assert summary.count == 150
        assert summary.avg_conf == 0.75


class TestExtractRouting:
    def test_extract_routing_basic(self):
        model = build_moe(in_dim=64, num_classes=10, device="cpu")
        data = torch.randn(32, 64)
        labels = torch.randint(0, 10, (32,))

        routing, summaries = extract_routing(
            model, data, labels,
            expert_names=["1-bit", "2-bit", "4-bit", "16-bit"],
            batch_size=16,
        )

        assert len(routing) == 32
        assert all(isinstance(r, RoutingInfo) for r in routing)
        assert all(0 <= r.expert < 4 for r in routing)
        assert all(0 <= r.confidence <= 1 for r in routing)

    def test_extract_routing_summaries(self):
        model = build_moe(in_dim=64, num_classes=10, device="cpu")
        data = torch.randn(100, 64)
        labels = torch.randint(0, 10, (100,))

        routing, summaries = extract_routing(model, data, labels, batch_size=32)

        # Should have at least one expert with samples
        total_samples = sum(s.count for s in summaries.values())
        assert total_samples == 100

        for exp_id, summary in summaries.items():
            assert isinstance(summary, ExpertSummary)
            assert summary.count > 0
            assert 0 <= summary.avg_conf <= 1

    def test_extract_routing_no_labels(self):
        model = build_moe(in_dim=64, num_classes=10, device="cpu")
        data = torch.randn(20, 64)

        routing, summaries = extract_routing(model, data, labels=None, batch_size=10)

        assert len(routing) == 20
        assert all(r.label is None for r in routing)


class TestBuildMoe:
    def test_build_moe_output_shape(self):
        model = build_moe(in_dim=64, num_classes=10, device="cpu")
        x = torch.randn(8, 64)
        out = model(x)

        assert len(out) == 4
        logits, router_p, lb_loss, curiosity = out
        assert logits.shape == (8, 10)
        assert router_p.shape == (8, 4)

    def test_build_moe_has_bayesian_router(self):
        model = build_moe(in_dim=64, num_classes=10, device="cpu")
        assert model.use_curiosity is True
        assert hasattr(model.router, "mc_samples")


class TestPlottingFunctions:
    """Test that plotting functions don't error (mocked matplotlib)."""

    def test_plot_routing_distribution(self, tmp_path):
        from scripts.analysis.routing_analysis import plot_routing_distribution

        summaries = {
            0: ExpertSummary(0, "1-bit", 50, 0.6, 0.02),
            1: ExpertSummary(1, "2-bit", 30, 0.7, 0.01),
            2: ExpertSummary(2, "4-bit", 15, 0.8, 0.01),
            3: ExpertSummary(3, "16-bit", 5, 0.9, 0.005),
        }
        output = tmp_path / "routing_dist.png"
        plot_routing_distribution(summaries, output)
        assert output.exists()

    def test_plot_confidence_by_expert(self, tmp_path):
        from scripts.analysis.routing_analysis import plot_confidence_by_expert

        summaries = {
            0: ExpertSummary(0, "1-bit", 50, 0.6, None),
            1: ExpertSummary(1, "2-bit", 30, 0.75, None),
        }
        output = tmp_path / "conf_by_expert.png"
        plot_confidence_by_expert(summaries, output)
        assert output.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
