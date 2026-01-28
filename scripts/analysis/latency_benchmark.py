#!/usr/bin/env python3
"""
Latency Benchmark for Quantized Models
======================================

Measures:
1. Inference latency variance across models (Q4-Base vs MoE)
2. Router overhead in MC-dropout based routing

Tasks 1.1 & 1.2 from CVPR 2026 rebuttal.

Usage:
    python scripts/analysis/latency_benchmark.py --synthetic --num-passes 100
    python scripts/analysis/latency_benchmark.py --csv /path/to/data.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from QWave.bitnnet import BitNetExpert
from QWave.moe import qMoEModelBatched, BayesianRouter


@dataclass
class LatencyStats:
    """Latency measurement results."""
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    variance_ms: float
    num_passes: int
    raw_latencies: List[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if k != "raw_latencies"}


@dataclass
class VarianceTestResult:
    """Levene's test result for variance comparison."""
    statistic: float
    p_value: float
    significant: bool
    variance_ratio: float
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["significant"] = bool(d["significant"])  # ensure JSON serializable
        return d


@dataclass
class RouterOverhead:
    """Router timing overhead measurement."""
    router_ms: float
    router_std: float
    total_ms: float
    overhead_pct: float
    mc_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def measure_latency(
    model: nn.Module,
    data: torch.Tensor,
    num_passes: int = 100,
    warmup: int = 10,
    batch_size: int = 64,
) -> LatencyStats:
    """Measure inference latency over multiple passes."""
    model.eval()
    device = next(model.parameters()).device
    latencies = []

    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            for i in range(0, len(data), batch_size):
                _ = model(data[i:i+batch_size].to(device))

        # Measure
        for _ in range(num_passes):
            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            for i in range(0, len(data), batch_size):
                out = model(data[i:i+batch_size].to(device))
                if isinstance(out, tuple):
                    _ = out[0]

            if device.type == "cuda":
                torch.cuda.synchronize()

            latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    return LatencyStats(
        mean_ms=float(np.mean(arr)),
        std_ms=float(np.std(arr)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        variance_ms=float(np.var(arr)),
        num_passes=num_passes,
        raw_latencies=latencies,
    )


def compare_variance(lat1: List[float], lat2: List[float], names=("A", "B")) -> VarianceTestResult:
    """Run Levene's test comparing variance between two latency distributions."""
    stat, p = stats.levene(lat1, lat2)
    var1, var2 = np.var(lat1), np.var(lat2)
    ratio = var1 / var2 if var2 > 0 else float('inf')

    sig = "*" if p < 0.05 else ""
    sig += "*" if p < 0.01 else ""
    sig += "*" if p < 0.001 else ""

    summary = f"{names[0]} var={var1:.2f}, {names[1]} var={var2:.2f}, ratio={ratio:.2f}x, p={p:.4f}{sig}"

    return VarianceTestResult(
        statistic=float(stat),
        p_value=float(p),
        significant=p < 0.05,
        variance_ratio=ratio,
        summary=summary,
    )


def measure_router_overhead(
    router: BayesianRouter,
    moe_model: nn.Module,
    data: torch.Tensor,
    num_trials: int = 100,
    batch_size: int = 64,
) -> RouterOverhead:
    """Measure MC-dropout router overhead as percentage of total inference."""
    device = next(router.parameters()).device
    router_times = []

    # Time router only
    with torch.no_grad():
        for _ in range(num_trials):
            t0 = time.perf_counter()
            for i in range(0, len(data), batch_size):
                router(data[i:i+batch_size].to(device), compute_uncertainty=True)
            router_times.append((time.perf_counter() - t0) * 1000)

    # Time full model
    full_stats = measure_latency(moe_model, data, num_passes=num_trials, batch_size=batch_size)

    router_mean = float(np.mean(router_times))
    overhead = (router_mean / full_stats.mean_ms) * 100 if full_stats.mean_ms > 0 else 0

    return RouterOverhead(
        router_ms=router_mean,
        router_std=float(np.std(router_times)),
        total_ms=full_stats.mean_ms,
        overhead_pct=overhead,
        mc_samples=router.mc_samples,
    )


def build_q4_model(in_dim=1536, num_classes=50, device="cpu") -> BitNetExpert:
    """Create 4-bit quantized model."""
    return BitNetExpert(in_dim, num_classes, [640, 320], 0.2, num_bits=4).to(device)


def build_moe_model(in_dim=1536, num_classes=50, device="cpu", mc_samples=10) -> qMoEModelBatched:
    """Create MoE model with Bayesian router and KL divergence curiosity."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "experiment": {
            "router": {
                "hidden_dim": 128,
                "expert_quantizations": ['bitnet', '4', '8'],  # FIXED: Match validated 5-fold CV config
                "num_experts": 3,  # FIXED: 3 experts (bitnet, 4-bit, 8-bit)
                "top_k": 1,
                "load_balancing_alpha": 1e-3,
                "use_curiosity": True,
                "curiosity_strategy": "kl_divergence",  # CRITICAL: Explicit KL divergence (Equation 8)
                "curiosity_alpha": 0.02,  # ADDED: Curiosity strength parameter
                "mc_samples": mc_samples,
            },
            "model": {"hidden_sizes": [640, 320], "dropout_prob": 0.2},
        }
    })
    return qMoEModelBatched(cfg, in_dim, num_classes, 3, 1).to(device)  # num_experts=3, top_k=1


def build_router(in_dim=1536, num_experts=4, mc_samples=10, device="cpu") -> BayesianRouter:
    """Create standalone Bayesian router."""
    return BayesianRouter(in_dim, 128, num_experts, 0.2, mc_samples).to(device)


def run_benchmark(args):
    """Run full benchmark."""
    device = torch.device(args.device)

    # Make output_dir relative to repository root, not current working directory
    if not Path(args.output_dir).is_absolute():
        output_dir = ROOT / args.output_dir
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    if args.synthetic:
        data = torch.randn(400, args.in_dim)
    else:
        import pandas as pd
        df = pd.read_csv(args.csv)
        cols = [c for c in df.columns if c not in ["folder", "name", "label", "category", "class_id"]]
        data = torch.tensor(df[cols].values, dtype=torch.float32)

    # Models
    q4 = build_q4_model(args.in_dim, args.num_classes, device)
    moe = build_moe_model(args.in_dim, args.num_classes, device, args.mc_samples)
    router = build_router(args.in_dim, 4, args.mc_samples, device)

    # Measure latency
    print(f"\nMeasuring Q4-Base ({args.num_passes} passes)...")
    q4_stats = measure_latency(q4, data, args.num_passes)
    print(f"  {q4_stats.mean_ms:.2f} +/- {q4_stats.std_ms:.2f} ms")

    print(f"\nMeasuring MoE ({args.num_passes} passes)...")
    moe_stats = measure_latency(moe, data, args.num_passes)
    print(f"  {moe_stats.mean_ms:.2f} +/- {moe_stats.std_ms:.2f} ms")

    # Compare variance
    print("\nVariance comparison (Levene's test):")
    var_test = compare_variance(q4_stats.raw_latencies, moe_stats.raw_latencies, ("Q4-Base", "MoE"))
    print(f"  {var_test.summary}")

    # Router overhead
    print(f"\nRouter overhead ({args.mc_samples} MC samples)...")
    overhead = measure_router_overhead(router, moe, data, args.num_passes)
    print(f"  Router: {overhead.router_ms:.2f} ms ({overhead.overhead_pct:.1f}% of total)")

    # Save results
    results = {
        "q4_base": q4_stats.to_dict(),
        "moe": moe_stats.to_dict(),
        "variance_test": var_test.to_dict(),
        "router_overhead": overhead.to_dict(),
    }

    with open(output_dir / "latency_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'latency_results.json'}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Latency benchmark for quantized models")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--csv", type=str, help="Path to data CSV")
    parser.add_argument("--num-passes", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="outputs/rebuttal_latency")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--in-dim", type=int, default=1536)
    parser.add_argument("--num-classes", type=int, default=50)
    parser.add_argument("--mc-samples", type=int, default=10)

    args = parser.parse_args()

    if not args.synthetic and not args.csv:
        parser.error("Provide --synthetic or --csv")

    run_benchmark(args)


if __name__ == "__main__":
    main()
