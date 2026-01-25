#!/usr/bin/env python3
"""
Routing Analysis for MoE Models
===============================

Analyzes expert routing patterns and generates visualizations.

Task 2.2 from CVPR 2026 rebuttal.

Usage:
    python scripts/analysis/routing_analysis.py --synthetic
    python scripts/analysis/routing_analysis.py --csv /path/to/data.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from QWave.moe import qMoEModelBatched


@dataclass
class RoutingInfo:
    """Single sample routing info."""
    idx: int
    expert: int
    expert_name: str
    confidence: float
    uncertainty: Optional[float]
    label: Optional[int]
    pred: int


@dataclass
class ExpertSummary:
    """Per-expert statistics."""
    expert: int
    name: str
    count: int
    avg_conf: float
    avg_unc: Optional[float]


def extract_routing(
    model: qMoEModelBatched,
    data: torch.Tensor,
    labels: torch.Tensor = None,
    expert_names: List[str] = None,
    batch_size: int = 64,
) -> Tuple[List[RoutingInfo], Dict[int, ExpertSummary]]:
    """Extract routing decisions from MoE model."""
    if expert_names is None:
        expert_names = ["1-bit", "2-bit", "4-bit", "16-bit"]

    model.eval()
    device = next(model.parameters()).device
    results = []
    expert_data = {i: [] for i in range(len(expert_names))}

    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch = data[start:start+batch_size].to(device)
            out = model(batch)

            logits, probs = out[0], out[1]
            unc = out[3] if len(out) > 3 else None

            experts = probs.argmax(dim=1).cpu()
            confs = probs.max(dim=1).values.cpu()
            preds = logits.argmax(dim=1).cpu()

            for i, (exp, conf, pred) in enumerate(zip(experts, confs, preds)):
                idx = start + i
                info = RoutingInfo(
                    idx=idx,
                    expert=int(exp),
                    expert_name=expert_names[int(exp)] if int(exp) < len(expert_names) else f"E{exp}",
                    confidence=float(conf),
                    uncertainty=float(unc[i]) if unc is not None else None,
                    label=int(labels[idx]) if labels is not None else None,
                    pred=int(pred),
                )
                results.append(info)
                expert_data[int(exp)].append(info)

    # Compute summaries
    summaries = {}
    for exp, items in expert_data.items():
        if not items:
            continue
        uncs = [x.uncertainty for x in items if x.uncertainty is not None]
        summaries[exp] = ExpertSummary(
            expert=exp,
            name=expert_names[exp] if exp < len(expert_names) else f"E{exp}",
            count=len(items),
            avg_conf=float(np.mean([x.confidence for x in items])),
            avg_unc=float(np.mean(uncs)) if uncs else None,
        )

    return results, summaries


def plot_routing_distribution(summaries: Dict[int, ExpertSummary], output: Path):
    """Bar chart of samples per expert."""
    import matplotlib.pyplot as plt

    experts = sorted(summaries.keys())
    counts = [summaries[e].count for e in experts]
    names = [summaries[e].name for e in experts]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(experts)), counts, color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(experts)))
    ax.set_xticklabels(names)
    ax.set_xlabel('Expert')
    ax.set_ylabel('Samples Routed')
    ax.set_title('Expert Routing Distribution')

    for i, c in enumerate(counts):
        ax.annotate(str(c), (i, c), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Saved {output}")


def plot_confidence_by_expert(summaries: Dict[int, ExpertSummary], output: Path):
    """Bar chart of average confidence per expert."""
    import matplotlib.pyplot as plt

    experts = sorted(summaries.keys())
    confs = [summaries[e].avg_conf for e in experts]
    names = [summaries[e].name for e in experts]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(experts)), confs, color='coral', edgecolor='black')
    ax.set_xticks(range(len(experts)))
    ax.set_xticklabels(names)
    ax.set_xlabel('Expert')
    ax.set_ylabel('Average Confidence')
    ax.set_ylim(0, 1)
    ax.set_title('Routing Confidence by Expert')

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Saved {output}")


def build_moe(in_dim=1536, num_classes=50, device="cpu") -> qMoEModelBatched:
    """Create MoE model."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "experiment": {
            "router": {
                "hidden_dim": 128,
                "expert_quantizations": [1, 2, 4, 16],
                "num_experts": 4,
                "top_k": 1,
                "load_balancing_alpha": 1e-3,
                "use_curiosity": True,
                "mc_samples": 10,
            },
            "model": {"hidden_sizes": [640, 320], "dropout_prob": 0.2},
        }
    })
    return qMoEModelBatched(cfg, in_dim, num_classes, 4, 1).to(device)


def run_analysis(args):
    """Run routing analysis."""
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    if args.synthetic:
        data = torch.randn(400, args.in_dim)
        labels = torch.randint(0, args.num_classes, (400,))
    else:
        import pandas as pd
        df = pd.read_csv(args.csv)
        cols = [c for c in df.columns if c not in ["folder", "name", "label", "category", "class_id"]]
        data = torch.tensor(df[cols].values, dtype=torch.float32)
        labels = torch.tensor(df["class_id"].values) if "class_id" in df else None

    # Model
    model = build_moe(args.in_dim, args.num_classes, device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # Extract routing
    print("Extracting routing decisions...")
    routing, summaries = extract_routing(model, data, labels, args.expert_names)

    print(f"\nRouting summary ({len(routing)} samples):")
    for exp, s in sorted(summaries.items()):
        print(f"  {s.name}: {s.count} samples, conf={s.avg_conf:.3f}")

    # Visualizations
    plot_routing_distribution(summaries, output_dir / "routing_distribution.png")
    plot_confidence_by_expert(summaries, output_dir / "confidence_by_expert.png")

    # Save data
    with open(output_dir / "routing_data.json", "w") as f:
        json.dump({
            "routing": [asdict(r) for r in routing[:100]],
            "summaries": {str(k): asdict(v) for k, v in summaries.items()},
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="MoE routing analysis")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--csv", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--output-dir", type=str, default="outputs/analysis")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--in-dim", type=int, default=1536)
    parser.add_argument("--num-classes", type=int, default=50)
    parser.add_argument("--expert-names", nargs="+", default=["1-bit", "2-bit", "4-bit", "16-bit"])

    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
