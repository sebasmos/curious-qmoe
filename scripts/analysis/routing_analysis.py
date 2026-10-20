#!/usr/bin/env python3
"""
Routing Analysis for MoE Models
===============================

Analyzes expert routing patterns and generates visualizations.

Uses held-out samples only: each fold's test samples are routed by that
fold's own checkpoint, with features scaled by the scaler fitted on the
fold's fit portion, and the folds are pooled.

Usage:
    python scripts/analysis/routing_analysis.py --run-dir outputs/esc_pprior_a1.0_s0/moe \
        --output-dir scripts/analysis/outputs-paper/routing
    python scripts/analysis/routing_analysis.py --synthetic
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
        expert_names = ["BitNet", "Q4", "Q8"]

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
            # Confidence must reflect sample difficulty, so it comes from the
            # classifier's class distribution, NOT the router's distribution
            # over experts (the previous version reported max router
            # probability, which measures router decisiveness instead).
            class_probs = torch.softmax(logits, dim=1)
            confs = class_probs.max(dim=1).values.cpu()
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


def build_moe(in_dim=1536, num_classes=50, device="cpu", mc_samples=10, curiosity_alpha=0.02, curiosity_strategy="kl_divergence") -> qMoEModelBatched:
    """Create MoE model with correct configuration matching latency_benchmark.py."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "experiment": {
            "router": {
                "hidden_dim": 128,
                "expert_quantizations": ['bitnet', '4', '8'],  # CRITICAL: Match validated 5-fold CV
                "num_experts": 3,
                "top_k": 1,
                "load_balancing_alpha": 1e-3,
                "use_curiosity": True,
                "curiosity_strategy": curiosity_strategy,
                "curiosity_alpha": curiosity_alpha,  # Curiosity strength parameter
                "mc_samples": mc_samples,
            },
            "model": {"hidden_sizes": [640, 320], "dropout_prob": 0.2},
        }
    })
    return qMoEModelBatched(cfg, in_dim, num_classes, 3, 1).to(device)  # num_experts=3, top_k=1


EXPERT_LABELS = {"bitnet": "BitNet", "esc": "FP32", "qesc": "PTQ-int8", "16": "Q16"}


def expert_names_from_cfg(cfg) -> List[str]:
    return [EXPERT_LABELS.get(str(q), f"Q{q}") for q in cfg.experiment.router.expert_quantizations]


def pooled_summaries(records: List[dict], expert_names: List[str]) -> Dict[int, ExpertSummary]:
    out = {}
    for e in sorted({r["expert"] for r in records}):
        items = [r for r in records if r["expert"] == e]
        uncs = [r["uncertainty"] for r in items if r["uncertainty"] is not None]
        out[e] = ExpertSummary(expert=e, name=expert_names[e] if e < len(expert_names) else f"E{e}",
                               count=len(items), avg_conf=float(np.mean([r["confidence"] for r in items])),
                               avg_unc=float(np.mean(uncs)) if uncs else None)
    return out


def run_analysis(args):
    """Routing of held-out samples only: each fold's test samples go through
    that fold's own checkpoint (never a model that trained on them), scaled
    with the fold's fit-portion scaler, and the folds are pooled."""
    if args.output_dir is None:
        output_dir = ROOT / "scripts" / "analysis" / "outputs-routing"
    elif not Path(args.output_dir).is_absolute():
        output_dir = ROOT / args.output_dir
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        model = build_moe(args.in_dim, args.num_classes, args.device, args.mc_samples, 1.0, "precision_prior")
        data, labels = torch.randn(400, args.in_dim), torch.randint(0, args.num_classes, (400,))
        routing, summaries = extract_routing(model, data, labels, args.expert_names)
        records, names, folds = [asdict(r) for r in routing], args.expert_names, []
    else:
        from scripts.analysis.run_loader import list_folds, load_trained_fold
        run_dir = Path(args.run_dir).resolve()
        folds = args.folds or list_folds(run_dir)
        records, names = [], None
        for fold in folds:
            tf = load_trained_fold(run_dir, fold, csv=args.csv, device=args.device)
            names = args.expert_names or expert_names_from_cfg(tf.cfg)
            torch.manual_seed(int(tf.cfg.experiment.get("seed", 0)))
            routing, _ = extract_routing(tf.model, tf.X_test, tf.y_test, names)
            for r in routing:
                rec = asdict(r)
                rec["fold"], rec["idx"] = fold, int(tf.test_idx[r.idx])
                records.append(rec)
        summaries = pooled_summaries(records, names)

    print(f"\nRouting summary ({len(records)} held-out samples, folds {folds}):")
    for exp, s in sorted(summaries.items()):
        print(f"  {s.name}: {s.count} samples, conf={s.avg_conf:.3f}")

    plot_routing_distribution(summaries, output_dir / "routing_distribution.png")
    plot_confidence_by_expert(summaries, output_dir / "confidence_by_expert.png")

    results = {
        "config": {
            "run_dir": None if args.synthetic else str(Path(args.run_dir).resolve()),
            "folds": folds,
            "expert_names": names,
            "num_samples": len(records),
            "samples": "held-out test folds only, each routed by its own fold's checkpoint",
        },
        # Full per-sample records: the confidence-distribution figure needs the
        # whole distribution, not a 100-sample prefix, and the file stays small.
        "routing": records,
        "summaries": {str(k): asdict(v) for k, v in summaries.items()},
    }
    with open(output_dir / "routing_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="MoE routing analysis on held-out test folds")
    parser.add_argument("--run-dir", type=str, help="outputs/<tag>/moe from scripts/benchmark.py")
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="default: every fold in the run")
    parser.add_argument("--csv", type=str, default=None, help="dataset CSV, if it moved since training")
    parser.add_argument("--synthetic", action="store_true", help="random untrained model, for smoke tests")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--in-dim", type=int, default=1536)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--mc-samples", type=int, default=10)
    parser.add_argument("--expert-names", nargs="+", default=None, help="default: from the run's config")

    args = parser.parse_args()
    if not args.synthetic and not args.run_dir:
        parser.error("Provide --run-dir (or --synthetic)")
    if args.synthetic and args.expert_names is None:
        args.expert_names = ["BitNet", "Q4", "Q8"]
    run_analysis(args)


if __name__ == "__main__":
    main()
