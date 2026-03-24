#!/usr/bin/env python3
"""
F1 Variance Comparison Across Experiments
==========================================

Compares F1 score variance reduction across different alpha values vs baseline.

CVPR 2026 Rebuttal Command:
    cd scripts
    python analysis/alpha_comparison.py \
        --baseline ../outputs-rebuttal/outputs/full_baseline_final/moe/summary.json \
        --experiments \
            ../outputs-rebuttal/outputs/alpha_020_fix/moe/summary.json \
            ../outputs-rebuttal/outputs/alpha_030_fix/moe/summary.json \
        --labels baseline alpha_0.2 alpha_0.3

Output: outputs-rebuttal/analysis/rebuttal_comparison/variance_comparison.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

ROOT = Path(__file__).resolve().parents[2]

def load_summary(path: Path) -> Dict[str, float]:
    """Load summary.json and extract F1 metrics."""
    with open(path) as f:
        data = json.load(f)

    return {
        "f1_mean": data.get("f1_mean", 0.0),
        "f1_std": data.get("f1_std", 0.0),
        "accuracy_mean": data.get("accuracy_mean", 0.0),
        "accuracy_std": data.get("accuracy_std", 0.0),
    }


def calculate_variance_reduction(
    baseline_std: float,
    current_std: float
) -> Dict[str, Any]:
    """Calculate variance reduction percentage."""
    if baseline_std == 0:
        return {"error": "baseline std is zero"}

    reduction_pct = ((baseline_std - current_std) / baseline_std) * 100

    return {
        "baseline_std": baseline_std,
        "current_std": current_std,
        "reduction_percent": reduction_pct,
        "improved": reduction_pct > 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare F1 variance across experiments"
    )
    parser.add_argument("--baseline", type=str, required=True,
                        help="Path to baseline summary.json")
    parser.add_argument("--experiments", nargs="+", required=True,
                        help="Paths to experiment summary.json files")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Labels for each experiment")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: outputs-rebuttal/analysis/rebuttal_comparison)")

    args = parser.parse_args()

    if len(args.experiments) + 1 != len(args.labels):
        parser.error("Number of labels must be number of experiments + 1 (for baseline)")

    # Output directory
    if args.output_dir is None:
        output_dir = ROOT / "outputs-rebuttal" / "analysis" / "rebuttal_comparison"
    elif not Path(args.output_dir).is_absolute():
        output_dir = ROOT / args.output_dir
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline
    baseline = load_summary(Path(args.baseline))
    print(f"Baseline: F1={baseline['f1_mean']:.3f}±{baseline['f1_std']:.3f}")

    # Compare experiments
    results = {
        "baseline": {
            "path": args.baseline,
            "label": args.labels[0],
            "f1_mean": baseline["f1_mean"],
            "f1_std": baseline["f1_std"],
            "accuracy_mean": baseline["accuracy_mean"],
            "accuracy_std": baseline["accuracy_std"],
        },
        "experiments": [],
    }

    print("\nVariance Reductions:")
    for exp_path, label in zip(args.experiments, args.labels[1:]):
        exp = load_summary(Path(exp_path))
        var_red = calculate_variance_reduction(baseline["f1_std"], exp["f1_std"])

        exp_result = {
            "path": exp_path,
            "label": label,
            "f1_mean": exp["f1_mean"],
            "f1_std": exp["f1_std"],
            "accuracy_mean": exp["accuracy_mean"],
            "accuracy_std": exp["accuracy_std"],
            "variance_reduction": var_red,
        }

        results["experiments"].append(exp_result)

        print(f"  {label}: F1={exp['f1_mean']:.3f}±{exp['f1_std']:.3f}")
        print(f"    Variance reduction: {var_red['reduction_percent']:.1f}%")

    # Save
    output_file = output_dir / "variance_comparison.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
