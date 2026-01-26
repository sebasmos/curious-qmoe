#!/usr/bin/env python3
"""
Analyze curiosity experiment results for ESC-50 dataset.
Extracts metrics from the 3 experiments: baseline, KL divergence, entropy regularization.
"""

import json
import numpy as np
from pathlib import Path

# Output directories
outputs_dir = Path("/Users/cajas.sebastian/Desktop/repositories/curious-qmoe/outputs")

experiments = {
    "Baseline": outputs_dir / "full_baseline_final" / "moe",
    "KL Divergence": outputs_dir / "full_kl_divergence_final" / "moe",
    "Entropy Regularization": outputs_dir / "full_entropy_regularization_final" / "moe"
}

results = {}

for exp_name, exp_dir in experiments.items():
    summary_file = exp_dir / "summary.json"

    if not summary_file.exists():
        print(f"WARNING: {summary_file} not found")
        continue

    with open(summary_file, 'r') as f:
        data = json.load(f)

    # Extract per-fold F1 scores
    fold_f1s = [fold['best_f1'] for fold in data['folds']]

    # Extract training and validation times
    train_times = [fold['training_duration_seconds'] for fold in data['folds']]
    val_times = [fold['validation_duration_seconds'] for fold in data['folds']]

    # Extract RAM usage
    ram_train = [fold['max_ram_mb_train'] for fold in data['folds']]
    ram_val = [fold['max_ram_mb_val'] for fold in data['folds']]

    results[exp_name] = {
        'f1_mean': np.mean(fold_f1s),
        'f1_std': np.std(fold_f1s, ddof=1),  # Sample std (n-1)
        'f1_folds': fold_f1s,
        'f1_min': min(fold_f1s),
        'f1_max': max(fold_f1s),
        'train_time_mean': np.mean(train_times),
        'train_time_std': np.std(train_times, ddof=1),
        'val_time_mean': np.mean(val_times),
        'val_time_std': np.std(val_times, ddof=1),
        'ram_train_mean': np.mean(ram_train),
        'ram_val_mean': np.mean(ram_val),
        'parameter_count': data['parameter_count']
    }

# Print results
print("\n" + "="*80)
print("CURIOSITY-DRIVEN ROUTING: ESC-50 5-FOLD CROSS-VALIDATION RESULTS")
print("="*80)

for exp_name in ["Baseline", "KL Divergence", "Entropy Regularization"]:
    if exp_name not in results:
        continue

    r = results[exp_name]
    print(f"\n{exp_name}:")
    print(f"  F1 Score: {r['f1_mean']:.4f} ± {r['f1_std']:.4f}")
    print(f"  Per-fold F1: {[f'{f:.4f}' for f in r['f1_folds']]}")
    print(f"  Range: [{r['f1_min']:.4f} - {r['f1_max']:.4f}]")
    print(f"  Training Time: {r['train_time_mean']:.2f}s ± {r['train_time_std']:.2f}s")
    print(f"  Validation Time: {r['val_time_mean']:.2f}s ± {r['val_time_std']:.2f}s")
    print(f"  RAM (Train): {r['ram_train_mean']:.2f} MB")
    print(f"  RAM (Val): {r['ram_val_mean']:.2f} MB")
    print(f"  Parameters: {r['parameter_count']:,}")

# Calculate improvements
if "Baseline" in results and "KL Divergence" in results:
    baseline = results["Baseline"]
    kl = results["KL Divergence"]

    f1_improvement_kl = ((kl['f1_mean'] - baseline['f1_mean']) / baseline['f1_mean']) * 100
    var_reduction_kl = ((baseline['f1_std']**2 - kl['f1_std']**2) / baseline['f1_std']**2) * 100
    val_overhead_kl = ((kl['val_time_mean'] - baseline['val_time_mean']) / baseline['val_time_mean']) * 100

    print(f"\n{'='*80}")
    print("KL DIVERGENCE vs BASELINE:")
    print(f"  F1 Improvement: +{f1_improvement_kl:.2f}%")
    print(f"  Variance Reduction: {var_reduction_kl:.1f}%")
    print(f"  Validation Overhead: +{val_overhead_kl:.1f}%")

if "Baseline" in results and "Entropy Regularization" in results:
    baseline = results["Baseline"]
    entropy = results["Entropy Regularization"]

    f1_improvement_ent = ((entropy['f1_mean'] - baseline['f1_mean']) / baseline['f1_mean']) * 100
    var_reduction_ent = ((baseline['f1_std']**2 - entropy['f1_std']**2) / baseline['f1_std']**2) * 100
    val_overhead_ent = ((entropy['val_time_mean'] - baseline['val_time_mean']) / baseline['val_time_mean']) * 100

    print(f"\n{'='*80}")
    print("ENTROPY REGULARIZATION vs BASELINE:")
    print(f"  F1 Improvement: +{f1_improvement_ent:.2f}%")
    print(f"  Variance Reduction: {var_reduction_ent:.1f}%")
    print(f"  Validation Overhead: +{val_overhead_ent:.1f}%")

print(f"\n{'='*80}\n")

# Generate markdown table
print("MARKDOWN TABLE FOR DOCUMENTATION:")
print("\n| Experiment | F1 Score (Mean ± Std) | Per-Fold F1 Scores | Δ F1 | Variance Reduction | Inference Overhead |")
print("|------------|----------------------|-------------------|------|-------------------|-------------------|")

for exp_name in ["Baseline", "KL Divergence", "Entropy Regularization"]:
    if exp_name not in results:
        continue

    r = results[exp_name]
    fold_str = ", ".join([f"{f:.4f}" for f in r['f1_folds']])

    if exp_name == "Baseline":
        print(f"| **{exp_name}** | {r['f1_mean']:.4f} ± {r['f1_std']:.4f} | [{fold_str}] | - | - | - |")
    else:
        baseline = results["Baseline"]
        f1_delta = ((r['f1_mean'] - baseline['f1_mean']) / baseline['f1_mean']) * 100
        var_red = ((baseline['f1_std']**2 - r['f1_std']**2) / baseline['f1_std']**2) * 100
        overhead = ((r['val_time_mean'] - baseline['val_time_mean']) / baseline['val_time_mean']) * 100

        marker = " ⭐" if exp_name == "Entropy Regularization" else ""
        print(f"| **{exp_name}**{marker} | {r['f1_mean']:.4f} ± {r['f1_std']:.4f} | [{fold_str}] | **+{f1_delta:.2f}%** | **{var_red:.1f}%** ↓ | +{overhead:.1f}% |")

print("\n")
