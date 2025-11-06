#!/usr/bin/env python3
"""
Test to exactly replicate what benchmark.py does
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from QWave.bitnnet import BitNetExpert
from QWave.memory import print_size_of_model

# Same dimensions as ESC-50
in_dim = 1536
num_classes = 50
hidden_sizes = [640, 320]
dropout_prob = 0.1953403862875243

print("Testing model size calculation as in benchmark.py...\n")

model_kinds = ['1', '2', '4', '8', '16']

results = []
for model_kind in model_kinds:
    # Replicate build_model logic
    if model_kind.isdigit() and int(model_kind) in {1,2,4,8,16}:
        model = BitNetExpert(
            in_dim, num_classes,
            hidden_sizes=hidden_sizes,
            dropout_prob=dropout_prob,
            num_bits=int(model_kind),
        )

    # Replicate final_model assignment
    final_model = model
    final_model.eval()

    # Calculate size - exactly as benchmark.py line 380
    model_size = print_size_of_model(model, f"Model {model_kind}-bit")

    # Also check num_bits in first layer
    first_bitlinear = None
    for name, module in model.named_modules():
        if hasattr(module, 'num_bits'):
            first_bitlinear = module
            break

    if first_bitlinear:
        print(f"  → First BitLinear layer has num_bits = {first_bitlinear.num_bits}")

    results.append({
        'kind': model_kind,
        'size_kb': model_size / 1000,
        'params': sum(p.numel() for p in model.parameters())
    })
    print()

print("\nSummary:")
print(f"{'Model':<10} {'Size (KB)':<15} {'Params':<10}")
print("-" * 40)
for r in results:
    print(f"{r['kind']:<10} {r['size_kb']:<15.3f} {r['params']:<10}")
