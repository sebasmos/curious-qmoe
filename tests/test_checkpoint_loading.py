#!/usr/bin/env python3
"""
Test if checkpoint loading affects model size calculation
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

print("Testing if checkpoint save/load affects model size...\n")

model_kinds = ['1', '2', '4', '8', '16']

for model_kind in model_kinds:
    print(f"\n{'='*60}")
    print(f"Model: {model_kind}-bit")
    print('='*60)

    # Build model - EXACT replica of benchmark.py
    if model_kind.isdigit() and int(model_kind) in {1,2,4,8,16}:
        model = BitNetExpert(
            in_dim, num_classes,
            hidden_sizes=hidden_sizes,
            dropout_prob=dropout_prob,
            num_bits=int(model_kind),
        )

    # Check num_bits attribute
    first_bitlinear = next((m for m in model.modules() if hasattr(m, 'num_bits')), None)
    if first_bitlinear:
        print(f"[ORIGINAL] First BitLinear has num_bits = {first_bitlinear.num_bits}")

    # Print size of original model
    print("\n[ORIGINAL MODEL]")
    size_orig = print_size_of_model(model, f"{model_kind}-bit_original")

    # Save checkpoint (as benchmark.py does)
    checkpoint_path = f'/tmp/test_checkpoint_{model_kind}.pth'
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nSaved checkpoint to {checkpoint_path}")
    print(f"Checkpoint file size: {os.path.getsize(checkpoint_path)/1e6:.3f} MB")

    # Create NEW model and load checkpoint (as benchmark.py does at line 308-314)
    print("\n[LOADING CHECKPOINT INTO NEW MODEL]")
    if model_kind.isdigit() and int(model_kind) in {1,2,4,8,16}:
        new_model = BitNetExpert(
            in_dim, num_classes,
            hidden_sizes=hidden_sizes,
            dropout_prob=dropout_prob,
            num_bits=int(model_kind),
        )

    # Check num_bits before loading
    first_bitlinear_new = next((m for m in new_model.modules() if hasattr(m, 'num_bits')), None)
    if first_bitlinear_new:
        print(f"[BEFORE LOAD] New model BitLinear has num_bits = {first_bitlinear_new.num_bits}")

    # Load state dict
    new_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    new_model.eval()

    # Check num_bits after loading
    first_bitlinear_loaded = next((m for m in new_model.modules() if hasattr(m, 'num_bits')), None)
    if first_bitlinear_loaded:
        print(f"[AFTER LOAD] Loaded model BitLinear has num_bits = {first_bitlinear_loaded.num_bits}")

    # Print size of loaded model
    print("\n[LOADED MODEL]")
    size_loaded = print_size_of_model(new_model, f"{model_kind}-bit_loaded")

    # Also test with final_model = model pattern from benchmark.py
    print("\n[USING final_model = model PATTERN]")
    final_model = model
    final_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    final_model.eval()

    first_bitlinear_final = next((m for m in final_model.modules() if hasattr(m, 'num_bits')), None)
    if first_bitlinear_final:
        print(f"[FINAL MODEL] BitLinear has num_bits = {first_bitlinear_final.num_bits}")

    size_final = print_size_of_model(final_model, f"{model_kind}-bit_final")

    print(f"\nSummary:")
    print(f"  Original:     {size_orig/1e6:.6f} MB")
    print(f"  Loaded:       {size_loaded/1e6:.6f} MB")
    print(f"  Final:        {size_final/1e6:.6f} MB")
    print(f"  Checkpoint:   {os.path.getsize(checkpoint_path)/1e6:.6f} MB")
    print(f"  All same? {size_orig == size_loaded == size_final}")

    # Cleanup
    os.remove(checkpoint_path)

print("\n" + "="*60)
print("Test complete")
