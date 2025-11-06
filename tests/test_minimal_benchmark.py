#!/usr/bin/env python3
"""
Minimal test to replicate the exact benchmark.py flow
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from QWave.bitnnet import BitNetExpert
from QWave.memory import print_size_of_model
from QWave.train_utils import train_pytorch_local
from omegaconf import OmegaConf

# Minimal config
cfg = OmegaConf.create({
    'experiment': {
        'model': {
            'batch_size': 64,
            'hidden_sizes': [640, 320],
            'learning_rate': 0.001,
            'dropout_prob': 0.1953403862875243,
            'epochs': 1,  # Just 1 epoch for testing
            'early_stopping': {'patience': 10, 'delta': 0.01},
            'weight_decay': 0.005,
            'label_smoothing': 0.1,
            'patience': 15,
            'factor': 0.6
        }
    }
})

# Same dimensions as ESC-50
in_dim = 1536
num_classes = 50
hidden_sizes = [640, 320]
dropout_prob = 0.1953403862875243
device = torch.device('cpu')

print("Testing benchmark.py flow with training...\n")

# Create dummy dataset
n_samples = 320
X_train = torch.randn(n_samples, in_dim)
y_train = torch.randint(0, num_classes, (n_samples,))
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

X_val = torch.randn(80, in_dim)
y_val = torch.randint(0, num_classes, (80,))
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model_kinds = ['1', '2', '4', '8', '16']

for model_kind in model_kinds:
    print(f"\n{'='*60}")
    print(f"Testing model_kind = {model_kind}")
    print('='*60)

    # Build model - EXACT replica of benchmark.py line 144-150
    if model_kind.isdigit() and int(model_kind) in {1,2,4,8,16}:
        model = BitNetExpert(
            in_dim, num_classes,
            hidden_sizes=hidden_sizes,
            dropout_prob=dropout_prob,
            num_bits=int(model_kind),
        )

    model = model.to(device)

    # Check num_bits BEFORE training
    first_bitlinear = next((m for m in model.modules() if hasattr(m, 'num_bits')), None)
    if first_bitlinear:
        print(f"[BEFORE TRAINING] First BitLinear has num_bits = {first_bitlinear.num_bits}")

    # Print size before training
    print("\n[BEFORE TRAINING]")
    size_before = print_size_of_model(model, f"{model_kind}-bit_before", debug=False)

    # Train for 1 epoch - exact replica of benchmark.py flow
    class_weights = torch.ones(num_classes).to(device)

    print(f"\n[TRAINING] Running 1 epoch...")
    model_trained, train_losses, val_losses, best_f1, _, _, _ = train_pytorch_local(
        cfg.experiment,
        model,
        train_loader,
        val_loader,
        class_weights,
        in_dim,
        device,
        fold_dir='/tmp/test_fold',
        resume=False,
        checkpoint_path='/tmp/test_checkpoint.pth'
    )

    # Replicate the exact benchmark.py flow: lines 308-315
    final_model = model
    if os.path.exists('/tmp/test_checkpoint.pth'):
        final_model.load_state_dict(torch.load('/tmp/test_checkpoint.pth', map_location=device))
    else:
        final_model.load_state_dict(model_trained.state_dict())
    final_model.eval()

    # Check num_bits AFTER training/loading
    first_bitlinear_after = next((m for m in final_model.modules() if hasattr(m, 'num_bits')), None)
    if first_bitlinear_after:
        print(f"\n[AFTER TRAINING] First BitLinear has num_bits = {first_bitlinear_after.num_bits}")

    # Print size after training - EXACT replica of benchmark.py line 386
    print("\n[AFTER TRAINING]")
    has_bitlinear = any(hasattr(m, 'num_bits') for m in model.modules())
    if has_bitlinear:
        first_bitlinear = next((m for m in model.modules() if hasattr(m, 'num_bits')), None)
        if first_bitlinear:
            print(f"[DEBUG] Model {model_kind} has BitLinear with num_bits={first_bitlinear.num_bits}")

    model_size = print_size_of_model(model, f"{model_kind}_model", debug=False)

    print(f"\nSize: {model_size/1e6:.3f} MB")
    print(f"Params: {sum(p.numel() for p in model.parameters())}")

    # Cleanup
    if os.path.exists('/tmp/test_checkpoint.pth'):
        os.remove('/tmp/test_checkpoint.pth')

print("\n" + "="*60)
print("Test complete")
