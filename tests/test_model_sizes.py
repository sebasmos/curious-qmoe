#!/usr/bin/env python3
"""
Test script to verify quantized model size calculations
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from QWave.bitnnet import BitNetExpert, BitLinear
from QWave.memory import print_size_of_model

# Same dimensions as ESC-50
in_dim = 1536
num_classes = 50
hidden_sizes = [640, 320]
dropout_prob = 0.1953403862875243

print("Creating models with different quantization bit-widths...\n")

models = {
    "1-bit": BitNetExpert(in_dim, num_classes, hidden_sizes, dropout_prob, num_bits=1),
    "2-bit": BitNetExpert(in_dim, num_classes, hidden_sizes, dropout_prob, num_bits=2),
    "4-bit": BitNetExpert(in_dim, num_classes, hidden_sizes, dropout_prob, num_bits=4),
    "8-bit": BitNetExpert(in_dim, num_classes, hidden_sizes, dropout_prob, num_bits=8),
    "16-bit": BitNetExpert(in_dim, num_classes, hidden_sizes, dropout_prob, num_bits=16),
    "bitnet": BitNetExpert(in_dim, num_classes, hidden_sizes, dropout_prob, num_bits="bitnet"),
}

print("=" * 100)
print("Testing model size calculations with DEBUG enabled:")
print("=" * 100)

for name, model in models.items():
    print_size_of_model(model, label=name, debug=True)
    print()
