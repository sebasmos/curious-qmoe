import torch
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from QWave.bitnnet import BitLinear, MLPBitnet

def test_bitlinear_forward_shapes():
    in_features, out_features, batch_size = 10, 5, 4
    layer = BitLinear(in_features, out_features)
    x = torch.randn(batch_size, in_features)
    y = layer(x)
    assert y.shape == (batch_size, out_features), "Output shape mismatch"

def test_bitlinear_quantization_ranges():
    in_features, out_features = 8, 3
    layer = BitLinear(in_features, out_features)
    wq = layer.quantize_weights()
    # Should be close to ternary values
    assert torch.all((wq.abs() < 1e-4) | (wq.abs() >= 1)), "Weights not ternary quantized"
    x = torch.randn(2, in_features)
    xq = layer.quantize_activations(x)
    assert torch.all(xq <= 128) and torch.all(xq >= -128), "Activations out of 8-bit range"

def test_mlpbitnet_forward():
    input_size, output_size = 16, 4
    model = MLPBitnet(input_size, output_size)