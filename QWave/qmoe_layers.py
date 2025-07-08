
import os, sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def ternary_quantize(x, threshold=0.05):
    """
    Quantize tensor x to {-1, 0, +1} with a sparsity threshold.
    """
    x_sign = torch.sign(x)
    x_sparse = torch.where(x.abs() < threshold, torch.zeros_like(x), x_sign)
    return x_sparse

def ternary_to_binary(x: torch.Tensor) -> torch.Tensor:
    """Convert {-1, 0, +1} to 2-bit binary encoding: [-1, 0, +1] → [1,0], [0,0], [0,1]."""
    neg = (x == -1).to(torch.uint8)
    pos = (x == 1).to(torch.uint8)
    return torch.stack([neg, pos], dim=-1)  # shape [..., 2]

def packbits2(tensor: torch.Tensor) -> torch.Tensor:
    """
    Packs last dim of tensor of bits (uint8) into uint8 bytes along that dim.
    Assumes the last dim is divisible by 8.
    """
    assert tensor.shape[-1] % 8 == 0, "Last dim must be divisible by 8 for bit packing"
    shape = tensor.shape[:-1] + (tensor.shape[-1] // 8,)
    tensor = tensor.view(*shape[:-1], 8)
    # Ensure torch.arange is on the same device as the tensor
    packed = (tensor << torch.arange(7, -1, -1, device=tensor.device)).sum(dim=-1)
    return packed.to(torch.uint8)

def bitwise_dot(x_bin, w_bin):
    """
    Simulates binary dot product via XOR and popcount (Hamming distance).
    Input:
      x_bin: [B, packed_bits]
      w_bin: [C, packed_bits]
    Output:
      [B, C] score matrix
    """
    # Ensure inputs are on the same device
    w_bin = w_bin.to(x_bin.device)
    xor = torch.bitwise_xor(x_bin.unsqueeze(1), w_bin.unsqueeze(0))  # [B, C, packed_bits]
    return (8 * xor.shape[-1] - xor.sum(dim=-1).float())  # Higher score for more matches

# === Layer Implementation ===
class BitwisePopcountLinear(nn.Module):
    def __init__(self, in_features, out_features, threshold=0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        B = x.shape[0]

        # Quantize input and weight
        x_q = ternary_quantize(x, self.threshold)  # [B, D]
        w_q = ternary_quantize(self.weight, self.threshold)  # [C, D]

        # Convert to binary encoding: [B, D, 2]
        x_bin = ternary_to_binary(x_q).reshape(B, -1)  # [B, D*2]
        w_bin = ternary_to_binary(w_q).reshape(self.out_features, -1)  # [C, D*2]

        # Pad to multiple of 8 for packing
        def pad8(t):
            L = t.shape[-1]
            pad_len = (8 - (L % 8)) % 8
            # Ensure padding is done on the correct device
            return F.pad(t, (0, pad_len), value=0)

        x_bin = pad8(x_bin)
        w_bin = pad8(w_bin)

        # Pack bits
        # Ensure that packing operates on the correct device
        x_pack = packbits2(x_bin)  # [B, L/8]
        w_pack = packbits2(w_bin)  # [C, L/8]

        # Compute approximate dot via popcount logic
        scores = bitwise_dot(x_pack, w_pack)  # [B, C]
        return scores
    
class BitNetPopcountExpert(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_sizes, dropout_prob, threshold=0.05):
        super().__init__()
        layers = []
        last_dim = in_dim
        for h in hidden_sizes:
            layers.append(BitwisePopcountLinear(last_dim, h, threshold))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            last_dim = h
        layers.append(BitwisePopcountLinear(last_dim, num_classes, threshold))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
          return self.net(x)

class BitwiseLinear(nn.Module):
    """
    BitNet-1.58b-style linear layer using ternary weights/activations and integer matrix multiply.
    """
    def __init__(self, in_features, out_features, threshold=0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # Quantize activations and weights to ternary values {-1, 0, 1}
        x_tern = ternary_quantize(x, self.threshold).to(torch.int8)       # [B, D]
        w_tern = ternary_quantize(self.weight, self.threshold).to(torch.int8)  # [C, D]

        # Matrix multiply using integer dot product
        # [B, D] x [D, C]ᵗ = [B, C]
        # Note: we convert to int32 to avoid overflow on dot product
        x_int = x_tern.to(torch.int32)
        w_int = w_tern.to(torch.int32)

        out = torch.matmul(x_int, w_int.T)  # [B, C]

        return out.float()  # Output remains float for downstream modules

class BitNetExpert158b(nn.Module):
    """
    An expert model built using the BitwiseLinear layers. This defines a single expert's architecture.
    """
    def __init__(self, in_dim, num_classes, hidden_sizes, dropout_prob, threshold=0.05):
        super().__init__()
        layers = []
        last_dim = in_dim
        for h in hidden_sizes:
            # Use the new BitNet1.58b linear layer
            layers.append(BitwiseLinear(last_dim, h, threshold))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            last_dim = h
        # Final layer to produce class logits
        layers.append(BitwiseLinear(last_dim, num_classes, threshold))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class BitLinear(nn.Module):
    """
    A BitLinear layer supporting both fixed bit-widths (1, 2, 4, 8, 16) and BitNet-style ternary quantization.
    """
    def __init__(self, in_features, out_features, num_bits=16):
        super(BitLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bits = num_bits
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        if self.num_bits == "bitnet":
            return self.forward_bitnet(x)
        
        elif isinstance(self.num_bits, int) and self.num_bits >= 16:
            return F.linear(x, self.weight)
        else:
            return self.forward_quantized(x)

    def forward_quantized(self, x):
        # Activation quantization (absmax scaling)
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_quant = (x * scale).round().clamp(-128, 127) / scale
        x_final = x + (x_quant - x).detach()  # STE for activation

        # Weight quantization (absmax scaling)
        w_centered = self.weight - self.weight.mean()
        if self.num_bits == 1:
            w_quant = torch.sign(w_centered)  # Ternary
        else:
            q_min = -2.**(self.num_bits - 1)
            q_max = 2.**(self.num_bits - 1) - 1
            w_scale = w_centered.abs().max() / q_max
            w_quant = torch.round(w_centered / w_scale.clamp(min=1e-5)).clamp(q_min, q_max)
            w_quant = w_quant * w_scale

        w_final = self.weight + (w_quant - self.weight).detach() # STE for weights
        return F.linear(x_final, w_final)

    def forward_bitnet(self, x):
        # Activation ternarization with STE
        x_codebook = torch.sign(x)
        x_final = x + (x_codebook - x).detach()

        # Weight ternarization with STE
        w_centered = self.weight - self.weight.mean()
        w_ternary = torch.sign(w_centered)
        w_final = self.weight + (w_ternary - self.weight).detach()

        return F.linear(x_final, w_final)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

class BitNetExpert(nn.Module):
    """An expert model using the original BitLinear layers and LayerNorm."""
    def __init__(self, in_dim, num_classes, hidden_sizes, dropout_prob, num_bits=16):
        super().__init__()
        layers = []
        last_dim = in_dim
        for hidden_dim in hidden_sizes:
            layers.append(BitLinear(last_dim, hidden_dim, num_bits=num_bits))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            last_dim = hidden_dim
        layers.append(BitLinear(last_dim, num_classes, num_bits=num_bits))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)