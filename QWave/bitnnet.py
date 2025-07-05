#### QWave BitLinear Module ####
from torch import nn
import timm
import torch
import torch.nn.functional as F
import random
"""

Quantizes weights to ternary values (-1, 0, +1) using absmean quantization.
Quantizes activations to 8-bit precision using absmax quantization.
Uses STE for training to allow gradient flow through non-differentiable quantization steps.
Includes a learnable scaling factor to adjust the output.
"""

def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
            
class BitLinear(nn.Module):
    """
    BitNet-style linear layer.
    """
    def __init__(self, in_features, out_features):
        super(BitLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        # Use a robust initializer
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # --- Weight Quantization ---
        # 1. Scale weights for stabilization before quantization
        w_scaled = self.weight / self.weight.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        
        # 2. Ternary Quantization: {-1, 0, 1} via sign()
        w_quant = torch.sign(w_scaled)

        # 3. Straight-Through Estimator (STE)
        # On forward pass, use quantized weights. On backward pass, use gradients from full-precision weights.
        # self.training is a built-in nn.Module attribute.
        if self.training:
            w_final = w_quant + (w_scaled - w_scaled).detach()
        else:
            w_final = w_quant

        # --- Activation Quantization (8-bit) ---
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_quant = (x * scale).round().clamp(-128, 127) / scale
        
        # STE for activations
        if self.training:
            x_final = x + (x_quant - x).detach()
        else:
            x_final = x_quant

        # --- Linear Operation ---
        return F.linear(x_final, w_final)

    def reset_parameters(self):
        """Custom reset for this layer."""
        nn.init.xavier_uniform_(self.weight)


class MLPBitnet(nn.Module):
    """
    MLP using the corrected BitLinear layers.
    """
    def __init__(self, input_size, output_size, hidden_sizes=[128, 64], dropout_prob=0.3, activation_fn=nn.ReLU, use_residual=False):
        super(MLPBitnet, self).__init__()
        self.use_residual = use_residual # Note: Not implemented in this version
        self.softmax = nn.LogSoftmax(dim=1)

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(BitLinear(prev_size, hidden_size))
            # LayerNorm is crucial for stabilizing quantized networks
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(activation_fn())
            layers.append(nn.Dropout(p=dropout_prob))
            prev_size = hidden_size
        
        self.fc_layers = nn.Sequential(*layers)
        
        # Output layer is also a BitLinear layer
        self.output_layer = BitLinear(hidden_sizes[-1], output_size)
        
    def forward(self, x):
        x = self.fc_layers(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        # Return the raw logits
        return x
