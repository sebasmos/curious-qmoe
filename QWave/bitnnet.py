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


## NOTE: The original BitLinear and BitNetExpert are kept for compatibility with other quantization schemes.
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
        # elif self.num_bits >= 16:
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


# def reset_weights(m):
#     for layer in m.children():
#         if hasattr(layer, 'reset_parameters'):
#             layer.reset_parameters()
            
# class BitLinear(nn.Module):
#     """
#     BitNet-style linear layer.
#     """
#     def __init__(self, in_features, out_features):
#         super(BitLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
#         # Use a robust initializer
#         nn.init.xavier_uniform_(self.weight)

#     def forward(self, x):
#         # --- Weight Quantization ---
#         # 1. Scale weights for stabilization before quantization
#         w_scaled = self.weight / self.weight.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
        
#         # 2. Ternary Quantization: {-1, 0, 1} via sign()
#         w_quant = torch.sign(w_scaled)

#         # 3. Straight-Through Estimator (STE)
#         # On forward pass, use quantized weights. On backward pass, use gradients from full-precision weights.
#         # self.training is a built-in nn.Module attribute.
#         if self.training:
#             w_final = w_quant + (w_scaled - w_scaled).detach()
#         else:
#             w_final = w_quant

#         # --- Activation Quantization (8-bit) ---
#         scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
#         x_quant = (x * scale).round().clamp(-128, 127) / scale
        
#         # STE for activations
#         if self.training:
#             x_final = x + (x_quant - x).detach()
#         else:
#             x_final = x_quant

#         # --- Linear Operation ---
#         return F.linear(x_final, w_final)

#     def reset_parameters(self):
#         """Custom reset for this layer."""
#         nn.init.xavier_uniform_(self.weight)


# class MLPBitnet(nn.Module):
#     """
#     MLP using the corrected BitLinear layers.
#     """
#     def __init__(self, input_size, output_size, hidden_sizes=[128, 64], dropout_prob=0.3, activation_fn=nn.ReLU, use_residual=False):
#         super(MLPBitnet, self).__init__()
#         self.use_residual = use_residual # Note: Not implemented in this version
#         self.softmax = nn.LogSoftmax(dim=1)

#         layers = []
#         prev_size = input_size
#         for hidden_size in hidden_sizes:
#             layers.append(BitLinear(prev_size, hidden_size))
#             # LayerNorm is crucial for stabilizing quantized networks
#             layers.append(nn.LayerNorm(hidden_size))
#             layers.append(activation_fn())
#             layers.append(nn.Dropout(p=dropout_prob))
#             prev_size = hidden_size
        
#         self.fc_layers = nn.Sequential(*layers)
        
#         # Output layer is also a BitLinear layer
#         self.output_layer = BitLinear(hidden_sizes[-1], output_size)
        
#     def forward(self, x):
#         x = self.fc_layers(x)
#         x = self.output_layer(x)
#         x = self.softmax(x)
#         # Return the raw logits
#         return x
