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


# QWave/bitnnet.py  — PATCH

from torch import nn
import torch
import torch.nn.functional as F

class BitLinear(nn.Module):
    """
    Linear layer supporting fixed k-bit or BitNet-style ternary ("bitnet").
    """
    def __init__(self, in_features, out_features, num_bits=16, bias=True, pre_ln=False):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.num_bits     = num_bits
        self.pre_ln       = pre_ln

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        # optional pre-LN helps quant stability
        self.act_ln = nn.LayerNorm(in_features) if pre_ln else nn.Identity()

    def forward(self, x):
        if self.num_bits == "bitnet":
            return self.forward_bitnet(x)
        elif isinstance(self.num_bits, int) and self.num_bits >= 16:
            return F.linear(x, self.weight, self.bias)
        else:
            return self.forward_quantized(x)

    def forward_quantized(self, x):
        # Activation quant (absmax) + STE
        x = self.act_ln(x)
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        xq = (x * scale).round().clamp(-128, 127) / scale
        x_final = x + (xq - x).detach()

        # Weight quant (k-bit, symmetric, per-layer)
        w_centered = self.weight - self.weight.mean()
        if self.num_bits == 1:
            # binary weights (fallback)
            wq = torch.sign(w_centered)
        else:
            q_min = -2. ** (self.num_bits - 1)
            q_max =  2. ** (self.num_bits - 1) - 1
            w_scale = w_centered.abs().max() / q_max
            wq = torch.round(w_centered / w_scale.clamp(min=1e-5)).clamp(q_min, q_max) * w_scale
        w_final = self.weight + (wq - self.weight).detach()
        return F.linear(x_final, w_final, self.bias)

    def forward_bitnet(self, x):
        # --- Pre-normalize activations ---
        x = self.act_ln(x)

        # --- Ternary activations with STE: {-1, 0, +1} ---
        # Threshold via mean-abs; you can tune delta if needed.
        delta_a = 0.0  # activations often okay at sign-only; set 0.05 to introduce zeros
        thr_a = delta_a * x.abs().mean(dim=-1, keepdim=True)
        xa = torch.where(x.abs() < thr_a, torch.zeros_like(x), torch.sign(x))
        x_final = x + (xa - x).detach()

        # --- Ternary weights with per-output-channel alpha ---
        W = self.weight
        Wc = W - W.mean(dim=1, keepdim=True)
        # ternary codebook
        delta_w = 0.05  # small threshold yields 0s for small magnitudes
        thr_w = delta_w * Wc.abs().mean(dim=1, keepdim=True)
        code = torch.where(Wc.abs() < thr_w, torch.zeros_like(Wc), torch.sign(Wc))
        # scale α per output channel (fan-out)
        alpha = (Wc.abs() * (code != 0).float()).sum(dim=1, keepdim=True)
        denom = (code != 0).float().sum(dim=1, keepdim=True).clamp(min=1.0)
        alpha = alpha / denom  # mean |W| over non-zeros
        Wq = alpha * code

        # STE
        W_final = W + (Wq - W).detach()

        return F.linear(x_final, W_final, self.bias)

# Keep fp32 classifier for stability
class BitNetExpert(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_sizes, dropout_prob, num_bits="bitnet",
                 pre_ln=True, bias=True):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_sizes:
            layers += [
                BitLinear(last, h, num_bits=num_bits, bias=bias, pre_ln=pre_ln),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
            ]
            last = h
        # final layer fp32
        layers.append(nn.Linear(last, num_classes, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
