import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import csv
from .memory import print_size_of_model
from .models import ESCModel 


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
    """Packs the last dimension of a binary tensor into uint8."""
    assert tensor.shape[-1] % 8 == 0, "Last dim must be divisible by 8 for bit packing"
    tensor = tensor.reshape(*tensor.shape[:-1], tensor.shape[-1] // 8, 8)
    # Packs bits into bytes: (b7 b6 b5 b4 b3 b2 b1 b0)
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
    def __init__(self, in_features, out_features, num_bits="bitnet", bias=True, pre_ln=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bits = num_bits  # "bitnet" or int in {1,2,4,8,16,...}
        self.pre_ln = pre_ln

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.act_ln = nn.LayerNorm(in_features) if pre_ln else nn.Identity()

    def forward(self, x):
        if self.num_bits == "bitnet":
            return self.forward_bitnet(x)
        elif isinstance(self.num_bits, int) and self.num_bits >= 16:
            return F.linear(self.act_ln(x), self.weight, self.bias)
        else:
            return self.forward_kbit(x)

    def forward_kbit(self, x):
        # Activation quant (absmax) with STE
        x = self.act_ln(x)
        x_scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        xq = (x * x_scale).round().clamp(-128, 127) / x_scale
        xa = x + (xq - x).detach()

        # Weight quant (symmetric, per-layer) with STE
        Wc = self.weight - self.weight.mean()
        if self.num_bits == 1:
            wq = torch.sign(Wc)
        else:
            qmin = -2**(self.num_bits - 1)
            qmax =  2**(self.num_bits - 1) - 1
            w_scale = Wc.abs().amax().clamp(min=1e-5) / qmax
            wq = (Wc / w_scale).round().clamp(qmin, qmax) * w_scale
        Wf = self.weight + (wq - self.weight).detach()

        return F.linear(xa, Wf, self.bias)

    def forward_bitnet(self, x):
        # --- Pre-normalize activations ---
        x = self.act_ln(x)

        # --- Ternary activations (optionally allow zeros) + STE ---
        delta_a = 0.05  # set to 0.05 to introduce zeros in activations
        thr_a = delta_a * x.abs().mean(dim=-1, keepdim=True)
        xa_code = torch.where(x.abs() < thr_a, torch.zeros_like(x), torch.sign(x))
        xa = x + (xa_code - x).detach()  # STE

        # --- Ternary weights with per-output-channel alpha + STE ---
        W  = self.weight
        Wc = W - W.mean(dim=1, keepdim=True)

        delta_w = 0.15   # small threshold -> some zeros, improves stability
        thr_w = delta_w * Wc.abs().mean(dim=1, keepdim=True)
        code = torch.where(Wc.abs() < thr_w, torch.zeros_like(Wc), torch.sign(Wc))  # {-1,0,+1}

        nz = (code != 0).float()
        alpha = (Wc.abs() * nz).sum(dim=1, keepdim=True) / nz.sum(dim=1, keepdim=True).clamp(min=1.0)
        wq = alpha * code
        Wf = W + (wq - W).detach()  # STE

        return F.linear(xa, Wf, self.bias)
# class BitLinear(nn.Module):
#     """
#     A BitLinear layer supporting both fixed bit-widths (1, 2, 4, 8, 16) and BitNet-style ternary quantization.
#     """
#     def __init__(self, in_features, out_features, num_bits=16):
#         super(BitLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.num_bits = num_bits
#         self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)

#     def forward(self, x):
#         if self.num_bits == "bitnet":
#             return self.forward_bitnet(x)
        
#         elif isinstance(self.num_bits, int) and self.num_bits >= 16:
#             return F.linear(x, self.weight)
#         else:
#             return self.forward_quantized(x)

#     def forward_quantized(self, x):
#         # Activation quantization (absmax scaling)
#         scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
#         x_quant = (x * scale).round().clamp(-128, 127) / scale
#         x_final = x + (x_quant - x).detach()  # STE for activation

#         # Weight quantization (absmax scaling)
#         w_centered = self.weight - self.weight.mean()
#         if self.num_bits == 1:
#             w_quant = torch.sign(w_centered)  # Ternary
#         else:
#             q_min = -2.**(self.num_bits - 1)
#             q_max = 2.**(self.num_bits - 1) - 1
#             w_scale = w_centered.abs().max() / q_max
#             w_quant = torch.round(w_centered / w_scale.clamp(min=1e-5)).clamp(q_min, q_max)
#             w_quant = w_quant * w_scale

#         w_final = self.weight + (w_quant - self.weight).detach() # STE for weights
#         return F.linear(x_final, w_final)

#     def forward_bitnet(self, x):
#         # Activation ternarization with STE
#         x_codebook = torch.sign(x)
#         x_final = x + (x_codebook - x).detach()

#         # Weight ternarization with STE
#         w_centered = self.weight - self.weight.mean()
#         w_ternary = torch.sign(w_centered)
#         w_final = self.weight + (w_ternary - self.weight).detach()

#         return F.linear(x_final, w_final)

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weight)

class BitNetExpert(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_sizes, dropout_prob,
                 num_bits="bitnet", pre_ln=True, bias=True):
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
        # FINAL CLASSIFIER IN FP32 (stabilizer)
        layers.append(nn.Linear(last, num_classes, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
if __name__ == "__main__":
    in_dim, num_classes, hidden_sizes, dropout_prob = 128, 10, [64, 32], 0.1
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    test_models = {
        "ESCModel (FP32 Baseline)": ESCModel(in_dim, num_classes, hidden_sizes, dropout_prob),
        "BitNetExpert (FP16/BF16/Full-Precision)": BitNetExpert(in_dim, num_classes, hidden_sizes, dropout_prob, num_bits=16),
        "BitNetExpert (8-bit Quant)": BitNetExpert(in_dim, num_classes, hidden_sizes, dropout_prob, num_bits=8),
        "BitNetExpert (4-bit Quant)": BitNetExpert(in_dim, num_classes, hidden_sizes, dropout_prob, num_bits=4),
        "BitNetExpert (2-bit Quant)": BitNetExpert(in_dim, num_classes, hidden_sizes, dropout_prob, num_bits=2),
        "BitNetExpert (1-bit Quant)": BitNetExpert(in_dim, num_classes, hidden_sizes, dropout_prob, num_bits=1),
        "BitNetExpert (Ternary 'bitnet' STE)": BitNetExpert(in_dim, num_classes, hidden_sizes, dropout_prob, num_bits="bitnet"),
        "BitNetExpert158b (Ternary Integer MatMul)": BitNetExpert158b(in_dim, num_classes, hidden_sizes, dropout_prob),
        "BitNetPopcountExpert (Ternary Bitwise Popcount)": BitNetPopcountExpert(in_dim, num_classes, hidden_sizes, dropout_prob),
    }

    for name, model in test_models.items(): model.to(device).eval()
    
    batch_size, num_warmup, num_runs = 128, 10, 100
    benchmark_input = torch.randn(batch_size, in_dim).to(device)
    results_times, results_memory_mb = {}, {}

    print("\nRunning benchmarks...")
    for name, model in test_models.items():
        with torch.no_grad():
            for _ in range(num_warmup): _ = model(benchmark_input)
            start_time = time.perf_counter()
            for _ in range(num_runs): _ = model(benchmark_input)
            end_time = time.perf_counter()
        results_times[name] = ((end_time - start_time) / num_runs) * 1000
        # --- FIX: Use the new, correct memory calculation function ---
        # results_memory_mb[name] = calculate_real_and_potential_model_size_mb(model, name)
        results_memory_mb[name] = print_size_of_model(model, name)

    csv_data = [["Model Name", "Average Inference Time (ms)", "X Times Faster (vs ESCModel FP32)", "Potential Storage (MB)", "Potential Storage Reduction"]]
    desired_order = list(test_models.keys())
    
    esc_baseline_time = results_times.get("ESCModel (FP32 Baseline)", 1.0)
    esc_baseline_mem = results_memory_mb.get("ESCModel (FP32 Baseline)", 1.0)
    
    for name in desired_order:
        if name not in results_times: continue
        avg_time = results_times[name]
        real_mem_mb = results_memory_mb[name]
        time_x_faster = f"{esc_baseline_time / avg_time:.2f}x" if avg_time > 0 else "N/A"
        mem_reduction = f"{esc_baseline_mem / real_mem_mb:.2f}x" if real_mem_mb > 0 else "N/A"
        csv_data.append([name, f"{avg_time:.4f}", time_x_faster, f"{real_mem_mb:.4f}", mem_reduction])

    fastest_model_name = min(results_times, key=results_times.get)
    csv_data.append([])
    csv_data.append(["Fastest Overall Model", fastest_model_name, f"{results_times[fastest_model_name]:.4f} ms"])

    print("\n--- Benchmark Results ---")
    for row in csv_data: print(f"{' | '.join(map(str, row))}")

    with open("results_fixed_memory.csv", 'w', newline='') as f:
        csv.writer(f).writerows(csv_data)
    print("\nResults saved to results_fixed_memory.csv")

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import time
# import csv
# # These custom local modules are assumed to be in the same directory or accessible.
# from memory import print_size_of_model
# from models import ESCModel

# def ternary_quantize(x, threshold=0.05):
#     """Quantizes a tensor to {-1, 0, 1} based on a threshold."""
#     x_sign = torch.sign(x)
#     x_sparse = torch.where(x.abs() < threshold, torch.zeros_like(x), x_sign)
#     return x_sparse

# def ternary_to_binary(x: torch.Tensor) -> torch.Tensor:
#     """Converts a ternary tensor {-1, 0, 1} to a binary representation."""
#     neg = (x == -1).to(torch.uint8)
#     pos = (x == 1).to(torch.uint8)
#     return torch.stack([neg, pos], dim=-1)

# def packbits2(tensor: torch.Tensor) -> torch.Tensor:
#     """Packs the last dimension of a binary tensor into uint8."""
#     assert tensor.shape[-1] % 8 == 0, "Last dim must be divisible by 8 for bit packing"
#     tensor = tensor.reshape(*tensor.shape[:-1], tensor.shape[-1] // 8, 8)
#     # Packs bits into bytes: (b7 b6 b5 b4 b3 b2 b1 b0)
#     packed = (tensor << torch.arange(7, -1, -1, device=tensor.device)).sum(dim=-1)
#     return packed.to(torch.uint8)

# def bitwise_dot(x_bin, w_bin):
#     """
#     Performs a dot product simulation using bitwise XOR and popcount.
#     This is equivalent to a matrix multiplication for binary tensors.
#     """
#     w_bin = w_bin.to(x_bin.device)
#     # XOR gives 1 if bits differ, 0 if same. Summing counts differing bits.
#     xor = torch.bitwise_xor(x_bin.unsqueeze(1), w_bin.unsqueeze(0))
#     # Total bits - differing bits = matching bits. Popcount is the sum of set bits.
#     # The formula below is a way to calculate dot product from XOR results.
#     return (8 * xor.shape[-1] - xor.sum(dim=-1).float())

# class BitwisePopcountLinear(nn.Module):
#     """
#     A linear layer optimized with bitwise operations.
#     Weights are pre-packed for efficient inference.
#     """
#     def __init__(self, in_features, out_features, threshold=0.05):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.threshold = threshold
#         self.weight = nn.Parameter(torch.empty(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)
#         # Buffer for the pre-packed weights for efficient inference
#         self.register_buffer('packed_weight', None)

#     def train(self, mode: bool = True):
#         # Clear the cached weight when switching to training mode
#         super().train(mode)
#         if mode:
#             self.packed_weight = None
#         return self

#     @staticmethod
#     def _pad8_last_dim(t):
#         """Pads the last dimension of a tensor to be a multiple of 8."""
#         last_dim = t.shape[-1]
#         pad_len = (8 - (last_dim % 8)) % 8
#         if pad_len > 0:
#             return F.pad(t, (0, pad_len), value=0)
#         return t

#     def forward(self, x):
#         B = x.shape[0]

#         # Activation processing (on-the-fly)
#         x_q = ternary_quantize(x, self.threshold)
#         x_bin = ternary_to_binary(x_q)
#         x_bin_padded = self._pad8_last_dim(x_bin)
#         x_bin_reshaped = x_bin_padded.reshape(B, -1)
#         x_pack = packbits2(x_bin_reshaped)

#         # Weight processing (cached for eval mode for speed)
#         if not self.training and self.packed_weight is not None:
#             w_pack = self.packed_weight
#         else:
#             # This block runs only once per layer in eval mode
#             w_q = ternary_quantize(self.weight, self.threshold)
#             w_bin = ternary_to_binary(w_q)
#             w_bin_padded = self._pad8_last_dim(w_bin)
#             w_bin_reshaped = w_bin_padded.reshape(self.out_features, -1)
#             w_pack = packbits2(w_bin_reshaped)
#             if not self.training:
#                 self.packed_weight = w_pack

#         scores = bitwise_dot(x_pack, w_pack.to(x_pack.device))
#         return scores

# class BitNetPopcountExpert(nn.Module):
#     def __init__(self, in_dim, num_classes, hidden_sizes, dropout_prob, threshold=0.05):
#         super().__init__()
#         layers = []
#         last_dim = in_dim
#         for h in hidden_sizes:
#             layers.append(BitwisePopcountLinear(last_dim, h, threshold))
#             layers.append(nn.LayerNorm(h))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(dropout_prob))
#             last_dim = h
#         layers.append(BitwisePopcountLinear(last_dim, num_classes, threshold))
#         self.net = nn.Sequential(*layers)

#     def forward(self, x):
#           return self.net(x)

# class BitwiseLinear(nn.Module):
#     """
#     A linear layer that uses true INT8 matrix multiplication for speed.
#     Weights are pre-quantized for efficient inference.
#     """
#     def __init__(self, in_features, out_features, threshold=0.05):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.threshold = threshold
#         self.weight = nn.Parameter(torch.empty(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)
#         # Buffer for the pre-quantized INT8 weights
#         self.register_buffer('quantized_w_int8', None)

#     def train(self, mode: bool = True):
#         # Clear the cached weight when switching to training mode
#         super().train(mode)
#         if mode:
#             self.quantized_w_int8 = None
#         return self

#     def forward(self, x):
#         x_tern = ternary_quantize(x, self.threshold).to(torch.int8)

#         # Use cached INT8 weight in eval mode for performance
#         if not self.training and self.quantized_w_int8 is not None:
#             w_tern = self.quantized_w_int8
#         else:
#             w_tern = ternary_quantize(self.weight, self.threshold).to(torch.int8)
#             if not self.training:
#                 self.quantized_w_int8 = w_tern

#         # Perform matrix multiplication using fast INT8 operations on CPU
#         out = torch.matmul(x_tern, w_tern.T.to(x_tern.device))
#         return out.float()

# class BitNetExpert158b(nn.Module):
#     def __init__(self, in_dim, num_classes, hidden_sizes, dropout_prob, threshold=0.05):
#         super().__init__()
#         layers = []
#         last_dim = in_dim
#         for h in hidden_sizes:
#             layers.append(BitwiseLinear(last_dim, h, threshold))
#             layers.append(nn.LayerNorm(h))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(dropout_prob))
#             last_dim = h
#         layers.append(BitwiseLinear(last_dim, num_classes, threshold))
#         self.net = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.net(x)

# class BitLinear(nn.Module):
#     """
#     A linear layer for Quantization-Aware Training (QAT) simulation.
#     Uses Straight-Through Estimators (STE) for gradients.
#     Weights are pre-quantized and cached for efficient inference simulation.
#     """
#     def __init__(self, in_features, out_features, num_bits=16):
#         super(BitLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.num_bits = num_bits
#         self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)
#         # Buffer for the cached quantized weight
#         self.register_buffer('quantized_w', None)

#     def train(self, mode: bool = True):
#         # Clear the cached weight when switching to training mode
#         super().train(mode)
#         if mode:
#             self.quantized_w = None
#         return self

#     def forward(self, x):
#         if self.quantized_w is None and not self.training:
#             # Cache the weight when switching to eval mode
#             self._quantize_weights()

#         if self.num_bits == "bitnet":
#             return self.forward_bitnet(x)
#         elif isinstance(self.num_bits, int) and self.num_bits >= 16:
#             return F.linear(x, self.weight)
#         else:
#             return self.forward_quantized(x)

#     def _quantize_weights(self):
#         """Helper to compute the simulated quantized weight for inference."""
#         if self.num_bits == "bitnet":
#             w_centered = self.weight - self.weight.mean()
#             w_ternary = torch.sign(w_centered)
#             w_final = self.weight + (w_ternary - self.weight).detach()
#         elif isinstance(self.num_bits, int) and self.num_bits < 16:
#             w_centered = self.weight - self.weight.mean()
#             if self.num_bits == 1:
#                 w_quant = torch.sign(w_centered)
#             else:
#                 q_min = -2.**(self.num_bits - 1)
#                 q_max = 2.**(self.num_bits - 1) - 1
#                 w_scale = w_centered.abs().max() / q_max
#                 w_quant = torch.round(w_centered / w_scale.clamp(min=1e-5)).clamp(q_min, q_max)
#                 w_quant = w_quant * w_scale # De-quantize back to float
#             w_final = self.weight + (w_quant - self.weight).detach()
#         else:
#             w_final = self.weight

#         self.quantized_w = w_final

#     def forward_quantized(self, x):
#         # Activation quantization (on-the-fly)
#         scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
#         x_quant = (x * scale).round().clamp(-128, 127) / scale
#         x_final = x + (x_quant - x).detach() # STE for activations

#         # Use the cached float weight (simulating quantization)
#         w_final = self.quantized_w if not self.training else self.weight
#         return F.linear(x_final, w_final)

#     def forward_bitnet(self, x):
#         # Activation quantization (on-the-fly)
#         x_codebook = torch.sign(x)
#         x_final = x + (x_codebook - x).detach() # STE for activations

#         # Use the cached float weight (simulating quantization)
#         w_final = self.quantized_w if not self.training else self.weight
#         return F.linear(x_final, w_final)

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weight)

# class BitNetExpert(nn.Module):
#     def __init__(self, in_dim, num_classes, hidden_sizes, dropout_prob, num_bits=16):
#         super().__init__()
#         layers = []
#         last_dim = in_dim
#         for hidden_dim in hidden_sizes:
#             layers.append(BitLinear(last_dim, hidden_dim, num_bits=num_bits))
#             layers.append(nn.LayerNorm(hidden_dim))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(dropout_prob))
#             last_dim = hidden_dim
#         layers.append(BitLinear(last_dim, num_classes, num_bits=num_bits))
#         self.net = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.net(x)

# if __name__ == "__main__":
#     in_dim = 128
#     num_classes = 10
#     hidden_sizes = [512, 256, 64, 32]
#     dropout_prob = 0.1

#     # --- FIX: Force the script to run on the CPU ---
#     device = torch.device("cpu")
#     print(f"Using device: {device}")

#     dummy_input = torch.randn(4, in_dim).to(device)

#     test_models = {
#         "BitNetExpert (FP16/BF16/Full-Precision)": BitNetExpert(in_dim, num_classes, hidden_sizes, dropout_prob, num_bits=16),
#         "BitNetExpert (8-bit Quant)": BitNetExpert(in_dim, num_classes, hidden_sizes, dropout_prob, num_bits=8),
#         "BitNetExpert (4-bit Quant)": BitNetExpert(in_dim, num_classes, hidden_sizes, dropout_prob, num_bits=4),
#         "BitNetExpert (2-bit Quant)": BitNetExpert(in_dim, num_classes, hidden_sizes, dropout_prob, num_bits=2),
#         "BitNetExpert (1-bit Quant)": BitNetExpert(in_dim, num_classes, hidden_sizes, dropout_prob, num_bits=1),
#         "BitNetExpert (Ternary 'bitnet' STE)": BitNetExpert(in_dim, num_classes, hidden_sizes, dropout_prob, num_bits="bitnet"),
#         "BitNetExpert158b (Ternary Integer MatMul)": BitNetExpert158b(in_dim, num_classes, hidden_sizes, dropout_prob),
#         "BitNetPopcountExpert (Ternary Bitwise Popcount)": BitNetPopcountExpert(in_dim, num_classes, hidden_sizes, dropout_prob),
#         "ESCModel (FP32 Baseline)": ESCModel(in_dim, num_classes, hidden_sizes, dropout_prob)
#     }

#     # Move all models to the target device
#     for name, model in test_models.items():
#         model.to(device)

#     batch_size = 128
#     num_warmup = 10
#     num_runs = 100

#     benchmark_input = torch.randn(batch_size, in_dim).to(device)

#     models_to_benchmark = {name: model.eval() for name, model in test_models.items()}

#     results_times = {}

#     for name, model in models_to_benchmark.items():
#         with torch.no_grad():
#             # Warmup runs to stabilize performance metrics
#             for _ in range(num_warmup):
#                 _ = model(benchmark_input)

#             # --- FIX: Removed torch.cuda.synchronize() as it's not needed for CPU ---
#             start_time = time.perf_counter()
#             for _ in range(num_runs):
#                 _ = model(benchmark_input)
#             end_time = time.perf_counter()

#         avg_time_ms = ((end_time - start_time) / num_runs) * 1000
#         results_times[name] = avg_time_ms

#     results_memory_mb = {}

#     # Measure memory usage on CPU for consistency
#     esc_model_for_mem = models_to_benchmark["ESCModel (FP32 Baseline)"]
#     esc_model_for_mem_cpu = esc_model_for_mem.to('cpu')
#     fp32_baseline_real_mem_mb = print_size_of_model(esc_model_for_mem_cpu, "ESCModel (FP32 Baseline)")
#     results_memory_mb["ESCModel (FP32 Baseline)"] = fp32_baseline_real_mem_mb

#     for name, model in models_to_benchmark.items():
#         if name == "ESCModel (FP32 Baseline)":
#             continue
#         model_cpu = model.to('cpu')
#         measured_mb = print_size_of_model(model_cpu, name)
#         results_memory_mb[name] = measured_mb

#     csv_data = [["Model Name", "Average Inference Time (ms)", "X Times Faster (vs ESCModel FP32)", "Real Storage (MB)", "Real Storage Reduction (vs ESCModel FP32)"]]

#     # Reorder results for clear comparison
#     desired_order = [
#         "ESCModel (FP32 Baseline)",
#         "BitNetExpert (FP16/BF16/Full-Precision)",
#         "BitNetExpert (8-bit Quant)",
#         "BitNetExpert (4-bit Quant)",
#         "BitNetExpert (2-bit Quant)",
#         "BitNetExpert (1-bit Quant)",
#         "BitNetExpert (Ternary 'bitnet' STE)",
#         "BitNetExpert158b (Ternary Integer MatMul)",
#         "BitNetPopcountExpert (Ternary Bitwise Popcount)",
#     ]

#     esc_baseline_time = results_times.get("ESCModel (FP32 Baseline)", 1.0)
#     if esc_baseline_time == 0: esc_baseline_time = 1.0

#     for name in desired_order:
#         if name not in results_times:
#             continue

#         avg_time = results_times.get(name, "N/A")
#         real_mem_mb = results_memory_mb.get(name, "N/A")

#         time_x_faster = "N/A"
#         if isinstance(avg_time, (int, float)) and avg_time > 0:
#             time_x_faster = f"{esc_baseline_time / avg_time:.2f}x"

#         mem_reduction = "N/A"
#         if isinstance(real_mem_mb, (int, float)) and fp32_baseline_real_mem_mb > 0:
#             mem_reduction = f"{fp32_baseline_real_mem_mb / real_mem_mb:.1f}x"

#         csv_data.append([
#             name,
#             f"{avg_time:.4f}",
#             time_x_faster,
#             f"{real_mem_mb:.2f}",
#             mem_reduction
#         ])

#     fastest_model_name = min(results_times, key=results_times.get)
#     fastest_time_ms = results_times[fastest_model_name]

#     csv_data.append([])
#     csv_data.append(["Fastest Overall Model", fastest_model_name, f"{fastest_time_ms:.4f} ms"])

#     # Print results to console
#     print("\n--- Benchmark Results ---")
#     for row in csv_data:
#         print(f"{' | '.join(map(str, row))}")

#     # Save results to CSV
#     csv_file = "results_cpu_only.csv"
#     try:
#         with open(csv_file, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerows(csv_data)
#         print(f"\nResults saved to {csv_file}")
#     except IOError:
#         print(f"\nCould not save results to {csv_file}")