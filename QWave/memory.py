import numpy as np
import os
import psutil
import torch

from pathlib import Path
import os, torch
import pandas as pd


def _load_cc_csv(csv_path: Path) -> dict:
    if not csv_path.is_file():
        return {}
    df = pd.read_csv(csv_path)
    return df.iloc[-1].to_dict() if len(df) else {}


def storage_bits(model, breakdown=None) -> int:
    """Bits needed to store `model` at the precision it is deployed at.

    BitLinear weights: k bits each plus one fp32 scale per output row for
    k-bit (k < 16, 1-bit included), 2-bit codes plus a per-row fp32 scale for
    ternary BitNet, 16 bits for fp16, 32 for fp32. Dynamic int8 Linear (PTQ):
    8 bits per weight plus an fp32 scale and zero point. Ternary BitwiseLinear
    and popcount weights: 2 bits. Everything else (biases, LayerNorm, routers,
    fp32 classifier heads): 32 bits per parameter. Buffers are not counted.
    Pass a list as `breakdown` to collect (module name, type, bits) rows.
    """
    from QWave.qmoe_layers import BitLinear, BitwiseLinear, BitwisePopcountLinear
    from torch.ao.nn.quantized.dynamic import Linear as DynamicInt8Linear

    total = 0
    for name, m in model.named_modules():
        if isinstance(m, BitLinear):
            n_w, rows, nb = m.weight.numel(), m.weight.shape[0], m.num_bits
            if nb == "bitnet":
                bits = 2 * n_w + 32 * rows
            elif nb >= 32:
                bits = 32 * n_w
            elif nb == 16:
                bits = 16 * n_w
            else:
                bits = nb * n_w + 32 * rows
            bits += 32 * m.bias.numel() if m.bias is not None else 0
            kind = f"BitLinear({nb})"
        elif isinstance(m, (BitwiseLinear, BitwisePopcountLinear)):
            bits, kind = 2 * m.weight.numel(), type(m).__name__
        elif isinstance(m, DynamicInt8Linear):
            b = m.bias()
            bits = 8 * m.weight().numel() + 64 + (32 * b.numel() if b is not None else 0)
            kind = "DynamicInt8Linear"
        else:
            bits = 32 * sum(p.numel() for p in m.parameters(recurse=False))
            kind = type(m).__name__
        if bits and breakdown is not None:
            breakdown.append((name, kind, bits))
        total += bits
    return total


def print_size_of_model(model, label="", debug=False):
    """Model storage in bytes at its deployed precision (see storage_bits).

    This used to recognize only QWave.bitnnet.BitLinear, so the models built
    from QWave.qmoe_layers fell through to serializing fp32 weights and every
    bit-width reported the same size."""
    rows = [] if debug else None
    size_bytes = storage_bits(model, rows) / 8
    if debug:
        print(f"\n=== DEBUG: Model {label} ===")
        for name, kind, bits in rows:
            print(f"  {name:40s} | {kind:25s} | {bits / 8 / 1e3:.3f} KB")
    print(f"model: {label} \t Size (KB): {size_bytes/1e3:.3f}")
    return size_bytes
    
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return bytes_to_mb(memory_info.rss)  # Returns the memory usage in bytes

def bytes_to_mb(memory_bytes):
    """
    Takes the memory usage in bytes as input and returns the memory usage converted to megabytes (MB).
    """
    return memory_bytes / (1024 * 1024)
