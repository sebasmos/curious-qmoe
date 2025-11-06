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


def print_size_of_model(model, label=""):
    """
    Calculate model size accounting for quantization.
    For quantized models, calculates theoretical size based on bit-width.
    For full-precision models, saves and measures actual file size.
    """
    from QWave.bitnnet import BitLinear
    from QWave.qmoe_layers import BitwisePopcountLinear

    total_bits = 0
    has_quantized_layers = False

    for name, param in model.named_parameters():
        # Check if this parameter belongs to a quantized layer
        module_name = '.'.join(name.split('.')[:-1])  # Get module path without param name
        param_name = name.split('.')[-1]  # weight, bias, etc.

        # Try to get the actual module
        module = model
        if module_name:
            for attr in module_name.split('.'):
                module = getattr(module, attr, None)
                if module is None:
                    break

        # Determine bit-width based on layer type
        if isinstance(module, BitLinear):
            has_quantized_layers = True
            if param_name == 'weight':
                if module.num_bits == "bitnet":
                    # Ternary: {-1, 0, 1} = 2 bits per weight
                    # Plus alpha scale per output channel (32-bit float)
                    num_weights = param.numel()
                    ternary_bits = num_weights * 2  # 2 bits per ternary value
                    alpha_bits = param.shape[0] * 32  # One alpha per output channel
                    total_bits += ternary_bits + alpha_bits
                elif isinstance(module.num_bits, int):
                    # k-bit quantization (1, 2, 4, 8, 16 bits)
                    bits = module.num_bits
                    # Add scale factor overhead (one 32-bit float per layer for symmetric quant)
                    total_bits += param.numel() * bits + 32
                else:
                    # Fallback to float32
                    total_bits += param.numel() * 32
            else:
                # bias and other params remain float32
                total_bits += param.numel() * 32

        elif isinstance(module, BitwisePopcountLinear):
            has_quantized_layers = True
            if param_name == 'weight':
                # Ternary with 2-bit encoding per value
                total_bits += param.numel() * 2
            else:
                total_bits += param.numel() * 32

        else:
            # Full-precision parameter (float32)
            total_bits += param.numel() * 32

    if has_quantized_layers:
        # Return theoretical quantized size
        size_bytes = total_bits / 8
        print(f"model: {label} \t Size (KB): {size_bytes/1e3:.3f} [quantized]")
        return size_bytes
    else:
        # Fallback: save and measure for non-quantized models
        torch.save(model.state_dict(), "temp.p")
        size = os.path.getsize("temp.p")
        print(f"model: {label} \t Size (KB): {size/1e3:.3f}")
        os.remove('temp.p')
        return size
    
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return bytes_to_mb(memory_info.rss)  # Returns the memory usage in bytes

def bytes_to_mb(memory_bytes):
    """
    Takes the memory usage in bytes as input and returns the memory usage converted to megabytes (MB).
    """
    return memory_bytes / (1024 * 1024)
