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
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
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
