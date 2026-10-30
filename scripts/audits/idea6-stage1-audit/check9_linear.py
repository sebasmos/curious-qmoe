"""Check 9: is the 'unseeded' 1536 x 1536 Linear of the EfficientNet extractor recoverable?

A fresh torch process starts from a fixed default seed, so building efficientnet_b3 (its __init__
draws every initial weight) and then nn.Linear(1536, 1536), as the extractor does, may give the same
W and b every time. If the stored embeddings E were made as E = f W^T + b with f the pooled
EfficientNet features (averages of SiLU outputs, so every entry >= -0.2785), then f = (E - b) W^-T
must respect that bound on every clip. A wrong W gives entries spread on both sides of zero.
Run in a fresh process (nothing else may touch the RNG first).
    python check9_linear.py OUT.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torchvision.models import efficientnet_b3
res = {"torch_initial_seed": int(torch.initial_seed())}
m = efficientnet_b3(weights=None)                 # same architecture and init draws as weights="IMAGENET1K_V1"
lin = nn.Linear(m.classifier[1].in_features, m.classifier[1].in_features)
W = lin.weight.detach().double().numpy(); b = lin.bias.detach().double().numpy()
res["W_checksum"] = float(W.sum()); res["W_00"] = float(W[0, 0])
# control: a Linear drawn from a different RNG state
g = torch.Generator().manual_seed(1)
Wc = (torch.rand(1536, 1536, generator=g, dtype=torch.float64) * 2 - 1).numpy() / np.sqrt(1536)
bc = (torch.rand(1536, generator=g, dtype=torch.float64) * 2 - 1).numpy() / np.sqrt(1536)
H = Path.home() / "Documents/DATASETS"
for ds, csv in {"esc50": H / "ESC-50/efficientnet_1536/esc-50.csv", "urban": H / "Urban8k/efficientnet/urbansound8k.csv"}.items():
    E = pd.read_csv(csv).loc[:, [str(i) for i in range(1536)]].to_numpy(np.float64)
    out = {}
    for tag, (WW, bb) in {"default_seed": (W, b), "control_other_rng": (Wc, bc)}.items():
        f = np.linalg.solve(WW, (E - bb).T).T
        silu_min = -0.27846455
        out[tag] = {"min": float(f.min()), "max": float(f.max()), "frac_below_silu_min": float((f < silu_min - 1e-4).mean()),
                    "median": float(np.median(f)),
                    "max_abs_reconstruction_error": float(np.abs(f @ WW.T + bb - E).max())}
    res[ds] = out
    print(ds, json.dumps(out, indent=1), flush=True)
Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
