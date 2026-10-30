"""Load a trained coarse expert (FP32 and the W8A8 tier) from a T22 checkpoint, on the CPU."""
from pathlib import Path
import numpy as np, torch
import t22_view as V
from QWave import efficientat_mn as M


def load_coarse(path: Path, K: int):
    m = V.Coarse(M.build_mn("mn10"), K)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return m.eval()


@torch.no_grad()
def run(f, X: np.ndarray, bs: int = 64) -> np.ndarray:
    """f: callable on [b, 1, 64, T] -> logits; X [n, 64, T] float32."""
    out = [f(torch.from_numpy(np.ascontiguousarray(X[s:s + bs])).unsqueeze(1)).float().numpy() for s in range(0, len(X), bs)]
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)
