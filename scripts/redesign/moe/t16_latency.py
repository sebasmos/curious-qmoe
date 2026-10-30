"""T16 measured CPU latency (batch 1, FP32 kernels, 2 threads, median of 50 runs after 10 warm-ups): the Kaldi fbank, the AST embedding layer plus b blocks, the remaining blocks plus LayerNorm and the 768-d head,
and the router MLP, on one full-length clip of UrbanSound8K (400 frames) and of ESC-50 (512 frames). E1's latency is not re-measured. python t16_latency.py --out DIR"""
import argparse, json, os, sys, time
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--warmup", type=int, default=10); ap.add_argument("--runs", type=int, default=50); a = ap.parse_args()
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = str(a.threads)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, torch
from torch import nn
from t16_common import *
from audio_shift import index_wavs, load_clip
torch.set_num_threads(a.threads)


def timeit(fn):
    for _ in range(a.warmup): fn()
    t = []
    for _ in range(a.runs): t0 = time.perf_counter(); fn(); t.append((time.perf_counter() - t0) * 1000)
    t = np.array(t); return dict(median_ms=float(np.median(t)), q25_ms=float(np.percentile(t, 25)), q75_ms=float(np.percentile(t, 75)))


res = {"threads": a.threads, "runs": a.runs, "warmup": a.warmup, "torch": torch.__version__}
for bed in ("urban", "esc50"):
    csv, aud, _, _ = csv_and_audio(bed); names = pd.read_csv(csv, usecols=["name"])["name"]; idx = index_wavs(aud); L = FRAMES[bed]; K = BEDS[bed]["ncls"]
    need = 63840 if bed == "urban" else 79000
    for nme in names:
        w = load_clip(idx[Path(nme).stem])
        if len(w) >= need: break
    ext, model = load_ast(L, "cpu"); head = nn.Linear(768, K).eval(); router = nn.Sequential(nn.Linear(1536 + K + 1, 128), nn.ReLU(), nn.Linear(128, 1)).eval()
    feats = ext([w], sampling_rate=16000, return_tensors="pt")["input_values"].contiguous(); o = {"clip": Path(nme).stem}
    with torch.no_grad():
        o["fbank"] = timeit(lambda: ext([w], sampling_rate=16000, return_tensors="pt"))
        for b in (4, 6, 8, 10, 12):
            def front():
                h = model.embeddings(feats)
                for l in model.encoder.layer[:b]: h = layer_fwd(l, h)
                return h
            o[f"embed_plus_{b}_blocks"] = timeit(front)
            h0 = front()
            def rest(h0=h0, b=b):
                h = h0
                for l in model.encoder.layer[b:]: h = layer_fwd(l, h)
                h = model.layernorm(h); return head((h[:, 0] + h[:, 1]) / 2)
            o[f"remaining_{12 - b}_blocks_plus_head"] = timeit(rest)
        x = torch.randn(1, 1536 + K + 1); o["router_mlp"] = timeit(lambda: router(x))
    o["tokens"] = int(model.embeddings(feats).shape[1]); res[bed] = o
a.out.mkdir(parents=True, exist_ok=True); (a.out / "latency.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
