"""T12 cost measurement: step time and memory of fine-tuning the last k AST blocks on MPS (no labels, random inputs of the real shapes).
    python t12_cost.py --out FILE.json [--device mps]"""
import argparse, json, time, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t12_common import *
ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--device", default="mps")
ap.add_argument("--L", type=int, default=400); ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4]); ap.add_argument("--bs", type=int, nargs="+", default=[16, 32, 64])
ap.add_argument("--steps", type=int, default=20); ap.add_argument("--only_cached", action="store_true")
a = ap.parse_args(); dev = torch.device(a.device)
sync = torch.mps.synchronize if a.device == "mps" else (lambda: None)
mem = (lambda: torch.mps.driver_allocated_memory() / 2**30) if a.device == "mps" else (lambda: 0.0)
ext, model = load_ast(a.L, dev); T = n_tokens(a.L); res = []
torch.manual_seed(0)


def bench(k, mode, cached, bs):
    B = NL - k; tail = Tail(model, B, mode)
    opt = torch.optim.AdamW(tail.param_groups(1e-4, 1e-3)); n_tr = sum(p.numel() for p in tail.parameters() if p.requires_grad)
    y = torch.randint(0, 10, (bs,), device=dev)
    tok = torch.randn(bs, T, 768, device=dev); feats = torch.randn(bs, a.L, 128, device=dev)
    ts, peak = [], 0.0
    for i in range(6 + a.steps):
        sync(); t0 = time.perf_counter()
        h = tok if cached else front(model, feats, B)
        tail.train(); loss = torch.nn.functional.cross_entropy(tail(h), y)
        opt.zero_grad(); loss.backward(); opt.step(); sync()
        ts.append(time.perf_counter() - t0); peak = max(peak, mem())
    return dict(k=k, mode=mode, cached=cached, bs=bs, L=a.L, tokens=T, trainable_params=n_tr, step_s=float(np.median(ts[6:])), step_s_p90=float(np.percentile(ts[6:], 90)),
                clips_per_s=bs / float(np.median(ts[6:])), mps_driver_gb_sampled=peak)


for cached in ([True] if a.only_cached else [True, False]):
    for k in a.ks:
        for mode in ("lora", "full"):
            for bs in a.bs:
                try: r = bench(k, mode, cached, bs)
                except RuntimeError as e: r = dict(k=k, mode=mode, cached=cached, bs=bs, error=str(e)[:200])
                res.append(r); print(r, flush=True); torch.mps.empty_cache() if a.device == "mps" else None
# forward-only throughput: whole model (no grad) and the tail from cache
for bs in (32,):
    feats = torch.randn(bs, a.L, 128, device=dev)
    with torch.no_grad():
        for i in range(8):
            if i == 3: sync(); t0 = time.perf_counter()
            front(model, feats, NL)
        sync(); full_fwd = (time.perf_counter() - t0) / 5 / bs
    res.append(dict(kind="forward_only_full_model_s_per_clip", bs=bs, L=a.L, value=full_fwd)); print(res[-1], flush=True)
a.out.write_text(json.dumps(res, indent=1))
