"""Measured CPU latency of the view experts (ANALYSIS_PLAN_T15.md section 3): front end + FP32 backbone kernels, batch 1, 2 threads, median (and p95) of 50 runs after
10 warm-ups, one clip of the dataset's length (ESC-50 5 s, UrbanSound8K 4 s, TAU 10 s). W8A8 is fake quantization in float, so no int8 speed is claimed.
python t15_latency.py --r R"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from QWave import efficientat_mn as M, panns_cnn14 as P
import t15_views as V
ap = argparse.ArgumentParser(); ap.add_argument("--r", type=Path, required=True); ap.add_argument("--views", nargs="+", default=V.ALL_VIEWS); a = ap.parse_args()
torch.set_num_threads(2); torch.manual_seed(0)
W = Path.home() / "Documents/qmoe-weights"; base = M.load_mn("mn10", W / "efficientat_release-v0.0.1/mn10_as_mAP_471.pt").eval(); cnn14 = P.load_cnn14(W / "panns-cnn14/Cnn14_mAP=0.431.pth").eval()
LEN = {"urban": 64000, "esc50": 80000, "tau": 160000}; res = {}
def timeit(fn, n=50, warm=10):
    for _ in range(warm): fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); t.append(1000 * (time.perf_counter() - t0))
    return dict(median_ms=float(np.median(t)), p95_ms=float(np.percentile(t, 95)))
with torch.no_grad():
    for ds, n in LEN.items():
        w = torch.randn(1, n) * 0.1; res[ds] = {}
        for v in a.views:
            net = cnn14 if v == "v6" else base
            x = V.mel_inputs(w, [v])[v]
            fe = timeit(lambda: V.mel_inputs(w, [v])); bb = timeit(lambda: net(x))
            res[ds][v] = dict(frontend=fe, backbone_fp32=bb, total_median_ms=fe["median_ms"] + bb["median_ms"], note="FP32 kernels; W8A8 fake quantization is not an int8 speed")
            print(ds, v, f"{res[ds][v]['total_median_ms']:.1f} ms", flush=True)
(a.r / "inputs").mkdir(exist_ok=True); (a.r / "inputs/view_latency.json").write_text(json.dumps(res, indent=1))
