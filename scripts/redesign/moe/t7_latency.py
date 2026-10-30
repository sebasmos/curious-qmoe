"""T7 measured CPU latency (ANALYSIS_PLAN_T7.md section 5): batch size 1, one UrbanSound8K clip (first full-length clip in CSV order), FP32 kernels (no low-bit kernels exist here).
Steps: Kaldi fbank extractor, energy selector, AST encoder with the gathered patch embedding at 470 (all patches), 236 and 119 tokens, the 768-d head.
    python t7_latency.py --out DIR [--threads 2] [--warmup 10] [--runs 50]"""
import argparse, json, os, sys, time
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--warmup", type=int, default=10); ap.add_argument("--runs", type=int, default=50); a = ap.parse_args()
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = str(a.threads)
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/redesign/idea6/stage2"))
import numpy as np, pandas as pd, torch
from audio_shift import index_wavs, load_clip
from QWave.ast_frames import shorten_ast
from QWave.backbone_quant import QuantAST
from QWave.qmoe_layers import BitNetExpert
import t7_mask as M
from transformers import ASTFeatureExtractor, ASTModel
torch.set_num_threads(a.threads)
E = Path.home() / "Documents/DATASETS/Urban8k"
meta = pd.read_csv(E / "ast_768/urbansound8k.csv", usecols=["name"]); idx = index_wavs(E / "audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio")
wave = None
for nme in meta["name"]:
    w = load_clip(idx[Path(nme).stem])
    if len(w) >= 63840: wave = w; break          # 3.99 s at 16 kHz: a full-length clip (398 real frames)
name = Path(nme).stem; L = 400
mdl, rev = "MIT/ast-finetuned-audioset-10-10-0.4593", "f826b80d28226b62986cc218e5cec390b1096902"
ext = ASTFeatureExtractor.from_pretrained(mdl, revision=rev, max_length=L)
qast = QuantAST(shorten_ast(ASTModel.from_pretrained(mdl, revision=rev), L).eval())      # default config: FP32 weights, no activation quantization
head = BitNetExpert(768, 10, [640, 320], 0.2, num_bits=32).eval()
feats = ext([wave], sampling_rate=16000, return_tensors="pt")["input_values"].contiguous()
n_samp = np.array([len(wave)])


def timeit(fn):
    for _ in range(a.warmup): fn()
    t = []
    for _ in range(a.runs):
        t0 = time.perf_counter(); fn(); t.append((time.perf_counter() - t0) * 1000)
    t = np.array(t); return {"median_ms": float(np.median(t)), "q25_ms": float(np.percentile(t, 25)), "q75_ms": float(np.percentile(t, 75))}


res = {"clip": name, "samples": int(len(wave)), "threads": a.threads, "runs": a.runs, "warmup": a.warmup, "torch": torch.__version__}
with torch.no_grad():
    res["fbank_extractor"] = timeit(lambda: ext([wave], sampling_rate=16000, return_tensors="pt"))
    res["selector_energy_25"] = timeit(lambda: M.select("energy", 25, feats, n_samp, [name], L))
    res["selector_energy_50"] = timeit(lambda: M.select("energy", 50, feats, n_samp, [name], L))
    for rho in (25, 50):
        keep = torch.as_tensor(M.select("energy", rho, feats, n_samp, [name], L), dtype=torch.long)
        res[f"encoder_masked_{rho}"] = timeit(lambda: M.masked_forward(qast, feats, keep)); res[f"encoder_masked_{rho}"]["tokens"] = M.k_keep(L, rho) + 2
    res["encoder_full"] = timeit(lambda: M.masked_forward(qast, feats, None)); res["encoder_full"]["tokens"] = 470
    res["encoder_full_qast_forward"] = timeit(lambda: qast(feats))
    e = torch.randn(1, 768); res["head_768"] = timeit(lambda: head(e))
for k in ("25", "50"):
    res[f"total_masked_{k}_ms"] = res["fbank_extractor"]["median_ms"] + res[f"selector_energy_{k}"]["median_ms"] + res[f"encoder_masked_{k}"]["median_ms"] + res["head_768"]["median_ms"]
res["total_full_ms"] = res["fbank_extractor"]["median_ms"] + res["encoder_full"]["median_ms"] + res["head_768"]["median_ms"]
a.out.mkdir(parents=True, exist_ok=True); (a.out / "latency.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
