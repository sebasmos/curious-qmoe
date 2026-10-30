"""T17(c) data: normalised AST filter banks (400 frames) of the UrbanSound8K fold-1 training clips (clean), fp16, for on-the-fly SpecAugment and mixup. CPU only, reads no label.
    python t17c_fbank.py --out DIR"""
import argparse, os, sys, time
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--threads", type=int, default=1); ap.add_argument("--limit", type=int, default=None); a = ap.parse_args()
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = str(a.threads)
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t17c_ast import *
from t17_common import T1C, load_bed
from common import load_table
from audio_shift import index_wavs, load_clip
torch.set_num_threads(a.threads)
AUD = Path.home() / "Documents/DATASETS/Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio"
S = np.load(T1C / "urban/raw_strength/s0_f1.npz"); tr = S["tr"]
if a.limit: tr = tr[:a.limit]
stems = load_table().stem.tolist(); index = index_wavs(AUD)
from transformers import ASTFeatureExtractor
ext = ASTFeatureExtractor.from_pretrained(MODEL, revision=REV, max_length=400)
a.out.mkdir(parents=True, exist_ok=True); out = np.lib.format.open_memmap(a.out / "train_fold1_fbank.npy", mode="w+", dtype=np.float16, shape=(len(tr), 400, 128)); t0 = time.time()
for s in range(0, len(tr), 64):
    r = tr[s:s + 64]; out[s:s + len(r)] = ext([load_clip(index[stems[i]]) for i in r], sampling_rate=16000, return_tensors="pt")["input_values"].numpy().astype(np.float16)
    if (s // 64) % 20 == 0: print(s, len(tr), "%.0fs" % (time.time() - t0), flush=True)
out.flush(); np.save(a.out / "train_fold1_rows.npy", tr); print("done", out.shape, "%.0fs" % (time.time() - t0))
