"""T20 training cache: normalised AST filter banks of the clean clips (float16, the clips' real length) for on-the-fly SpecAugment and mixup,
as T17 part (c) (`t17c_fbank.py`) for any task. TAU: device-A clips only (the training pool). CPU only, reads no label.
    python t20_fbank.py --task urban|esc50|tau [--threads 2] [--limit N]"""
import argparse, os, sys, time
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--out", type=Path, default=None); a = ap.parse_args()
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = str(a.threads)
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "idea6/stage2"))
from t20_common import *
from t17c_ast import MODEL, REV
from audio_shift import index_wavs, load_clip
torch.set_num_threads(a.threads)
T = TASKS[a.task]; D = bed(a.task); st = stems(a.task); rows = cache_rows(a.task, D)
if a.limit: rows = rows[:a.limit]
index = index_wavs(T["aud"])
from transformers import ASTFeatureExtractor
ext = ASTFeatureExtractor.from_pretrained(MODEL, revision=REV, max_length=T["L"]); assert ext.max_length == T["L"]
out_d = a.out or CACHE / a.task; out_d.mkdir(parents=True, exist_ok=True)
mm = np.lib.format.open_memmap(out_d / "clean_fbank.partial.npy", mode="w+", dtype=np.float16, shape=(len(rows), T["L"], 128)); t0 = time.time()
for s in range(0, len(rows), 64):
    r = rows[s:s + 64]; mm[s:s + len(r)] = ext([load_clip(index[st[i]]) for i in r], sampling_rate=16000, return_tensors="pt")["input_values"].numpy().astype(np.float16)
    if (s // 64) % 20 == 0: print(s, len(rows), "%.0fs" % (time.time() - t0), flush=True)
mm.flush(); assert np.isfinite(mm[:: max(1, len(rows) // 100)].astype(np.float32)).all(); del mm
os.replace(out_d / "clean_fbank.partial.npy", out_d / "clean_fbank.npy"); np.save(out_d / "clean_rows.npy", rows)
print("done", len(rows), "%.0fs" % (time.time() - t0), flush=True)
