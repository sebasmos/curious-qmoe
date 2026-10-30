"""T22 clean coarse-view cache (ANALYSIS_PLAN.md C2): padded clean 16 kHz waveform -> t22_view.logmel, float32 [rows, 64, frames],
for every ESC-50 and UrbanSound8K clip and the TAU device A clips. CPU only.   python t22_cache.py --task esc50"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t22_common import *  # noqa
import t22_view as V
from audio_shift import index_wavs, load_clip

ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--bs", type=int, default=64); a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 2)))
T = TASKS[a.task]; B = bed(a.task); st = stems(a.task)
assert names_crc(a.task) == B["crc"], "CSV row order differs from the stored arrays"
rows = cache_rows(a.task, B); idx = index_wavs(T["aud"])
d = CACHE / a.task; d.mkdir(parents=True, exist_ok=True)
fr = V.frames_of(V.PAD16[a.task])
X = np.lib.format.open_memmap(d / "clean_mel.npy", mode="w+", dtype=np.float32, shape=(len(rows), V.N_MELS, fr))
lens = np.zeros(len(rows), np.int64); t0 = time.time()
for s in range(0, len(rows), a.bs):
    rr = rows[s:s + a.bs]
    w = [load_clip(idx[st[r]]) for r in rr]
    lens[s:s + len(rr)] = [len(x) for x in w]
    x = torch.from_numpy(np.stack([V.pad16(v, a.task) for v in w]))
    with torch.no_grad():
        X[s:s + len(rr)] = V.logmel(x)[:, 0].numpy()
    if (s // a.bs) % 20 == 0:
        print(f"{a.task} {s + len(rr)}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
X.flush(); np.save(d / "clean_rows.npy", rows); np.save(d / "clean_lengths16k.npy", lens)
assert np.isfinite(np.asarray(X)).all()
(d / "meta.json").write_text(json.dumps({"rows": len(rows), "frames": fr, "n_mels": V.N_MELS, "pad16": V.PAD16[a.task], "seconds": time.time() - t0,
                                         "names_crc32": int(B["crc"]), "max_len16": int(lens.max())}, indent=1))
print("done", a.task, X.shape, f"{time.time() - t0:.0f}s", "max len", lens.max(), flush=True)
