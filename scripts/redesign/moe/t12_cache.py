"""T12 token cache: tokens entering block B of the FP32 AST (fp16 memmap) and the frozen pooled embedding (fp32), for UrbanSound8K rows under one condition.
    python t12_cache.py --cond clean|bg:0|field --rows all|fold:1 --block 10 --max-length 400 --out DIR [--check]
Files: <name>_tok.npy [n,T,768] fp16, <name>_frozen_emb.npy [n,768] fp32, <name>_rows.npy. field = the test field-mix assignment (seed 5001), rows restricted by --rows. No label is read."""
import argparse, json, os, sys, time
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--cond", required=True); ap.add_argument("--rows", default="all"); ap.add_argument("--block", type=int, default=10)
ap.add_argument("--max-length", type=int, default=400); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--device", default="mps")
ap.add_argument("--threads", type=int, default=2); ap.add_argument("--chunk", type=int, default=64); ap.add_argument("--check", action="store_true"); ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = str(a.threads)
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t12_common import *
from common import load_table, Q, S2
from audio_shift import FIELD_SEEDS, NoiseMaker, field_assignment, index_wavs, load_clip, mix
torch.set_num_threads(a.threads)
E = Path.home() / "Documents/DATASETS"; AUD = E / "Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio"; BGA = E / "ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio"
d = load_table(); N = len(d); stems = d.stem.tolist(); fold = d.folder.values
rows = np.arange(N) if a.rows == "all" else np.flatnonzero(fold == int(a.rows.split(":")[1]))
if a.limit: rows = rows[:a.limit]
if a.cond == "clean": cond_of = {int(r): ("clean", None) for r in rows}; name = "clean"
elif a.cond.startswith("bg:"): cond_of = {int(r): ("bg", float(a.cond[3:])) for r in rows}; name = f"bg{int(float(a.cond[3:]))}"
elif a.cond == "field":
    asg = field_assignment(N, FIELD_SEEDS["test"]); name = "field"
    cond_of = {int(r): (("clean", None) if asg[r] == "clean" else (asg[r].split("__")[0], float(asg[r].split("__")[1][:-2]))) for r in rows}
else: raise SystemExit(a.cond)
name = f"fold{a.rows.split(':')[1]}_{name}" if a.rows != "all" else name
index = index_wavs(AUD); bg_names = [Path(s).stem for s in pd.read_csv(E / "ESC-50/ast_768/esc-50.csv", usecols=["name"])["name"]]
bg_index = index_wavs(BGA); nm = NoiseMaker([b for b in bg_names if b in bg_index], bg_index, True)
dev = torch.device(a.device); ext, model = load_ast(a.max_length, dev); T = n_tokens(a.max_length); B = a.block
a.out.mkdir(parents=True, exist_ok=True)


def wave(r):
    x = load_clip(index[stems[r]]); kind, snr = cond_of[int(r)]
    if kind != "clean": x, _ = mix(x, nm(stems[r], kind, len(x)), snr)
    return x


@torch.no_grad()
def run(idx):
    feats = ext([wave(r) for r in idx], sampling_rate=16000, return_tensors="pt")["input_values"].contiguous().to(dev)
    h = front(model, feats, B); tok = h.half().cpu().numpy(); emb = h
    for l in model.encoder.layer[B:]: emb = layer_fwd(l, emb)
    emb = model.layernorm(emb); emb = ((emb[:, 0] + emb[:, 1]) / 2).float().cpu().numpy()
    return tok, emb, h.float()


if a.check:
    from QWave.backbone_quant import QuantAST, parse_config
    sub = rows[:: max(1, len(rows) // 64)][:64]; feats = ext([wave(r) for r in sub], sampling_rate=16000, return_tensors="pt")["input_values"].contiguous().to(dev)
    q = QuantAST(model); q.set_config(*parse_config("w4g128")); ref = np.concatenate([q(feats[i:i + 16]).float().cpu().numpy() for i in range(0, 64, 16)])
    stored = np.load(S2 / "emb_ast_short/urban8k/w4g128__clean.npy", mmap_mode="r")[sub]
    rel = lambda x, y: np.linalg.norm(x - y, axis=1) / np.linalg.norm(y, axis=1)
    tok, emb, h32 = run(sub); fp16_tail = []
    with torch.no_grad():
        h = torch.from_numpy(tok).to(dev).float()
        for l in model.encoder.layer[B:]: h = layer_fwd(l, h)
        h = model.layernorm(h); fp16_tail = ((h[:, 0] + h[:, 1]) / 2).cpu().numpy()
    res = dict(rows=sub.tolist(), w4g128_vs_stored_E2c_rel_l2_median=float(np.median(rel(ref, stored))), w4g128_vs_stored_E2c_rel_l2_max=float(rel(ref, stored).max()),
               fp16_cache_vs_fp32_pooled_rel_l2_median=float(np.median(rel(fp16_tail, emb))), fp16_cache_vs_fp32_pooled_rel_l2_max=float(rel(fp16_tail, emb).max()),
               token_abs_max=float(np.abs(tok.astype(np.float32)).max()), tokens=T, block=B)
    (a.out / "check_cache.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1)); sys.exit(0)

tokp = a.out / f"{name}_tok.npy"; mm = np.lib.format.open_memmap(tokp, mode="w+", dtype=np.float16, shape=(len(rows), T, 768)); E_ = np.zeros((len(rows), 768), np.float32)
t0 = time.time()
for s in range(0, len(rows), a.chunk):
    idx = rows[s:s + a.chunk]; tok, emb, _ = run(idx); mm[s:s + len(idx)] = tok; E_[s:s + len(idx)] = emb
    if (s // a.chunk) % 10 == 0: print(f"{name}: {s + len(idx)}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
mm.flush(); assert np.isfinite(E_).all() and np.isfinite(mm[:: max(1, len(rows) // 50)].astype(np.float32)).all()
np.save(a.out / f"{name}_frozen_emb.npy", E_); np.save(a.out / f"{name}_rows.npy", rows)
(a.out / f"{name}_meta.json").write_text(json.dumps(dict(cond=a.cond, rows=a.rows, block=B, tokens=T, max_length=a.max_length, n=len(rows), seconds=time.time() - t0,
    tok_bytes=int(os.path.getsize(tokp)), device=a.device, model=MODEL, revision=REV), indent=1))
print(f"{name}: done {len(rows)} rows in {time.time() - t0:.0f}s, {os.path.getsize(tokp) / 2**30:.2f} GB", flush=True)
