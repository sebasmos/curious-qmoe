"""T16 extraction (GPU): token-0 and patch-mean statistics of the FP32 AST after blocks 4, 6, 8, 10 and 12 (the last after the final LayerNorm), and the standard pooled embedding.
    python t16_extract.py --bed urban|esc50|tau [--cells clean bg__0db ...] [--check] [--limit N] [--out DIR]
Output per cell: <bed>__<cell>.npy float32 [N,5,2,768] (depth, 0 = token 0 / 1 = mean of tokens 2..), <bed>__<cell>_emb.npy float32 [N,768] ((token0+token1)/2 after the LayerNorm), NaN rows not extracted.
Same audio, noise plan and mixing as idea 5 stage 2 (audio_shift.py). No label is read."""
import argparse, json, os, sys, time, zlib
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--bed", required=True); ap.add_argument("--cells", nargs="*", default=None); ap.add_argument("--out", type=Path, default=None)
ap.add_argument("--device", default="mps"); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--chunk", type=int, default=128)
ap.add_argument("--cfg", default="fp32"); ap.add_argument("--check", action="store_true"); ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[v] = str(a.threads)
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t16_common import *
from audio_shift import NoiseMaker, index_wavs, load_clip, mix
torch.set_num_threads(a.threads)
bed = a.bed; L = FRAMES[bed]; OUT = a.out or (FEAT if a.cfg == "fp32" else FEAT.parent / f"feat_{a.cfg}"); OUT.mkdir(parents=True, exist_ok=True)
csv, aud, bgcsv, bgaud = csv_and_audio(bed)
stems = pd.read_csv(csv, usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist(); N = len(stems)
index = index_wavs(aud); assert all(s in index for s in stems)
nm = NoiseMaker([], {}, False)
if bgcsv is not None:
    bgn = pd.read_csv(bgcsv, usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist(); bgi = index_wavs(bgaud); nm = NoiseMaker([b for b in bgn if b in bgi], bgi, True)
dev = torch.device(a.device)


def wave(i, cell):
    x = load_clip(index[stems[i]])
    if cell != "clean":
        kind, snr = cell.split("__"); x, _ = mix(x, nm(stems[i], kind, len(x)), float(snr[:-2]))
    return x


@torch.no_grad()
def run(model, ext, idx, cell):
    feats = ext([wave(i, cell) for i in idx], sampling_rate=16000, return_tensors="pt")["input_values"].contiguous().to(dev)
    S = np.zeros((len(idx), len(DEPTHS), 2, 768), np.float32); E = np.zeros((len(idx), 768), np.float32)
    for s in range(0, len(idx), a.batch):
        h = model.embeddings(feats[s:s + a.batch])
        for k, layer in enumerate(model.encoder.layer, 1):
            h = layer_fwd(layer, h) if QA is None else QA._layer(h, layer, QA.weights[k - 1], QA.bias[k - 1])
            if k in DEPTHS[:-1]: j = DEPTHS.index(k); S[s:s + len(h), j, 0] = h[:, 0].float().cpu().numpy(); S[s:s + len(h), j, 1] = h[:, 2:].mean(1).float().cpu().numpy()
        h = model.layernorm(h); S[s:s + len(h), -1, 0] = h[:, 0].float().cpu().numpy(); S[s:s + len(h), -1, 1] = h[:, 2:].mean(1).float().cpu().numpy()
        E[s:s + len(h)] = ((h[:, 0] + h[:, 1]) / 2).float().cpu().numpy()
    return S, E


QA = None
if a.check and a.cfg != "fp32":
    from QWave.backbone_quant import QuantAST, parse_config
    ext, model = load_ast(L, dev); QA = QuantAST(model); QA.set_config(*parse_config(a.cfg)); sub = np.arange(N)[:: max(1, N // 64)][:64]
    ref_p = {"urban": S2 / "emb_ast_short/urban8k", "esc50": S2 / "emb_ast_short/esc50", "tau": OPTB / "emb_ast/tau"}[bed] / f"{a.cfg}__clean.npy"
    _, E = run(model, ext, sub, "clean"); ref = np.asarray(np.load(ref_p, mmap_mode="r")[sub], np.float32); r = np.linalg.norm(E - ref, axis=1) / np.linalg.norm(ref, axis=1)
    res = {f"{a.cfg}_{L}f_vs_stored__clean": dict(rel_l2_median=float(np.median(r)), rel_l2_max=float(r.max()), stored=str(ref_p))}; OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"check_{bed}.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1)); sys.exit(0)
if a.check:
    # (1) the 12-block pooled embedding against the stored FP32 embedding (the stored one is the 1024-frame graph): bed's own length for TAU, a 1024-frame model for the others, clean and bg 0 dB
    ext, model = load_ast(1024, dev); res = {}
    sub = np.arange(N)[:: max(1, N // 64)][:64]
    stored = {"urban": I5 / "emb/urban8k", "esc50": I5 / "emb/esc50", "tau": OPTB / "emb_ast/tau"}[bed]
    rel = lambda x, y: (float(np.median(np.linalg.norm(x - y, axis=1) / np.linalg.norm(y, axis=1))), float((np.linalg.norm(x - y, axis=1) / np.linalg.norm(y, axis=1)).max()))
    for cell in (["clean"] if bed == "tau" else ["clean", "bg__0db"]):
        _, E = run(model, ext, sub, cell); ref = np.load(stored / f"fp32__{cell}.npy", mmap_mode="r")[sub]
        res[f"1024f_vs_stored_fp32__{cell}"] = dict(zip(["rel_l2_median", "rel_l2_max"], rel(E, np.asarray(ref, np.float32))))
    # (2) the task's own frame count against the stored 1024-frame FP32 embedding (information only: a different graph on UrbanSound8K and ESC-50)
    if L != 1024:
        ext2, model2 = load_ast(L, dev); _, E2 = run(model2, ext2, sub, "clean"); ref = np.load(stored / "fp32__clean.npy", mmap_mode="r")[sub]
        res[f"{L}f_vs_stored_1024f_fp32__clean"] = dict(zip(["rel_l2_median", "rel_l2_max"], rel(E2, np.asarray(ref, np.float32))))
    res["rows"] = sub.tolist(); OUT.mkdir(parents=True, exist_ok=True); (OUT / f"check_{bed}.json").write_text(json.dumps(res, indent=1)); print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1)); sys.exit(0)

ext, model = load_ast(L, dev); QA = None
if a.cfg != "fp32":
    from QWave.backbone_quant import QuantAST, parse_config
    QA = QuantAST(model); QA.set_config(*parse_config(a.cfg))
plan = cell_rows(bed, N) if bed != "tau" else {"clean": np.arange(N)}
for cell in (a.cells or list(plan)):
    rows = plan[cell]
    if a.limit: rows = rows[: a.limit]
    fp = OUT / f"{bed}__{cell}.npy"
    if fp.exists(): print(f"{cell}: exists", flush=True); continue
    part = OUT / f"_part_{bed}__{cell}.npz"; S = np.full((N, len(DEPTHS), 2, 768), np.nan, np.float32); E = np.full((N, 768), np.nan, np.float32); done = np.zeros(N, bool)
    if part.exists(): d = np.load(part); S, E, done = d["S"], d["E"], d["done"]; print(f"{cell}: resuming {int(done.sum())}", flush=True)
    left = rows[~done[rows]]; t0 = time.time()
    for ci, s in enumerate(range(0, len(left), a.chunk)):
        idx = left[s:s + a.chunk]; Sx, Ex = run(model, ext, idx, cell); S[idx], E[idx], done[idx] = Sx, Ex, True
        if ci % 8 == 7: np.savez(part, S=S, E=E, done=done)
        if ci % 4 == 0: print(f"{bed} {cell}: {int(done[rows].sum())}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    assert np.isfinite(S[rows]).all() and np.isfinite(E[rows]).all()
    if not a.limit:
        np.save(OUT / f"{bed}__{cell}_emb.npy", E); np.save(fp, S)
        (OUT / f"{bed}__{cell}.json").write_text(json.dumps(dict(bed=bed, cell=cell, frames=L, depths=DEPTHS, rows=int(len(rows)), seconds=time.time() - t0, device=a.device, model=MODEL, revision=REV,
            names_crc32=zlib.crc32("\n".join(stems).encode()), layout="[N, depth(4,6,8,10,12), (token0, mean of tokens 2..), 768]; depth 12 is after the final LayerNorm"), indent=1))
    if part.exists(): part.unlink()
    print(f"{bed} {cell}: done {len(rows)} rows in {time.time() - t0:.0f}s", flush=True)
