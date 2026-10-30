"""T21 EfficientAT MN10 extraction on the CPU, each clip at its own length (ANALYSIS_PLAN.md C3), FP32 and W8A8.

Audio and conditions as t21_extract_ast.py (stage 2's audio_shift with the 10.24 s rule, corrupt.py, the field cells
for the rows the two assignments give them). Front end and models: QWave.efficientat_mn (load_mn, model_input,
build_backbone), as stage 3c's extract_cnn.py. With --mel-cache DIR the log-mel input of every extracted row is also
written per condition (float16, frames concatenated, with offsets) for the ONNX Runtime int8 path, which has no
torchaudio. Output per (config, condition): mn10_<config>__<cond>.npy [N, 960] (NaN outside the extracted rows) and a
sidecar with per-clip costs (means), names checksum and timing.

    python t21_extract_mn.py --csv META --audio-dir AUDIO --bg-csv U8K_CSV --bg-audio-dir U8K_AUDIO --out-dir OUT \
        --configs fp32 w8a8 --conditions clean bg:10 ... [--field] [--mel-cache DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zlib
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--bg-audio-dir", type=Path, default=None)
    p.add_argument("--bg-csv", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--configs", nargs="+", default=["fp32", "w8a8"])
    p.add_argument("--conditions", nargs="*", default=[])
    p.add_argument("--field", action="store_true")
    p.add_argument("--mel-cache", type=Path, default=None)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--available-only", action="store_true",
                   help="process only clips whose audio is on disk; a condition stays partial until every row is done")
    return p.parse_args(argv)


ARGS = parse_args() if __name__ == "__main__" else None
if ARGS is not None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(ARGS.threads)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t21_common as C  # noqa: E402

C.patch_loader()
from audio_shift import NoiseMaker, index_wavs, mix  # noqa: E402
import audio_shift as AS  # noqa: E402
import corrupt as CR  # noqa: E402
from QWave import efficientat_mn as M  # noqa: E402
from t21_extract_ast import cname_of, plan_conditions  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    a = ARGS
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    t_start = time.time()
    meta = pd.read_csv(a.csv, usecols=["folder", "name", "class_id"])
    stems = meta["name"].map(lambda s: Path(s).stem).tolist()
    n = len(stems)
    crc = zlib.crc32("\n".join(stems).encode())
    index = index_wavs(a.audio_dir)
    missing = [s for s in stems if s not in index]
    avail = np.array([s in index for s in stems])
    if missing and not a.available_only:
        raise SystemExit(f"{len(missing)} clips missing, e.g. {missing[:5]}")
    plan = plan_conditions(a.conditions, a.field, n)
    noise_maker = NoiseMaker([], {}, False)
    if any(k == "bg" for k, _, _ in plan):
        bg_names = pd.read_csv(a.bg_csv, usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist()
        bg_index = index_wavs(a.bg_audio_dir)
        noise_maker = NoiseMaker([b for b in bg_names if b in bg_index], bg_index, True)
    wf = C.WEIGHTS / "mn10_as_mAP_471.pt"
    base = M.load_mn("mn10", wf)
    nets = {c: M.build_backbone(c, base, "cpu") for c in a.configs}
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    clean_cache = {}
    for kind, snr, rows in plan:
        cname = cname_of(kind, snr)
        todo = [c for c in a.configs if not (out / f"mn10_{c}__{cname}.npy").exists()]
        if not todo:
            continue
        melf = a.mel_cache / f"{cname}__part{int(time.time())}.npz" if a.mel_cache else None
        if a.limit:
            rows = rows[:a.limit]
        X = {c: np.full((n, base.emb_dim), np.nan, np.float32) for c in todo}
        lens = np.zeros(n, np.int64)
        done = np.zeros(n, bool)
        part = out / f"_partial__{cname}.npz"
        if part.exists():
            d = np.load(part)
            if sorted(d["configs"].tolist()) == sorted(todo):
                done, lens = d["done"].copy(), d["lens"].copy()
                for c in todo:
                    X[c] = d[c].copy()
                print(f"{cname}: resuming, {int(done[rows].sum())} rows done", flush=True)
        all_rows = rows
        rows = rows[~done[rows] & avail[rows]]
        mel_parts, mel_rows = [], []
        fwd = {c: 0.0 for c in todo}
        prep, n_silent, t0 = 0.0, 0, time.time()
        for s in range(0, len(rows), a.chunk):
            idx = rows[s:s + a.chunk]
            tp = time.time()
            waves = []
            for i in idx:
                x = clean_cache.get(i)
                if x is None:
                    x = AS.load_clip(index[stems[i]])
                    if len(clean_cache) < 12000:
                        clean_cache[i] = x
                if kind in CR.KINDS:
                    x = CR.corrupt(x, kind, snr)
                elif kind != "clean":
                    x, mixed = mix(x, noise_maker(stems[i], kind, len(x)), snr)
                    n_silent += not mixed
                waves.append(x)
            lens[idx] = [len(w) for w in waves]
            with torch.no_grad():
                groups = M.model_input(waves)
            prep += time.time() - tp
            for gi, xin in groups:
                r = idx[np.asarray(gi)]
                if melf is not None:
                    for j, row in enumerate(r):
                        mel_parts.append(xin[j, 0].numpy().astype(np.float16))
                        mel_rows.append(int(row))
                for c in todo:
                    tf = time.time()
                    with torch.no_grad():
                        X[c][r] = nets[c](xin).float().numpy()
                    fwd[c] += time.time() - tf
            print(f"  {cname}: {min(s + a.chunk, len(rows))}/{len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
        done[rows] = True
        dt = time.time() - t0
        if melf is not None and mel_parts:
            a.mel_cache.mkdir(parents=True, exist_ok=True)
            off = np.cumsum([0] + [p.shape[1] for p in mel_parts])
            tmp = a.mel_cache / f"_{melf.name}.tmp.npz"
            np.savez(tmp, mel=np.concatenate(mel_parts, axis=1), offsets=off, rows=np.array(mel_rows),
                     names_crc32=np.int64(crc))
            os.replace(tmp, melf)
        if not done[all_rows].all():
            np.savez(part, configs=np.array(todo), done=done, lens=lens, **{c: X[c] for c in todo})
            print(f"{cname}: {int(done[all_rows].sum())}/{len(all_rows)} rows done (audio missing for the rest); partial saved", flush=True)
            continue
        rows = all_rows
        rl = lens[rows]
        cache = {}
        fe, bo, macs = [], [], []
        for L in rl:
            fr = M.frames_of(M.n32_of(int(L)))
            if fr not in cache:
                cache[fr] = M.feature_macs(base, fr)
            fe.append(M.frontend_macs(int(L)))
            macs.append(cache[fr])
        for c in todo:
            if not np.isfinite(X[c][rows]).all():
                raise SystemExit(f"mn10_{c}__{cname}: non-finite embeddings")
            np.save(out / f"mn10_{c}__{cname}.npy", X[c])
            bb = [M.backbone_bitops({"macs": m_["macs"], "out_elems": m_["out_elems"], "aggregate_macs": 0.0}, c, 1024.0)
                  for m_ in macs]
            side = {"config": f"mn10_{c}", "condition": cname, "noise": kind, "snr_db": snr, "n_clips": n,
                    "rows_extracted": int(len(rows)), "emb_dim": base.emb_dim, "silent_clips_left_clean": n_silent,
                    "length_rule": "first 163840 samples at 16 kHz", "device": "cpu", "threads": a.threads,
                    "forward_seconds": fwd[c], "prep_seconds": prep, "condition_seconds": dt, "names_crc32": crc,
                    "weights": str(wf), "weights_sha256": sha256(wf), "torch": torch.__version__,
                    "frontend_bitops_mean": float(np.mean(fe)) * 1024.0, "backbone_bitops_rescale1024_mean": float(np.mean(bb)),
                    "backbone_macs_mean": float(np.mean([m_["macs"] for m_ in macs]))}
            (out / f"mn10_{c}__{cname}.json").write_text(json.dumps(side, indent=2))
        np.save(out / f"lengths__{cname}.npy", lens)
        if part.exists():
            part.unlink()
        if kind == "bg":
            planf = out / "bg_plan.json"
            old = json.loads(planf.read_text()) if planf.exists() else {}
            old.update(noise_maker.plan)
            planf.write_text(json.dumps(old, indent=0))
        print(f"{cname}: {len(rows)} rows in {dt:.0f}s (prep {prep:.0f}s, " + ", ".join(f"{c} {fwd[c]:.0f}s" for c in todo)
              + f"), silent {n_silent}", flush=True)
    print(f"done in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
