"""T25 extraction of the stage 3c CNNs (MN04, MN05, MN20, DyMN04) on T21's D1 pool, CPU, each clip at its own length.

This is t21_extract_mn.py (T21's MN10 extractor, unchanged audio, length rule, mixer, field cells and sidecars) with
the model list widened through stage 3c's loaders: QWave.efficientat_mn (mn04, mn05, mn10, mn20) and
QWave.efficientat_dymn (dymn04), the same weight files and the simulated W8A8 of build_backbone. Every model runs on
the same waveform of a chunk, so the audio is loaded and mixed once per condition and every model reads the audio
T21's MN10 and AST read. No mel cache (no int8 path in T25).

Output per (model, config, condition): <model>_<config>__<cond>.npy [N, emb_dim] (NaN outside the extracted rows),
a JSON sidecar (per-clip cost means, names checksum, timing), lengths__<cond>.npy, bg_plan.json.

    python t25_extract_cnn.py --csv POOL --audio-dir AUDIO --bg-csv U8K_CSV --bg-audio-dir U8K_AUDIO --out-dir OUT \
        --models mn04 mn05 mn20 dymn04 --configs w8a8 --conditions clean bg:10 ... [--field] [--limit N]
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

WEIGHT_FILES = {"mn04": "mn04_as_mAP_432.pt", "mn05": "mn05_as_mAP_443.pt", "mn10": "mn10_as_mAP_471.pt",
                "mn20": "mn20_as_mAP_478.pt", "dymn04": "dymn04_as.pt"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--bg-audio-dir", type=Path, default=None)
    p.add_argument("--bg-csv", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--models", nargs="+", default=["mn04", "mn05", "mn20", "dymn04"])
    p.add_argument("--configs", nargs="+", default=["w8a8"])
    p.add_argument("--conditions", nargs="*", default=[])
    p.add_argument("--field", action="store_true")
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args(argv)


ARGS = parse_args() if __name__ == "__main__" else None
if ARGS is not None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(ARGS.threads)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "t21"))
import t21_common as C  # noqa: E402

C.patch_loader()
from audio_shift import NoiseMaker, index_wavs, mix  # noqa: E402
import audio_shift as AS  # noqa: E402
import corrupt as CR  # noqa: E402
from QWave import efficientat_dymn as DY  # noqa: E402
from QWave import efficientat_mn as M  # noqa: E402
from t21_extract_ast import cname_of, plan_conditions  # noqa: E402


def lib_of(name):
    return DY if name.startswith("dymn") else M


def load_model(name, path):
    return DY.load_dymn(name, path) if name.startswith("dymn") else M.load_mn(name, path)


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
    if missing:
        raise SystemExit(f"{len(missing)} clips missing, e.g. {missing[:5]}")
    plan = plan_conditions(a.conditions, a.field, n)
    noise_maker = NoiseMaker([], {}, False)
    if any(k == "bg" for k, _, _ in plan):
        bg_names = pd.read_csv(a.bg_csv, usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist()
        bg_index = index_wavs(a.bg_audio_dir)
        noise_maker = NoiseMaker([b for b in bg_names if b in bg_index], bg_index, True)
    wfs = {m: C.WEIGHTS / WEIGHT_FILES[m] for m in a.models}
    bases = {m: load_model(m, wfs[m]) for m in a.models}
    nets = {(m, c): lib_of(m).build_backbone(c, bases[m], "cpu") for m in a.models for c in a.configs}
    wsha = {m: sha256(wfs[m]) for m in a.models}
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    clean_cache = {}
    for kind, snr, rows in plan:
        cname = cname_of(kind, snr)
        todo = [(m, c) for m in a.models for c in a.configs if not (out / f"{m}_{c}__{cname}.npy").exists()]
        if not todo:
            continue
        if a.limit:
            rows = rows[:a.limit]
        X = {mc: np.full((n, bases[mc[0]].emb_dim), np.nan, np.float32) for mc in todo}
        lens = np.zeros(n, np.int64)
        fwd = {mc: 0.0 for mc in todo}
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
                groups = M.model_input(waves)  # DyMN shares MN's front end (QWave/efficientat_dymn.py)
            prep += time.time() - tp
            for gi, xin in groups:
                r = idx[np.asarray(gi)]
                for mc in todo:
                    tf = time.time()
                    with torch.no_grad():
                        X[mc][r] = nets[mc](xin).float().numpy()
                    fwd[mc] += time.time() - tf
            print(f"  {cname}: {min(s + a.chunk, len(rows))}/{len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
        dt = time.time() - t0
        rl = lens[rows]
        fe = np.array([M.frontend_macs(int(L)) for L in rl])
        for m in sorted({mc[0] for mc in todo}):
            L = lib_of(m)
            cache = {}
            macs = []
            for Lc in rl:
                fr = M.frames_of(M.n32_of(int(Lc)))
                if fr not in cache:
                    cache[fr] = L.feature_macs(bases[m], fr)
                macs.append(cache[fr])
            for c in [c for (mm, c) in todo if mm == m]:
                if not np.isfinite(X[(m, c)][rows]).all():
                    raise SystemExit(f"{m}_{c}__{cname}: non-finite embeddings")
                np.save(out / f"{m}_{c}__{cname}.npy", X[(m, c)])
                bb = [L.backbone_bitops({"aggregate_macs": 0.0, **mm_}, c, 1024.0) for mm_ in macs]
                side = {"config": f"{m}_{c}", "condition": cname, "noise": kind, "snr_db": snr, "n_clips": n,
                        "rows_extracted": int(len(rows)), "emb_dim": bases[m].emb_dim, "silent_clips_left_clean": n_silent,
                        "length_rule": "first 163840 samples at 16 kHz", "device": "cpu", "threads": a.threads,
                        "forward_seconds": fwd[(m, c)], "prep_seconds": prep, "condition_seconds": dt, "names_crc32": crc,
                        "weights": str(wfs[m]), "weights_sha256": wsha[m], "torch": torch.__version__,
                        "frontend_bitops_mean": float(fe.mean()) * 1024.0, "backbone_bitops_rescale1024_mean": float(np.mean(bb)),
                        "backbone_macs_mean": float(np.mean([mm_["macs"] for mm_ in macs])),
                        "aggregate_macs_mean": float(np.mean([mm_.get("aggregate_macs", 0.0) for mm_ in macs]))}
                (out / f"{m}_{c}__{cname}.json").write_text(json.dumps(side, indent=2))
        np.save(out / f"lengths__{cname}.npy", lens)
        if kind == "bg":
            planf = out / "bg_plan.json"
            old = json.loads(planf.read_text()) if planf.exists() else {}
            old.update(noise_maker.plan)
            planf.write_text(json.dumps(old, indent=0))
        print(f"{cname}: {len(rows)} rows in {dt:.0f}s (prep {prep:.0f}s, "
              + ", ".join(f"{m}_{c} {fwd[(m, c)]:.0f}s" for m, c in todo) + f"), silent {n_silent}", flush=True)
    print(f"done in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
