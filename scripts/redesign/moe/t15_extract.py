"""T15/T18 view embeddings: one pass per condition loads, corrupts (stage 2's code) and pads each clip once, builds the input of every requested view
(t15_views.py) and runs the frozen W8A8 backbones (mn10 for the mel views, CNN14 for v6), on one fixed input shape per dataset.
Output per (view, condition): <out>/<view>__<cond>.npy float32 [N, D] (rows not extracted are NaN), a JSON sidecar, and <out>/lengths.npy (true 16 kHz lengths).
Partial results every --save-every chunks (<out>/_partial__<cond>.npz), picked up on restart. The GPU lock is taken by the calling shell script, not here.

    python t15_extract.py --dataset urban --views v0 v1 --conditions clean bg:0 --field --out-dir OUT [--limit 64]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DS = Path.home() / "Documents/DATASETS"
WT = Path.home() / "Documents/qmoe-weights"
DATA = {
    "urban": dict(csv=DS / "Urban8k/ast_768/urbansound8k.csv", audio=DS / "Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio", pad=64587,
                  bg_csv=DS / "ESC-50/ast_768/esc-50.csv", bg_audio=DS / "ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio"),
    "esc50": dict(csv=DS / "ESC-50/ast_768/esc-50.csv", audio=DS / "ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio", pad=80000,
                  bg_csv=DS / "Urban8k/ast_768/urbansound8k.csv", bg_audio=DS / "Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio"),
    "tau": dict(csv=DS / "TAU2020Mobile/ast_768/tau2020mobile.csv",
                audio=DS / "TAU2020Mobile/audio_raw/tau2020mobile_zenodo_3819968/TAU-urban-acoustic-scenes-2020-mobile-development/audio", pad=160000,
                bg_csv=None, bg_audio=None),
}
BATCH = {"v0": 64, "v1": 64, "v2": 64, "v3": 32, "v4": 64, "lra": 64, "lrb": 64, "v6": 16}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True, choices=list(DATA))
    p.add_argument("--views", nargs="+", required=True)
    p.add_argument("--conditions", nargs="*", default=["clean"], help="clean or KIND:SNR (bg, pink, white), all rows")
    p.add_argument("--field", action="store_true", help="add the nine field cells for the rows the test (5001) or validation (5002) assignment gives them")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--mn10", type=Path, default=WT / "efficientat_release-v0.0.1/mn10_as_mAP_471.pt")
    p.add_argument("--cnn14", type=Path, default=WT / "panns-cnn14/Cnn14_mAP=0.431.pth")
    p.add_argument("--device", default="mps")
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--save-every", type=int, default=8)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--fp32", action="store_true", help="on the clean condition also write <view>_fp32__clean.npy: the FP32 backbone embedding the heads are trained on (repo convention)")
    return p.parse_args(argv)


ARGS = parse_args() if __name__ == "__main__" else None
if ARGS is not None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(ARGS.threads)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts/redesign/idea6/stage2"))
from QWave import efficientat_mn as M  # noqa: E402
from QWave import panns_cnn14 as P  # noqa: E402
from audio_shift import FIELD_CELLS, FIELD_SEEDS, NoiseMaker, cond_name, field_assignment, index_wavs, load_clip, mix  # noqa: E402
import t15_views as V  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_condition(c):
    if c == "clean":
        return "clean", None
    kind, snr = c.split(":")
    if kind not in ("bg", "pink", "white"):
        raise SystemExit(f"bad condition {c}")
    return kind, float(snr)


def plan_conditions(conditions, field, n):
    plan = [(k, s, np.arange(n)) for k, s in map(parse_condition, conditions)]
    if field:
        full = {cond_name(k, s) for k, s, _ in plan}
        assign = [np.array(field_assignment(n, seed)) for seed in FIELD_SEEDS.values()]
        for kind, snr in FIELD_CELLS:
            name = cond_name(kind, snr)
            if name not in full:
                plan.append((kind, float(snr), np.flatnonzero((assign[0] == name) | (assign[1] == name))))
    return plan


def main():
    a = ARGS
    cfg = DATA[a.dataset]
    t_start = time.time()
    meta = pd.read_csv(cfg["csv"], usecols=["folder", "name", "class_id"])
    if a.limit:
        meta = meta.iloc[:a.limit]
    stems = meta["name"].map(lambda s: Path(s).stem).tolist()
    n = len(stems)
    crc = zlib.crc32("\n".join(stems).encode())
    index = index_wavs(cfg["audio"])
    missing = [s for s in stems if s not in index]
    if missing:
        raise SystemExit(f"{len(missing)} clips missing, e.g. {missing[:5]}")
    plan = plan_conditions(a.conditions, a.field, n)
    noise_maker = NoiseMaker([], {}, False)
    if any(k == "bg" for k, _, _ in plan):
        bg_names = pd.read_csv(cfg["bg_csv"], usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist()
        bg_index = index_wavs(cfg["bg_audio"])
        noise_maker = NoiseMaker([b for b in bg_names if b in bg_index], bg_index, True)

    torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    views = list(a.views)
    nets, wsha = {}, {}
    if any(v in V.MEL_VIEWS for v in views):
        base = M.load_mn("mn10", a.mn10)
        wsha["mel"] = sha256(a.mn10)
        net = M.build_backbone("w8a8", base, device)
        for v in views:
            if v in V.MEL_VIEWS:
                nets[v] = net
    if "v6" in views:
        wsha["v6"] = sha256(a.cnn14)
        nets["v6"] = P.QuantCnn14(P.load_cnn14(a.cnn14), 8, 8, device)
    nets32 = {}
    if a.fp32:
        if any(v in V.MEL_VIEWS for v in views):
            net32 = M.build_backbone("fp32", base, device)
            nets32.update({v: net32 for v in views if v in V.MEL_VIEWS})
        if "v6" in views:
            nets32["v6"] = P.load_cnn14(a.cnn14).to(device).eval()
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(1)
    cache = {}
    n_silent = [0]

    def clean_wave(i):
        if i not in cache and len(cache) < 12000:
            cache[i] = load_clip(index[stems[i]])
        return cache[i] if i in cache else load_clip(index[stems[i]])

    lens_path = out / ("lengths.npy" if not a.limit else "lengths_limit.npy")
    lens_all = np.load(lens_path) if lens_path.exists() else np.zeros(n, np.int64)

    for kind, snr, rows in plan:
        cname = cond_name(kind, snr)
        todo = [v for v in views if not (out / f"{v}__{cname}.npy").exists() or (a.fp32 and kind == "clean" and not (out / f"{v}_fp32__clean.npy").exists())]
        if not todo:
            continue
        fp = a.fp32 and kind == "clean"
        part = out / f"_partial__{cname}.npz"
        X = {v: np.full((n, V.DIM[v]), np.nan, np.float32) for v in todo}
        if fp:
            X.update({v + "_fp32": np.full((n, V.DIM[v]), np.nan, np.float32) for v in todo})
        done = np.zeros(n, bool)
        if part.exists():
            d = np.load(part)
            if sorted(d["views"].tolist()) == sorted(todo):
                done = d["done"]
                for v in X:
                    X[v] = d[v]
                print(f"{cname}: resuming, {int(done.sum())} rows done", flush=True)
        left = rows[~done[rows]]
        chunks = [left[s:s + a.chunk] for s in range(0, len(left), a.chunk)]
        prep_s, fwd_s = 0.0, {v: 0.0 for v in todo}
        n_silent[0] = 0
        t0 = time.time()

        def prepare(idx):
            tp = time.time()
            waves = []
            for i in idx:
                x = clean_wave(i)
                lens_all[i] = len(x)
                if kind != "clean":
                    x, mixed = mix(x, noise_maker(stems[i], kind, len(x)), snr)
                    n_silent[0] += not mixed
                if len(x) > cfg["pad"]:
                    raise SystemExit(f"{stems[i]}: {len(x)} samples exceed the padded shape {cfg['pad']}")
                waves.append(np.pad(x, (0, cfg["pad"] - len(x))))
            with torch.no_grad():
                inp = V.mel_inputs(torch.from_numpy(np.stack(waves)).float(), todo)
            return inp, time.time() - tp

        nxt = pool.submit(prepare, chunks[0]) if chunks else None
        for ci, idx in enumerate(chunks):
            inp, dt_prep = nxt.result()
            prep_s += dt_prep
            if ci + 1 < len(chunks):
                nxt = pool.submit(prepare, chunks[ci + 1])
            for v in todo:
                tf = time.time()
                bs = BATCH[v]
                outs = []
                for b in range(0, len(idx), bs):
                    xb = inp[v][b:b + bs]
                    nb = len(xb)
                    if nb < bs:  # fixed batch shape: fill a short last batch with repeated rows, drop their outputs
                        xb = xb[torch.arange(bs) % nb]
                    with torch.no_grad():
                        outs.append(nets[v](xb.to(device)).float().cpu().numpy()[:nb])
                X[v][idx] = np.concatenate(outs)
                fwd_s[v] += time.time() - tf
                if fp:
                    outs = []
                    for b in range(0, len(idx), bs):
                        xb = inp[v][b:b + bs]
                        nb = len(xb)
                        if nb < bs:
                            xb = xb[torch.arange(bs) % nb]
                        with torch.no_grad():
                            outs.append(nets32[v](xb.to(device)).float().cpu().numpy()[:nb])
                    X[v + "_fp32"][idx] = np.concatenate(outs)
            done[idx] = True
            if device.type == "mps":
                torch.mps.empty_cache()
            if (ci + 1) % a.save_every == 0:
                np.savez(part, views=np.array(todo), done=done, **X)
                np.save(lens_path, lens_all)
            el = time.time() - t0
            print(f"  {cname}: {int(done[rows].sum())}/{len(rows)} rows, {el:.0f}s", flush=True)
        dt = time.time() - t0
        for v in list(X):
            if not np.isfinite(X[v][rows]).all():
                raise SystemExit(f"{v}__{cname}: non-finite embeddings")
            np.save(out / f"{v}__{cname}.npy", X[v])
            side = {"view": v, "backbone": "fp32" if v.endswith("_fp32") else "w8a8", "condition": cname, "noise": kind, "snr_db": snr, "dataset": a.dataset, "n_clips": n, "rows_extracted": int(len(rows)),
                    "emb_dim": X[v].shape[1], "pad_samples": cfg["pad"], "silent_clips_left_clean": n_silent[0], "forward_seconds": fwd_s[v.replace("_fp32", "")], "condition_seconds": dt,
                    "prep_seconds": prep_s, "views_in_pass": todo, "device": str(device), "batch_size": BATCH[v.replace("_fp32", "")], "chunk": a.chunk,
                    "weights_sha256": wsha["v6" if v.startswith("v6") else "mel"], "torch": torch.__version__, "names_crc32": crc, "csv": str(cfg["csv"])}
            (out / f"{v}__{cname}.json").write_text(json.dumps(side, indent=2))
        np.save(lens_path, lens_all)
        if kind == "bg":
            planf = out / "bg_plan.json"
            old = json.loads(planf.read_text()) if planf.exists() else {}
            old.update(noise_maker.plan)
            planf.write_text(json.dumps(old, indent=0))
        if part.exists():
            part.unlink()
        print(f"{cname}: {len(left)} rows x {len(todo)} views in {dt:.0f}s (prep {prep_s:.0f}s, forward " + ", ".join(f"{v} {fwd_s[v]:.0f}s" for v in todo)
              + f"), silent {n_silent[0]}", flush=True)
    print(f"done in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
