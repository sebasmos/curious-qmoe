"""T23 waveform-expert embeddings (ANALYSIS_PLAN.md C3): one pass per condition loads each clip (stage 2's load_clip, 16 kHz), applies
the condition at 16 kHz (stage 2's mixer and plan for bg/pink/white, the final run's corrupt.py for reverb/clip/lowpass), zero-pads to
the task's fixed length, resamples the batch to 32 kHz and runs the Wavegram-CNN14 backbone (W8A8, and FP32 on clean with --fp32).
Adapted from T15's t15_extract.py (same datasets, pads, resume logic, sidecars, bg_plan.json, lengths.npy).
Output per condition: <out>/wg__<cond>.npy float32 [N, 2048] (rows not extracted are NaN) and a JSON sidecar; <out>/wg_fp32__clean.npy.
The GPU queue is taken by the calling shell (fair_gpu_run.sh), not here. Run with python -I (the checkpoint is a download).

    python -I t23_extract.py --dataset urban --conditions clean bg:0 reverb:0.3 --fp32 --out-dir OUT [--limit 64]
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DS = Path.home() / "Documents/DATASETS"
WT = Path.home() / "Documents/qmoe-weights"
DATA = {
    "urban": dict(csv=DS / "Urban8k/ast_768/urbansound8k.csv", audio=DS / "Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio", pad=64587,
                  bg_csv=DS / "ESC-50/ast_768/esc-50.csv", bg_audio=DS / "ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio", batch=32),
    "esc50": dict(csv=DS / "ESC-50/ast_768/esc-50.csv", audio=DS / "ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio", pad=80000,
                  bg_csv=DS / "Urban8k/ast_768/urbansound8k.csv", bg_audio=DS / "Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio", batch=32),
    "tau": dict(csv=DS / "TAU2020Mobile/ast_768/tau2020mobile.csv",
                audio=DS / "TAU2020Mobile/audio_raw/tau2020mobile_zenodo_3819968/TAU-urban-acoustic-scenes-2020-mobile-development/audio", pad=160000,
                bg_csv=None, bg_audio=None, batch=16),
}
CKPT = WT / "panns-wavegram/Wavegram_Cnn14_mAP=0.389.pth"
CKPT_MD5 = "1e3506ab640371e0b5a417b15fd66d21"

p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
p.add_argument("--dataset", required=True, choices=list(DATA))
p.add_argument("--conditions", nargs="*", default=["clean"], help="clean, KIND:SNR (bg, pink, white) or reverb:RT60, clip:Q, lowpass:HZ; all rows")
p.add_argument("--out-dir", type=Path, required=True)
p.add_argument("--device", default="mps")
p.add_argument("--chunk", type=int, default=128)
p.add_argument("--save-every", type=int, default=8)
p.add_argument("--threads", type=int, default=2)
p.add_argument("--limit", type=int, default=None)
p.add_argument("--fp32", action="store_true", help="on clean also write wg_fp32__clean.npy (FP32 backbone; heads are trained on it)")
p.add_argument("--max-minutes", type=float, default=75.0, help="save the partial file and exit 3 once this much time has passed")
a = p.parse_args()
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = str(a.threads)
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/redesign/idea6/stage2")); sys.path.insert(0, str(ROOT / "scripts/redesign/idea6/final"))
import numpy as np, pandas as pd, torch  # noqa: E402
from QWave import efficientat_mn as M  # noqa: E402
from QWave.panns_wavegram import QuantWavegram, load_wavegram  # noqa: E402
from audio_shift import NoiseMaker, cond_name, index_wavs, load_clip, mix  # noqa: E402
import corrupt as CR  # noqa: E402


def digest(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_condition(c):
    if c == "clean":
        return "clean", None
    kind, lv = c.split(":")
    if kind not in ("bg", "pink", "white", *CR.KINDS):
        raise SystemExit(f"bad condition {c}")
    return kind, float(lv)


def cname_of(kind, lv):
    return CR.cond_label(kind, lv) if kind in CR.KINDS else cond_name(kind, lv)


def main():
    t_start = time.time()
    cfg = DATA[a.dataset]
    if digest(CKPT, "md5") != CKPT_MD5:
        raise SystemExit("checkpoint md5 differs from Zenodo's")
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
    plan = [parse_condition(c) for c in a.conditions]
    noise_maker = NoiseMaker([], {}, False)
    if any(k == "bg" for k, _ in plan):
        bg_names = pd.read_csv(cfg["bg_csv"], usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist()
        bg_index = index_wavs(cfg["bg_audio"])
        noise_maker = NoiseMaker([b for b in bg_names if b in bg_index], bg_index, True)
    torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    base = load_wavegram(CKPT)
    net8 = QuantWavegram(base, 8, 8, device)
    net32 = base.to(device).eval() if a.fp32 else None
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(1)
    cache = {}
    lens_path = out / ("lengths.npy" if not a.limit else "lengths_limit.npy")
    lens_all = np.load(lens_path) if lens_path.exists() else np.zeros(n, np.int64)
    bs = cfg["batch"]

    use_cache = len(plan) > 1 and n * cfg["pad"] * 4 < 3e9  # TAU (14.7 GB of clean audio) is never cached

    def clean_wave(i):
        if not use_cache:
            return load_clip(index[stems[i]])
        if i not in cache:
            cache[i] = load_clip(index[stems[i]])
        return cache[i]

    for kind, lv in plan:
        cname = cname_of(kind, lv)
        fp = a.fp32 and kind == "clean"
        keys = ["wg"] + (["wg_fp32"] if fp else [])
        if all((out / f"{k}__{cname}.npy").exists() for k in keys):
            continue
        part = out / f"_partial__{cname}.npz"
        X = {k: np.full((n, 2048), np.nan, np.float32) for k in keys}
        done = np.zeros(n, bool)
        if part.exists():
            d = np.load(part)
            if sorted(d["keys"].tolist()) == sorted(keys):
                done = d["done"]
                for k in keys:
                    X[k] = d[k]
                print(f"{cname}: resuming, {int(done.sum())} rows done", flush=True)
        left = np.flatnonzero(~done)
        chunks = [left[s:s + a.chunk] for s in range(0, len(left), a.chunk)]
        stats = {"silent": 0, "level_db": []}
        prep_s, fwd_s, t0 = 0.0, 0.0, time.time()

        def prepare(idx):
            tp = time.time()
            waves = []
            for i in idx:
                x = clean_wave(i)
                lens_all[i] = len(x)
                if kind in CR.KINDS:
                    y = CR.corrupt(x, kind, lv)
                    dlev = CR.level_change_db(x, y)
                    if dlev is not None:
                        stats["level_db"].append(dlev)
                    x = y
                elif kind != "clean":
                    x, mixed = mix(x, noise_maker(stems[i], kind, len(x)), lv)
                    stats["silent"] += not mixed
                if len(x) > cfg["pad"]:
                    raise SystemExit(f"{stems[i]}: {len(x)} samples exceed the padded shape {cfg['pad']}")
                waves.append(np.pad(x, (0, cfg["pad"] - len(x))))
            with torch.no_grad():
                w32 = M.resample_to_32k(torch.from_numpy(np.stack(waves)).float())
            return w32, time.time() - tp

        nxt = pool.submit(prepare, chunks[0]) if chunks else None
        timed_out = False
        for ci, idx in enumerate(chunks):
            inp, dt = nxt.result()
            prep_s += dt
            if ci + 1 < len(chunks):
                nxt = pool.submit(prepare, chunks[ci + 1])
            tf = time.time()
            for k, net in (("wg", net8), ("wg_fp32", net32)):
                if k not in keys:
                    continue
                outs = []
                for b in range(0, len(idx), bs):
                    xb = inp[b:b + bs]
                    nb = len(xb)
                    if nb < bs:  # fixed batch shape: fill with repeated rows, drop their outputs
                        xb = xb[torch.arange(bs) % nb]
                    with torch.no_grad():
                        outs.append(net(xb.to(device)).float().cpu().numpy()[:nb])
                X[k][idx] = np.concatenate(outs)
            fwd_s += time.time() - tf
            done[idx] = True
            if device.type == "mps":
                torch.mps.empty_cache()
            over = (time.time() - t_start) / 60 > a.max_minutes
            if (ci + 1) % a.save_every == 0 or over:
                np.savez(part, keys=np.array(keys), done=done, **X)
                np.save(lens_path, lens_all)
            print(f"  {cname}: {int(done.sum())}/{n} rows, {time.time() - t0:.0f}s", flush=True)
            if over and ci + 1 < len(chunks):
                timed_out = True
                break
        if timed_out:
            print(f"{cname}: time limit, partial saved", flush=True)
            if kind == "bg":
                planf = out / "bg_plan.json"
                old = json.loads(planf.read_text()) if planf.exists() else {}
                old.update(noise_maker.plan)
                planf.write_text(json.dumps(old, indent=0))
            sys.exit(3)
        for k in keys:
            if not np.isfinite(X[k]).all():
                raise SystemExit(f"{k}__{cname}: non-finite embeddings")
            np.save(out / f"{k}__{cname}.npy", X[k])
            side = {"view": k, "backbone": "fp32" if k.endswith("_fp32") else "w8a8", "condition": cname, "kind": kind, "level": lv, "dataset": a.dataset,
                    "n_clips": n, "emb_dim": 2048, "pad_samples_16k": cfg["pad"], "silent_clips_left_clean": stats["silent"],
                    "mean_level_change_db": float(np.mean(stats["level_db"])) if stats["level_db"] else None, "n_level": len(stats["level_db"]),
                    "forward_seconds": fwd_s, "prep_seconds": prep_s, "condition_seconds": time.time() - t0, "device": str(device), "batch_size": bs,
                    "chunk": a.chunk, "weights_md5": CKPT_MD5, "torch": torch.__version__, "names_crc32": crc, "csv": str(cfg["csv"])}
            (out / f"{k}__{cname}.json").write_text(json.dumps(side, indent=2))
        np.save(lens_path, lens_all)
        if kind == "bg":
            planf = out / "bg_plan.json"
            old = json.loads(planf.read_text()) if planf.exists() else {}
            old.update(noise_maker.plan)
            planf.write_text(json.dumps(old, indent=0))
        if part.exists():
            part.unlink()
        print(f"{cname}: done in {time.time() - t0:.0f}s (prep {prep_s:.0f}s, forward {fwd_s:.0f}s)", flush=True)
    print(f"all done in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
