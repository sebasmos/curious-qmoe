"""Idea 6 final round: Stage 3c's extract_cnn.py (scripts/redesign/idea6/stage3c/extract_cnn.py at 545bb54) with
three more condition kinds, reverb:RT60, clip:Q and lowpass:HZ (corrupt.py). The only changes are parse_condition,
Audio.get (calls corrupt.corrupt for the new kinds and records the mean level change), the condition file-name
helper cname_of and the extra sidecar fields. Conditions of the existing kinds are handled by the unchanged code.

EfficientAT clip embeddings for idea 6 Stage 3c: Stage 3b's extractor
(scripts/redesign/idea6/stage3b/extract_cnn.py, unchanged audio, front end,
chunking and sidecars) extended to the siblings mn05, mn20 (QWave.efficientat_mn)
and dymn04 (QWave.efficientat_dymn), FP32 and W8A8, and with a mode for the
P5 unpadded UrbanSound8K extraction (--tail-cpu, below).

Audio: Stage 2's audio_shift.py (idea 5 Stage 2's file, unchanged): load_clip
(mono, mean removed, 16 kHz), NoiseMaker and mix, called exactly as Stage 2's
extract_eff.py calls them (same CSVs, folders, seeds and background plan), so
the CNN reads the 16 kHz waveforms the EfficientNet and the AST read. The
model's documented front end then resamples to 32 kHz and computes its log
mel (QWave/efficientat_mn.py). Each clip runs at its own length; clips of
equal length are batched.

Models: QWave.efficientat_mn.load_mn on the released AudioSet weights
(--weights-dir, torch.load with weights_only=True), names mn04 and mn10.
Output per (model, configuration, condition):
<out>/<model>_<config>__<condition>.npy, float32 [N, emb_dim] in the CSV's
row order, and a JSON sidecar (front end, weight file SHA-256, costs as
means over the clips' lengths, timing, names checksum); lengths__<condition>.npy
once per condition. Partial results every --save-every chunks, picked up on
restart.

--pad-to-samples N (Stage 3b amendment 1, UrbanSound8K only): every 16 kHz
waveform shorter than N samples is zero-padded at the end to N before the
front end, so every clip has one input shape. Running UrbanSound8K's 354
distinct lengths on MPS compiled one graph per (batch, length) and grew the
first extraction process to 40 GB; costs are then counted at the padded
length. After every chunk the MPS allocator cache is emptied.

--tail-cpu (Stage 3c, P5): no padding at all. Within a chunk, a group of
clips of one exact length with at least --mps-min-group clips runs on the
MPS device in batches of exactly --batch-size (a short last batch is filled
with repeated rows, whose outputs are dropped, so the device compiles one
graph per length that occurs that often: on UrbanSound8K only the 4 s
length); every smaller group runs on the CPU at its exact length, where no
graph is compiled. Lengths are the clips' own; costs use them.

    python scripts/redesign/idea6/stage3c/extract_cnn.py --csv AST_CSV --audio-dir AUDIO \
        --bg-audio-dir OTHER_AUDIO --bg-csv OTHER_CSV --out-dir OUT --weights-dir W \
        --models mn05 mn20 dymn04 --configs fp32 w8a8 --conditions clean bg:10 ... [--limit 64]
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

WEIGHT_FILES = {"mn04": "mn04_as_mAP_432.pt", "mn05": "mn05_as_mAP_443.pt", "mn10": "mn10_as_mAP_471.pt",
                "mn20": "mn20_as_mAP_478.pt", "dymn04": "dymn04_as.pt"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--bg-audio-dir", type=Path, default=None)
    p.add_argument("--bg-csv", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--weights-dir", type=Path, required=True)
    p.add_argument("--models", nargs="+", default=["mn04", "mn10"])
    p.add_argument("--configs", nargs="+", default=["fp32", "w8a8"])
    p.add_argument("--conditions", nargs="+", required=True, help="clean or KIND:SNR (bg, pink, white)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--save-every", type=int, default=4)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pad-to-samples", type=int, default=0, help="zero-pad shorter clips to this many 16 kHz samples")
    p.add_argument("--tail-cpu", action="store_true", help="P5: exact lengths; small length groups run on the CPU")
    p.add_argument("--mps-min-group", type=int, default=32)
    return p.parse_args(argv)


ARGS = parse_args() if __name__ == "__main__" else None
if ARGS is not None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(ARGS.threads)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "redesign" / "idea6" / "stage2"))

from QWave import efficientat_dymn as D  # noqa: E402
from QWave import efficientat_mn as M  # noqa: E402
from audio_shift import NoiseMaker, cond_name, index_wavs, load_clip, mix  # noqa: E402
import corrupt as CR  # noqa: E402


def parse_condition(c):
    if c == "clean":
        return "clean", None
    kind, snr = c.split(":")
    if kind not in ("bg", "pink", "white", *CR.KINDS):
        raise SystemExit(f"bad condition {c}")
    return kind, float(snr)


def cname_of(kind, level):
    """File-name suffix of a condition: Stage 2's for the noise kinds, corrupt.cond_label for the new ones."""
    return CR.cond_label(kind, level) if kind in CR.KINDS else cond_name(kind, level)


class Audio:
    """Clean 16 kHz waveforms (cached) and their corrupted versions (Stage 2's extract_eff.Audio)."""

    def __init__(self, stems, index, noise_maker):
        self.stems, self.index, self.noise, self.cache = stems, index, noise_maker, {}
        self.n_silent = 0
        self.level_db = []

    def clean(self, i):
        if i not in self.cache:
            self.cache[i] = load_clip(self.index[self.stems[i]])
        return self.cache[i]

    def get(self, i, kind, snr):
        x = self.clean(i)
        if kind == "clean":
            return x
        if kind in CR.KINDS:
            y = CR.corrupt(x, kind, snr)
            d = CR.level_change_db(x, y)
            if d is not None:
                self.level_db.append(d)
            return y
        y, mixed = mix(x, self.noise(self.stems[i], kind, len(x)), snr)
        self.n_silent += not mixed
        return y


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def lib_of(name):
    return D if name.startswith("dymn") else M


def load_model(name, path):
    return lib_of(name).load_dymn(name, path) if name.startswith("dymn") else M.load_mn(name, path)


def clip_costs(name, base, lengths) -> dict:
    """Means over the clips of the front end and the feature extractor's MACs and output elements (and, for a
    DyMN, the kernel aggregation MACs)."""
    L = lib_of(name)
    cache = {}
    fe, macs, outs, agg = [], [], [], []
    for n in lengths:
        fr = M.frames_of(M.n32_of(int(n)))
        if fr not in cache:
            cache[fr] = L.feature_macs(base, fr)
        fe.append(M.frontend_macs(int(n)))
        macs.append(cache[fr]["macs"])
        outs.append(cache[fr]["out_elems"])
        agg.append(cache[fr].get("aggregate_macs", 0.0))
    return {"frontend_macs_mean": float(np.mean(fe)), "backbone_macs_mean": float(np.mean(macs)),
            "backbone_out_elems_mean": float(np.mean(outs)), "aggregate_macs_mean": float(np.mean(agg)),
            "n_distinct_frames": len(cache), "frames_min": int(min(cache)), "frames_max": int(max(cache))}


def main():
    args = ARGS
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    t_start = time.time()
    meta = pd.read_csv(args.csv, usecols=["folder", "name", "class_id"])
    if args.limit:
        meta = meta.iloc[:args.limit]
    stems = meta["name"].map(lambda s: Path(s).stem).tolist()
    n = len(stems)
    crc = zlib.crc32("\n".join(stems).encode())
    index = index_wavs(args.audio_dir)
    missing = [s for s in stems if s not in index]
    if missing:
        raise SystemExit(f"{len(missing)} clips missing, e.g. {missing[:5]}")
    plan = [parse_condition(c) for c in args.conditions]
    noise_maker = NoiseMaker([], {}, False)
    if any(k == "bg" for k, _ in plan):
        bg_names = pd.read_csv(args.bg_csv, usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist()
        bg_index = index_wavs(args.bg_audio_dir)
        noise_maker = NoiseMaker([b for b in bg_names if b in bg_index], bg_index, True)
    audio = Audio(stems, index, noise_maker)

    device = torch.device(args.device)
    cpu = torch.device("cpu")
    if args.tail_cpu and args.pad_to_samples:
        raise SystemExit("--tail-cpu runs clips at their own length; drop --pad-to-samples")
    bases, nets, cpu_nets, wsha, dims = {}, {}, {}, {}, {}
    for m in args.models:
        wf = args.weights_dir / WEIGHT_FILES[m]
        wsha[m] = sha256(wf)
        bases[m] = load_model(m, wf)
        dims[m] = bases[m].emb_dim
        for c in args.configs:
            nets[(m, c)] = lib_of(m).build_backbone(c, bases[m], device)
            if args.tail_cpu and device.type != "cpu":
                cpu_nets[(m, c)] = lib_of(m).build_backbone(c, bases[m], cpu)
    names = {(m, c): f"{m}_{c}" for m in args.models for c in args.configs}
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(1)

    for kind, snr in plan:
        cname = cname_of(kind, snr)
        todo = [k for k in nets if not (out / f"{names[k]}__{cname}.npy").exists()]
        if not todo:
            continue
        part = out / f"_partial__{cname}.npz"
        X = {k: np.full((n, dims[k[0]]), np.nan, np.float32) for k in todo}
        lens = np.zeros(n, np.int64)
        done = np.zeros(n, bool)
        if part.exists():
            d = np.load(part)
            if sorted(d["configs"].tolist()) == sorted(names[k] for k in todo):
                done, lens = d["done"], d["lens"]
                for k in todo:
                    X[k] = d[names[k]]
                print(f"{cname}: resuming, {int(done.sum())} rows done", flush=True)
        audio.n_silent = 0
        audio.level_db = []
        prep_s, fwd_s, n_cpu = 0.0, {k: 0.0 for k in todo}, {k: 0 for k in todo}
        t0 = time.time()
        left = np.flatnonzero(~done)
        chunks = [left[s:s + args.chunk] for s in range(0, len(left), args.chunk)]

        def prepare(idx):
            tp = time.time()
            waves = [audio.get(i, kind, snr) for i in idx]
            lens_ = np.array([len(w) for w in waves])
            if args.pad_to_samples:
                waves = [np.pad(w, (0, max(0, args.pad_to_samples - len(w)))) for w in waves]
            with torch.no_grad():
                groups = M.model_input(waves)
            return groups, lens_, time.time() - tp

        nxt = pool.submit(prepare, chunks[0]) if chunks else None
        for ci, idx in enumerate(chunks):
            groups, ln, dt_prep = nxt.result()
            prep_s += dt_prep
            if ci + 1 < len(chunks):
                nxt = pool.submit(prepare, chunks[ci + 1])
            lens[idx] = ln
            for gi, x in groups:
                rows = idx[np.asarray(gi)]
                small = args.tail_cpu and len(rows) < args.mps_min_group and cpu_nets
                xd = x if small else x.to(device)
                for k in todo:
                    tf = time.time()
                    with torch.no_grad():
                        if small:
                            outs = [cpu_nets[k](xd[b:b + args.batch_size]).float().numpy()
                                    for b in range(0, len(rows), args.batch_size)]
                            n_cpu[k] += len(rows)
                        elif args.tail_cpu:  # fixed batch shape: fill a short last batch with repeated rows
                            outs = []
                            for b in range(0, len(rows), args.batch_size):
                                xb = xd[b:b + args.batch_size]
                                nb = len(xb)
                                if nb < args.batch_size:
                                    xb = xb[torch.arange(args.batch_size, device=xb.device) % nb]
                                outs.append(nets[k](xb).float().cpu().numpy()[:nb])
                        else:
                            outs = [nets[k](xd[b:b + args.batch_size]).float().cpu().numpy()
                                    for b in range(0, len(rows), args.batch_size)]
                    X[k][rows] = np.concatenate(outs)
                    fwd_s[k] += time.time() - tf
            done[idx] = True
            if device.type == "mps":
                torch.mps.empty_cache()
            if (ci + 1) % args.save_every == 0:
                np.savez(part, configs=np.array([names[k] for k in todo]), done=done, lens=lens,
                         **{names[k]: X[k] for k in todo})
            el = time.time() - t0
            print(f"  {cname}: {int(done.sum())}/{n} rows, {el:.0f}s, "
                  f"{(done.sum() - (n - len(left))) / max(el, 1e-9):.1f} clips/s", flush=True)
        dt = time.time() - t0
        run_lens = np.maximum(lens, args.pad_to_samples)  # what the front end and the network ran on
        costs = {m: clip_costs(m, bases[m], run_lens) for m in {k[0] for k in todo}}
        for k in todo:
            m, c = k
            if not np.isfinite(X[k]).all():
                raise SystemExit(f"{names[k]}__{cname}: non-finite embeddings")
            np.save(out / f"{names[k]}__{cname}.npy", X[k])
            cc = costs[m]
            mac = {"macs": cc["backbone_macs_mean"], "out_elems": cc["backbone_out_elems_mean"],
                   "aggregate_macs": cc["aggregate_macs_mean"]}
            L = lib_of(m)
            side = {"config": names[k], "model": m, "backbone_config": c, "condition": cname, "noise": kind,
                    "snr_db": snr, "n_clips": n, "emb_dim": dims[m], "pad_to_samples": args.pad_to_samples,
                    "silent_clips_left_clean": audio.n_silent,
                    "level": snr, "mean_level_change_db": float(np.mean(audio.level_db)) if audio.level_db else None, "forward_seconds": fwd_s[k],
                    "clips_per_s_forward": len(left) / max(fwd_s[k], 1e-9), "condition_seconds": dt,
                    "prep_seconds": prep_s, "configs_in_pass": [names[t] for t in todo], "device": str(device),
                    "batch_size": args.batch_size, "chunk": args.chunk, "torch_seed": args.seed,
                    "tail_cpu": bool(args.tail_cpu), "mps_min_group": args.mps_min_group if args.tail_cpu else None,
                    "rows_on_cpu": int(n_cpu[k]), "model_module": L.__name__,
                    "csv": str(args.csv), "audio_dir": str(args.audio_dir), "bg_csv": str(args.bg_csv),
                    "bg_audio_dir": str(args.bg_audio_dir),
                    "weights": str(args.weights_dir / WEIGHT_FILES[m]), "weights_sha256": wsha[m],
                    "frontend": {"sr_in": M.SR_IN, "sr": M.SR, "n_fft": M.N_FFT, "win": M.WIN, "hop": M.HOP,
                                 "n_mels": M.N_MELS, "f_min": M.F_MIN, "f_max": M.F_MAX,
                                 "resample_taps": M.resample_taps()[2]},
                    "torch": torch.__version__, "names_crc32": crc,
                    "cost_per_clip": {**cc, "backbone_bitops": L.backbone_bitops(mac, c),
                                      "backbone_bitops_rescale1024": L.backbone_bitops(mac, c, 1024.0),
                                      "frontend_bitops_mean": cc["frontend_macs_mean"] * 32 * 32,
                                      "storage_bits": L.weight_bits_stored(bases[m], c)}}
            (out / f"{names[k]}__{cname}.json").write_text(json.dumps(side, indent=2))
        np.save(out / f"lengths__{cname}.npy", lens)
        if kind == "bg":
            planf = out / "bg_plan.json"
            old = json.loads(planf.read_text()) if planf.exists() else {}
            old.update(noise_maker.plan)
            planf.write_text(json.dumps(old, indent=0))
        if part.exists():
            part.unlink()
        print(f"{cname}: {len(left)} rows x {len(todo)} backbones in {dt:.0f}s (prep {prep_s:.0f}s, forward "
              + ", ".join(f"{names[k]} {fwd_s[k]:.0f}s" for k in todo) + f"), silent {audio.n_silent}"
              + (f", rows on cpu {n_cpu[todo[0]]}" if args.tail_cpu else ""), flush=True)
    print(f"done in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
