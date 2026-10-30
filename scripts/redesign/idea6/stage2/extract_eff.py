"""EfficientNet-B3 clip embeddings for idea 6 Stage 2 (the cheap tier), FP32
and W8A8, on the same corrupted audio as the AST tier of idea 5 Stage 2.

Audio: audio_shift.py is idea 5 Stage 2's file copied unchanged (SHA-256
48b44fb2...0633, branch redesign/idea5-stage2 since a422003): load_clip
(mono, mean removed, 16 kHz), NoiseMaker (white, pink or a background clip
of the other dataset, seeded per clip by CRC32 of "<name>|<kind>", so the
same background clip and offset at every SNR) and mix (RMS SNR over the whole
clip). Called with the same CSVs and audio folders as Stage 2's extraction,
it gives the AST tier's waveforms; check_same_audio.py checks that end to
end, and bg_plan.json can be compared with Stage 2's.

Images and backbone: QWave/effnet_quant.py (MelImageSpec defaults, pooled
1536-d features, no random layer). Nothing here is random besides the seeded
noise; torch is seeded anyway (--seed) and the seed is recorded.

Output per (configuration, condition): <out>/<config>__<condition>.npy,
float32 [N, 1536] in the CSV's row order, a JSON sidecar (rendering spec,
costs, timing, names checksum), and once per condition
audiostat__<condition>.npy, the spectral flatness of each clip's mel power
spectrogram (the audio-statistic gate's score). Partial results are saved
every --save-every chunks and picked up on restart.

    python scripts/redesign/idea6/stage2/extract_eff.py --csv AST_CSV --audio-dir AUDIO \
        --bg-audio-dir OTHER_AUDIO --bg-csv OTHER_CSV --out-dir OUT --configs fp32 w8a8 \
        --conditions clean bg:20 bg:10 ... [--limit 64]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--bg-audio-dir", type=Path, default=None)
    p.add_argument("--bg-csv", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--configs", nargs="+", default=["fp32", "w8a8"])
    p.add_argument("--conditions", nargs="+", required=True, help="clean or KIND:SNR (bg, pink, white)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--save-every", type=int, default=4)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from QWave.effnet_quant import (EMB_DIM, MelImageSpec, backbone_bitops, build_backbone, conv_macs,  # noqa: E402
                                efficientnet_b3_pooled, mel_image_frontend_macs, mel_images, weight_bits_stored)
from audio_shift import NoiseMaker, cond_name, index_wavs, load_clip, mix  # noqa: E402


def parse_condition(c):
    if c == "clean":
        return "clean", None
    kind, snr = c.split(":")
    if kind not in ("bg", "pink", "white"):
        raise SystemExit(f"bad condition {c}")
    return kind, float(snr)


class Audio:
    """Clean 16 kHz waveforms (cached) and their corrupted versions."""

    def __init__(self, stems, index, noise_maker):
        self.stems, self.index, self.noise, self.cache = stems, index, noise_maker, {}
        self.n_silent = 0

    def clean(self, i):
        if i not in self.cache:
            self.cache[i] = load_clip(self.index[self.stems[i]])
        return self.cache[i]

    def get(self, i, kind, snr):
        x = self.clean(i)
        if kind == "clean":
            return x
        y, mixed = mix(x, self.noise(self.stems[i], kind, len(x)), snr)
        self.n_silent += not mixed
        return y


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

    spec = MelImageSpec()
    device = torch.device(args.device)
    base = efficientnet_b3_pooled()
    macs = conv_macs(base, spec.size)
    nets = {c: build_backbone(c, base, device) for c in args.configs}
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(1)

    for kind, snr in plan:
        cname = cond_name(kind, snr)
        todo = [c for c in args.configs if not (out / f"{c}__{cname}.npy").exists()]
        stat_path = out / f"audiostat__{cname}.npy"
        if not todo and stat_path.exists():
            continue
        part = out / f"_partial__{cname}.npz"
        X = {c: np.full((n, EMB_DIM), np.nan, np.float32) for c in todo}
        flat = np.full(n, np.nan)
        lens = np.zeros(n, np.int64)
        done = np.zeros(n, bool)
        if part.exists():
            d = np.load(part)
            if sorted(d["configs"].tolist()) == sorted(todo):
                done, flat, lens = d["done"], d["flat"], d["lens"]
                for c in todo:
                    X[c] = d[c]
                print(f"{cname}: resuming, {int(done.sum())} rows done", flush=True)
        audio.n_silent = 0
        prep_s, fwd_s = 0.0, {c: 0.0 for c in todo}
        t0 = time.time()
        left = np.flatnonzero(~done)
        chunks = [left[s:s + args.chunk] for s in range(0, len(left), args.chunk)]

        def prepare(idx):
            tp = time.time()
            waves = [audio.get(i, kind, snr) for i in idx]
            imgs, fl = mel_images(waves, spec, with_flatness=True)
            return imgs, fl, np.array([len(w) for w in waves]), time.time() - tp

        nxt = pool.submit(prepare, chunks[0]) if chunks else None
        for ci, idx in enumerate(chunks):
            imgs, fl, ln, dt_prep = nxt.result()
            prep_s += dt_prep
            if ci + 1 < len(chunks):
                nxt = pool.submit(prepare, chunks[ci + 1])
            flat[idx], lens[idx] = fl, ln
            x = imgs.to(device)
            for c in todo:
                tf = time.time()
                with torch.no_grad():
                    outs = [nets[c](x[b:b + args.batch_size]).float().cpu().numpy()
                            for b in range(0, len(idx), args.batch_size)]
                X[c][idx] = np.concatenate(outs)
                fwd_s[c] += time.time() - tf
            done[idx] = True
            if (ci + 1) % args.save_every == 0:
                np.savez(part, configs=np.array(todo), done=done, flat=flat, lens=lens, **X)
            el = time.time() - t0
            print(f"  {cname}: {int(done.sum())}/{n} rows, {el:.0f}s, "
                  f"{(done.sum() - (n - len(left))) / max(el, 1e-9):.1f} clips/s", flush=True)
        dt = time.time() - t0
        front = float(np.mean([mel_image_frontend_macs(int(m), spec) for m in lens]))
        for c in todo:
            if not np.isfinite(X[c]).all():
                raise SystemExit(f"{c}__{cname}: non-finite embeddings")
            np.save(out / f"{c}__{cname}.npy", X[c])
            side = {"config": c, "condition": cname, "noise": kind, "snr_db": snr, "n_clips": n,
                    "silent_clips_left_clean": audio.n_silent, "forward_seconds": fwd_s[c],
                    "clips_per_s_forward": len(left) / max(fwd_s[c], 1e-9), "condition_seconds": dt,
                    "prep_seconds": prep_s, "configs_in_pass": todo, "device": str(device),
                    "batch_size": args.batch_size, "chunk": args.chunk, "torch_seed": args.seed,
                    "csv": str(args.csv), "audio_dir": str(args.audio_dir), "bg_csv": str(args.bg_csv),
                    "bg_audio_dir": str(args.bg_audio_dir), "mel_image": spec.as_dict(),
                    "backbone": "torchvision efficientnet_b3 IMAGENET1K_V1, classifier = Identity (pooled 1536-d)",
                    "torch": torch.__version__, "names_crc32": crc,
                    "cost_per_clip": {"backbone_macs": macs["macs"], "backbone_out_elems": macs["out_elems"],
                                      "parts_macs": macs["parts_macs"],
                                      "backbone_bitops": backbone_bitops(macs, c),
                                      "backbone_bitops_rescale1024": backbone_bitops(macs, c, 1024.0),
                                      "frontend_macs_mean": front, "frontend_bitops_mean": front * 32 * 32,
                                      "storage_bits": weight_bits_stored(base, c)}}
            (out / f"{c}__{cname}.json").write_text(json.dumps(side, indent=2))
        np.save(stat_path, flat)
        np.save(out / f"lengths__{cname}.npy", lens)
        if kind == "bg":
            planf = out / "bg_plan.json"
            old = json.loads(planf.read_text()) if planf.exists() else {}
            old.update(noise_maker.plan)
            planf.write_text(json.dumps(old, indent=0))
        if part.exists():
            part.unlink()
        print(f"{cname}: {len(left)} rows x {len(todo)} configs in {dt:.0f}s (prep {prep_s:.0f}s, forward "
              + ", ".join(f"{c} {fwd_s[c]:.0f}s" for c in todo) + f"), silent {audio.n_silent}", flush=True)
    print(f"done in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
