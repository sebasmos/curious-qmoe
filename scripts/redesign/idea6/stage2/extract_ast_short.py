"""AST clip embeddings at a shorter input length (idea 6 Stage 2): idea 5
Stage 2's extract_stage2.py (branch redesign/idea5-stage2, a422003) with one
addition, --max-length. The feature extractor pads each clip's fbank to
max_length frames instead of 1024 and the model is cut to that length by
QWave.ast_frames.shorten_ast (the first time positions of the position
table), so a clip that fits sees the same patches and position embeddings
as in the 1024-frame model, without the padding patches. Everything else
(corruption, field cells, configurations, chunks, sidecars) is unchanged.
ESC-50: 512 frames (clips are 5 s, 498 frames); UrbanSound8K: 400 (at most
4 s, 398 frames). Sidecars record max_length; their costs use the shorter
token count.

Idea 6 Stage 3b adds --frame-offset (default 0, which changes nothing): the
fbank is padded or cut to frame_offset + max_length frames, as the feature
extractor does, and the model reads frames frame_offset to frame_offset +
max_length - 1, so a second 256-frame window of a clip uses the same fbank
values as the first-window and full-length inputs. Sidecars record it.

Original description:
AST clip embeddings for idea 5 Stage 2: many backbone configurations on
the same corrupted audio, batched on one device.

For each condition (clean, or a noise kind at an SNR, Stage 1's corruption
code via audio_shift.py) the clips are read and turned into the model's
1024-frame fbank input once per chunk, then every requested configuration
(QWave.backbone_quant.parse_config names: fp32, w8, w4g128, n4, ...) embeds
the chunk in turn: one QuantAST re-quantizes its weights from the FP32
master at each switch. Output per (configuration, condition):
<out>/<config>__<condition>.npy, float32 [N, 768] in the CSV's row order,
and a JSON sidecar with timing and the configuration's per-clip costs.

--field adds the field-mixture conditions: every cell of
audio_shift.FIELD_CELLS that is not also asked for in full is extracted only
for the clips that the test or the validation field assignment gives it
(other rows are NaN).

A partial result is saved every --save-every chunks
(<config>__<condition>.partial.npz) and picked up on restart.

    python scripts/redesign/idea5/extract_stage2.py --csv CLEAN_AST_CSV --audio-dir AUDIO \
        --bg-audio-dir OTHER_AUDIO --bg-csv OTHER_CSV --out-dir OUT \
        --configs fp32 w8 w4 n4 --conditions clean bg:10 [--field] [--limit 64]
"""
from __future__ import annotations

import argparse
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
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--conditions", nargs="*", default=[], help="clean or KIND:SNR (bg, pink, white)")
    p.add_argument("--field", action="store_true", help="add the field-mixture cells for the clips that need them")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--model", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    p.add_argument("--revision", default="f826b80d28226b62986cc218e5cec390b1096902")
    p.add_argument("--max-length", type=int, required=True, help="input frames (1024 in the pretrained model)")
    p.add_argument("--frame-offset", type=int, default=0, help="first fbank frame the model reads (Stage 3b)")
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--chunk", type=int, default=256, help="clips prepared at once and run through every config")
    p.add_argument("--save-every", type=int, default=4, help="chunks between partial saves")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--cache-bg", action="store_true")
    p.add_argument("--p-bits", type=int, default=16, help="bits of the attention probabilities (ablation: 8)")
    p.add_argument("--name-suffix", default="", help="appended to every config's file names (ablations)")
    p.add_argument("--wait-for", nargs="*", default=[],
                   help="before each condition, wait while a process matching any of these patterns runs")
    return p.parse_args(argv)


def wait_for(patterns, poll=60):
    """Block while `pgrep -f` finds another process matching a pattern (the
    GPU is shared: one extraction at a time)."""
    import subprocess

    while True:
        busy = []
        for pat in patterns:
            r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
            busy += [p for p in r.stdout.split() if int(p) != os.getpid()]
        if not busy:
            return
        print(f"  waiting: processes {busy} match {patterns}", flush=True)
        time.sleep(poll)


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

from QWave.ast_frames import shorten_ast  # noqa: E402
from QWave.backbone_quant import QuantAST, backbone_cost, parse_config  # noqa: E402
from audio_shift import (FIELD_CELLS, FIELD_SEEDS, NoiseMaker, cond_name, field_assignment,  # noqa: E402
                         index_wavs, load_clip, mix)

TARGET_SR = 16000


def parse_condition(c):
    if c == "clean":
        return "clean", None
    kind, snr = c.split(":")
    if kind not in ("bg", "pink", "white"):
        raise SystemExit(f"bad condition {c}")
    return kind, float(snr)


def plan_conditions(conditions, field, n):
    """[(kind, snr, row indices)] to extract; field cells not asked for in
    full get only the rows the test or validation assignment gives them."""
    plan = [(k, s, np.arange(n)) for k, s in map(parse_condition, conditions)]
    if field:
        full = {cond_name(k, s) for k, s, _ in plan}
        assign = [np.array(field_assignment(n, seed)) for seed in FIELD_SEEDS.values()]
        for kind, snr in FIELD_CELLS:
            name = cond_name(kind, snr)
            if name not in full:
                rows = np.flatnonzero((assign[0] == name) | (assign[1] == name))
                plan.append((kind, float(snr), rows))
    return plan


def main():
    args = ARGS
    t_start = time.time()
    meta = pd.read_csv(args.csv, usecols=["folder", "name", "class_id"])
    if args.limit:
        meta = meta.iloc[:args.limit]
    stems = meta["name"].map(lambda s: Path(s).stem).tolist()
    n = len(stems)
    index = index_wavs(args.audio_dir)
    missing = [s for s in stems if s not in index]
    if missing:
        raise SystemExit(f"{len(missing)} clips missing, e.g. {missing[:5]}")
    plan = plan_conditions(args.conditions, args.field, n)
    noise_maker = NoiseMaker([], {}, False)
    if any(k == "bg" for k, _, _ in plan):
        bg_names = pd.read_csv(args.bg_csv, usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist()
        bg_index = index_wavs(args.bg_audio_dir)
        noise_maker = NoiseMaker([b for b in bg_names if b in bg_index], bg_index, args.cache_bg)

    from transformers import ASTFeatureExtractor, ASTModel

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    ext_len = args.frame_offset + args.max_length
    extractor = ASTFeatureExtractor.from_pretrained(args.model, revision=args.revision, max_length=ext_len)
    model = shorten_ast(ASTModel.from_pretrained(args.model, revision=args.revision), args.max_length)
    model = model.to(device).eval()
    if extractor.max_length != ext_len:
        raise SystemExit("feature extractor ignored max_length")
    qast = QuantAST(model, p_bits=args.p_bits)
    specs = {c: parse_config(c) for c in args.configs}
    costs = {c: backbone_cost(model, *specs[c], p_bits=args.p_bits) for c in args.configs}
    sfx = args.name_suffix
    current = None

    for kind, snr, rows in plan:
        cname = cond_name(kind, snr)
        todo = [c for c in args.configs if not (out / f"{c}{sfx}__{cname}.npy").exists()]
        if not todo:
            continue
        wait_for(args.wait_for)
        part = out / f"_partial{sfx}__{cname}.npz"
        X = {c: np.full((n, 768), np.nan, np.float32) for c in todo}
        done = np.zeros(n, bool)
        if part.exists():
            d = np.load(part)
            if sorted(d["configs"].tolist()) == sorted(todo):
                done = d["done"]
                for c in todo:
                    X[c] = d[c]
                print(f"{cname}: resuming, {int(done.sum())} rows done", flush=True)
        prep_s, fwd_s, n_silent = 0.0, {c: 0.0 for c in todo}, 0
        t0 = time.time()
        left = rows[~done[rows]]
        for ci, start in enumerate(range(0, len(left), args.chunk)):
            idx = left[start:start + args.chunk]
            tp = time.time()
            waves = []
            for i in idx:
                x = load_clip(index[stems[i]])
                if kind != "clean":
                    x, mixed = mix(x, noise_maker(stems[i], kind, len(x)), snr)
                    n_silent += not mixed
                waves.append(x)
            feats = extractor(waves, sampling_rate=TARGET_SR, return_tensors="pt")["input_values"]
            feats = feats[:, args.frame_offset:args.frame_offset + args.max_length].contiguous().to(device)
            prep_s += time.time() - tp
            for c in todo:
                if current != c:
                    qast.set_config(*specs[c])
                    current = c
                tf = time.time()
                outs = [qast(feats[b:b + args.batch_size]).float().cpu().numpy()
                        for b in range(0, len(idx), args.batch_size)]
                X[c][idx] = np.concatenate(outs)
                fwd_s[c] += time.time() - tf
            done[idx] = True
            if (ci + 1) % args.save_every == 0:
                np.savez(part, configs=np.array(todo), done=done, **X)
            el = time.time() - t0
            print(f"  {cname}: {int(done[rows].sum())}/{len(rows)} rows, {el:.0f}s, "
                  f"{done[rows].sum() * len(todo) / max(el, 1e-9):.2f} clip-configs/s", flush=True)
        dt = time.time() - t0
        for c in todo:
            if not np.isfinite(X[c][rows]).all():
                raise SystemExit(f"{c}{sfx}__{cname}: non-finite embeddings")
            np.save(out / f"{c}{sfx}__{cname}.npy", X[c])
            side = {"config": c + sfx, "p_bits": args.p_bits, "max_length": args.max_length,
                    "frame_offset": args.frame_offset,
                    "condition": cname, "noise": kind, "snr_db": snr,
                    "n_clips": n,
                    "rows_extracted": int(len(rows)), "silent_clips_left_clean": n_silent,
                    "forward_seconds": fwd_s[c], "clips_per_s_forward": len(left) / max(fwd_s[c], 1e-9),
                    "condition_seconds": dt, "prep_seconds": prep_s, "configs_in_pass": todo,
                    "device": str(device), "batch_size": args.batch_size, "chunk": args.chunk,
                    "csv": str(args.csv), "model": args.model, "revision": args.revision,
                    "backbone_cost_per_clip": costs[c], "names_crc32": zlib.crc32("\n".join(stems).encode())}
            (out / f"{c}{sfx}__{cname}.json").write_text(json.dumps(side, indent=2))
        if kind == "bg":
            planf = out / "bg_plan.json"
            old = json.loads(planf.read_text()) if planf.exists() else {}
            old.update(noise_maker.plan)
            planf.write_text(json.dumps(old, indent=0))
        if part.exists():
            part.unlink()
        rate = len(left) * len(todo) / dt
        print(f"{cname}: {len(left)} rows x {len(todo)} configs in {dt:.0f}s ({rate:.2f} clip-configs/s, "
              f"prep {prep_s:.0f}s), silent {n_silent}", flush=True)
    print(f"done in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
