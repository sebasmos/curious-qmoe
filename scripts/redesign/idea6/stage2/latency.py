"""CPU wall clock of both tiers at batch size 1 (ANALYSIS_PLAN.md amendment 1).

One real clip per dataset (the first full-length clip in CSV order, read with
audio_shift.load_clip).
Steps timed separately: the mel image front end (QWave/effnet_quant.py),
EfficientNet-B3 FP32, EfficientNet-B3 W8A8 in fake quantization (a
simulation: it runs FP32 kernels plus the rounding, so it is slower than FP32
and says nothing about int8 speed), the cheap heads (scripts/benchmark.py's
networks, kinds esc and 4, random weights: the time does not depend on them),
the AST feature extractor (Kaldi fbank) and the FP32 AST at 1024 frames and at
the short length, and the AST heads. 20 warm-up and 100 timed passes each,
torch threads fixed, median and interquartile range in milliseconds. No
low-bit kernels exist here, so a quantized AST runs at FP32 time.

    python scripts/redesign/idea6/stage2/latency.py --out R/latency [--threads 2]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

E = Path.home() / "Documents" / "DATASETS"
CLIPS = {"esc50": (E / "ESC-50/ast_768/esc-50.csv", E / "ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio", 512, 50),
         "urban8k": (E / "Urban8k/ast_768/urbansound8k.csv",
                     E / "Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio", 400, 10)}


def timeit(fn, warm, reps):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    t = np.array(t) * 1000
    return {"median_ms": float(np.median(t)), "q25_ms": float(np.percentile(t, 25)), "q75_ms": float(np.percentile(t, 75))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    from omegaconf import OmegaConf
    from transformers import ASTFeatureExtractor, ASTModel

    from audio_shift import index_wavs, load_clip
    from QWave.ast_frames import shorten_ast
    from QWave.effnet_quant import MelImageSpec, build_backbone, efficientnet_b3_pooled, mel_images
    from scripts.benchmark import build_model

    cfg = OmegaConf.load(ROOT / "config" / "esc50.yaml")
    name, rev = "MIT/ast-finetuned-audioset-10-10-0.4593", "f826b80d28226b62986cc218e5cec390b1096902"
    base = efficientnet_b3_pooled()
    eff = {"fp32": build_backbone("fp32", base), "w8a8_fakequant": build_backbone("w8a8", base)}
    ast_full = ASTModel.from_pretrained(name, revision=rev).eval()
    rows, res = [], {"threads": a.threads, "warm": a.warm, "reps": a.reps, "torch": torch.__version__,
                     "load_average_start": os.getloadavg()}
    for ds, (csv, audio_dir, short_len, K) in CLIPS.items():
        index = index_wavs(audio_dir)
        for nm in pd.read_csv(csv, usecols=["name"])["name"]:  # first clip of full length (5 s ESC-50, 4 s US8K)
            wave = load_clip(index[Path(nm).stem])
            if len(wave) >= (80000 if ds == "esc50" else 64000):
                break
        res[f"{ds}_clip"] = {"name": nm, "samples_16k": len(wave)}
        spec = MelImageSpec()
        img = mel_images([wave], spec)
        with contextlib.redirect_stdout(io.StringIO()):
            heads = {(k, d): build_model(k, d, K, cfg).eval() for k in ("esc", "4") for d in (1536, 768)}
        fe_full = ASTFeatureExtractor.from_pretrained(name, revision=rev)
        fe_short = ASTFeatureExtractor.from_pretrained(name, revision=rev, max_length=short_len)
        ast_short = shorten_ast(ASTModel.from_pretrained(name, revision=rev), short_len).eval()
        x_full = fe_full([wave], sampling_rate=16000, return_tensors="pt")["input_values"]
        x_short = fe_short([wave], sampling_rate=16000, return_tensors="pt")["input_values"]
        e1536, e768 = torch.randn(1, 1536), torch.randn(1, 768)
        steps = {
            "mel_image_front_end": lambda: mel_images([wave], spec),
            "effnet_fp32": lambda: eff["fp32"](img),
            "effnet_w8a8_fakequant_simulation": lambda: eff["w8a8_fakequant"](img),
            "cheap_head_esc": lambda: heads[("esc", 1536)](e1536),
            "cheap_head_4": lambda: heads[("4", 1536)](e1536),
            "ast_fbank_1024": lambda: fe_full([wave], sampling_rate=16000, return_tensors="pt"),
            f"ast_fbank_{short_len}": lambda: fe_short([wave], sampling_rate=16000, return_tensors="pt"),
            "ast_fp32_1024": lambda: ast_full(input_values=x_full).pooler_output,
            f"ast_fp32_{short_len}": lambda: ast_short(input_values=x_short).pooler_output,
            "ast_head_esc": lambda: heads[("esc", 768)](e768),
        }
        with torch.no_grad():
            for step, fn in steps.items():
                r = timeit(fn, a.warm, a.reps)
                rows.append({"dataset": ds, "step": step, **r})
                print(ds, step, f"{r['median_ms']:.2f} ms", flush=True)
    res["load_average_end"] = os.getloadavg()
    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out / "latency.csv", index=False)
    (a.out / "latency.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
