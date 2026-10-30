"""CPU wall clock of every PyTorch step of every tier, batch size 1, one thread
count per process (the launcher sets OMP_NUM_THREADS and
VECLIB_MAXIMUM_THREADS to --threads before Python starts; Accelerate reads the
latter when it loads).

Clips: the first full-length clip in CSV order (ESC-50 1-100032-A-0, 5 s;
UrbanSound8K 100263-2-0-117, 4 s), read by audio_shift.load_clip.
Steps (dataset in brackets):
  mel_front_end [esc50, urban8k]   Stage 2's mel image front end of the
                                   EfficientNet tier (STFT, mel, dB, magma,
                                   300 x 300 resize)
  mn_front_end [esc50, urban8k]    EfficientAT's documented front end
                                   (resample 16 to 32 kHz, pre-emphasis, STFT,
                                   128 Kaldi mel bands, log, normalization);
                                   the UrbanSound8K clip zero-padded to 4.04 s
                                   as in Stage 3b
  fbank_512 [esc50], fbank_400 [urban8k]
                                   the AST feature extractor (Kaldi fbank,
                                   padded to 512 or 400 frames)
  effnet_torch_fp32, effnet_torch_int8_fx [both]
                                   EfficientNet-B3 on the ESC-50 image (the
                                   image is 300 x 300 for every clip); the
                                   int8 FX qnnpack model of the first study,
                                   state dict reloaded and checked against its
                                   reference output
  mn10_torch_fp32, mn20_torch_fp32 [esc50, urban8k]
                                   EfficientAT mn10 and mn20, released AudioSet
                                   weights, on each clip's log mel (W8A8 is fake
                                   quantization on FP32 kernels, not timed)
  head_* [both]                    the heads of the cascades (fold 1, seed 0,
                                   prepared as scored): Stage 3c's mn20 ESC-50/50
                                   4-bit head, Stage 3b's mn10 UrbanSound8K FP32
                                   head, Stage 2's AST FP32 heads; the first
                                   study's EfficientNet heads kept for reference
  ast_<frames>_fp32, ast_<frames>_int8_dynamic, ast_<frames>_n4_kleidiai
                                   the short AST at 512 [esc50] and 400 [urban8k]
                                   frames on the matching clip
--warm passes, then --reps timed passes, each step a block; before each
block the 1-minute load must be below --max-load (waits up to 20 min, then
runs and flags it); the load before and after each block and the top CPU
users after it are recorded. Raw passes: raw/torch_passes_t<threads>.csv.gz;
summary: raw/torch_summary_t<threads>.json.

    OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
        python scripts/time_torch.py --threads 2
"""
from __future__ import annotations

import argparse
import contextlib
import fnmatch
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

HEADS = {  # step -> (results folder, run dir under its outputs, input dim, task)
    "head_cheap_mn20_esc50_4bit": (C.S3C, "mn20_esc50_s0/4", 1920, "esc50"),
    "head_cheap_mn10_urban_fp32": (C.S3B, "mn10_urban_s0/esc", 960, "urban"),
    "head_cheap_esc50_fp32": (C.S2, "effp_esc50_s0/esc", 1536, "esc50"),
    "head_cheap_urban_8bit": (C.S2, "effp_urban_s0/8", 1536, "urban"),
    "head_ast_esc50_fp32": (C.S2, "astshort_esc50_s0/esc", 768, "esc50"),
    "head_ast_urban_fp32": (C.S2, "astshort_urban_s0/esc", 768, "urban"),
}
N_CLS = {"esc50": 50, "esc5": 5, "urban": 10}


def build_steps():
    from scripts.analysis.cascade import load_model

    from QWave.effnet_quant import MelImageSpec, mel_images

    steps, info = {}, {}
    spec = MelImageSpec()
    waves = {}
    for ds in ("esc50", "urban8k"):
        nm, wave = C.timing_clip(ds)
        waves[ds] = wave
        info[f"clip_{ds}"] = {"name": nm, "samples_16k": len(wave)}
        steps[("mel_front_end", ds)] = lambda w=wave: mel_images([w], spec)
        mw = C.mn_wave(ds, wave)
        steps[("mn_front_end", ds)] = lambda w=mw: C.mn_input(w)
        fe = C.ast_extractor(C.SHORT[ds])
        steps[(f"fbank_{C.SHORT[ds]}", ds)] = lambda w=wave, f=fe: f([w], sampling_rate=16000, return_tensors="pt")

    img = torch.from_numpy(np.load(C.OLD / "cache/timing_esc50_u8.npy")).float() / 255.0
    fp32 = C.effnet_fp32()
    cal = torch.from_numpy(np.load(C.OLD / "cache/calib_u8.npy")[:1]).float() / 255.0
    q = C.load_effnet_int8(C.OLD / "models/effnet_int8_fx.pt", fp32, cal)
    ref = torch.load(C.OLD / "models/effnet_int8_ref_out.pt", weights_only=False)
    te = torch.from_numpy(np.load(C.OLD / "cache/test_u8.npy")[:4]).float() / 255.0
    with torch.no_grad():
        if not torch.equal(q(te), ref["out"]):
            raise SystemExit("int8 EfficientNet does not reproduce its reference output")
    info["effnet_int8_reload_equal"] = True
    steps[("effnet_torch_fp32", "both")] = lambda: fp32(img)
    steps[("effnet_torch_int8_fx", "both")] = lambda: q(img)

    for name in C.MN_FILES:
        m = C.mn_fp32(name)
        for ds in ("esc50", "urban8k"):
            x = torch.from_numpy(np.load(C.OUT / f"cache/mn_mel_{ds}.npy"))
            with torch.no_grad():
                saved = np.load(C.OUT / f"cache/mn_emb_{name}_{ds}_torch.npy")
                err = float(np.abs(m(x).numpy() - saved).max())
            info[f"{name}_{ds}_maxabs_vs_prep"] = err
            info[f"{name}_{ds}_mel_frames"] = int(x.shape[-1])
            steps[(f"{name}_torch_fp32", ds)] = lambda m=m, x=x: m(x)

    for step, (res, run, dim, task) in HEADS.items():
        with contextlib.redirect_stdout(io.StringIO()):
            m, _ = load_model(res / "outputs" / run, 1, dim, N_CLS[task])
        e = torch.randn(1, dim)
        steps[(step, "both")] = lambda m=m, e=e: m(e)

    for ds in ("esc50", "urban8k"):
        L = C.SHORT[ds]
        fe = C.ast_extractor(L)
        x = fe([waves[ds]], sampling_rate=16000, return_tensors="pt")["input_values"]
        a32 = C.ast_fp32(L)
        a8, n_lin = C.ast_int8_dynamic(a32)
        a4, n_kai = C.ast_n4_kleidiai(a32)
        info[f"ast_{L}_linear_layers"] = {"int8_dynamic": n_lin, "n4_kleidiai": n_kai}
        for name, m in (("fp32", a32), ("int8_dynamic", a8), ("n4_kleidiai", a4)):
            steps[(f"ast_{L}_{name}", ds)] = lambda m=m, x=x: m(input_values=x).pooler_output
    return steps, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--warm", type=int, default=50)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--max-load", type=float, default=9.0)
    ap.add_argument("--only", nargs="*", default=["*"], help="globs over step names")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    C.set_threads(a.threads)
    steps, info = build_steps()
    info.update({"threads": a.threads, "warm": a.warm, "reps": a.reps, "max_load": a.max_load,
                 "torch": torch.__version__, "torch_num_threads": torch.get_num_threads(),
                 "quantized_engine": torch.backends.quantized.engine,
                 "load_start": C.loadavg(), "busy_start": C.busy_processes(8)})
    rows, raw = [], []
    with torch.no_grad():
        for (step, ds), fn in steps.items():
            if not any(fnmatch.fnmatch(step, g) for g in a.only):
                continue
            q = C.wait_quiet(a.max_load)
            t = C.time_fn(fn, a.warm, a.reps)
            rows.append({"step": step, "dataset": ds, "threads": a.threads, "runtime": f"torch {torch.__version__}",
                         **C.summarize(t), **q, "load_after": C.loadavg(), "busy_after": C.busy_processes(5)})
            raw.append(pd.DataFrame({"step": step, "dataset": ds, "threads": a.threads, "pass": np.arange(len(t)),
                                     "ms": t}))
            r = rows[-1]
            print(f"t{a.threads} {step} [{ds}]: {r['median_ms']:.2f} ms (IQR {r['iqr_ms']:.2f}, p95 {r['p95_ms']:.2f}), "
                  f"load {q['load_before'][0]} -> {r['load_after'][0]}{' WAITED' if q['waited_s'] else ''}"
                  f"{' HIGH' if q['still_high'] else ''}", flush=True)
    info["load_end"] = C.loadavg()
    out = C.OUT / "raw"
    out.mkdir(exist_ok=True)
    (out / f"torch_summary_t{a.threads}{a.tag}.json").write_text(json.dumps({"info": info, "rows": rows}, indent=1))
    pd.concat(raw).to_csv(out / f"torch_passes_t{a.threads}{a.tag}.csv.gz", index=False)


if __name__ == "__main__":
    main()
