"""T19 step 19.4 (ANALYSIS_PLAN_T19.md D1, D2, B2): calibration of one task, on the GPU.

256 calibration inputs from the other dataset(s), each filled to the task's input length by concatenating randomly drawn clean clips
(numpy default_rng(19)); no clip of the task is used (asserted by name). Then:
  1. FP32 forward with hooks: per-channel absmax of the four Linear inputs (SmoothQuant scales, alpha 0.5) and token samples
     (clip bounds at the 99.9th and 99.99th percentiles after smoothing);
  2. clip choice per activation bit-width, p in {99.9, 99.99, none}, by the lowest mean squared error of the pooled embedding against
     FP32 (A4 judged on n4g128, A6 on w4g128, A8 on w8g128), label free;
  3. sensitivity scan for the mixed models: n4g128 with A8 everywhere (A8's chosen clip) as reference, block i alone at A4; relative
     squared error of the pooled embedding, mean over the inputs. Order least sensitive first.
Writes <results>/calib/<task>.pt (scales and bounds) and <task>.json (choices, errors, order, clip names).

    python t19_calib.py --task esc50
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t19_common import CALIB_SOURCES, MODEL, REVISION, TASKS, results_dir  # noqa: E402
from QWave.ast_frames import shorten_ast  # noqa: E402
from QWave.backbone_quant import (QuantAST, T19Act, WeightSpec, calibration_tables, collect_calibration,  # noqa: E402
                                  parse_config)
from audio_shift import index_wavs, load_clip  # noqa: E402

PCTS = (99.9, 99.99, None)
JUDGE = {4: "n4g128", 6: "w4g128", 8: "w8g128"}


def stems(task):
    return pd.read_csv(TASKS[task]["csv"], usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist()


def build_inputs(task, rng, cap=None):
    n_samples = TASKS[task]["length"] * 160 + 400
    waves, used = [], []
    for src, n in CALIB_SOURCES[task]:
        n = n if cap is None else min(n, cap)
        names = stems(src)
        index = index_wavs(TASKS[src]["audio"])
        for _ in range(n):
            parts, tot, picked = [], 0, []
            while tot < n_samples:
                nm = names[int(rng.integers(len(names)))]
                x = load_clip(index[nm])
                parts.append(x); tot += len(x); picked.append(nm)
            waves.append(np.concatenate(parts)[:n_samples].astype(np.float32))
            used.append({"source": src, "clips": picked})
    own = set(stems(task)) if task != "tau" else set(stems("tau"))
    assert not any(c in own for u in used for c in u["clips"]), "a calibration clip belongs to the task"
    return waves, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["esc50", "urban", "tau"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--smoke", type=int, default=None, help="inputs per source, output to --out (code test only)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    t0 = time.time()
    R = results_dir()
    L = TASKS[a.task]["length"]
    rng = np.random.default_rng(19)
    waves, used = build_inputs(a.task, rng, a.smoke)
    from transformers import ASTFeatureExtractor, ASTModel

    ext = ASTFeatureExtractor.from_pretrained(MODEL, revision=REVISION, max_length=L)
    feats = ext(waves, sampling_rate=16000, return_tensors="pt")["input_values"]
    assert feats.shape[1] == L
    dev = torch.device(a.device)
    model = shorten_ast(ASTModel.from_pretrained(MODEL, revision=REVISION), L).to(dev).eval()
    qa = QuantAST(model, p_bits=8)
    batches = [feats[i:i + a.batch_size].to(dev) for i in range(0, len(feats), a.batch_size)]
    print(f"{a.task}: {len(feats)} inputs of {L} frames, prep {time.time() - t0:.0f}s", flush=True)

    cal = collect_calibration(qa, batches, tokens_per_clip=8, seed=19)
    tab = calibration_tables(qa, cal, percentiles=(99.9, 99.99))
    print(f"calibration statistics {time.time() - t0:.0f}s", flush=True)

    def run(spec, act):
        if act is None:
            qa.set_config(spec, None)
        else:
            qa.set_t19(spec, act)
        return torch.cat([qa(b).float().cpu() for b in batches])

    ref = run(WeightSpec(32), None)

    def clipd(p):
        return None if p is None else tab["clip"][p]

    mse, choice = {}, {}
    for bits, wname in JUDGE.items():
        spec = parse_config(wname)[0]
        for p in PCTS:
            e = run(spec, T19Act(layer_bits=(bits,) * 12, p_bits=8, smooth=tab["smooth"],
                                 clip=None if p is None else {bits: clipd(p)}))
            mse[f"a{bits}_p{p}"] = float(((e - ref) ** 2).mean())
            mse[f"rel_a{bits}_p{p}"] = float((((e - ref) ** 2).sum(1) / (ref ** 2).sum(1)).mean())
            print(f"  A{bits} on {wname}, clip {p}: mse {mse[f'a{bits}_p{p}']:.5g}  {time.time() - t0:.0f}s", flush=True)
        choice[bits] = min(PCTS, key=lambda p: mse[f"a{bits}_p{p}"])
    clip_by_bits = {b: clipd(choice[b]) for b in (4, 6, 8)}

    def act(bits_list):
        cl = {b: clip_by_bits[b] for b in set(bits_list)}  # a None entry means no clamp at that width
        return T19Act(layer_bits=tuple(bits_list), p_bits=8, smooth=tab["smooth"], clip=cl)

    n4 = WeightSpec(4, 128, nested=True)
    base = run(n4, act([8] * 12))
    sens = []
    for i in range(12):
        bl = [8] * 12
        bl[i] = 4
        e = run(n4, act(bl))
        sens.append(float((((e - base) ** 2).sum(1) / (base ** 2).sum(1)).mean()))
        print(f"  block {i}: {sens[-1]:.5g}  {time.time() - t0:.0f}s", flush=True)
    order = [int(i) for i in np.argsort(sens, kind="stable")]
    out = a.out or R / "calib"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"smooth": tab["smooth"], "clip": tab["clip"], "choice": choice}, out / f"{a.task}.pt")
    info = {"task": a.task, "frames": L, "n_inputs": len(waves), "sources": CALIB_SOURCES[a.task], "rng_seed": 19,
            "tokens_per_clip": 8, "alpha": 0.5, "clip_mse_vs_fp32": mse,
            "clip_choice": {str(b): choice[b] for b in choice}, "sensitivity_rel_sq_err": sens,
            "sensitivity_order_least_first": order, "mp6_a4_blocks": sorted(order[:6]), "mp8_a4_blocks": sorted(order[:8]),
            "inputs": used, "seconds": time.time() - t0, "device": str(dev), "batch_size": a.batch_size}
    (out / f"{a.task}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "inputs"}, indent=1))


if __name__ == "__main__":
    main()
