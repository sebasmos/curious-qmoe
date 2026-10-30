"""CPU wall clock of the ONNX Runtime models, batch size 1, in the conda env
qmoe-ort (onnxruntime 1.31): CPU execution provider, intra-op threads =
--threads, inter-op 1, sequential execution, all graph optimizations.

Models (dataset in brackets):
  effnet_ort_fp32, effnet_ort_int8_qdq [both]   the first study's EfficientNet-B3
      (models/effnet_fp32_pre.onnx, models/effnet_int8_qdq.onnx: static QDQ,
      Percentile calibration on 160 clean ESC-50 clips) on its timing image
  mn10_ort_fp32, mn20_ort_fp32 [esc50, urban8k]   Stage 3c's FP32 exports
      (models/<model>_fp32_pre.onnx), checked against the torch embedding of
      prep_inputs.py
  mn10_ort_int8, mn20_ort_int8 [esc50, urban8k]   Stage 3c's static QDQ int8
      models, fold 1 (models/esc50/mn10_int8_f1.onnx and
      models/urban8k/mn10_int8_f1.onnx; mn20 has ESC-50 builds only, so
      models/esc50/mn20_int8_f1.onnx runs on both clips: the folds differ in
      their calibration scales only, the graph is the same)
Inputs: cache/mn_mel_<ds>.npy from prep_inputs.py. --warm and --reps as
time_torch.py, the same load gate (--max-load) and load records. Raw passes:
raw/ort_passes_t<threads>.csv.gz; summary raw/ort_summary_t<threads>.json.

    PYO=<python of the qmoe-ort environment>
    OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 $PYO scripts/time_ort.py --threads 2
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402


def session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def models():
    img = np.load(C.OLD / "cache/timing_esc50_u8.npy").astype(np.float32) / np.float32(255.0)
    out = [("effnet_ort_fp32", "both", C.OLD / "models/effnet_fp32_pre.onnx", {"image": img}),
           ("effnet_ort_int8_qdq", "both", C.OLD / "models/effnet_int8_qdq.onnx", {"image": img})]
    mel = {ds: np.load(C.OUT / f"cache/mn_mel_{ds}.npy").astype(np.float32) for ds in ("esc50", "urban8k")}
    for name in ("mn10", "mn20"):
        for ds in ("esc50", "urban8k"):
            out.append((f"{name}_ort_fp32", ds, C.S3C / f"models/{name}_fp32_pre.onnx", {"mel": mel[ds]}))
    out += [("mn10_ort_int8", "esc50", C.S3C / "models/esc50/mn10_int8_f1.onnx", {"mel": mel["esc50"]}),
            ("mn10_ort_int8", "urban8k", C.S3C / "models/urban8k/mn10_int8_f1.onnx", {"mel": mel["urban8k"]}),
            ("mn20_ort_int8", "esc50", C.S3C / "models/esc50/mn20_int8_f1.onnx", {"mel": mel["esc50"]}),
            ("mn20_ort_int8", "urban8k", C.S3C / "models/esc50/mn20_int8_f1.onnx", {"mel": mel["urban8k"]})]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--warm", type=int, default=50)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--max-load", type=float, default=9.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    info = {"threads": a.threads, "warm": a.warm, "reps": a.reps, "max_load": a.max_load,
            "onnxruntime": ort.__version__, "load_start": C.loadavg(), "busy_start": C.busy_processes(8),
            "models": {}}
    rows, raw = [], []
    for step, ds, path, feed in models():
        s = session(path, a.threads)
        info["models"][f"{step}/{ds}"] = str(path)
        if step.endswith("_ort_fp32") and step.startswith("mn"):
            ref = np.load(C.OUT / f"cache/mn_emb_{step[:4]}_{ds}_torch.npy")
            info[f"{step}_{ds}_maxabs_vs_torch"] = float(np.abs(s.run(None, feed)[0] - ref).max())
        q = C.wait_quiet(a.max_load)
        t = C.time_fn(lambda: s.run(None, feed), a.warm, a.reps)
        rows.append({"step": step, "dataset": ds, "threads": a.threads, "runtime": f"onnxruntime {ort.__version__}",
                     **C.summarize(t), **q, "load_after": C.loadavg(), "busy_after": C.busy_processes(5)})
        raw += [{"step": step, "dataset": ds, "threads": a.threads, "pass": i, "ms": v} for i, v in enumerate(t)]
        r = rows[-1]
        print(f"t{a.threads} {step} [{ds}]: {r['median_ms']:.2f} ms (IQR {r['iqr_ms']:.2f}, p95 {r['p95_ms']:.2f}), "
              f"load {q['load_before'][0]} -> {r['load_after'][0]}{' WAITED' if q['waited_s'] else ''}"
              f"{' HIGH' if q['still_high'] else ''}", flush=True)
    info["load_end"] = C.loadavg()
    out = C.OUT / "raw"
    out.mkdir(exist_ok=True)
    (out / f"ort_summary_t{a.threads}{a.tag}.json").write_text(json.dumps({"info": info, "rows": rows}, indent=1))
    with gzip.open(out / f"ort_passes_t{a.threads}{a.tag}.csv.gz", "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "dataset", "threads", "pass", "ms"])
        w.writeheader()
        w.writerows(raw)


if __name__ == "__main__":
    main()
