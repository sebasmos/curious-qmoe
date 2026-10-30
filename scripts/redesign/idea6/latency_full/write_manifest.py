"""run_manifest.json: machine, code, env versions, thread settings, the load before and after every timed block,
the other CPU users seen, the models and inputs used (sha256), the scripts (sha256), timing of the run.

    python scripts/write_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

PY = os.environ.get("PY", sys.executable)
PYO = os.environ.get("PYO", "python")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def versions(py, mods):
    code = "import json,importlib\nout={}\n" + "\n".join(
        f"\ntry:\n    out['{m}']=importlib.import_module('{m}').__version__\nexcept Exception as e:\n    out['{m}']=str(e)"
        for m in mods) + "\nprint(json.dumps(out))"
    return json.loads(subprocess.run([py, "-c", code], capture_output=True, text=True, env={"PYTHONDONTWRITEBYTECODE": "1"}).stdout)


def main():
    m = {"study": "redesign idea 6, full CPU latency protocol of the tiers and the Stage 3c cascade picks",
         "folder": str(C.OUT),
         "machine": {"cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
                     "cores": subprocess.run(["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True).stdout.strip(),
                     "perf_cores": subprocess.run(["sysctl", "-n", "hw.perflevel0.logicalcpu"], capture_output=True, text=True).stdout.strip(),
                     "eff_cores": subprocess.run(["sysctl", "-n", "hw.perflevel1.logicalcpu"], capture_output=True, text=True).stdout.strip(),
                     "macos": platform.mac_ver()[0], "ram_bytes": subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip(),
                     "idle_load_note": "idle 1-minute load 6 to 7.5 (VS Code helpers and an IT agent each hold a core); "
                                       "gate: 1-minute load below 9 at the start of each block, waiting up to 6 h"},
         "code": {"worktree": C.git_head(C.W), "note": "imported read-only (PYTHONDONTWRITEBYTECODE=1); QWave/effnet_quant.py, "
                  "ast_frames.py, backbone_quant.py and stage2/audio_shift.py are byte-identical to the Stage 2 worktree at c892c2f"},
         "envs": {"curious-qmoe": versions(PY, ["torch", "torchaudio", "torchvision", "transformers", "numpy", "pandas"]),
                  "qmoe-ort": versions(PYO, ["onnxruntime", "onnx", "numpy"])},
         "protocol": {"batch": 1, "threads": [1, 2, 4], "warm_up": 50, "timed_passes": 500, "statistics": "median, IQR, p95, mean",
                      "thread_env": "OMP_NUM_THREADS, VECLIB_MAXIMUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS = threads; "
                                    "torch.set_num_threads, interop 1; ORT intra-op = threads, inter-op 1, sequential",
                      "load_gate": "1-minute load below 9 before each block (common.wait_quiet), load recorded before and after"},
         "inputs": json.loads((C.OUT / "cache/inputs.json").read_text()),
         "blocks": {}, "logs": {}}
    for p in sorted((C.OUT / "raw").glob("*_summary_t*.json")):
        if "smoke" in p.name:
            continue
        d = json.loads(p.read_text())
        m["blocks"][p.stem] = {"info": {k: v for k, v in d["info"].items() if k != "models"},
                               "rows": [{k: r[k] for k in ("step", "dataset", "threads", "runtime", "n", "median_ms", "iqr_ms",
                                                            "p95_ms", "mean_ms", "load_before", "waited_s", "still_high",
                                                            "load_after", "busy_after")} for r in d["rows"]]}
    if (C.OUT / "logs/timing.done").exists():
        m["logs"]["timing.done"] = (C.OUT / "logs/timing.done").read_text().splitlines()
    files = {"old_effnet_int8_fx": C.OLD / "models/effnet_int8_fx.pt", "old_effnet_fp32_pre_onnx": C.OLD / "models/effnet_fp32_pre.onnx",
             "old_effnet_int8_qdq_onnx": C.OLD / "models/effnet_int8_qdq.onnx", "old_timing_image": C.OLD / "cache/timing_esc50_u8.npy",
             "mn10_weights": C.MN_WEIGHTS / C.MN_FILES["mn10"], "mn20_weights": C.MN_WEIGHTS / C.MN_FILES["mn20"],
             "mn10_fp32_pre_onnx": C.S3C / "models/mn10_fp32_pre.onnx", "mn20_fp32_pre_onnx": C.S3C / "models/mn20_fp32_pre.onnx",
             "mn10_int8_esc50_f1": C.S3C / "models/esc50/mn10_int8_f1.onnx", "mn10_int8_urban8k_f1": C.S3C / "models/urban8k/mn10_int8_f1.onnx",
             "mn20_int8_esc50_f1": C.S3C / "models/esc50/mn20_int8_f1.onnx",
             "head_mn20_esc50_4bit": C.S3C / "outputs/mn20_esc50_s0/4/fold_1/best_model.pth",
             "head_mn10_urban_fp32": C.S3B / "outputs/mn10_urban_s0/esc/fold_1/best_model.pth",
             "rates_esc50": C.S3C / "analysis/esc50/selected.csv", "rates_urban": C.S3C / "analysis/urban/selected.csv"}
    m["files_used"] = {k: {"path": str(p), "sha256": sha(p)} for k, p in files.items() if p.exists()}
    m["cache"] = {p.name: sha(p) for p in sorted((C.OUT / "cache").glob("*.npy"))}
    m["scripts"] = {p.name: sha(p) for p in sorted((C.OUT / "scripts").glob("*.*"))}
    m["raw"] = {p.name: sha(p) for p in sorted((C.OUT / "raw").glob("*")) if p.is_file()}
    if (C.OUT / "tables/rates.json").exists():
        m["rates"] = json.loads((C.OUT / "tables/rates.json").read_text())
    (C.OUT / "run_manifest.json").write_text(json.dumps(m, indent=1))
    print("manifest written:", len(m["blocks"]), "blocks")


if __name__ == "__main__":
    main()
