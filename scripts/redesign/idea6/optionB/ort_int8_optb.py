"""Real int8 EfficientAT MN on ONNX Runtime for idea 6 option B (S4), conda env qmoe-ort, reusing Stage 3c's
ort_int8.py (export, session, quantize, node types). Differences from Stage 3c: the universe CSV of TAU 2020 Mobile
(every device), five location-grouped folds whose test rows are the fold's rows of every device, calibration clips
drawn from the fold's device-A training rows that are in no seed's inner-validation split of this stage's runs,
and the calibration method fixed to Stage 3c's label-free choice (Percentile; `--method` records it).

mels   float16 log-mel cache of every universe row (ort_mels.py's front end) -> R/cache_mel/tau/clean.npy
export the FP32 graph (Stage 3c's exportable rewrite) -> R/models/<model>_fp32[_pre].onnx, checked on 8 clips
build  one int8 model per fold -> R/models/tau/<model>_int8_f<f>.onnx, R/models/tau/<model>_build.json
embed  fold f's model embeds every universe row -> R/emb_int8/tau/<model>_int8_f<f>__clean.npy (the analysis reads
       fold f's test rows and fold f's fit and inner-validation rows from it); <model>_int8__clean.npy holds each
       row by the model of its own fold; ORT FP32 on every row -> <model>_ortfp32__clean.npy; errors in the sidecar.

    PYO=<python of the qmoe-ort environment>
    $PYO scripts/redesign/idea6/optionB/ort_int8_optb.py export --model mn10 --r R
    $PYO ... build --model mn10 --r R --meta CSV ; $PYO ... embed --model mn10 --r R --meta CSV
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "stage3c"))

import ort_int8 as O  # noqa: E402  Stage 3c's pipeline

SEEDS = [0, 1, 2]
KINDS = ["esc", "8", "4", "bitnet"]


class Meta:
    """folder and device columns of the universe CSV (the qmoe-ort env has no pandas)."""

    def __init__(self, path: Path):
        import csv

        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        self.folder = np.array([int(r["folder"]) for r in rows])
        self.device = np.array([r["device"] for r in rows])


def splits(r: Path, model: str, meta):
    """Per fold: the fold's rows (every device) and the union over seeds and kinds of the inner-validation rows."""
    fold_u = meta.folder
    out = {}
    for f in sorted(np.unique(fold_u)):
        iv = set()
        for s in SEEDS:
            for k in KINDS:
                sp = json.loads((r / "outputs" / f"{model}_tau_s{s}" / k / f"fold_{f}" / "split.json").read_text())
                iv |= set(sp["inner_val_idx"])
        out[int(f)] = (np.flatnonzero(fold_u == f), np.array(sorted(iv)))
    return out


def do_export(a):
    import torch

    from QWave import efficientat_mn as M
    from onnxruntime.quantization import quant_pre_process

    torch.set_num_threads(O.THREADS)
    md = a.r / "models"
    md.mkdir(parents=True, exist_ok=True)
    wf = a.weights / O.WEIGHT_FILES[a.model] if hasattr(O, "WEIGHT_FILES") else a.weights / {
        "mn04": "mn04_as_mAP_432.pt", "mn05": "mn05_as_mAP_443.pt", "mn10": "mn10_as_mAP_471.pt", "mn20": "mn20_as_mAP_478.pt"}[a.model]
    base = M.load_mn(a.model, wf)
    ex = O.exportable(base)
    fp32, pre = md / f"{a.model}_fp32.onnx", md / f"{a.model}_fp32_pre.onnx"
    torch.onnx.export(ex, (torch.zeros(1, 1, 128, 1001),), str(fp32), opset_version=17, dynamo=False,
                      input_names=["mel"], output_names=["embedding"],
                      dynamic_axes={"mel": {0: "batch", 3: "frames"}, "embedding": {0: "batch"}})
    quant_pre_process(str(fp32), str(pre))
    mel = np.load(a.r / "cache_mel" / "tau" / "clean.npy", mmap_mode="r")
    x = torch.from_numpy(mel[:8].astype(np.float32))[:, None]
    with torch.no_grad():
        ref, exv = base(x).numpy(), ex(x).numpy()
    o = O.run(O.session(pre), mel, np.arange(8))
    info = {"model": a.model, "weights": str(wf), "opset": 17, "exportable_vs_base_maxabs": float(np.abs(exv - ref).max()),
            "ort_fp32_vs_torch_maxabs": float(np.abs(o - ref).max()), "fp32_pre_nodes": O.node_types(pre), "torch": torch.__version__}
    (md / f"{a.model}_export.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1), flush=True)


def do_build(a):
    meta = Meta(a.meta)
    md = a.r / "models" / "tau"
    md.mkdir(parents=True, exist_ok=True)
    pre = a.r / "models" / f"{a.model}_fp32_pre.onnx"
    mel = np.load(a.r / "cache_mel" / "tau" / "clean.npy", mmap_mode="r")
    n_a = int((meta.device == "a").sum())
    sp = splits(a.r, a.model, meta)
    info = {"model": a.model, "method": a.method, "n_cal": O.N_CAL, "seed": O.CAL_SEED, "folds": {}}
    for f, (te, iv) in sp.items():
        out = md / f"{a.model}_int8_f{f}.onnx"
        m = np.ones(n_a, bool)  # device-A rows only
        m[te[te < n_a]] = False
        m[iv] = False
        el = np.flatnonzero(m)
        cal = np.sort(np.random.default_rng(O.CAL_SEED).choice(el, O.N_CAL, replace=False))
        secs = O.quantize(pre, out, mel, cal, a.method) if not out.exists() else None
        fi = {"n_test_rows_all_devices": int(len(te)), "n_iv_union": int(len(iv)), "n_eligible": int(len(el)),
              "calibration_rows": cal.tolist(), "seconds": secs}
        if f == min(sp):
            opt = md / f"{a.model}_int8_f{f}_optimized.onnx"
            O.session(out, O.THREADS, opt)
            fi["int8_model_nodes"] = O.node_types(out)
            fi["int8_optimized_nodes"] = O.node_types(opt)
        info["folds"][str(f)] = fi
        print(f"fold {f}: int8 built ({secs}s), {len(el)} eligible", flush=True)
    import onnxruntime as ort

    info["onnxruntime"] = ort.__version__
    (md / f"{a.model}_build.json").write_text(json.dumps(info, indent=1))


def do_embed(a):
    meta = Meta(a.meta)
    md = a.r / "models" / "tau"
    ed = a.r / "emb_int8" / "tau"
    ed.mkdir(parents=True, exist_ok=True)
    sp = splits(a.r, a.model, meta)
    w8 = json.loads((a.r / "emb_cnn" / "tau" / f"{a.model}_w8a8__clean.json").read_text())
    cf = a.r / "cache_mel" / "tau" / "clean.npy"
    side = json.loads(cf.with_suffix(".json").read_text())
    if side["names_crc32"] != w8["names_crc32"]:
        raise SystemExit("names checksum differs from the W8A8 sidecar")
    t0 = time.time()
    mel = np.load(cf, mmap_mode="r")
    n = len(mel)
    X = np.full((n, w8["emb_dim"]), np.nan, np.float32)
    for f, (te, _) in sp.items():
        pf = ed / f"{a.model}_int8_f{f}__clean.npy"
        if pf.exists():
            full = np.load(pf)
        else:
            full = O.run(O.session(md / f"{a.model}_int8_f{f}.onnx"), mel, np.arange(n))
            np.save(pf, full)
            (ed / f"{a.model}_int8_f{f}__clean.json").write_text(json.dumps(
                {"config": f"{a.model}_int8_f{f}", "model": a.model, "fold": f, "condition": "clean", "n_clips": n,
                 "emb_dim": w8["emb_dim"], "names_crc32": side["names_crc32"], "rows": "every universe row by fold f's model",
                 "cost_per_clip": w8["cost_per_clip"]}, indent=1))
        X[te] = full[te]
        print(f"  fold {f}: {time.time() - t0:.0f}s", flush=True)
    if not np.isfinite(X).all():
        raise SystemExit("rows left unfilled")
    np.save(ed / f"{a.model}_int8__clean.npy", X)
    pre = a.r / "models" / f"{a.model}_fp32_pre.onnx"
    p32 = ed / f"{a.model}_ortfp32__clean.npy"
    o32 = np.load(p32) if p32.exists() else O.run(O.session(pre), mel, np.arange(n))
    np.save(p32, o32)
    ref = np.load(a.r / "emb_cnn" / "tau" / f"{a.model}_fp32__clean.npy")
    w8e = np.load(a.r / "emb_cnn" / "tau" / f"{a.model}_w8a8__clean.npy")
    rel = lambda Z: float(np.mean(np.linalg.norm(Z - ref, axis=1) / np.linalg.norm(ref, axis=1)))  # noqa: E731
    isa = (meta.device == "a")
    meta_out = {"config": f"{a.model}_int8", "model": a.model, "backbone_config": "int8_ort", "condition": "clean", "n_clips": n,
                "emb_dim": w8["emb_dim"], "names_crc32": side["names_crc32"], "frames": side["frames"], "per_fold_models": True,
                "rows": "each row by the model of its own fold; <model>_int8_f<f>__clean.npy: every row by fold f's model",
                "cost_per_clip": w8["cost_per_clip"], "cost_note": "priced as the W8A8 model (plan)",
                "weights_sha256": w8["weights_sha256"], "seconds": time.time() - t0, "threads": O.THREADS,
                "rel_err_int8_vs_torch_fp32_mean": rel(X), "rel_err_ortfp32_vs_torch_fp32_mean": rel(o32),
                "rel_err_w8a8_vs_torch_fp32_mean": rel(w8e),
                "rel_err_int8_vs_torch_fp32_mean_device_a": float(np.mean((np.linalg.norm(X - ref, axis=1) / np.linalg.norm(ref, axis=1))[isa])),
                "rel_err_int8_vs_torch_fp32_mean_other_devices": float(np.mean((np.linalg.norm(X - ref, axis=1) / np.linalg.norm(ref, axis=1))[~isa]))}
    (ed / f"{a.model}_int8__clean.json").write_text(json.dumps(meta_out, indent=1))
    print(json.dumps({k: v for k, v in meta_out.items() if k.startswith("rel_err")}), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["export", "build", "embed"])
    ap.add_argument("--model", default="mn10")
    ap.add_argument("--r", type=Path, required=True)
    ap.add_argument("--meta", type=Path, default=Path.home() / "Documents/DATASETS/TAU2020Mobile/ast_768/tau2020mobile.csv")
    ap.add_argument("--weights", type=Path, default=Path.home() / "Documents/qmoe-weights/efficientat_release-v0.0.1")
    ap.add_argument("--method", default="Percentile", help="Stage 3c's label-free choice (models/mn10_calib_select.json)")
    a = ap.parse_args(argv)
    {"export": do_export, "build": do_build, "embed": do_embed}[a.what](a)


if __name__ == "__main__":
    main()
