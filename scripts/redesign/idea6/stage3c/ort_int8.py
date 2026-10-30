"""Real int8 EfficientAT MN on ONNX Runtime for idea 6 Stage 3c (P4), in the conda env qmoe-ort (onnxruntime, onnx,
torch and torchvision for the export; no torchaudio, so the log-mel inputs come from ort_mels.py's float16 cache).

export  the feature extractor of QWave.efficientat_mn.load_mn(MODEL) with hardswish rewritten as
        x * clip(x + 3, 0, 6) / 6 and the squeeze-excitation mean as a global average pool (both exact in float),
        opset 17, dynamic batch and frame axes -> models/<model>_fp32.onnx, quant_pre_process -> _pre.onnx; checked
        against torch on eight clean clips.
select  calibration method, label-free, on ESC-50 fold 1: MinMax, Percentile, Entropy on 128 calibration clips;
        the lowest mean relative embedding error against ORT FP32 on 64 other eligible training clips wins
        -> models/calib_select.json.
build   one int8 model per outer fold: quantize_static in QDQ format (int8 symmetric per-channel weights, uint8
        per-tensor activations), calibrated on 128 clean clips drawn with numpy seed 0 from the fold's training
        clips that are in no (task, seed) inner-validation split of the Stage 3b runs -> models/<ds>/<model>_int8_f<f>.onnx,
        the optimized graph's node types and the calibration rows in models/<ds>/<model>_build.json.
embed   per condition: fold f's model embeds fold f's test rows -> emb_int8/<ds>/<model>_int8__<cond>.npy [N, D];
        on clean every fold's model embeds every row -> <model>_int8_f<f>__clean.npy; ORT FP32 on clean -> <model>_ortfp32__clean.npy.
        Sidecars carry the names checksum and the W8A8 sidecar's costs (the plan prices the int8 model as W8A8).

    PYO=<python of the qmoe-ort environment>
    $PYO scripts/redesign/idea6/stage3c/ort_int8.py export --model mn10 --r R --s3b S3B
    $PYO ... select --model mn10 --r R --s3b S3B
    $PYO ... build --model mn10 --ds esc50 --r R --s3b S3B ; ... embed --model mn10 --ds esc50 --r R --s3b S3B
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

DS = {"esc50": dict(tasks=["esc50", "esc5", "esc5off"], seeds=[0, 1, 2, 3, 4], folds=list(range(1, 6))),
      "urban8k": dict(tasks=["urban"], seeds=[0, 1, 2], folds=list(range(1, 11)))}
N_CAL, N_SEL, CAL_SEED = 128, 64, 0
THREADS = 2
BATCH = 32


def session(path, threads=THREADS, optimized_out=None):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if optimized_out:
        so.optimized_model_filepath = str(optimized_out)
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def node_types(path):
    import onnx

    m = onnx.load(str(path))
    return dict(sorted(collections.Counter(f"{n.domain or 'ai.onnx'}:{n.op_type}" for n in m.graph.node).items()))


def run(sess, mel, rows, batch=BATCH):
    out = []
    for b in range(0, len(rows), batch):
        x = mel[rows[b:b + batch]].astype(np.float32)[:, None]
        out.append(sess.run(None, {"mel": x})[0])
    return np.concatenate(out).astype(np.float32)


def splits(s3b: Path, model: str, ds: str):
    """Per fold: test rows (seed 0 of the first task; equal across runs) and the union of every (task, seed)
    inner-validation split of the Stage 3b runs of this model."""
    d = DS[ds]
    out = {}
    for f in d["folds"]:
        te, iv = None, set()
        for t in d["tasks"]:
            for s in d["seeds"]:
                sp = json.loads((s3b / "outputs" / f"{model}_{t}_s{s}" / "esc" / f"fold_{f}" / "split.json").read_text())
                if te is None:
                    te = np.array(sp["test_idx"])
                elif not np.array_equal(te, np.array(sp["test_idx"])):
                    raise SystemExit(f"{model} {t} s{s} fold {f}: test rows differ")
                iv |= set(sp["inner_val_idx"])
        out[f] = (te, np.array(sorted(iv)))
    return out


def eligible(n, te, iv):
    m = np.ones(n, bool)
    m[te] = False
    m[iv] = False
    return np.flatnonzero(m)


def exportable(base):
    import torch
    import torch.nn.functional as F
    from torch import nn

    from QWave import efficientat_mn as M

    class HS(nn.Module):
        def forward(self, x):
            return x * torch.clamp(x + 3.0, 0.0, 6.0) * (1.0 / 6.0)

    class SE(nn.Module):
        def __init__(self, se):
            super().__init__()
            self.fc1, self.fc2 = se.fc1, se.fc2

        def forward(self, x):
            s = F.adaptive_avg_pool2d(x, 1).flatten(1)
            s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
            return x * s[:, :, None, None]

    m = copy.deepcopy(base).eval()
    for mod in list(m.features.modules()):
        for cn, ch in list(mod.named_children()):
            if isinstance(ch, nn.Hardswish):
                setattr(mod, cn, HS())
            elif isinstance(ch, M.SqueezeExcitation):
                setattr(mod, cn, SE(ch))

    class Wrap(nn.Module):
        def __init__(self, f):
            super().__init__()
            self.f = f

        def forward(self, x):
            return F.adaptive_avg_pool2d(self.f(x), 1).flatten(1)

    return Wrap(m.features).eval()


def do_export(a):
    import torch

    from QWave import efficientat_mn as M
    from onnxruntime.quantization import quant_pre_process

    torch.set_num_threads(THREADS)
    md = a.r / "models"
    md.mkdir(parents=True, exist_ok=True)
    wf = a.weights / {"mn04": "mn04_as_mAP_432.pt", "mn05": "mn05_as_mAP_443.pt", "mn10": "mn10_as_mAP_471.pt",
                      "mn20": "mn20_as_mAP_478.pt"}[a.model]
    base = M.load_mn(a.model, wf)
    ex = exportable(base)
    fp32, pre = md / f"{a.model}_fp32.onnx", md / f"{a.model}_fp32_pre.onnx"
    torch.onnx.export(ex, (torch.zeros(1, 1, 128, 501),), str(fp32), opset_version=17, dynamo=False,
                      input_names=["mel"], output_names=["embedding"],
                      dynamic_axes={"mel": {0: "batch", 3: "frames"}, "embedding": {0: "batch"}})
    quant_pre_process(str(fp32), str(pre))
    mel = np.load(a.r / "cache_mel" / "esc50" / "clean.npy", mmap_mode="r")
    x = torch.from_numpy(mel[:8].astype(np.float32))[:, None]
    with torch.no_grad():
        ref, exv = base(x).numpy(), ex(x).numpy()
    o = run(session(pre), mel, np.arange(8))
    info = {"model": a.model, "weights": str(wf), "opset": 17, "exportable_vs_base_maxabs": float(np.abs(exv - ref).max()),
            "ort_fp32_vs_torch_maxabs": float(np.abs(o - ref).max()), "fp32_pre_nodes": node_types(pre),
            "torch": torch.__version__}
    (md / f"{a.model}_export.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1), flush=True)


class Reader:
    def __init__(self, mel, rows):
        self.mel, self.rows, self.i = mel, rows, 0

    def get_next(self):
        if self.i >= len(self.rows):
            return None
        self.i += 1
        return {"mel": self.mel[self.rows[self.i - 1]:self.rows[self.i - 1] + 1].astype(np.float32)[:, None]}

    def rewind(self):
        self.i = 0


def quantize(pre, out, mel, rows, method):
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

    class R(Reader, CalibrationDataReader):
        pass

    t0 = time.time()
    quantize_static(str(pre), str(out), R(mel, rows), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=getattr(CalibrationMethod, method))
    return time.time() - t0


def do_select(a):
    md = a.r / "models"
    pre = md / f"{a.model}_fp32_pre.onnx"
    mel = np.load(a.r / "cache_mel" / "esc50" / "clean.npy", mmap_mode="r")
    te, iv = splits(a.s3b, a.model, "esc50")[1]
    el = eligible(len(mel), te, iv)
    rng = np.random.default_rng(CAL_SEED)
    pick = rng.choice(el, N_CAL + N_SEL, replace=False)
    cal, sel = np.sort(pick[:N_CAL]), np.sort(pick[N_CAL:])
    ref = run(session(pre), mel, sel)
    res = {}
    for meth in ("MinMax", "Percentile", "Entropy"):
        p = md / f"_select_{meth}.onnx"
        secs = quantize(pre, p, mel, cal, meth)
        o = run(session(p), mel, sel)
        res[meth] = {"rel_err_mean": float(np.mean(np.linalg.norm(o - ref, axis=1) / np.linalg.norm(ref, axis=1))),
                     "seconds": round(secs, 1)}
        p.unlink()
        print(meth, res[meth], flush=True)
    best = min(res, key=lambda k: res[k]["rel_err_mean"])
    info = {"model": a.model, "fold": 1, "n_eligible": int(len(el)), "calibration_rows": cal.tolist(),
            "selection_rows": sel.tolist(), "errors": res, "chosen": best}
    (md / f"{a.model}_calib_select.json").write_text(json.dumps(info, indent=1))
    print("chosen", best, flush=True)


def do_build(a):
    md = a.r / "models" / a.ds
    md.mkdir(parents=True, exist_ok=True)
    pre = a.r / "models" / f"{a.model}_fp32_pre.onnx"
    meth = json.loads((a.r / "models" / f"{a.model}_calib_select.json").read_text())["chosen"]
    mel = np.load(a.r / "cache_mel" / a.ds / "clean.npy", mmap_mode="r")
    sp = splits(a.s3b, a.model, a.ds)
    info = {"model": a.model, "ds": a.ds, "method": meth, "n_cal": N_CAL, "seed": CAL_SEED, "folds": {}}
    for f, (te, iv) in sp.items():
        out = md / f"{a.model}_int8_f{f}.onnx"
        el = eligible(len(mel), te, iv)
        cal = np.sort(np.random.default_rng(CAL_SEED).choice(el, N_CAL, replace=False))
        secs = quantize(pre, out, mel, cal, meth) if not out.exists() else None
        fi = {"n_test": int(len(te)), "n_iv_union": int(len(iv)), "n_eligible": int(len(el)),
              "calibration_rows": cal.tolist(), "seconds": secs}
        if f == min(sp):
            opt = md / f"{a.model}_int8_f{f}_optimized.onnx"
            session(out, THREADS, opt)
            fi["int8_model_nodes"] = node_types(out)
            fi["int8_optimized_nodes"] = node_types(opt)
        info["folds"][str(f)] = fi
        print(f"{a.ds} fold {f}: int8 built ({secs}s), {len(el)} eligible", flush=True)
    import onnxruntime as ort

    info["onnxruntime"] = ort.__version__
    (md / f"{a.model}_build.json").write_text(json.dumps(info, indent=1))


def do_embed(a):
    md = a.r / "models" / a.ds
    ed = a.r / "emb_int8" / a.ds
    ed.mkdir(parents=True, exist_ok=True)
    sp = splits(a.s3b, a.model, a.ds)
    sess = {f: session(md / f"{a.model}_int8_f{f}.onnx") for f in sp}
    w8 = json.loads((a.s3b / "emb_cnn" / a.ds / f"{a.model}_w8a8__clean.json").read_text())
    for cond in a.conditions:
        cf = a.r / "cache_mel" / a.ds / f"{cond}.npy"
        side = json.loads(cf.with_suffix(".json").read_text())
        if side["names_crc32"] != w8["names_crc32"]:
            raise SystemExit(f"{cond}: names checksum differs from the W8A8 sidecar")
        done = (ed / f"{a.model}_int8__{cond}.npy").exists() and (cond != "clean" or
                all((ed / f"{a.model}_int8_f{f}__clean.npy").exists() for f in sp))
        if done:
            continue
        t0 = time.time()
        mel = np.load(cf, mmap_mode="r")
        n = len(mel)
        X = np.full((n, w8["emb_dim"]), np.nan, np.float32)
        for f, (te, _) in sp.items():
            if cond == "clean":
                full = run(sess[f], mel, np.arange(n))
                np.save(ed / f"{a.model}_int8_f{f}__clean.npy", full)
                (ed / f"{a.model}_int8_f{f}__clean.json").write_text(json.dumps(
                    {"config": f"{a.model}_int8_f{f}", "model": a.model, "fold": f, "condition": "clean", "n_clips": n,
                     "emb_dim": w8["emb_dim"], "names_crc32": side["names_crc32"], "rows": "every row by fold f's model",
                     "cost_per_clip": w8["cost_per_clip"]}, indent=1))
                X[te] = full[te]
            else:
                X[te] = run(sess[f], mel, te)
            print(f"  {cond} fold {f}: {time.time() - t0:.0f}s", flush=True)
        if not np.isfinite(X).all():
            raise SystemExit(f"{cond}: rows left unfilled")
        np.save(ed / f"{a.model}_int8__{cond}.npy", X)
        meta = {"config": f"{a.model}_int8", "model": a.model, "backbone_config": "int8_ort", "condition": cond,
                "n_clips": n, "emb_dim": w8["emb_dim"], "names_crc32": side["names_crc32"],
                "pad_to_samples": side["pad_to_samples"], "frames": side["frames"], "per_fold_models": True,
                "rows": "test rows by their fold's model" + ("; <model>_int8_f<f>__clean.npy: every row by fold f's model"
                                                            if cond == "clean" else ""),
                "cost_per_clip": w8["cost_per_clip"], "cost_note": "priced as the W8A8 model (plan)",
                "weights_sha256": w8["weights_sha256"], "seconds": time.time() - t0, "threads": THREADS}
        if cond == "clean":
            pre = a.r / "models" / f"{a.model}_fp32_pre.onnx"
            o32 = run(session(pre), mel, np.arange(n))
            np.save(ed / f"{a.model}_ortfp32__clean.npy", o32)
            ref = np.load(a.s3b / "emb_cnn" / a.ds / f"{a.model}_fp32__clean.npy")
            err = np.linalg.norm(X - ref, axis=1) / np.linalg.norm(ref, axis=1)
            meta["rel_err_int8_vs_torch_fp32_mean"] = float(err.mean())
            meta["rel_err_ortfp32_vs_torch_fp32_mean"] = float(np.mean(np.linalg.norm(o32 - ref, axis=1) / np.linalg.norm(ref, axis=1)))
            w8e = np.load(a.s3b / "emb_cnn" / a.ds / f"{a.model}_w8a8__clean.npy")
            meta["rel_err_w8a8_vs_torch_fp32_mean"] = float(np.mean(np.linalg.norm(w8e - ref, axis=1) / np.linalg.norm(ref, axis=1)))
        (ed / f"{a.model}_int8__{cond}.json").write_text(json.dumps(meta, indent=1))
        print(f"{cond}: done in {time.time() - t0:.0f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["export", "select", "build", "embed"])
    ap.add_argument("--model", default="mn10")
    ap.add_argument("--ds", default="esc50", choices=list(DS))
    ap.add_argument("--r", type=Path, required=True)
    ap.add_argument("--s3b", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=Path.home() / "Documents/qmoe-weights/efficientat_release-v0.0.1")
    ap.add_argument("--conditions", nargs="*", default=["clean", "bg__10db", "bg__5db", "bg__0db", "pink__10db", "pink__5db",
                                                        "pink__0db", "white__10db", "white__5db", "white__0db", "bg__20db",
                                                        "pink__20db", "white__20db"])
    a = ap.parse_args(argv)
    {"export": do_export, "select": do_select, "build": do_build, "embed": do_embed}[a.what](a)


if __name__ == "__main__":
    main()
