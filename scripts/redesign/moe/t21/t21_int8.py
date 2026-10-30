"""T21 real int8 MN10 on ONNX Runtime (ANALYSIS_PLAN.md C1), run in the qmoe-ort env (no torchaudio: the log-mel inputs come
from t21_extract_mn.py's mel cache). Uses stage 3c's ort_int8.py functions unchanged (exportable, quantize, session).

export  MN10 features -> ONNX (opset 17, dynamic batch and frames), quant_pre_process; ORT FP32 checked against the
        torch FP32 embeddings of 16 cached clean clips.
build   one int8 model per outer fold, method Percentile (stage 3c's choice, fixed), 128 clean clips drawn with numpy
        seed 0 from the fold's training rows that are in no seed's inner-validation split (split.json of the mn10 heads).
embed   per condition cache file: each test row by its fold's model -> mn10_int8__<cond>.npy (NaN elsewhere);
        on clean also every fold's model on every row -> mn10_int8_f<k>__clean.npy (inner-validation and refit rows).
Each clip runs at its own length (batch 1).

    $PYO t21_int8.py export|build|embed --r R --d d1
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/redesign/idea6/stage3c"))
import ort_int8 as O  # noqa: E402

WEIGHTS = Path.home() / "Documents/qmoe-weights/efficientat_release-v0.0.1/mn10_as_mAP_471.pt"
METHOD = "Percentile"
CAL_FRAMES = 1025  # frames of a 10.24 s clip; calibration clips are tiled (or cut) to this length
SEEDS = {"d1": [0, 1, 2, 3, 4], "d2": [0, 1, 2]}


class Mel:
    """Rows of one condition's mel cache (one or more part files)."""

    def __init__(self, files):
        self.parts = [np.load(f) for f in files]
        self.where = {}
        for pi, p in enumerate(self.parts):
            for j, r in enumerate(p["rows"]):
                self.where[int(r)] = (pi, j)
        crcs = {int(p["names_crc32"]) for p in self.parts}
        if len(crcs) != 1:
            raise SystemExit(f"names checksums differ across {files}")
        self.crc = crcs.pop()

    def get(self, r):
        pi, j = self.where[int(r)]
        p = self.parts[pi]
        o = p["offsets"]
        return np.asarray(p["mel"][:, o[j]:o[j + 1]], np.float32)[None, None]


def mel_files(r, d, cond):
    return sorted(glob.glob(str(r / "mel_cache" / d / f"{cond}__part*.npz")))


class Meta:
    """pool.csv read without pandas (the qmoe-ort env has none): names and outer folds."""

    def __init__(self, path):
        import csv

        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.names = [r["name"] for r in rows]
        self.folder = np.array([int(r["folder"]) for r in rows])

    def __len__(self):
        return len(self.names)


def pool_meta(r, d):
    return Meta(r / "csv" / d / "pool.csv")


def iv_union(r, d, fold):
    rows, te = set(), None
    for s in SEEDS[d]:
        sp = json.loads((r / "outputs" / f"{d}_pool_mn10_s{s}" / "esc" / f"fold_{fold}" / "split.json").read_text())
        rows |= set(sp["inner_val_idx"])
        t = np.array(sp["test_idx"])
        if te is not None and not np.array_equal(te, t):
            raise SystemExit("test rows differ between seeds")
        te = t
    return te, np.array(sorted(rows))


def do_export(a):
    import torch

    from QWave import efficientat_mn as M
    from onnxruntime.quantization import quant_pre_process

    md = a.r / "models"
    md.mkdir(parents=True, exist_ok=True)
    base = M.load_mn("mn10", WEIGHTS)
    ex = O.exportable(base)
    fp32, pre = md / "mn10_fp32.onnx", md / "mn10_fp32_pre.onnx"
    torch.onnx.export(ex, (torch.zeros(1, 1, 128, 501),), str(fp32), opset_version=17, dynamo=False,
                      input_names=["mel"], output_names=["embedding"],
                      dynamic_axes={"mel": {0: "batch", 3: "frames"}, "embedding": {0: "batch"}})
    quant_pre_process(str(fp32), str(pre))
    mel = Mel(mel_files(a.r, a.d, "clean"))
    ref = np.load(a.r / "emb" / a.d / "mn_pool" / "mn10_fp32__clean.npy")
    rows = sorted(mel.where)[:16]
    s = O.session(pre)
    o = np.concatenate([s.run(None, {"mel": mel.get(r_)})[0] for r_ in rows])
    with torch.no_grad():
        t = np.concatenate([ex(torch.from_numpy(mel.get(r_))).numpy() for r_ in rows])
    info = {"opset": 17, "ort_fp32_vs_cached_torch_fp32_maxrel": float(np.max(np.linalg.norm(o - ref[rows], axis=1) / np.linalg.norm(ref[rows], axis=1))),
            "exportable_vs_cached_maxrel": float(np.max(np.linalg.norm(t - ref[rows], axis=1) / np.linalg.norm(ref[rows], axis=1))),
            "fp32_pre_nodes": O.node_types(pre), "torch": torch.__version__}
    (md / ("mn10_export.json" if a.d == "d1" else f"mn10_export_{a.d}.json")).write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1), flush=True)


def do_build(a):
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

    md = a.r / "models" / a.d
    md.mkdir(parents=True, exist_ok=True)
    pre = a.r / "models" / "mn10_fp32_pre.onnx"
    mel = Mel(mel_files(a.r, a.d, "clean"))
    meta = pool_meta(a.r, a.d)
    info = {"method": METHOD, "n_cal": O.N_CAL, "seed": O.CAL_SEED, "folds": {}}
    for f in sorted(set(meta.folder.tolist())):
        te, iv = iv_union(a.r, a.d, int(f))
        if not np.array_equal(np.sort(te), np.flatnonzero(meta.folder == f)):
            raise SystemExit(f"fold {f}: split.json test rows differ from the CSV folder column")
        el = O.eligible(len(meta), te, iv)
        cal = np.sort(np.random.default_rng(O.CAL_SEED).choice(el, O.N_CAL, replace=False))
        assert not (set(cal) & set(te)) and not (set(cal) & set(iv))
        out = md / f"mn10_int8_f{f}.onnx"

        class R(CalibrationDataReader):
            def __init__(self):
                self.i = 0

            def get_next(self):
                if self.i >= len(cal):
                    return None
                self.i += 1
                x = mel.get(cal[self.i - 1])  # ORT stacks calibration outputs: one length for all, by tiling
                return {"mel": np.ascontiguousarray(np.tile(x, (1, 1, 1, -(-CAL_FRAMES // x.shape[3])))[..., :CAL_FRAMES])}

            def rewind(self):
                self.i = 0

        t0 = time.time()
        if not out.exists():
            quantize_static(str(pre), str(out), R(), quant_format=QuantFormat.QDQ, per_channel=True,
                            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                            calibrate_method=getattr(CalibrationMethod, METHOD))
        info["folds"][str(int(f))] = {"n_test": int(len(te)), "n_iv_union": int(len(iv)), "n_eligible": int(len(el)),
                                      "calibration_rows": cal.tolist(), "seconds": time.time() - t0}
        print(f"fold {f}: built in {time.time() - t0:.0f}s", flush=True)
    opt = md / "mn10_int8_f1_optimized.onnx"
    O.session(md / "mn10_int8_f1.onnx", O.THREADS, opt)
    info["int8_optimized_nodes_f1"] = O.node_types(opt)
    import onnxruntime as ort

    info["onnxruntime"] = ort.__version__
    (md / "mn10_build.json").write_text(json.dumps(info, indent=1))


def do_embed(a):
    md = a.r / "models" / a.d
    ed = a.r / "emb" / a.d / "mn_int8"
    ed.mkdir(parents=True, exist_ok=True)
    meta = pool_meta(a.r, a.d)
    n = len(meta)
    crc = zlib.crc32("\n".join(Path(s).stem for s in meta.names).encode())
    folds = sorted(set(meta.folder.tolist()))
    sess = {f: O.session(md / f"mn10_int8_f{f}.onnx") for f in folds}
    fold_of = meta.folder
    conds = sorted({Path(p).name.split("__part")[0] for p in glob.glob(str(a.r / "mel_cache" / a.d / "*__part*.npz"))},
                   key=lambda c: (c == "clean", c))  # clean last: it holds its cache longest (every fold model)
    for cond in conds:
        if (ed / f"mn10_int8__{cond}.npy").exists():
            continue
        t0 = time.time()
        mel = Mel(mel_files(a.r, a.d, cond))
        if mel.crc != crc:
            raise SystemExit(f"{cond}: names checksum differs")
        rows = np.array(sorted(mel.where))
        X = np.full((n, 960), np.nan, np.float32)
        full = {f: np.full((n, 960), np.nan, np.float32) for f in folds} if cond == "clean" else None
        for r_ in rows:
            x = mel.get(r_)
            if cond == "clean":
                for f in folds:
                    full[f][r_] = sess[f].run(None, {"mel": x})[0][0]
                X[r_] = full[fold_of[r_]][r_]
            else:
                X[r_] = sess[fold_of[r_]].run(None, {"mel": x})[0][0]
        if not np.isfinite(X[rows]).all():
            raise SystemExit(f"{cond}: non-finite rows")
        np.save(ed / f"mn10_int8__{cond}.npy", X)
        side = {"config": "mn10_int8", "condition": cond, "n_clips": n, "rows": int(len(rows)), "names_crc32": crc,
                "method": METHOD, "seconds": time.time() - t0, "threads": O.THREADS, "per_fold_models": True}
        if cond == "clean":
            for f in folds:
                np.save(ed / f"mn10_int8_f{f}__clean.npy", full[f])
            side["every_fold_model_on_every_row"] = True
        (ed / f"mn10_int8__{cond}.json").write_text(json.dumps(side, indent=1))
        print(f"{cond}: {len(rows)} rows in {time.time() - t0:.0f}s", flush=True)
        if a.delete_cache:  # disk: this condition's mel cache is not needed again
            for f in mel_files(a.r, a.d, cond):
                Path(f).unlink()
            print(f"{cond}: mel cache deleted", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["export", "build", "embed"])
    ap.add_argument("--r", type=Path, required=True)
    ap.add_argument("--d", default="d1")
    ap.add_argument("--delete-cache", action="store_true", help="embed: delete each condition's mel cache once embedded")
    a = ap.parse_args(argv)
    {"export": do_export, "build": do_build, "embed": do_embed}[a.what](a)


if __name__ == "__main__":
    main()
