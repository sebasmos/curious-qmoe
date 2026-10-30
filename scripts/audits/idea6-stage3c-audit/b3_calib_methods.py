"""Check (b)/(c): on mn04, build int8 models with MinMax, Entropy, Percentile on the stage 3c fold-1 calibration rows
and compare their activation scales (are MinMax and Entropy the same model?) and the embedding error on the fold-1
selection rows; also Entropy with 2048 histogram bins (a real entropy search) and per-tensor weights."""
import json, sys, time
from pathlib import Path
import numpy as np, onnx
from onnx import numpy_helper
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
R = Path(sys.argv[1]); mel_dir = Path(sys.argv[2]); out = Path(sys.argv[3]); model = sys.argv[4] if len(sys.argv) > 4 else "mn04"
md = out / "models_calib"; md.mkdir(parents=True, exist_ok=True)
pre = R / "models" / f"{model}_fp32_pre.onnx"
sel = json.loads((R / "models" / f"{model}_calib_select.json").read_text())
cal, selrows = np.array(sel["calibration_rows"]), np.array(sel["selection_rows"])
mel = np.load(mel_dir / "clean.npy", mmap_mode="r")

class Reader(CalibrationDataReader):
    def __init__(self, rows): self.rows, self.i = rows, 0
    def get_next(self):
        if self.i >= len(self.rows): return None
        self.i += 1; r = self.rows[self.i - 1]
        return {"mel": mel[r:r + 1].astype(np.float32)[:, None]}
    def rewind(self): self.i = 0

def sess(p):
    so = ort.SessionOptions(); so.intra_op_num_threads = 2; so.inter_op_num_threads = 1
    return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])

def run(s, rows):
    return np.concatenate([s.run(None, {"mel": mel[rows[b:b + 32]].astype(np.float32)[:, None]})[0] for b in range(0, len(rows), 32)])

def scales(p):
    g = onnx.load(str(p)).graph
    return {i.name: numpy_helper.to_array(i).ravel() for i in g.initializer if i.name.endswith("_scale")}

ref = run(sess(pre), selrows)
res, S = {}, {}
variants = {"MinMax": dict(calibrate_method=CalibrationMethod.MinMax),
            "Entropy": dict(calibrate_method=CalibrationMethod.Entropy),
            "Percentile": dict(calibrate_method=CalibrationMethod.Percentile),
            "Entropy_2048bins": dict(calibrate_method=CalibrationMethod.Entropy, extra_options={"CalibNumBins": 2048}),
            "Percentile_pertensorW": dict(calibrate_method=CalibrationMethod.Percentile, per_channel=False)}
for name, kw in variants.items():
    p = md / f"{model}_{name}.onnx"; t0 = time.time()
    kw = {"quant_format": QuantFormat.QDQ, "per_channel": True, "activation_type": QuantType.QUInt8, "weight_type": QuantType.QInt8, **kw}
    quantize_static(str(pre), str(p), Reader(cal), **kw)
    o = run(sess(p), selrows)
    S[name] = scales(p)
    res[name] = {"rel_err_mean": float(np.mean(np.linalg.norm(o - ref, axis=1) / np.linalg.norm(ref, axis=1))), "seconds": round(time.time() - t0, 1),
                 "out_scale": float(S[name].get("/GlobalAveragePool_output_0_scale", [np.nan])[0])}
    print(name, res[name], flush=True)
def same(a, b):
    ka = set(S[a]) & set(S[b]); return {"n_common": len(ka), "n_equal": int(sum(np.array_equal(S[a][k], S[b][k]) for k in ka)),
                                       "max_rel_diff": float(max(np.abs(S[a][k] - S[b][k]).max() / (np.abs(S[a][k]).max() + 1e-30) for k in ka))}
res["MinMax_vs_Entropy_scales"] = same("MinMax", "Entropy"); res["MinMax_vs_Percentile_scales"] = same("MinMax", "Percentile")
res["Entropy_vs_Entropy2048_scales"] = same("Entropy", "Entropy_2048bins")
res["stage3c_select_errors"] = sel["errors"]
json.dump(res, open(out / "outputs" / f"b3_calib_methods_{model}.json", "w"), indent=1); print(json.dumps(res, indent=1))
