"""Check (d): corrected real-int8 mn10 on ESC-50. Same export (stage 3c's mn10_fp32_pre.onnx), same Percentile method,
the same 128 calibration rows per fold as stage 3c's build.json, two variants of what is quantized:
  v1_poolfp32   every op as stage 3c except the final GlobalAveragePool (and so the output), nodes_to_exclude
  v2_convonly   only Conv and Gemm (the fake-quant W8A8 rule: SE, hardswish, residuals and the pool in float)
  v0_stage3c    stage 3c's own fold models re-run here (a check that my mels reproduce its embeddings)
Embeds every row on clean with every fold's model (for the scaler refit and the inner validation) and fold f's test
rows on the nine field conditions. Waits for a condition's mel file if it is not there yet."""
import json, sys, time, os
from pathlib import Path
import numpy as np, onnx
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
import onnxruntime as ort
R = Path(sys.argv[1]); MEL = Path(sys.argv[2]); A = Path(sys.argv[3]); model = sys.argv[4]
FIELD = ["bg__10db", "bg__5db", "bg__0db", "pink__10db", "pink__5db", "pink__0db", "white__10db", "white__5db", "white__0db"]
pre = R / "models" / f"{model}_fp32_pre.onnx"
build = json.loads((R / "models/esc50" / f"{model}_build.json").read_text())
assert build["method"] == "Percentile"
md = A / "models_fixed"; md.mkdir(exist_ok=True); ed = A / "emb_int8_fixed"; ed.mkdir(exist_ok=True)
mel_clean = np.load(MEL / "clean.npy", mmap_mode="r")
n = len(mel_clean)

def sess(p):
    so = ort.SessionOptions(); so.intra_op_num_threads = 2; so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])

def run(s, mel, rows):
    return np.concatenate([s.run(None, {"mel": mel[rows[b:b + 32]].astype(np.float32)[:, None]})[0] for b in range(0, len(rows), 32)]).astype(np.float32)

class Reader(CalibrationDataReader):
    def __init__(self, rows): self.rows, self.i = rows, 0
    def get_next(self):
        if self.i >= len(self.rows): return None
        self.i += 1; r = self.rows[self.i - 1]
        return {"mel": mel_clean[r:r + 1].astype(np.float32)[:, None]}
    def rewind(self): self.i = 0

# 0. my mels reproduce stage 3c's ORT FP32 embeddings?
o32 = run(sess(pre), mel_clean, np.arange(64))
ref32 = np.load(R / "emb_int8/esc50" / f"{model}_ortfp32__clean.npy")[:64]
chk = {"ortfp32_my_mels_vs_stage3c_maxabs": float(np.abs(o32 - ref32).max())}
print(chk, flush=True)
# node names
g = onnx.load(str(pre)).graph
pool_nodes = [nd.name for nd in g.node if nd.op_type == "GlobalAveragePool" and nd.output[0] == "/GlobalAveragePool_output_0"]
assert len(pool_nodes) == 1, pool_nodes
variants = {"v1_poolfp32": dict(nodes_to_exclude=pool_nodes), "v2_convonly": dict(op_types_to_quantize=["Conv", "Gemm"])}
splits = {}
for f, fi in build["folds"].items():
    sp = json.loads((R.parent / "2026-10-22_1652_redesign-idea6-stage3b_code-07dc1db" / "outputs" / f"{model}_esc50_s0/esc" / f"fold_{f}/split.json").read_text())
    splits[int(f)] = (np.array(sp["test_idx"]), np.array(fi["calibration_rows"]))
    assert fi["n_test"] == len(sp["test_idx"])
info = {"check": chk, "pool_node": pool_nodes, "variants": {}}
for v, kw in variants.items():
    t0 = time.time(); paths = {}
    for f, (te, cal) in splits.items():
        p = md / f"{model}_{v}_f{f}.onnx"
        if not p.exists():
            quantize_static(str(pre), str(p), Reader(cal), quant_format=QuantFormat.QDQ, per_channel=True, activation_type=QuantType.QUInt8,
                            weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.Percentile, **kw)
        paths[f] = p
    so = ort.SessionOptions(); so.optimized_model_filepath = str(md / f"{model}_{v}_f1_optimized.onnx"); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort.InferenceSession(str(paths[1]), so, providers=["CPUExecutionProvider"])
    import collections
    nodes = dict(collections.Counter(nd.op_type for nd in onnx.load(str(md / f"{model}_{v}_f1_optimized.onnx")).graph.node))
    go = onnx.load(str(paths[1])).graph; prod = {o: nd for nd in go.node for o in nd.output}
    out_op = prod[go.output[0].name].op_type
    info["variants"][v] = {"build_seconds": round(time.time() - t0, 1), "optimized_nodes": nodes, "output_producer_op": out_op}
    print(v, info["variants"][v], flush=True)
# embed: v0 (stage 3c models), v1, v2
allv = {"v0_stage3c": {f: R / "models/esc50" / f"{model}_int8_f{f}.onnx" for f in splits}}
allv.update({v: {f: md / f"{model}_{v}_f{f}.onnx" for f in splits} for v in variants})
for v, paths in allv.items():
    S = {f: sess(p) for f, p in paths.items()}
    t0 = time.time()
    if not (ed / f"{model}_{v}__clean.npy").exists():
        X = np.full((n, ref32.shape[1]), np.nan, np.float32)
        for f, (te, _) in splits.items():
            full = run(S[f], mel_clean, np.arange(n)); np.save(ed / f"{model}_{v}_f{f}__clean.npy", full); X[te] = full[te]
        assert np.isfinite(X).all(); np.save(ed / f"{model}_{v}__clean.npy", X)
    for c in FIELD:
        if (ed / f"{model}_{v}__{c}.npy").exists(): continue
        while not (MEL / f"{c}.json").exists(): time.sleep(20)
        mel = np.load(MEL / f"{c}.npy", mmap_mode="r")
        X = np.full((n, ref32.shape[1]), np.nan, np.float32)
        for f, (te, _) in splits.items(): X[te] = run(S[f], mel, te)
        assert np.isfinite(X).all(); np.save(ed / f"{model}_{v}__{c}.npy", X)
    print(f"{v}: embedded in {time.time() - t0:.0f}s", flush=True)
if (R / "emb_int8/esc50" / f"{model}_int8__clean.npy").exists():
    a = np.load(ed / f"{model}_v0_stage3c__clean.npy"); b = np.load(R / "emb_int8/esc50" / f"{model}_int8__clean.npy")
    info["check"]["v0_rerun_vs_stage3c_int8_clean_maxabs"] = float(np.abs(a - b).max())
    for c in FIELD:
        a = np.load(ed / f"{model}_v0_stage3c__{c}.npy"); b = np.load(R / "emb_int8/esc50" / f"{model}_int8__{c}.npy")
        info["check"][f"v0_rerun_vs_stage3c_int8_{c}_maxabs"] = float(np.abs(a - b).max())
json.dump(info, open(A / "outputs" / f"d1_int8_fixed_build_{model}.json", "w"), indent=1); print(json.dumps(info["check"], indent=1))
