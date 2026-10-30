"""Check (b): the quantized graph's output path, the last Q/DQ scale, node types, and the per-fold output grid."""
import json, sys
import numpy as np, onnx
from onnx import numpy_helper
R = sys.argv[1]; out = sys.argv[2]
res = {}
for ds, m in (("esc50", "mn04"), ("esc50", "mn10"), ("esc50", "mn20"), ("urban8k", "mn10")):
    g = onnx.load(f"{R}/models/{ds}/{m}_int8_f1.onnx").graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    outname = g.output[0].name
    chain = []; t = outname
    for _ in range(8):
        n = prod.get(t)
        if n is None: break
        d = {"op": n.op_type, "name": n.name, "inputs": list(n.input)}
        if n.op_type in ("DequantizeLinear", "QuantizeLinear"):
            d["scale"] = float(inits[n.input[1]].ravel()[0]) if n.input[1] in inits else "?"
            d["zero_point"] = int(inits[n.input[2]].ravel()[0]) if len(n.input) > 2 and n.input[2] in inits else "?"
            d["dtype"] = str(inits[n.input[2]].dtype) if len(n.input) > 2 and n.input[2] in inits else "?"
        chain.append(d); t = n.input[0]
    r = {"output": outname, "chain_from_output": chain}
    # per-fold output grid
    folds = sorted(int(p.stem.split("_f")[1].split("__")[0]) for p in __import__("pathlib").Path(f"{R}/emb_int8/{ds}").glob(f"{m}_int8_f*__clean.npy"))
    r["per_fold_unique_values"] = {}
    for f in folds[:3]:
        x = np.load(f"{R}/emb_int8/{ds}/{m}_int8_f{f}__clean.npy")
        u = np.unique(x); gaps = np.diff(u); gaps = gaps[gaps > 1e-9]
        r["per_fold_unique_values"][f] = {"n_unique": int(len(u)), "step_median": float(np.median(gaps)), "min": float(u.min()), "max": float(u.max())}
    b = json.load(open(f"{R}/models/{ds}/{m}_build.json"))
    r["method"] = b["method"]; r["int8_optimized_nodes"] = b["folds"]["1"].get("int8_optimized_nodes"); r["int8_model_nodes"] = b["folds"]["1"].get("int8_model_nodes")
    # activation-quantization scales: all per-tensor activation DQ scales (scalar scale, uint8 zp)
    act_scales = [float(inits[n.input[1]]) for n in g.node if n.op_type == "DequantizeLinear" and inits.get(n.input[1]) is not None and inits[n.input[1]].ndim == 0 and len(n.input) > 2 and inits[n.input[2]].dtype == np.uint8]
    r["n_activation_dq"] = len(act_scales); r["act_scale_min_median_max"] = [float(np.min(act_scales)), float(np.median(act_scales)), float(np.max(act_scales))] if act_scales else None
    res[f"{ds}/{m}"] = r
    print("==", ds, m, json.dumps(r, indent=1), flush=True)
json.dump(res, open(out, "w"), indent=1)
