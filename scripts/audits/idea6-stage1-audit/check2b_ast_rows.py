"""Check 2b: each quantized AST embedding row is nearest (cosine) to the FP32 AST embedding of the
same row and to the training CSV row of the same file name. Read only.
    python check2b_ast_rows.py OUT.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
H = Path.home() / "Documents"
import os; RD = Path(os.environ.get("RESULTS_DIR", str(H / "results")))  # results root (archive edit: path only)
S2 = RD / "2026-10-20_2359_redesign-idea5-stage2_code-3c18c38/emb"
CSV = {"esc50": H / "DATASETS/ESC-50/ast_768/esc-50.csv", "urban8k": H / "DATASETS/Urban8k/ast_768/urbansound8k.csv"}
def unit(x):
    x = x - x.mean(0)
    return x / np.linalg.norm(x, axis=1, keepdims=True)
res = {}
for ds, csv in CSV.items():
    df = pd.read_csv(csv)
    X = df[[str(i) for i in range(768)]].to_numpy(np.float32)
    ref = unit(X)
    for cfg in ["fp32", "w4g128", "w8g64", "w3g64", "w4"]:
        p = S2 / ds / f"{cfg}__clean.npy"
        if not p.exists():
            continue
        E = np.load(p).astype(np.float32)
        u = unit(E)
        nn = np.concatenate([np.argmax(u[i:i + 1024] @ ref.T, 1) for i in range(0, len(u), 1024)])
        res[f"{ds}/{cfg}"] = {"rows": len(E), "nearest_csv_row_is_same_row": float((nn == np.arange(len(E))).mean()),
                              "max_abs_diff_to_csv": float(np.abs(E - X).max()) if cfg == "fp32" else None}
        print(ds, cfg, res[f"{ds}/{cfg}"], flush=True)
Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
