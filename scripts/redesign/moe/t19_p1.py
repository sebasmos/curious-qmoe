"""T19 step 19.5: preconditions P1 and P2 on fold 1 (ANALYSIS_PLAN_T19.md B6), from the fold-1 heads' clean inner-validation logits only.
Writes <results>/precondition/p1_p2.json and p1_table.csv."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t19_common import TASKS, results_dir  # noqa: E402

R = results_dir()
CFGS = ["w8a8", "n4a4", "n4a6", "w4a6", "w4a4"]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def margin(p):
    s = np.sort(p, 1)
    return s[:, -1] - s[:, -2]


def iv(task, cfg, seed, fold=1):
    z = np.load(R / f"scores/{task}/{cfg}/s{seed}/{cfg}__clean.npz")
    assert "test_logits" not in z.files, "P1 files must not hold test logits"
    return z["y"][z[f"iv_idx_{fold}"]], z[f"iv_logits_{fold}"].astype(np.float64), z[f"iv_idx_{fold}"]


rows, res = [], {}
for task in ("esc50", "urban"):
    f1 = {}
    for cfg in CFGS:
        v = []
        for s in TASKS[task]["seeds"]:
            y, z, _ = iv(task, cfg, s)
            v.append(100 * f1_score(y, z.argmax(1), average="macro"))
        f1[cfg] = float(np.mean(v))
        rows.append(dict(task=task, config=cfg, iv_f1_mean=f1[cfg], iv_f1_seeds=json.dumps(np.round(v, 4).tolist()),
                         gap_to_w8a8=f1["w8a8"] - f1[cfg] if "w8a8" in f1 else 0.0))
    res[task] = f1
p1_n4a4 = all(res[t]["w8a8"] - res[t]["n4a4"] <= 8.0 for t in res)
p1_n4a6 = all(res[t]["w8a8"] - res[t]["n4a6"] <= 8.0 for t in res)
cheap = "n4a4" if p1_n4a4 else ("n4a6" if p1_n4a6 else None)
out = {"p1": {"iv_f1": res, "n4a4_within_8": p1_n4a4, "n4a6_within_8": p1_n4a6, "cheap_tier": cheap,
              "a8_reproduction": json.loads((R / "logs/repro_a8.json").read_text())}}
if cheap is not None:
    p2 = {}
    for task in ("esc50", "urban"):
        au = []
        for s in TASKS[task]["seeds"]:
            y, zc, ic = iv(task, cheap, s)
            y8, z8, i8 = iv(task, "w8a8", s)
            assert np.array_equal(ic, i8)
            target = (zc.argmax(1) != y) & (z8.argmax(1) == y)
            au.append(float(roc_auc_score(target, -margin(softmax(zc)))) if 0 < target.sum() < len(y) else float("nan"))
        p2[task] = {"auroc_seeds": au, "auroc_mean": float(np.nanmean(au))}
    out["p2"] = {**p2, "pass": all(p2[t]["auroc_mean"] >= 0.70 for t in p2)}
out["stop"] = cheap is None or not out.get("p2", {}).get("pass", False)
(R / "precondition").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(R / "precondition/p1_table.csv", index=False)
(R / "precondition/p1_p2.json").write_text(json.dumps(out, indent=1))
print(pd.DataFrame(rows).to_string(index=False))
print(json.dumps({k: v for k, v in out.items() if k != "p1"} | {"cheap": cheap}, indent=1))
