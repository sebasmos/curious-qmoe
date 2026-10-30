"""Redesign idea 1, stage 2: per-fold logits (inner val and test), bit-ops and weight
bytes of every single model in one protocol 3 <ds>_singles_s<seed> run.

Read-only on the run. Checkpoints are loaded with the code that trained them
(--code, commit bc4ff1c, put first on sys.path) through extract.py's helpers,
the way scripts/benchmark.py builds and scores them; test logits are computed in
the run's batch size, in order. Bit-ops per clip and weight bytes come from
QWave.costs (cost, weight_bytes). Every fold's test macro F1 must reproduce
metrics.json exactly (assert); inner-val F1 is checked and recorded. A seed is
extracted only when every model in the run has written summary.json, and the
1, 2 and 8-bit heads must be among them.

usage: python stage2_extract.py --run .../outputs/esc_singles_s0 --out DIR [--code DIR]
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

from extract import fold_rows, load_frame, load_model, logits

ALL_MODELS = ["bitnet", "1", "2", "4", "8", "16", "esc", "qesc"]
REQUIRED = ["1", "2", "8"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--code", type=Path, default=Path("/Users/cajas.sebastian/Desktop/repositories/curious-qmoe-run-v3-bc4ff1c"))
    a = ap.parse_args()
    sys.path.insert(0, str(a.code))
    from QWave.costs import cost, weight_bytes
    torch.set_num_threads(2)

    models = [m for m in ALL_MODELS if (a.run / m / "config.yaml").exists()]
    missing = [m for m in REQUIRED if m not in models]
    unfinished = [m for m in models if not (a.run / m / "summary.json").exists()]
    if missing or unfinished:
        sys.exit(f"{a.run}: missing {missing}, unfinished {unfinished}")
    a.out.mkdir(parents=True, exist_ok=True)

    cfg8 = OmegaConf.load(a.run / "8" / "config.yaml")
    seed, bs = int(cfg8.experiment.seed), int(cfg8.experiment.train.batch_size)
    df = load_frame(cfg8)
    in_dim, n_cls = df.shape[1] - 1, int(df["class_id"].nunique())
    folds = sorted(f["fold"] for f in json.loads((a.run / "8" / "summary.json").read_text())["folds"])
    checks, costs, wbytes = [], {}, {}
    for fold in folds:
        split = json.loads((a.run / "8" / f"fold_{fold}" / "split.json").read_text())
        assert split.get("inner_split", {}).get("groups") == "recording", (a.run, fold, split.get("inner_split"))
        for m in models:
            s = json.loads((a.run / m / f"fold_{fold}" / "split.json").read_text())
            assert s["test_idx"] == split["test_idx"] and s["inner_val_idx"] == split["inner_val_idx"], (m, fold)
        rows = fold_rows(df, cfg8, split)
        (X_iv, y_iv), (X_te, y_te) = rows["inner_val"], rows["test"]
        arrs = {"y_iv": y_iv, "y_te": y_te, "test_idx": np.asarray(split["test_idx"]),
                "iv_idx": np.asarray(split["inner_val_idx"])}
        for m in models:
            model = load_model(a.run / m, fold, in_dim, n_cls)
            costs[m] = cost(model).tolist()
            wbytes[m] = float(weight_bytes(model))
            arrs[f"l_iv_{m}"] = logits(model, X_iv, bs)
            arrs[f"l_te_{m}"] = logits(model, X_te, bs)
            met = json.loads((a.run / m / f"fold_{fold}" / "metrics.json").read_text())
            f1_te = f1_score(y_te, arrs[f"l_te_{m}"].argmax(1), average="macro", zero_division=0)
            f1_iv = f1_score(y_iv, arrs[f"l_iv_{m}"].argmax(1), average="macro", zero_division=0)
            checks.append({"fold": fold, "model": m, "f1_te": f1_te, "metrics_f1": met["f1_macro"],
                           "diff_te": abs(f1_te - met["f1_macro"]), "f1_iv": f1_iv,
                           "diff_iv": abs(f1_iv - met["inner_val_f1_macro"]),
                           "metrics_size_bytes": met["model_size_bytes"], "weight_bytes": wbytes[m],
                           "n_test": met["n_test"], "protocol_version": met.get("protocol_version")})
            assert checks[-1]["diff_te"] < 1e-9, checks[-1]
            assert met["n_test"] == len(y_te) and met["n_inner_val"] == len(y_iv), (m, fold)
        np.savez_compressed(a.out / f"fold_{fold}.npz", **arrs)
        fc = [c for c in checks if c["fold"] == fold]
        print(fold, "max |F1 - metrics.json| test", max(c["diff_te"] for c in fc),
              "inner val", max(c["diff_iv"] for c in fc), flush=True)
    (a.out / "checks.json").write_text(json.dumps(
        {"run": str(a.run), "code": str(a.code), "seed": seed, "folds": folds, "n_classes": n_cls,
         "models": models, "batch_size": bs, "checks": checks, "costs": costs, "weight_bytes": wbytes}, indent=1))


if __name__ == "__main__":
    main()
