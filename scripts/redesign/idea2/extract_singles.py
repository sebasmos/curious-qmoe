#!/usr/bin/env python3
"""Softmax of every protocol-v2 single model on its fold's inner validation
split and held-out test fold, plus per-sample bit-op cost of each model.
Read-only on the run. Copied from the routing-audit headroom extract.py
(itself built on scripts/analysis/cascade.py: load_frame, fold_rows,
load_model, layer_cost) without the MC-dropout passes; it also stores the
inner-validation row indices. Must run with the code that trained the
models on sys.path; the test F1 of every model is checked against its
metrics.json.

    python extract_singles.py <code checkout> <outputs>/<ds>_singles_s<seed> <out dir>
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from torch import nn

CODE, RUN, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(CODE))
torch.set_num_threads(2)
MODELS = ["bitnet", "1", "2", "4", "8", "16", "esc", "qesc"]
NON_FEATURE_COLS = ["folder", "name", "label", "category"]


def load_frame(cfg):
    esc = cfg.experiment.datasets.esc
    df = pd.read_csv(str(esc.csv))
    lc = esc.get("label_col", "class_id")
    if lc != "class_id":
        df["class_id"] = df[lc]
    return df.drop(columns=NON_FEATURE_COLS, errors="ignore")


def fold_rows(df_feat, cfg, split):
    from QWave.datasets import EmbeddingAdaptDataset
    test_idx, iv_idx = np.asarray(split["test_idx"]), np.asarray(split["inner_val_idx"])
    fit_idx = np.setdiff1d(np.arange(len(df_feat)), np.concatenate([test_idx, iv_idx]))
    norm = cfg.experiment.datasets.esc.normalization_type
    rows = {}
    with contextlib.redirect_stdout(io.StringIO()):
        fit = EmbeddingAdaptDataset(df_feat.iloc[fit_idx].reset_index(drop=True), normalization_type=norm)
        for name, idx in (("inner_val", iv_idx), ("test", test_idx)):
            d = EmbeddingAdaptDataset(df_feat.iloc[idx].reset_index(drop=True), normalization_type=norm,
                                      scaler=fit.get_scaler())
            rows[name] = (d.features, d.labels.numpy())
    return rows


def load_model(run_dir, fold, in_dim, n_cls):
    from scripts.benchmark import build_model, checkpoint_name, prepare_eval_model
    cfg = OmegaConf.load(run_dir / "config.yaml")
    cfg.experiment.device = "cpu"
    cfg.experiment.current_fold = fold
    kind = run_dir.name
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_model(kind, in_dim, n_cls, cfg)
        model.load_state_dict(torch.load(run_dir / f"fold_{fold}" / checkpoint_name(kind), map_location="cpu"))
        model = prepare_eval_model(kind, model, cfg)
    return model.eval()


def layer_cost(m, kind):
    from QWave.qmoe_layers import BitLinear
    if kind == "qesc":  # dynamic int8 PTQ: int8 weights and activations
        if type(m).__name__ == "Linear" and "quantized" in type(m).__module__:
            macs = m.in_features * m.out_features
            return np.array([macs, macs * 8 * 8], dtype=float)
        return None
    if isinstance(m, BitLinear):
        nb = m.num_bits
        w, a = (2, 2) if nb == "bitnet" else (nb, 8 if nb < 16 else 32)
    elif isinstance(m, nn.Linear):
        w, a = 32, 32
    else:
        return None
    macs = m.in_features * m.out_features
    return np.array([macs, macs * w * a], dtype=float)


def cost(module, kind):
    return sum((c for c in (layer_cost(m, kind) for m in module.modules()) if c is not None), np.zeros(2))


@torch.no_grad()
def softmax(model, X, bs=64):  # batches of 64 in order, as the benchmark's test loader
    return torch.cat([F.softmax(model(X[i:i + bs]), dim=1) for i in range(0, len(X), bs)]).numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg4 = OmegaConf.load(RUN / "4" / "config.yaml")
    df = load_frame(cfg4)
    in_dim, n_cls = df.shape[1] - 1, int(df["class_id"].nunique())
    folds = sorted(int(p.name.split("_")[1]) for p in (RUN / "esc").glob("fold_*"))
    checks, costs = [], {}
    for fold in folds:
        split = json.loads((RUN / "esc" / f"fold_{fold}" / "split.json").read_text())
        for m in MODELS:
            s = json.loads((RUN / m / f"fold_{fold}" / "split.json").read_text())
            assert s["test_idx"] == split["test_idx"] and s["inner_val_idx"] == split["inner_val_idx"], (m, fold)
        rows = fold_rows(df, cfg4, split)
        (X_iv, y_iv), (X_te, y_te) = rows["inner_val"], rows["test"]
        arrs = {"y_iv": y_iv, "y_te": y_te, "test_idx": np.asarray(split["test_idx"]),
                "iv_idx": np.asarray(split["inner_val_idx"])}
        for m in MODELS:
            model = load_model(RUN / m, fold, in_dim, n_cls)
            costs[m] = cost(model, m).tolist()
            arrs[f"p_iv_{m}"] = softmax(model, X_iv)
            arrs[f"p_te_{m}"] = softmax(model, X_te)
            met = json.loads((RUN / m / f"fold_{fold}" / "metrics.json").read_text())
            f1_te = f1_score(y_te, arrs[f"p_te_{m}"].argmax(1), average="macro", zero_division=0)
            checks.append({"fold": fold, "model": m, "f1_te": f1_te, "metrics_f1": met["f1_macro"],
                           "diff_te": abs(f1_te - met["f1_macro"]), "n_test": met["n_test"]})
        np.savez_compressed(OUT / f"fold_{fold}.npz", **arrs)
        print(fold, max(c["diff_te"] for c in checks if c["fold"] == fold), flush=True)
    (OUT / "checks.json").write_text(json.dumps({"run": str(RUN), "seed": int(cfg4.experiment.seed),
                                                 "checks": checks, "costs": costs}, indent=1))


if __name__ == "__main__":
    main()
