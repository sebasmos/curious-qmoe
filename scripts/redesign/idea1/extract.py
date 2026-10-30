"""Redesign idea 1, stage 1: per-fold logits (inner val and test) of every single
model of one <ds>_singles_s<seed> run, plus MC-dropout passes of the cheap heads.

Read-only on the run. Checkpoints are loaded with the code that trained them
(--code, commit 5db7e75), the same way scripts/benchmark.py builds and scores
them; every fold's test and inner-val macro F1 is checked against metrics.json.
Only folds where all eight models finished are extracted.

usage: python extract.py --run .../outputs/urban_singles_s0 --out DIR [--code DIR] [--mc-passes 10]
"""
import argparse, contextlib, io, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from torch import nn

MODELS = ["bitnet", "1", "2", "4", "8", "16", "esc", "qesc"]
CHEAP = ["bitnet", "1", "2", "4"]
NON_FEATURE_COLS = ["folder", "name", "label", "category"]


def load_frame(cfg):
    esc = cfg.experiment.datasets.esc
    df = pd.read_csv(str(esc.csv))
    lc = esc.get("label_col", "class_id")
    if lc != "class_id":
        df["class_id"] = df[lc]
    return df.drop(columns=NON_FEATURE_COLS, errors="ignore")


def fold_rows(df_feat, cfg, split):
    """Inner-val and test features standardized with the fit split's scaler, as in training."""
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


def load_model(model_dir, fold, in_dim, n_cls):
    from scripts.benchmark import build_model, checkpoint_name, prepare_eval_model
    cfg = OmegaConf.load(model_dir / "config.yaml")
    cfg.experiment.device = "cpu"
    cfg.experiment.current_fold = fold
    kind = model_dir.name
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_model(kind, in_dim, n_cls, cfg)
        model.load_state_dict(torch.load(model_dir / f"fold_{fold}" / checkpoint_name(kind), map_location="cpu"))
        model = prepare_eval_model(kind, model, cfg)
    return model.eval()


def layer_cost(m, kind):
    """[MACs, bit-ops] of one linear layer: MACs x weight bits x activation bits (ternary = 2 bits),
    as scripts/analysis/cascade.py; the dynamic int8 PTQ model counts 8 x 8 on every Linear."""
    from QWave.qmoe_layers import BitLinear
    if kind == "qesc":
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
def logits(model, X, bs=64):
    # batches of 64 in order, as the benchmark's test loader: dynamic int8 PTQ
    # (qesc) picks its activation scale per batch
    return torch.cat([model(X[i:i + bs]) for i in range(0, len(X), bs)]).numpy()


@torch.no_grad()
def mc_probs(model, X, passes, seed):
    torch.manual_seed(seed)
    drops = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    for m in drops:
        m.train()
    try:
        P = torch.stack([torch.softmax(model(X), dim=1) for _ in range(passes)])
    finally:
        for m in drops:
            m.eval()
    return P.numpy().astype(np.float32)


def complete_folds(run):
    folds = sorted(int(p.name.split("_")[1]) for p in (run / "esc").glob("fold_*"))
    return [k for k in folds if all((run / m / f"fold_{k}" / "metrics.json").exists() for m in MODELS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--code", type=Path, default=Path("/Users/cajas.sebastian/Desktop/repositories/curious-qmoe-run-5db7e75"))
    ap.add_argument("--mc-passes", type=int, default=10)
    a = ap.parse_args()
    sys.path.insert(0, str(a.code))
    torch.set_num_threads(2)
    a.out.mkdir(parents=True, exist_ok=True)

    cfg4 = OmegaConf.load(a.run / "4" / "config.yaml")
    seed = int(cfg4.experiment.seed)
    df = load_frame(cfg4)
    in_dim, n_cls = df.shape[1] - 1, int(df["class_id"].nunique())
    folds = complete_folds(a.run)
    checks, costs = [], {}
    for fold in folds:
        split = json.loads((a.run / "esc" / f"fold_{fold}" / "split.json").read_text())
        for m in MODELS:
            s = json.loads((a.run / m / f"fold_{fold}" / "split.json").read_text())
            assert s["test_idx"] == split["test_idx"] and s["inner_val_idx"] == split["inner_val_idx"], (m, fold)
        rows = fold_rows(df, cfg4, split)
        (X_iv, y_iv), (X_te, y_te) = rows["inner_val"], rows["test"]
        arrs = {"y_iv": y_iv, "y_te": y_te, "test_idx": np.asarray(split["test_idx"]),
                "iv_idx": np.asarray(split["inner_val_idx"])}
        for m in MODELS:
            model = load_model(a.run / m, fold, in_dim, n_cls)
            costs[m] = cost(model, m).tolist()
            arrs[f"l_iv_{m}"] = logits(model, X_iv)
            arrs[f"l_te_{m}"] = logits(model, X_te)
            met = json.loads((a.run / m / f"fold_{fold}" / "metrics.json").read_text())
            f1_te = f1_score(y_te, arrs[f"l_te_{m}"].argmax(1), average="macro", zero_division=0)
            f1_iv = f1_score(y_iv, arrs[f"l_iv_{m}"].argmax(1), average="macro", zero_division=0)
            d_iv = abs(f1_iv - met["inner_val_f1_macro"]) if "inner_val_f1_macro" in met else None
            checks.append({"fold": fold, "model": m, "f1_te": f1_te, "metrics_f1": met["f1_macro"],
                           "diff_te": abs(f1_te - met["f1_macro"]), "f1_iv": f1_iv, "diff_iv": d_iv,
                           "size_bytes": met["model_size_bytes"], "n_test": met["n_test"]})
            assert checks[-1]["diff_te"] < 1e-9, checks[-1]
            if m in CHEAP:
                arrs[f"mc_iv_{m}"] = mc_probs(model, X_iv, a.mc_passes, seed)
                arrs[f"mc_te_{m}"] = mc_probs(model, X_te, a.mc_passes, seed)
        np.savez_compressed(a.out / f"fold_{fold}.npz", **arrs)
        print(fold, "max |F1 - metrics.json| test", max(c["diff_te"] for c in checks if c["fold"] == fold),
              "inner val", max((c["diff_iv"] or 0) for c in checks if c["fold"] == fold), flush=True)
    (a.out / "checks.json").write_text(json.dumps(
        {"run": str(a.run), "seed": seed, "folds": folds, "n_classes": n_cls, "mc_passes": a.mc_passes,
         "checks": checks, "costs": costs}, indent=1))


if __name__ == "__main__":
    main()
