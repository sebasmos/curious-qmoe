"""Score trained single heads on every condition's embeddings.

Idea 5, Stage 1. Heads come from scripts/benchmark.py run directories
(outputs/<tag>/<kind>), trained on the clean FP32-backbone AST CSV. For each
fold, the scaler is fitted on the clean CSV's fit portion of that fold
(training folds minus the fold's inner validation split, as in benchmark.py)
and applied unchanged to every condition's embeddings, so a corrupted or
quantized backbone is seen exactly as a deployed head would see it.

Per (run, condition) this writes <out>/<tag>/<kind>/<condition>.npz with
  test_logits [N, C], test_mcvar [N], test_fold [N]   each clip scored by the
      head of the fold that holds it out (test rows only)
  iv_idx_<k>, iv_logits_<k>, iv_mcvar_<k>            the fold's inner
      validation rows, for thresholds chosen without test labels
mcvar is the MC-dropout predictive variance of the head (dropout on, M passes,
variance of the softmax summed over classes, seeded with the run seed per
fold), as in scripts/analysis/cascade.py.

The clean FP32 condition is read from the training CSV itself (fp32__clean);
every other condition is a float32 .npy from extract_conditions.py in the
CSV's row order.

    python scripts/redesign/idea5/score_heads.py --runs outputs/astesc5_a_s0/4 ... \
        --emb-dir EMB --out SCORES [--mc-passes 10]
"""
from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.analysis.cascade import load_model, mc_variance  # noqa: E402

NON_FEATURE_COLS = ["folder", "name", "label", "category"]


def clean_frame(cfg):
    esc = cfg.experiment.datasets.esc
    df = pd.read_csv(esc.csv)
    label_col = esc.get("label_col", "class_id")
    y = df[label_col].to_numpy()
    X = df[[str(i) for i in range(768)]].to_numpy(np.float32)
    return X, y, df["folder"].to_numpy()


def fold_scaler(X_clean, y, split, norm):
    from QWave.datasets import EmbeddingAdaptDataset

    n = len(X_clean)
    fit_idx = np.setdiff1d(np.arange(n), np.concatenate([split["test_idx"], split["inner_val_idx"]]))
    df = pd.DataFrame(X_clean[fit_idx], columns=[str(i) for i in range(X_clean.shape[1])])
    df["class_id"] = y[fit_idx]
    with contextlib.redirect_stdout(io.StringIO()):
        return EmbeddingAdaptDataset(df, normalization_type=norm).get_scaler()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--emb-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mc-passes", type=int, default=10)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)

    for run in args.runs:
        run = run.resolve()
        cfg = OmegaConf.load(run / "config.yaml")
        seed = int(cfg.experiment.seed)
        norm = cfg.experiment.datasets.esc.normalization_type
        X_clean, y, _ = clean_frame(cfg)
        n_cls = len(np.unique(y))
        conds = {"fp32__clean": X_clean}
        for p in sorted(args.emb_dir.glob("*.npy")):
            if p.stem != "fp32__clean":
                conds[p.stem] = p
        out_dir = args.out / run.parent.name / run.name
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [c for c in conds if not (out_dir / f"{c}.npz").exists()]
        if not todo:
            continue
        summary = {f["fold"]: f for f in json.loads((run / "summary.json").read_text())["folds"]}
        folds = sorted(summary)
        res = {c: {"test_logits": np.full((len(y), n_cls), np.nan, np.float32),
                   "test_mcvar": np.full(len(y), np.nan, np.float32),
                   "test_fold": np.zeros(len(y), np.int16)} for c in todo}
        Xs = {c: (conds[c] if isinstance(conds[c], np.ndarray) else np.load(conds[c])) for c in todo}
        for c, X in Xs.items():
            if X.shape != X_clean.shape:
                raise SystemExit(f"{c}: shape {X.shape} != {X_clean.shape}")
        for fold in folds:
            split = json.loads((run / f"fold_{fold}" / "split.json").read_text())
            te, iv = np.asarray(split["test_idx"]), np.asarray(split["inner_val_idx"])
            scaler = fold_scaler(X_clean, y, split, norm)
            model, _ = load_model(run, fold, X_clean.shape[1], n_cls)
            for c in todo:
                r = res[c]
                for name, idx in (("test", te), ("iv", iv)):
                    Z = torch.tensor(scaler.transform(Xs[c][idx]), dtype=torch.float32)
                    with torch.no_grad():
                        logits = model(Z).numpy()
                    mcv = mc_variance(model, Z, args.mc_passes, seed)
                    if name == "test":
                        r["test_logits"][te], r["test_mcvar"][te], r["test_fold"][te] = logits, mcv, fold
                    else:
                        r[f"iv_idx_{fold}"], r[f"iv_logits_{fold}"], r[f"iv_mcvar_{fold}"] = iv, logits, mcv
                if c == "fp32__clean":
                    f1 = f1_score(y[te], r["test_logits"][te].argmax(1), average="macro", zero_division=0)
                    if abs(f1 - summary[fold]["f1_macro"]) > 1e-6:
                        raise SystemExit(f"{run} fold {fold}: rescored clean F1 {f1:.6f} != summary "
                                         f"{summary[fold]['f1_macro']:.6f}")
        for c in todo:
            np.savez_compressed(out_dir / f"{c}.npz", y=y, **res[c])
        print(f"{run.parent.name}/{run.name}: scored {len(todo)} condition(s)", flush=True)


if __name__ == "__main__":
    main()
