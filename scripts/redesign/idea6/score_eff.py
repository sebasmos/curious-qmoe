"""Score trained EfficientNet heads (scripts/benchmark.py run directories) on
their own embedding CSV: test-fold and inner-validation logits for every clip.

Same layout as Stage 2 of idea 5 (scripts/redesign/idea5/score_stage2.py on
redesign/idea5-stage2), so the cheap tier and the AST tier can be joined row
by row. For each fold the scaler is fitted on the fold's fit portion (training
folds minus the inner validation split) and the fold's selected checkpoint is
prepared exactly as benchmark.py scored it. Before writing, every fold's
macro F1 on its test rows must equal the fold's F1 in summary.json.

Per run directory <outputs>/<tag>/<kind> this writes <out>/<tag>/<kind>.npz:
  y, test_fold [N]             labels and the fold that holds each clip out
  test_logits [N, C]           each clip scored by the head of its test fold
  iv_idx_<k>, iv_logits_<k>    fold k's inner validation rows and their logits
  names_crc32                  zlib.crc32 of the CSV's file stems joined by newlines

    python scripts/redesign/idea6/score_eff.py --runs OUT/outputs/effesc50_s0/esc ... --out SCORES/esc50
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.analysis.cascade import load_model  # noqa: E402


def frame(cfg):
    esc = cfg.experiment.datasets.esc
    df = pd.read_csv(esc.csv)
    feats = [c for c in df.columns if c.isdigit()]
    if feats != [str(i) for i in range(len(feats))]:
        raise SystemExit(f"{esc.csv}: feature columns are not 0..D-1")
    y = df[esc.get("label_col", "class_id")].to_numpy()
    stems = df["name"].map(lambda s: Path(s).stem).tolist()
    return df[feats].to_numpy(np.float32), y, zlib.crc32("\n".join(stems).encode())


def fold_scaler(X, y, split, norm):
    from QWave.datasets import EmbeddingAdaptDataset

    fit = np.setdiff1d(np.arange(len(X)), np.concatenate([split["test_idx"], split["inner_val_idx"]]))
    df = pd.DataFrame(X[fit], columns=[str(i) for i in range(X.shape[1])])
    df["class_id"] = y[fit]
    with contextlib.redirect_stdout(io.StringIO()):
        return EmbeddingAdaptDataset(df, normalization_type=norm).get_scaler()


@torch.no_grad()
def logits_of(model, scaler, X):
    return model(torch.tensor(scaler.transform(X), dtype=torch.float32)).numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    cache = {}
    for run in args.runs:
        run = run.resolve()
        dest = args.out / run.parent.name / f"{run.name}.npz"
        if dest.exists():
            continue
        cfg = OmegaConf.load(run / "config.yaml")
        key = (cfg.experiment.datasets.esc.csv, cfg.experiment.datasets.esc.get("label_col", "class_id"))
        if key not in cache:
            cache[key] = frame(cfg)
        X, y, crc = cache[key]
        n_cls = len(np.unique(y))
        norm = cfg.experiment.datasets.esc.normalization_type
        summary = {f["fold"]: f for f in json.loads((run / "summary.json").read_text())["folds"]}
        res = {"y": y, "test_fold": np.full(len(y), -1, np.int16),
               "test_logits": np.full((len(y), n_cls), np.nan, np.float32), "names_crc32": np.int64(crc)}
        for fold in sorted(summary):
            split = json.loads((run / f"fold_{fold}" / "split.json").read_text())
            model, _ = load_model(run, fold, X.shape[1], n_cls)
            te, iv = np.asarray(split["test_idx"]), np.asarray(split["inner_val_idx"])
            scaler = fold_scaler(X, y, split, norm)
            lt = logits_of(model, scaler, X[te])
            f1 = f1_score(y[te], lt.argmax(1), average="macro", zero_division=0)
            if abs(f1 - summary[fold]["f1_macro"]) > 1e-6:
                raise SystemExit(f"{run} fold {fold}: rescored F1 {f1:.6f} != summary {summary[fold]['f1_macro']:.6f}")
            res["test_logits"][te] = lt
            res["test_fold"][te] = fold
            res[f"iv_idx_{fold}"], res[f"iv_logits_{fold}"] = iv, logits_of(model, scaler, X[iv])
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, **res)
        print(f"{run.parent.name}/{run.name}: {len(summary)} folds, F1 equal to summary.json", flush=True)


if __name__ == "__main__":
    main()
