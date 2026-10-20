"""Rebuild a trained fold from a benchmark run directory.

A run directory is outputs/<tag>/<model_kind>/ as written by
scripts/benchmark.py: config.yaml (resolved), and per fold fold_<k>/split.json
plus the selected checkpoint. load_trained_fold returns the network exactly as
it was scored (PTQ applied) and that fold's held-out test rows, scaled with
the scaler fitted on the fold's fit portion (training folds minus the inner
validation split), so analyses never see raw features or training samples.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

NON_FEATURE_COLS = ["folder", "name", "label", "category"]


@dataclass
class TrainedFold:
    cfg: object
    model_kind: str
    fold: int
    model: torch.nn.Module
    X_test: torch.Tensor
    y_test: torch.Tensor
    test_idx: np.ndarray


def list_folds(run_dir) -> list[int]:
    return sorted(int(p.name.split("_")[1]) for p in Path(run_dir).glob("fold_*") if (p / "split.json").exists())


def load_heldout(cfg, split: dict, csv: str | None = None):
    """Held-out test rows of one fold, scaled with that fold's fit-portion scaler."""
    from QWave.datasets import EmbeddingAdaptDataset

    df = pd.read_csv(csv or cfg.experiment.datasets.esc.csv)
    label_col = cfg.experiment.datasets.esc.get("label_col", "class_id")
    if label_col != "class_id":
        df["class_id"] = df[label_col]
    df_feat = df.drop(columns=NON_FEATURE_COLS, errors="ignore")
    test_idx = np.asarray(split["test_idx"])
    fit_idx = np.setdiff1d(np.arange(len(df)), np.concatenate([test_idx, np.asarray(split["inner_val_idx"])]))
    norm = cfg.experiment.datasets.esc.normalization_type
    fit_ds = EmbeddingAdaptDataset(df_feat.iloc[fit_idx].reset_index(drop=True), normalization_type=norm)
    te_ds = EmbeddingAdaptDataset(df_feat.iloc[test_idx].reset_index(drop=True), normalization_type=norm,
                                  scaler=fit_ds.get_scaler())
    return te_ds.features, te_ds.labels, test_idx, df_feat.shape[1] - 1, int(df["class_id"].nunique())


def load_trained_fold(run_dir, fold: int = 1, csv: str | None = None, device: str = "cpu") -> TrainedFold:
    from scripts.benchmark import build_model, checkpoint_name, prepare_eval_model

    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / "config.yaml")
    if csv:
        cfg.experiment.datasets.esc.csv = str(csv)
    cfg.experiment.device = device
    kind = run_dir.name
    split = json.loads((run_dir / f"fold_{fold}" / "split.json").read_text())
    X, y, test_idx, in_dim, n_cls = load_heldout(cfg, split, csv)
    cfg.experiment.current_fold = fold
    model = build_model(kind, in_dim, n_cls, cfg).to(device)
    state = torch.load(run_dir / f"fold_{fold}" / checkpoint_name(kind), map_location=device)
    model.load_state_dict(state)
    model = prepare_eval_model(kind, model, cfg)
    model.eval()
    return TrainedFold(cfg, kind, fold, model, X, y, test_idx)
