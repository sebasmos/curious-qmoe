#!/usr/bin/env python
"""
K-Fold CV on embedding CSVs
"""
from pathlib import Path
import os, sys
ROOT = Path(__file__).resolve().parents[2]      # → /…/QWave
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import os, json, warnings
import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from QWave.datasets import EmbeddingDataset
from QWave.models import SClassifier, reset_weights, ESCModel
from QWave.train_utils import train_pytorch_local
from QWave.graphics import plot_multiclass_roc_curve, plot_losses
from QWave.utils import get_device 

def run_cv(csv_path: str, cfg: DictConfig):
    df_full = pd.read_csv(csv_path)
    labels = df_full["class_id"].values
    df = df_full.drop(columns=["folder", "name", "label", "category"])

    print("Data shape:", df.shape)

    skf = StratifiedKFold(
        n_splits=cfg.experiment.cross_validation.n_splits,
        shuffle=cfg.experiment.cross_validation.shuffle,
        random_state=cfg.experiment.cross_validation.random_seed
    )

    fold_metrics = []
    tag = cfg.experiment.metadata.tag
    out_dir = os.path.abspath(os.path.join("outputs", tag))
    os.makedirs(out_dir, exist_ok=True)

    # Select device once
    device = get_device(cfg)
    print(f"Using device: {device}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n--- FOLD {fold+1}/{skf.n_splits} ---")

        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)

        tr_ds = EmbeddingDataset(df_train)
        va_ds = EmbeddingDataset(df_val)

        tr_ld = torch.utils.data.DataLoader(tr_ds, batch_size=cfg.experiment.model.batch_size, shuffle=True)
        va_ld = torch.utils.data.DataLoader(va_ds, batch_size=cfg.experiment.model.batch_size, shuffle=False)

        class_weights = torch.tensor(1.0 / np.bincount(tr_ds.labels.numpy()), dtype=torch.float32).to(device)
        in_dim = tr_ds.features.shape[1]
        num_classes = len(np.unique(labels))

        model = ESCModel(
            in_dim,
            num_classes,
            hidden_sizes=cfg.experiment.model.hidden_sizes,
            dropout_prob=cfg.experiment.model.dropout_prob
        ).to(device)
        model.apply(reset_weights)

        fold_dir = os.path.join(out_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        ckpt_path = os.path.join(fold_dir, "best_model.pth")
        resume = cfg.experiment.logging.resume and os.path.exists(ckpt_path)

        model, train_losses, val_losses, best_f1, all_labels, all_preds, all_probs = train_pytorch_local(
            args=cfg.experiment,
            model=model,
            train_loader=tr_ld,
            val_loader=va_ld,
            class_weights=class_weights,
            num_columns=in_dim,
            device=device,
            fold_folder=fold_dir,
            resume_checkpoint=resume,
            checkpoint_path=ckpt_path
        )

        plot_multiclass_roc_curve(all_labels, all_probs, EXPERIMENT_NAME=fold_dir)
        plot_losses(train_losses, val_losses, fold_dir)
        fold_metrics.append(dict(best_f1=best_f1))

        with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
            json.dump({
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_f1": best_f1
            }, f, indent=4)

    summary = {
        "f1_mean": float(np.mean([m["best_f1"] for m in fold_metrics])),
        "f1_std": float(np.std([m["best_f1"] for m in fold_metrics])),
        "metadata": dict(cfg.experiment.metadata),
        "folds": fold_metrics
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(json.dumps(summary, indent=4))


# @hydra.main(version_base=None, config_path=".", config_name="configs")
@hydra.main(version_base=None, config_path="config", config_name="esc50")
def main(cfg: DictConfig):
    for name, meta in cfg.experiment.datasets.items():
        csv_path = meta.csv
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(csv_path)

        print(f"\n=== DATASET {name.upper()} → outputs/{cfg.experiment.metadata.tag}")
        run_cv(csv_path, cfg)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()