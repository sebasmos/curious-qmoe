#!/usr/bin/env python
"""
K-Fold CV on embedding CSVs with CodeCarbon energy tracking
epochs=200,learning_rate=0.01, patience=50
python train_mlpbitnnet.py \
  --config-path /Users/sebasmos/Desktop/QWave/config \
  --config-name esc50 \
  experiment.datasets.esc.normalization_type=standard \
  experiment.model.epochs=200 \
  experiment.model.patience=50 \
  experiment.model.learning_rate=0.01 \
  experiment.datasets.esc.csv=/Users/sebasmos/Documents/DATASETS/data_VE/ESC-50-master/VE_soundscapes/efficientnet_1536/esc-50.csv \
  experiment.device=mps \
  experiment.metadata.tag=bitnnet_esc50_standard
"""

from pathlib import Path
import os, sys
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import os, json, warnings
import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from codecarbon import EmissionsTracker
from memory_profiler import memory_usage
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch.nn.functional as F
from QWave.bitnnet import MLPBitnet, reset_weights
from QWave.datasets import EmbeddingDataset, EmbeddingAdaptDataset
from QWave.train_utils import train_pytorch_local, _validate_single_epoch
from QWave.graphics import plot_multiclass_roc_curve, plot_losses
from QWave.utils import get_device, set_seed

def reset_weights(m):
    """
    Resets the parameters of a model's layers.
    """
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Resetting weights for {layer}')
            layer.reset_parameters()

def _load_cc_csv(csv_path: Path) -> dict:
    if not csv_path.is_file():
        return {}
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return {}
    return df.iloc[-1].to_dict()

cc_metrics_to_track = [
    "duration", "emissions", "emissions_rate", "cpu_power", "gpu_power", "ram_power",
    "cpu_energy", "gpu_energy", "ram_energy", "energy_consumed",
    "cpu_count", "cpu_model", "gpu_count", "gpu_model", "ram_total_size"
]

def run_cv(csv_path: str, cfg: DictConfig):
    set_seed(1)
    df_full = pd.read_csv(csv_path)
    labels = df_full["class_id"].values
    df = df_full.drop(columns=["folder", "name", "label", "category"], errors="ignore")

    print("Data shape:", df.shape)
    print("Unique labels:", np.unique(labels))

    skf = StratifiedKFold(
        n_splits=cfg.experiment.cross_validation.n_splits,
        shuffle=cfg.experiment.cross_validation.shuffle,
        random_state=cfg.experiment.cross_validation.random_seed
    )

    fold_metrics = []
    tag = cfg.experiment.metadata.tag
    out_dir = Path("outputs") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(cfg)
    print(f"Using device: {device}")

    all_final_f1_scores = []
    all_final_accuracy_scores = []
    all_train_cc_data_agg = {k: [] for k in cc_metrics_to_track}
    all_val_cc_data_agg = {k: [] for k in cc_metrics_to_track}
    all_max_ram_mb_train = []
    all_max_ram_mb_val = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        
        print(f"\n--- FOLD {fold+1}/{skf.n_splits} ---")
        
        fold_dir = out_dir / f"fold_{fold}"
        
        fold_dir.mkdir(exist_ok=True)

        df_train = df.iloc[train_idx].reset_index(drop=True)
        
        df_val = df.iloc[val_idx].reset_index(drop=True)
        
        normalization_type = cfg.experiment.datasets.esc.get("normalization_type", "raw") # Default to "raw" if not specified

        train_ds = EmbeddingAdaptDataset(df_train,normalization_type=normalization_type, scaler=None)
        
        fitted_scaler = train_ds.get_scaler()
        
        val_ds = EmbeddingAdaptDataset(df_val,normalization_type=normalization_type, scaler=fitted_scaler)
        
        print("Label distribution:", np.bincount(train_ds.labels.numpy()))
        print("Feature range:", train_ds.features.min().item(), train_ds.features.max().item())

        train_ld = torch.utils.data.DataLoader(train_ds, batch_size=cfg.experiment.model.batch_size, shuffle=True)
        val_ld = torch.utils.data.DataLoader(val_ds, batch_size=cfg.experiment.model.batch_size, shuffle=False)

        in_dim = train_ds.features.shape[1]
        num_classes = len(np.unique(labels))

        model = MLPBitnet(
            in_dim,
            num_classes,
            hidden_sizes=cfg.experiment.model.hidden_sizes,
            dropout_prob=cfg.experiment.model.dropout_prob,
            use_residual=False
        ).to(device)
        model.apply(reset_weights)

        class_weights = torch.tensor(1.0 / np.bincount(train_ds.labels.numpy()), dtype=torch.float32)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = class_weights.to(device)
        print("Class weights:", class_weights.tolist())

        ckpt_path = fold_dir / "best_model.pth"
        resume = False # Disable resuming to ensure fresh training per fold

        train_tracker = EmissionsTracker(
            project_name=f"{tag}_fold{fold}_train",
            output_dir=str(fold_dir),
            output_file="emissions_train.csv",
            save_to_file=True
        )
        train_tracker.start()

        mem_train_usage, (model_trained_state, train_losses, val_losses, best_f1_from_train,
                          all_labels_best_epoch, all_preds_best_epoch, all_probs_best_epoch) = \
            memory_usage(
                (train_pytorch_local, (
                    cfg.experiment,
                    model,
                    train_ld,
                    val_ld,
                    class_weights,
                    in_dim,
                    device,
                    str(fold_dir),
                    resume,
                    str(ckpt_path)
                )),
                interval=0.1,
                retval=True
            )

        max_ram_mb_train = sum(mem_train_usage) / len(mem_train_usage)
        
        print("Train losses (last 5):", train_losses[-5:])
        print("Val losses (last 5):", val_losses[-5:])

        train_tracker.stop()

        val_tracker = EmissionsTracker(
            project_name=f"{tag}_fold{fold}_val",
            output_dir=str(fold_dir),
            output_file="emissions_val.csv",
            save_to_file=True
        )
        val_tracker.start()

        final_model = MLPBitnet(
            in_dim,
            num_classes,
            hidden_sizes=cfg.experiment.model.hidden_sizes,
            dropout_prob=cfg.experiment.model.dropout_prob,
            use_residual=False
        ).to(device)
        final_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        final_model.eval()

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        mem_val_usage, (_, _, all_labels_final, all_preds_final, all_probs_final) = \
            memory_usage(
                (_validate_single_epoch, (
                    final_model,
                    val_ld,
                    criterion,
                    device
                )),
                interval=0.1,
                retval=True
            )

        max_ram_mb_val = sum(mem_val_usage) / len(mem_val_usage)

        val_tracker.stop()

        train_stats = _load_cc_csv(fold_dir / "emissions_train.csv")
        val_stats = _load_cc_csv(fold_dir / "emissions_val.csv")

        final_accuracy = accuracy_score(all_labels_final, all_preds_final)
        final_f1_weighted = f1_score(all_labels_final, all_preds_final, average="weighted", zero_division=0)

        print(f"  Fold {fold+1}: Final Accuracy={final_accuracy:.4f} | Final weighted-F1={final_f1_weighted:.4f}")

        fold_result = {
            "best_f1": final_f1_weighted,
            "accuracy": final_accuracy,
            "max_ram_mb_train": float(max_ram_mb_train),
            "max_ram_mb_val": float(max_ram_mb_val),
        }

        for k in cc_metrics_to_track:
            fold_result[f"train_{k}"] = train_stats.get(k, 0.0)
            all_train_cc_data_agg[k].append(train_stats.get(k, 0.0))

        for k in cc_metrics_to_track:
            fold_result[f"val_{k}"] = val_stats.get(k, 0.0)
            all_val_cc_data_agg[k].append(val_stats.get(k, 0.0))

        all_final_f1_scores.append(final_f1_weighted)
        all_final_accuracy_scores.append(final_accuracy)
        all_max_ram_mb_train.append(max_ram_mb_train)
        all_max_ram_mb_val.append(max_ram_mb_val)

        plot_multiclass_roc_curve(all_labels_final, all_probs_final, EXPERIMENT_NAME=str(fold_dir))
        plot_losses(train_losses, val_losses, str(fold_dir))

        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(fold_result, f, indent=4)

        fold_metrics.append(fold_result)

    summary = {
        "f1_mean": float(np.mean(all_final_f1_scores)),
        "f1_std": float(np.std(all_final_f1_scores)),
        "accuracy_mean": float(np.mean(all_final_accuracy_scores)),
        "accuracy_std": float(np.std(all_final_accuracy_scores)),
        "metadata": dict(cfg.experiment.metadata),
        "folds": fold_metrics
    }

    if all_max_ram_mb_train:
        summary["avg_ram_mb_train_mean"] = float(np.mean(all_max_ram_mb_train))
        summary["avg_ram_mb_train_std"] = float(np.std(all_max_ram_mb_train))
    if all_max_ram_mb_val:
        summary["avg__ram_mb_val_mean"] = float(np.mean(all_max_ram_mb_val))
        summary["avg_ram_mb_val_std"] = float(np.std(all_max_ram_mb_val))

    for phase_data_agg, prefix in [(all_train_cc_data_agg, "train"), (all_val_cc_data_agg, "val")]:
        for k in cc_metrics_to_track:
            if phase_data_agg[k]:
                if all(isinstance(val, (int, float)) for val in phase_data_agg[k]):
                    summary[f"{prefix}_{k}_mean"] = float(np.mean(phase_data_agg[k]))
                    summary[f"{prefix}_{k}_std"] = float(np.std(phase_data_agg[k]))
                else:
                    unique_values = list(set(phase_data_agg[k]))
                    if len(unique_values) == 1:
                        summary[f"{prefix}_{k}_common"] = unique_values[0]
                    else:
                        summary[f"{prefix}_{k}_all"] = unique_values

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\n== CV SUMMARY ==")
    print(json.dumps(summary, indent=4))

@hydra.main(version_base=None, config_path=str(ROOT / "config"), config_name="esc50")
def main(cfg: DictConfig):
    for name, meta in cfg.experiment.datasets.items():
        csv_path = Path(meta.csv)
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        print(f"\n=== DATASET {name.upper()} → outputs/{cfg.experiment.metadata.tag}")
        run_cv(str(csv_path), cfg)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()