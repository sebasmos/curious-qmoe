#!/usr/bin/env python
"""
K-Fold CV on embedding CSVs with inner split per fold 

python Baseline.py \
  --config-path /Users/sebasmos/Desktop/QWave/config \
  --config-name esc50 \
  experiment.datasets.esc.csv=/Users/sebasmos/Documents/DATASETS/data_VE/ESC-50-master/VE_soundscapes/efficientnet_1536/esc-50.csv \
  experiment.device=mps \
  experiment.metadata.tag=EfficientNet_esc50Baseline
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from codecarbon import EmissionsTracker

from QWave.datasets import EmbeddingDataset
from QWave.models import ESCModel, reset_weights
from QWave.train_utils import train_pytorch_local, _validate_single_epoch
from QWave.graphics import plot_multiclass_roc_curve, plot_losses
from QWave.utils import get_device

def _load_cc_csv(csv_path: Path) -> dict:
    """Helper to load the last row of a CodeCarbon CSV."""
    if not csv_path.is_file():
        return {}
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return {}
    return df.iloc[-1].to_dict()

# Keys for CodeCarbon metrics to track for each phase (duration included here)
cc_metrics_to_track = [
    "duration", "emissions", "emissions_rate", "cpu_power", "gpu_power", "ram_power",
    "cpu_energy", "gpu_energy", "ram_energy", "energy_consumed",
    "cpu_count", "cpu_model", "gpu_count", "gpu_model", "ram_total_size"
]

def run_cv(csv_path: str, cfg: DictConfig):
    df_full = pd.read_csv(csv_path)
    labels = df_full["class_id"].values
    df = df_full.drop(columns=["folder", "name", "label", "category"], errors="ignore")

    print("Data shape:", df.shape)

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

    # Lists to collect metrics across all folds for overall mean/std
    all_final_f1_scores = []
    all_final_accuracy_scores = []
    
    # Dictionaries to aggregate CodeCarbon metrics for training and test inference across folds
    all_train_cc_data_agg = {k: [] for k in cc_metrics_to_track}
    all_test_inference_cc_data_agg = {k: [] for k in cc_metrics_to_track}


    for fold, (train_outer_idx, test_outer_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n--- FOLD {fold+1}/{skf.n_splits} ---")
        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)

        # Step 1: Split the outer train_outer_idx into sub_train and sub_val for internal training loop
        df_train_outer = df.iloc[train_outer_idx].reset_index(drop=True)
        df_test_outer = df.iloc[test_outer_idx].reset_index(drop=True) # This is the true test set for the fold

        train_outer_labels = df_train_outer["class_id"].values

        sub_val_ld = None
        if cfg.experiment.cross_validation.validation_split_ratio > 0:
            sub_train_df, sub_val_df, _, _ = train_test_split(
                df_train_outer, train_outer_labels,
                test_size=cfg.experiment.cross_validation.validation_split_ratio,
                stratify=train_outer_labels,
                random_state=cfg.experiment.cross_validation.random_seed
            )
            sub_val_ds = EmbeddingDataset(sub_val_df)
            sub_val_ld = torch.utils.data.DataLoader(sub_val_ds, batch_size=cfg.experiment.model.batch_size, shuffle=False)
        else:
            sub_train_df = df_train_outer
            warnings.warn(f"No internal validation set for Fold {fold+1} as validation_split_ratio is 0. Early stopping might not be effective.")

        sub_train_ds = EmbeddingDataset(sub_train_df)
        sub_train_ld = torch.utils.data.DataLoader(sub_train_ds, batch_size=cfg.experiment.model.batch_size, shuffle=True)
        
        test_ds = EmbeddingDataset(df_test_outer)
        test_ld = torch.utils.data.DataLoader(test_ds, batch_size=cfg.experiment.model.batch_size, shuffle=False)

        # Initialize model for this fold
        in_dim = sub_train_ds.features.shape[1]
        num_classes = len(np.unique(labels)) 
        
        model = ESCModel(
            in_dim,
            num_classes,
            hidden_sizes=cfg.experiment.model.hidden_sizes,
            dropout_prob=cfg.experiment.model.dropout_prob
        ).to(device)
        model.apply(reset_weights)

        class_weights = torch.tensor(1.0 / np.bincount(sub_train_ds.labels.numpy()), dtype=torch.float32).to(device)
        
        ckpt_path = fold_dir / "best_model.pth"
        resume = cfg.experiment.logging.resume and ckpt_path.exists()

        train_tracker = EmissionsTracker(
            project_name=f"{tag}_fold{fold}_train",
            output_dir=str(fold_dir),
            output_file="emissions_train.csv", 
            save_to_file=True
        )
        train_tracker.start()

        # Phase 1: Model Training 
        model_trained_state, train_losses, val_losses, best_f1_from_train, \
        all_labels_best_epoch, all_preds_best_epoch, all_probs_best_epoch = train_pytorch_local(
            args=cfg.experiment,
            model=model,
            train_loader=sub_train_ld,
            val_loader=sub_val_ld,
            class_weights=class_weights,
            num_columns=in_dim,
            device=device,
            fold_folder=str(fold_dir),
            resume_checkpoint=resume,
            checkpoint_path=str(ckpt_path)
        )
        train_tracker.stop()

        test_inference_tracker = EmissionsTracker(
            project_name=f"{tag}_fold{fold}_test_inference",
            output_dir=str(fold_dir),
            output_file="emissions_test_inference.csv",
            save_to_file=True
        )
        test_inference_tracker.start()

        final_model = ESCModel(
            in_dim,
            num_classes,
            hidden_sizes=cfg.experiment.model.hidden_sizes,
            dropout_prob=cfg.experiment.model.dropout_prob
        ).to(device)
        final_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
        # Perform final evaluation on the TEST_LD
        _, _, all_labels_final, all_preds_final, all_probs_final = _validate_single_epoch(
            final_model, test_ld, torch.nn.CrossEntropyLoss(), device
        )
        
        test_inference_tracker.stop()

        train_stats = _load_cc_csv(fold_dir / "emissions_train.csv") # Load from new train file
        test_inference_stats = _load_cc_csv(fold_dir / "emissions_test_inference.csv")

        # Calculate final metrics from the dedicated test inference pass
        final_accuracy = accuracy_score(all_labels_final, all_preds_final)
        final_f1_weighted = f1_score(all_labels_final, all_preds_final, average="weighted", zero_division=0)
        
        print(f"  Fold {fold}: Final Accuracy={final_accuracy:.4f} | Final weighted-F1={final_f1_weighted:.4f}")

        fold_result = {
            "best_f1": final_f1_weighted, # Use final F1 from test inference pass
            "accuracy": final_accuracy,   # Use final accuracy from test inference pass
        }

        # Add all CodeCarbon metrics for the training phase
        for k in cc_metrics_to_track:
            fold_result[f"codecarbon_train_{k}"] = train_stats.get(k, 0.0)
            all_train_cc_data_agg[k].append(train_stats.get(k, 0.0))

        # Add all CodeCarbon metrics for the final test inference phase
        for k in cc_metrics_to_track:
            fold_result[f"codecarbon_test_inference_{k}"] = test_inference_stats.get(k, 0.0)
            all_test_inference_cc_data_agg[k].append(test_inference_stats.get(k, 0.0))

        all_final_f1_scores.append(final_f1_weighted)
        all_final_accuracy_scores.append(final_accuracy)

        # Plotting uses results from the final test inference and internal training history
        plot_multiclass_roc_curve(all_labels_final, all_probs_final, EXPERIMENT_NAME=str(fold_dir))
        # plot_losses uses train_losses and val_losses from the internal training loop
        plot_losses(train_losses, val_losses, str(fold_dir)) 

        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(fold_result, f, indent=4)

        fold_metrics.append(fold_result)

    # Calculate overall summary metrics
    summary = {
        "f1_mean": float(np.mean(all_final_f1_scores)),
        "f1_std": float(np.std(all_final_f1_scores)),
        "accuracy_mean": float(np.mean(all_final_accuracy_scores)),
        "accuracy_std": float(np.std(all_final_accuracy_scores)),
        "metadata": dict(cfg.experiment.metadata),
        "folds": fold_metrics
    }

    # Aggregate CodeCarbon metrics for all tracked phases
    for phase_data_agg, prefix in [(all_train_cc_data_agg, "train"), (all_test_inference_cc_data_agg, "test_inference")]:
        for k in cc_metrics_to_track:
            if phase_data_agg[k]:
                if all(isinstance(val, (int, float)) for val in phase_data_agg[k]):
                    summary[f"codecarbon_{prefix}_{k}_mean"] = float(np.mean(phase_data_agg[k]))
                    summary[f"codecarbon_{prefix}_{k}_std"] = float(np.std(phase_data_agg[k]))
                else: # For non-numeric values like 'cpu_model'
                    unique_values = list(set(phase_data_agg[k]))
                    if len(unique_values) == 1:
                        summary[f"codecarbon_{prefix}_{k}_common"] = unique_values[0]
                    else:
                        summary[f"codecarbon_{prefix}_{k}_all"] = unique_values # List all unique values


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