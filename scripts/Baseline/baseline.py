#!/usr/bin/env python
"""
K-Fold CV on embedding CSVs with CodeCarbon energy tracking

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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from codecarbon import EmissionsTracker
import psutil
import threading
import time

from QWave.datasets import EmbeddingDataset
from QWave.models import ESCModel, reset_weights
from QWave.train_utils import train_pytorch_local, _validate_single_epoch
from QWave.graphics import plot_multiclass_roc_curve, plot_losses
from QWave.utils import get_device

class SystemMonitor(threading.Thread):
    def __init__(self, interval: float = 1.0):
        super().__init__()
        self._stop_event = threading.Event()
        self.interval = interval
        self.cpu_usages_percent = []
        self.ram_usages_mb = []
        self.daemon = True 

    def run(self):
        psutil.cpu_percent(interval=None) 
        while not self._stop_event.is_set():
            try:
                self.cpu_usages_percent.append(psutil.cpu_percent(interval=None))
                vm = psutil.virtual_memory()
                self.ram_usages_mb.append(vm.used / (1024 * 1024))
                time.sleep(self.interval)
            except Exception as e:
                warnings.warn(f"Error during system monitoring: {e}")
                break

    def stop(self):
        self._stop_event.set()

    def get_metrics(self):
        avg_cpu_percent = np.mean(self.cpu_usages_percent) if self.cpu_usages_percent else 0.0
        avg_ram_mb = np.mean(self.ram_usages_mb) if self.ram_usages_mb else 0.0
        return avg_cpu_percent, avg_ram_mb

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

    all_final_f1_scores = []
    all_final_accuracy_scores = []
    
    all_train_cc_data_agg = {k: [] for k in cc_metrics_to_track}
    all_val_cc_data_agg = {k: [] for k in cc_metrics_to_track}
    all_avg_cpu_util_percent_train = []
    all_avg_ram_util_mb_train = []
    all_avg_cpu_util_percent_val = []
    all_avg_ram_util_mb_val = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n--- FOLD {fold+1}/{skf.n_splits} ---")
        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)

        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True) 

        train_ds = EmbeddingDataset(df_train)
        val_ds = EmbeddingDataset(df_val)
        
        train_ld = torch.utils.data.DataLoader(train_ds, batch_size=cfg.experiment.model.batch_size, shuffle=True)
        val_ld = torch.utils.data.DataLoader(val_ds, batch_size=cfg.experiment.model.batch_size, shuffle=False)

        in_dim = train_ds.features.shape[1]
        num_classes = len(np.unique(labels)) 
        
        model = ESCModel(
            in_dim,
            num_classes,
            hidden_sizes=cfg.experiment.model.hidden_sizes,
            dropout_prob=cfg.experiment.model.dropout_prob
        ).to(device)
        model.apply(reset_weights)

        class_weights = torch.tensor(1.0 / np.bincount(train_ds.labels.numpy()), dtype=torch.float32).to(device)
        
        ckpt_path = fold_dir / "best_model.pth"
        resume = cfg.experiment.logging.resume and ckpt_path.exists()

        train_tracker = EmissionsTracker(
            project_name=f"{tag}_fold{fold}_train",
            output_dir=str(fold_dir),
            output_file="emissions_train.csv", 
            save_to_file=True
        )
        train_tracker.start()
        train_monitor = SystemMonitor(interval=0.5)
        train_monitor.start()

        model_trained_state, train_losses, val_losses, best_f1_from_train, \
        all_labels_best_epoch, all_preds_best_epoch, all_probs_best_epoch = train_pytorch_local(
            args=cfg.experiment,
            model=model,
            train_loader=train_ld,
            val_loader=val_ld,  
            class_weights=class_weights,
            num_columns=in_dim,
            device=device,
            fold_folder=str(fold_dir), 
            resume_checkpoint=resume,
            checkpoint_path=str(ckpt_path)
        )
        train_monitor.stop()
        train_monitor.join()
        train_tracker.stop()
        avg_cpu_util_percent_train, avg_ram_util_mb_train = train_monitor.get_metrics()

        val_tracker = EmissionsTracker( 
            project_name=f"{tag}_fold{fold}_val",
            output_dir=str(fold_dir),
            output_file="emissions_val.csv",
            save_to_file=True
        )
        val_tracker.start()
        val_monitor = SystemMonitor(interval=0.5)
        val_monitor.start()

        final_model = ESCModel(
            in_dim,
            num_classes,
            hidden_sizes=cfg.experiment.model.hidden_sizes,
            dropout_prob=cfg.experiment.model.dropout_prob
        ).to(device)
        final_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
        _, _, all_labels_final, all_preds_final, all_probs_final = _validate_single_epoch(
            final_model, val_ld, torch.nn.CrossEntropyLoss(), device 
        )
        
        val_monitor.stop()
        val_monitor.join()
        val_tracker.stop()
        avg_cpu_util_percent_val, avg_ram_util_mb_val = val_monitor.get_metrics()

        train_stats = _load_cc_csv(fold_dir / "emissions_train.csv") 
        val_stats = _load_cc_csv(fold_dir / "emissions_val.csv")

        final_accuracy = accuracy_score(all_labels_final, all_preds_final)
        final_f1_weighted = f1_score(all_labels_final, all_preds_final, average="weighted", zero_division=0)
        
        print(f"  Fold {fold}: Final Accuracy={final_accuracy:.4f} | Final weighted-F1={final_f1_weighted:.4f}")

        fold_result = {
            "best_f1": final_f1_weighted, 
            "accuracy": final_accuracy,   
            "avg_cpu_util_percent_train": float(avg_cpu_util_percent_train),
            "avg_ram_util_mb_train": float(avg_ram_util_mb_train),
            "avg_cpu_util_percent_val": float(avg_cpu_util_percent_val),
            "avg_ram_util_mb_val": float(avg_ram_util_mb_val),
        }

        for k in cc_metrics_to_track:
            fold_result[f"codecarbon_train_{k}"] = train_stats.get(k, 0.0)
            all_train_cc_data_agg[k].append(train_stats.get(k, 0.0))

        for k in cc_metrics_to_track:
            fold_result[f"codecarbon_val_{k}"] = val_stats.get(k, 0.0)
            all_val_cc_data_agg[k].append(val_stats.get(k, 0.0))

        all_final_f1_scores.append(final_f1_weighted)
        all_final_accuracy_scores.append(final_accuracy)
        all_avg_cpu_util_percent_train.append(avg_cpu_util_percent_train)
        all_avg_ram_util_mb_train.append(avg_ram_util_mb_train)
        all_avg_cpu_util_percent_val.append(avg_cpu_util_percent_val)
        all_avg_ram_util_mb_val.append(avg_ram_util_mb_val)

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

    if all_avg_cpu_util_percent_train:
        summary["avg_cpu_util_percent_train_mean"] = float(np.mean(all_avg_cpu_util_percent_train))
        summary["avg_cpu_util_percent_train_std"] = float(np.std(all_avg_cpu_util_percent_train))
        summary["avg_ram_util_mb_train_mean"] = float(np.mean(all_avg_ram_util_mb_train))
        summary["avg_ram_util_mb_train_std"] = float(np.std(all_avg_ram_util_mb_train))
    if all_avg_cpu_util_percent_val:
        summary["avg_cpu_util_percent_val_mean"] = float(np.mean(all_avg_cpu_util_percent_val))
        summary["avg_cpu_util_percent_val_std"] = float(np.std(all_avg_cpu_util_percent_val))
        summary["avg_ram_util_mb_val_mean"] = float(np.mean(all_avg_ram_util_mb_val))
        summary["avg_ram_util_mb_val_std"] = float(np.std(all_avg_ram_util_mb_val))

    for phase_data_agg, prefix in [(all_train_cc_data_agg, "train"), (all_val_cc_data_agg, "val")]:
        for k in cc_metrics_to_track:
            if phase_data_agg[k]:
                if all(isinstance(val, (int, float)) for val in phase_data_agg[k]):
                    summary[f"codecarbon_{prefix}_{k}_mean"] = float(np.mean(phase_data_agg[k]))
                    summary[f"codecarbon_{prefix}_{k}_std"] = float(np.std(phase_data_agg[k]))
                else:
                    unique_values = list(set(phase_data_agg[k]))
                    if len(unique_values) == 1:
                        summary[f"codecarbon_{prefix}_{k}_common"] = unique_values[0]
                    else:
                        summary[f"codecarbon_{prefix}_{k}_all"] = unique_values 


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