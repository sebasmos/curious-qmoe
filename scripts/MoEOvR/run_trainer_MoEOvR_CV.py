#!/usr/bin/env python
"""
K-Fold Mixture-of-Experts (Multi-class Experts & Router)

Cross-validation version of the MoE pipeline. Keeps the same folder/metrics
conventions as the original K-Fold CV on embedding CSVs reference.
This version uses a two-stage Experts -> Router approach where:

Key points
----------
- Expert phase – Multiple `ESCModel` instances, each trained as a **binary classifier**
                 (One-vs-Rest) on the training data within the fold.
                 Each expert will output a probability for its specialized class.
                 NOTE: Requires `train_pytorch_local` to support binary classification
                       with `BCEWithLogitsLoss` or a custom loop for experts.
- Router phase – A multi-layer perceptron (MLP) router trained with CrossEntropyLoss
                 to combine the outputs from all binary experts.
- Metrics – Includes training/inference duration, CodeCarbon energy metrics,
            CPU utilization percentage, and RAM utilization in MB per fold and in summary.

python run_trainer_MoEOvR_CV.py  --config-name=esc50 \
  experiment.datasets.esc.csv=/Users/sebasmos/Documents/DATASETS/data_VE/ESC-50-master/VE_soundscapes/efficientnet_1536/esc-50.csv \
  experiment.device=mps \
  experiment.metadata.tag="EfficientNet_esc50MoEOvR_CV"

"""

from pathlib import Path
import os, sys
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import os, json, warnings
from typing import List, Dict

import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

from codecarbon import EmissionsTracker
import psutil
import threading
import time

from QWave.datasets import EmbeddingDataset
from QWave.models import ESCModel, reset_weights
# IMPORT CHANGED: Import train_pytorch_adapter instead of train_pytorch_local
from QWave.train_utils import train_pytorch_adapter # This is your new adapted function
from QWave.graphics import plot_multiclass_roc_curve
from QWave.utils import get_device


# SystemMonitor class to collect CPU utilization percentage and RAM utilization in MB
class SystemMonitor(threading.Thread):
    def __init__(self, interval: float = 1.0):
        super().__init__()
        self._stop_event = threading.Event()
        self.interval = interval
        self.cpu_usages_percent = [] # Stores CPU usage in percentage (0-100)
        self.ram_usages_mb = []     # Stores RAM usage in Megabytes
        self.daemon = True 

    def run(self):
        # Call cpu_percent once with interval=None to prime it for accurate subsequent calls
        psutil.cpu_percent(interval=None) 
        while not self._stop_event.is_set():
            try:
                self.cpu_usages_percent.append(psutil.cpu_percent(interval=None))
                # Get virtual memory stats and convert used memory from bytes to MB
                vm = psutil.virtual_memory()
                self.ram_usages_mb.append(vm.used / (1024 * 1024)) # Convert bytes to MB
                time.sleep(self.interval)
            except Exception as e:
                warnings.warn(f"Error during system monitoring: {e}")
                break # Exit loop on error

    def stop(self):
        self._stop_event.set()

    def get_metrics(self):
        avg_cpu_percent = np.mean(self.cpu_usages_percent) if self.cpu_usages_percent else 0.0
        avg_ram_mb = np.mean(self.ram_usages_mb) if self.ram_usages_mb else 0.0
        return avg_cpu_percent, avg_ram_mb

class Router(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, drop_prob: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# This function now collects a single probability (or logit) from each expert.
# Each expert is assumed to be a binary classifier for a specific class.
def _expert_output_binary_scores(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_scores = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            # The model (expert) now outputs a single logit for the positive class
            scores = model(xb).squeeze(1).cpu() # Squeeze to remove the 1-dimension if present
            all_scores.append(scores)
    return torch.cat(all_scores, dim=0) # Concatenate along the batch dimension


def _train_router(router: Router, X: torch.Tensor, y: np.ndarray, device: torch.device,
                   *, epochs: int, lr: float, batch_size: int, weight_decay: float = 0.0):
    # For multi-class classification, use CrossEntropyLoss, and y should be class indices (long tensor).
    ds = TensorDataset(X, torch.from_numpy(y).long())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    crit = nn.CrossEntropyLoss()
    opt  = torch.optim.Adam(router.parameters(), lr=lr, weight_decay=weight_decay)
    router.to(device)
    router.train()
    for ep in range(1, epochs+1):
        run_loss = 0.0 # Renamed variable for clarity
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad();  loss = crit(router(xb), yb);  loss.backward();  opt.step()
            run_loss += loss.item() * len(xb)
        if ep % 15 == 0:
            print(f"    Router epoch {ep:3d} | CrossEntropyLoss: {run_loss/len(dl.dataset):.4f}")


def _load_cc_csv(csv_path: Path) -> Dict:
    if not csv_path.is_file():
        return {}
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return {}
    return df.iloc[-1].to_dict()

cc_keys_overall = [
    "project_name", "emissions",
    "emissions_rate","cpu_power","gpu_power","ram_power","cpu_energy","gpu_energy",
    "ram_energy","energy_consumed",
    "cpu_count","cpu_model","gpu_count","gpu_model",
    "ram_total_size"
]


def run_cv_moe(csv_path: str, cfg: DictConfig):
    df_full = pd.read_csv(csv_path)
    labels  = df_full["class_id"].astype(int).values
    df      = df_full.drop(columns=["folder", "name", "label", "category"], errors="ignore")
    n_classes = int(labels.max() + 1)

    skf = StratifiedKFold(
        n_splits = cfg.experiment.cross_validation.n_splits,
        shuffle  = cfg.experiment.cross_validation.shuffle,
        random_state = cfg.experiment.cross_validation.random_seed,
    )

    tag       = cfg.experiment.metadata.tag
    out_dir = Path("outputs") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(cfg)
    print(f"Using device: {device}\nData shape: {df.shape}")

    fold_results = []
    
    all_f1_scores = []
    all_accuracy_scores = []
    all_overall_codecarbon_data = {k: [] for k in cc_keys_overall} 
    all_train_duration_data = [] 
    all_val_duration_data = []   
    # Lists to store psutil metrics across folds
    all_avg_cpu_util_percent_train = []
    all_avg_ram_util_mb_train = []
    all_avg_cpu_util_percent_val = []
    all_avg_ram_util_mb_val = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n--- FOLD {fold+1}/{skf.n_splits} ---")
        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)

        overall_tracker = EmissionsTracker(
            project_name=f"{tag}_fold{fold}_overall",
            output_dir=str(fold_dir),
            output_file="emissions_overall.csv",
            save_to_file=True,
        )
        overall_tracker.start()

        df_tr_orig, df_va_orig = df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)
        y_tr_orig, y_va_orig   = labels[tr_idx], labels[va_idx] # Keep original labels for router and final eval

        train_duration_tracker = EmissionsTracker(
            project_name=f"{tag}_fold{fold}_train_experts_duration",
            output_dir=str(fold_dir),
            output_file="emissions_train_duration.csv",
            save_to_file=True,
        )
        train_duration_tracker.start()
        # Start system monitor for expert training phase
        train_monitor = SystemMonitor(interval=0.5)
        train_monitor.start()

        print("  == PHASE 1: training binary experts (One-vs-Rest) ==")
        experts: List[nn.Module] = []
        for cls in range(n_classes):
            print(f"    Training Expert for Class {cls}...")
            # Create binary labels for the current expert (1 for target class, 0 for others)
            y_tr_binary = (y_tr_orig == cls).astype(int)
            y_va_binary = (y_va_orig == cls).astype(int)

            # --- FIX STARTS HERE ---
            # Create temporary DataFrames for this expert, incorporating the binary labels
            df_tr_expert = df_tr_orig.copy()
            df_va_expert = df_va_orig.copy()
            df_tr_expert['class_id'] = y_tr_binary 
            df_va_expert['class_id'] = y_va_binary

            # Now pass these modified DataFrames to EmbeddingDataset
            tr_ds = EmbeddingDataset(df_tr_expert)
            va_ds = EmbeddingDataset(df_va_expert)
            # --- FIX ENDS HERE ---
            
            tr_ld = DataLoader(tr_ds, batch_size=cfg.experiment.model.batch_size, shuffle=True)
            va_ld = DataLoader(va_ds, batch_size=cfg.experiment.model.batch_size, shuffle=False)
            
            in_dim  = tr_ds.features.shape[1]
            # ESCModel's output_dim is 1 for binary classification
            model   = ESCModel(in_dim, 1, # Output dimension is 1 for binary classification (logit)
                                 hidden_sizes=cfg.experiment.model.hidden_sizes,
                                 dropout_prob=cfg.experiment.model.dropout_prob).to(device)
            model.apply(reset_weights)

            exp_dir = fold_dir / f"expert_{cls}"; exp_dir.mkdir(exist_ok=True)
            
            # Calculate pos_weight for BCEWithLogitsLoss
            num_positive_samples = y_tr_binary.sum()
            num_negative_samples = len(y_tr_binary) - num_positive_samples
            
            # Ensure pos_weight_val is not 0 if num_positive_samples is 0
            pos_weight_val = num_negative_samples / num_positive_samples if num_positive_samples > 0 else 1.0
            
            # Call the adapted training function for binary experts
            # trained_model, _, _, _, _, _, _ = train_pytorch_adapter(
            #     args=cfg.experiment.model, # Pass model-specific args for consistency (epochs, lr, patience etc.)
            #     model=model,
            #     train_loader=tr_ld,
            #     val_loader=va_ld,
            #     num_columns=in_dim, # num_columns is not strictly needed by train_pytorch_adapter but kept for signature
            #     device=device,
            #     fold_folder=str(exp_dir),
            #     task_type='binary', # <<< IMPORTANT: Specify binary task type
            #     pos_weight=pos_weight_val, # <<< Pass pos_weight for BCEWithLogitsLoss
            #     class_weights=None, # Not used for binary tasks
            #     resume_checkpoint=False,
            #     checkpoint_path=None # No resume for fresh expert training
            # )
            train_pytorch_adapter(
                args=cfg.experiment, # This passes the correct `model` DictConfig
                model=model,
                train_loader=tr_ld,
                val_loader=va_ld,
                num_columns=in_dim,
                device=device,
                fold_folder=str(exp_dir),
                task_type='binary',
                pos_weight=pos_weight_val,
                class_weights=None,
                resume_checkpoint=False,
                checkpoint_path=None
            )
            
            # Load the best model state saved by train_pytorch_adapter
            model.load_state_dict(torch.load(str(exp_dir / "best_model.pth"), map_location=device))
            experts.append(model.eval()) # Append the best-trained expert model

        train_monitor.stop()
        train_monitor.join()
        train_duration_tracker.stop() 
        
        avg_cpu_util_percent_train, avg_ram_util_mb_train = train_monitor.get_metrics()


        tr_full_ds = EmbeddingDataset(df_tr_orig); tr_ld_full = DataLoader(tr_full_ds, batch_size=cfg.experiment.model.batch_size)
        va_full_ds = EmbeddingDataset(df_va_orig); va_ld_full = DataLoader(va_full_ds, batch_size=cfg.experiment.model.batch_size)
        
        # Collects a single score/logit from each expert.
        # Concatenate them to form the router input.
        collect = lambda ld: torch.cat([_expert_output_binary_scores(exp, ld, device).unsqueeze(1) for exp in experts], dim=1)
        X_tr, X_va = collect(tr_ld_full), collect(va_ld_full)

        print("  == PHASE 2: training router ==")
        router_cfg = getattr(cfg, "router", {})
        
        # Router input_dim is now n_classes (one score from each binary expert)
        router = Router(
            input_dim=n_classes, # Router input is now n_classes (one score per expert)
            hidden_dim=router_cfg.get("hidden_dim", 128),
            output_dim=n_classes, # Router output is still n_classes (for final multi-class prediction)
            drop_prob=router_cfg.get("drop_prob", 0.2)
        )
        _train_router(router, X_tr, y_tr_orig, device, # Router trains on original multi-class labels
                      epochs=router_cfg.get("epochs", 75),
                      lr=router_cfg.get("lr", 2e-3),
                      batch_size=router_cfg.get("batch_size", 256),
                      weight_decay=router_cfg.get("weight_decay", 0.0))
        
        val_duration_tracker = EmissionsTracker(
            project_name=f"{tag}_fold{fold}_router_val_duration",
            output_dir=str(fold_dir),
            output_file="emissions_val_duration.csv",
            save_to_file=True,
        )
        val_duration_tracker.start()
        # Start system monitor for router inference phase
        val_monitor = SystemMonitor(interval=0.5)
        val_monitor.start()

        router.eval()
        with torch.no_grad():
            logits_va = router(X_va.to(device))
            probs_va  = torch.softmax(logits_va, dim=1).cpu().numpy()
            y_pred    = probs_va.argmax(axis=1)
        
        val_monitor.stop()
        val_monitor.join()
        val_duration_tracker.stop() 
        
        avg_cpu_util_percent_val, avg_ram_util_mb_val = val_monitor.get_metrics()

        overall_tracker.stop() 
        
        overall_stats = _load_cc_csv(fold_dir / "emissions_overall.csv")
        train_dur_stats = _load_cc_csv(fold_dir / "emissions_train_duration.csv")
        val_dur_stats = _load_cc_csv(fold_dir / "emissions_val_duration.csv")

        acc  = accuracy_score(y_va_orig, y_pred) # Evaluate against original multi-class labels
        w_f1 = f1_score(y_va_orig, y_pred, average="weighted")
        print(f"  Fold {fold}: Accuracy={acc:.4f} | weighted-F1={w_f1:.4f}")

        plot_multiclass_roc_curve(y_va_orig, probs_va, EXPERIMENT_NAME=str(fold_dir))

        fold_result = {
            "best_f1": w_f1,
            "accuracy": acc,
            "codecarbon_train_duration": train_dur_stats.get("duration"), 
            "codecarbon_val_duration": val_dur_stats.get("duration"),     
            "codecarbon_duration": overall_stats.get("duration"),         
            # Add psutil metrics to fold results (CPU in percent, RAM in MB)
            "avg_cpu_util_percent_train": float(avg_cpu_util_percent_train),
            "avg_ram_util_mb_train": float(avg_ram_util_mb_train),
            "avg_cpu_util_percent_val": float(avg_cpu_util_percent_val),
            "avg_ram_util_mb_val": float(avg_ram_util_mb_val),
        }
        for k in cc_keys_overall:
            fold_result[f"codecarbon_{k}"] = overall_stats.get(k)
        
            if k in overall_stats:
                all_overall_codecarbon_data[k].append(overall_stats[k])

        all_f1_scores.append(w_f1)
        all_accuracy_scores.append(acc)
        all_train_duration_data.append(train_dur_stats.get("duration"))
        all_val_duration_data.append(val_dur_stats.get("duration"))
        # Append psutil metrics to all_ lists for summary
        all_avg_cpu_util_percent_train.append(avg_cpu_util_percent_train)
        all_avg_ram_util_mb_train.append(avg_ram_util_mb_train)
        all_avg_cpu_util_percent_val.append(avg_cpu_util_percent_val)
        all_avg_ram_util_mb_val.append(avg_ram_util_mb_val) # Corrected typo here from `all_avg_ram_util_mb_val` to `avg_ram_util_mb_val`

        fold_results.append(fold_result)
        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(fold_result, f, indent=4)

    summary = {
        "f1_mean": float(np.mean(all_f1_scores)),
        "f1_std" : float(np.std(all_f1_scores)),
        "accuracy_mean": float(np.mean(all_accuracy_scores)),
        "accuracy_std": float(np.std(all_accuracy_scores)),
        "metadata": dict(cfg.experiment.metadata),
        "folds"   : fold_results,
    }

    if all_train_duration_data:
        summary["codecarbon_train_duration_mean"] = float(np.mean(all_train_duration_data))
        summary["codecarbon_train_duration_std"] = float(np.std(all_train_duration_data))
    if all_val_duration_data:
        summary["codecarbon_val_duration_mean"] = float(np.mean(all_val_duration_data))
        summary["codecarbon_val_duration_std"] = float(np.std(all_val_duration_data))

    # Add aggregated psutil metrics to the summary
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

    for k in all_overall_codecarbon_data:
        if all_overall_codecarbon_data[k]:
            if all(isinstance(val, (int, float)) for val in all_overall_codecarbon_data[k]):
                summary[f"codecarbon_{k}_mean"] = float(np.mean(all_overall_codecarbon_data[k]))
                summary[f"codecarbon_{k}_std"] = float(np.std(all_overall_codecarbon_data[k]))
            else:
                unique_values = list(set(all_overall_codecarbon_data[k]))
                if len(unique_values) == 1:
                    summary[f"codecarbon_{k}_common"] = unique_values[0]
                else:
                    summary[f"codecarbon_{k}_all"] = unique_values

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    print("\n== CV SUMMARY ==")
    print(json.dumps(summary, indent=4))


@hydra.main(version_base=None, config_path=str(Path(__file__).resolve().parents[2] / "config"), config_name="esc50")
def main(cfg: DictConfig):
    for name, meta in cfg.experiment.datasets.items():
        csv_path = Path(meta.csv)
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        print(f"\n=== DATASET {name.upper()} → outputs/{cfg.experiment.metadata.tag}_moe")
        run_cv_moe(str(csv_path), cfg)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()