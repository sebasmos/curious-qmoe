#!/usr/bin/env python
"""
K-Fold Mixture-of-Experts (Multi-class Experts & Router)

Cross-validation version of the MoE pipeline. Keeps the same folder/metrics
conventions as the original K-Fold CV on embedding CSVs reference.
This version uses a two-stage Experts -> Router approach where:

Key points
----------
- Expert phase – Multiple `ESCModel` instances, each trained as a **full multi-class classifier**
                 on the training data within the fold.
                 Each expert will output probabilities for ALL N_classes.
- Router phase – A multi-layer perceptron (MLP) router trained with CrossEntropyLoss
                 to combine the outputs from all multi-class experts.
- Metrics – Includes training/inference duration, CodeCarbon energy metrics,
            **CPU utilization percentage**, and **RAM utilization in MB** per fold and in summary.

python run_trainer_MoEData_CV.py  --config-name=esc50 \
  experiment.datasets.esc.normalization_type=standard \
  experiment.datasets.esc.csv=/Users/sebasmos/Documents/DATASETS/data_VE/ESC-50-master/VE_soundscapes/efficientnet_1536/esc-50.csv \
  experiment.device=mps \
  experiment.metadata.tag="EfficientNet_esc50MoEData_CV_standard_rawoutput"

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
from memory_profiler import memory_usage

from QWave.datasets import EmbeddingAdaptDataset
from QWave.models import ESCModel, reset_weights
from QWave.train_utils import train_pytorch_local
from QWave.graphics import plot_multiclass_roc_curve
from QWave.utils import get_device


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


def _expert_output_full_probs(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            probs = torch.softmax(model(xb), dim=1).cpu()
            all_probs.append(probs)
    return torch.cat(all_probs, dim=0)


def _train_router(router: Router, X: torch.Tensor, y: np.ndarray, device: torch.device,
                   *, epochs: int, lr: float, batch_size: int, weight_decay: float = 0.0):
    ds = TensorDataset(X, torch.from_numpy(y).long())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    crit = nn.CrossEntropyLoss()
    opt  = torch.optim.Adam(router.parameters(), lr=lr, weight_decay=weight_decay)
    router.to(device)
    router.train()
    for ep in range(1, epochs+1):
        run = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad();  loss = crit(router(xb), yb);  loss.backward();  opt.step()
            run += loss.item() * len(xb)
        if ep % 15 == 0:
            print(f"    Router epoch {ep:3d} | CrossEntropyLoss: {run/len(dl.dataset):.4f}")


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

# --- Helper functions for memory_usage ---
def _train_experts_phase_wrapper(cfg, df_tr_orig, df_va_orig, y_tr_orig, y_va_orig, n_classes, cfg_experiment, device, fold_dir):
    """Encapsulates the entire expert training loop for memory profiling."""
    experts: List[nn.Module] = []
    for cls in range(n_classes):

        normalization_type = cfg.experiment.datasets.esc.get("normalization_type", "raw") # Default to "raw" if not specified

        tr_ds = EmbeddingAdaptDataset(df_tr_orig, normalization_type=normalization_type, scaler=None)
        
        fitted_scaler = tr_ds.get_scaler()
        
        va_ds = EmbeddingAdaptDataset(df_va_orig,normalization_type=normalization_type, scaler=fitted_scaler)
                              
        tr_ld = DataLoader(tr_ds, batch_size=cfg_experiment.model.batch_size, shuffle=True)
        
        va_ld = DataLoader(va_ds, batch_size=cfg_experiment.model.batch_size, shuffle=False)

        class_w_vals = np.bincount(y_tr_orig)
        
        class_w = torch.tensor(1.0 / class_w_vals, dtype=torch.float32).to(device)
        
        in_dim  = tr_ds.features.shape[1]
        
        model   = ESCModel(in_dim, n_classes, # Multi-class expert
                            hidden_sizes=cfg_experiment.model.hidden_sizes,
                            dropout_prob=cfg_experiment.model.dropout_prob).to(device)
        model.apply(reset_weights)

        exp_dir = fold_dir / f"expert_{cls}"; exp_dir.mkdir(exist_ok=True)
        train_pytorch_local(
            args=cfg_experiment, model=model, train_loader=tr_ld, val_loader=va_ld,
            class_weights=class_w, num_columns=in_dim, device=device,
            fold_folder=str(exp_dir), resume_checkpoint=False,
            checkpoint_path=str(exp_dir / "best.pth"),
        )
        experts.append(model.eval()) # Append expert in evaluation mode
    return experts

def _train_router_phase_wrapper(router, X_tr, y_tr, device, router_cfg):
    """Encapsulates the router training for memory profiling."""
    _train_router(router, X_tr, y_tr, device,
                  epochs=router_cfg.get("epochs", 75),
                  lr=router_cfg.get("lr", 2e-3),
                  batch_size=router_cfg.get("batch_size", 256),
                  weight_decay=router_cfg.get("weight_decay", 0.0))
    return router # Return router instance

def _run_router_inference_phase_wrapper(router, X_va, device):
    """Encapsulates the router inference/validation for memory profiling."""
    router.eval()
    with torch.no_grad():
        logits_va = router(X_va.to(device))
        probs_va  = torch.softmax(logits_va, dim=1).cpu().numpy()
        y_pred    = probs_va.argmax(axis=1)
    return probs_va, y_pred
# --- End Helper functions ---


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
    
    # NEW: Lists to store peak RAM utilization in MB for each phase
    all_max_ram_mb_experts_training = []
    all_max_ram_mb_router_training = []
    all_max_ram_mb_router_inference = []

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

        df_tr, df_va = df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)
        y_tr, y_va   = labels[tr_idx], labels[va_idx]

        # --- PHASE 1: Expert Training (Memory Profiled) ---
        train_duration_tracker = EmissionsTracker(
            project_name=f"{tag}_fold{fold}_train_experts_duration",
            output_dir=str(fold_dir),
            output_file="emissions_train_duration.csv",
            save_to_file=True,
        )
        train_duration_tracker.start()
        
        print("  == PHASE 1: training experts ==")
        # Use memory_usage to profile the entire expert training phase
        mem_experts_training_usage, experts = memory_usage(
            (_train_experts_phase_wrapper, (cfg, df_tr, df_va, y_tr, y_va, n_classes, cfg.experiment, device, fold_dir)),
            interval=0.1, 
            retval=True
        )
        max_ram_mb_experts_training = sum(mem_experts_training_usage) / len(mem_experts_training_usage)
    
        train_duration_tracker.stop() 
        # --- End PHASE 1 ---

        # Collect expert outputs for router

        normalization_type = cfg.experiment.datasets.esc.get("normalization_type", "raw") # Default to "raw" if not specified

        tr_full_ds = EmbeddingAdaptDataset(df_tr, normalization_type=normalization_type, scaler=None)
        # If using 'standard' or 'min_max' scaling, get the fitted scaler from the training dataset
        fitted_scaler = tr_full_ds.get_scaler()
        
        va_full_ds = EmbeddingAdaptDataset(df_va,normalization_type=normalization_type, scaler=fitted_scaler)
                            
        tr_ld_full = DataLoader(tr_full_ds, batch_size=cfg.experiment.model.batch_size)
        
        va_ld_full = DataLoader(va_full_ds, batch_size=cfg.experiment.model.batch_size)
        
        collect_expert_outputs = lambda ld: torch.cat([_expert_output_full_probs(exp, ld, device) for exp in experts], dim=1)
        X_tr, X_va = collect_expert_outputs(tr_ld_full), collect_expert_outputs(va_ld_full)

        # --- PHASE 2: Router Training (Memory Profiled) ---
        print("  == PHASE 2: training router ==")
        router_cfg = getattr(cfg, "router", {})
        
        router = Router(
            input_dim=n_classes * n_classes,
            hidden_dim=router_cfg.get("hidden_dim", 128),
            output_dim=n_classes,
            drop_prob=router_cfg.get("drop_prob", 0.2)
        )
        
        # Use memory_usage to profile the router training phase
        mem_router_training_usage, _ = memory_usage(
            (_train_router_phase_wrapper, (router, X_tr, y_tr, device, router_cfg)),
            interval=0.1,
            retval=True
        )
        max_ram_mb_router_training = sum(mem_router_training_usage) / len(mem_router_training_usage)
        # --- End PHASE 2 ---
        
        # --- PHASE 3: Router Inference/Validation (Memory Profiled) ---
        val_duration_tracker = EmissionsTracker(
            project_name=f"{tag}_fold{fold}_router_val_duration",
            output_dir=str(fold_dir),
            output_file="emissions_val_duration.csv",
            save_to_file=True,
        )
        val_duration_tracker.start()
        
        # Use memory_usage to profile the router inference phase
        mem_router_inference_usage, (probs_va, y_pred) = memory_usage(
            (_run_router_inference_phase_wrapper, (router, X_va, device)),
            interval=0.1,
            retval=True
        )
        max_ram_mb_router_inference =  sum(mem_router_inference_usage) / len(mem_router_inference_usage)
       
        val_duration_tracker.stop() 
        # --- End PHASE 3 ---

        overall_tracker.stop() 
        
        overall_stats = _load_cc_csv(fold_dir / "emissions_overall.csv")
        train_dur_stats = _load_cc_csv(fold_dir / "emissions_train_duration.csv")
        val_dur_stats = _load_cc_csv(fold_dir / "emissions_val_duration.csv")

        acc  = accuracy_score(y_va, y_pred)
        w_f1 = f1_score(y_va, y_pred, average="weighted", zero_division=0)
        print(f"  Fold {fold}: Accuracy={acc:.4f} | weighted-F1={w_f1:.4f}")

        plot_multiclass_roc_curve(y_va, probs_va, EXPERIMENT_NAME=str(fold_dir))

        fold_result = {
            "best_f1": w_f1,
            "accuracy": acc,
            "train_duration": train_dur_stats.get("duration"), 
            "val_duration": val_dur_stats.get("duration"),     
            "duration": overall_stats.get("duration"),         
            # NEW: Add peak RAM metrics to fold results
            "avg_ram_mb_experts_training": float(max_ram_mb_experts_training),
            "avg_ram_mb_router_training": float(max_ram_mb_router_training),
            "avg_ram_mb_router_inference": float(max_ram_mb_router_inference),
        }
        for k in cc_keys_overall:
            fold_result[f"codecarbon_{k}"] = overall_stats.get(k)
        
            if k in overall_stats:
                all_overall_codecarbon_data[k].append(overall_stats[k])

        all_f1_scores.append(w_f1)
        all_accuracy_scores.append(acc)
        all_train_duration_data.append(train_dur_stats.get("duration"))
        all_val_duration_data.append(val_dur_stats.get("duration"))
        
        # NEW: Append peak RAM metrics to all_ lists for summary
        all_max_ram_mb_experts_training.append(max_ram_mb_experts_training)
        all_max_ram_mb_router_training.append(max_ram_mb_router_training)
        all_max_ram_mb_router_inference.append(max_ram_mb_router_inference)

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
        summary["train_duration_mean"] = float(np.mean(all_train_duration_data))
        summary["train_duration_std"] = float(np.std(all_train_duration_data))
    if all_val_duration_data:
        summary["val_duration_mean"] = float(np.mean(all_val_duration_data))
        summary["val_duration_std"] = float(np.std(all_val_duration_data))

    # NEW: Add aggregated peak RAM metrics to the summary
    if all_max_ram_mb_experts_training:
        summary["avg_ram_mb_experts_training_mean"] = float(np.mean(all_max_ram_mb_experts_training))
        summary["avg_ram_mb_experts_training_std"] = float(np.std(all_max_ram_mb_experts_training))
    if all_max_ram_mb_router_training:
        summary["avg_ram_mb_router_training_mean"] = float(np.mean(all_max_ram_mb_router_training))
        summary["avg_ram_mb_router_training_std"] = float(np.std(all_max_ram_mb_router_training))
    if all_max_ram_mb_router_inference:
        summary["avg_ram_mb_router_inference_mean"] = float(np.mean(all_max_ram_mb_router_inference))
        summary["avg_ram_mb_router_inference_std"] = float(np.std(all_max_ram_mb_router_inference))

    for k in all_overall_codecarbon_data:
        if all_overall_codecarbon_data[k]:
            if all(isinstance(val, (int, float)) for val in all_overall_codecarbon_data[k]):
                summary[f"{k}_mean"] = float(np.mean(all_overall_codecarbon_data[k]))
                summary[f"{k}_std"] = float(np.std(all_overall_codecarbon_data[k]))
            else:
                unique_values = list(set(all_overall_codecarbon_data[k]))
                if len(unique_values) == 1:
                    summary[f"{k}_common"] = unique_values[0]
                else:
                    summary[f"{k}_all"] = unique_values

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
        print(f"\n=== DATASET {name.upper()} → outputs/{cfg.experiment.metadata.tag}_moe")
        run_cv_moe(str(csv_path), cfg)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()