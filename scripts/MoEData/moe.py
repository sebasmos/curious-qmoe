# 
# #!/usr/bin/env python
"""
K-Fold CV on embedding CSVs with CodeCarbon energy tracking

CUDA_VISIBLE_DEVICES=1 python moe.py \
    --config-path /home/sebastian/codes/repo_clean/QWave/config \
    --config-name esc50 \
    experiment.datasets.esc.normalization_type=standard \
    experiment.datasets.esc.csv=/home/sebastian/codes/data/ESC-50-master/VE_soundscapes/efficientnet_1536/esc-50.csv \
    experiment.device=cuda \
    experiment.metadata.tag=moe_deeprouter

# mac:
python moe_vanilla.py \
    --config-path /Users/sebasmos/Desktop/QWave/config \
    --config-name esc50 \
    experiment.datasets.esc.normalization_type=l2 \
    experiment.datasets.esc.csv=/Users/sebasmos/Documents/DATASETS/data_VE/ESC-50-master/VE_soundscapes/efficientnet_1536/esc-50.csv \
    experiment.device=mps \
    experiment.metadata.tag=delete
"""

from pathlib import Path
import os, sys
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import os, sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from codecarbon import EmissionsTracker
from memory_profiler import memory_usage
from QWave.datasets import EmbeddingAdaptDataset
from QWave.graphics import plot_multiclass_roc_curve, plot_losses
from QWave.models import ESCModel, reset_weights
import time # Import the time module
from QWave.utils import get_device

# --- NEW ROUTER CLASS ---
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

class MoEModel(nn.Module):
    def __init__(self, cfg, in_dim, num_classes):
        super().__init__()
        # Initialize router && experts
        self.num_experts = cfg.experiment.router.num_experts
        # Updated Router initialization
        self.router = Router(in_dim, cfg.experiment.router.hidden_dim, self.num_experts, cfg.experiment.router.dropout_prob)
        self.experts = nn.ModuleList([ESCModel(in_dim,num_classes, 
                                               hidden_sizes=cfg.experiment.model.hidden_sizes, 
                                               dropout_prob=cfg.experiment.model.dropout_prob) 
                                    for _ in range(self.num_experts)])
        self.num_classes = num_classes
        self.top_k = cfg.experiment.router.top_k


    def forward(self, x):
        # 1. Router scores and probabilities
        router_scores = self.router(x)
        router_probs = F.softmax(router_scores, dim=1)
        # 2. Select top-k experts based on router probabilities
        topk_vals, topk_indices = torch.topk(router_probs, self.top_k, dim=1)
        # 3. Compute outputs from selected experts
        outputs = torch.zeros(x.size(0), self.num_classes, device=x.device)
        for i in range(x.size(0)):
            out_sum = 0
            for k in range(self.top_k):
                idx = topk_indices[i, k].item()
                weight = topk_vals[i, k].item()
                expert_out = self.experts[idx](x[i].unsqueeze(0)).squeeze(0)
                out_sum += weight * expert_out
            outputs[i] = out_sum / self.top_k
        return outputs, router_probs

class MoEModelBatched(nn.Module):
    def __init__(self, cfg, in_dim, num_classes, num_experts=4, top_k=2):
        super().__init__()
        # Updated Router initialization
        self.router = Router(in_dim, cfg.experiment.router.hidden_dim, num_experts, cfg.experiment.model.dropout_prob)
        self.experts = nn.ModuleList([
            ESCModel(in_dim, num_classes, 
                     hidden_sizes=cfg.experiment.model.hidden_sizes, 
                     dropout_prob=cfg.experiment.model.dropout_prob)
            for _ in range(num_experts)
        ])
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balancing_alpha = cfg.experiment.model.get("load_balancing_alpha", 10e-3)
        self.load_balancing_enabled = cfg.experiment.model.get("load_balancing_enabled", True)

    def forward(self, x):
        B = x.size(0)
        
        router_scores = self.router(x)                      
        router_probs = F.softmax(router_scores, dim=1)    

        topk_vals, topk_indices = torch.topk(router_probs, self.top_k, dim=1)  

        outputs = torch.zeros(B, self.num_classes, device=x.device)

        load_balancing_loss = 0.0 

        if self.training and self.load_balancing_enabled: 
            load_balancing_loss = torch.sum(torch.mean(router_probs, dim=0) ** 2)

        for expert_idx in range(self.num_experts):
            mask = (topk_indices == expert_idx)  

            if not mask.any():
                continue

            example_indices, slot_indices = torch.nonzero(mask, as_tuple=True)

            x_selected = x[example_indices]  
            weight_selected = topk_vals[example_indices, slot_indices]  

            expert_output = self.experts[expert_idx](x_selected)  

            outputs[example_indices] += expert_output * weight_selected.unsqueeze(1)

        outputs /= self.top_k  
        return outputs, router_probs, load_balancing_loss
    

def train_moe_local(cfg,load_balancing, model, train_loader, val_loader, class_weights, in_dim, device, fold_dir, resume, ckpt_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.experiment.router.lr_moe_train)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_f1 = 0
    load_balancing_alpha = cfg.experiment.router.load_balancing_alpha
    
    train_losses, val_losses = [], []

    for epoch in range(cfg.experiment.model.epochs):
        model.train()
        total_loss = 0
        for embeddings, labels in train_loader:
            X, y = embeddings.to(device), labels.to(device)
            if load_balancing:
                outputs, router_probs, load_balancing_loss_term = model(X) # Unpack load_balancing_loss_term
                # Calculate the primary classification loss.
                classification_loss = criterion(outputs, y)
                # Combine the classification loss with the scaled load balancing loss.
                loss = classification_loss + load_balancing_alpha * load_balancing_loss_term                  
            else:
                outputs, _, _ = model(X) # Ensure model returns 3 values even if load_balancing is off, or adjust the model call
                loss = criterion(outputs, y)# classification loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for embeddings, labels in val_loader:
                X, y = embeddings.to(device), labels.to(device)
                outputs, _, _ = model(X) # Ensure model returns 3 values even if load_balancing is off, or adjust the model call
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        print(f"Epoch {epoch+1}/{cfg.experiment.model.epochs}, Train Loss: {total_loss:.4f}, Val F1: {f1:.4f}")
        val_losses.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), ckpt_path)
            best_state = (model.state_dict(), train_losses, val_losses, best_f1, all_labels, all_preds, [])

    return best_state

def _validate_moe_epoch(model, val_loader, criterion, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for embeddings, labels in val_loader:
            X, y = embeddings.to(device), labels.to(device)
            outputs, _, _ = model(X) # Ensure model returns 3 values
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())
    return 0, 0, all_labels, all_preds, all_probs

def _load_cc_csv(csv_path: Path) -> dict:
    if not csv_path.is_file():
        return {}
    df = pd.read_csv(csv_path)
    return df.iloc[-1].to_dict() if len(df) else {}

cc_metrics_to_track = [
    "duration", "emissions", "emissions_rate", "cpu_power", "gpu_power", "ram_power",
    "cpu_energy", "gpu_energy", "ram_energy", "energy_consumed",
    "cpu_count", "cpu_model", "gpu_count", "gpu_model", "ram_total_size"
]

@hydra.main(version_base=None, config_path="config", config_name="esc50")
def main(cfg: DictConfig):
    df_full = pd.read_csv(cfg.experiment.datasets.esc.csv)
    labels = df_full["class_id"].values
    df = df_full.drop(columns=["folder", "name", "label", "category"], errors="ignore")
    skf = StratifiedKFold(n_splits=cfg.experiment.cross_validation.n_splits, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = get_device(cfg)
    print(f"Final selected device: {device}\n")
    tag = cfg.experiment.metadata.tag
    out_dir = Path("outputs") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    all_final_f1_scores, all_final_accuracy_scores = [], []
    all_avg_ram_mb_train, all_avg_ram_mb_val = [], [] # Renamed for clarity with average
    all_train_cc_data_agg = {k: [] for k in cc_metrics_to_track}
    all_val_cc_data_agg = {k: [] for k in cc_metrics_to_track}
    all_training_durations = [] # New list to store training durations
    all_validation_durations = [] # New list to store validation durations
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n--- FOLD {fold+1}/{skf.n_splits} ---")
        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)

        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)

        normalization_type = cfg.experiment.datasets.esc.get("normalization_type", "raw")
        train_ds = EmbeddingAdaptDataset(df_train, normalization_type=normalization_type, scaler=None)
        fitted_scaler = train_ds.get_scaler()
        val_ds = EmbeddingAdaptDataset(df_val, normalization_type=normalization_type, scaler=fitted_scaler)

        train_ld = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_ld = DataLoader(val_ds, batch_size=16, shuffle=False)

        in_dim = train_ds.features.shape[1]
        num_classes = len(np.unique(labels))
        
        model = MoEModelBatched(cfg, in_dim, num_classes, cfg.experiment.router.num_experts, cfg.experiment.router.top_k).to(device)
        
        class_weights = torch.tensor(1.0 / np.bincount(train_ds.labels.numpy()), dtype=torch.float32).to(device)
        ckpt_path = fold_dir / "best_model.pth"

        # --- Training Phase Timing ---
        train_start_time = time.perf_counter()
        train_tracker = EmissionsTracker(project_name=f"{tag}_fold{fold}_train", output_dir=str(fold_dir), output_file="emissions_train.csv")
        train_tracker.start()
        load_balancing = cfg.experiment.get("load_balancing", True)# set as True by default
        mem_train_usage, (state_dict, train_losses, val_losses, best_f1, all_labels_best, all_preds_best, _) = \
            memory_usage((train_moe_local, (cfg,load_balancing, model, train_ld, val_ld, class_weights, in_dim, device, str(fold_dir), False, ckpt_path)), interval=0.1, retval=True)
        train_tracker.stop()
        avg_ram_mb_train = sum(mem_train_usage) / len(mem_train_usage) # Kept as average
        train_end_time = time.perf_counter()
        training_duration = train_end_time - train_start_time
        print(f"Manual Timer: Training for Fold {fold+1} took {training_duration:.2f} seconds.")
        all_training_durations.append(training_duration)
        

        # --- Clear CUDA cache between training and validation ---
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache emptied after training.")

        val_tracker = EmissionsTracker(project_name=f"{tag}_fold{fold}_val", output_dir=str(fold_dir), output_file="emissions_val.csv")
        val_tracker.start()

        # The model used here should also be MoEModelBatched for consistency if that's what's being trained
        final_model = MoEModelBatched(cfg,in_dim, num_classes, cfg.experiment.router.num_experts, cfg.experiment.router.top_k).to(device)
        final_model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # --- Validation Phase Timing ---
        val_start_time = time.perf_counter()

        mem_val_usage, (_, _, all_labels_final, all_preds_final, all_probs_final) = memory_usage((_validate_moe_epoch, (final_model, val_ld, nn.CrossEntropyLoss(), device)), interval=0.1, retval=True)
        avg_ram_mb_val = sum(mem_val_usage) / len(mem_val_usage) # Kept as average
        val_end_time = time.perf_counter()
        
        validation_duration = val_end_time - val_start_time
        print(f"Manual Timer: Validation for Fold {fold+1} took {validation_duration:.2f} seconds.")
        val_tracker.stop()
        
        all_validation_durations.append(validation_duration)
        

        train_stats = _load_cc_csv(fold_dir / "emissions_train.csv")
        val_stats = _load_cc_csv(fold_dir / "emissions_val.csv")
        final_accuracy = accuracy_score(all_labels_final, all_preds_final)
        final_f1_weighted = f1_score(all_labels_final, all_preds_final, average="weighted", zero_division=0)

        print(f"  Fold {fold}: Final Accuracy={final_accuracy:.4f} | Final weighted-F1={final_f1_weighted:.4f}")

        fold_result = {
            "best_f1": final_f1_weighted,
            "accuracy": final_accuracy,
            "avg_ram_mb_train": float(avg_ram_mb_train), # Using avg
            "avg_ram_mb_val": float(avg_ram_mb_val),     # Using avg
            "training_duration_seconds": float(training_duration),  # Add training duration
            "validation_duration_seconds": float(validation_duration) # Add validation duration
        }

        for k in cc_metrics_to_track:
            fold_result[f"train_{k}"] = train_stats.get(k, 0.0)
            all_train_cc_data_agg[k].append(train_stats.get(k, 0.0))
        for k in cc_metrics_to_track:
            fold_result[f"val_{k}"] = val_stats.get(k, 0.0)
            all_val_cc_data_agg[k].append(val_stats.get(k, 0.0))

        all_final_f1_scores.append(final_f1_weighted)
        all_final_accuracy_scores.append(final_accuracy)
        all_avg_ram_mb_train.append(avg_ram_mb_train) # Using avg
        all_avg_ram_mb_val.append(avg_ram_mb_val)     # Using avg

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
        "training_duration_mean_seconds": float(np.mean(all_training_durations)), # Add mean training duration
        "training_duration_std_seconds": float(np.std(all_training_durations)),   # Add std training duration
        "validation_duration_mean_seconds": float(np.mean(all_validation_durations)), # Add mean validation duration
        "validation_duration_std_seconds": float(np.std(all_validation_durations)),   # Add std validation duration
        "metadata": dict(cfg.experiment.metadata),
        "folds": fold_metrics
    }

    if all_avg_ram_mb_train:
        summary["avg_ram_mb_train_mean"] = float(np.mean(all_avg_ram_mb_train))
        summary["avg_ram_mb_train_std"] = float(np.std(all_avg_ram_mb_train))
    if all_avg_ram_mb_val:
        summary["avg_ram_mb_val_mean"] = float(np.mean(all_avg_ram_mb_val))
        summary["avg_ram_mb_val_std"] = float(np.std(all_avg_ram_mb_val))

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

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()