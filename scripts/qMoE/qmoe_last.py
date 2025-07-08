"""
bitnet158b, bitnet 

CUDA_VISIBLE_DEVICES=1 python qmoe_last.py \
  --config-path /home/sebastian/codes/repo_clean/QWave/config \
  --config-name esc50 \
  experiment.router.expert_quantizations=""[1,2,4,16]"" \
  experiment.router.num_experts=4 \
  experiment.datasets.esc.normalization_type=standard \
  experiment.datasets.esc.csv=/home/sebastian/codes/data/ESC-50-master/VE_soundscapes/efficientnet_1536/esc-50.csv \
  experiment.device=cuda \
  experiment.metadata.tag=qmoe_last

CUDA_VISIBLE_DEVICES=1 python qmoe_last.py \
  --config-path /home/sebastian/codes/repo_clean/QWave/config \
  --config-name esc50 \
  experiment.router.expert_quantizations=[bitnet,4,8,16] \
  experiment.router.num_experts=4 \
  experiment.datasets.esc.normalization_type=standard \
  experiment.datasets.esc.csv=/home/sebastian/codes/data/ESC-50-master/VE_soundscapes/efficientnet_1536/esc-50.csv \
  experiment.device=cuda \
  experiment.metadata.tag=qmoe_p7
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
from QWave.moe import Router, train_moe_local, _validate_moe_epoch
from QWave.utils import get_num_parameters
import time # Import the time module
from QWave.utils import get_device
from QWave.qmoe_layers import BitNetExpert158b, BitNetExpert, BitwisePopcountLinear, BitNetPopcountExpert, BitwiseLinear
from fvcore.nn import FlopCountAnalysis

class qMoEModelBatched(nn.Module):
    def __init__(self, cfg, in_dim, num_classes, num_experts=4, top_k=2):
        super().__init__()
        self.router = Router(in_dim, cfg.experiment.router.hidden_dim, num_experts, cfg.experiment.model.dropout_prob)
        
        expert_quantizations = cfg.experiment.router.expert_quantizations
        print(f"Initializing experts with quantizations: {expert_quantizations}")

        experts = []
        for bit_width in expert_quantizations:
            if bit_width == "bitnet158b":
                print("  -> Creating a BitNet1.58b expert.")
                experts.append(BitNetExpert158b(
                    in_dim,
                    num_classes,
                    hidden_sizes=cfg.experiment.model.hidden_sizes,
                    dropout_prob=cfg.experiment.model.dropout_prob
                ))
            elif bit_width == "bitnet":
                print("  -> Creating a standard BitNetExpert with ternary mode.")
                experts.append(BitNetExpert(
                    in_dim,
                    num_classes,
                    hidden_sizes=cfg.experiment.model.hidden_sizes,
                    dropout_prob=cfg.experiment.model.dropout_prob,
                    num_bits="bitnet"
                ))
            elif bit_width == "popcount": # Added popcount expert type
                print("  -> Creating a BitNetPopcountExpert.")
                experts.append(BitNetPopcountExpert(
                    in_dim,
                    num_classes,
                    hidden_sizes=cfg.experiment.model.hidden_sizes,
                    dropout_prob=cfg.experiment.model.dropout_prob
                ))
            else:
                print(f"  -> Creating a standard BitNetExpert with num_bits={bit_width}.")
                experts.append(BitNetExpert(
                    in_dim,
                    num_classes,
                    hidden_sizes=cfg.experiment.model.hidden_sizes,
                    dropout_prob=cfg.experiment.model.dropout_prob,
                    num_bits=int(bit_width)  # cast to int if needed
                ))
        self.experts = nn.ModuleList(experts)
        
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balancing_alpha = cfg.experiment.router.load_balancing_alpha
        

    def forward(self, x):
        B = x.size(0)
        
        router_scores = self.router(x)                      
        router_probs = F.softmax(router_scores, dim=1)    

        topk_vals, topk_indices = torch.topk(router_probs, self.top_k, dim=1)  

        outputs = torch.zeros(B, self.num_classes, device=x.device)

        load_balancing_loss = 0.0 

        if self.training: 
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
    device = get_device(cfg)
    print(f"Final selected device: {device}\n")
    tag = cfg.experiment.metadata.tag
    out_dir = Path("outputs") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    all_final_f1_scores, all_final_accuracy_scores = [], []
    all_avg_ram_mb_train, all_avg_ram_mb_val = [], []
    all_train_cc_data_agg = {k: [] for k in cc_metrics_to_track}
    all_val_cc_data_agg = {k: [] for k in cc_metrics_to_track}
    all_training_durations = []
    all_validation_durations = []
    all_training_flops = [] # Store training FLOPs per fold
    all_validation_flops = [] # Store validation FLOPs per fold
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n--- FOLD {fold+1}/{skf.n_splits} ---")
        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)

        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)

        normalization_type = cfg.experiment.datasets.esc.normalization_type
        train_ds = EmbeddingAdaptDataset(df_train, normalization_type=normalization_type, scaler=None)
        fitted_scaler = train_ds.get_scaler()
        val_ds = EmbeddingAdaptDataset(df_val, normalization_type=normalization_type, scaler=fitted_scaler)

        train_ld = DataLoader(train_ds, batch_size=cfg.experiment.model.batch_size, shuffle=True)
        val_ld = DataLoader(val_ds, batch_size=cfg.experiment.model.batch_size, shuffle=False)

        in_dim = train_ds.features.shape[1]
        num_classes = len(np.unique(labels))
        
        # Instantiate the qMoEModelBatched as per your original request
        model = qMoEModelBatched(cfg, in_dim, num_classes, cfg.experiment.router.num_experts, cfg.experiment.router.top_k).to(device)
        
        # Calculate parameter count for the MoE model
        moe_parameter_count = get_num_parameters(model)
        print(f"MoE Model Parameter Count: {moe_parameter_count}")

        class_weights = torch.tensor(1.0 / np.bincount(train_ds.labels.numpy()), dtype=torch.float32).to(device)
        ckpt_path = fold_dir / "best_model.pth"

        # Calculate FLOPs for training
        if len(train_ld) > 0:
            sample_input_train, _ = next(iter(train_ld))
            sample_input_train = sample_input_train.to(device)
            flops_train_analyzer = FlopCountAnalysis(model, sample_input_train)
            training_flops = flops_train_analyzer.total() * cfg.experiment.model.epochs * len(train_ld)
            print(f"Estimated Training FLOPs for Fold {fold+1}: {training_flops}")
            all_training_flops.append(training_flops)
        else:
            all_training_flops.append(0)

        train_start_time = time.perf_counter()
        train_tracker = EmissionsTracker(project_name=f"{tag}_fold{fold}_train", output_dir=str(fold_dir), output_file="emissions_train.csv")
        train_tracker.start()
        load_balancing = cfg.experiment.router.load_balancing # Use load balancing from config
        mem_train_usage, (state_dict, train_losses, val_losses, best_f1, all_labels_best, all_preds_best, _) = \
            memory_usage((train_moe_local, (cfg, load_balancing, model, train_ld, val_ld, class_weights, in_dim, device, str(fold_dir), False, ckpt_path)), interval=0.1, retval=True)
        
        train_tracker.stop()
        avg_ram_mb_train = sum(mem_train_usage) / len(mem_train_usage)
        train_end_time = time.perf_counter()
        training_duration = train_end_time - train_start_time
        print(f"Manual Timer: Training for Fold {fold+1} took {training_duration:.2f} seconds.")
        all_training_durations.append(training_duration)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache emptied after training.")

        val_tracker = EmissionsTracker(project_name=f"{tag}_fold{fold}_val", output_dir=str(fold_dir), output_file="emissions_val.csv")
        val_tracker.start()

        final_model = qMoEModelBatched(cfg,in_dim, num_classes, cfg.experiment.router.num_experts, cfg.experiment.router.top_k).to(device)
        final_model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # Calculate FLOPs for validation
        if len(val_ld) > 0:
            sample_input_val, _ = next(iter(val_ld))
            sample_input_val = sample_input_val.to(device)
            flops_val_analyzer = FlopCountAnalysis(final_model, sample_input_val)
            validation_flops = flops_val_analyzer.total() * len(val_ld)
            print(f"Estimated Validation FLOPs for Fold {fold+1}: {validation_flops}")
            all_validation_flops.append(validation_flops)
        else:
            all_validation_flops.append(0)


        val_start_time = time.perf_counter()

        # Capture latency directly from _validate_moe_epoch
        mem_val_usage, (_, _, all_labels_final, all_preds_final, all_probs_final) = \
            memory_usage((_validate_moe_epoch, (final_model, val_ld, nn.CrossEntropyLoss(), device)), interval=0.1, retval=True)
        
        val_end_time = time.perf_counter()
        validation_duration = val_end_time - val_start_time
        avg_ram_mb_val = sum(mem_val_usage) / len(mem_val_usage)
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
            "avg_ram_mb_train": float(avg_ram_mb_train),
            "avg_ram_mb_val": float(avg_ram_mb_val), # Peak memory (RAM) during validation
            "training_duration_seconds": float(training_duration),
            "validation_duration_seconds": float(validation_duration),
            "parameter_count": int(moe_parameter_count), # Parameter count
            "training_flops": float(training_flops),
            "validation_flops": float(validation_flops)
        }

        for k in cc_metrics_to_track:
            fold_result[f"train_{k}"] = train_stats.get(k, 0.0)
            all_train_cc_data_agg[k].append(train_stats.get(k, 0.0))
        for k in cc_metrics_to_track:
            fold_result[f"val_{k}"] = val_stats.get(k, 0.0)
            all_val_cc_data_agg[k].append(val_stats.get(k, 0.0))

        all_final_f1_scores.append(final_f1_weighted)
        all_final_accuracy_scores.append(final_accuracy)
        all_avg_ram_mb_train.append(avg_ram_mb_train)
        all_avg_ram_mb_val.append(avg_ram_mb_val)

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
        "training_duration_mean_seconds": float(np.mean(all_training_durations)),
        "training_duration_std_seconds": float(np.std(all_training_durations)),
        "validation_duration_mean_seconds": float(np.mean(all_validation_durations)),
        "validation_duration_std_seconds": float(np.std(all_validation_durations)),
        "parameter_count": int(moe_parameter_count), # Parameter count (same for all folds of this MoE config)
        "training_flops_mean": float(np.mean(all_training_flops)) if all_training_flops else 0.0,
        "training_flops_std": float(np.std(all_training_flops)) if all_training_flops else 0.0,
        "validation_flops_mean": float(np.mean(all_validation_flops)) if all_validation_flops else 0.0,
        "validation_flops_std": float(np.std(all_validation_flops)) if all_validation_flops else 0.0,
        "metadata": dict(cfg.experiment.metadata),
        "folds": fold_metrics
    }

    if all_avg_ram_mb_train:
        summary["avg_ram_mb_train_mean"] = float(np.mean(all_avg_ram_mb_train))
        summary["avg_ram_mb_train_std"] = float(np.std(all_avg_ram_mb_train))
    if all_avg_ram_mb_val:
        summary["avg_ram_mb_val_mean"] = float(np.mean(all_avg_ram_mb_val)) # Peak RAM mean
        summary["avg_ram_mb_val_std"] = float(np.std(all_avg_ram_mb_val))   # Peak RAM std

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
    
    filtered_csv_path = out_dir / "summary_filtered.csv"
    filtered_data = {
        "name": summary.get("metadata", {}).get("tag", "unknown"),
        "parameter_count": summary.get("parameter_count"),
        "accuracy_mean": summary.get("accuracy_mean"),
        "f1_mean": summary.get("f1_mean"),
        "avg_ram_gb_val_mean": summary.get("avg_ram_mb_val_mean", 0.0) / 1024,
        "training_duration_mean_seconds": summary.get("training_duration_mean_seconds"),
        "validation_duration_mean_seconds": summary.get("validation_duration_mean_seconds"),
        "train_energy_consumed_mean": summary.get("train_energy_consumed_mean"),
        "val_energy_consumed_mean": summary.get("val_energy_consumed_mean"),
        "train_emissions_mean": summary.get("train_emissions_mean"),
        "val_emissions_mean": summary.get("val_emissions_mean"),
        "train_gpu_power_mean": summary.get("train_gpu_power_mean"),
        "val_gpu_power_mean": summary.get("val_gpu_power_mean"),
        "training_flops_mean": summary.get("training_flops_mean"),
        "validation_flops_mean": summary.get("validation_flops_mean")
    }
    pd.DataFrame([filtered_data]).to_csv(filtered_csv_path, index=False)
    print(f"\nFiltered summary saved to {filtered_csv_path}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()