"""
bitnet158b, bitnet 

  CUDA_VISIBLE_DEVICES=1 python qmoe.py \
  --config-path /home/sebastian/codes/repo_clean/QWave/config \
  --config-name esc50 \
  experiment.router.expert_quantizations=""[esc,esc,esc,esc]"" \
  experiment.router.num_experts=4 \
  experiment.datasets.esc.normalization_type=standard \
  experiment.datasets.esc.csv=/home/sebastian/codes/data/ESC-50-master/VE_soundscapes/efficientnet_1536/esc-50.csv \
  experiment.device=cpu \
  experiment.metadata.tag=p1

  CUDA_VISIBLE_DEVICES=1 python qmoe.py \
  --config-path /home/sebastian/codes/repo_clean/QWave/config \
  --config-name esc50 \
  experiment.router.expert_quantizations=""[qesc,qesc,qesc,qesc]"" \
  experiment.router.num_experts=4 \
  experiment.datasets.esc.normalization_type=standard \
  experiment.datasets.esc.csv=/home/sebastian/codes/data/ESC-50-master/VE_soundscapes/efficientnet_1536/esc-50.csv \
  experiment.device=cpu \
  experiment.metadata.tag=q1

  CUDA_VISIBLE_DEVICES=1 python qmoe.py \
  --config-path /home/sebastian/codes/repo_clean/QWave/config \
  --config-name esc50 \
  experiment.router.expert_quantizations=""[1,2,4,16]"" \
  experiment.router.num_experts=4 \
  experiment.datasets.esc.normalization_type=standard \
  experiment.datasets.esc.csv=/home/sebastian/codes/data/ESC-50-master/VE_soundscapes/efficientnet_1536/esc-50.csv \
  experiment.device=cpu \
  experiment.metadata.tag=p2

CUDA_VISIBLE_DEVICES=1 python qmoe.py \
  --config-path /home/sebastian/codes/repo_clean/QWave/config \
  --config-name esc50 \
  experiment.router.expert_quantizations=[popcount,popcount,popcount,popcount] \
  experiment.router.num_experts=4 \
  experiment.datasets.esc.normalization_type=standard \
  experiment.datasets.esc.csv=/home/sebastian/codes/data/ESC-50-master/VE_soundscapes/efficientnet_1536/esc-50.csv \
  experiment.device=cpu \
  experiment.metadata.tag=p3

mac: 
python qmoe.py \
  --config-path /Users/sebasmos/Desktop/QWave/config \
  --config-name esc50 \
  "experiment.router.expert_quantizations=[esc,esc,esc,esc]" \
  experiment.router.num_experts=4 \
  experiment.datasets.esc.normalization_type=standard \
  experiment.datasets.esc.csv=/Users/sebasmos/Documents/DATASETS/data_VE/ESC-50-master/VE_soundscapes/efficientnet_1536/esc-50.csv \
  experiment.device=cpu \
  experiment.metadata.tag=p1

python qmoe.py \
  --config-path /Users/sebasmos/Desktop/QWave/config \
  --config-name esc50 \
  "experiment.router.expert_quantizations=[qesc,qesc,qesc,qesc,qesc,qesc,qesc,qesc,qesc,qesc]" \
  experiment.router.num_experts=10 \
  experiment.datasets.esc.normalization_type=standard \
  experiment.datasets.esc.csv=/Users/sebasmos/Documents/DATASETS/data_VE/ESC-50-master/VE_soundscapes/efficientnet_1536/esc-50.csv \
  experiment.device=cpu \
  experiment.metadata.tag=q2

python qmoe.py \
  --config-path /Users/sebasmos/Desktop/QWave/config \
  --config-name esc50 \
  "experiment.router.expert_quantizations=[1,2,4,16]"\
  experiment.router.num_experts=4 \
  experiment.datasets.esc.normalization_type=standard \
  experiment.datasets.esc.csv=/Users/sebasmos/Documents/DATASETS/data_VE/ESC-50-master/VE_soundscapes/efficientnet_1536/esc-50.csv \
  experiment.device=cpu \
  experiment.metadata.tag=p2
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
from torch.utils.data import DataLoader
from codecarbon import EmissionsTracker
from memory_profiler import memory_usage
from QWave.datasets import EmbeddingAdaptDataset
from QWave.graphics import plot_multiclass_roc_curve, plot_losses
from QWave.models import ESCModel
from QWave.moe import train_moe_local, _validate_moe_epoch, qMoEModelBatched
from QWave.utils import get_num_parameters, get_device
from QWave.memory import print_size_of_model, _load_cc_csv
import platform
import time 
from fvcore.nn import FlopCountAnalysis


cc_metrics_to_track = [
    "duration", "emissions", "emissions_rate", "cpu_power", "gpu_power", "ram_power",
    "cpu_energy", "gpu_energy", "ram_energy", "energy_consumed",
    "cpu_count", "cpu_model", "gpu_count", "gpu_model", "ram_total_size"
]

@hydra.main(version_base=None, config_path="config", config_name="esc50")
def main(cfg: DictConfig):
    warnings.filterwarnings("ignore", category=UserWarning)
    df_full = pd.read_csv(cfg.experiment.datasets.esc.csv)
    if df_full.empty:
        print(f"[ERROR] Input CSV {cfg.experiment.datasets.esc.csv} is empty. Exiting.")
        return

    labels = df_full["class_id"].values
    df = df_full.drop(columns=["folder", "name", "label", "category"], errors="ignore")
    skf = StratifiedKFold(n_splits=cfg.experiment.cross_validation.n_splits, shuffle=True, random_state=42)
    device = get_device(cfg)
    print(f"Final selected device: {device}\n")
    tag = cfg.experiment.metadata.tag
    out_dir = Path("outputs") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    all_final_f1_scores, all_final_accuracy_scores = [], []
    all_max_ram_mb_train, all_max_ram_mb_val = [], [] 
    all_model_size_mb = []
    all_train_cc_data_agg = {k: [] for k in cc_metrics_to_track}
    all_val_cc_data_agg = {k: [] for k in cc_metrics_to_track}
    all_training_durations = []
    all_validation_durations = []
    all_training_flops = [] # Store training FLOPs per fold
    all_validation_flops = [] # Store validation FLOPs per fold
    fold_metrics = []
    in_dim_placeholder = df.shape[1] - (df_full.shape[1] - df.shape[1]) # Adjust for dropped columns if they were features
    if "class_id" in df.columns: # Adjust in_dim if class_id is still in df
        in_dim_placeholder -= 1
    if in_dim_placeholder <= 0: # Ensure in_dim is valid after drops
        print("[ERROR] Input dimension (in_dim) is not positive after dropping columns. Check your DataFrame structure.")
        return
    num_classes_placeholder = len(np.unique(labels))
    
    dummy_model_for_params = qMoEModelBatched(cfg, in_dim_placeholder, num_classes_placeholder, cfg.experiment.router.num_experts, cfg.experiment.router.top_k).to(device)
    moe_parameter_count = get_num_parameters(dummy_model_for_params)
    print(f"MoE Model Parameter Count: {moe_parameter_count:,}")
    del dummy_model_for_params # Free up memory

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
        # num_classes already determined globally, but can re-confirm for safety
        num_classes = len(np.unique(labels)) 
        
        # Instantiate the qMoEModelBatched as per your original request
        model = qMoEModelBatched(cfg, in_dim, num_classes, cfg.experiment.router.num_experts, cfg.experiment.router.top_k).to(device)
        # if model is quantized QAT
        # print("Expert 0 mode:", model.experts[0].net[0].num_bits) 
        # print("Expert 1 mode:", model.experts[1].net[0].num_bits) 
        class_weights = torch.tensor(1.0 / np.bincount(train_ds.labels.numpy()), dtype=torch.float32).to(device)
        ckpt_path = fold_dir / "best_model.pth"

        # Calculate FLOPs for training
        if len(train_ld) > 0:
            sample_input_train, _ = next(iter(train_ld))
            sample_input_train = sample_input_train.to(device)
            flops_train_analyzer = FlopCountAnalysis(model, sample_input_train)
            training_flops = flops_train_analyzer.total() * cfg.experiment.model.epochs * len(train_ld)
            print(f"Estimated Training FLOPs for Fold {fold+1}: {training_flops:,}")
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
        
        max_ram_mb_train = float(np.max(mem_train_usage)) if mem_train_usage else 0.0
        
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
            print(f"Estimated Validation FLOPs for Fold {fold+1}: {validation_flops:,}")
            all_validation_flops.append(validation_flops)
        else:
            all_validation_flops.append(0)

        print_size_of_model(final_model, "Original_Model")
        
        expert_quantizations = cfg.experiment.router.expert_quantizations

        if "qesc" in expert_quantizations:
            print("  -> Using qESC expert for validation.")
            if platform.system() == "Darwin":  # macOS
                    torch.backends.quantized.engine = 'qnnpack'
            else:
                    torch.backends.quantized.engine = 'fbgemm'
            
            # ----- quantise experts once -------------
            for i, exp in enumerate(final_model.experts):
                # Check if expert i should be quantized based on its position in the list
                # This assumes expert_quantizations is a list corresponding to expert indices
                if i < len(expert_quantizations) and "qesc" in expert_quantizations[i]:
                    if isinstance(exp, ESCModel):
                        final_model.experts[i] = torch.quantization.quantize_dynamic(
                            exp.cpu(), {nn.Linear}, dtype=torch.qint8
                        )
                        final_model.experts[i].to(device) # Move quantized expert back to device
                    else:
                        warnings.warn(f"Expert at index {i} is not an ESCModel, skipping qesc quantization.")
                
            final_model.eval()
            for param in final_model.parameters():
                    param.requires_grad_(False)
            val_start_time = time.perf_counter()
            mem_val_usage, (_, _, all_labels_final, all_preds_final, all_probs_final) = \
                memory_usage((_validate_moe_epoch, (final_model, val_ld, nn.CrossEntropyLoss(), device)), interval=0.01, retval=True)
            dur_val = time.perf_counter() - val_start_time
            model_size = print_size_of_model(final_model, "Quantized model")
            
        else: 
            val_start_time = time.perf_counter()
            mem_val_usage, (_, _, all_labels_final, all_preds_final, all_probs_final) = \
                memory_usage((_validate_moe_epoch, (final_model, val_ld, nn.CrossEntropyLoss(), device)), interval=0.01, retval=True)
            dur_val = time.perf_counter() - val_start_time
            model_size =  print_size_of_model(final_model, "Model after validation")
        
        max_ram_mb_val = float(np.max(mem_val_usage)) if mem_val_usage else 0.0
        print("Model size: ", model_size)
        val_tracker.stop()
        
        all_validation_durations.append(dur_val)

        train_stats = _load_cc_csv(fold_dir / "emissions_train.csv")
        val_stats = _load_cc_csv(fold_dir / "emissions_val.csv")
        final_accuracy = accuracy_score(all_labels_final, all_preds_final)
        final_f1_weighted = f1_score(all_labels_final, all_preds_final, average="weighted", zero_division=0)

        print(f"  Fold {fold}: Final Accuracy={final_accuracy:.4f} | Final weighted-F1={final_f1_weighted:.4f}")

        fold_result = {
            "best_f1": final_f1_weighted,
            "accuracy": final_accuracy,
            "model_size": model_size / 1e6, # Model size in MB
            "max_ram_mb_train": float(max_ram_mb_train),
            "max_ram_mb_val": float(max_ram_mb_val), # Peak memory (RAM) during validation
            "training_duration_seconds": float(training_duration),
            "validation_duration_seconds": float(dur_val),
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

        all_max_ram_mb_train.append(max_ram_mb_train)
        all_max_ram_mb_val.append(max_ram_mb_val)
        all_model_size_mb.append(model_size)

        print(f"    Fold {fold} Results:")
        print(f"        F1 Score (Weighted): {final_f1_weighted:.4f}")
        print(f"        Validation RAM Usage (Max MB): {max_ram_mb_val:.2f}")
        print(f"        Validation Inference Time (s): {dur_val:.4f}")


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
        "model_size_mb": float(np.mean([m["model_size"] for m in fold_metrics])),
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

    if all_max_ram_mb_train:
        summary["max_ram_mb_train_mean"] = float(np.mean(all_max_ram_mb_train))
        summary["max_ram_mb_train_std"] = float(np.std(all_max_ram_mb_train))
    if all_max_ram_mb_val:
        summary["max_ram_mb_val_mean"] = float(np.mean(all_max_ram_mb_val)) # Peak RAM mean
        summary["max_ram_mb_val_std"] = float(np.std(all_max_ram_mb_val))   # Peak RAM std

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
        "name": summary.get("metadata", {}).get("tag", "unknown"), # Kept as tag, as explained above
        "Param. Count": summary.get("parameter_count"),
        "Acc (Mean)": summary.get("accuracy_mean"),
        "F1 (Mean)": summary.get("f1_mean"),
        "Model size (MB)": summary.get("model_size_mb"),
        "Infer. RAM (GB)": summary.get("max_ram_mb_val_mean", 0.0) / 1024,
        "Train. Runtime (s)": summary.get("training_duration_mean_seconds"),
        "Latency (s)": summary.get("validation_duration_mean_seconds"),
        "Energy-train": summary.get("train_energy_consumed_mean"),
        "Energy-val": summary.get("val_energy_consumed_mean"),
        "Emissions-train": summary.get("train_emissions_mean"),
        "Emissions-val": summary.get("val_emissions_mean"),
        "GPU-Power-train": summary.get("train_gpu_power_mean"),
        "GPU-Power-val": summary.get("val_gpu_power_mean"),
        "FLOPs-train": summary.get("training_flops_mean"),
        "FLOPs-val": summary.get("validation_flops_mean")
    }
    pd.DataFrame([filtered_data]).to_csv(filtered_csv_path, index=False)
    print(f"\nFiltered summary saved to {filtered_csv_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()