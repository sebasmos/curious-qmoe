# run_three_models.py
#!/usr/bin/env python
"""
Train **single-network** versions of:
  1. BitNet        (ternary)        – key: "bitnet"
  2. BitNet-158B   (ternary v2)     – key: "bitnet158b"
  3. ESCModel      (full-precision) – key: "esc"

Optional extras: numeric quantisations **1,2,4,8,16** (k-bit) are supported
with keys "1", "2", … "16".

Select which to run via
    experiment.models_to_run="[bitnet,bitnet158b,esc]"

The script keeps *all* variable names, CSV/JSON summaries, plots, CodeCarbon,
FLOP count, memory profiling, etc., but removes every MoE-specific element.

CUDA_VISIBLE_DEVICES=1 python benchmark_baselines.py \
    --config-path /home/sebastian/codes/repo_clean/QWave/config \
    --config-name esc50 \
    experiment.datasets.esc.normalization_type=standard \
    experiment.datasets.esc.csv=/home/sebastian/codes/data/ESC-50-master/VE_soundscapes/efficientnet_1536/esc-50.csv \
    experiment.device=cuda \
    experiment.models_to_run="['1','2','4','8','16',esc]" \
    experiment.metadata.tag=benchmark_baselines

python benchmark_baselines.py \
    --config-path /home/sebastian/codes/repo_clean/QWave/config \
    --config-name esc50 \
    experiment.datasets.esc.normalization_type=standard \
    experiment.datasets.esc.csv=/home/sebastian/codes/data/ESC-50-master/VE_soundscapes/efficientnet_1536/esc-50.csv \
    experiment.device=cpu \
    experiment.models_to_run="[qesc]" \
    experiment.metadata.tag=benchmark_baselines



Mac:
CUDA_VISIBLE_DEVICES=1 python benchmark_baselines.py \
    --config-path /Users/sebasmos/Desktop/QWave/config \
    --config-name esc50 \
    experiment.datasets.esc.normalization_type=standard \
    experiment.datasets.esc.csv=/Users/sebasmos/Documents/DATASETS/data_VE/ESC-50-master/VE_soundscapes/efficientnet_1536/esc-50.csv \
    experiment.device=cpu \
    experiment.models_to_run="[esc,qesc,bitnet,'1','2','4','8','16']" \
    experiment.metadata.tag=benchmark_baselines

python benchmark.py \
    --config-path /Users/sebasmos/Desktop/QWave/config \
    --config-name esc50 \
    experiment.datasets.esc.normalization_type=standard \
    experiment.datasets.esc.csv=/Users/sebasmos/Documents/DATASETS/data_VE/ESC-50-master/VE_soundscapes/efficientnet_1536/esc-50.csv \
    experiment.device=cpu \
    experiment.models_to_run="[moe]" \
    "experiment.router.expert_quantizations=[1,2,4,16]"\
    experiment.router.num_experts=4 \
    experiment.metadata.tag=benchmark

# qmoe with quantizations
python benchmark.py \
    --config-path /Users/sebasmos/Desktop/QWave/config \
    --config-name esc50 \
    experiment.datasets.esc.normalization_type=standard \
    experiment.datasets.esc.csv=/Users/sebasmos/Documents/DATASETS/data_VE/ESC-50-master/VE_soundscapes/efficientnet_1536/esc-50.csv \
    experiment.device=cpu \
    experiment.models_to_run="[qmoe]" \
    "experiment.router.expert_quantizations=[qesc,qesc,qesc,qesc]"\
    experiment.router.num_experts=4 \
    experiment.metadata.tag=benchmark
"""

from __future__ import annotations

from pathlib import Path
import os, sys, json, time, warnings

import hydra
from omegaconf import DictConfig, ListConfig
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fvcore.nn import FlopCountAnalysis
from memory_profiler import memory_usage
from codecarbon import EmissionsTracker
import platform
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT); sys.path.insert(0, str(ROOT))

# ─── QWave ------------------------------------------------------------------
from QWave.datasets import EmbeddingAdaptDataset
from QWave.graphics import plot_multiclass_roc_curve, plot_losses
from QWave.models import ESCModel
from QWave.memory import print_size_of_model, _load_cc_csv
from QWave.utils import get_device, get_num_parameters
from QWave.train_utils import train_pytorch_local, _validate_single_epoch
from QWave.qmoe_layers import BitNetExpert, BitNetExpert158b
from QWave.moe import train_moe_local, _validate_moe_epoch, qMoEModelBatched,BayesianRouter
# ─── Model factory ──────────────────────────────────────────────────────────

def build_model(model_kind: str, in_dim: int, n_cls: int, cfg: DictConfig):
    """Return a *single* network according to `model_kind`."""
    if model_kind == "bitnet":
        return BitNetExpert(
            in_dim, n_cls,
            hidden_sizes=cfg.experiment.model.hidden_sizes,
            dropout_prob=cfg.experiment.model.dropout_prob,
            num_bits="bitnet",
        )
    if model_kind == "moe" or model_kind == "qmoe":
        # qmoe is initialized as MoE 
        device = get_device(cfg)
        num_classes = len(np.unique(pd.read_csv(cfg.experiment.datasets.esc.csv)["class_id"].values)) 
        return qMoEModelBatched(cfg, in_dim, 
            num_classes, 
            cfg.experiment.router.num_experts, 
            cfg.experiment.router.top_k).to(device)

    if model_kind == "bitnet158b":
        return BitNetExpert158b(
            in_dim, n_cls,
            hidden_sizes=cfg.experiment.model.hidden_sizes,
            dropout_prob=cfg.experiment.model.dropout_prob,
        )
    if model_kind == "esc":
        return ESCModel(
            in_dim, n_cls,
            hidden_sizes=cfg.experiment.model.hidden_sizes,
            dropout_prob=cfg.experiment.model.dropout_prob,
        )
    if model_kind == "qesc":
        print("Quantized ESCModel selected.")
        model = ESCModel(
            in_dim, n_cls,
            hidden_sizes=cfg.experiment.model.hidden_sizes,
            dropout_prob=cfg.experiment.model.dropout_prob,
        ) 
        torch.backends.quantized.engine = 'qnnpack'
        return model 
    if model_kind.isdigit() and int(model_kind) in {1,2,4,8,16}:
        return BitNetExpert(
            in_dim, n_cls,
            hidden_sizes=cfg.experiment.model.hidden_sizes,
            dropout_prob=cfg.experiment.model.dropout_prob,
            num_bits=int(model_kind),
        )
    raise ValueError(f"Unknown model_kind '{model_kind}'")

# ─── helpers ───────────────────────────────────────────────────────────────
cc_metrics_to_track = [
    "duration", "emissions", "emissions_rate", "cpu_power", "gpu_power", "ram_power",
    "cpu_energy", "gpu_energy", "ram_energy", "energy_consumed",
    "cpu_count", "cpu_model", "gpu_count", "gpu_model", "ram_total_size",
]

# ─── Cross-Validation Run Function ──────────────────────────────────────────
def run_cv(csv_path: str, cfg: DictConfig):
    device = get_device(cfg)
    tag = cfg.experiment.metadata.tag
    out_dir = Path("outputs") / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | Output dir: {out_dir}\n")
    print(f"DEBUG: experiment.metadata.tag is '{tag}'")

    # 1. data --------------------------------------------------------------
    df_full = pd.read_csv(csv_path)
    if df_full.empty:
        print(f"[ERROR] Input CSV {csv_path} is empty. Skipping dataset.")
        return {}

    labels = df_full["class_id"].values
    df_feat = df_full.drop(columns=["folder","name","label","category"], errors="ignore")
    
    if len(np.unique(labels)) < cfg.experiment.cross_validation.n_splits:
        print(f"[ERROR] Not enough unique classes ({len(np.unique(labels))}) for {cfg.experiment.cross_validation.n_splits} folds. Please adjust n_splits or provide more data.")
        return {}
    if len(df_feat) < cfg.experiment.cross_validation.n_splits:
        print(f"[ERROR] Not enough samples ({len(df_feat)}) for {cfg.experiment.cross_validation.n_splits} folds. Please adjust n_splits or provide more data.")
        return {}

    skf = StratifiedKFold(cfg.experiment.cross_validation.n_splits, shuffle=True, random_state=42)

    to_run = cfg.experiment.get("models_to_run", ["bitnet", "bitnet158b", "esc"])
    if isinstance(to_run, ListConfig):
        to_run = list(to_run)
    print("Models to run:", to_run)

    global_summary = {}
    all_filtered_rows = []

    # ── iterate over requested models ─────────────────────────────────────
    for model_kind in to_run:
        print(f"\n===== MODEL: {model_kind} =====")
        model_out = out_dir / model_kind
        model_out.mkdir(exist_ok=True)

        all_final_accuracy_scores, all_final_f1_scores = [], []
        all_training_durations, all_validation_durations = [], []
        all_max_ram_mb_train, all_max_ram_mb_val = [], [] 
        all_model_size_mb = []
        all_training_flops, all_validation_flops = [], []
        
        all_train_cc_data_agg = {k: [] for k in cc_metrics_to_track}
        all_val_cc_data_agg = {k: [] for k in cc_metrics_to_track}

        fold_metrics = []

        in_dim = df_feat.shape[1] - 1
        num_classes = len(np.unique(labels))
        
        print(f"DEBUG: Dataset determined input dimension (in_dim): {in_dim}")
        
        dummy_model_for_params = build_model(str(model_kind), in_dim, num_classes, cfg).to(device)
        moe_parameter_count = get_num_parameters(dummy_model_for_params)
        print(f"Params: {moe_parameter_count:,}")
        del dummy_model_for_params

        for fold, (tr_idx, vl_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
            print(f"── Fold {fold}/{skf.n_splits}")
            fold_dir = model_out / f"fold_{fold}"
            fold_dir.mkdir(exist_ok=True)

            # datasets ───────────────────────────────────────────────────
            tr_ds = EmbeddingAdaptDataset(
                df_feat.iloc[tr_idx].reset_index(drop=True),
                normalization_type=cfg.experiment.datasets.esc.normalization_type)
            vl_ds = EmbeddingAdaptDataset(
                df_feat.iloc[vl_idx].reset_index(drop=True),
                normalization_type=cfg.experiment.datasets.esc.normalization_type,
                scaler=tr_ds.get_scaler())
            
            if len(tr_ds) == 0 or len(vl_ds) == 0:
                print(f"[WARN] Fold {fold} has empty train or validation dataset. Skipping this fold.")
                continue

            tr_ld = DataLoader(tr_ds, batch_size=cfg.experiment.model.batch_size, shuffle=True)
            vl_ld = DataLoader(vl_ds, batch_size=cfg.experiment.model.batch_size, shuffle=False)

            # model ──────────────────────────────────────────────────────
            model = build_model(str(model_kind), in_dim, num_classes, cfg).to(device)
            
            if len(tr_ld) > 0:
                sample_input_train, _ = next(iter(tr_ld))
                sample_input_train = sample_input_train.to(device)
                flops_train_analyzer = FlopCountAnalysis(model, sample_input_train)
                training_flops = flops_train_analyzer.total() * cfg.experiment.model.epochs * len(tr_ld)
                all_training_flops.append(training_flops)
            else:
                all_training_flops.append(0)

            if len(vl_ld) > 0:
                sample_input_val, _ = next(iter(vl_ld))
                sample_input_val = sample_input_val.to(device)
                flops_val_analyzer = FlopCountAnalysis(model, sample_input_val)
                validation_flops = flops_val_analyzer.total() * len(vl_ld)
                all_validation_flops.append(validation_flops)
            else:
                all_validation_flops.append(0)

            class_weights = torch.tensor(1/np.bincount(tr_ds.labels.numpy()), dtype=torch.float32).to(device)

            ckpt_path = fold_dir / "best.pth"

            # training ───────────────────────────────────────────────────
            tracker_tr = EmissionsTracker(output_dir=str(fold_dir), output_file="emissions_train.csv",
                                          project_name=f"{tag}_{model_kind}_fold{fold}_train")
            t0_train = time.perf_counter(); tracker_tr.start()
            
            if model_kind == "moe" or model_kind == "qmoe":
                print(f"Training {model_kind} model...")
                load_balancing = cfg.experiment.router.load_balancing # Use load balancing from config
                mem_train_usage, (state_dict, train_losses, val_losses, best_f1, all_labels_best, all_preds_best, _) = \
                    memory_usage((train_moe_local, (cfg, load_balancing, model, tr_ld, vl_ld, class_weights, in_dim, device, str(fold_dir), False, ckpt_path)), interval=0.1, retval=True)
            else: 
                mem_train_usage, (
                    model_trained, train_losses, val_losses, best_f1_from_train,
                    all_labels_best_epoch, all_preds_best_epoch, all_probs_best_epoch
                ) = memory_usage(
                    (train_pytorch_local, (
                        cfg.experiment,
                        model,
                        tr_ld,
                        vl_ld,
                        class_weights,
                        in_dim,
                        device,
                        str(fold_dir),
                        False,
                        str(ckpt_path)
                    )), interval=0.1, retval=True)

            tracker_tr.stop()
            dur_tr = time.perf_counter() - t0_train
            
            max_ram_mb_train = float(np.max(mem_train_usage)) if mem_train_usage else 0.0
            all_max_ram_mb_train.append(max_ram_mb_train)
            all_training_durations.append(dur_tr)
            
            train_stats = _load_cc_csv(fold_dir / "emissions_train.csv")
            for k in cc_metrics_to_track:
                all_train_cc_data_agg[k].append(train_stats.get(k, 0.0))

            # reload best & validate ────────────────────────────────────
            final_model = model

            if ckpt_path.is_file():
                final_model.load_state_dict(torch.load(ckpt_path, map_location=device))
            else:
                warnings.warn(f"Checkpoint file not found at {ckpt_path}. Final validation will use the model's state returned from training for fold {fold}.")
                final_model.load_state_dict(model_trained.state_dict())
            final_model.eval()
            
            tracker_val = EmissionsTracker(output_dir=str(fold_dir), output_file="emissions_val.csv",
                                          project_name=f"{tag}_{model_kind}_fold{fold}_val")
            tracker_val.start()
        
            print_size_of_model(final_model, "Original_Model")


            if model_kind == "qmoe":
                if platform.system() == "Darwin":  
                    torch.backends.quantized.engine = 'qnnpack'
                else:
                    torch.backends.quantized.engine = 'fbgemm'
                print(torch.backends.quantized.engine)
                expert_quantizations = cfg.experiment.router.expert_quantizations
                # ----- quantise experts once -------------
                for i, exp in enumerate(final_model.experts):
                    # Check if expert i should be quantized based on its position in the list
                    # This assumes expert_quantizations is a list corresponding to expert indices
                    # import pdb; pdb.set_trace()  # Debugging line to inspect expert_quantizations
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
                mem_val_usage, (_, _, y_true, y_pred, y_prob) = \
    memory_usage((_validate_moe_epoch, (final_model, vl_ld, nn.CrossEntropyLoss(), device, str(fold_dir))), interval=0.01, retval=True)

                dur_val = time.perf_counter() - val_start_time
                model_size = print_size_of_model(final_model, "Quantized model")
            elif model_kind == "qesc":
                if platform.system() == "Darwin":  
                    torch.backends.quantized.engine = 'qnnpack'
                else:
                    torch.backends.quantized.engine = 'fbgemm'
                print(torch.backends.quantized.engine)
                qmodel = torch.quantization.quantize_dynamic(final_model, {nn.Linear}, dtype=torch.qint8)
                val_start_time = time.perf_counter()
                mem_val_usage, (_, _, y_true, y_pred, y_prob) = memory_usage(
                    (_validate_single_epoch, (qmodel, vl_ld, nn.CrossEntropyLoss(), device)),
                    interval=0.01, retval=True)
                dur_val = time.perf_counter() - val_start_time
                model_size = print_size_of_model(qmodel, "Quantized_Model")
            elif model_kind == "moe":
                
                val_start_time = time.perf_counter()
                mem_val_usage, (_, _, y_true, y_pred, y_prob) = \
    memory_usage((_validate_moe_epoch, (final_model, vl_ld, nn.CrossEntropyLoss(), device, str(fold_dir))), interval=0.01, retval=True)

            
                dur_val = time.perf_counter() - val_start_time
                model_size = print_size_of_model(model, "Model after validation")
            
            else: 
                val_start_time = time.perf_counter()
                mem_val_usage, (_, _, y_true, y_pred, y_prob) = memory_usage(
                    (_validate_single_epoch, (final_model, vl_ld, nn.CrossEntropyLoss(), device)),
                    interval=0.01, retval=True)
                dur_val = time.perf_counter() - val_start_time
                model_size = print_size_of_model(model, "Model after validation")

            tracker_val.stop()
        
            max_ram_mb_val = float(np.max(mem_val_usage)) if mem_val_usage else 0.0
            all_max_ram_mb_val.append(max_ram_mb_val)
            all_validation_durations.append(dur_val)
            all_model_size_mb.append(model_size)

            val_stats = _load_cc_csv(fold_dir / "emissions_val.csv")
            for k in cc_metrics_to_track:
                all_val_cc_data_agg[k].append(val_stats.get(k, 0.0))

            final_accuracy = accuracy_score(y_true, y_pred)
            final_f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            all_final_accuracy_scores.append(final_accuracy)
            all_final_f1_scores.append(final_f1_weighted)

            print(f"    Fold {fold} Results:")
            print(f"        F1 Score (Weighted): {final_f1_weighted:.4f}")
            print(f"        Validation RAM Usage (Max MB): {max_ram_mb_val:.2f}")
            print(f"        Validation Inference Time (s): {dur_val:.4f}")

            plot_losses(train_losses, val_losses, str(fold_dir))
            plot_multiclass_roc_curve(y_true, y_prob, EXPERIMENT_NAME=str(fold_dir))

            fold_result = {
                "best_f1": final_f1_weighted,
                "accuracy": final_accuracy,
                "model_size": model_size / 1e6, # Model size in MB
                "max_ram_mb_train": float(max_ram_mb_train),
                "max_ram_mb_val": float(max_ram_mb_val), # Peak memory (RAM) during validation
                "training_duration_seconds": float(dur_tr),
                "validation_duration_seconds": float(dur_val),
                "parameter_count": int(moe_parameter_count), # Parameter count
                "training_flops": float(training_flops),
                "validation_flops": float(validation_flops)
            }
            for k in cc_metrics_to_track:
                fold_result[f"train_{k}"] = train_stats.get(k, 0.0)
                fold_result[f"val_{k}"] = val_stats.get(k, 0.0)

            with open(fold_dir/"metrics.json", "w") as fp:
                json.dump(fold_result, fp, indent=4)
            fold_metrics.append(fold_result)

        # model summary JSON ──────────────────────────────────────────────────
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
            summary["max_ram_mb_val_mean"] = float(np.mean(all_max_ram_mb_val))
            summary["max_ram_mb_val_std"] = float(np.std(all_max_ram_mb_val))

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

        with open(model_out / "summary.json", "w") as fp:
            json.dump(summary, fp, indent=4)
        print("\n== MODEL SUMMARY ==\n", json.dumps(summary, indent=4))
        global_summary[model_kind] = summary

        # Individual Model Summary CSV ──────────────────────────────────
        # Ensure 'name' in summary_filtered.csv is tag_model kind
        model_name_for_csv = f"{tag}_{model_kind}" 
        current_model_filtered_data = {
            "name": model_name_for_csv, # Changed this line
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
        
        individual_csv_path = model_out / "summary_filtered.csv"
        pd.DataFrame([current_model_filtered_data]).to_csv(individual_csv_path, index=False)
        print(f"Individual model summary saved to {individual_csv_path}")

        all_filtered_rows.append(current_model_filtered_data)

    # Global Combined Summary CSV ─────────────────────────────────────────
    if all_filtered_rows:
        pd.DataFrame(all_filtered_rows).to_csv(out_dir / "summary_filtered.csv", index=False)
        print(f"\nGlobal combined summary saved to {out_dir / 'summary_filtered.csv'}")
    else:
        print("\nNo model runs completed to generate global combined summary_filtered.csv.")
    
    return global_summary

# ─── main ──────────────────────────────────────────────────────────────────
@hydra.main(version_base=None, config_path=str(ROOT / "config"), config_name="esc50")
def main(cfg: DictConfig):
    warnings.filterwarnings("ignore", category=UserWarning)

    for name, meta in cfg.experiment.datasets.items():
        csv_path = Path(meta.csv)
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        print(f"\n=== DATASET {name.upper()} → outputs/{cfg.experiment.metadata.tag}")
        run_cv(str(csv_path), cfg)

if __name__ == "__main__":
    main()

"""

"""    