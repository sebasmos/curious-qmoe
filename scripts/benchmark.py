# run_three_models.py
#!/usr/bin/env python

"""
Train **single-network** versions of:
  1. BitNet        (ternary)        – key: "bitnet"
  2. BitNet-158B   (ternary v2)     – key: "bitnet158b"
  3. FP32 BitNetExpert (pre-LN, same architecture as k-bit) – key: "esc"

Optional extras: numeric quantisations **1,2,4,8,16** (k-bit) are supported
with keys "1", "2", … "16".

Select which to run via
    experiment.models_to_run="[bitnet,bitnet158b,esc]"

The script keeps *all* variable names, CSV/JSON summaries, plots, CodeCarbon,
FLOP count, memory profiling, etc., but removes every MoE-specific element.

python benchmark_baselines.py --config-path /home/sebastian/codes/repo_clean/QWave/config --config-name esc50 experiment.datasets.esc.normalization_type=standard experiment.datasets.esc.csv=/home/sebasmos/Documents/DATASET/esc-50.csv  experiment.device=cuda  experiment.models_to_run="['1','2','4','8','16',esc]"  experiment.metadata.tag=benchmark_baselines

CUDA_VISIBLE_DEVICES=1 python benchmark_baselines.py --config-path /home/sebasmos/Desktop/QWave/config \
    --config-name esc50 \
    experiment.datasets.esc.normalization_type=standard \
    experiment.datasets.esc.csv=/home/sebasmos/Documents/DATASET/esc-50.csv \
    experiment.device=cuda \
    experiment.models_to_run="["bitnet", "qesc", '1','2','4','8','16',esc]" \
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
    experiment.models_to_run="[qmoe, moe , esc,qesc,bitnet,'1','2','4','8','16']" \
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
from omegaconf import DictConfig, OmegaConf, open_dict, ListConfig
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fvcore.nn import FlopCountAnalysis
from memory_profiler import memory_usage
from codecarbon import EmissionsTracker
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT); sys.path.insert(0, str(ROOT))

# ─── QWave ------------------------------------------------------------------
from QWave.datasets import EmbeddingAdaptDataset, make_outer_folds, inner_split
from QWave.graphics import plot_multiclass_roc_curve, plot_losses
from QWave.memory import print_size_of_model, _load_cc_csv
from QWave.utils import get_device, get_num_parameters, set_seed
from QWave.train_utils import train_pytorch_local, _validate_single_epoch
from QWave.qmoe_layers import BitNetExpert, BitNetExpert158b, ptq_dynamic_int8
from QWave.moe import train_moe_local, _validate_moe_epoch, qMoEModelBatched
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
        _label_col = cfg.experiment.datasets.esc.get("label_col", "class_id")
        num_classes = len(np.unique(pd.read_csv(cfg.experiment.datasets.esc.csv)[_label_col].values))
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
    if model_kind in ("esc", "qesc"):
        # FP32 baseline on the same pre-LayerNorm architecture as the k-bit
        # models; "qesc" is the same network, dynamically int8-quantized after
        # training (ptq_dynamic_int8) before the test pass.
        return BitNetExpert(
            in_dim, n_cls,
            hidden_sizes=cfg.experiment.model.hidden_sizes,
            dropout_prob=cfg.experiment.model.dropout_prob,
            num_bits=32,
        )
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

# Bumped whenever the evaluation protocol changes, so the collector can refuse
# to pool summaries produced under different protocols. 2 = predefined folds
# where they exist, inner validation split for model selection, per-fold
# seeding, macro F1 primary.
PROTOCOL_VERSION = 2

def checkpoint_name(model_kind: str) -> str:
    """train_moe_local writes best.pth, train_pytorch_local best_model.pth."""
    return "best.pth" if model_kind in ("moe", "qmoe") else "best_model.pth"

def prepare_eval_model(model_kind: str, model, cfg):
    """The network scored on the test fold. The PTQ single model ("qesc") and
    every PTQ expert ("qesc") inside an MoE are dynamically int8-quantized
    after training; everything else is evaluated as trained."""
    model.eval()
    if model_kind == "qesc":
        return ptq_dynamic_int8(model)
    if model_kind in ("moe", "qmoe") and model.shared is None:
        for i, q in enumerate(cfg.experiment.router.expert_quantizations):
            if str(q) == "qesc":
                model.experts[i] = ptq_dynamic_int8(model.experts[i])
    return model

def route_test_fold(model, loader, device, seed):
    """Top-1 expert, epistemic uncertainty (None for a deterministic router)
    and prediction for every sample in loader, with MC dropout seeded."""
    torch.manual_seed(seed)
    model.eval()
    experts, unc, preds = [], [], []
    with torch.no_grad():
        for X, _ in loader:
            out, router_p, _, u = model(X.to(device))
            experts.append(router_p.argmax(dim=1).cpu())
            preds.append(out.argmax(dim=1).cpu())
            if u is not None:
                unc.append(u.cpu())
    return (torch.cat(experts).numpy(), torch.cat(unc).numpy() if unc else None,
            torch.cat(preds).numpy())

def routing_fractions(experts, uncertainty, n_experts):
    """Share of samples sent to each expert, overall and per quartile of the
    router's uncertainty (quartile 1 = most confident)."""
    block = {"overall": (np.bincount(experts, minlength=n_experts) / len(experts)).tolist()}
    if uncertainty is not None:
        edges = np.quantile(uncertainty, [0.25, 0.5, 0.75])
        quart = np.searchsorted(edges, uncertainty, side="right")
        block["by_uncertainty_quartile"] = [
            (np.bincount(experts[quart == q], minlength=n_experts) / max(1, int((quart == q).sum()))).tolist()
            for q in range(4)]
        block["quartile_counts"] = [int((quart == q).sum()) for q in range(4)]
        block["uncertainty_quartile_edges"] = edges.tolist()
    return block

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

    # label_col selects the supervision target. In the ESC-50 CSV, "class_id"
    # holds the 5 super-categories (Animals, Natural, Human, Interior,
    # Exterior) while "label" holds the true 50 ESC-50 classes. Historical
    # paper runs trained on class_id, i.e. the 5-way task.
    label_col = cfg.experiment.datasets.esc.get("label_col", "class_id")
    if label_col != "class_id":
        df_full["class_id"] = df_full[label_col]

    labels = df_full["class_id"].values
    df_feat = df_full.drop(columns=["folder","name","label","category"], errors="ignore")
    
    if len(np.unique(labels)) < cfg.experiment.cross_validation.n_splits:
        print(f"[ERROR] Not enough unique classes ({len(np.unique(labels))}) for {cfg.experiment.cross_validation.n_splits} folds. Please adjust n_splits or provide more data.")
        return {}
    if len(df_feat) < cfg.experiment.cross_validation.n_splits:
        print(f"[ERROR] Not enough samples ({len(df_feat)}) for {cfg.experiment.cross_validation.n_splits} folds. Please adjust n_splits or provide more data.")
        return {}

    cv_cfg = cfg.experiment.cross_validation
    folds = make_outer_folds(df_full, labels, cv_cfg)
    fold_scheme = ({"type": "predefined", "column": str(cv_cfg.fold_column)} if cv_cfg.get("fold_column")
                   else {"type": "stratified_kfold", "n_splits": int(cv_cfg.n_splits),
                         "random_state": int(cv_cfg.get("random_seed", 42))})
    inner_val_ratio = float(cv_cfg.get("validation_split_ratio", 0.1))
    seed = int(cfg.experiment.get("seed", 0))
    if cfg.experiment.get("num_threads"):
        torch.set_num_threads(int(cfg.experiment.num_threads))
    print(f"Folds: {fold_scheme} | seed {seed} | inner val {inner_val_ratio} | torch threads {torch.get_num_threads()}")

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
        # resolved config next to the checkpoints, so analysis scripts can
        # rebuild the exact model and data split from the run directory alone
        OmegaConf.save(cfg, model_out / "config.yaml", resolve=True)

        all_final_accuracy_scores, all_final_f1_scores, all_final_f1_weighted = [], [], []
        all_training_durations, all_validation_durations = [], []
        all_max_ram_mb_train, all_max_ram_mb_val = [], []
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

        for fold, tr_idx, te_idx in folds:
            print(f"── Fold {fold} ({len(folds)} folds)")
            fold_dir = model_out / f"fold_{fold}"
            fold_dir.mkdir(exist_ok=True)

            # Seed everything at the start of each fold, so any single fold is
            # reproducible on its own: init, dropout, MC dropout, loader order
            # and the inner split all follow the run seed.
            loader_gen = set_seed(seed)
            # Model selection runs on an inner validation split of the
            # training folds; the held-out fold te_idx is evaluated once, at
            # the end, with the selected checkpoint.
            fit_idx, iv_idx = inner_split(tr_idx, labels, inner_val_ratio, seed)
            with open(fold_dir / "split.json", "w") as fp:
                json.dump({"fold": fold, "seed": seed, "fold_scheme": fold_scheme,
                           "test_idx": te_idx.tolist(), "inner_val_idx": iv_idx.tolist()}, fp)

            # datasets ───────────────────────────────────────────────────
            # the scaler is fit on the fit portion only, then applied to the
            # inner validation split and the test fold
            tr_ds = EmbeddingAdaptDataset(
                df_feat.iloc[fit_idx].reset_index(drop=True),
                normalization_type=cfg.experiment.datasets.esc.normalization_type)
            iv_ds = EmbeddingAdaptDataset(
                df_feat.iloc[iv_idx].reset_index(drop=True),
                normalization_type=cfg.experiment.datasets.esc.normalization_type,
                scaler=tr_ds.get_scaler())
            te_ds = EmbeddingAdaptDataset(
                df_feat.iloc[te_idx].reset_index(drop=True),
                normalization_type=cfg.experiment.datasets.esc.normalization_type,
                scaler=tr_ds.get_scaler())

            if len(tr_ds) == 0 or len(iv_ds) == 0 or len(te_ds) == 0:
                print(f"[WARN] Fold {fold} has an empty train, inner validation or test set. Skipping this fold.")
                continue

            tr_ld = DataLoader(tr_ds, batch_size=cfg.experiment.train.batch_size, shuffle=True, generator=loader_gen)
            iv_ld = DataLoader(iv_ds, batch_size=cfg.experiment.train.batch_size, shuffle=False)
            te_ld = DataLoader(te_ds, batch_size=cfg.experiment.train.batch_size, shuffle=False)

            # model ──────────────────────────────────────────────────────
            # expose the fold index so per-fold warm-start paths resolve to a
            # checkpoint trained without this fold's validation split
            with open_dict(cfg):
                cfg.experiment.current_fold = fold
            model = build_model(str(model_kind), in_dim, num_classes, cfg).to(device)

            # FLOPs are traced on fixed rows rather than a loader batch, so
            # counting does not draw from the training loader's generator; the
            # state is restored afterwards because tracing runs a train-mode
            # forward that would update the router's running uncertainty range
            bs = cfg.experiment.train.batch_size
            pre_flops_state = {k: v.clone() for k, v in model.state_dict().items()}
            if len(tr_ld) > 0:
                sample_input_train = tr_ds.features[:bs].to(device)
                flops_train_analyzer = FlopCountAnalysis(model, sample_input_train)
                training_flops = flops_train_analyzer.total() * cfg.experiment.train.max_epochs * len(tr_ld)
                all_training_flops.append(training_flops)
            else:
                all_training_flops.append(0)

            if len(te_ld) > 0:
                sample_input_val = te_ds.features[:bs].to(device)
                flops_val_analyzer = FlopCountAnalysis(model, sample_input_val)
                validation_flops = flops_val_analyzer.total() * len(te_ld)
                all_validation_flops.append(validation_flops)
            else:
                all_validation_flops.append(0)
            model.load_state_dict(pre_flops_state)

            class_weights = torch.tensor(1/np.bincount(tr_ds.labels.numpy()), dtype=torch.float32).to(device)

            ckpt_path = fold_dir / checkpoint_name(str(model_kind))

            # training ───────────────────────────────────────────────────
            track_emissions = cfg.experiment.get("track_emissions", False)
            if track_emissions:
                tracker_tr = EmissionsTracker(output_dir=str(fold_dir), output_file="emissions_train.csv",
                                              project_name=f"{tag}_{model_kind}_fold{fold}_train")
                tracker_tr.start()
            t0_train = time.perf_counter()
            
            if model_kind == "moe" or model_kind == "qmoe":
                print(f"Training {model_kind} model...")
                load_balancing = cfg.experiment.router.load_balancing # Use load balancing from config
                mem_train_usage, (state_dict, train_losses, val_losses, inner_val_f1, all_labels_best, all_preds_best, _) = \
                    memory_usage((train_moe_local, (cfg, load_balancing, model, tr_ld, iv_ld, class_weights, in_dim, device, str(fold_dir), False, ckpt_path)), interval=0.1, retval=True, max_iterations=1)
            else:
                mem_train_usage, (
                    model_trained, train_losses, val_losses, inner_val_f1,
                    all_labels_best_epoch, all_preds_best_epoch, all_probs_best_epoch
                ) = memory_usage(
                    (train_pytorch_local, (
                        cfg.experiment,
                        model,
                        tr_ld,
                        iv_ld,
                        class_weights,
                        in_dim,
                        device,
                        str(fold_dir),
                        False,
                        str(ckpt_path)
                    )), interval=0.1, retval=True, max_iterations=1)

            if track_emissions:
                tracker_tr.stop()
            dur_tr = time.perf_counter() - t0_train
            
            max_ram_mb_train = float(np.max(mem_train_usage)) if mem_train_usage else 0.0
            all_max_ram_mb_train.append(max_ram_mb_train)
            all_training_durations.append(dur_tr)
            
            if track_emissions:
                train_stats = _load_cc_csv(fold_dir / "emissions_train.csv")
                for k in cc_metrics_to_track:
                    all_train_cc_data_agg[k].append(train_stats.get(k, 0.0))
            else:
                for k in cc_metrics_to_track:
                    all_train_cc_data_agg[k].append(0.0)

            # reload best & validate ────────────────────────────────────
            final_model = model

            if ckpt_path.is_file():
                final_model.load_state_dict(torch.load(ckpt_path, map_location=device))
            else:
                warnings.warn(f"Checkpoint file not found at {ckpt_path}. Final validation will use the model's state returned from training for fold {fold}.")
                final_model.load_state_dict(model_trained.state_dict())
            final_model.eval()
            
            if track_emissions:
                tracker_val = EmissionsTracker(output_dir=str(fold_dir), output_file="emissions_val.csv",
                                              project_name=f"{tag}_{model_kind}_fold{fold}_val")
                tracker_val.start()
        
            model_size_bytes = print_size_of_model(final_model, "Original_Model")


            # PTQ single models and PTQ experts inside an MoE are int8-quantized
            # here, after training, in both the "moe" and "qmoe" paths
            eval_model = prepare_eval_model(str(model_kind), final_model, cfg)
            for param in eval_model.parameters():
                param.requires_grad_(False)
            is_moe = model_kind in ("moe", "qmoe")
            # fixed MC-dropout noise for the test pass, so the routing pass
            # below sees exactly the same draws
            torch.manual_seed(seed)
            val_start_time = time.perf_counter()
            if is_moe:
                mem_val_usage, (_, _, y_true, y_pred, y_prob) = \
                    memory_usage((_validate_moe_epoch, (eval_model, te_ld, nn.CrossEntropyLoss(), device, str(fold_dir))), interval=0.01, retval=True, max_iterations=1)
            else:
                mem_val_usage, (_, _, y_true, y_pred, y_prob) = memory_usage(
                    (_validate_single_epoch, (eval_model, te_ld, nn.CrossEntropyLoss(), device)),
                    interval=0.01, retval=True, max_iterations=1)
            dur_val = time.perf_counter() - val_start_time

            if track_emissions:
                tracker_val.stop()

            max_ram_mb_val = float(np.max(mem_val_usage)) if mem_val_usage else 0.0
            all_max_ram_mb_val.append(max_ram_mb_val)
            all_validation_durations.append(dur_val)

            if track_emissions:
                val_stats = _load_cc_csv(fold_dir / "emissions_val.csv")
                for k in cc_metrics_to_track:
                    all_val_cc_data_agg[k].append(val_stats.get(k, 0.0))
            else:
                for k in cc_metrics_to_track:
                    all_val_cc_data_agg[k].append(0.0)

            final_accuracy = accuracy_score(y_true, y_pred)
            final_f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
            final_f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            all_final_accuracy_scores.append(final_accuracy)
            all_final_f1_scores.append(final_f1_macro)
            all_final_f1_weighted.append(final_f1_weighted)

            print(f"    Fold {fold} Results (held-out test fold):")
            print(f"        F1 Score (Macro): {final_f1_macro:.4f} | Weighted: {final_f1_weighted:.4f} | Acc: {final_accuracy:.4f}")
            print(f"        Test-pass RAM Usage (Max MB): {max_ram_mb_val:.2f}")
            print(f"        Test-pass duration, whole fold (s): {dur_val:.4f}")

            plot_losses(train_losses, val_losses, str(fold_dir))
            plot_multiclass_roc_curve(y_true, y_prob, EXPERIMENT_NAME=str(fold_dir))

            fold_result = {
                "fold": int(fold),
                "seed": seed,
                "n_fit": int(len(fit_idx)),
                "n_inner_val": int(len(iv_idx)),
                "n_test": int(len(te_idx)),
                "f1_macro": float(final_f1_macro),
                "f1_weighted": float(final_f1_weighted),
                "accuracy": float(final_accuracy),
                "inner_val_f1_macro": float(inner_val_f1),  # selection score of the chosen checkpoint
                "model_size_bytes": float(model_size_bytes),
                "max_ram_mb_train": float(max_ram_mb_train),
                "max_ram_mb_val": float(max_ram_mb_val), # peak process RSS during the test-fold pass
                "training_duration_seconds": float(dur_tr),
                # whole test-fold pass under memory_profiler, NOT per-sample
                # latency (scripts/analysis/efficiency_benchmark.py measures that)
                "validation_duration_seconds": float(dur_val),
                "parameter_count": int(moe_parameter_count), # Parameter count
                "training_flops": float(training_flops),
                "validation_flops": float(validation_flops)
            }
            if is_moe:
                # held-out routing: which expert each test sample reaches
                n_exp = eval_model.num_experts
                experts, unc, preds = route_test_fold(eval_model, te_ld, device, seed)
                fold_result["routing"] = routing_fractions(experts, unc, n_exp)
                fold_result["routing"]["matches_scored_pass"] = bool(np.array_equal(preds, np.asarray(y_pred)))
                # Same weights, prior switched off at inference: the paired
                # ablation that isolates what the precision prior itself does.
                if (getattr(eval_model, "use_curiosity", False) and eval_model.curiosity_strategy == "precision_prior"
                        and eval_model.curiosity_alpha != 0):
                    alpha = eval_model.curiosity_alpha
                    eval_model.curiosity_alpha = 0.0
                    experts0, unc0, preds0 = route_test_fold(eval_model, te_ld, device, seed)
                    eval_model.curiosity_alpha = alpha
                    y_np = np.asarray(y_true)
                    fold_result["alpha0_eval"] = {
                        "f1_macro": float(f1_score(y_np, preds0, average="macro", zero_division=0)),
                        "f1_weighted": float(f1_score(y_np, preds0, average="weighted", zero_division=0)),
                        "accuracy": float(accuracy_score(y_np, preds0)),
                        "expert_changed_fraction": float(np.mean(experts0 != experts)),
                        "prediction_changed_fraction": float(np.mean(preds0 != preds)),
                        "routing": routing_fractions(experts0, unc0, n_exp),
                    }
            if track_emissions:
                for k in cc_metrics_to_track:
                    fold_result[f"train_{k}"] = train_stats.get(k, 0.0)
                    fold_result[f"val_{k}"] = val_stats.get(k, 0.0)
            else:
                for k in cc_metrics_to_track:
                    fold_result[f"train_{k}"] = 0.0
                    fold_result[f"val_{k}"] = 0.0

            with open(fold_dir/"metrics.json", "w") as fp:
                json.dump(fold_result, fp, indent=4)
            fold_metrics.append(fold_result)

        # model summary JSON ──────────────────────────────────────────────────
        # Per-fold values (with fold ids and the seed) live in "folds", so the
        # collector can pair arms on (seed, fold); the means and population
        # stds here cover this run's folds only and are for a quick look.
        summary = {
            "protocol_version": PROTOCOL_VERSION,
            "model_kind": str(model_kind),
            "seed": seed,
            "fold_scheme": fold_scheme,
            "inner_val_ratio": inner_val_ratio,
            "primary_metric": "f1_macro",
            "num_threads": torch.get_num_threads(),
            "train_recipe": OmegaConf.to_container(cfg.experiment.train, resolve=True),
            "f1_macro_mean": float(np.mean(all_final_f1_scores)),
            "f1_macro_std": float(np.std(all_final_f1_scores)),
            "f1_weighted_mean": float(np.mean(all_final_f1_weighted)),
            "f1_weighted_std": float(np.std(all_final_f1_weighted)),
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
        
        if fold_metrics and all("routing" in f for f in fold_metrics):
            summary["routing_overall_mean"] = np.mean([f["routing"]["overall"] for f in fold_metrics], axis=0).tolist()
            if all("by_uncertainty_quartile" in f["routing"] for f in fold_metrics):
                summary["routing_by_uncertainty_quartile_mean"] = np.mean(
                    [f["routing"]["by_uncertainty_quartile"] for f in fold_metrics], axis=0).tolist()
        alpha0 = [f["alpha0_eval"] for f in fold_metrics if "alpha0_eval" in f]
        if alpha0:
            summary["alpha0_eval"] = {
                "f1_macro_mean": float(np.mean([a["f1_macro"] for a in alpha0])),
                "f1_macro_std": float(np.std([a["f1_macro"] for a in alpha0])),
                "accuracy_mean": float(np.mean([a["accuracy"] for a in alpha0])),
                "expert_changed_fraction_mean": float(np.mean([a["expert_changed_fraction"] for a in alpha0])),
                "note": "same checkpoints, precision prior off (alpha=0) at inference; per-fold values in folds[*].alpha0_eval",
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
            "F1 macro (Mean)": summary.get("f1_macro_mean"),
            "Infer. RAM (GB)": summary.get("max_ram_mb_val_mean", 0.0) / 1024,
            "Train. Runtime (s)": summary.get("training_duration_mean_seconds"),
            "Test pass, whole fold (s)": summary.get("validation_duration_mean_seconds"),
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