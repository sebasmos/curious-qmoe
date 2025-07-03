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
- Router phase – A linear layer trained with CrossEntropyLoss to combine the outputs
                 from all multi-class experts.
- Outputs – each fold gets its own directory, ROC plot, and `metrics.json`.
- Summary – aggregated weighted-F1 across folds saved to `summary.json`.

python run_trainer_MoEData_CV.py  --config-name=esc50 \
  experiment.datasets.esc.csv=/Users/sebasmos/Documents/DATASETS/data_VE/ESC-50-master/VE_soundscapes/efficientnet_1536/esc-50.csv \
  experiment.device=mps \
  experiment.metadata.tag="EfficientNet_esc50MoEMultiClassExperts"

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

from QWave.datasets import EmbeddingDataset
from QWave.models import ESCModel, reset_weights
from QWave.train_utils import train_pytorch_local # Assumes train_pytorch_local is updated for multi-class
from QWave.graphics import plot_multiclass_roc_curve
from QWave.utils import get_device


class Router(nn.Module):
    # CHANGED: Router now takes the combined input dimension from all experts (N_experts * N_classes)
    # and outputs N_classes logits for the final classification.
    def __init__(self, input_dim_from_experts: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim_from_experts, n_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    def forward(self, x: torch.Tensor):
        return self.fc(x)


# CHANGED: This function now collects ALL n_classes probabilities from a multi-class expert.
# Renamed from _pos_prob for clarity.
def _expert_output_full_probs(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for xb, _ in loader: # xb are inputs, _ are dummy labels since we don't need them here
            xb = xb.to(device)
            # The model (expert) now outputs n_classes logits, apply softmax to get full probability distribution
            probs = torch.softmax(model(xb), dim=1).cpu()
            all_probs.append(probs)
    return torch.cat(all_probs, dim=0) # Concatenate along the batch dimension


def _train_router(router: Router, X: torch.Tensor, y: np.ndarray, device: torch.device,
                   *, epochs: int, lr: float, batch_size: int):
    # CHANGED: For multi-class classification, use CrossEntropyLoss, and y should be class indices (long tensor).
    # No need for y_onehot conversion here, as CrossEntropyLoss expects class indices.
    ds = TensorDataset(X, torch.from_numpy(y).long()) # Ensure y is a LongTensor for CrossEntropyLoss
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    crit = nn.CrossEntropyLoss() # CHANGED: Criterion is now CrossEntropyLoss
    opt  = torch.optim.Adam(router.parameters(), lr=lr)
    router.to(device)
    router.train()
    for ep in range(1, epochs+1):
        run = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad();  loss = crit(router(xb), yb);  loss.backward();  opt.step()
            run += loss.item() * len(xb)
        if ep % 15 == 0:
            # CHANGED: Print CrossEntropyLoss instead of BCE
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
        y_tr, y_va   = labels[tr_idx], labels[va_idx] # Keep original multi-class labels for training experts

        train_duration_tracker = EmissionsTracker(
            project_name=f"{tag}_fold{fold}_train_experts_duration",
            output_dir=str(fold_dir),
            output_file="emissions_train_duration.csv",
            save_to_file=True,
        )
        train_duration_tracker.start()

        print("  == PHASE 1: training experts ==")
        experts: List[nn.Module] = []
        for cls in range(n_classes): # We still create N_classes experts
            # CHANGED: Experts are now trained on the full multi-class dataset, not binary versions.
            # Remove df_tr_bin, df_va_bin creation.
            tr_ds = EmbeddingDataset(df_tr); va_ds = EmbeddingDataset(df_va) # Use original multi-class datasets
            tr_ld = DataLoader(tr_ds, batch_size=cfg.experiment.model.batch_size, shuffle=True)
            va_ld = DataLoader(va_ds, batch_size=cfg.experiment.model.batch_size, shuffle=False)

            # CHANGED: Class weights are calculated based on the original multi-class training labels (y_tr).
            class_w_vals = np.bincount(y_tr)
            class_w = torch.tensor(1.0 / class_w_vals, dtype=torch.float32).to(device)
            
            in_dim  = tr_ds.features.shape[1]
            # CHANGED: Each ESCModel (expert) now outputs n_classes logits.
            model   = ESCModel(in_dim, n_classes, # Output dimension is n_classes
                                 hidden_sizes=cfg.experiment.model.hidden_sizes,
                                 dropout_prob=cfg.experiment.model.dropout_prob).to(device)
            model.apply(reset_weights)

            exp_dir = fold_dir / f"expert_{cls}"; exp_dir.mkdir(exist_ok=True)
            # train_pytorch_local is called for each expert, now training a multi-class model.
            train_pytorch_local(
                args=cfg.experiment, model=model, train_loader=tr_ld, val_loader=va_ld,
                class_weights=class_w, num_columns=in_dim, device=device,
                fold_folder=str(exp_dir), resume_checkpoint=False,
                checkpoint_path=str(exp_dir / "best.pth"),
            )
            experts.append(model.eval())
        
        train_duration_tracker.stop() 

        tr_full_ds = EmbeddingDataset(df_tr); tr_ld_full = DataLoader(tr_full_ds, batch_size=cfg.experiment.model.batch_size)
        va_full_ds = EmbeddingDataset(df_va); va_ld_full = DataLoader(va_full_ds, batch_size=cfg.experiment.model.batch_size)
        
        # CHANGED: Use the new _expert_output_full_probs function.
        # It already returns (batch_size, n_classes), so no .unsqueeze(1) needed per expert output.
        # The concatenation still happens along dim=1, resulting in (num_samples, n_experts * n_classes).
        collect = lambda ld: torch.cat([_expert_output_full_probs(exp, ld, device) for exp in experts], dim=1)
        X_tr, X_va = collect(tr_ld_full), collect(va_ld_full)

        print("  == PHASE 2: training router ==")
        router_cfg = getattr(cfg, "router", {})
        # CHANGED: Router initialization now reflects the new input dimension.
        # It takes N_experts * N_classes as input and outputs N_classes logits.
        router = Router(input_dim_from_experts=n_classes * n_classes, n_classes=n_classes)
        _train_router(router, X_tr, y_tr, device, # y_tr passed as class indices for CrossEntropyLoss
                      epochs=router_cfg.get("epochs", 75),
                      lr=router_cfg.get("lr", 2e-3),
                      batch_size=router_cfg.get("batch_size", 256))
        
        val_duration_tracker = EmissionsTracker(
            project_name=f"{tag}_fold{fold}_router_val_duration",
            output_dir=str(fold_dir),
            output_file="emissions_val_duration.csv",
            save_to_file=True,
        )
        val_duration_tracker.start()

        router.eval()
        with torch.no_grad():
            logits_va = router(X_va.to(device))
            # CHANGED: Apply softmax (not sigmoid) to router's logits for multi-class probabilities
            # when using CrossEntropyLoss.
            probs_va  = torch.softmax(logits_va, dim=1).cpu().numpy()
            y_pred    = probs_va.argmax(axis=1)
        
        val_duration_tracker.stop() 

        overall_tracker.stop() 
        
        overall_stats = _load_cc_csv(fold_dir / "emissions_overall.csv")
        train_dur_stats = _load_cc_csv(fold_dir / "emissions_train_duration.csv")
        val_dur_stats = _load_cc_csv(fold_dir / "emissions_val_duration.csv")

        acc  = accuracy_score(y_va, y_pred)
        w_f1 = f1_score(y_va, y_pred, average="weighted")
        print(f"  Fold {fold}: Accuracy={acc:.4f} | weighted-F1={w_f1:.4f}")

        plot_multiclass_roc_curve(y_va, probs_va, EXPERIMENT_NAME=str(fold_dir))

        fold_result = {
            "best_f1": w_f1,
            "accuracy": acc,
            "codecarbon_train_duration": train_dur_stats.get("duration"), 
            "codecarbon_val_duration": val_dur_stats.get("duration"),     
            "codecarbon_duration": overall_stats.get("duration"),         
        }
        for k in cc_keys_overall:
            fold_result[f"codecarbon_{k}"] = overall_stats.get(k)
        
            if k in overall_stats:
                all_overall_codecarbon_data[k].append(overall_stats[k])

        all_f1_scores.append(w_f1)
        all_accuracy_scores.append(acc)
        all_train_duration_data.append(train_dur_stats.get("duration"))
        all_val_duration_data.append(val_dur_stats.get("duration"))

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