#!/usr/bin/env python
"""
K-Fold Mixture-of-Experts (BCE router)

Cross-validation version of the MoE pipeline. Keeps the same folder/metrics
conventions as the original K-Fold CV on embedding CSVs reference, but the
modeling logic is the two-stage Experts ➜ Router approach with a
BCE-with-Logits Router.

Key points
----------
- Expert phase  – one binary `ESCModel` per class, trained within the fold.
- Router phase – linear layer trained with BCE on one-hot targets.
- Outputs – each fold gets its own directory, ROC plot, and `metrics.json`.
- Summary – aggregated weighted-F1 across folds saved to `summary.json`.

python run_trainer_MoETask_CV.py  --config-name=esc50 \
  experiment.datasets.esc.csv=/Users/sebasmos/Documents/DATASETS/data_VE/ESC-50-master/VE_soundscapes/efficientnet_1536/esc-50.csv \
  experiment.device=mps \
  experiment.metadata.tag="EfficientNet_esc50MoECV2"

"""

#!/usr/bin/env python
from pathlib import Path
import os, sys
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import json, warnings
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

def _pos_prob(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            out.append(torch.softmax(model(xb), dim=1)[:, 1].cpu())
    return torch.cat(out)

def _train_router(router: Router, X: torch.Tensor, y: np.ndarray, device: torch.device,
                  epochs: int, lr: float, batch_size: int, weight_decay: float):
    y_onehot = torch.eye(router.net[-1].out_features)[y].float()
    ds = TensorDataset(X, y_onehot)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(router.parameters(), lr=lr, weight_decay=weight_decay)
    router.to(device)
    router.train()
    for ep in range(1, epochs + 1):
        run = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(router(xb), yb)
            loss.backward()
            opt.step()
            run += loss.item() * len(xb)
        if ep % 15 == 0:
            print(f"    Router epoch {ep:3d} | BCE: {run/len(dl.dataset):.4f}")

def _load_cc_csv(csv_path: Path) -> Dict:
    if not csv_path.is_file():
        return {}
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return {}
    return df.iloc[-1].to_dict()

cc_keys_overall = [
    "project_name", "emissions",
    "emissions_rate", "cpu_power", "gpu_power", "ram_power", "cpu_energy", "gpu_energy",
    "ram_energy", "energy_consumed",
    "cpu_count", "cpu_model", "gpu_count", "gpu_model",
    "ram_total_size"
]

def run_cv_moe(csv_path: str, cfg: DictConfig):
    df_full = pd.read_csv(csv_path)
    labels = df_full["class_id"].astype(int).values
    df = df_full.drop(columns=["folder", "name", "label", "category"], errors="ignore")
    n_classes = int(labels.max() + 1)
    skf = StratifiedKFold(
        n_splits=cfg.experiment.cross_validation.n_splits,
        shuffle=cfg.experiment.cross_validation.shuffle,
        random_state=cfg.experiment.cross_validation.random_seed,
    )
    tag = cfg.experiment.metadata.tag
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
        y_tr, y_va = labels[tr_idx], labels[va_idx]
        train_duration_tracker = EmissionsTracker(
            project_name=f"{tag}_fold{fold}_train_experts_duration",
            output_dir=str(fold_dir),
            output_file="emissions_train_duration.csv",
            save_to_file=True,
        )
        train_duration_tracker.start()
        print("  == PHASE 1: training experts ==")
        experts: List[nn.Module] = []
        for cls in range(n_classes):
            df_tr_bin, df_va_bin = df_tr.copy(), df_va.copy()
            df_tr_bin["class_id"] = (y_tr == cls).astype(int)
            df_va_bin["class_id"] = (y_va == cls).astype(int)
            tr_ds = EmbeddingDataset(df_tr_bin)
            va_ds = EmbeddingDataset(df_va_bin)
            tr_ld = DataLoader(tr_ds, batch_size=cfg.experiment.model.batch_size, shuffle=True)
            va_ld = DataLoader(va_ds, batch_size=cfg.experiment.model.batch_size, shuffle=False)
            class_w = torch.tensor(1.0/np.bincount(tr_ds.labels.numpy()), dtype=torch.float32).to(device)
            in_dim = tr_ds.features.shape[1]
            model = ESCModel(in_dim, 2, hidden_sizes=cfg.experiment.model.hidden_sizes,
                             dropout_prob=cfg.experiment.model.dropout_prob).to(device)
            model.apply(reset_weights)
            exp_dir = fold_dir / f"expert_{cls}"
            exp_dir.mkdir(exist_ok=True)
            train_pytorch_local(
                args=cfg.experiment, model=model, train_loader=tr_ld, val_loader=va_ld,
                class_weights=class_w, num_columns=in_dim, device=device,
                fold_folder=str(exp_dir), resume_checkpoint=False,
                checkpoint_path=str(exp_dir / "best.pth"),
            )
            experts.append(model.eval())
        train_duration_tracker.stop()
        tr_full_ds = EmbeddingDataset(df_tr)
        tr_ld_full = DataLoader(tr_full_ds, batch_size=cfg.experiment.model.batch_size)
        va_full_ds = EmbeddingDataset(df_va)
        va_ld_full = DataLoader(va_full_ds, batch_size=cfg.experiment.model.batch_size)
        collect = lambda ld: torch.cat([_pos_prob(exp, ld, device).unsqueeze(1) for exp in experts], dim=1)
        X_tr, X_va = collect(tr_ld_full), collect(va_ld_full)
        print("  == PHASE 2: training router ==")
        router = Router(input_dim=n_classes, hidden_dim=cfg.experiment.router.hidden_dim, output_dim=n_classes, drop_prob=cfg.experiment.router.drop_prob)
        _train_router(router, X_tr, y_tr, device,
                      epochs=cfg.experiment.epochs,
                      lr=cfg.experiment.lr,
                      batch_size=cfg.experiment.batch_size,
                      weight_decay=cfg.experiment.weight_decay)
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
            probs_va = torch.sigmoid(logits_va).cpu().numpy()
            y_pred = probs_va.argmax(axis=1)
        val_duration_tracker.stop()
        overall_tracker.stop()
        overall_stats = _load_cc_csv(fold_dir / "emissions_overall.csv")
        train_dur_stats = _load_cc_csv(fold_dir / "emissions_train_duration.csv")
        val_dur_stats = _load_cc_csv(fold_dir / "emissions_val_duration.csv")
        acc = accuracy_score(y_va, y_pred)
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
        "f1_std": float(np.std(all_f1_scores)),
        "accuracy_mean": float(np.mean(all_accuracy_scores)),
        "accuracy_std": float(np.std(all_accuracy_scores)),
        "metadata": dict(cfg.experiment.metadata),
        "folds": fold_results,
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

# from pathlib import Path
# import os, sys
# ROOT = Path(__file__).resolve().parents[2]
# os.chdir(ROOT)
# sys.path.insert(0, str(ROOT))

# import os, json, warnings
# from typing import List, Dict

# import hydra
# from omegaconf import DictConfig
# import numpy as np
# import pandas as pd
# import torch
# from torch import nn
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import f1_score, accuracy_score

# from codecarbon import EmissionsTracker

# from QWave.datasets import EmbeddingDataset
# from QWave.models import ESCModel, reset_weights
# from QWave.train_utils import train_pytorch_local
# from QWave.graphics import plot_multiclass_roc_curve
# from QWave.utils import get_device


# class Router(nn.Module):
#     def __init__(self, num_experts: int):
#         super().__init__()
#         self.fc = nn.Linear(num_experts, num_experts)
#         nn.init.xavier_uniform_(self.fc.weight)
#         nn.init.zeros_(self.fc.bias)
#     def forward(self, x: torch.Tensor):
#         return self.fc(x)


# def _pos_prob(model: nn.Module, loader: DataLoader, device: torch.device):
#     model.eval()
#     out = []
#     with torch.no_grad():
#         for xb, _ in loader:
#             xb = xb.to(device)
#             out.append(torch.softmax(model(xb), dim=1)[:, 1].cpu())
#     return torch.cat(out)


# def _train_router(router: Router, X: torch.Tensor, y: np.ndarray, device: torch.device,
#                    *, epochs: int, lr: float, batch_size: int):
#     y_onehot = torch.eye(router.fc.out_features)[y].float()
#     ds = TensorDataset(X, y_onehot)
#     dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
#     crit = nn.BCEWithLogitsLoss()
#     opt  = torch.optim.Adam(router.parameters(), lr=lr)
#     router.to(device)
#     router.train()
#     for ep in range(1, epochs+1):
#         run = 0.0
#         for xb, yb in dl:
#             xb, yb = xb.to(device), yb.to(device)
#             opt.zero_grad();  loss = crit(router(xb), yb);  loss.backward();  opt.step()
#             run += loss.item() * len(xb)
#         if ep % 15 == 0:
#             print(f"    Router epoch {ep:3d} | BCE: {run/len(dl.dataset):.4f}")


# def _load_cc_csv(csv_path: Path) -> Dict:
#     if not csv_path.is_file():
#         return {}
#     df = pd.read_csv(csv_path)
#     if len(df) == 0:
#         return {}
#     return df.iloc[-1].to_dict()

# cc_keys_overall = [
#     "project_name", "emissions",
#     "emissions_rate","cpu_power","gpu_power","ram_power","cpu_energy","gpu_energy",
#     "ram_energy","energy_consumed",
#     "cpu_count","cpu_model","gpu_count","gpu_model",
#     "ram_total_size"
# ]

# def run_cv_moe(csv_path: str, cfg: DictConfig):
#     df_full = pd.read_csv(csv_path)
#     labels  = df_full["class_id"].astype(int).values
#     df      = df_full.drop(columns=["folder", "name", "label", "category"], errors="ignore")
#     n_classes = int(labels.max() + 1)

#     skf = StratifiedKFold(
#         n_splits = cfg.experiment.cross_validation.n_splits,
#         shuffle  = cfg.experiment.cross_validation.shuffle,
#         random_state = cfg.experiment.cross_validation.random_seed,
#     )

#     tag       = cfg.experiment.metadata.tag
#     out_dir = Path("outputs") / tag
#     out_dir.mkdir(parents=True, exist_ok=True)

#     device = get_device(cfg)
#     print(f"Using device: {device}\nData shape: {df.shape}")

#     fold_results = []
    
#     all_f1_scores = []
#     all_accuracy_scores = []
#     all_overall_codecarbon_data = {k: [] for k in cc_keys_overall} 
#     all_train_duration_data = [] 
#     all_val_duration_data = []   

#     for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
#         print(f"\n--- FOLD {fold+1}/{skf.n_splits} ---")
#         fold_dir = out_dir / f"fold_{fold}"
#         fold_dir.mkdir(exist_ok=True)

#         overall_tracker = EmissionsTracker(
#             project_name=f"{tag}_fold{fold}_overall",
#             output_dir=str(fold_dir),
#             output_file="emissions_overall.csv",
#             save_to_file=True,
#         )
#         overall_tracker.start()

#         df_tr, df_va = df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)
#         y_tr, y_va   = labels[tr_idx], labels[va_idx]

#         train_duration_tracker = EmissionsTracker(
#             project_name=f"{tag}_fold{fold}_train_experts_duration",
#             output_dir=str(fold_dir),
#             output_file="emissions_train_duration.csv",
#             save_to_file=True,
#         )
#         train_duration_tracker.start()

#         print("  == PHASE 1: training experts ==")
#         experts: List[nn.Module] = []
#         for cls in range(n_classes):
#             df_tr_bin, df_va_bin = df_tr.copy(), df_va.copy()
#             df_tr_bin["class_id"] = (y_tr == cls).astype(int)
#             df_va_bin["class_id"] = (y_va == cls).astype(int)

#             tr_ds = EmbeddingDataset(df_tr_bin); va_ds = EmbeddingDataset(df_va_bin)
#             tr_ld = DataLoader(tr_ds, batch_size=cfg.experiment.model.batch_size, shuffle=True)
#             va_ld = DataLoader(va_ds, batch_size=cfg.experiment.model.batch_size, shuffle=False)

#             class_w = torch.tensor(1.0/np.bincount(tr_ds.labels.numpy()), dtype=torch.float32).to(device)
#             in_dim  = tr_ds.features.shape[1]
#             model   = ESCModel(in_dim, 2, hidden_sizes=cfg.experiment.model.hidden_sizes,
#                                  dropout_prob=cfg.experiment.model.dropout_prob).to(device)
#             model.apply(reset_weights)

#             exp_dir = fold_dir / f"expert_{cls}"; exp_dir.mkdir(exist_ok=True)
#             train_pytorch_local(
#                 args=cfg.experiment, model=model, train_loader=tr_ld, val_loader=va_ld,
#                 class_weights=class_w, num_columns=in_dim, device=device,
#                 fold_folder=str(exp_dir), resume_checkpoint=False,
#                 checkpoint_path=str(exp_dir / "best.pth"),
#             )
#             experts.append(model.eval())
        
#         train_duration_tracker.stop() 

#         tr_full_ds = EmbeddingDataset(df_tr); tr_ld_full = DataLoader(tr_full_ds, batch_size=cfg.experiment.model.batch_size)
#         va_full_ds = EmbeddingDataset(df_va); va_ld_full = DataLoader(va_full_ds, batch_size=cfg.experiment.model.batch_size)
#         collect = lambda ld: torch.cat([_pos_prob(exp, ld, device).unsqueeze(1) for exp in experts], dim=1)
#         X_tr, X_va = collect(tr_ld_full), collect(va_ld_full) # this calculates the expert probabilities

#         print("  == PHASE 2: training router ==")
#         router_cfg = getattr(cfg, "router", {})
#         router = Router(num_experts=n_classes)
#         _train_router(router, X_tr, y_tr, device,
#                       epochs=router_cfg.get("epochs", 75),
#                       lr=router_cfg.get("lr", 2e-3),
#                       batch_size=router_cfg.get("batch_size", 256))
        
#         val_duration_tracker = EmissionsTracker(
#             project_name=f"{tag}_fold{fold}_router_val_duration",
#             output_dir=str(fold_dir),
#             output_file="emissions_val_duration.csv",
#             save_to_file=True,
#         )
#         val_duration_tracker.start()

#         router.eval()
#         with torch.no_grad():
#             logits_va = router(X_va.to(device))
#             probs_va  = torch.sigmoid(logits_va).cpu().numpy()
#             y_pred    = probs_va.argmax(axis=1)
        
#         val_duration_tracker.stop() 

#         overall_tracker.stop() 
        
#         overall_stats = _load_cc_csv(fold_dir / "emissions_overall.csv")
#         train_dur_stats = _load_cc_csv(fold_dir / "emissions_train_duration.csv")
#         val_dur_stats = _load_cc_csv(fold_dir / "emissions_val_duration.csv")

#         acc  = accuracy_score(y_va, y_pred)
#         w_f1 = f1_score(y_va, y_pred, average="weighted")
#         print(f"  Fold {fold}: Accuracy={acc:.4f} | weighted-F1={w_f1:.4f}")

#         plot_multiclass_roc_curve(y_va, probs_va, EXPERIMENT_NAME=str(fold_dir))

#         fold_result = {
#             "best_f1": w_f1,
#             "accuracy": acc,
#             "codecarbon_train_duration": train_dur_stats.get("duration"), 
#             "codecarbon_val_duration": val_dur_stats.get("duration"),     
#             "codecarbon_duration": overall_stats.get("duration"),         
#         }
#         for k in cc_keys_overall:
#             fold_result[f"codecarbon_{k}"] = overall_stats.get(k)
        
#             if k in overall_stats:
#                 all_overall_codecarbon_data[k].append(overall_stats[k])

#         all_f1_scores.append(w_f1)
#         all_accuracy_scores.append(acc)
#         all_train_duration_data.append(train_dur_stats.get("duration"))
#         all_val_duration_data.append(val_dur_stats.get("duration"))

#         fold_results.append(fold_result)
#         with open(fold_dir / "metrics.json", "w") as f:
#             json.dump(fold_result, f, indent=4)

#     summary = {
#         "f1_mean": float(np.mean(all_f1_scores)),
#         "f1_std" : float(np.std(all_f1_scores)),
#         "accuracy_mean": float(np.mean(all_accuracy_scores)),
#         "accuracy_std": float(np.std(all_accuracy_scores)),
#         "metadata": dict(cfg.experiment.metadata),
#         "folds"   : fold_results,
#     }

#     if all_train_duration_data:
#         summary["codecarbon_train_duration_mean"] = float(np.mean(all_train_duration_data))
#         summary["codecarbon_train_duration_std"] = float(np.std(all_train_duration_data))
#     if all_val_duration_data:
#         summary["codecarbon_val_duration_mean"] = float(np.mean(all_val_duration_data))
#         summary["codecarbon_val_duration_std"] = float(np.std(all_val_duration_data))

#     for k in all_overall_codecarbon_data:
#         if all_overall_codecarbon_data[k]:
#             if all(isinstance(val, (int, float)) for val in all_overall_codecarbon_data[k]):
#                 summary[f"codecarbon_{k}_mean"] = float(np.mean(all_overall_codecarbon_data[k]))
#                 summary[f"codecarbon_{k}_std"] = float(np.std(all_overall_codecarbon_data[k]))
#             else:
#                 unique_values = list(set(all_overall_codecarbon_data[k]))
#                 if len(unique_values) == 1:
#                     summary[f"codecarbon_{k}_common"] = unique_values[0]
#                 else:
#                     summary[f"codecarbon_{k}_all"] = unique_values

#     with open(out_dir / "summary.json", "w") as f:
#         json.dump(summary, f, indent=4)
#     print("\n== CV SUMMARY ==")
#     print(json.dumps(summary, indent=4))


# @hydra.main(version_base=None, config_path=str(ROOT / "config"), config_name="esc50")
# def main(cfg: DictConfig):
#     for name, meta in cfg.experiment.datasets.items():
#         csv_path = Path(meta.csv)
#         if not csv_path.is_file():
#             raise FileNotFoundError(csv_path)
#         print(f"\n=== DATASET {name.upper()} → outputs/{cfg.experiment.metadata.tag}_moe")
#         run_cv_moe(str(csv_path), cfg)


# if __name__ == "__main__":
#     warnings.filterwarnings("ignore", category=UserWarning)
#     main()