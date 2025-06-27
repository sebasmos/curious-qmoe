#!/usr/bin/env python
"""
K-Fold Mixture-of-Experts (BCE router)
=====================================
Cross-validation version of the MoE pipeline. Keeps the same folder/metrics
conventions as the original *K-Fold CV on embedding CSVs* reference, but the
modeling logic is the two-stage Experts ➜ Router approach with a
**BCE-with-Logits Router**.

Key points
----------
* **Expert phase** – one binary `ESCModel` per class, trained within the fold.
* **Router phase** – linear layer trained with BCE on one-hot targets.
* **Outputs** – each fold gets its own directory, ROC plot, and `metrics.json`.
* **Summary** – aggregated weighted-F1 across folds saved to `summary.json`.


python run_trainer_MoETask.py  --config-name=esc50 \
  experiment.datasets.esc.csv=/Users/sebasmos/Documents/DATASETS/data_VE/ESC-50-master/VE_soundscapes/efficientnet_1536/esc-50.csv \
  experiment.device=cpu \
  experiment.metadata.tag="EfficientNet_esc50MoEDataCV"

"""

import os, json, warnings
import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

from QWave.datasets import EmbeddingDataset
from QWave.models import ESCModel, reset_weights
from QWave.train_utils import train_pytorch_local
from QWave.graphics import plot_multiclass_roc_curve, plot_losses
from QWave.utils import get_device

#######################################################################################
# Router                                                                              #
#######################################################################################

class Router(nn.Module):
    """Simple linear router: logits = W·p_expert."""
    def __init__(self, num_experts: int):
        super().__init__()
        self.fc = nn.Linear(num_experts, num_experts)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    def forward(self, x):
        return self.fc(x)  # raw logits

#######################################################################################
# Helpers                                                                             #
#######################################################################################

def _pos_prob(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            out.append(torch.softmax(model(xb), dim=1)[:, 1].cpu())
    return torch.cat(out)


def _train_router(router: Router, X: torch.Tensor, y: np.ndarray, device: torch.device,
                  *, epochs: int, lr: float, batch_size: int):
    y_onehot = torch.eye(router.fc.out_features)[y].float()
    ds = TensorDataset(X, y_onehot)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    crit = nn.BCEWithLogitsLoss()
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
            print(f"    Router epoch {ep:3d} | BCE: {run/len(dl.dataset):.4f}")

#######################################################################################
# CV pipeline                                                                         #
#######################################################################################

def run_cv_moe(csv_path: str, cfg: DictConfig):
    df_full = pd.read_csv(csv_path)
    labels  = df_full["class_id"].astype(int).values
    df      = df_full.drop(columns=["folder", "name", "label", "category"], errors="ignore")
    n_classes = int(labels.max() + 1)

    skf = StratifiedKFold(
        n_splits = cfg.experiment.cross_validation.n_splits,
        shuffle   = cfg.experiment.cross_validation.shuffle,
        random_state = cfg.experiment.cross_validation.random_seed,
    )

    tag     = cfg.experiment.metadata.tag 
    out_dir = os.path.abspath(os.path.join("outputs", tag))
    os.makedirs(out_dir, exist_ok=True)

    device = get_device(cfg)
    print(f"Using device: {device}\nData shape: {df.shape}")

    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n--- FOLD {fold+1}/{skf.n_splits} ---")
        fold_dir = os.path.join(out_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        df_tr, df_va = df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)
        y_tr, y_va   = labels[tr_idx], labels[va_idx]

        # ───────────────── PHASE 1: per-class experts ────────────────────
        print("  == PHASE 1: training experts ==")
        experts = []
        for cls in range(n_classes):
            df_tr_bin, df_va_bin = df_tr.copy(), df_va.copy()
            df_tr_bin["class_id"] = (y_tr == cls).astype(int)
            df_va_bin["class_id"] = (y_va == cls).astype(int)

            tr_ds = EmbeddingDataset(df_tr_bin); va_ds = EmbeddingDataset(df_va_bin)
            tr_ld = DataLoader(tr_ds, batch_size=cfg.experiment.model.batch_size, shuffle=True)
            va_ld = DataLoader(va_ds, batch_size=cfg.experiment.model.batch_size, shuffle=False)

            class_w = torch.tensor(1.0/np.bincount(tr_ds.labels.numpy()), dtype=torch.float32).to(device)
            in_dim  = tr_ds.features.shape[1]
            model   = ESCModel(in_dim, 2, hidden_sizes=cfg.experiment.model.hidden_sizes,
                                dropout_prob=cfg.experiment.model.dropout_prob).to(device)
            model.apply(reset_weights)

            exp_dir = os.path.join(fold_dir, f"expert_{cls}"); os.makedirs(exp_dir, exist_ok=True)
            train_pytorch_local(
                args=cfg.experiment, model=model, train_loader=tr_ld, val_loader=va_ld,
                class_weights=class_w, num_columns=in_dim, device=device,
                fold_folder=exp_dir, resume_checkpoint=False,
                checkpoint_path=os.path.join(exp_dir, "best.pth")
            )
            experts.append(model.eval())

        # collect expert probs --------------------------------------------------------
        tr_full_ds = EmbeddingDataset(df_tr); tr_ld_full = DataLoader(tr_full_ds, batch_size=cfg.experiment.model.batch_size)
        va_full_ds = EmbeddingDataset(df_va); va_ld_full = DataLoader(va_full_ds, batch_size=cfg.experiment.model.batch_size)
        collect = lambda ld: torch.cat([_pos_prob(exp, ld, device).unsqueeze(1) for exp in experts], dim=1)
        X_tr, X_va = collect(tr_ld_full), collect(va_ld_full)

        # ───────────────── PHASE 2: router ──────────────────────────────
        print("  == PHASE 2: training router ==")
        router_cfg = getattr(cfg, "router", {})
        router = Router(num_experts=n_classes)
        _train_router(router, X_tr, y_tr, device,
                      epochs=router_cfg.get("epochs", 75),
                      lr=router_cfg.get("lr", 2e-3),
                      batch_size=router_cfg.get("batch_size", 256))

        # evaluation ---------------------------------------------------------------
        router.eval()
        with torch.no_grad():
            logits_va = router(X_va.to(device))
            probs_va  = torch.sigmoid(logits_va).cpu().numpy()
            y_pred    = probs_va.argmax(axis=1)

        acc  = accuracy_score(y_va, y_pred)
        w_f1 = f1_score(y_va, y_pred, average="weighted")
        print(f"  Fold {fold}: acc={acc:.4f} | w-F1={w_f1:.4f}")

        # diagnostics --------------------------------------------------------------
        plot_multiclass_roc_curve(y_va, probs_va, EXPERIMENT_NAME=fold_dir)
        fold_metrics.append({"accuracy": acc, "best_f1": w_f1})
        with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
            json.dump({"accuracy": acc, "weighted_f1": w_f1}, f, indent=4)

    # summary -----------------------------------------------------------------------
    summary = {
        "f1_mean": float(np.mean([m["best_f1"] for m in fold_metrics])),
        "f1_std" : float(np.std([m["best_f1"]  for m in fold_metrics])),
        "acc_mean": float(np.mean([m["accuracy"] for m in fold_metrics])),
        "metadata": dict(cfg.experiment.metadata),
        "folds"   : fold_metrics,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    print("\n== CV SUMMARY ==")
    print(json.dumps(summary, indent=4))

#######################################################################################
# CLI                                                                                 #
#######################################################################################

@hydra.main(version_base=None, config_path="config", config_name="esc50")
def main(cfg: DictConfig):
    for name, meta in cfg.experiment.datasets.items():
        csv_path = meta.csv
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(csv_path)
        print(f"\n=== DATASET {name.upper()} → outputs/{cfg.experiment.metadata.tag}_moe")
        run_cv_moe(csv_path, cfg)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()