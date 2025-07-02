
#!/usr/bin/env python
"""
Hold-out Mixture-of-Experts with BCE router
===========================================
Training the Router with **BCE-with-Logits loss on one-hot targets**
--------------
1. **Router loss** – `BCEWithLogitsLoss` and uses
   one-hot labels, giving each output neuron an independent logistic objective.
2. **Router forward** – no softmax; the evaluation converts logits → sigmoid.
3. **Evaluation prints** – arg-max accuracy + weighted-F1.

All other behaviour (per-class experts, single hold-out split, ROC plot) is unchanged.

python run_trainer_MoETask_holdout.py \
  --config-path /Users/sebasmos/Desktop/QWave/config \
  --config-name esc50 \
  experiment.datasets.esc.csv=/Users/sebasmos/Documents/DATASETS/data_VE/ESC-50-master/VE_soundscapes/efficientnet_1536/esc-50.csv \
  experiment.device=cpu \
  experiment.metadata.tag=EfficientNet_esc50MoEData
"""
# Ensure qwave is in the path
from pathlib import Path, os, sys         
ROOT = Path(__file__).resolve().parents[2] 
os.chdir(ROOT)                             
sys.path.insert(0, str(ROOT))              
import os, json, warnings, sys
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from QWave.datasets import EmbeddingDataset
from QWave.models import ESCModel, reset_weights
from QWave.train_utils import train_pytorch_local
from QWave.graphics import plot_multiclass_roc_curve
from QWave.utils import get_device

class Router(nn.Module):
    """Linear layer → logits (no softmax)."""
    def __init__(self, num_experts: int):
        super().__init__()
        self.fc = nn.Linear(num_experts, num_experts)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # raw logits per class

def _pos_prob(model: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    model.eval()
    probs: List[torch.Tensor] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            pos = torch.softmax(model(xb), dim=1)[:, 1]
            probs.append(pos.cpu())
    return torch.cat(probs)


def _train_router(router: Router, X: torch.Tensor, y: np.ndarray, device: torch.device,
                  *, epochs: int, lr: float, batch_size: int):
    y_onehot = torch.eye(router.fc.out_features)[y].float()
    ds = TensorDataset(X, y_onehot)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    crit = nn.BCEWithLogitsLoss()
    opt  = torch.optim.Adam(router.parameters(), lr=lr)

    router.to(device)
    router.train()
    for ep in range(1, epochs + 1):
        running = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(router(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        if ep % 15 == 0:
            print(f"  epoch {ep:3d}/{epochs} | BCE: {running / len(dl.dataset):.4f}")


def run_holdout_moe(csv_path: str, cfg: DictConfig):
    df_full = pd.read_csv(csv_path)
    labels  = df_full["class_id"].astype(int).values
    df      = df_full.drop(columns=["folder", "name", "label", "category"], errors="ignore")
    n_classes = int(labels.max() + 1)

    val_frac = OmegaConf.select(cfg, "experiment.validation_split", default=0.2)
    tr_idx, va_idx = train_test_split(np.arange(len(labels)), test_size=val_frac,
                                      stratify=labels, random_state=42)

    df_tr, df_va = df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)
    y_tr, y_va   = labels[tr_idx], labels[va_idx]

    device = get_device(cfg)
    out_dir = os.path.abspath(os.path.join("outputs", cfg.experiment.metadata.tag))
    os.makedirs(out_dir, exist_ok=True)

    # Phase 1
    print("\n== PHASE 1: training per-class experts ==")
    experts: List[nn.Module] = []
    for cls in range(n_classes):
        print(f"  ↳ expert for class {cls}")
        df_tr_bin = df_tr.copy(); df_va_bin = df_va.copy()
        df_tr_bin["class_id"] = (y_tr == cls).astype(int)
        df_va_bin["class_id"] = (y_va == cls).astype(int)

        tr_ds = EmbeddingDataset(df_tr_bin)
        va_ds = EmbeddingDataset(df_va_bin)
        tr_ld = DataLoader(tr_ds, batch_size=cfg.experiment.model.batch_size, shuffle=True)
        va_ld = DataLoader(va_ds, batch_size=cfg.experiment.model.batch_size, shuffle=False)

        class_weights = torch.tensor(1.0 / np.bincount(tr_ds.labels.numpy()), dtype=torch.float32).to(device)
        in_dim = tr_ds.features.shape[1]

        model = ESCModel(in_dim, 2, hidden_sizes=cfg.experiment.model.hidden_sizes,
                         dropout_prob=cfg.experiment.model.dropout_prob).to(device)
        model.apply(reset_weights)

        expert_dir = os.path.join(out_dir, f"expert_{cls}")
        os.makedirs(expert_dir, exist_ok=True)
        train_pytorch_local(
            args=cfg.experiment, model=model, train_loader=tr_ld, val_loader=va_ld,
            class_weights=class_weights, num_columns=in_dim, device=device,
            fold_folder=expert_dir, resume_checkpoint=False, checkpoint_path=os.path.join(expert_dir, "best.pth")
        )
        experts.append(model.eval())

    tr_full_ds = EmbeddingDataset(df_tr); tr_ld_full = DataLoader(tr_full_ds, batch_size=cfg.experiment.model.batch_size)
    va_full_ds = EmbeddingDataset(df_va); va_ld_full = DataLoader(va_full_ds, batch_size=cfg.experiment.model.batch_size)

    def collect(loader):
        mats = [_pos_prob(exp, loader, device).unsqueeze(1) for exp in experts]
        return torch.cat(mats, dim=1)

    X_tr, X_va = collect(tr_ld_full), collect(va_ld_full)

    # Phase 2
    print("\n== PHASE 2: training router (BCE+one-hot) ==")
    router_cfg = OmegaConf.select(cfg, "router", default={})
    router = Router(num_experts=n_classes)
    _train_router(router, X_tr, y_tr, device,
                  epochs=router_cfg.get("epochs", 75),
                  lr=router_cfg.get("lr", 2e-3),
                  batch_size=router_cfg.get("batch_size", 256))

    # Evaluation
    router.eval()
    with torch.no_grad():
        logits_va = router(X_va.to(device))
        probs_va  = torch.sigmoid(logits_va).cpu().numpy()
        y_pred    = probs_va.argmax(axis=1)

    w_f1  = f1_score(y_va, y_pred, average="weighted")
    acc   = accuracy_score(y_va, y_pred)

    print("\n== RESULTS ==")
    print(f" Validation accuracy     : {acc:.4f}")
    print(f" Validation weighted-F1 : {w_f1:.4f}\n")

    # Save diagnostics ---------------------------------------------------------------
    plot_multiclass_roc_curve(y_va, probs_va, EXPERIMENT_NAME=out_dir)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"accuracy": acc, "weighted_f1": w_f1}, f, indent=4)

@hydra.main(version_base=None, config_path="config", config_name="esc50")
def main(cfg: DictConfig):
    for name, meta in cfg.experiment.datasets.items():
        csv_path: str = meta.csv
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(csv_path)
        print(f"\n=== DATASET {name.upper()} → outputs/{cfg.experiment.metadata.tag}_moe")
        run_holdout_moe(csv_path, cfg)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
