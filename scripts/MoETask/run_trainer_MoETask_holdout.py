
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

# ── Bootstrap so Hydra finds ./config no matter where the script lives ────
from pathlib import Path
import os, sys
ROOT = Path(__file__).resolve().parents[2]      # → /…/QWave
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
# -------------------------------------------------------------------------

import json, warnings
from typing import List, Dict

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from codecarbon import EmissionsTracker

from QWave.datasets import EmbeddingDataset
from QWave.models import ESCModel, reset_weights
from QWave.train_utils import train_pytorch_local
from QWave.graphics import plot_multiclass_roc_curve
from QWave.utils import get_device

class Router(nn.Module):
    """Linear router (no softmax)."""
    def __init__(self, num_experts: int):
        super().__init__()
        self.fc = nn.Linear(num_experts, num_experts)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    def forward(self, x: torch.Tensor):
        return self.fc(x)

def _pos_prob(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval(); out = []
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
    crit = nn.BCEWithLogitsLoss(); opt = torch.optim.Adam(router.parameters(), lr=lr)
    router.to(device); router.train()
    for ep in range(1, epochs + 1):
        run = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(router(xb), yb); loss.backward(); opt.step()
            run += loss.item() * len(xb)
        if ep % 15 == 0:
            print(f"    Router epoch {ep:3d} | BCE: {run/len(dl.dataset):.4f}")


def _start_tracker(phase: str, out_dir: Path, tag: str) -> EmissionsTracker:
    return EmissionsTracker(
        project_name=f"{tag}_{phase}",
        output_dir=str(out_dir),
        output_file=f"emissions_{phase}.csv",
        save_to_file=True,
    )


def _load_cc_csv(csv_path: Path) -> Dict:
    """Return the *last row* of a CodeCarbon CSV as a dict."""
    if not csv_path.is_file():
        return {}
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return {}
    return df.iloc[-1].to_dict()


def run_holdout_moe(csv_path: str, cfg: DictConfig):
    # ───── data ───────────────────────────────────────────────────────────
    df_full = pd.read_csv(csv_path)
    labels  = df_full["class_id"].astype(int).values
    df      = df_full.drop(columns=["folder", "name", "label", "category"], errors="ignore")
    n_classes = int(labels.max() + 1)

    val_frac = OmegaConf.select(cfg, "experiment.validation_split", default=0.2)
    tr_idx, va_idx = train_test_split(np.arange(len(labels)), test_size=val_frac,
                                      stratify=labels, random_state=42)
    df_tr, df_va = df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)
    y_tr, y_va   = labels[tr_idx], labels[va_idx]

    device  = get_device(cfg)
    out_dir = Path("outputs") / cfg.experiment.metadata.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tracker = _start_tracker("train", out_dir, cfg.experiment.metadata.tag)
    train_tracker.start()

    # ───── Phase 1: experts ──────────────────────────────────────────────
    print("\n== PHASE 1: training per-class experts ==")
    experts: List[nn.Module] = []
    for cls in range(n_classes):
        print(f"  ↳ expert for class {cls}")
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

        exp_dir = out_dir / f"expert_{cls}"; exp_dir.mkdir(exist_ok=True)
        train_pytorch_local(
            args=cfg.experiment, model=model, train_loader=tr_ld, val_loader=va_ld,
            class_weights=class_w, num_columns=in_dim, device=device,
            fold_folder=str(exp_dir), resume_checkpoint=False,
            checkpoint_path=str(exp_dir / "best.pth"),
        )
        experts.append(model.eval())

    tr_full_ds = EmbeddingDataset(df_tr); tr_ld_full = DataLoader(tr_full_ds, batch_size=cfg.experiment.model.batch_size)
    va_full_ds = EmbeddingDataset(df_va); va_ld_full = DataLoader(va_full_ds, batch_size=cfg.experiment.model.batch_size)
    collect = lambda ld: torch.cat([_pos_prob(exp, ld, device).unsqueeze(1) for exp in experts], dim=1)
    X_tr, X_va = collect(tr_ld_full), collect(va_ld_full)

    print("\n== PHASE 2: training router ==")
    router_cfg = getattr(cfg, "router", {})
    router = Router(num_experts=n_classes)
    _train_router(router, X_tr, y_tr, device,
                  epochs=router_cfg.get("epochs", 75),
                  lr=router_cfg.get("lr", 2e-3),
                  batch_size=router_cfg.get("batch_size", 256))

    train_tracker.stop()
    train_stats = _load_cc_csv(out_dir / "emissions_train.csv")

    val_tracker = _start_tracker("val", out_dir, cfg.experiment.metadata.tag)
    val_tracker.start()

    router.eval()
    with torch.no_grad():
        logits_va = router(X_va.to(device))
        probs_va  = torch.sigmoid(logits_va).cpu().numpy()
        y_pred    = probs_va.argmax(axis=1)

    val_tracker.stop()
    val_stats = _load_cc_csv(out_dir / "emissions_val.csv")

    acc  = accuracy_score(y_va, y_pred)
    w_f1 = f1_score(y_va, y_pred, average="weighted")

    print("\n== RESULTS ==")
    print(f" Validation accuracy     : {acc:.4f}")
    print(f" Validation weighted-F1 : {w_f1:.4f}\n")

    plot_multiclass_roc_curve(y_va, probs_va, EXPERIMENT_NAME=str(out_dir))

    
    cc_keys = [
        "project_name", "duration","emissions",
        "emissions_rate","cpu_power","gpu_power","ram_power","cpu_energy","gpu_energy",
        "ram_energy","energy_consumed", "cpu_count","cpu_model","gpu_count","gpu_model",
        "ram_total_size"
    ]

    metrics = {"accuracy": acc, "weighted_f1": w_f1}
    for k in cc_keys:
        metrics[f"train_{k}"] = train_stats.get(k)
        metrics[f"val_{k}"]   = val_stats.get(k)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

@hydra.main(version_base=None, config_path=str(ROOT / "config"), config_name="esc50")
def main(cfg: DictConfig):
    for name, meta in cfg.experiment.datasets.items():
        csv_path = Path(meta.csv)
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        print(f"\n=== DATASET {name.upper()} → outputs/{cfg.experiment.metadata.tag}")
        run_holdout_moe(str(csv_path), cfg)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
