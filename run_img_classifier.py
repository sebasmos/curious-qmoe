#!/usr/bin/env python
"""
K-Fold CV on spectrogram images using EfficientNetB3 pretrained on ImageNet
"""
import os, json, warnings
import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torch.utils.data import Dataset, DataLoader
from torch import nn
from PIL import Image
from QWave.train_utils import train_pytorch_local
from QWave.graphics import plot_multiclass_roc_curve, plot_losses
from QWave.utils import get_device

CATEGORY_TO_ID = {
    "Animals": 0,
    "Natural soundscapes": 1,
    "Human non-speech": 2,
    "Interior sounds": 3,
    "Exterior noises": 4
}

class ESCImageDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['name']  # should contain full image path
        label = row['class_id']
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

def run_cv(image_root: str, cfg: DictConfig):
    # Load metadata from image filenames
    image_paths = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith(".png")])
    if not image_paths:
        raise RuntimeError(f"No PNG images found in {image_root}")

    rows = []
    for path in image_paths:
        fname = os.path.basename(path)
        parts = fname[:-4].split("-")
        if len(parts) != 4:
            continue
        fold, clip_id, take, label = parts
        label = int(label)
        category = { 0:"Animals",1:"Animals",2:"Animals",3:"Animals",4:"Animals",
        5:"Natural soundscapes",6:"Natural soundscapes",7:"Natural soundscapes",
        8:"Natural soundscapes",9:"Natural soundscapes",10:"Human non-speech",
        11:"Human non-speech",12:"Human non-speech",13:"Human non-speech",
        14:"Human non-speech",15:"Interior sounds",16:"Interior sounds",
        17:"Interior sounds",18:"Interior sounds",19:"Interior sounds",
        20:"Exterior noises",21:"Exterior noises",22:"Exterior noises",
        23:"Exterior noises",24:"Exterior noises",25:"Animals",26:"Animals",
        27:"Animals",28:"Animals",29:"Animals",30:"Natural soundscapes",
        31:"Natural soundscapes",32:"Natural soundscapes",33:"Natural soundscapes",
        34:"Natural soundscapes",35:"Human non-speech",36:"Human non-speech",
        37:"Human non-speech",38:"Human non-speech",39:"Human non-speech",
        40:"Interior sounds",41:"Interior sounds",42:"Interior sounds",
        43:"Interior sounds",44:"Interior sounds",45:"Exterior noises",
        46:"Exterior noises",47:"Exterior noises",48:"Exterior noises",
        49:"Exterior noises" }[label]
        class_id = CATEGORY_TO_ID[category]
        rows.append((path, label, category, class_id))

    df_full = pd.DataFrame(rows, columns=["name", "label", "category", "class_id"])
    labels = df_full["class_id"].values
    print("Found", len(df_full), "images → class IDs:", np.unique(labels))

    skf = StratifiedKFold(
        n_splits=cfg.experiment.cross_validation.n_splits,
        shuffle=cfg.experiment.cross_validation.shuffle,
        random_state=cfg.experiment.cross_validation.random_seed
    )

    fold_metrics = []
    tag = cfg.experiment.metadata.tag
    out_dir = os.path.abspath(os.path.join("outputs", tag))
    os.makedirs(out_dir, exist_ok=True)

    device = get_device(cfg)
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n--- FOLD {fold+1}/{skf.n_splits} ---")

        df_train = df_full.iloc[train_idx].reset_index(drop=True)
        df_val = df_full.iloc[val_idx].reset_index(drop=True)

        tr_ds = ESCImageDataset(df_train, transform)
        va_ds = ESCImageDataset(df_val, transform)

        tr_ld = DataLoader(tr_ds, batch_size=cfg.experiment.model.batch_size, shuffle=True)
        va_ld = DataLoader(va_ds, batch_size=cfg.experiment.model.batch_size, shuffle=False)

        model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_dim = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_dim, len(CATEGORY_TO_ID))
        model.to(device)

        class_weights = torch.tensor(1.0 / np.bincount(df_train["class_id"].values), dtype=torch.float32).to(device)

        fold_dir = os.path.join(out_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        ckpt_path = os.path.join(fold_dir, "best_model.pth")
        resume = cfg.experiment.logging.resume and os.path.exists(ckpt_path)
        if resume:
            try:
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                print(f"Resumed from checkpoint: {ckpt_path}")
            except RuntimeError as e:
                print(f"⚠️  Checkpoint load failed: {e}\n→ Starting from scratch.")
                resume = False

        model, train_losses, val_losses, best_f1, all_labels, all_preds, all_probs = train_pytorch_local(
            args=cfg.experiment,
            model=model,
            train_loader=tr_ld,
            val_loader=va_ld,
            class_weights=class_weights,
            num_columns=300*300,
            device=device,
            fold_folder=fold_dir,
            resume_checkpoint=resume,
            checkpoint_path=ckpt_path
        )

        plot_multiclass_roc_curve(all_labels, all_probs, EXPERIMENT_NAME=fold_dir)
        plot_losses(train_losses, val_losses, fold_dir)
        fold_metrics.append(dict(best_f1=best_f1))
        print(f"Best F1: {best_f1:.4f}")
        print(f"Saving metrics to {fold_dir}/metrics.json")
        with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
            json.dump({
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_f1": best_f1
            }, f, indent=4)

    summary = {
        "f1_mean": float(np.mean([m["best_f1"] for m in fold_metrics])),
        "f1_std": float(np.std([m["best_f1"] for m in fold_metrics])),
        "metadata": dict(cfg.experiment.metadata),
        "folds": fold_metrics
    }
    print(f"Saving summary to {out_dir}/summary.json")
    print(summary)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(json.dumps(summary, indent=4))

@hydra.main(version_base=None, config_path=".", config_name="configs")
def main(cfg: DictConfig):
    for name, meta in cfg.experiment.datasets.items():
        image_root = meta.imgs  # we overload this to mean "image folder"
        if not os.path.isdir(image_root):
            raise FileNotFoundError(image_root)
        print(f"\n=== DATASET {name.upper()} → outputs/{cfg.experiment.metadata.tag}")
        run_cv(image_root, cfg)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()