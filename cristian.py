#!/usr/bin/env python
"""
K-Fold CV on spectrogram images using GFNet pretrained on ImageNet
"""
import os, json, warnings, time, datetime
from functools import partial
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn
from PIL import Image
from pathlib import Path

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from gfnet import GFNetPyramid
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
import QWave.utils2 as utils2
from QWave.utils import get_device

CATEGORY_TO_ID = {
    "Animals": 0,
    "Natural soundscapes": 1,
    "Human non-speech": 2,
    "Interior sounds": 3,
    "Exterior noises": 4
}
ID_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_ID.items()}

class ESCImageDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['name']
        label = row['class_id']
        try:
            img = Image.open(path).convert("RGB")
            return (self.transform(img), label), 0  # Provide dummy domain for unpacking
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return (torch.zeros((3, 224, 224)), -1), 0

def get_transforms(cfg: DictConfig):
    input_size = cfg.data.input_size
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=cfg.data.get('random_horiz_flip', 0.5)),
        transforms.ColorJitter(
            brightness=cfg.data.get('jitter', 0.4),
            contrast=cfg.data.get('jitter', 0.4),
            saturation=cfg.data.get('jitter', 0.4),
            hue=0
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def run_fold(fold_id: int, df_train: pd.DataFrame, df_val: pd.DataFrame, cfg: DictConfig, out_dir: str):
    print(f"\n--- FOLD {fold_id + 1}/{cfg.experiment.cross_validation.n_splits} ---")
    fold_out_dir = os.path.join(out_dir, f"fold_{fold_id + 1}")
    os.makedirs(fold_out_dir, exist_ok=True)
    output_dir_path = Path(fold_out_dir)

    device = get_device(cfg)
    print(f"Using device: {device}")

    train_transform = get_transforms(cfg)
    val_transform = get_transforms(cfg)

    train_dataset = ESCImageDataset(df_train, train_transform)
    val_dataset = ESCImageDataset(df_val, val_transform)

    data_loader_train = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_mem,
        drop_last=True,
    )
    data_loader_val = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_mem,
        drop_last=False,
    )

    n_classes = len(ID_TO_CATEGORY)
    print(f"Number of classes: {n_classes}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    mixup_fn = None
    mixup_active = cfg.training.mixup > 0 or cfg.training.cutmix > 0. or cfg.training.cutmix_minmax is not None
    if mixup_active:
        print("Mixup/Cutmix enabled.")
        mixup_fn = Mixup(
            mixup_alpha=cfg.training.mixup, cutmix_alpha=cfg.training.cutmix, cutmix_minmax=cfg.training.cutmix_minmax,
            prob=cfg.training.mixup_prob, switch_prob=cfg.training.mixup_switch_prob, mode=cfg.training.mixup_mode,
            label_smoothing=cfg.training.smoothing, num_classes=n_classes)
    else:
        print("Mixup/Cutmix disabled.")

    model = GFNetPyramid(
        img_size=cfg.data.input_size,
        patch_size=4,
        num_classes=n_classes,
        embed_dim=[64, 128, 256, 512], depth=[3, 3, 10, 3],
        mlp_ratio=[4, 4, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_path_rate=cfg.model.drop_path
    )

    if cfg.model.finetune:
        finetune_path = Path(cfg.model.finetune)
        if finetune_path.is_dir():
            finetune_path = finetune_path / (cfg.model.arch + ".pth")
        if finetune_path.exists():
            checkpoint = torch.load(finetune_path, map_location='cpu')
            checkpoint_model = checkpoint.get('model', checkpoint)
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint due to shape mismatch.")
                    del checkpoint_model[k]
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print("Pretrained weights load status:", msg)

    model.to(device)

    optimizer = create_optimizer(cfg.optimizer, model)
    lr_scheduler, _ = create_scheduler(cfg.scheduler, optimizer)
    loss_scaler = NativeScaler() if cfg.training.amp else None

    if mixup_active:
        criterion = SoftTargetCrossEntropy()
    elif cfg.training.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=cfg.training.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    criterion = DistillationLoss(criterion, None, 'none', 0.0, 1.0).to(device)

    train_stats = train_one_epoch(
        model, criterion, data_loader_train,
        optimizer, device, 0, loss_scaler,
        cfg.optimizer.clip_grad, None, mixup_fn,
        set_training_mode=cfg.training.get('set_training_mode', True),
        amp=cfg.training.amp
    )

    val_stats = evaluate(data_loader_val, model, device)
    return val_stats['acc1'], 0

@hydra.main(config_path=".", config_name="configs_2")
def main(cfg: DictConfig):
    image_root = hydra.utils.to_absolute_path(cfg.data.image_root)
    image_paths = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith(".png")])
    rows = []
    for path in image_paths:
        fname = os.path.basename(path)
        parts = fname[:-4].split("-")
        if len(parts) != 4:
            continue
        _, _, _, label_str = parts
        label = int(label_str)
        category_map = {
            **{i: "Animals" for i in [0,1,2,3,4,25,26,27,28,29]},
            **{i: "Natural soundscapes" for i in [5,6,7,8,9,30,31,32,33,34]},
            **{i: "Human non-speech" for i in [10,11,12,13,14,35,36,37,38,39]},
            **{i: "Interior sounds" for i in [15,16,17,18,19,40,41,42,43,44]},
            **{i: "Exterior noises" for i in [20,21,22,23,24,45,46,47,48,49]}
        }
        category = category_map.get(label, "Unknown")
        if category == "Unknown":
            continue
        class_id = CATEGORY_TO_ID[category]
        rows.append((path, label, category, class_id))

    df_full = pd.DataFrame(rows, columns=["name", "label", "category", "class_id"])
    labels = df_full["class_id"].values

    skf = StratifiedKFold(
        n_splits=cfg.experiment.cross_validation.n_splits,
        shuffle=cfg.experiment.cross_validation.shuffle,
        random_state=cfg.experiment.cross_validation.random_seed
    )

    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        df_train = df_full.iloc[train_idx]
        df_val = df_full.iloc[val_idx]
        val_acc, _ = run_fold(fold, df_train, df_val, cfg, out_dir)
        fold_metrics.append(val_acc)

    mean_acc = np.mean(fold_metrics)
    std_acc = np.std(fold_metrics)
    print("\n--- Cross-Validation Summary ---")
    print(f"Fold Accuracies: {[f'{acc:.2f}%' for acc in fold_metrics]}")
    print(f"Mean Validation Accuracy: {mean_acc:.2f}%")
    print(f"Standard Deviation: {std_acc:.2f}%")

    results = {
        "fold_accuracies": fold_metrics,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "config": OmegaConf.to_container(cfg, resolve=True)
    }
    results_path = os.path.join(out_dir, "cv_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"CV results saved to {results_path}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="Argument interpolation should be")
    warnings.filterwarnings("ignore", category=UserWarning, module='torchvision.transforms.functional')
    main()
