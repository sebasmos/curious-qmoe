#!/usr/bin/env python
"""
K-Fold CV on precomputed Mel embeddings using CLIP's text encoder
"""
import os, json, warnings
import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader, TensorDataset
from QWave.utils import get_device

def generate_clip_text_embeddings(class_names, processor, model, device):
    prompts = [f"This is a Soundscape Spectrogram capturing an area characterized by {name}" for name in class_names]
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    text_features = torch.nn.functional.normalize(text_features, dim=-1)
    return prompts, text_features

def evaluate_clip_embeddings(features, true_labels, text_embeddings, label_names):
    features = torch.nn.functional.normalize(features, dim=-1)
    sims = features @ text_embeddings.T
    preds = torch.argmax(sims, dim=1).cpu().numpy()
    true_labels = np.array(true_labels)
    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='macro')
    return acc, f1

def run_cv(csv_path: str, cfg: DictConfig):
    df_full = pd.read_csv(csv_path)
    labels = df_full["class_id"].values
    features = df_full.drop(columns=["folder","class_id", "name", "label", "category"])
    class_names = df_full["category"].unique()

    print("Data shape:", features.shape)    
    print("Class names:", class_names)
    device = get_device(cfg)
    print(f"Using device: {device}")

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()
    _, text_embeddings = generate_clip_text_embeddings(class_names, processor, model, device)

    skf = StratifiedKFold(
        n_splits=cfg.experiment.cross_validation.n_splits,
        shuffle=cfg.experiment.cross_validation.shuffle,
        random_state=cfg.experiment.cross_validation.random_seed
    )

    fold_metrics = []
    out_dir = os.path.abspath(os.path.join("outputs", cfg.experiment.metadata.tag))
    os.makedirs(out_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n--- FOLD {fold+1}/{skf.n_splits} ---")

        train_features = torch.tensor(features.iloc[train_idx].values, dtype=torch.float32, device=device)
        train_labels   = torch.tensor(labels[train_idx], dtype=torch.long,  device=device)

        val_features   = torch.tensor(features.iloc[val_idx].values, dtype=torch.float32, device=device)
        val_labels     = labels[val_idx]

        projection = torch.nn.Linear(1536, 512, bias=False).to(device)
        optimizer  = torch.optim.AdamW(
            projection.parameters(),
            lr=cfg.experiment.model.learning_rate,
            weight_decay=cfg.experiment.model.weight_decay
        )
        criterion  = torch.nn.CrossEntropyLoss()

        train_ds = TensorDataset(train_features, train_labels)
        train_dl = DataLoader(train_ds, batch_size=cfg.experiment.model.batch_size, shuffle=True, drop_last=False)

        for epoch in range(cfg.experiment.model.epochs):
            projection.train()
            running_loss = 0.0

            for xb, yb in train_dl:
                logits = torch.nn.functional.normalize(projection(xb), dim=-1) @ text_embeddings.T
                loss   = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)

            epoch_loss = running_loss / len(train_ds)
            print(f"  epoch {epoch+1:02d}/{cfg.experiment.model.epochs} | loss {epoch_loss:.4f}")

        projection.eval()
        with torch.no_grad():
            val_proj = torch.nn.functional.normalize(projection(val_features), dim=-1)

        acc, f1 = evaluate_clip_embeddings(val_proj, val_labels, text_embeddings, class_names)

        fold_dir = os.path.join(out_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        torch.save(projection.state_dict(), os.path.join(fold_dir, "projection.pt"))

        fold_metrics.append(dict(clip_acc=acc, clip_f1=f1))
        with open(os.path.join(fold_dir, "clip_eval.json"), "w") as f:
            json.dump(fold_metrics[-1], f, indent=4)

    summary = {
        "acc_mean": float(np.mean([m["clip_acc"] for m in fold_metrics])),
        "acc_std" : float(np.std ([m["clip_acc"] for m in fold_metrics])),
        "f1_mean" : float(np.mean([m["clip_f1"] for m in fold_metrics])),
        "f1_std"  : float(np.std ([m["clip_f1"] for m in fold_metrics])),
        "metadata": dict(cfg.experiment.metadata),
        "folds"   : fold_metrics
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(json.dumps(summary, indent=4))

@hydra.main(version_base=None, config_path=".", config_name="configs")
def main(cfg: DictConfig):
    for name, meta in cfg.experiment.datasets.items():
        csv_path = meta.csv
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(csv_path)
        print(f"\n=== DATASET {name.upper()} → outputs/{cfg.experiment.metadata.tag}")
        run_cv(csv_path, cfg)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
