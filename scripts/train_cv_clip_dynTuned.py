#!/usr/bin/env python
"""
Optuna-integrated K-Fold CV for tuning the projection layer from 1536 → 512 in CLIP-style audio classifier
"""
import os, json, warnings
import hydra
import optuna
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader, TensorDataset
from QWave.utils import get_device

def generate_clip_text_embeddings(class_names, processor, model, device):
    # prompts = [f"This is a Soundscape Spectrogram capturing an area characterized by {name}" for name in class_names]
    prompts = [f"sound of a {name}" for name in class_names]
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    text_features = torch.nn.functional.normalize(text_features, dim=-1)
    return prompts, text_features

def evaluate_clip_embeddings(features, true_labels, text_embeddings):
    features = torch.nn.functional.normalize(features, dim=-1)
    sims = features @ text_embeddings.T
    preds = torch.argmax(sims, dim=1).cpu().numpy()
    true_labels = np.array(true_labels)
    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='macro')
    return acc, f1

def objective(trial, cfg, csv_path):
    OmegaConf.set_struct(cfg, False)

    cfg.experiment.model.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    cfg.experiment.model.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    cfg.experiment.model.batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    cfg.experiment.model.epochs = trial.suggest_int("epochs", 10, 100)

    print(f"Trial {trial.number} - lr={cfg.experiment.model.learning_rate:.2e}, wd={cfg.experiment.model.weight_decay:.2e}, bs={cfg.experiment.model.batch_size}, epochs={cfg.experiment.model.epochs}")

    df_full = pd.read_csv(csv_path)
    labels = df_full["class_id"].values
    features = df_full.drop(columns=["folder", "class_id", "name", "label", "category"])
    class_names = df_full["category"].unique()

    device = get_device(cfg)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()
    _, text_embeddings = generate_clip_text_embeddings(class_names, processor, model, device)

    skf = StratifiedKFold(
        n_splits=cfg.experiment.cross_validation.n_splits,
        shuffle=cfg.experiment.cross_validation.shuffle,
        random_state=cfg.experiment.cross_validation.random_seed
    )

    fold_f1s = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"  Fold {fold+1}/{skf.n_splits}")

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
        criterion = torch.nn.CrossEntropyLoss()

        train_ds = TensorDataset(train_features, train_labels)
        train_dl = DataLoader(train_ds, batch_size=cfg.experiment.model.batch_size, shuffle=True)

        for epoch in range(cfg.experiment.model.epochs):
            projection.train()
            epoch_loss = 0.0
            for xb, yb in train_dl:
                logits = torch.nn.functional.normalize(projection(xb), dim=-1) @ text_embeddings.T
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            # print(f"    Epoch {epoch+1:02d} - Loss: {epoch_loss / len(train_ds):.4f}")

        projection.eval()
        with torch.no_grad():
            val_proj = torch.nn.functional.normalize(projection(val_features), dim=-1)

        acc, f1 = evaluate_clip_embeddings(val_proj, val_labels, text_embeddings)
        print(f"    Fold {fold+1} F1: {f1:.4f}, Accuracy: {acc:.4f}")
        fold_f1s.append(f1)

    mean_f1 = float(np.mean(fold_f1s))
    print(f"Trial {trial.number} completed - Mean F1: {mean_f1:.4f}")
    return mean_f1

@hydra.main(version_base=None, config_path=".", config_name="configs")
def main(cfg: DictConfig):
    dataset_name = list(cfg.experiment.datasets.keys())[0]
    csv_path = cfg.experiment.datasets[dataset_name].csv
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    print("Starting Optuna hyperparameter search...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, cfg, csv_path), n_trials=50)

    print("\nBest trial:")
    print(f"  F1 Score: {study.best_trial.value:.4f}")
    print("  Hyperparameters:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
