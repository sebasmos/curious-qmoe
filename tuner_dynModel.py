#!/usr/bin/env python
"""
Optuna-integrated K-Fold CV for tuning ESC audio classifier with dynamic MLP architecture search
"""
import os, json, warnings
import hydra
import optuna
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from QWave.datasets import EmbeddingDataset
from QWave.train_utils import train_pytorch_local
from QWave.graphics import plot_multiclass_roc_curve, plot_losses


def build_dynamic_mlp(input_size, output_size, trial):
    num_layers = trial.suggest_int("num_layers", 2, 6)
    activation_name = trial.suggest_categorical("activation", ["ReLU", "GELU", "ELU"])
    activation_cls = getattr(nn, activation_name)
    dropout_prob = trial.suggest_float("dropout_prob", 0.0, 0.5)
    use_layernorm = trial.suggest_categorical("use_layernorm", [True, False])

    layers = []
    in_dim = input_size

    for i in range(num_layers):
        out_dim = trial.suggest_int(f"hidden_dim_{i}", 64, 1024, step=64)
        layers.append(nn.Linear(in_dim, out_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(out_dim))
        layers.append(activation_cls())
        layers.append(nn.Dropout(dropout_prob))
        in_dim = out_dim

    layers.append(nn.Linear(in_dim, output_size))
    layers.append(nn.LogSoftmax(dim=1))
    return nn.Sequential(*layers)


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def objective(trial, base_cfg, csv_path):
    OmegaConf.set_struct(base_cfg, False)

    base_cfg.experiment.model.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    base_cfg.experiment.model.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    base_cfg.experiment.model.label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
    base_cfg.experiment.model.dropout_prob = trial.suggest_float("dropout_prob", 0.0, 0.5)
    base_cfg.experiment.model.batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    base_cfg.experiment.model.epochs = trial.suggest_int("epochs", 50, 150)
    base_cfg.experiment.model.patience = trial.suggest_int("patience", 5, 20)
    base_cfg.experiment.model.factor = trial.suggest_float("factor", 0.5, 0.9)
    base_cfg.experiment.metadata.tag = f"optuna_trial_{trial.number}"
    base_cfg.experiment.logging.resume = False

    df_full = pd.read_csv(csv_path)
    labels = df_full["class_id"].values
    df = df_full.drop(columns=["folder", "name", "label", "category"])

    skf = StratifiedKFold(
        n_splits=base_cfg.experiment.cross_validation.n_splits,
        shuffle=base_cfg.experiment.cross_validation.shuffle,
        random_state=base_cfg.experiment.cross_validation.random_seed
    )

    fold_f1s = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)

        tr_ds = EmbeddingDataset(df_train)
        va_ds = EmbeddingDataset(df_val)

        tr_ld = torch.utils.data.DataLoader(tr_ds, batch_size=base_cfg.experiment.model.batch_size, shuffle=True)
        va_ld = torch.utils.data.DataLoader(va_ds, batch_size=base_cfg.experiment.model.batch_size, shuffle=False)

        class_weights = torch.tensor(1.0 / np.bincount(tr_ds.labels.numpy()), dtype=torch.float32)
        in_dim = tr_ds.features.shape[1]
        num_classes = len(np.unique(labels))

        model = build_dynamic_mlp(in_dim, num_classes, trial)
        model.apply(reset_weights)

        model, _, _, best_f1, _, _, _ = train_pytorch_local(
            args=base_cfg.experiment,
            model=model,
            train_loader=tr_ld,
            val_loader=va_ld,
            class_weights=class_weights,
            num_columns=in_dim,
            device="cpu",
            fold_folder="/tmp",
            resume_checkpoint=False,
            checkpoint_path=None
        )
        fold_f1s.append(best_f1)

    return float(np.mean(fold_f1s))


@hydra.main(version_base=None, config_path=".", config_name="configs")
def main(cfg: DictConfig):
    dataset_name = list(cfg.experiment.datasets.keys())[0]
    csv_path = cfg.experiment.datasets[dataset_name].csv
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, cfg, csv_path), n_trials=100)

    print("\nBest trial:")
    print(f"  F1 Score: {study.best_trial.value:.4f}")
    print("  Architecture and hyperparameters:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()