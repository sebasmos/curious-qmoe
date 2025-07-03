#!/usr/bin/env python
"""
Optuna-integrated K-Fold CV for tuning ESC audio classifier
"""
import os, json, warnings
import hydra
import optuna
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from QWave.datasets import EmbeddingDataset
from QWave.train_utils import train_pytorch_local
from QWave.graphics import plot_multiclass_roc_curve, plot_losses

from torch import nn


class ESCModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[128, 64], dropout_prob=0.3, activation_fn=nn.ReLU, use_residual=True):
        super(ESCModel, self).__init__()
        self.use_residual = use_residual
        self.fc_layers = self._create_fc_layers(input_size, hidden_sizes, dropout_prob, activation_fn)
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.activation_fn = activation_fn()
        self.softmax = nn.LogSoftmax(dim=1)

    def _create_fc_layers(self, input_size, hidden_sizes, dropout_prob, activation_fn):
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))  
            layers.append(activation_fn())
            layers.append(nn.Dropout(p=dropout_prob))
            prev_size = hidden_size
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        x = self.fc_layers(x)
        if self.use_residual and residual.shape == x.shape:
            x += residual 
        x = self.output_layer(x)
        x = self.softmax(x)
        return x


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def objective(trial, base_cfg, csv_path):
    base_cfg.experiment.model.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    base_cfg.experiment.model.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    base_cfg.experiment.model.label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
    base_cfg.experiment.model.dropout_prob = trial.suggest_float("dropout_prob", 0.0, 0.5)
    base_cfg.experiment.model.batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    base_cfg.experiment.model.epochs = trial.suggest_int("epochs", 50, 150)
    base_cfg.experiment.model.patience = trial.suggest_int("patience", 5, 20)
    base_cfg.experiment.model.factor = trial.suggest_float("factor", 0.5, 0.9)
    base_cfg.experiment.model.hidden_sizes = [
        trial.suggest_int("hidden_size1", 256, 1024, step=128),
        trial.suggest_int("hidden_size2", 128, 512, step=64),
        trial.suggest_int("hidden_size3", 64, 256, step=32),
    ]
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

        model = ESCModel(
            in_dim, num_classes,
            hidden_sizes=base_cfg.experiment.model.hidden_sizes,
            dropout_prob=base_cfg.experiment.model.dropout_prob
        )
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
    study.optimize(lambda trial: objective(trial, cfg, csv_path), n_trials=200)

    print("Best trial:")
    print(study.best_trial.value)
    print(study.best_trial.params)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
