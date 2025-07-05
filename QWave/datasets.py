# EmbeddingDataset
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler # Import these

from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, shuffle=False):
        self.df = df.copy()
        labels = self.df["class_id"].to_numpy(dtype=np.int64)
        features = self.df.drop(columns=["class_id"]).to_numpy(dtype=np.float32)

        if shuffle:
            indices = np.random.permutation(len(features))
            features, labels = features[indices], labels[indices]

        print("Data shape:")
        print(f"  Features shape: {features.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Unique labels: {np.unique(labels)}")
        print(f"  Feature min/max: {features.min():.2f} / {features.max():.2f}")

        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class EmbeddingAdaptDataset(Dataset):
    def __init__(self, df: pd.DataFrame, normalization_type: str = "raw", scaler=None, shuffle=False):
        self.df = df.copy()

        # Identify non-feature columns
        # Ensure 'folder', 'name', 'label', 'category' are dropped from features
        # 'class_id' is explicitly handled as labels
        non_feature_cols = ["folder", "name", "label", "category", "class_id"]
        feature_df = self.df.drop(columns=non_feature_cols, errors='ignore')

        labels = self.df["class_id"].to_numpy(dtype=np.int64)
        features = feature_df.to_numpy(dtype=np.float32)

        if shuffle:
            indices = np.random.permutation(len(features))
            features, labels = features[indices], labels[indices]

        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

        self.normalization_type = normalization_type.lower()
        self.scaler = scaler # Will be None for training dataset, or fitted scaler for val/test

        print("--- Initial Feature Stats ---")
        print(f"  Features shape: {self.features.shape}")
        print(f"  Labels shape: {self.labels.shape}")
        print(f"  Unique labels: {np.unique(labels)}")
        print(f"  Feature min/max: {self.features.min().item():.4f} / {self.features.max().item():.4f}")
        print(f"  Feature mean/std: {self.features.mean().item():.4f} / {self.features.std().item():.4f}")

        self._apply_normalization()

        print("--- Feature Stats After Normalization ---")
        print(f"  Feature min/max: {self.features.min().item():.4f} / {self.features.max().item():.4f}")
        print(f"  Feature mean/std: {self.features.mean().item():.4f} / {self.features.std().item():.4f}")
        if self.normalization_type == "l2":
            # Check a few norms to ensure they're close to 1
            sample_norms = torch.linalg.norm(self.features[:5], dim=1).cpu().numpy()
            print(f"  L2 Norms (sample): {[f'{n:.4f}' for n in sample_norms]}")
            print(f"  L2 Norms (mean): {torch.linalg.norm(self.features, dim=1).mean().item():.4f}")


    def _apply_normalization(self):
        if self.normalization_type == "raw":
            print("Normalization: RAW (no scaling applied).")
            # No action needed, features are already tensors
        elif self.normalization_type == "standard":
            print("Normalization: STANDARD SCALING (Z-score).")
            if self.scaler is None: # This is the training dataset
                self.scaler = StandardScaler()
                self.features = torch.tensor(self.scaler.fit_transform(self.features.cpu().numpy()), dtype=torch.float32)
            else: # This is validation/test dataset, use pre-fitted scaler
                self.features = torch.tensor(self.scaler.transform(self.features.cpu().numpy()), dtype=torch.float32)
        elif self.normalization_type == "min_max":
            print("Normalization: MIN-MAX SCALING ([0, 1] range).")
            if self.scaler is None: # This is the training dataset
                self.scaler = MinMaxScaler()
                self.features = torch.tensor(self.scaler.fit_transform(self.features.cpu().numpy()), dtype=torch.float32)
            else: # This is validation/test dataset, use pre-fitted scaler
                self.features = torch.tensor(self.scaler.transform(self.features.cpu().numpy()), dtype=torch.float32)
        elif self.normalization_type == "l2":
            print("Normalization: L2 NORMALIZATION (unit vectors).")
            # L2 normalization is applied per sample, so no need for a pre-fitted scaler
            # Add a small epsilon to prevent division by zero for zero vectors (unlikely)
            self.features = self.features / (torch.linalg.norm(self.features, dim=1, keepdim=True) + 1e-6)
        else:
            raise ValueError(f"Unknown normalization_type: {self.normalization_type}. Choose from 'raw', 'standard', 'min_max', 'l2'.")

    def get_scaler(self):
        """Returns the fitted scaler for 'standard' or 'min_max' normalization."""
        if self.normalization_type in ["standard", "min_max"]:
            if self.scaler is None:
                raise RuntimeError("Scaler has not been fitted yet. Call this only after initializing a training dataset.")
            return self.scaler
        return None # Return None if L2 or raw (no external scaler needed)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]