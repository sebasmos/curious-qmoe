# EmbeddingDataset
import torch
import os
import pandas as pd
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
from torch.ao.quantization.observer import MinMaxObserver
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.ao.quantization.observer import MinMaxObserver
import os

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.quantization import MinMaxObserver

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