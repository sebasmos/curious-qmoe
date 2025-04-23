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
# from torch.ao.quantization.observer import MinMaxObserver

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset



from torch.quantization import MinMaxObserver

class QuantizedEmbeddingDatasetv31(Dataset):
    def __init__(self, csv_path, save_dir='quantized_embeddings', train=True, save_embeddings=False):
        self.train = train
        self.save_dir = save_dir

        # Load data directly as tensors
        df = pd.read_csv(csv_path)
        self.labels = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)
        features_tensor = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)

        if self.train:
            indices = torch.randperm(len(df))
            features_tensor = features_tensor[indices]
            self.labels = self.labels[indices]

        # Quantize once during initialization
        observer = MinMaxObserver(dtype=torch.qint8, quant_min=-128, quant_max=127)
        observer(features_tensor)
        scale, zero_point = observer.calculate_qparams()
        
        self.features = torch.quantize_per_tensor(
            features_tensor,
            scale=scale,
            zero_point=zero_point,
            dtype=torch.qint8
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return quantized tensor WITHOUT dequantization
        return self.features[idx], self.labels[idx]

    
from torch.ao.quantization.observer import MinMaxObserver

class QuantizedEmbeddingDataset(Dataset):
    """
    A dataset class for handling quantized embeddings derived from a CSV file. 
    This class processes the data by reading from the provided CSV path, applying 
    quantization to the feature columns, and optionally saving the quantized 
    embeddings to a specified directory. The dataset can be used for training 
    or validation purposes, with the capability to shuffle the data.

    Inputs:
        csv_path (str): The path to the CSV file containing the dataset. 
                        The last column of the CSV is assumed to be the labels.
        save_dir (str): The directory where quantized embeddings will be saved. 
                        Default is 'quantized_embeddings'.
        train (bool): A flag to indicate whether the dataset is for training or validation. 
                      Default is True, which shuffles the dataset.
        save_embeddings (bool): A flag to indicate whether to save the quantized 
                                embeddings to a CSV file. Default is False.

    Outputs:
        A PyTorch Dataset object that provides access to quantized features 
        and corresponding labels.
    """
    def __init__(self, csv_path, save_dir='quantized_embeddings', train=True, save_embeddings=False):
        self.train = train
        self.save_dir = save_dir

        # Load CSV efficiently
        df = pd.read_csv(csv_path)
        self.original_column_names = df.columns.tolist()

        features = df.iloc[:, :-1].to_numpy(dtype=np.float32)
        labels = df.iloc[:, -1].to_numpy(dtype=np.int64)

        print("Data shape: ", features.shape, labels.shape)

        if self.train:
            indices = np.random.permutation(len(df))
            features, labels = features[indices], labels[indices]

        self.labels = labels

        # Optimize observer usage
        observer = MinMaxObserver(dtype=torch.qint8, quant_min=-128, quant_max=127)
        observer(torch.tensor(features))  # Apply observer
        self.scale, self.zero_point = observer.calculate_qparams()
        self.scale, self.zero_point = self.scale.item(), self.zero_point.item()

        # Directly quantize with NumPy for efficiency
        self.features = np.round(features / self.scale + self.zero_point).astype(np.int8)
        
        if save_embeddings:
            self.save_embeddings_to_parquet()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        quantized = self.features[idx]
        dequantized = (quantized.astype(np.float32) - self.zero_point) * self.scale
        return torch.tensor(dequantized, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

    def save_embeddings_to_parquet(self):
        """Save embeddings efficiently using Parquet instead of CSV."""
        os.makedirs(self.save_dir, exist_ok=True)
        df = pd.DataFrame(self.features, columns=self.original_column_names[:-1])
        df['labels'] = self.labels
        file_name = 'quantized_embeddings_train.parquet' if self.train else 'quantized_embeddings_val.parquet'
        file_path = os.path.join(self.save_dir, file_name)
        df.to_parquet(file_path, index=False)
        print(f"Quantized embeddings saved to {file_path}")
# class QuantizedEmbeddingDataset(Dataset):
#     """
#     A dataset class for handling quantized embeddings derived from a CSV file. 
#     This class processes the data by reading from the provided CSV path, applying 
#     quantization to the feature columns, and optionally saving the quantized 
#     embeddings to a specified directory. The dataset can be used for training 
#     or validation purposes, with the capability to shuffle the data.

#     Inputs:
#         csv_path (str): The path to the CSV file containing the dataset. 
#                         The last column of the CSV is assumed to be the labels.
#         save_dir (str): The directory where quantized embeddings will be saved. 
#                         Default is 'quantized_embeddings'.
#         train (bool): A flag to indicate whether the dataset is for training or validation. 
#                       Default is True, which shuffles the dataset.
#         save_embeddings (bool): A flag to indicate whether to save the quantized 
#                                 embeddings to a CSV file. Default is False.

#     Outputs:
#         A PyTorch Dataset object that provides access to quantized features 
#         and corresponding labels.
#     """
#     def __init__(self, csv_path, save_dir='quantized_embeddings', train=True, save_embeddings=False):
#         self.train = train
#         self.save_dir = save_dir
#         df = pd.read_csv(csv_path)
        
#         self.original_column_names = df.columns.tolist()
#         features = df.iloc[:, :-1].values.astype(np.float32)
#         labels = df.iloc[:, -1].values.astype(np.int64)
#         print("structure of data is: ",features.shape, labels.shape)
#         if self.train:
#             indices = np.random.permutation(len(df))
#             features = features[indices]
#             labels = labels[indices]
        
#         self.labels = labels
#         features_tensor = torch.tensor(features)
#         observer = MinMaxObserver(dtype=torch.qint8, quant_min=-128, quant_max=127)
#         observer(features_tensor)
#         self.scale, zero_point = observer.calculate_qparams()
#         self.scale = self.scale.item()
#         self.zero_point = zero_point.item()
        
#         quantized_tensor = torch.quantize_per_tensor(
#             features_tensor, 
#             self.scale, 
#             int(self.zero_point), 
#             dtype=torch.qint8
#         )
#         self.features = quantized_tensor.int_repr().numpy()

#         if save_embeddings:
#             self.save_embeddings_to_csv()

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         quantized = self.features[idx].astype(np.float32)
#         dequantized = (quantized - self.zero_point) * self.scale
#         return (
#             torch.tensor(dequantized, dtype=torch.float32),
#             torch.tensor(self.labels[idx], dtype=torch.long)
#         )

#     def save_embeddings_to_csv(self):
#         quantized_df = pd.DataFrame(self.features, columns=self.original_column_names[:-1])
#         quantized_df['labels'] = self.labels
#         file_name = 'quantized_embeddings_train.csv' if self.train else 'quantized_embeddings_val.csv'
#         file_path = os.path.join(self.save_dir, file_name)
#         quantized_df.to_csv(file_path, index=False)
#         print(f"Quantized embeddings saved to {file_path}")

class EmbeddingDatasetESC(Dataset):
    def __init__(self, data, shuffle=False):
        self.data = data
        self.features = torch.tensor(self.data.iloc[:, :-1].values, dtype=torch.float32)
        self.labels = torch.tensor(self.data.iloc[:, -1].values, dtype=torch.long)
        if shuffle:
            self.features = self.features.sample(frac=1).reset_index(drop=True)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return (feature, label)
    

class EmbeddingDataset_ESC(Dataset):
    def __init__(self, data_frame, shuffle=False):
        self.data = data_frame
        self.features = torch.tensor(self.data.filter(like="feat").values, dtype=torch.float32)
        self.labels = torch.tensor(self.data["label"].values, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    


    
class EmbeddingDataset(Dataset):
    def __init__(self, csv_file, shuffle=False):
        self.data = pd.read_csv(csv_file)
        # Convert features and labels to tensors
        self.features = torch.tensor(self.data.iloc[:, :-1].values, dtype=torch.float32)
        self.labels = torch.tensor(self.data.iloc[:, -1].values, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return (feature, label)
        
class EmbeddingDataset_stateless(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.labels = self.data.iloc[:, -1].values 
        self.features = self.data.iloc[:, :-1].values
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = self.features[idx].astype(np.float32)
        label = self.labels[idx]
        return torch.tensor(features), torch.tensor(label, dtype=torch.long)

def apply_pca(X, variance_threshold=0.95):
    pca = PCA(n_components=variance_threshold)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca

class classifier_embeddings(nn.Module):
    def __init__(self, base_model, feat_space, model_name):
        super(classifier_embeddings, self).__init__()
        self.base_model = base_model
        self.model_name = model_name
        self.feat_space = feat_space
        # Example: Adding a new classifier layer
        if model_name in ("mobilenetv4_r448", "mobilenetv4_r448_trained"):
            self.new_classifier = nn.Linear(self.base_model.conv_head.out_channels, out_features=self.feat_space)
        elif model_name in ("eva02_large_patch14_448_embeddings_imageNet"):
            self.new_classifier = nn.Linear(in_features=1024, out_features=feat_space)
    def forward(self, x):
        x = self.base_model(x)
        x = self.new_classifier(x)
        return x

def extract_embeddings(model, data_loader, save_path, device, preprocess=None,data_config=None, transforms=None):
    embeddings_list = []
    targets_list = []
    total_batches = len(data_loader)
    with torch.no_grad(), tqdm(total=total_batches) as pbar:
        model.eval()  # Set the model to evaluation mode
        model.to(device)
        for images, targets in data_loader:

            if preprocess:
                images = preprocess(images).squeeze()
                images = images.to(device)
                embeddings = model(images)
            if transforms: # for timm models
                images = images.to(device)   
                embeddings = model(transforms(images))# output is (batch_size, num_features) shaped tensor
            else:
                images = images.to(device)
                embeddings = model(images)
            embeddings_list.append(embeddings.cpu().detach().numpy())  # Move to CPU and convert to NumPy
            targets_list.append(targets.numpy())  # Convert targets to NumPy
            pbar.update(1)

    # Concatenate embeddings and targets from all batches
    embeddings = np.concatenate(embeddings_list).squeeze()
    targets = np.concatenate(targets_list)
    num_embeddings = embeddings.shape[1]
    column_names = [f"feat_{i}" for i in range(num_embeddings)]
    column_names.append("label")

    embeddings_with_targets = np.hstack((embeddings, np.expand_dims(targets, axis=1)))

    # Create a DataFrame with column names
    df = pd.DataFrame(embeddings_with_targets, columns=column_names)
    
    df.to_csv(save_path, index=False)

    return df


def extract_PCA_embeddings(model, data_loader, save_path, device, preprocess=None,data_config=None, transforms=None):
    embeddings_list = []
    targets_list = []
    total_batches = len(data_loader)
    with torch.no_grad(), tqdm(total=total_batches) as pbar:
        model.eval()  # Set the model to evaluation mode
        model.to(device)
        for images, targets in data_loader:

            if preprocess:
                images = preprocess(images).squeeze()
                images = images.to(device)
                embeddings = model(images)
            if transforms: # for timm models
                images = images.to(device)   
                embeddings = model(transforms(images))# output is (batch_size, num_features) shaped tensor
            else:
                images = images.to(device)
                embeddings = model(images)
            embeddings_list.append(embeddings.cpu().detach().numpy())  # Move to CPU and convert to NumPy
            targets_list.append(targets.numpy())  # Convert targets to NumPy
            pbar.update(1)

    # Concatenate embeddings and targets from all batches
    embeddings = np.concatenate(embeddings_list).squeeze()
    targets = np.concatenate(targets_list)
    num_embeddings = embeddings.shape[1]
    column_names = [f"feat_{i}" for i in range(num_embeddings)]
    column_names.append("label")

    embeddings_with_targets = np.hstack((embeddings, np.expand_dims(targets, axis=1)))

    # Create a DataFrame with column names
    df = pd.DataFrame(embeddings_with_targets, columns=column_names)
    
    df.to_csv(save_path, index=False)