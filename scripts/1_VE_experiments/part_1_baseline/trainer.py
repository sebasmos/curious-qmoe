import os
import time
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import random
import pickle
import sys
import time
import argparse
from torch.profiler import profile, ProfilerActivity
import pandas as pd
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from qnet.graphics import *
from qnet.metrics import *
from qnet.memory import *
from qnet.embeddings import *
from qnet.model import *
from qnet.utils import *
from qnet.train import *

def main(args):

    for seed in range(args.total_num_seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f'SEED {seed}')
        print('--------------------------------')
        
        torch.manual_seed(42)
        device = torch.device(args.device)
        train_csv = os.path.join(args.data_path, 'train_embeddings.csv')
        val_csv = os.path.join(args.data_path, 'val_embeddings.csv')
        print(f"Train data: {train_csv}")
        print(f"Val data: {val_csv}")
        
        experiment_folder = f"{args.model}_{args.data_path.split('/')[-1]}"
        output_folder = os.path.join(os.getcwd(), args.output_dir, experiment_folder)
        seed_dir = os.path.join(output_folder, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        
        train_dataset = EmbeddingDataset(train_csv, shuffle=True)
        val_dataset = EmbeddingDataset(val_csv, shuffle=False)
            
        print(f"Embeddings shapes".center(60, "-"))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

            # Class weights
        class_counts = np.bincount(np.asarray(train_dataset.labels, int))
        class_weights = 1.0 / class_counts
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        num_columns = train_dataset.features.shape[1]
            
        model = SClassifier(num_columns, args.num_classes, hidden_sizes=[256, 128, 64]).to(device)
        model.apply(reset_weights)

        start_train_time = time.time()
        model_trained, train_losses, val_losses, f1 = train_pytorch(args, model, train_loader, val_loader, class_weights, num_columns, device, seed_dir)
        end_train_time = time.time()

        model_memory_train = get_memory_usage()
        print("model_memory_train: ", model_memory_train)
        print("F1 on eval data: ", f1)

        # Prepare output folder
        output_folder = args.output_dir
        os.makedirs(output_folder, exist_ok=True)
        combined_metrics = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "f1_score": f1,
            "model_memory_train": model_memory_train,
            "train_time_seconds": end_train_time - start_train_time
        }

        output_path = os.path.join(output_folder, seed_dir, "training_results.json")

        with open(output_path, "w") as f:
            json.dump(combined_metrics, f, indent=4)

        plot_losses(train_losses, val_losses, seed_dir)                   
        print(f"Results saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a simple MLP or SVC with K-Fold Cross Validation")
    parser.add_argument('--data_path', type=str, default='./data', help='Path to the dataset directory containing CSV files')
    parser.add_argument('--device', type=str, default='cpu', help='')
    parser.add_argument('--output_dir', type=str, default='./results_jan_29_test', help='Directory to save output metrics and models')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of K-Folds for cross-validation')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs (only applicable for MLP)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (only applicable for MLP)')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer (only applicable for MLP)')
    parser.add_argument('--model', type=str, choices=['mlp', 'svc'], default='mlp', help='Model to use: "mlp" or "svc"')
    parser.add_argument('--num_classes', type=int, default=5, help='num_classes')
    parser.add_argument('--total_num_seed', type=int, default=1, help='total_num_seed')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)

    