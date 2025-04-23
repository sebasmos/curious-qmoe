import os
import time
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import psutil
import argparse
import sys
import json
# export PYTHONPATH="../quantumVM:$PYTHONPATH"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from qnet.graphics import *
from qnet.metrics import *
from qnet.memory import *
from qnet.embeddings import *
from qnet.model import *
from qnet.utils import *
from qnet.train import *

"""
python k_fold_validation.py --data_path /home/sebastian/codes/repo_clean/VE_paper/0_VE_extraction/part0_ESC-VE-extraction/efficientnet_b3_16_bs64  --num_epochs 2 --model mlp
"""
import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import argparse

def save_split_data(fold_dir, train_data, test_data):
    train_data.to_csv(os.path.join(fold_dir, 'train_split.csv'), index=False)
    test_data.to_csv(os.path.join(fold_dir, 'test_split.csv'), index=False)

def calculate_avg_metrics(fold_dir, k_folds):
    
    metrics_list = []
    for fold in range(k_folds):
        metrics_file = os.path.join(fold_dir, f'fold_{fold}', f'metrics_seed_{fold}.csv')
        # import pdb;pdb.set_trace()
        if os.path.exists(metrics_file):
            df = pd.read_csv(metrics_file)
            metrics_list.append(df)
        else:
            print(f"Metrics file for fold {fold} not found at {metrics_file}. Skipping this fold.")
    if not metrics_list:
        print("No metrics files found. Cannot calculate average metrics.")
        return
    avg_metrics = pd.concat(metrics_list).mean().to_frame().T
    avg_metrics.to_csv(os.path.join(fold_dir, 'average_metrics.csv'), index=False)
    print("Average metrics calculated and saved to 'average_metrics.csv'.")

def main(args):
    torch.manual_seed(42)
    
    device = torch.device(args.device)
    all_data = pd.DataFrame()
    for i in range(args.k_folds):
        csv_path = os.path.join(args.data_path, f'fold_{i+1}.csv')
        fold_data = pd.read_csv(csv_path)
        fold_data['fold'] = i
        all_data = pd.concat([all_data, fold_data], ignore_index=True)
    
    for fold in range(args.k_folds):
        print(f'FOLD {fold}')
        print('--------------------------------')
                
        experiment_folder = f"{args.model}_{args.data_path.split('/')[-1]}"
        output_folder = os.path.join(args.output_dir, experiment_folder)  # Use os.path.join to combine paths
        fold_dir = os.path.join(output_folder, "fold_"+str(fold))
        os.makedirs(fold_dir, exist_ok=True)
        
        test_data = all_data[all_data['fold'] == fold]
        train_data = all_data[all_data['fold'] != fold]

        test_data = test_data.drop(columns=["fold"])
        train_data = train_data.drop(columns=["fold"])

        print(test_data.shape, train_data.shape)
        
        train_dataset = EmbeddingDatasetESC(train_data)
        test_dataset = EmbeddingDatasetESC(test_data)

        output_folder = os.path.join( args.output_dir, f"{args.model}_{args.data_path.split('/')[-1]}")
        os.makedirs(output_folder, exist_ok=True)

        save_split_data(fold_dir, train_data, test_data)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        num_columns = train_dataset.features.shape[1]
        model = SClassifier(num_columns, args.num_classes, hidden_sizes=[64]).to(args.device)
        model.apply(reset_weights)

        class_counts = np.bincount(np.asarray(train_dataset.labels, int))
        
        class_weights = 1.0 / class_counts
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        
        start_train_time = time.time()
        model_trained, train_losses, val_losses, f1 = train_pytorch(args, model, train_loader, val_loader, class_weights, num_columns, args.device, fold_dir)
        end_train_time = time.time()
        print("Class weights shape:", class_weights.shape)

        model_memory_train = get_memory_usage()
        print("model_memory_train: ", model_memory_train)
        print("F1 on eval data: ", f1)

        combined_metrics = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "f1_score": f1,
            "model_memory_train": model_memory_train,
            "train_time_seconds": end_train_time - start_train_time
        }
        output_path = os.path.join(output_folder, fold_dir, "training_results.json")

        with open(output_path, "w") as f:
            json.dump(combined_metrics, f, indent=4)

        plot_losses(train_losses, val_losses, fold_dir)                   
        print(f"Results saved to {output_path}")

        start_inference_time = time.time()
        all_labels, all_preds = eval_pytorch_model(model, val_loader, args.device)
        end_inference_time = time.time()

        model_memory_eval = get_memory_usage()
        inference_time = end_inference_time - start_inference_time

        print("total_cpu_memory_inference (MB): ", model_memory_eval)

        metrics = get_all_metrics(
            train_dataset.features.shape[1], all_labels, all_preds, model_memory_train, model_memory_eval, end_train_time - start_train_time, inference_time
        )
        print(f'Inference time: {inference_time:.2f} seconds')
        print(metrics)
        save_predictions(fold_dir, experiment_folder, model, metrics, all_labels, all_preds, fold)
        
        plot_multiclass_roc_curve(all_labels, all_preds, fold_dir)
        
        save_confusion_matrix(all_labels, all_preds, np.unique(train_dataset.labels.numpy()), fold_dir, "val")
    
    calculate_avg_metrics(output_folder, args.k_folds)
    print("K-Fold Cross Validation and testing completed. Results saved to the output directory.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a simple MLP or SVC with K-Fold Cross Validation")
    parser.add_argument('--data_path', type=str, default='./data', help='Path to the dataset directory containing CSV files')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output metrics and models')
    parser.add_argument('--device', type=str, default='cpu', help='')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of K-Folds for cross-validation')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of K-Folds for cross-validation')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs (only applicable for MLP)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training (only applicable for MLP)')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer (only applicable for MLP)')
    parser.add_argument('--model', type=str, default='mlp', help='Model to use: "mlp" or "svc"')

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)