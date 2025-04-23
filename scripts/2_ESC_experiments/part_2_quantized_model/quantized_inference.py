import os
import time
import torch
import pandas as pd
import torch.quantization
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

        device = torch.device(args.device)
        val_csv = os.path.join(args.data_path, 'val_embeddings.csv')
        print(f"Val data: {val_csv}")

        source_experiment_folder = f"{args.model}_{os.path.basename(args.data_path)}"
        target_experiment_folder = f"{source_experiment_folder}_quantized_model"
        source_output_folder = os.path.join(args.output_dir, source_experiment_folder)
        target_output_folder = os.path.join(args.output_dir, target_experiment_folder)

        source_seed_dir = os.path.join(source_output_folder, f"seed_{seed}")
        target_seed_dir = os.path.join(target_output_folder, f"seed_{seed}")

        # Create necessary directories
        os.makedirs(source_seed_dir, exist_ok=True)
        os.makedirs(target_output_folder, exist_ok=True)
        os.makedirs(target_seed_dir, exist_ok=True)

        # Load validation dataset
        val_dataset = EmbeddingDataset(val_csv, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        num_columns = val_dataset.features.shape[1]
        model = SClassifier(num_columns, args.num_classes, hidden_sizes=[256, 128, 64]).to(device)

        # Load the trained model
        model_path = os.path.join(source_seed_dir, "best_model.pth")
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found.")
            continue

        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Start Quantization
        torch.backends.quantized.engine = 'qnnpack'  # Set quantization engine for CPU: Ensures quantization is optimized for mobile/CPU execution.
        
        model_quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

        model_quantized.eval()
        for param in model_quantized.parameters():
            param.requires_grad_(False)

        # End Quantization
        
        # Perform inference
        start_inference_time = time.time()
        all_labels, all_preds = eval_pytorch_model(model_quantized, val_loader, device)
        end_inference_time = time.time()

        # Compute memory usage
        model_memory_eval = get_memory_usage()
        inference_time = end_inference_time - start_inference_time

        print("total_cpu_memory_inference (MB): ", model_memory_eval)

        training_results_path = os.path.join(source_seed_dir, 'training_results.json')
        if not os.path.exists(training_results_path):
            print(f"Error: Training results file {training_results_path} not found.")
            continue

        with open(training_results_path, 'r') as file:
            training_results = json.load(file)

        model_memory_train = training_results.get("model_memory_train", 0)
        training_time = training_results.get("train_time_seconds", 0)

        # Compute metrics
        metrics = get_all_metrics(
            val_dataset.features.shape[1], all_labels, all_preds, model_memory_train, model_memory_eval, training_time, inference_time
        )
        print(f'Inference time: {inference_time:.2f} seconds')
        print(metrics)

        # Save predictions and metrics
        save_predictions(target_seed_dir, target_experiment_folder, model, metrics, all_labels, all_preds, seed)
        plot_multiclass_roc_curve(all_labels, all_preds, target_seed_dir)
        save_confusion_matrix(all_labels, all_preds, np.unique(val_dataset.labels), target_seed_dir, "val")

    consolidate_and_average_metrics(args, target_output_folder)
    print("Evaluation across different seeds completed. Results saved to the output directory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference on trained models across multiple seeds")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset directory containing CSV files')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for inference')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save quantized results')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
    parser.add_argument('--model', type=str, choices=['mlp', 'svc'], default='mlp', help='Model type used during training')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--total_num_seed', type=int, default=1, help='Total number of seeds for evaluation')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)