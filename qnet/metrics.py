from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
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
from torch.profiler import profile, ProfilerActivity


def get_all_metrics_total(num_feat, all_labels, all_preds, total_cpu_memory_train,training_time):
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_075 = fbeta_score(all_labels, all_preds, beta=0.75, average='weighted', zero_division=0)
    # print("total_cpu_memory (MB): ", total_mem)
    metrics = {
        'EMB_size_out': num_feat,
        'Accuracy': accuracy,
        'F1 Score': f1,
        'F0.75 Score': f1_075,
        'Precision': precision,
        'Recall': recall,
        'Total time (s)': training_time,
        'Mem. Total (MB)': total_cpu_memory_train, 
        }
    
    print(metrics)
    
    return metrics

def get_all_metrics(num_feat, all_labels, all_preds, total_cpu_memory_train, total_cpu_memory_inference,training_time, inference_time):
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_075 = fbeta_score(all_labels, all_preds, beta=0.75, average='weighted', zero_division=0)
    # print("total_cpu_memory (MB): ", total_mem)
    metrics = {
        'EMB_size_out': num_feat,
        'Accuracy': accuracy,
        'F1 Score': f1,
        'F0.75 Score': f1_075,
        'Precision': precision,
        'Recall': recall,
        'Training time': training_time,
        'Inference time': inference_time,
        'Mem. Train (MB)': total_cpu_memory_train, 
        'Mem. Infer (MB)': total_cpu_memory_inference, 
        }
    
    print(metrics)

    return metrics
    
def compute_f0_75_score_mean(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    beta = 0.75
    f0_75_scores = []
    for p, r in zip(precision, recall):
        if p + r == 0:
            f0_75 = 0.0
        else:
            f0_75 = (1 + beta**2) * (p * r) / (beta**2 * p + r)
        f0_75_scores.append(f0_75)
    
    return np.mean(f0_75_scores)