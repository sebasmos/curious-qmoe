# train_pytorch(), plot_lossesimport os
import time
import torch
import pandas as pd
import numpy as np
from torch import nn

import torch.optim as optim
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
import gc

import torch
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

def plot_multiclass_roc_curve(all_labels, all_predictions, EXPERIMENT_NAME="."):
    # Step 1: Label Binarization
    label_binarizer = LabelBinarizer()
    y_onehot = label_binarizer.fit_transform(all_labels)
    all_predictions_hot = label_binarizer.transform(all_predictions)

    # Step 2: Calculate ROC curves
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    unique_classes = range(y_onehot.shape[1])
    for i in unique_classes:
        fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], all_predictions_hot[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Step 3: Plot ROC curves
    fig, ax = plt.subplots(figsize=(8, 8))

    # Micro-average ROC curve
    fpr_micro, tpr_micro, _ = roc_curve(y_onehot.ravel(), all_predictions_hot.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(
        fpr_micro,
        tpr_micro,
        label=f"micro-average ROC curve (AUC = {roc_auc_micro:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    # Macro-average ROC curve
    all_fpr = np.unique(np.concatenate([fpr[i] for i in unique_classes]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in unique_classes:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(unique_classes)
    fpr_macro = all_fpr
    tpr_macro = mean_tpr
    roc_auc_macro = auc(fpr_macro, tpr_macro)
    plt.plot(
        fpr_macro,
        tpr_macro,
        label=f"macro-average ROC curve (AUC = {roc_auc_macro:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    # Individual class ROC curves with unique colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
    for class_id, color in zip(unique_classes, colors):
        plt.plot(
            fpr[class_id],
            tpr[class_id],
            color=color,
            label=f"ROC curve for Class {class_id} (AUC = {roc_auc[class_id]:.2f})",
            linewidth=2,
        )

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2)  # Add diagonal line for reference
    plt.axis("equal")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Extension of Receiver Operating Characteristic\n to One-vs-Rest multiclass")
    plt.legend()
    plt.savefig(f'{EXPERIMENT_NAME}/roc_curve.png')

def plot_losses(train_losses, val_losses, EXPERIMENT_NAME="."):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))

    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    os.makedirs(EXPERIMENT_NAME, exist_ok=True)
    plt.savefig(os.path.join(EXPERIMENT_NAME, 'losses.png'))
    plt.close()

def save_confusion_matrix(all_labels, all_preds, unique_classes, experiment_name, phase):
    cm = confusion_matrix(all_labels, all_preds)
    conf_matrix = pd.DataFrame(cm, index=unique_classes, columns=unique_classes)

    plt.figure(figsize=(5, 4))
    ax = sns.heatmap(conf_matrix, annot=True, fmt='.1f', annot_kws={"size": 8},
                     cmap="crest", linewidths=0.1, cbar=True)

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.set_xticks(range(len(unique_classes)))
    ax.set_yticks(range(len(unique_classes)))

    ax.set_xticks([i + 0.5 for i in range(len(unique_classes))])
    ax.set_yticks([i + 0.5 for i in range(len(unique_classes))])

    plt.savefig(f'{experiment_name}/confusion_matrix_{phase}.png', dpi=300, bbox_inches='tight')
    plt.close()
def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def eval_pytorch_model(model, val_loader, device):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device) 
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
    return all_labels, all_preds
    
def train_pytorch(args, model, train_loader, val_loader, class_weights, num_columns, device, seed_dir,
                  resume_checkpoint=False, checkpoint_path=None):
    best_f1 = 0.0
    patience_counter = 0
    patience = 50

    optimizer = torch.optim.Adam(model.parameters(), lr=args.model.learning_rate)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.8)

    train_losses, val_losses = [], []

    if resume_checkpoint and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))

    for epoch in range(args.training.epochs):
        model.train()
        epoch_train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        epoch_val_loss = 0
        all_labels, all_preds = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        print(f"Epoch {epoch+1}/{args.training.epochs}, F1 Score: {f1:.4f}")
        scheduler.step(f1)

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            if args.logging.save_checkpoint:
                torch.save(model.state_dict(), os.path.join(seed_dir, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    return model, train_losses, val_losses, best_f1


def k_trainer_pytorch_esc(args, model, train_loader, val_loader, class_weights, num_columns, device, seed_dir):
    patience = args.training.early_stopping_patience  
    best_f1 = 0.0
    patience_counter = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.8)

    train_losses, val_losses = [], []
    
    
    start_train_time = time.time()
    for epoch in range(args.training.num_epochs):
        print(f"Starting epoch {epoch + 1}/{args.training.num_epochs}")
        model.train()
        epoch_train_loss = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss /= max(1, len(train_loader.dataset))
        train_losses.append(epoch_train_loss)

        model.eval()
        epoch_val_loss = 0
        all_labels, all_preds = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        epoch_val_loss /= max(1, len(val_loader.dataset))
        val_losses.append(epoch_val_loss)
        
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        print(f"Epoch {epoch+1}/{args.training.num_epochs}, F1 Score: {f1:.4f}")
        scheduler.step(f1)

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            model_path = os.path.join(seed_dir, 'best_model.pth')
            os.makedirs(seed_dir, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved at {model_path}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break
    
    return model, train_losses, val_losses, f1, all_labels, all_preds