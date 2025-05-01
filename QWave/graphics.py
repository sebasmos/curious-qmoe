import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import torch
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def plot_training_curves(train_losses, val_losses, fold_folder):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", linestyle='dashed')
    plt.plot(val_losses, label="Validation Loss", linestyle='solid')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fold_folder, "loss_plot.png"))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, fold_folder):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(fold_folder, "confusion_matrix.png"))
    plt.close()



def plot_multiclass_roc_curve(all_labels, all_probs, EXPERIMENT_NAME="."):
    os.makedirs(EXPERIMENT_NAME, exist_ok=True)

    # Convert list of probs to numpy array
    all_probs = np.array(all_probs)
    # Binarize true labels
    label_binarizer = LabelBinarizer()
    y_onehot = label_binarizer.fit_transform(all_labels)

    # If binary classification, reshape
    # Convert to numpy array (important!)
    if y_onehot.shape[1] == 1:
        y_onehot = np.hstack((1 - y_onehot, y_onehot))
        all_probs = np.array(all_probs)
        if all_probs.shape[1] == 1:
            all_probs = np.hstack((1 - all_probs, all_probs))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_onehot.shape[1]

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot.ravel(), all_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label=f"micro-average ROC (AUC = {roc_auc['micro']:.2f})",
             color="deeppink", linestyle=":", linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],
             label=f"macro-average ROC (AUC = {roc_auc['macro']:.2f})",
             color="navy", linestyle=":", linewidth=4)

    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                 label=f"Class {label_binarizer.classes_[i]} (AUC = {roc_auc[i]:.2f})", linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{EXPERIMENT_NAME}/roc_curve.png")
    plt.close()
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
    ax.set_xticks([i + 0.5 for i in range(len(unique_classes))])
    ax.set_yticks([i + 0.5 for i in range(len(unique_classes))])

    os.makedirs(experiment_name, exist_ok=True)
    plt.savefig(f'{experiment_name}/confusion_matrix_{phase}.png', dpi=300, bbox_inches='tight')
    plt.close()


def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    # no plt.show() here
    return