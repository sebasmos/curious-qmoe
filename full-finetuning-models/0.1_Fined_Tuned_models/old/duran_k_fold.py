# 02/07/2024
import os
import csv
import argparse
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from PIL import Image
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def get_args_parser():
    parser = argparse.ArgumentParser(description="Train a model with k-fold cross-validation.")
    parser.add_argument('--experiment_name', type=str, default="BS_Pytorch_EPOCHS_CV")
    parser.add_argument('--im_size', type=tuple, default=(224, 224))
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=65)
    parser.add_argument('--data_path', type=str, default='/home/sebastian/codes/data/ABGQI_mel_spectrograms')
    parser.add_argument('--csv_file', type=str, default='/home/sebastian/codes/data/ABGQI_mel_spectrograms/Pruebas_Rutas.csv')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--early_stopping', type=int, default=5, help="Number of epochs with no improvement after which training will be stopped.")
    return parser

def parse_args():
    parser = get_args_parser()
    if 'ipykernel' in sys.modules:
        args, unknown = parser.parse_known_args()
    else:
        args = parser.parse_args()
    return args

def generate_images_data_csv(main_folder, csv_file):
    images_data = []
    for subdir, _, files in os.walk(main_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                name = file
                class_name = os.path.basename(subdir)
                path = os.path.join(subdir, file).replace('\\', '/')
                images_data.append([name, class_name, path])
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['name', 'class', 'path'])
        writer.writerows(images_data)
    print(f'Se ha generado el archivo {csv_file} con los datos de las imágenes.')

def validate_model(model, criterion, data_loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_loss = running_loss / len(data_loader)
    val_accuracy = correct / total
    return val_loss, val_accuracy

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNet, self).__init__()
        self.base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

def plot_training_history(history, fold, experiment_name, epochs):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    experiment_folder = f"{experiment_name}/fold_{fold + 1}"
    os.makedirs(experiment_folder, exist_ok=True)
    plt.savefig(f'{experiment_folder}/loss_plot.png')
    plt.show()
    plt.figure()
    plt.plot(epochs, history['train_accuracy'], label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{experiment_folder}/accuracy_plot.png')
    plt.show()

def train_k_fold_model(fold, train_loader, valid_loader, num_classes, experiment_name, early_stopping_patience,epochs):
    model = CustomEfficientNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': [], 'lr': []}
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
    total_train_time = 0
    total_val_time = 0
    for epoch in range(epochs):
        if epoch < 25:
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        else:
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_start_time = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_end_time = time.time()
        train_duration = train_end_time - train_start_time
        total_train_time += train_duration
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        val_start_time = time.time()
        val_loss, val_accuracy = validate_model(model, criterion, valid_loader)
        val_end_time = time.time()
        val_duration = val_end_time - val_start_time
        total_val_time += val_duration
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        print(f"Training time for epoch [{epoch+1}/{epochs}]: {train_duration:.4f} seconds")
        print(f"Validation time for epoch [{epoch+1}/{epochs}]: {val_duration:.4f} seconds")
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    fold_training_time = total_train_time
    fold_validation_time = total_val_time
    experiment_folder = f"{experiment_name}/fold_{fold + 1}"
    os.makedirs(experiment_folder, exist_ok=True)
    model_path = os.path.join(experiment_folder, 'model_weights.pth')
    torch.save(model.state_dict(), model_path)
    plot_training_history(history, fold, experiment_name, epochs)
    val_labels, val_preds = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())
    val_labels = np.array(val_labels)
    val_preds = np.array(val_preds)
    y_true = label_binarize(val_labels, classes=np.arange(num_classes))
    y_pred = label_binarize(val_preds, classes=np.arange(num_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve (class {i}) (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for each class')
    plt.legend(loc='lower right')
    plt.savefig(f'{experiment_folder}/roc_curve_plot.png')
    plt.show()
    precision = precision_score(val_labels, val_preds, average='micro', zero_division=0)
    recall = recall_score(val_labels, val_preds, average='micro', zero_division=0)
    f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
    beta = 0.75
    f0_75 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    class_precision = precision_score(val_labels, val_preds, average=None, zero_division=0)
    class_recall = recall_score(val_labels, val_preds, average=None, zero_division=0)
    fold_confusion_matrix = confusion_matrix(val_labels, val_preds)
    confusion_matrix_df = pd.DataFrame(fold_confusion_matrix)
    confusion_matrix_df.to_csv(f'{experiment_folder}/confusion_matrix.csv')
    metrics_df = pd.DataFrame({
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1],
        'f0.75': [f0_75],
        'training_time': [fold_training_time],
        'validation_time': [fold_validation_time]
    })
    metrics_df.to_csv(f'{experiment_folder}/metrics.csv', index=False)
    class_metrics_df = pd.DataFrame({
        'class': class_names,
        'precision': class_precision,
        'recall': class_recall
    })
    class_metrics_df.to_csv(f'{experiment_folder}/class_metrics.csv', index=False)
    print(f"Fold {fold + 1} - Precision: {precision}, Recall: {recall}, F1 Score: {f1}, F0.75 Score: {f0_75}, Training Time: {fold_training_time:.2f} seconds, Validation Time: {fold_validation_time:.2f} seconds")
    return precision, recall, f1, f0_75, fold_confusion_matrix, class_precision, class_recall, fold_training_time, fold_validation_time

def main():
    args = parse_args()
    if torch.cuda.is_available():
        print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available.")
    experiment_name = args.experiment_name
    im_size = args.im_size
    batch_size = args.batch_size
    num_folds = args.num_folds
    epochs = args.epochs
    data_path = args.data_path
    csv_file = args.csv_file
    seed = args.seed
    early_stopping_patience = args.early_stopping
    np.random.seed(seed)
    torch.manual_seed(seed)
    global class_names
    class_names = sorted(os.listdir(data_path))
    df = pd.read_csv(csv_file)
    print("Columnas del DataFrame:", df.columns)
    label_encoder = LabelEncoder()
    df['class'] = label_encoder.fit_transform(df['class'])
    data_transform = transforms.Compose([
        transforms.Resize(im_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.RandomResizedCrop(im_size),
        transforms.ToTensor()
    ])
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, df, transform=None):
            self.df = df
            self.transform = transform
        def __len__(self):
            return len(self.df)
        def __getitem__(self, idx):
            img_path = self.df.iloc[idx]['path']
            image = Image.open(img_path).convert('RGB')
            label = self.df.iloc[idx]['class']
            label = torch.tensor(label).long()
            if self.transform:
                image = self.transform(image)
            return image, label
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    all_f0_75_scores = []
    all_class_precisions = []
    all_class_recalls = []
    all_training_times = []
    all_validation_times = []
    start_time = time.time()
    for fold, (train_idx, val_idx) in enumerate(skf.split(df['path'], df['class'])):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        train_dataset = CustomDataset(train_df, transform=data_transform)
        val_dataset = CustomDataset(val_df, transform=data_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        print(f"Training fold {fold + 1} of {num_folds}")
        precision, recall, f1, f0_75, fold_confusion_matrix, class_precision, class_recall, fold_training_time, fold_validation_time = train_k_fold_model(fold, train_loader, val_loader, len(df['class'].unique()), experiment_name, early_stopping_patience, epochs)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1_scores.append(f1)
        all_f0_75_scores.append(f0_75)
        all_class_precisions.append(class_precision)
        all_class_recalls.append(class_recall)
        all_training_times.append(fold_training_time)
        all_validation_times.append(fold_validation_time)
    mean_precision = np.mean(all_precisions)
    mean_recall = np.mean(all_recalls)
    mean_f1_score = np.mean(all_f1_scores)
    mean_f0_75_score = np.mean(all_f0_75_scores)
    mean_class_precision = np.mean(all_class_precisions, axis=0)
    mean_class_recall = np.mean(all_class_recalls, axis=0)
    mean_training_time = np.mean(all_training_times)
    mean_validation_time = np.mean(all_validation_times)
    mean_total_time = mean_training_time + mean_validation_time
    end_time = time.time()
    execution_time = end_time - start_time
    total_minutes, total_seconds = divmod(execution_time, 60)
    print(f"Average Precision: {mean_precision:.4f}")
    print(f"Average Recall: {mean_recall:.4f}")
    print(f"Average F1 Score: {mean_f1_score:.4f}")
    print(f"Average F0.75 Score: {mean_f0_75_score:.4f}")
    print(f"Class Precision: {mean_class_precision}")
    print(f"Class Recall: {mean_class_recall}")
    print(f"Average Training Time per Fold: {mean_training_time:.2f} seconds")
    print(f"Average Validation Time per Fold: {mean_validation_time:.2f} seconds")
    print(f"Average Total Time per Fold: {mean_total_time:.2f} seconds")
    print(f"Total Execution Time: {total_minutes:.0f} minutes {total_seconds:.2f} seconds")
    results_folder = experiment_name
    os.makedirs(results_folder, exist_ok=True)
    final_metrics_df = pd.DataFrame({
        'mean_precision': [mean_precision],
        'mean_recall': [mean_recall],
        'mean_f1_score': [mean_f1_score],
        'mean_f0.75_score': [mean_f0_75_score],
        'mean_training_time': [mean_training_time],
        'mean_validation_time': [mean_validation_time],
        'mean_total_time': [mean_total_time],
        'total_execution_time': [f"{total_minutes:.0f} minutes {total_seconds:.2f} seconds"]
    })
    final_metrics_df.to_csv(os.path.join(results_folder, 'final_metrics.csv'), index=False)
    class_metrics_df = pd.DataFrame({
        'class': class_names,
        'mean_class_precision': mean_class_precision,
        'mean_class_recall': mean_class_recall
    })
    class_metrics_df.to_csv(os.path.join(results_folder, 'class_metrics.csv'), index=False)

if __name__ == "__main__":
    main()

# Para ejecutar el script en Jupyter Notebook:
main_folder = '/home/sebastian/codes/data/ABGQI_mel_spectrograms'
csv_file = '/home/sebastian/codes/data/ABGQI_mel_spectrograms/Pruebas_Rutas.csv'
# Descomentar la siguiente línea para ejecutar en Jupyter Notebook
generate_images_data_csv(main_folder, csv_file)