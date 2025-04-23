import os
import csv
import argparse
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from PIL import Image
import time
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
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
    parser = argparse.ArgumentParser(description="Train a model without k-fold cross-validation.")
    parser.add_argument('--experiment_name', type=str, default="BS_Pytorch_EPOCHS_BCE")
    parser.add_argument('--im_size', type=tuple, default=(224, 224))
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=65)
    parser.add_argument('--data_path', type=str, default='/home/sebastian/codes/data/ABGQI_mel_spectrograms')
    parser.add_argument('--csv_file', type=str, default='/home/sebastian/codes/data/ABGQI_mel_spectrograms/Pruebas_Rutas.csv')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--early_stopping', type=int, default=5)
    return parser

def parse_args():
    parser = get_args_parser()
    if 'ipykernel' in sys.modules:
        args, unknown = parser.parse_known_args()
    else:
        args = parser.parse_args()
    return args

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNet, self).__init__()
        self.base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

def validate_model(model, criterion, data_loader,num_classes):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels.argmax(dim=1)).sum().item()
            total += labels.size(0)
    val_loss = running_loss / len(data_loader)
    val_accuracy = correct / total
    return val_loss, val_accuracy

def train_model(train_loader, valid_loader, num_classes, experiment_name, early_stopping_patience, epochs):

    model = CustomEfficientNet(num_classes=num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()  # BinaryCrossentropy loss
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
    total_train_time = 0
    total_val_time = 0

    for epoch in range(epochs):
        # Apply learning rate decay after the first 25 epochs
        if epoch < 25:
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        else:
            optimizer = optim.Adam(model.parameters(), lr=0.00001)  # Learning rate after 25 epochs

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_start_time = time.time()

        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
            loss = criterion(outputs, labels.float())  # Convert labels to float for BCEWithLogitsLoss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            # correct += (preds == labels).sum().item()
            correct += (preds == labels.argmax(dim=1)).sum().item()
            total += labels.size(0)

        train_duration = time.time() - train_start_time
        total_train_time += train_duration
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        val_loss, val_accuracy = validate_model(model, criterion, valid_loader, num_classes)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    torch.save(model.state_dict(), f"{experiment_name}/model_weights.pth")
    return history

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    global class_names
    class_names = sorted(os.listdir(args.data_path))
    df = pd.read_csv(args.csv_file)
    label_encoder = LabelEncoder()
    df['class'] = label_encoder.fit_transform(df['class'])

    data_transform = transforms.Compose([
        transforms.Resize(args.im_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.RandomResizedCrop(args.im_size),
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

    dataset = CustomDataset(df, transform=data_transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    os.makedirs(args.experiment_name, exist_ok=True)

    history = train_model(train_loader, val_loader, len(class_names), args.experiment_name, args.early_stopping, args.epochs)

if __name__ == "__main__":
    main()