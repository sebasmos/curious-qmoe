import os
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
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
    parser = argparse.ArgumentParser(description="Train and test a model with folder-based data structure.")
    parser.add_argument('--experiment_name', type=str, default="BS_Pytorch_EPOCHS_BCE")
    parser.add_argument('--im_size', type=tuple, default=(300, 300))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=65)
    parser.add_argument('--data_path', type=str, default='/home/sebastian/codes/data/ABGQI_mel_spectrograms/')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--early_stopping', type=int, default=10)
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

def create_data_loaders(data_path, im_size, batch_size):
    train_transform = transforms.Compose([
        transforms.Resize(im_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.RandomResizedCrop(im_size),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize(im_size),
        transforms.CenterCrop(im_size),
        transforms.ToTensor()
    ])

    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')
    test_dir = os.path.join(data_path, 'test')

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(train_dataset.classes)
    return train_loader, val_loader, test_loader, num_classes

def validate_model(model, criterion, data_loader, num_classes):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.argmax(dim=1).cpu().numpy())

    val_loss = running_loss / len(data_loader)
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    return val_loss, val_accuracy, val_f1, all_labels, all_preds

def train_model(train_loader, val_loader, num_classes, experiment_name, early_stopping_patience, epochs):
    model = CustomEfficientNet(num_classes=num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(patience=early_stopping_patience)

    for epoch in range(epochs):
        optimizer = optim.Adam(model.parameters(), lr=0.001 if epoch < 25 else 0.0001)
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels.argmax(dim=1)).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        val_loss, val_accuracy, val_f1, _, _ = validate_model(model, criterion, val_loader, num_classes)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    torch.save(model.state_dict(), f"{experiment_name}/model_weights.pth")
    return model

def test_model(model, test_loader, num_classes):
    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_accuracy, test_f1, all_labels, all_preds = validate_model(model, criterion, test_loader, num_classes)

    print(f"\nTest Metrics:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score (weighted): {test_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_loader, val_loader, test_loader, num_classes = create_data_loaders(args.data_path, args.im_size, args.batch_size)
    os.makedirs(args.experiment_name, exist_ok=True)

    model = train_model(train_loader, val_loader, num_classes, args.experiment_name, args.early_stopping, args.epochs)
    test_model(model, test_loader, num_classes)

if __name__ == "__main__":
    main()