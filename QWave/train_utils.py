import os
import time
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from QWave.graphics import plot_training_curves, plot_multiclass_roc_curve, plot_losses, show_image, save_confusion_matrix, plot_confusion_matrix


def train_pytorch_local(args, model, train_loader, val_loader, class_weights, num_columns, device, fold_folder,
                        resume_checkpoint=False, checkpoint_path=None):
    best_f1 = 0.0
    patience_counter = 0
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.model.learning_rate, weight_decay=args.model.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.model.label_smoothing)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=args.model.patience, factor=args.model.factor)

    train_losses, val_losses = [], []

    if resume_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model = model.to(device)

    for epoch in range(args.model.epochs):
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
        all_labels, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                probs = torch.exp(outputs)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        print(f"Epoch {epoch+1}/{args.model.epochs}, F1 Score: {f1:.4f}")
        scheduler.step(f1)

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            if getattr(args, "save_checkpoint", True):
                torch.save(model.state_dict(), os.path.join(fold_folder, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= args.model.patience:
                print("Early stopping triggered")
                break

    plot_training_curves(train_losses, val_losses, fold_folder)
    plot_confusion_matrix(all_labels, all_preds, fold_folder)

    return model, train_losses, val_losses, best_f1, all_labels, all_preds, all_probs


def train_pytorch(args, model, train_loader, val_loader, class_weights, num_columns, device, seed_dir,
                  resume_checkpoint=False, checkpoint_path=None):
    best_f1 = 0.0
    patience_counter = 0
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.model.learning_rate)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.model.label_smoothing)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=args.model.patience, factor=args.model.factor)

    train_losses, val_losses = [], []

    if resume_checkpoint and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model = model.to(device)

    for epoch in range(args.model.epochs):
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
        print(f"Epoch {epoch+1}/{args.model.epochs}, F1 Score: {f1:.4f}")
        scheduler.step(f1)

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            if args.logging.save_checkpoint:
                torch.save(model.state_dict(), os.path.join(seed_dir, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= args.model.patience:
                print("Early stopping triggered")
                break

    return model, train_losses, val_losses, best_f1


def eval_pytorch_model(model, val_loader, device):
    model.eval()
    model = model.to(device)
    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    return all_labels, all_preds