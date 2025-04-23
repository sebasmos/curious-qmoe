import os
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
    

def train_pytorch(args, model, train_loader, val_loader, class_weights, num_columns, device, seed_dir):
        patience = 50  
        best_f1 = 0.0
        patience_counter = 0

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.8)

        train_losses = []
        val_losses = []
        epoch_train_loss, epoch_val_loss = 0, 0
        for epoch in range(args.num_epochs):
            print(f"Starting epoch {epoch + 1}/{args.num_epochs}")
            model.train()
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
            print(f'Epoch {epoch+1}/{args.num_epochs}, F1 Score: {f1}')
            scheduler.step(f1)
            
            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
                
                torch.save(model.state_dict(), os.path.join(seed_dir, 'best_model.pth'))
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered")
                break
        return model, train_losses, val_losses, f1

def k_trainer_pytorch_esc(args, model, train_loader, val_loader, class_weights, num_columns, device, seed_dir):
    patience = args.early_stopping_patience  
    best_f1 = 0.0
    patience_counter = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.8)

    train_losses, val_losses = [], []
    
    
    start_train_time = time.time()
    for epoch in range(args.num_epochs):
        print(f"Starting epoch {epoch + 1}/{args.num_epochs}")
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
        print(f"Epoch {epoch+1}/{args.num_epochs}, F1 Score: {f1:.4f}")
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