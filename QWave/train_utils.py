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

import os
import time
import torch
import numpy as np # Import numpy
from torch import nn
from torch.utils.data import DataLoader # Keep if used
from sklearn.metrics import f1_score, confusion_matrix # Keep if used
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Make sure these imports are correct based on your QWave project structure
from QWave.graphics import plot_training_curves, plot_multiclass_roc_curve, plot_losses, show_image, save_confusion_matrix, plot_confusion_matrix

# This _validate_single_epoch function is correct and should be imported/used by train_pytorch_local
def _validate_single_epoch(model, val_loader, criterion, device):
    """Performs a single validation epoch and collects metrics."""
    model.eval()
    epoch_val_loss = 0
    all_labels, all_preds, all_probs = [], [], []
    # Add robust handling for empty val_loader
    if val_loader is None or len(val_loader.dataset) == 0:
        return float('inf'), 0.0, [], [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            probs = torch.softmax(outputs, dim=1) # Correct: softmax for probabilities
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Handle empty val_loader.dataset if it's possible after loop
    if len(val_loader.dataset) == 0:
        epoch_val_loss = float('inf')
    else:
        epoch_val_loss /= len(val_loader.dataset)

    # Handle case where all_labels might be empty
    if len(all_labels) == 0:
        f1 = 0.0
    else:
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return epoch_val_loss, f1, all_labels, all_preds, all_probs


def train_pytorch_local(args, model, train_loader, val_loader, class_weights, num_columns, device, fold_folder,
                        resume_checkpoint=False, checkpoint_path=None):
    best_f1 = -1.0 # Initialize with a value that any valid F1 will beat
    patience_counter = 0
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.model.learning_rate, weight_decay=args.model.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.model.label_smoothing)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=args.model.patience, factor=args.model.factor) # mode='max' for F1

    train_losses, val_losses = [], []
    # These will store the labels/preds/probs from the epoch that achieved the best_f1
    all_labels_best_epoch, all_preds_best_epoch, all_probs_best_epoch = [], [], []

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

        # Call the helper _validate_single_epoch for consistent validation
        epoch_val_loss, f1, current_labels, current_preds, current_probs = \
            _validate_single_epoch(model, val_loader, criterion, device)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{args.model.epochs}, Val F1 Score: {f1:.4f}, Val Loss: {epoch_val_loss:.4f}")
        scheduler.step(f1) # Schedule based on F1

        # Check for best model and save its state and corresponding metrics
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            if getattr(args, "save_checkpoint", True):
                torch.save(model.state_dict(), os.path.join(fold_folder, "best_model.pth"))
                # print(f"Saved best model with F1: {best_f1:.4f} at epoch {epoch+1}")
            
            # Update best epoch metrics
            all_labels_best_epoch = current_labels
            all_preds_best_epoch = current_preds
            all_probs_best_epoch = current_probs
        else:
            patience_counter += 1
            if patience_counter >= args.model.patience:
                print("Early stopping triggered")
                break
    
    # Plotting using the overall training history and the metrics from the best epoch
    plot_training_curves(train_losses, val_losses, fold_folder)
    # Only plot confusion matrix if there was at least one valid validation step
    if all_labels_best_epoch and all_preds_best_epoch:
        plot_confusion_matrix(all_labels_best_epoch, all_preds_best_epoch, fold_folder)
    else:
        print(f"Warning: No valid data to plot confusion matrix for fold: {fold_folder}. Val loader might be empty or training ended too early.")


    # Return the model (or its final state), losses, best F1, and metrics from the best validation epoch
    return model, train_losses, val_losses, best_f1, all_labels_best_epoch, all_preds_best_epoch, all_probs_best_epoch


# def train_pytorch_adapter(args, model, train_loader, val_loader, num_columns, device, fold_folder,
#                           task_type='multiclass', class_weights=None, pos_weight=None, 
#                           resume_checkpoint=False, checkpoint_path=None):
#     best_f1 = -1.0 # Initialize with a value that any valid F1 will beat
#     patience_counter = 0
#     model = model.to(device)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.model.learning_rate, weight_decay=args.model.weight_decay)
    
#     # --- CRITICAL: Select criterion based on task_type ---
#     if task_type == 'binary':
#         # BCEWithLogitsLoss expects pos_weight as a tensor of shape [1]
#         if pos_weight is not None:
#             # Ensure pos_weight is a tensor and on the correct device
#             criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device, dtype=torch.float32))
#         else:
#             criterion = nn.BCEWithLogitsLoss()

#         print(f"  Using BCEWithLogitsLoss for {task_type} task.")
#     else: # 'multiclass'
#         # CrossEntropyLoss expects class_weights as a tensor of shape [num_classes]
#         weights_tensor = None
#         if class_weights is not None:
#             # Ensure class_weights is a tensor and on the correct device if provided
#             weights_tensor = torch.tensor(class_weights, device=device, dtype=torch.float32)
#         criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=args.label_smoothing)
#         print(f"  Using CrossEntropyLoss for {task_type} task.")
    

#     scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=args.model.patience, factor=args.model.factor) # mode='max' for F1

#     train_losses, val_losses = [], []
#     # These will store the labels/preds/probs from the epoch that achieved the best_f1
#     all_labels_best_epoch, all_preds_best_epoch, all_probs_best_epoch = [], [], []

#     if resume_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
#         print(f"Resuming from checkpoint: {checkpoint_path}")
#         model.load_state_dict(torch.load(checkpoint_path, map_location=device))
#         model = model.to(device)

#     for epoch in range(args.model.epochs):
#         model.train()
#         epoch_train_loss = 0
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)

#             # --- FIX STARTS HERE ---
#             if task_type == 'binary':
#                 # Reshape labels to [batch_size, 1] to match model output
#                 labels = labels.float().unsqueeze(1) 
#             # --- FIX ENDS HERE ---
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             epoch_train_loss += loss.item() * inputs.size(0)

#         epoch_train_loss /= len(train_loader.dataset)
#         train_losses.append(epoch_train_loss)

#         # Call the helper _validate_single_epoch for consistent validation
#         epoch_val_loss, f1, current_labels, current_preds, current_probs = \
#             _validate_single_epoch(model, val_loader, criterion, device)
#         val_losses.append(epoch_val_loss)

#         print(f"Epoch {epoch+1}/{args.model.epochs}, Val F1 Score: {f1:.4f}, Val Loss: {epoch_val_loss:.4f}")
#         scheduler.step(f1) # Schedule based on F1

#         # Check for best model and save its state and corresponding metrics
#         if f1 > best_f1:
#             best_f1 = f1
#             patience_counter = 0
#             if getattr(args, "save_checkpoint", True):
#                 torch.save(model.state_dict(), os.path.join(fold_folder, "best_model.pth"))
#                 # print(f"Saved best model with F1: {best_f1:.4f} at epoch {epoch+1}")
            
#             # Update best epoch metrics
#             all_labels_best_epoch = current_labels
#             all_preds_best_epoch = current_preds
#             all_probs_best_epoch = current_probs
#         else:
#             patience_counter += 1
#             if patience_counter >= args.model.patience:
#                 print("Early stopping triggered")
#                 break
    
#     # Plotting using the overall training history and the metrics from the best epoch
#     plot_training_curves(train_losses, val_losses, fold_folder)
#     # Only plot confusion matrix if there was at least one valid validation step
#     if all_labels_best_epoch and all_preds_best_epoch:
#         plot_confusion_matrix(all_labels_best_epoch, all_preds_best_epoch, fold_folder)
#     else:
#         print(f"Warning: No valid data to plot confusion matrix for fold: {fold_folder}. Val loader might be empty or training ended too early.")


# --- Update _validate_single_epoch as well (no changes needed here from last iteration) ---
def _validate_single_epoch_adapter(model, val_loader, criterion, device, task_type='multiclass'):
    model.eval()
    epoch_val_loss = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            if task_type == 'binary':
                labels_for_loss = labels.float().unsqueeze(1) 
            else:
                labels_for_loss = labels 

            loss = criterion(outputs, labels_for_loss)
            epoch_val_loss += loss.item() * inputs.size(0)

            if task_type == 'binary': 
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long()
                probs = probs.cpu().numpy()
            else: 
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                probs = probs.cpu().numpy()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs)
    
    epoch_val_loss /= len(val_loader.dataset)
    
    from sklearn.metrics import f1_score
    f1 = f1_score(all_labels, all_preds, average='weighted') 

    return epoch_val_loss, f1, all_labels, all_preds, all_probs


def train_pytorch_adapter(args, model, train_loader, val_loader, num_columns, device, fold_folder,
                          task_type='multiclass', class_weights=None, pos_weight=None, 
                          resume_checkpoint=False, checkpoint_path=None):
    best_f1 = -1.0 
    patience_counter = 0
    model = model.to(device)

    # FIX: Access learning_rate and weight_decay via args.model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.model.learning_rate, weight_decay=args.model.weight_decay)
    
    if task_type == 'binary':
        if pos_weight is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device, dtype=torch.float32))
        else:
            criterion = nn.BCEWithLogitsLoss()
        print(f"  Using BCEWithLogitsLoss for {task_type} task.")
    else: # 'multiclass'
        weights_tensor = None
        if class_weights is not None:
            weights_tensor = torch.tensor(class_weights, device=device, dtype=torch.float32)
        # FIX: Access label_smoothing via args.model
        criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=args.model.label_smoothing)
        print(f"  Using CrossEntropyLoss for {task_type} task.")
    
    # This line, which you state is correct, implies args.model exists for patience and factor.
    # It also has the duplicate 'max' mode, which typically causes a TypeError.
    # If this specific line works for you, it's an unusual case.
    scheduler = ReduceLROnPlateau(optimizer,  mode='max', patience=args.model.patience, factor=args.model.factor) # mode='max' for F1


    train_losses, val_losses = [], []
    all_labels_best_epoch, all_preds_best_epoch, all_probs_best_epoch = [], [], []

    if resume_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model = model.to(device)

    for epoch in range(args.model.epochs): # Access epochs via args.model.epochs
        model.train()
        epoch_train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            if task_type == 'binary':
                labels = labels.float().unsqueeze(1) 

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        epoch_val_loss, f1, current_labels, current_preds, current_probs = \
            _validate_single_epoch_adapter(model, val_loader, criterion, device, task_type=task_type) 
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{args.model.epochs}, Val F1 Score: {f1:.4f}, Val Loss: {epoch_val_loss:.4f}")
        scheduler.step(f1) 

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            # FIX: Access save_checkpoint via args.logging (or args.model if you defined it there)
            # Based on your YAML, save_checkpoint is under 'logging'
            if getattr(args.logging, "save_checkpoint", True): 
                torch.save(model.state_dict(), os.path.join(fold_folder, "best_model.pth"))
            
            all_labels_best_epoch = current_labels
            all_preds_best_epoch = current_preds
            all_probs_best_epoch = current_probs
        else:
            patience_counter += 1
            # FIX: Access patience from args.model.early_stopping
            if patience_counter >= args.model.early_stopping.patience: 
                print("Early stopping triggered")
                break
    
    return model, train_losses, val_losses, best_f1, all_labels_best_epoch, all_preds_best_epoch, all_probs_best_epoch


    

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