
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F

class Router(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_prob: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

def train_moe_local(cfg, load_balancing, model, train_loader, val_loader, class_weights, in_dim, device, fold_dir, resume, ckpt_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.experiment.router.lr_moe_train)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    best_accuracy = 0.0
    patience_counter = 0
    best_state = None

    early_stopping_config = cfg.experiment.router.get("early_stopping")
    early_stopping_enabled = early_stopping_config is not None

    if early_stopping_enabled:
        patience = early_stopping_config.patience
        delta = early_stopping_config.delta
        print(f"Early stopping enabled with patience={patience} and delta={delta}")

    train_losses, val_losses = [], []

    for epoch in range(cfg.experiment.model.epochs):
        model.train()
        total_loss = 0
        for embeddings, labels in train_loader:
            X, y = embeddings.to(device), labels.to(device)
            if load_balancing:
                outputs, router_probs, load_balancing_loss_term = model(X)
                classification_loss = criterion(outputs, y)
                loss = classification_loss + cfg.experiment.router.load_balancing_alpha * load_balancing_loss_term
            else:
                outputs, _, _ = model(X)
                loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for embeddings, labels in val_loader:
                X, y = embeddings.to(device), labels.to(device)
                outputs, _, _ = model(X)
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        
        print(f"Epoch {epoch+1}/{cfg.experiment.model.epochs}, Train Loss: {total_loss:.4f}, Val Accuracy: {accuracy:.4f}, Val F1: {f1:.4f}")
        
        val_losses.append(f1) 

        if f1 > best_accuracy + (delta if early_stopping_enabled else 0):
            best_accuracy = f1
            print(f"  -> Validation f1 improved to {best_accuracy:.4f}. Saving model.")
            torch.save(model.state_dict(), ckpt_path)
            best_state = (model.state_dict(), train_losses, val_losses, best_accuracy, all_labels, all_preds, [])
            if early_stopping_enabled:
                patience_counter = 0  # Reset patience counter
        else:
            if early_stopping_enabled:
                patience_counter += 1
                print(f"  -> No improvement for {patience_counter} epoch(s). Patience is {patience}.")

        if early_stopping_enabled and patience_counter >= patience:
            print(f"\nEARLY STOPPING: Validation accuracy has not improved by >{delta} for {patience} epochs.")
            break
        
    if best_state is None:
        print("No improvement observed; saving final model state.")
        torch.save(model.state_dict(), ckpt_path)
        best_state = (model.state_dict(), train_losses, val_losses, accuracy, all_labels, all_preds, [])

    return best_state


def _validate_moe_epoch(model, val_loader, criterion, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for embeddings, labels in val_loader:
            X, y = embeddings.to(device), labels.to(device)
            outputs, _, _ = model(X) # Ensure model returns 3 values
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())
    return 0, 0, all_labels, all_preds, all_probs