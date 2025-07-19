
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F
from .models import ESCModel, reset_weights
from .qmoe_layers import BitNetExpert158b, BitNetExpert, BitNetPopcountExpert

class qMoEModelBatched(nn.Module):
    def __init__(self, cfg, in_dim, num_classes, num_experts=4, top_k=2):
        super().__init__()
        self.router = Router(in_dim, cfg.experiment.router.hidden_dim, num_experts, cfg.experiment.model.dropout_prob)
        
        expert_quantizations = cfg.experiment.router.expert_quantizations
        print(f"Initializing experts with quantizations: {expert_quantizations}")

        experts = []
        for bit_width in expert_quantizations:
            if bit_width == "esc":
                print("  -> Creating a ESC expert.")
                experts.append(ESCModel(
                    in_dim,
                    num_classes,
                    hidden_sizes=cfg.experiment.model.hidden_sizes,
                    dropout_prob=cfg.experiment.model.dropout_prob
                ))
            elif bit_width == "qesc":
                print("  -> Creating a qESC expert.")
                experts.append(ESCModel(
                    in_dim,
                    num_classes,
                    hidden_sizes=cfg.experiment.model.hidden_sizes,
                    dropout_prob=cfg.experiment.model.dropout_prob
                ))
                torch.backends.quantized.engine = 'qnnpack'
            elif bit_width == "bitnet158b":
                print("  -> Creating a BitNet1.58b expert.")
                experts.append(BitNetExpert158b(
                    in_dim,
                    num_classes,
                    hidden_sizes=cfg.experiment.model.hidden_sizes,
                    dropout_prob=cfg.experiment.model.dropout_prob,
                    threshold=0.05
                ))
            elif bit_width == "bitnet":
                print("  -> Creating a standard BitNetExpert with ternary mode.")
                experts.append(BitNetExpert(
                    in_dim,
                    num_classes,
                    hidden_sizes=cfg.experiment.model.hidden_sizes,
                    dropout_prob=cfg.experiment.model.dropout_prob,
                    num_bits="bitnet"
                ))
            elif bit_width == "popcount": # Added popcount expert type
                print("  -> Creating a BitNetPopcountExpert.")
                experts.append(BitNetPopcountExpert(
                    in_dim,
                    num_classes,
                    hidden_sizes=cfg.experiment.model.hidden_sizes,
                    dropout_prob=cfg.experiment.model.dropout_prob
                ))
            else:
                print(f"  -> Creating a standard BitNetExpert with num_bits={bit_width}.")
                experts.append(BitNetExpert(
                    in_dim,
                    num_classes,
                    hidden_sizes=cfg.experiment.model.hidden_sizes,
                    dropout_prob=cfg.experiment.model.dropout_prob,
                    num_bits=int(bit_width)  # cast to int if needed
                ))
        self.experts = nn.ModuleList(experts)
        print(self.experts)
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balancing_alpha = cfg.experiment.router.load_balancing_alpha
        
    def forward(self, x: torch.Tensor):
        """
        Vectorised MoE forward pass
        ---------------------------
        • Routes each sample to `top_k` experts via soft-max scores  
        • Batches all rows that go to the **same** expert before the call,
        making INT8 / GEMM kernels much more efficient.
        """
        B = x.size(0)                        # batch size
        router_p   = F.softmax(self.router(x), dim=1)          # (B, E)
        k_val, k_idx = torch.topk(router_p, self.top_k, dim=1) # (B, K)

        out = x.new_zeros(B, self.num_classes)  # pre-allocate result

        # load-balancing L2 loss (auxiliary)
        lb_loss = torch.sum(router_p.mean(0) ** 2) if self.training else 0.0

        # ------------------------------------------------------------------
        # Vectorised expert dispatch
        # ------------------------------------------------------------------
        for e_idx, expert in enumerate(self.experts):
            rows = (k_idx == e_idx).nonzero(as_tuple=True)[0]  # indices in batch
            if rows.numel() == 0:
                continue                   # no sample picked this expert

            # column position inside the top-k list for these rows
            cols = (k_idx[rows] == e_idx).nonzero(as_tuple=True)[1]
            weights = k_val[rows, cols]                     # (n_rows,)

            # single expert call on a *batched* tensor
            logits = expert(x[rows])                        # (n_rows, C)

            # weighted contribution
            out[rows] += logits * weights.unsqueeze(1)      # broadcast

        out = out / self.top_k            # average over top-k experts
        return out, router_p, lb_loss


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