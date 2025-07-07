"""
bitnet158b, bitnet 

CUDA_VISIBLE_DEVICES=1 python qmoe_last.py \
  --config-path /home/sebastian/codes/repo_clean/QWave/config \
  --config-name esc50 \
  experiment.router.expert_quantizations=""[1,2,4,16]"" \
  experiment.router.num_experts=4 \
  experiment.datasets.esc.normalization_type=standard \
  experiment.datasets.esc.csv=/home/sebastian/codes/data/ESC-50-master/VE_soundscapes/efficientnet_1536/esc-50.csv \
  experiment.device=cuda \
  experiment.metadata.tag=test
"""
from pathlib import Path
import os, sys
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import os, sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from codecarbon import EmissionsTracker
from memory_profiler import memory_usage
from QWave.datasets import EmbeddingAdaptDataset
from QWave.graphics import plot_multiclass_roc_curve, plot_losses
from QWave.models import ESCModel, reset_weights
import time # Import the time module

# Import get_device from QWave.utils or define it if it's not universally available
try:
    from QWave.utils import get_device
except ImportError:
    print("Warning: QWave.utils.get_device not found. Defining a simple get_device function.")
    def get_device(cfg):
        if cfg.experiment.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

def ternary_quantize(x, threshold=0.05):
    """
    Quantize tensor x to {-1, 0, +1} with a sparsity threshold.
    """
    x_sign = torch.sign(x)
    x_sparse = torch.where(x.abs() < threshold, torch.zeros_like(x), x_sign)
    return x_sparse

def ternary_to_binary(x: torch.Tensor) -> torch.Tensor:
    """Convert {-1, 0, +1} to 2-bit binary encoding: [-1, 0, +1] → [1,0], [0,0], [0,1]."""
    neg = (x == -1).to(torch.uint8)
    pos = (x == 1).to(torch.uint8)
    return torch.stack([neg, pos], dim=-1)  # shape [..., 2]

def packbits2(tensor: torch.Tensor) -> torch.Tensor:
    """
    Packs last dim of tensor of bits (uint8) into uint8 bytes along that dim.
    Assumes the last dim is divisible by 8.
    """
    assert tensor.shape[-1] % 8 == 0, "Last dim must be divisible by 8 for bit packing"
    shape = tensor.shape[:-1] + (tensor.shape[-1] // 8,)
    tensor = tensor.view(*shape[:-1], 8)
    # Ensure torch.arange is on the same device as the tensor
    packed = (tensor << torch.arange(7, -1, -1, device=tensor.device)).sum(dim=-1)
    return packed.to(torch.uint8)

def bitwise_dot(x_bin, w_bin):
    """
    Simulates binary dot product via XOR and popcount (Hamming distance).
    Input:
      x_bin: [B, packed_bits]
      w_bin: [C, packed_bits]
    Output:
      [B, C] score matrix
    """
    # Ensure inputs are on the same device
    w_bin = w_bin.to(x_bin.device)
    xor = torch.bitwise_xor(x_bin.unsqueeze(1), w_bin.unsqueeze(0))  # [B, C, packed_bits]
    return (8 * xor.shape[-1] - xor.sum(dim=-1).float())  # Higher score for more matches

# === Layer Implementation ===
class BitwisePopcountLinear(nn.Module):
    def __init__(self, in_features, out_features, threshold=0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        B = x.shape[0]

        # Quantize input and weight
        x_q = ternary_quantize(x, self.threshold)  # [B, D]
        w_q = ternary_quantize(self.weight, self.threshold)  # [C, D]

        # Convert to binary encoding: [B, D, 2]
        x_bin = ternary_to_binary(x_q).reshape(B, -1)  # [B, D*2]
        w_bin = ternary_to_binary(w_q).reshape(self.out_features, -1)  # [C, D*2]

        # Pad to multiple of 8 for packing
        def pad8(t):
            L = t.shape[-1]
            pad_len = (8 - (L % 8)) % 8
            # Ensure padding is done on the correct device
            return F.pad(t, (0, pad_len), value=0)

        x_bin = pad8(x_bin)
        w_bin = pad8(w_bin)

        # Pack bits
        # Ensure that packing operates on the correct device
        x_pack = packbits2(x_bin)  # [B, L/8]
        w_pack = packbits2(w_bin)  # [C, L/8]

        # Compute approximate dot via popcount logic
        scores = bitwise_dot(x_pack, w_pack)  # [B, C]
        return scores
    
class BitNetPopcountExpert(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_sizes, dropout_prob, threshold=0.05):
        super().__init__()
        layers = []
        last_dim = in_dim
        for h in hidden_sizes:
            layers.append(BitwisePopcountLinear(last_dim, h, threshold))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            last_dim = h
        layers.append(BitwisePopcountLinear(last_dim, num_classes, threshold))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
          return self.net(x)

class BitwiseLinear(nn.Module):
    """
    BitNet-1.58b-style linear layer using ternary weights/activations and integer matrix multiply.
    """
    def __init__(self, in_features, out_features, threshold=0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # Quantize activations and weights to ternary values {-1, 0, 1}
        x_tern = ternary_quantize(x, self.threshold).to(torch.int8)       # [B, D]
        w_tern = ternary_quantize(self.weight, self.threshold).to(torch.int8)  # [C, D]

        # Matrix multiply using integer dot product
        # [B, D] x [D, C]ᵗ = [B, C]
        # Note: we convert to int32 to avoid overflow on dot product
        x_int = x_tern.to(torch.int32)
        w_int = w_tern.to(torch.int32)

        out = torch.matmul(x_int, w_int.T)  # [B, C]

        return out.float()  # Output remains float for downstream modules

class BitNetExpert158b(nn.Module):
    """
    An expert model built using the BitwiseLinear layers. This defines a single expert's architecture.
    """
    def __init__(self, in_dim, num_classes, hidden_sizes, dropout_prob, threshold=0.05):
        super().__init__()
        layers = []
        last_dim = in_dim
        for h in hidden_sizes:
            # Use the new BitNet1.58b linear layer
            layers.append(BitwiseLinear(last_dim, h, threshold))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            last_dim = h
        # Final layer to produce class logits
        layers.append(BitwiseLinear(last_dim, num_classes, threshold))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class BitLinear(nn.Module):
    """
    A BitLinear layer supporting both fixed bit-widths (1, 2, 4, 8, 16) and BitNet-style ternary quantization.
    """
    def __init__(self, in_features, out_features, num_bits=16):
        super(BitLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bits = num_bits
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        if self.num_bits == "bitnet":
            return self.forward_bitnet(x)
        
        elif isinstance(self.num_bits, int) and self.num_bits >= 16:
            return F.linear(x, self.weight)
        else:
            return self.forward_quantized(x)

    def forward_quantized(self, x):
        # Activation quantization (absmax scaling)
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_quant = (x * scale).round().clamp(-128, 127) / scale
        x_final = x + (x_quant - x).detach()  # STE for activation

        # Weight quantization (absmax scaling)
        w_centered = self.weight - self.weight.mean()
        if self.num_bits == 1:
            w_quant = torch.sign(w_centered)  # Ternary
        else:
            q_min = -2.**(self.num_bits - 1)
            q_max = 2.**(self.num_bits - 1) - 1
            w_scale = w_centered.abs().max() / q_max
            w_quant = torch.round(w_centered / w_scale.clamp(min=1e-5)).clamp(q_min, q_max)
            w_quant = w_quant * w_scale

        w_final = self.weight + (w_quant - self.weight).detach() # STE for weights
        return F.linear(x_final, w_final)

    def forward_bitnet(self, x):
        # Activation ternarization with STE
        x_codebook = torch.sign(x)
        x_final = x + (x_codebook - x).detach()

        # Weight ternarization with STE
        w_centered = self.weight - self.weight.mean()
        w_ternary = torch.sign(w_centered)
        w_final = self.weight + (w_ternary - self.weight).detach()

        return F.linear(x_final, w_final)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

class BitNetExpert(nn.Module):
    """An expert model using the original BitLinear layers and LayerNorm."""
    def __init__(self, in_dim, num_classes, hidden_sizes, dropout_prob, num_bits=16):
        super().__init__()
        layers = []
        last_dim = in_dim
        for hidden_dim in hidden_sizes:
            layers.append(BitLinear(last_dim, hidden_dim, num_bits=num_bits))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            last_dim = hidden_dim
        layers.append(BitLinear(last_dim, num_classes, num_bits=num_bits))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

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

class qMoEModelBatched(nn.Module):
    def __init__(self, cfg, in_dim, num_classes, num_experts=4, top_k=2):
        super().__init__()
        self.router = Router(in_dim, cfg.experiment.router.hidden_dim, num_experts, cfg.experiment.model.dropout_prob)
        
        expert_quantizations = cfg.experiment.router.expert_quantizations
        print(f"Initializing experts with quantizations: {expert_quantizations}")

        experts = []
        for bit_width in expert_quantizations:
            if bit_width == "bitnet158b":
                print("  -> Creating a BitNet1.58b expert.")
                experts.append(BitNetExpert158b(
                    in_dim,
                    num_classes,
                    hidden_sizes=cfg.experiment.model.hidden_sizes,
                    dropout_prob=cfg.experiment.model.dropout_prob
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
        
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balancing_alpha = cfg.experiment.router.load_balancing_alpha
        

    def forward(self, x):
        B = x.size(0)
        
        router_scores = self.router(x)                      
        router_probs = F.softmax(router_scores, dim=1)    

        topk_vals, topk_indices = torch.topk(router_probs, self.top_k, dim=1)  

        outputs = torch.zeros(B, self.num_classes, device=x.device)

        load_balancing_loss = 0.0 

        if self.training: 
            load_balancing_loss = torch.sum(torch.mean(router_probs, dim=0) ** 2)

        for expert_idx in range(self.num_experts):
            mask = (topk_indices == expert_idx)  

            if not mask.any():
                continue

            example_indices, slot_indices = torch.nonzero(mask, as_tuple=True)

            x_selected = x[example_indices]  
            weight_selected = topk_vals[example_indices, slot_indices]  

            expert_output = self.experts[expert_idx](x_selected)  

            outputs[example_indices] += expert_output * weight_selected.unsqueeze(1)

        outputs /= self.top_k  
        return outputs, router_probs, load_balancing_loss


def get_num_parameters(model):
    """Calculates the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
from memory_profiler import memory_usage

def _validate_moe_epoch(model, val_loader, criterion, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_latency_ms = 0
    num_samples = 0

    mem_snapshots = []

    with torch.no_grad():
        for embeddings, labels in val_loader:
            X, y = embeddings.to(device), labels.to(device)

            def forward_step():
                outputs, _, _ = model(X)

            mem = memory_usage((forward_step,), interval=0.01, max_iterations=1)
            mem_snapshots.append(np.mean(mem))

            # Measure latency
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                outputs, _, _ = model(X)
                end_event.record()
                torch.cuda.synchronize()
                batch_latency_ms = start_event.elapsed_time(end_event)
            else:
                start_time = time.perf_counter()
                outputs, _, _ = model(X)
                end_time = time.perf_counter()
                batch_latency_ms = (end_time - start_time) * 1000
            
            total_latency_ms += batch_latency_ms
            num_samples += X.size(0)

            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())

    avg_latency_ms_per_sample = (total_latency_ms / num_samples) if num_samples > 0 else 0
    avg_memory_mb = np.mean(mem_snapshots) if mem_snapshots else 0
    return 0, 0, all_labels, all_preds, all_probs, avg_latency_ms_per_sample, avg_memory_mb

def _load_cc_csv(csv_path: Path) -> dict:
    if not csv_path.is_file():
        return {}
    df = pd.read_csv(csv_path)
    return df.iloc[-1].to_dict() if len(df) else {}

cc_metrics_to_track = [
    "duration", "emissions", "emissions_rate", "cpu_power", "gpu_power", "ram_power",
    "cpu_energy", "gpu_energy", "ram_energy", "energy_consumed",
    "cpu_count", "cpu_model", "gpu_count", "gpu_model", "ram_total_size"
]


@hydra.main(version_base=None, config_path="config", config_name="esc50")
def main(cfg: DictConfig):
    df_full = pd.read_csv(cfg.experiment.datasets.esc.csv)
    labels = df_full["class_id"].values
    df = df_full.drop(columns=["folder", "name", "label", "category"], errors="ignore")
    skf = StratifiedKFold(n_splits=cfg.experiment.cross_validation.n_splits, shuffle=True, random_state=42)
    device = get_device(cfg)
    print(f"Final selected device: {device}\n")
    tag = cfg.experiment.metadata.tag
    out_dir = Path("outputs") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    all_final_f1_scores, all_final_accuracy_scores = [], []
    all_avg_ram_mb_train, all_avg_ram_mb_val = [], [] 
    all_train_cc_data_agg = {k: [] for k in cc_metrics_to_track}
    all_val_cc_data_agg = {k: [] for k in cc_metrics_to_track}
    all_training_durations = [] 
    all_validation_durations = [] 
    all_validation_latencies = [] # Store validation latencies per fold
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n--- FOLD {fold+1}/{skf.n_splits} ---")
        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)

        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)

        normalization_type = cfg.experiment.datasets.esc.normalization_type
        train_ds = EmbeddingAdaptDataset(df_train, normalization_type=normalization_type, scaler=None)
        fitted_scaler = train_ds.get_scaler()
        val_ds = EmbeddingAdaptDataset(df_val, normalization_type=normalization_type, scaler=fitted_scaler)

        train_ld = DataLoader(train_ds, batch_size=cfg.experiment.model.batch_size, shuffle=True)
        val_ld = DataLoader(val_ds, batch_size=cfg.experiment.model.batch_size, shuffle=False)

        in_dim = train_ds.features.shape[1]
        num_classes = len(np.unique(labels))
        
        # Instantiate the qMoEModelBatched as per your original request
        model = qMoEModelBatched(cfg, in_dim, num_classes, cfg.experiment.router.num_experts, cfg.experiment.router.top_k).to(device)
        
        # Calculate parameter count for the MoE model
        moe_parameter_count = get_num_parameters(model)
        print(f"MoE Model Parameter Count: {moe_parameter_count}")

        class_weights = torch.tensor(1.0 / np.bincount(train_ds.labels.numpy()), dtype=torch.float32).to(device)
        ckpt_path = fold_dir / "best_model.pth"

        train_start_time = time.perf_counter()
        train_tracker = EmissionsTracker(project_name=f"{tag}_fold{fold}_train", output_dir=str(fold_dir), output_file="emissions_train.csv")
        train_tracker.start()
        load_balancing = cfg.experiment.router.load_balancing # Use load balancing from config
        mem_train_usage, (state_dict, train_losses, val_losses, best_f1, all_labels_best, all_preds_best, _) = \
            memory_usage((train_moe_local, (cfg, load_balancing, model, train_ld, val_ld, class_weights, in_dim, device, str(fold_dir), False, ckpt_path)), interval=0.1, retval=True)
        
        train_tracker.stop()
        avg_ram_mb_train = sum(mem_train_usage) / len(mem_train_usage) 
        train_end_time = time.perf_counter()
        training_duration = train_end_time - train_start_time
        print(f"Manual Timer: Training for Fold {fold+1} took {training_duration:.2f} seconds.")
        all_training_durations.append(training_duration)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache emptied after training.")

        val_tracker = EmissionsTracker(project_name=f"{tag}_fold{fold}_val", output_dir=str(fold_dir), output_file="emissions_val.csv")
        val_tracker.start()

        final_model = qMoEModelBatched(cfg,in_dim, num_classes, cfg.experiment.router.num_experts, cfg.experiment.router.top_k).to(device)
        final_model.load_state_dict(torch.load(ckpt_path, map_location=device))

        val_start_time = time.perf_counter()

        # Capture latency directly from _validate_moe_epoch
        # mem_val_usage, (_, _, all_labels_final, all_preds_final, all_probs_final, avg_latency_ms_per_sample) = \
        #     memory_usage((_validate_moe_epoch, (final_model, val_ld, nn.CrossEntropyLoss(), device)), interval=0.1, retval=True)
        
        # This already returns clean validation memory usage
        _, _, all_labels_final, all_preds_final, all_probs_final, avg_latency_ms_per_sample, avg_ram_mb_val = \
            _validate_moe_epoch(final_model, val_ld, nn.CrossEntropyLoss(), device)
        val_end_time = time.perf_counter()
        validation_duration = val_end_time - val_start_time
        print(f"Manual Timer: Validation for Fold {fold+1} took {validation_duration:.2f} seconds.")
        print(f"Average Latency per sample for Fold {fold+1}: {avg_latency_ms_per_sample:.4f} ms")

        val_tracker.stop()
        
        all_validation_durations.append(validation_duration)
        all_validation_latencies.append(avg_latency_ms_per_sample) # Store latency

        train_stats = _load_cc_csv(fold_dir / "emissions_train.csv")
        val_stats = _load_cc_csv(fold_dir / "emissions_val.csv")
        final_accuracy = accuracy_score(all_labels_final, all_preds_final)
        final_f1_weighted = f1_score(all_labels_final, all_preds_final, average="weighted", zero_division=0)

        print(f"  Fold {fold}: Final Accuracy={final_accuracy:.4f} | Final weighted-F1={final_f1_weighted:.4f}")

        fold_result = {
            "best_f1": final_f1_weighted,
            "accuracy": final_accuracy,
            "avg_ram_mb_train": float(avg_ram_mb_train),
            "avg_ram_mb_val": float(avg_ram_mb_val), # Peak memory (RAM) during validation
            "training_duration_seconds": float(training_duration),
            "validation_duration_seconds": float(validation_duration),
            "average_latency_ms_per_sample": float(avg_latency_ms_per_sample), # Average latency
            "parameter_count": int(moe_parameter_count) # Parameter count
        }

        for k in cc_metrics_to_track:
            fold_result[f"train_{k}"] = train_stats.get(k, 0.0)
            all_train_cc_data_agg[k].append(train_stats.get(k, 0.0))
        for k in cc_metrics_to_track:
            fold_result[f"val_{k}"] = val_stats.get(k, 0.0)
            all_val_cc_data_agg[k].append(val_stats.get(k, 0.0))

        all_final_f1_scores.append(final_f1_weighted)
        all_final_accuracy_scores.append(final_accuracy)
        all_avg_ram_mb_train.append(avg_ram_mb_train)
        all_avg_ram_mb_val.append(avg_ram_mb_val)

        plot_multiclass_roc_curve(all_labels_final, all_probs_final, EXPERIMENT_NAME=str(fold_dir))
        plot_losses(train_losses, val_losses, str(fold_dir))

        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(fold_result, f, indent=4)
        fold_metrics.append(fold_result)

    summary = {
        "f1_mean": float(np.mean(all_final_f1_scores)),
        "f1_std": float(np.std(all_final_f1_scores)),
        "accuracy_mean": float(np.mean(all_final_accuracy_scores)),
        "accuracy_std": float(np.std(all_final_accuracy_scores)),
        "training_duration_mean_seconds": float(np.mean(all_training_durations)),
        "training_duration_std_seconds": float(np.std(all_training_durations)),
        "validation_duration_mean_seconds": float(np.mean(all_validation_durations)),
        "validation_duration_std_seconds": float(np.std(all_validation_durations)),
        "average_latency_ms_per_sample_mean": float(np.mean(all_validation_latencies)), # Avg latency mean
        "average_latency_ms_per_sample_std": float(np.std(all_validation_latencies)),   # Avg latency std
        "parameter_count": int(moe_parameter_count), # Parameter count (same for all folds of this MoE config)
        "metadata": dict(cfg.experiment.metadata),
        "folds": fold_metrics
    }

    if all_avg_ram_mb_train:
        summary["avg_ram_mb_train_mean"] = float(np.mean(all_avg_ram_mb_train))
        summary["avg_ram_mb_train_std"] = float(np.std(all_avg_ram_mb_train))
    if all_avg_ram_mb_val:
        summary["avg_ram_mb_val_mean"] = float(np.mean(all_avg_ram_mb_val)) # Peak RAM mean
        summary["avg_ram_mb_val_std"] = float(np.std(all_avg_ram_mb_val))   # Peak RAM std

    for phase_data_agg, prefix in [(all_train_cc_data_agg, "train"), (all_val_cc_data_agg, "val")]:
        for k in cc_metrics_to_track:
            if phase_data_agg[k]:
                if all(isinstance(val, (int, float)) for val in phase_data_agg[k]):
                    summary[f"{prefix}_{k}_mean"] = float(np.mean(phase_data_agg[k]))
                    summary[f"{prefix}_{k}_std"] = float(np.std(phase_data_agg[k]))
                else:
                    unique_values = list(set(phase_data_agg[k]))
                    if len(unique_values) == 1:
                        summary[f"{prefix}_{k}_common"] = unique_values[0]
                    else:
                        summary[f"{prefix}_{k}_all"] = unique_values

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\n== CV SUMMARY ==")
    print(json.dumps(summary, indent=4))
    # === Create filtered summary CSV ===
    filtered_csv_path = out_dir / "summary_filtered.csv"
    filtered_data = {
        "name": summary.get("metadata", {}).get("tag", "unknown"),
        "parameter_count": summary.get("parameter_count"),
        "accuracy_mean": summary.get("accuracy_mean"),
        "f1_mean": summary.get("f1_mean"),
        "avg_ram_gb_val_mean": summary.get("avg_ram_mb_val_mean", 0.0) / 1024,
        "average_latency_ms_per_sample_mean": summary.get("average_latency_ms_per_sample_mean"),
        "training_duration_mean_seconds": summary.get("training_duration_mean_seconds"),
        "validation_duration_mean_seconds": summary.get("validation_duration_mean_seconds"),
        "train_energy_consumed_mean": summary.get("train_energy_consumed_mean"),
        "val_energy_consumed_mean": summary.get("val_energy_consumed_mean"),
        "train_emissions_mean": summary.get("train_emissions_mean"),
        "val_emissions_mean": summary.get("val_emissions_mean"),
        "train_gpu_power_mean": summary.get("train_gpu_power_mean")
    }
    pd.DataFrame([filtered_data]).to_csv(filtered_csv_path, index=False)
    print(f"\nFiltered summary saved to {filtered_csv_path}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()