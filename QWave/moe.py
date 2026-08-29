import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F
from .models import ESCModel, reset_weights
from .qmoe_layers import BitNetExpert158b, BitNetExpert, BitNetPopcountExpert, SharedTrunkMoE


class qMoEModelBatched(nn.Module):
    def __init__(self, cfg, in_dim, num_classes, num_experts=4, top_k=2):
        super().__init__()

        # ──────────────────────────────────────────────────────────────
        # Router: deterministic vs Bayesian (curiosity via MC Dropout)
        # ──────────────────────────────────────────────────────────────
        use_curiosity = getattr(cfg.experiment.router, "use_curiosity", False)
        if use_curiosity:
            print("Using BayesianRouter with curiosity (Monte Carlo Dropout).")
            self.router = BayesianRouter(
                in_dim,
                cfg.experiment.router.hidden_dim,
                num_experts,
                dropout_prob=cfg.experiment.model.dropout_prob,
                mc_samples=getattr(cfg.experiment.router, "mc_samples", 10),
            )
            self.use_curiosity = True
        else:
            self.router = Router(
                in_dim,
                cfg.experiment.router.hidden_dim,
                num_experts,
                cfg.experiment.model.dropout_prob,
            )
            self.use_curiosity = False

        # ──────────────────────────────────────────────────────────────
        # Curiosity parameters (for modifying routing probabilities)
        # ──────────────────────────────────────────────────────────────
        self.curiosity_alpha = getattr(cfg.experiment.router, "curiosity_alpha", 0.1)
        self.curiosity_strategy = getattr(cfg.experiment.router, "curiosity_strategy", "precision_prior")

        # Validate strategy
        valid_strategies = ["kl_divergence", "entropy_regularization", "precision_prior",
                            "precision_sharp", "escalation", "soft_escalation"]
        if self.use_curiosity and self.curiosity_strategy not in valid_strategies:
            raise ValueError(f"Invalid curiosity_strategy: {self.curiosity_strategy}. "
                             f"Must be one of {valid_strategies}")

        if self.use_curiosity:
            print(f"[MoE] Curiosity enabled: strategy='{self.curiosity_strategy}', α={self.curiosity_alpha}")

        # Per-expert precision weights in [0, 1] for the precision_prior
        # strategy: 0 for the lowest-precision expert, 1 for the highest.
        # Uncertain samples get routed toward high beta (high precision) BY
        # CONSTRUCTION, which is the behavior the paper claims for the
        # uncertainty-aware router.
        def _bits(q):
            if q == "bitnet":
                return 1.58
            if q == "bitnet158b":
                return 1.58
            if q == "popcount":
                return 1.0
            if q == "qesc":
                return 8.0
            if q == "esc":
                return 32.0
            return float(q)
        bit_values = [_bits(q) for q in cfg.experiment.router.expert_quantizations]
        lo, hi = min(bit_values), max(bit_values)
        if hi - lo < 1e-8:
            beta = [0.0 for _ in bit_values]
        else:
            beta = [(b - lo) / (hi - lo) for b in bit_values]
        self.register_buffer("precision_beta", torch.tensor(beta, dtype=torch.float32))

        # ──────────────────────────────────────────────────────────────
        # Expert initialization
        # ──────────────────────────────────────────────────────────────
        expert_quantizations = cfg.experiment.router.expert_quantizations
        if len(expert_quantizations) != num_experts:
            raise ValueError(
                f"num_experts={num_experts} but expert_quantizations has "
                f"{len(expert_quantizations)} entries ({list(expert_quantizations)}). "
                "They must match: one quantization per expert. Set "
                "experiment.router.expert_quantizations and "
                "experiment.router.num_experts to the same length."
            )
        print(f"Initializing experts with quantizations: {expert_quantizations}")

        # Efficiency option: experts share one quantized trunk and differ only
        # in a per-expert head, cutting parameters and per-sample compute
        # roughly in half relative to fully replicated experts.
        self.gate_threshold = getattr(cfg.experiment.router, "gate_threshold", 0.0)
        if self.gate_threshold > 0:
            print(f"  -> Uncertainty gating enabled: MC passes only when top-1 routing prob < {self.gate_threshold}")
        self.shared_trunk = getattr(cfg.experiment.router, "shared_trunk", False)
        if self.shared_trunk:
            trunk_bits = getattr(cfg.experiment.router, "trunk_bits", 8)
            print(f"  -> Shared-trunk MoE (trunk at {trunk_bits}-bit, {len(expert_quantizations)} heads).")
            self.shared = SharedTrunkMoE(
                in_dim, num_classes,
                hidden_sizes=cfg.experiment.model.hidden_sizes,
                dropout_prob=cfg.experiment.model.dropout_prob,
                expert_bits=list(expert_quantizations),
                trunk_bits=trunk_bits,
            )
            self.experts = nn.ModuleList()
        else:
            self.shared = None

        experts = []
        for bit_width in ([] if self.shared_trunk else expert_quantizations):
            if bit_width == "esc":
                print("  -> Creating a ESC expert.")
                experts.append(
                    ESCModel(
                        in_dim,
                        num_classes,
                        hidden_sizes=cfg.experiment.model.hidden_sizes,
                        dropout_prob=cfg.experiment.model.dropout_prob,
                    )
                )
            elif bit_width == "qesc":
                print("  -> Creating a qESC expert.")
                experts.append(
                    ESCModel(
                        in_dim,
                        num_classes,
                        hidden_sizes=cfg.experiment.model.hidden_sizes,
                        dropout_prob=cfg.experiment.model.dropout_prob,
                    )
                )
                torch.backends.quantized.engine = "qnnpack"
            elif bit_width == "bitnet158b":
                print("  -> Creating a BitNet1.58b expert.")
                experts.append(
                    BitNetExpert158b(
                        in_dim,
                        num_classes,
                        hidden_sizes=cfg.experiment.model.hidden_sizes,
                        dropout_prob=cfg.experiment.model.dropout_prob,
                        threshold=0.05,
                    )
                )
            elif bit_width == "bitnet":
                print("  -> Creating a standard BitNetExpert with ternary mode.")
                experts.append(
                    BitNetExpert(
                        in_dim,
                        num_classes,
                        hidden_sizes=cfg.experiment.model.hidden_sizes,
                        dropout_prob=cfg.experiment.model.dropout_prob,
                        num_bits="bitnet",
                    )
                )
            elif bit_width == "popcount":
                print("  -> Creating a BitNetPopcountExpert.")
                experts.append(
                    BitNetPopcountExpert(
                        in_dim,
                        num_classes,
                        hidden_sizes=cfg.experiment.model.hidden_sizes,
                        dropout_prob=cfg.experiment.model.dropout_prob,
                    )
                )
            else:
                print(f"  -> Creating a standard BitNetExpert with num_bits={bit_width}.")
                experts.append(
                    BitNetExpert(
                        in_dim,
                        num_classes,
                        hidden_sizes=cfg.experiment.model.hidden_sizes,
                        dropout_prob=cfg.experiment.model.dropout_prob,
                        num_bits=int(bit_width),
                    )
                )
        if not self.shared_trunk:
            self.experts = nn.ModuleList(experts)
            print(self.experts)

        # Efficiency/accuracy option: warm-start each expert from the trained
        # single model of the SAME bit-width, so the MoE begins at
        # single-model accuracy and the router learns allocation on top of
        # already-competent experts rather than training everything jointly
        # from scratch. Paths are given per bit-width in the config as
        # warm_start: {'4': '/path/best_model.pth', ...}.
        warm = getattr(cfg.experiment.router, "warm_start", None)
        if warm and not self.shared_trunk:
            import os
            for e_idx, bit_width in enumerate(expert_quantizations):
                key = str(bit_width)
                path = warm.get(key) if hasattr(warm, "get") else None
                # Per-fold warm start: {fold} is substituted with the current
                # fold so each fold loads a checkpoint trained WITHOUT its own
                # validation split. Using one fixed fold's checkpoint for every
                # fold leaks validation data and inflates F1 (observed: fold 1
                # 0.747 honest vs folds 2-5 at 0.91-0.95).
                if path and "{fold}" in str(path):
                    fold_id = getattr(cfg.experiment, "current_fold", None)
                    if fold_id is None:
                        print("  -> warm start: {fold} placeholder but no experiment.current_fold set; skipping")
                        continue
                    path = str(path).replace("{fold}", str(fold_id))
                if not path or not os.path.exists(path):
                    print(f"  -> warm start: no checkpoint for expert '{key}', training from scratch")
                    continue
                state = torch.load(path, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                missing, unexpected = self.experts[e_idx].load_state_dict(state, strict=False)
                print(f"  -> warm start: expert '{key}' <- {os.path.basename(path)} "
                      f"({len(missing)} missing, {len(unexpected)} unexpected keys)")

        # ──────────────────────────────────────────────────────────────
        # Other attributes
        # ──────────────────────────────────────────────────────────────
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balancing_alpha = cfg.experiment.router.load_balancing_alpha

    def forward(self, x: torch.Tensor):
        """
        Vectorised MoE forward pass
        ---------------------------
        • Routes each sample to `top_k` experts via soft-max scores
        • Supports optional Bayesian (curiosity) routing with uncertainty
        Returns:
            out:        (B, C) logits
            router_p:   (B, E) softmax over experts
            lb_loss:    scalar (aux load-balancing term)
            curiosity:  (B,) aggregate epistemic uncertainty or None
        """
        B = x.size(0)

        # Router
        if isinstance(self.router, BayesianRouter):
            # Efficiency option: the MC-dropout pass costs ~18% of inference,
            # but it only changes the routing decision for samples the router
            # is unsure about. With gating, one cheap deterministic pass runs
            # first and the MC passes run ONLY on the ambiguous rows (those
            # whose top-1 routing probability falls below `gate_threshold`).
            if self.use_curiosity and getattr(self, "gate_threshold", 0.0) > 0.0:
                base_logits = self.router.net(x)
                base_p = F.softmax(base_logits, dim=1)
                ambiguous = base_p.max(dim=1).values < self.gate_threshold
                uncertainty = x.new_zeros(x.size(0))
                if ambiguous.any():
                    _, unc_sub = self.router(x[ambiguous], compute_uncertainty=True)
                    uncertainty[ambiguous] = unc_sub
                router_out = base_logits
                self._last_gate_fraction = ambiguous.float().mean().item()
            else:
                router_out, uncertainty = self.router(
                    x, compute_uncertainty=self.use_curiosity
                )
        else:
            router_out, uncertainty = self.router(x), None

        # CURIOSITY MECHANISM: Modify routing probabilities based on uncertainty
        if self.use_curiosity and uncertainty is not None:
            router_p = self._apply_curiosity(router_out, uncertainty)
        else:
            router_p = F.softmax(router_out, dim=1)  # Standard routing
        k_val, k_idx = torch.topk(router_p, self.top_k, dim=1)  # (B, K)

        out = x.new_zeros(B, self.num_classes)

        # load-balancing L2 loss (auxiliary)
        lb_loss = torch.sum(router_p.mean(0) ** 2) if self.training else 0.0

        # Shared-trunk dispatch: the expensive trunk runs ONCE for the whole
        # batch, then only the small per-expert head runs on its assigned rows.
        if self.shared_trunk:
            z = self.shared.trunk_forward(x)
            for e_idx in range(self.num_experts):
                rows = (k_idx == e_idx).nonzero(as_tuple=True)[0]
                if rows.numel() == 0:
                    continue
                cols = (k_idx[rows] == e_idx).nonzero(as_tuple=True)[1]
                weights = k_val[rows, cols]
                logits = self.shared.head_forward(z[rows], e_idx)
                out[rows] += logits * weights.unsqueeze(1)
            out = out / self.top_k
            return out, router_p, lb_loss, uncertainty

        # Vectorised expert dispatch
        for e_idx, expert in enumerate(self.experts):
            rows = (k_idx == e_idx).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue

            cols = (k_idx[rows] == e_idx).nonzero(as_tuple=True)[1]
            weights = k_val[rows, cols]

            logits = expert(x[rows])
            out[rows] += logits * weights.unsqueeze(1)

        out = out / self.top_k

        # Return 4-tuple (backward-compatible callers can ignore the 4th)
        return out, router_p, lb_loss, uncertainty

    def _apply_curiosity(self, router_logits: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Apply curiosity-driven routing modification based on epistemic uncertainty.

        Args:
            router_logits: Raw router outputs (B, E)
            uncertainty: Epistemic uncertainty per sample (B,)

        Returns:
            router_p: Modified routing probabilities (B, E)
        """
        router_p_base = F.softmax(router_logits, dim=1)  # (B, E)

        # ============================================================
        # Strategy 1: KL Divergence (Paper's Equation 8)
        # ============================================================
        if self.curiosity_strategy == "kl_divergence":
            # p_i^curious ∝ p_i · exp(α · KL(p_i||p̄))  (Eq. 8)
            # BUG (fixed 2026-08-28): the previous code summed the KL term over
            # experts into one (B, 1) scalar, so exp(alpha * kl_div) broadcast
            # identically onto every expert. Scaling a distribution by the same
            # constant and renormalizing returns it unchanged, an identity map;
            # alpha had zero effect (verified: max |delta| ~1.8e-7 sweeping
            # alpha in [0, 50]). Fixed to keep a PER-EXPERT bonus so the term
            # actually reshapes the distribution instead of only rescaling it.
            p_uniform = torch.ones_like(router_p_base) / router_p_base.shape[1]

            # Per-expert log-ratio term (not summed across experts): this is
            # expert i's own contribution to KL(p||p_uniform), so it differs
            # per expert and can push some experts up and others down.
            log_ratio_per_expert = (router_p_base + 1e-8).log() - (p_uniform + 1e-8).log()  # (B, E)

            # Apply curiosity bonus per expert (exploration for confident routing)
            curiosity_bonus = torch.exp(self.curiosity_alpha * log_ratio_per_expert)  # (B, E)
            router_p = router_p_base * curiosity_bonus
            router_p = router_p / router_p.sum(dim=1, keepdim=True)

            return router_p

        # ============================================================
        # Strategy 2: Entropy Regularization
        # ============================================================
        elif self.curiosity_strategy == "entropy_regularization":
            # BUG (fixed 2026-08-28): same identity-map defect as Strategy 1.
            # `sharpening` was a (B, 1) scalar applied to every expert equally,
            # so it always renormalized back to router_p_base. Fixed to scale
            # each expert by its own deviation from uniform, so sharpening
            # actually redistributes probability mass instead of only
            # rescaling it uniformly.
            u_min, u_max = uncertainty.min(), uncertainty.max()
            if u_max - u_min < 1e-8:
                return router_p_base

            u_norm = (uncertainty - u_min) / (u_max - u_min + 1e-8)
            u_norm = u_norm.unsqueeze(1).clamp(0, 1)  # (B, 1)

            # Per-expert deviation from uniform routing (signed): experts
            # already favored get pushed further up, disfavored ones further
            # down, scaled by how uncertain the sample is.
            p_uniform = torch.ones_like(router_p_base) / router_p_base.shape[1]
            deviation_per_expert = router_p_base - p_uniform  # (B, E)

            sharpening = 1.0 + self.curiosity_alpha * u_norm * deviation_per_expert  # (B, E)
            sharpening = sharpening.clamp(min=1e-4)  # keep probabilities non-negative

            router_p = router_p_base * sharpening
            router_p = router_p / (router_p.sum(dim=1, keepdim=True) + 1e-8)

            return router_p

        # ============================================================
        # Strategy 3: Precision Prior (uncertainty-directed routing)
        # ============================================================
        elif self.curiosity_strategy == "precision_prior":
            # p_i^curious ∝ p_i · exp(α · u · β_i), where β_i in [0, 1] is
            # expert i's normalized precision. High epistemic uncertainty
            # shifts routing mass toward higher-precision experts by
            # construction; confident samples keep the base routing, so easy
            # inputs stay on the cheap low-bit experts.
            u_min, u_max = uncertainty.min(), uncertainty.max()
            if u_max - u_min < 1e-8:
                return router_p_base

            u_norm = ((uncertainty - u_min) / (u_max - u_min + 1e-8)).unsqueeze(1).clamp(0, 1)  # (B, 1)
            beta = self.precision_beta.unsqueeze(0)  # (1, E)

            router_p = router_p_base * torch.exp(self.curiosity_alpha * u_norm * beta)
            router_p = router_p / router_p.sum(dim=1, keepdim=True)

            return router_p

        # ============================================================
        # Strategy 4: Precision Sharp (polarization + precision prior)
        # ============================================================
        elif self.curiosity_strategy == "precision_sharp":
            # p_i^curious ∝ p_i^(1+α) · exp(α · u · β_i): the kl strategy's
            # temperature sharpening (stability) combined with the
            # uncertainty-to-precision bias (specialization), one alpha.
            sharpened = router_p_base.pow(1.0 + self.curiosity_alpha)
            sharpened = sharpened / sharpened.sum(dim=1, keepdim=True)

            u_min, u_max = uncertainty.min(), uncertainty.max()
            if u_max - u_min < 1e-8:
                return sharpened
            u_norm = ((uncertainty - u_min) / (u_max - u_min + 1e-8)).unsqueeze(1).clamp(0, 1)
            beta = self.precision_beta.unsqueeze(0)

            router_p = sharpened * torch.exp(self.curiosity_alpha * u_norm * beta)
            router_p = router_p / router_p.sum(dim=1, keepdim=True)
            return router_p

        # ============================================================
        # Strategy 5: Escalation (hard uncertainty guard)
        # ============================================================
        elif self.curiosity_strategy == "escalation":
            # The top-alpha fraction of most-uncertain samples in the batch
            # routes outright to the highest-precision expert; everything
            # else keeps base routing. alpha in (0, 1) is the escalation
            # quantile; alpha=0 escalates nothing (identity).
            if self.curiosity_alpha <= 0:
                return router_p_base
            u_min, u_max = uncertainty.min(), uncertainty.max()
            if u_max - u_min < 1e-8:
                return router_p_base
            threshold = torch.quantile(uncertainty, 1.0 - min(self.curiosity_alpha, 1.0))
            escalate = (uncertainty >= threshold).unsqueeze(1).float()  # (B, 1)

            onehot = torch.zeros_like(router_p_base)
            onehot[:, int(self.precision_beta.argmax())] = 1.0

            return escalate * onehot + (1.0 - escalate) * router_p_base

        # ============================================================
        # Strategy 6: Soft Escalation (convex blend toward high precision)
        # ============================================================
        elif self.curiosity_strategy == "soft_escalation":
            # p^curious = (1 - m) · p + m · onehot(highest precision), with
            # m = clamp(alpha · u_norm, 0, 1): smooth version of escalation.
            u_min, u_max = uncertainty.min(), uncertainty.max()
            if u_max - u_min < 1e-8:
                return router_p_base
            u_norm = ((uncertainty - u_min) / (u_max - u_min + 1e-8)).unsqueeze(1).clamp(0, 1)
            mix = (self.curiosity_alpha * u_norm).clamp(0, 1)

            onehot = torch.zeros_like(router_p_base)
            onehot[:, int(self.precision_beta.argmax())] = 1.0

            return (1.0 - mix) * router_p_base + mix * onehot

        else:
            # Unknown strategy, return base routing
            return router_p_base


class Router(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_prob: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
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

            # Handle 3- or 4-value forward returns (curiosity-aware)
            forward_result = model(X)
            if len(forward_result) == 4:
                outputs, router_probs, load_balancing_loss_term, curiosity = forward_result
            else:
                outputs, router_probs, load_balancing_loss_term = forward_result
                curiosity = None

            if load_balancing:
                classification_loss = criterion(outputs, y)
                loss = classification_loss + cfg.experiment.router.load_balancing_alpha * load_balancing_loss_term
            else:
                loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # (Optional) quick curiosity log
            # if curiosity is not None:
            #     print(f"  Curiosity (mean uncertainty): {curiosity.mean().item():.6f}")

        train_losses.append(total_loss)

        # Validation within training loop
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for embeddings, labels in val_loader:
                X, y = embeddings.to(device), labels.to(device)

                # Handle 3- or 4-value forward returns (curiosity-aware)
                forward_result = model(X)
                if len(forward_result) == 4:
                    outputs, _, _, _ = forward_result
                else:
                    outputs, _, _ = forward_result

                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        print(
            f"Epoch {epoch+1}/{cfg.experiment.model.epochs}, "
            f"Train Loss: {total_loss:.4f}, Val Accuracy: {accuracy:.4f}, Val F1: {f1:.4f}"
        )

        val_losses.append(f1)

        if f1 > best_accuracy + (delta if early_stopping_enabled else 0):
            best_accuracy = f1
            print(f"  -> Validation f1 improved to {best_accuracy:.4f}. Saving model.")
            torch.save(model.state_dict(), ckpt_path)
            best_state = (model.state_dict(), train_losses, val_losses, best_accuracy, all_labels, all_preds, [])
            if early_stopping_enabled:
                patience_counter = 0
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

def _validate_moe_epoch(model, val_loader, criterion, device, fold_dir: str = "outputs"):
    """
    Validation helper for MoE models.
    Extended to collect and visualize curiosity (uncertainty) from BayesianRouter.

    Saves inside the current fold directory:
        • curiosity_values.json
        • curiosity_histogram.png
        • curiosity_per_class.png
    """
    import os, json
    import numpy as np
    import matplotlib.pyplot as plt

    model.eval()
    all_preds, all_labels, all_probs, all_curiosity = [], [], [], []

    with torch.no_grad():
        for embeddings, labels in val_loader:
            X, y = embeddings.to(device), labels.to(device)

            forward_result = model(X)
            if len(forward_result) == 4:
                outputs, _, _, curiosity = forward_result
                if curiosity is not None:
                    all_curiosity.extend(curiosity.cpu().numpy().tolist())
            else:
                outputs, _, _ = forward_result

            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())

    # ------------------------------------------------------------------
    # Save curiosity data if available
    # ------------------------------------------------------------------
    if len(all_curiosity) > 0:
        os.makedirs(fold_dir, exist_ok=True)

        curiosity_path = os.path.join(fold_dir, "curiosity_values.json")
        with open(curiosity_path, "w") as f:
            json.dump(
                {
                    "curiosity": all_curiosity,
                    "labels": list(map(int, all_labels[:len(all_curiosity)])),
                    "preds": list(map(int, all_preds[:len(all_curiosity)])),
                },
                f,
                indent=4,
            )
        print(f"[INFO] Saved curiosity values to {curiosity_path}")

        # ------------------------------------------------------------------
        # 1️⃣ Plot histogram of curiosity values
        # ------------------------------------------------------------------
        plt.figure(figsize=(7, 5))
        plt.hist(all_curiosity, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
        plt.title("Distribution of Curiosity (Epistemic Uncertainty)")
        plt.xlabel("Curiosity (variance across MC samples)")
        plt.ylabel("Number of samples")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        hist_path = os.path.join(fold_dir, "curiosity_histogram.png")
        plt.savefig(hist_path)
        plt.close()
        print(f"[INFO] Saved curiosity histogram to {hist_path}")

        # ------------------------------------------------------------------
        # 2️⃣ Plot average curiosity per predicted class
        # ------------------------------------------------------------------
        labels_np = np.array(all_labels[:len(all_curiosity)])
        preds_np = np.array(all_preds[:len(all_curiosity)])
        curiosity_np = np.array(all_curiosity)

        n_classes = int(np.max(preds_np)) + 1
        avg_curiosity_per_class = [
            curiosity_np[preds_np == c].mean() if np.any(preds_np == c) else 0
            for c in range(n_classes)
        ]

        plt.figure(figsize=(7, 5))
        plt.bar(range(n_classes), avg_curiosity_per_class, color="orange", edgecolor="black")
        plt.title("Average Curiosity per Predicted Class")
        plt.xlabel("Predicted Class ID")
        plt.ylabel("Mean Curiosity (uncertainty)")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        class_path = os.path.join(fold_dir, "curiosity_per_class.png")
        plt.savefig(class_path)
        plt.close()
        print(f"[INFO] Saved curiosity-per-class plot to {class_path}")

    else:
        print("[INFO] No curiosity values found — router is deterministic.")

    return 0, 0, all_labels, all_preds, all_probs




############# LUIS CURIOSITY IMPLEMENTATION #############
class BayesianRouter(nn.Module):
    """Router with Monte Carlo Dropout for uncertainty estimation (curiosity)."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout_prob: float = 0.2, mc_samples: int = 10):
        super().__init__()
        self.mc_samples = mc_samples
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),          # keep dropout layers in the graph
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        # init like the plain Router
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, compute_uncertainty: bool = False):
        """
        Returns:
            logits:      (B, E)
            uncertainty: (B,) epistemic (sum of per-expert variance) or None
        Notes:
            • During training, Monte Carlo Dropout uses multiple stochastic passes.
            • During eval (model.eval()), we default to a single pass with dropout disabled
              unless you explicitly call with compute_uncertainty=True AND ensure dropout
              is active (advanced use only).
        """
        # Fast path: no uncertainty requested -> single deterministic pass
        if not compute_uncertainty:
            return self.net(x), None

        # MC Dropout: multiple stochastic forward passes.
        # Ensure dropout is active for stochasticity: temporarily set train mode
        was_training = self.training
        self.train(True)
        preds = []
        for _ in range(self.mc_samples):
            preds.append(F.softmax(self.net(x), dim=1))
        # restore previous mode
        self.train(was_training)

        predictions = torch.stack(preds, dim=0)  # (mc, B, E)
        mean_pred = predictions.mean(dim=0)      # (B, E)
        # Epistemic uncertainty: aggregate variance across experts
        # Population variance (unbiased=False): with mc_samples=1 the unbiased
        # estimator divides by zero and yields NaN uncertainty, which crashed
        # downstream plotting; a single pass now gives zero uncertainty, and
        # uncertainty-consuming strategies correctly fall back to base routing.
        uncertainty = predictions.var(dim=0, unbiased=False).sum(dim=1)  # (B,)

        # Return logits consistent with routing path (use mean logits proxy via log-softmax inverse)
        # Safer: re-run a single pass for logits to keep numerical path identical
        logits = self.net(x)
        return logits, uncertainty
