import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F
from .models import reset_weights
from .qmoe_layers import BitNetExpert158b, BitNetExpert, SharedTrunkMoE


def expected_loss_router_loss(router_logits, expert_ce, entropy_bonus=0.0):
    """Router loss of the dense mode: mean over clips of sum_i p_i * CE_i,
    with CE_i expert i's per-clip loss (detached), minus entropy_bonus * H(p).
    Without the bonus the optimum is one-hot on the lowest-loss expert; with
    it the optimum is p_i proportional to exp(-CE_i / entropy_bonus)."""
    logp = F.log_softmax(router_logits, dim=1)
    p = logp.exp()
    loss = (p * expert_ce.detach()).sum(1).mean()
    if entropy_bonus:
        loss = loss + entropy_bonus * (p * logp).sum(1).mean()
    return loss


class qMoEModelBatched(nn.Module):
    def __init__(self, cfg, in_dim, num_classes, num_experts=4, top_k=2):
        super().__init__()

        # ──────────────────────────────────────────────────────────────
        # Router redesign options (redesign idea 4). The defaults keep the
        # original behavior and the original state dict.
        #   dense_experts: in training every expert runs on every clip and is
        #     trained on it; the router is trained on sum_i p_i * CE_i with the
        #     expert losses detached. Inference stays top-k and the output is
        #     the selected expert's logits, not scaled by the routing prob.
        #   router_entropy_bonus: subtracts bonus * H(p) from that router loss.
        #   error_head: one extra router output trained (BCE, detached target)
        #     to predict whether the lowest-precision expert misclassifies.
        #   uncertainty_source: "mc_variance" (MC-dropout routing variance) or
        #     "error_head" (MC mean of the predicted error probability).
        #   escalation_rule: "threshold" sends clips whose uncertainty is above
        #     the escalation_tau buffer to the highest-precision expert at
        #     inference; tau is chosen on the inner validation split
        #     (select_escalation_tau), after training.
        #   router_holdout: share of the fit split kept away from the experts
        #     and used only to train the router and error head, on the
        #     experts' losses for clips they never trained on (benchmark.py
        #     makes the split, train_moe_local the two phases). The experts
        #     interpolate their own training clips, so in-sample expert
        #     losses carry no error signal.
        # ──────────────────────────────────────────────────────────────
        r = cfg.experiment.router
        self.dense_experts = bool(getattr(r, "dense_experts", False))
        if float(getattr(r, "router_holdout", 0.0)) > 0 and not self.dense_experts:
            raise ValueError("router_holdout needs dense_experts=true")
        self.router_entropy_bonus = float(getattr(r, "router_entropy_bonus", 0.0))
        self.error_head = bool(getattr(r, "error_head", False))
        self.uncertainty_source = str(getattr(r, "uncertainty_source", "mc_variance"))
        self.escalation_rule = str(getattr(r, "escalation_rule", "none"))
        use_curiosity = getattr(cfg.experiment.router, "use_curiosity", False)
        if self.uncertainty_source not in ("mc_variance", "error_head"):
            raise ValueError(f"uncertainty_source must be mc_variance or error_head, got {self.uncertainty_source}")
        if self.escalation_rule not in ("none", "threshold"):
            raise ValueError(f"escalation_rule must be none or threshold, got {self.escalation_rule}")
        if self.dense_experts and getattr(r, "shared_trunk", False):
            raise ValueError("dense_experts is not implemented for the shared trunk")
        if (self.error_head or self.router_entropy_bonus) and not self.dense_experts:
            raise ValueError("error_head and router_entropy_bonus are trained by the dense_experts loss")
        if (self.error_head or self.escalation_rule != "none") and not use_curiosity:
            raise ValueError("error_head and escalation_rule need the Bayesian router (use_curiosity=true)")
        if self.uncertainty_source == "error_head" and not self.error_head:
            raise ValueError("uncertainty_source=error_head needs error_head=true")
        if self.error_head and getattr(r, "gate_threshold", 0.0) > 0:
            raise ValueError("error_head is not implemented with gate_threshold")

        # ──────────────────────────────────────────────────────────────
        # Router: deterministic vs Bayesian (curiosity via MC Dropout)
        # ──────────────────────────────────────────────────────────────
        if use_curiosity:
            print("Using BayesianRouter with curiosity (Monte Carlo Dropout).")
            self.router = BayesianRouter(
                in_dim,
                cfg.experiment.router.hidden_dim,
                num_experts,
                dropout_prob=cfg.experiment.model.dropout_prob,
                mc_samples=getattr(cfg.experiment.router, "mc_samples", 10),
                error_head=self.error_head,
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

        # Running range of the router's epistemic uncertainty, BatchNorm-style.
        # Train mode normalizes with the batch range and updates these EMA
        # buffers (the first tracked batch initializes them); eval mode
        # normalizes with the buffers and clamps to [0, 1], so a sample's
        # routing does not depend on the rest of its batch and the prior still
        # acts at batch size 1. The escalation strategy keeps a running
        # threshold the same way. Saved in the checkpoint with the weights.
        self.unc_momentum = 0.1
        self.register_buffer("unc_running_min", torch.tensor(0.0))
        self.register_buffer("unc_running_max", torch.tensor(1.0))
        self.register_buffer("unc_running_threshold", torch.tensor(float("inf")))
        self.register_buffer("unc_num_batches_tracked", torch.tensor(0, dtype=torch.long))
        if self.escalation_rule == "threshold":
            # inf escalates nothing; set after training from the inner
            # validation split and saved in the checkpoint
            self.register_buffer("escalation_tau", torch.tensor(float("inf")))

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
            if bit_width in ("esc", "qesc"):
                # FP32 expert on the same pre-LN architecture as the k-bit
                # experts; "qesc" is int8-quantized after training
                # (ptq_dynamic_int8) before the test pass.
                print(f"  -> Creating an FP32 expert ({bit_width}).")
                experts.append(
                    BitNetExpert(
                        in_dim,
                        num_classes,
                        hidden_sizes=cfg.experiment.model.hidden_sizes,
                        dropout_prob=cfg.experiment.model.dropout_prob,
                        num_bits=32,
                    )
                )
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
                # BitwisePopcountLinear sums XOR'd bytes instead of counting
                # set bits, so it does not compute an XNOR-popcount dot
                # product, and it has no gradient path, so it cannot train.
                raise ValueError("the 'popcount' expert is not a correct XNOR-popcount layer and "
                                 "cannot be trained; it is not part of any paper configuration")
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
                base_logits = self.router.net(x)[:, :self.num_experts]
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
        if self.uncertainty_source == "error_head" and uncertainty is not None:
            uncertainty = self.router.aux["error_prob"]

        # CURIOSITY MECHANISM: Modify routing probabilities based on uncertainty
        if self.use_curiosity and uncertainty is not None:
            router_p = self._apply_curiosity(router_out, uncertainty)
        else:
            router_p = F.softmax(router_out, dim=1)  # Standard routing
        if self.escalation_rule == "threshold" and not self.training and uncertainty is not None:
            escalate = (uncertainty > self.escalation_tau).unsqueeze(1)
            onehot = torch.zeros_like(router_p)
            onehot[:, int(self.precision_beta.argmax())] = 1.0
            router_p = torch.where(escalate, onehot, router_p)
        k_val, k_idx = torch.topk(router_p, self.top_k, dim=1)  # (B, K)

        out = x.new_zeros(B, self.num_classes)

        # load-balancing L2 loss (auxiliary)
        lb_loss = torch.sum(router_p.mean(0) ** 2) if self.training else 0.0

        if self.dense_experts:
            if self.training:
                # every expert on every clip; dense_loss reads this cache
                expert_logits = torch.stack([e(x) for e in self.experts])  # (E, B, C)
                self._dense = {"expert_logits": expert_logits, "router_logits": router_out}
                out = expert_logits[k_idx, torch.arange(B, device=x.device).unsqueeze(1)].mean(1)
                return out, router_p, lb_loss, uncertainty
            for e_idx, expert in enumerate(self.experts):
                rows = (k_idx == e_idx).nonzero(as_tuple=True)[0]
                if rows.numel() > 0:
                    out[rows] += expert(x[rows])  # unscaled logits
            return out / self.top_k, router_p, lb_loss, uncertainty

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

    def dense_loss(self, y, criterion, part="all"):
        """Training loss of the dense mode for the last train-mode forward:
        each expert's criterion loss on every clip, the expected-loss router
        loss on the router's base logits (before the precision prior), and,
        with error_head, BCE of the error output against whether the
        lowest-precision expert misclassified the clip (detached).
        part: "all", "experts" (expert losses only) or "router" (router and
        error head only, for clips held out from the experts)."""
        d = self._dense
        logits = d["expert_logits"]
        loss = sum(criterion(l, y) for l in logits) if part in ("all", "experts") else 0.0
        if part == "experts":
            return loss
        with torch.no_grad():
            ce = torch.stack([F.cross_entropy(l, y, reduction="none") for l in logits], dim=1)  # (B, E)
        loss = loss + expected_loss_router_loss(d["router_logits"], ce, self.router_entropy_bonus)
        if self.error_head:
            cheap = int(self.precision_beta.argmin())
            target = (logits[cheap].detach().argmax(1) != y).float()
            loss = loss + F.binary_cross_entropy_with_logits(self.router.aux["error_logit"], target)
        return loss

    def _ema(self, buf, value):
        if self.unc_num_batches_tracked == 0:
            buf.copy_(value)
        else:
            buf.mul_(1.0 - self.unc_momentum).add_(self.unc_momentum * value)

    def _normalized_uncertainty(self, uncertainty: torch.Tensor):
        """Uncertainty mapped to [0, 1] as a (B, 1) column, or None when the
        range is degenerate (then the strategies fall back to base routing).
        Train mode: batch min-max, and the running range is updated. Eval
        mode: the running range, clamped, independent of batch composition."""
        if self.training:
            u_min, u_max = uncertainty.min(), uncertainty.max()
            if u_max - u_min < 1e-8:
                return None
            with torch.no_grad():
                self._ema(self.unc_running_min, u_min.detach())
                self._ema(self.unc_running_max, u_max.detach())
                self.unc_num_batches_tracked += 1
        else:
            u_min, u_max = self.unc_running_min, self.unc_running_max
            if u_max - u_min < 1e-8:
                return None
        return ((uncertainty - u_min) / (u_max - u_min + 1e-8)).unsqueeze(1).clamp(0, 1)

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
            u_norm = self._normalized_uncertainty(uncertainty)  # (B, 1)
            if u_norm is None:
                return router_p_base

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
            # inputs stay on the cheap low-bit experts. u is normalized to
            # [0, 1] (running range in eval mode, see _normalized_uncertainty).
            u_norm = self._normalized_uncertainty(uncertainty)  # (B, 1)
            if u_norm is None:
                return router_p_base
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

            u_norm = self._normalized_uncertainty(uncertainty)
            if u_norm is None:
                return sharpened
            beta = self.precision_beta.unsqueeze(0)

            router_p = sharpened * torch.exp(self.curiosity_alpha * u_norm * beta)
            router_p = router_p / router_p.sum(dim=1, keepdim=True)
            return router_p

        # ============================================================
        # Strategy 5: Escalation (hard uncertainty guard)
        # ============================================================
        elif self.curiosity_strategy == "escalation":
            # Samples above the (1 - alpha) uncertainty quantile route
            # outright to the highest-precision expert; everything else keeps
            # base routing. alpha=0 escalates nothing (identity). Train mode
            # takes the quantile of the batch and tracks it as a running
            # threshold; eval mode uses the running threshold, so the
            # decision is per sample rather than relative to the batch.
            if self.curiosity_alpha <= 0:
                return router_p_base
            if self.training:
                if uncertainty.max() - uncertainty.min() < 1e-8:
                    return router_p_base
                threshold = torch.quantile(uncertainty, 1.0 - min(self.curiosity_alpha, 1.0))
                with torch.no_grad():
                    self._ema(self.unc_running_threshold, threshold.detach())
                    self.unc_num_batches_tracked += 1
            else:
                threshold = self.unc_running_threshold
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
            u_norm = self._normalized_uncertainty(uncertainty)
            if u_norm is None:
                return router_p_base
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


def train_moe_local(cfg, load_balancing, model, train_loader, val_loader, class_weights, in_dim, device, fold_dir, resume, ckpt_path,
                    router_loader=None):
    # router_loader (dense mode with router_holdout): clips the experts never
    # train on; each epoch the experts train on train_loader, then the router
    # and error head train on router_loader against the experts' eval-mode
    # losses on those clips.
    # val_loader drives early stopping and checkpoint selection, so it must
    # never be the fold that is reported (benchmark.py passes an inner
    # validation split carved out of the training folds).
    # Same recipe as the single models (experiment.train): optimizer, loss,
    # LR schedule, early stopping and gradient clipping.
    from .train_utils import build_training_recipe
    rc = cfg.experiment.train
    optimizer, criterion, scheduler = build_training_recipe(rc, model, class_weights)

    best_accuracy = -1.0
    patience_counter = 0
    best_state = None
    patience, delta = rc.patience, rc.min_delta

    train_losses, val_losses = [], []

    for epoch in range(rc.max_epochs):
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

            dense = getattr(model, "dense_experts", False)
            if dense:
                classification_loss = model.dense_loss(y, criterion, "all" if router_loader is None else "experts")
            else:
                classification_loss = criterion(outputs, y)
            if load_balancing and router_loader is None:
                loss = classification_loss + cfg.experiment.router.load_balancing_alpha * load_balancing_loss_term
            else:
                loss = classification_loss

            optimizer.zero_grad()
            loss.backward()
            if dense:
                # each expert (and the router) clipped on its own, as a single
                # model trained alone would be
                for part in [*model.experts, model.router]:
                    torch.nn.utils.clip_grad_norm_(part.parameters(), rc.grad_clip_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), rc.grad_clip_norm)
            optimizer.step()
            total_loss += loss.item()

            # (Optional) quick curiosity log
            # if curiosity is not None:
            #     print(f"  Curiosity (mean uncertainty): {curiosity.mean().item():.6f}")

        if router_loader is not None:
            for embeddings, labels in router_loader:
                X, y = embeddings.to(device), labels.to(device)
                model.train()
                for expert in model.experts:
                    expert.eval()  # targets: the experts as they run at test time
                _, _, lb_term, _ = model(X)
                loss = model.dense_loss(y, criterion, "router")
                if load_balancing:
                    loss = loss + cfg.experiment.router.load_balancing_alpha * lb_term
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.router.parameters(), rc.grad_clip_norm)
                optimizer.step()
                total_loss += loss.item()

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
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        print(
            f"Epoch {epoch+1}/{rc.max_epochs}, "
            f"Train Loss: {total_loss:.4f}, Val Accuracy: {accuracy:.4f}, Val F1: {f1:.4f}"
        )

        val_losses.append(f1)
        scheduler.step(f1)

        if f1 > best_accuracy + delta:
            best_accuracy = f1
            print(f"  -> Validation f1 improved to {best_accuracy:.4f}. Saving model.")
            torch.save(model.state_dict(), ckpt_path)
            best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_state = (best_weights, train_losses, val_losses, best_accuracy, all_labels, all_preds, [])
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  -> No improvement for {patience_counter} epoch(s). Patience is {patience}.")

        if patience_counter >= patience:
            print(f"\nEARLY STOPPING: Validation F1 has not improved by >{delta} for {patience} epochs.")
            break

    # leave the model at the selected checkpoint, as train_pytorch_local does
    model.load_state_dict(best_state[0])
    return best_state


@torch.no_grad()
def select_escalation_tau(model, loader, device, seed, rule="target_share", target_share=0.3):
    """Escalation threshold for escalation_rule=threshold, from one split, the
    inner validation split. Each clip's uncertainty and routed prediction are
    taken with the hard rule off (MC dropout seeded as in the scored pass);
    a clip escalates when its uncertainty is above tau. "target_share" picks
    the candidate whose escalation share is closest to target_share,
    "max_f1" the one with the highest macro F1; ties go to fewer escalations.
    Sets model.escalation_tau and returns (tau, info)."""
    model.eval()
    model.escalation_tau.fill_(float("inf"))
    hi = int(model.precision_beta.argmax())
    torch.manual_seed(seed)
    us, preds, preds_hi, ys = [], [], [], []
    for X, y in loader:
        X = X.to(device)
        out, _, _, u = model(X)
        us.append(u.cpu()); preds.append(out.argmax(1).cpu()); ys.append(y.cpu())
        preds_hi.append(model.experts[hi](X).argmax(1).cpu())
    u, pred, pred_hi, y = (torch.cat(t).numpy() for t in (us, preds, preds_hi, ys))
    vals = np.unique(u)
    best = None
    for t in np.concatenate([[-np.inf], (vals[:-1] + vals[1:]) / 2, [np.inf]]):
        esc = u > t
        f1 = f1_score(y, np.where(esc, pred_hi, pred), average="macro", zero_division=0)
        if rule == "max_f1":
            key = (-f1, esc.mean())
        elif rule == "target_share":
            key = (abs(esc.mean() - target_share), esc.mean())
        else:
            raise ValueError(f"unknown escalation_tau_rule '{rule}'")
        if best is None or key < best[0]:
            best = (key, float(t), float(esc.mean()), float(f1))
    _, tau, share, f1 = best
    model.escalation_tau.fill_(tau)
    return tau, {"tau_rule": rule, "target_share": float(target_share) if rule == "target_share" else None,
                 "tau": tau, "inner_val_escalation_share": share, "inner_val_f1_macro": f1,
                 "inner_val_f1_macro_rule_off": float(f1_score(y, pred, average="macro", zero_division=0))}

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
        print("[INFO] No curiosity values found, router is deterministic.")

    return 0, 0, all_labels, all_preds, all_probs




############# LUIS CURIOSITY IMPLEMENTATION #############
class BayesianRouter(nn.Module):
    """Router with Monte Carlo Dropout for uncertainty estimation (curiosity)."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout_prob: float = 0.2, mc_samples: int = 10, error_head: bool = False):
        super().__init__()
        self.mc_samples = mc_samples
        # error_head: one extra output column, a logit for "the cheapest
        # expert misclassifies this clip"; routing uses the first output_dim
        self.num_routes = output_dim
        self.error_head = error_head
        # per-forward side outputs: mc_variance, and with the error head
        # error_logit (last pass) and error_prob (MC mean of the sigmoid)
        self.aux = {}
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),          # keep dropout layers in the graph
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim // 2, output_dim + int(error_head)),
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
        E = self.num_routes
        # Fast path: no uncertainty requested -> single deterministic pass
        if not compute_uncertainty:
            out = self.net(x)
            self.aux = {"error_logit": out[:, E]} if self.error_head else {}
            return out[:, :E], None

        # MC Dropout: multiple stochastic forward passes.
        # Ensure dropout is active for stochasticity: temporarily set train mode
        was_training = self.training
        self.train(True)
        preds, errs = [], []
        for _ in range(self.mc_samples):
            o = self.net(x)
            preds.append(F.softmax(o[:, :E], dim=1))
            if self.error_head:
                errs.append(torch.sigmoid(o[:, E]))
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
        out = self.net(x)
        self.aux = {"mc_variance": uncertainty}
        if self.error_head:
            self.aux.update(error_prob=torch.stack(errs).mean(0), error_logit=out[:, E])
        return out[:, :E], uncertainty
