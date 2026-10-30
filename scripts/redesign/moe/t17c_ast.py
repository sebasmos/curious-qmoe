"""Copy of t12_common.py of the T12 branch (not in this worktree), used by T17(c). Shared pieces for T12 (ANALYSIS_PLAN_T12.md): the AST split at block B, a LoRA tail, cache reading. Stored results are read only."""
from __future__ import annotations
import copy, os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # T20: the pinned revision is cached; avoid network timeouts
from pathlib import Path
import numpy as np, torch
from torch import nn
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/redesign/idea6/stage2")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from QWave.ast_frames import shorten_ast  # noqa: E402

MODEL, REV = "MIT/ast-finetuned-audioset-10-10-0.4593", "f826b80d28226b62986cc218e5cec390b1096902"
NL = 12


def n_tokens(L: int) -> int: return 2 + 12 * ((L - 16) // 10 + 1)


def load_ast(L: int, device="cpu"):
    from transformers import ASTFeatureExtractor, ASTModel
    ext = ASTFeatureExtractor.from_pretrained(MODEL, revision=REV, max_length=L)
    m = shorten_ast(ASTModel.from_pretrained(MODEL, revision=REV), L).to(device).eval()
    for p in m.parameters(): p.requires_grad_(False)
    return ext, m


def layer_fwd(layer, h):
    o = layer(h)
    return o[0] if isinstance(o, tuple) else o


@torch.no_grad()
def front(model, feats, B: int):
    """Tokens entering block B (embeddings, then blocks 0..B-1), fp32."""
    h = model.embeddings(feats)
    for l in model.encoder.layer[:B]: h = layer_fwd(l, h)
    return h


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base, self.s = base, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / np.sqrt(base.in_features))
        self.Bm = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
    def forward(self, x): return self.base(x) + (x @ self.A.t() @ self.Bm.t()) * self.s
    def merged(self) -> nn.Linear:
        lin = copy.deepcopy(self.base)
        with torch.no_grad(): lin.weight += (self.Bm @ self.A) * self.s
        return lin


def _linears(layer):
    a = layer.attention.attention
    return [(a, "query"), (a, "key"), (a, "value"), (layer.attention.output, "dense"), (layer.intermediate, "dense"), (layer.output, "dense")]


class Tail(nn.Module):
    """Blocks B..11, final LayerNorm, pooling, linear head. mode: 'full' (all weights of the k blocks trainable), 'lora' (LoRA rank r on the six linears, rest frozen),
    'head' (nothing but the head and the final LayerNorm)."""
    def __init__(self, model, B: int, mode: str, ncls=10, r=16, alpha=32.0, drop=0.1):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(l) for l in model.encoder.layer[B:]])
        self.ln = copy.deepcopy(model.layernorm); self.head = nn.Linear(768, ncls); self.drop = nn.Dropout(drop); self.mode = mode
        for p in self.parameters(): p.requires_grad_(False)
        for p in list(self.ln.parameters()) + list(self.head.parameters()): p.requires_grad_(True)
        for l in self.layers:
            if mode == "full":
                for p in l.parameters(): p.requires_grad_(True)
            elif mode == "lora":
                for mod, name in _linears(l): setattr(mod, name, LoRALinear(getattr(mod, name), r, alpha))
        self.to(model.embeddings.position_embeddings.device)
    def pooled(self, h):
        for l in self.layers: h = layer_fwd(l, h)
        h = self.ln(h); return (h[:, 0] + h[:, 1]) / 2
    def forward(self, h): return self.head(self.drop(self.pooled(h)))
    def param_groups(self, lr_body, lr_head):
        body = [p for l in self.layers for p in l.parameters() if p.requires_grad]
        return [{"params": body + list(self.ln.parameters()), "lr": lr_body}, {"params": list(self.head.parameters()), "lr": lr_head}]
    def write_back(self, model, B: int):
        """Put the (merged) tail weights into `model` (an ASTModel) so QuantAST can quantize them."""
        with torch.no_grad():
            for i, l in enumerate(self.layers):
                tgt = model.encoder.layer[B + i]
                for (m1, n1), (m2, n2) in zip(_linears(l), _linears(tgt)):
                    w = getattr(m1, n1); w = w.merged() if isinstance(w, LoRALinear) else w
                    getattr(m2, n2).weight.copy_(w.weight); getattr(m2, n2).bias.copy_(w.bias)
                tp = dict(tgt.named_parameters())
                for n, p in l.named_parameters():
                    if "layernorm" in n: tp[n].copy_(p)
            model.layernorm.load_state_dict(self.ln.state_dict())


class TokCache:
    """fp16 memmap [n, T, 768] with the global row ids it holds."""
    def __init__(self, d: Path, name: str):
        self.rows = np.load(d / f"{name}_rows.npy"); self.pos = {int(r): i for i, r in enumerate(self.rows)}
        self.tok = np.lib.format.open_memmap(d / f"{name}_tok.npy", mode="r")
        self.emb = np.load(d / f"{name}_frozen_emb.npy")
    def get(self, rows, device):
        idx = np.array([self.pos[int(r)] for r in rows]); o = np.argsort(idx)
        x = np.empty((len(idx),) + self.tok.shape[1:], np.float16); x[o] = self.tok[idx[o]]
        return torch.from_numpy(x).to(device).float()
