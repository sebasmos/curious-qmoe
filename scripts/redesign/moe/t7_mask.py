"""T7 shared pieces (ANALYSIS_PLAN_T7.md sections 3 and 5): patch grid, the three patch selectors, the masked forward of QuantAST and the priced cost.
No AST pass is used to choose patches."""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

NF = 12                      # frequency patches (128 mel bins, patch 16, stride 10)
PATCH, STRIDE = 16, 10
MASKS = [(s, r) for r in (25, 50) for s in ("energy", "random", "crop")]     # rho in percent


def n_time(max_length: int) -> int:
    return (max_length - PATCH) // STRIDE + 1


def n_patches(max_length: int) -> int:
    return NF * n_time(max_length)


def k_keep(max_length: int, rho_pct: int) -> int:
    return int(round(n_patches(max_length) * rho_pct / 100))


def valid_frames(n_samples: int, max_length: int) -> int:
    """Real Kaldi fbank frames of a clip (25 ms window, 10 ms shift, snip_edges), capped at the model's input length."""
    return int(min(max_length, 0 if n_samples < 400 else 1 + (n_samples - 400) // 160))


def patch_valid(n_v: int, max_length: int) -> np.ndarray:
    """[NF * nt] bool, grid index f * nt + t: patch valid when its first frame 10 t is a real frame."""
    nt = n_time(max_length)
    t = np.tile(np.arange(nt), NF)
    return (STRIDE * t) < n_v


@torch.no_grad()
def energy_scores(feats: torch.Tensor, n_v: np.ndarray) -> torch.Tensor:
    """feats [B, T, 128] (the extractor's normalized log-mel, CPU float32) -> [B, NF * nt] mean over each patch's bins and real frames."""
    B, T, _ = feats.shape
    x = feats.transpose(1, 2).unsqueeze(1).contiguous()                       # [B,1,128,T]
    m = (torch.arange(T)[None, :] < torch.as_tensor(n_v)[:, None]).float()    # [B,T]
    m = m[:, None, None, :].expand(B, 1, x.shape[2], T).contiguous()
    s = F.unfold(x * m, kernel_size=PATCH, stride=STRIDE).sum(1)
    c = F.unfold(m, kernel_size=PATCH, stride=STRIDE).sum(1)
    return s / c.clamp_min(1.0)


def select_energy(score: torch.Tensor, valid: np.ndarray, k: int) -> np.ndarray:
    """score [B, L] CPU, valid [B, L] bool -> [B, k] sorted grid indices. Highest valid scores; ties and the fill with padded patches go by grid index."""
    s = score.clone()
    s[torch.as_tensor(~valid)] = float("-inf")
    order = torch.argsort(s, dim=1, descending=True, stable=True)[:, :k]
    return np.sort(order.numpy(), axis=1)


def _fill(chosen: np.ndarray, valid_row: np.ndarray, k: int) -> np.ndarray:
    if len(chosen) >= k:
        return np.sort(chosen[:k])
    pad = np.flatnonzero(~valid_row)[:k - len(chosen)]
    return np.sort(np.concatenate([chosen, pad]))


def select_random(stem: str, rho_pct: int, valid_row: np.ndarray, k: int) -> np.ndarray:
    rng = np.random.default_rng(zlib.crc32(f"{stem}|{rho_pct}".encode()))
    v = np.flatnonzero(valid_row)
    chosen = rng.permutation(v)[:k] if len(v) >= k else v
    return _fill(chosen, valid_row, k)


def select_crop(valid_row: np.ndarray, k: int, max_length: int) -> np.ndarray:
    nt = n_time(max_length)
    v = np.flatnonzero(valid_row)
    order = v[np.argsort((v % nt) * NF + (v // nt), kind="stable")]           # time-major
    V = len(order)
    start = int(np.clip((V - k) // 2, 0, max(V - k, 0)))
    return _fill(order[start:start + k], valid_row, k)


def select(sel: str, rho_pct: int, feats: torch.Tensor, n_samples: np.ndarray, stems: list[str], max_length: int, score=None) -> np.ndarray:
    k = k_keep(max_length, rho_pct)
    n_v = np.array([valid_frames(int(n), max_length) for n in n_samples])
    valid = np.stack([patch_valid(int(v), max_length) for v in n_v])
    if sel == "energy":
        return select_energy(energy_scores(feats, n_v) if score is None else score, valid, k)
    if sel == "random":
        return np.stack([select_random(s, rho_pct, valid[i], k) for i, s in enumerate(stems)])
    if sel == "crop":
        return np.stack([select_crop(valid[i], k, max_length) for i in range(len(stems))])
    raise ValueError(sel)


@torch.no_grad()
def masked_forward(qast, feats: torch.Tensor, keep: torch.Tensor | None) -> torch.Tensor:
    """pooled output of a QuantAST (already shortened to feats.shape[1] frames) on the kept patches only.
    keep [B, k] grid indices (f * nt + t) on feats' device, or None for all patches. The patch embedding is the convolution evaluated on the kept
    patches (unfold, gather, matrix product); kept patches keep their position rows; the CLS and distillation tokens are always present."""
    emb = qast.ast.embeddings
    B = feats.shape[0]
    proj = emb.patch_embeddings.projection
    x = feats.unsqueeze(1).transpose(2, 3).contiguous()                       # [B,1,128,T]
    cols = F.unfold(x, kernel_size=proj.kernel_size, stride=proj.stride)      # [B,256,L], block order = conv output order
    L = cols.shape[-1]
    if keep is None:
        keep = torch.arange(L, device=feats.device)[None].expand(B, -1)
    g = cols.gather(2, keep[:, None, :].expand(-1, cols.shape[1], -1))        # [B,256,k]
    patches = torch.einsum("bck,dc->bkd", g, proj.weight.flatten(1)) + proj.bias
    pos = emb.position_embeddings
    if pos.shape[1] != 2 + L:
        raise ValueError(f"position table {tuple(pos.shape)} vs {L} patches")
    cls = emb.cls_token.expand(B, -1, -1) + pos[:, 0:1]
    dist = emb.distillation_token.expand(B, -1, -1) + pos[:, 1:2]
    h = torch.cat([cls, dist, patches + pos[0, 2:][keep]], dim=1)
    for layer, w, b in zip(qast.ast.encoder.layer, qast.weights, qast.bias):
        h = qast._layer(h, layer, w, b)
    h = qast.ast.layernorm(h)
    return (h[:, 0] + h[:, 1]) / 2


# ---- priced cost (stage 3c convention: idea 6 stage 2 analyze.ast_bitops_tok with PRICE 1024, plus fbank and head) ----
AST_L, AST_D, AST_FF = 12, 768, 3072
BLOCKS = [(3 * AST_D, AST_D), (AST_D, AST_D), (AST_FF, AST_D), (AST_D, AST_FF)]
PRICE = 1024.0
FBANK_BITOPS_US8K = 3112228956.408612             # stage 3c analysis/urban/costs.json fbank_bitops_mean
HEAD_ESC_768 = 716308480.0                        # stage 3c costs.json heads['768']['esc']['bitops']
E1_COST = 28363537408.0                           # cnn:mn10:w8a8:esc (T0 static_recomputed)
F_COST_STORED = 2085697280092.4087                # ast:w4g128@400+esc


def ast_bitops_tok(wb: int, g: int, tokens: int, patches: int, pbits: int = 16, price: float = PRICE) -> float:
    lin_macs = AST_L * sum(o * i for o, i in BLOCKS) * tokens
    att_half = tokens * tokens * AST_D * AST_L
    patch_macs = AST_D * 16 * 16 * patches
    resc = tokens * AST_L * sum(o * (i // g) for o, i in BLOCKS)
    return float(lin_macs * wb * 8 + att_half * 8 * 8 + att_half * pbits * 8 + patch_macs * 32 * 32 + resc * price)


def selector_bitops(sel: str, max_length: int) -> float:
    """one 32-bit add per input element per covering patch (2.56 patches cover an element), 1024 bit-ops per add; random and crop cost nothing."""
    return 128 * max_length * (PATCH / STRIDE) ** 2 * 1024 if sel == "energy" else 0.0


def priced_cost(sel: str | None, rho_pct: int, max_length: int = 400, fbank: float = FBANK_BITOPS_US8K, head: float = HEAD_ESC_768, front: bool = True) -> float:
    """per-clip bit-ops of the AST with w4g128 at the given selector/rho (sel None = full view). front=False drops the fbank and selector terms."""
    P = n_patches(max_length) if sel is None else k_keep(max_length, rho_pct)
    c = ast_bitops_tok(4, 128, P + 2, P) + head
    if front:
        c += fbank + (0.0 if sel is None else selector_bitops(sel, max_length))
    return c
