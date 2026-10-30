"""An AST that reads fewer input frames, for clips shorter than the 1024
frames (10.24 s) the pretrained model pads every clip to.

transformers' ASTModel turns its [frames, mel bins] input into a grid of
patches ordered frequency first (index f * n_time + t) and adds one learned
position embedding per patch. shorten_ast keeps the first n_time' time
positions of every frequency row of that table (and the two special tokens'
entries), so a clip padded to max_length frames meets exactly the position
embeddings its patches had in the 1024-frame model; only the patches that
held padding beyond max_length are gone. Nothing is retrained. ESC-50 clips
(5 s, 498 fbank frames) fit in 512 frames and UrbanSound8K clips (at most 4 s,
398 frames) in 400, so the shorter model sees the whole clip.

Costs follow from the config: QWave.costs.ast_token_count reads
config.max_length, so QWave.backbone_quant.backbone_cost prices the shorter
model (602 tokens at 512 frames, 470 at 400) with no change.
"""
from __future__ import annotations

import torch
from torch import nn


def time_patches(config, max_length: int | None = None) -> int:
    n = config.max_length if max_length is None else max_length
    return (n - config.patch_size) // config.time_stride + 1


def freq_patches(config) -> int:
    return (config.num_mel_bins - config.patch_size) // config.frequency_stride + 1


@torch.no_grad()
def shorten_ast(model, max_length: int):
    """Truncate an ASTModel (or anything with .embeddings and .config) to
    max_length input frames in place and return it."""
    cfg = model.config
    if max_length > cfg.max_length:
        raise ValueError(f"max_length {max_length} above the model's {cfg.max_length}")
    nf, nt, nt2 = freq_patches(cfg), time_patches(cfg), time_patches(cfg, max_length)
    pos = model.embeddings.position_embeddings  # [1, 2 + nf * nt, D]
    if pos.shape[1] != 2 + nf * nt:
        raise ValueError(f"position table {tuple(pos.shape)} does not match a {nf} x {nt} patch grid")
    grid = pos[:, 2:].reshape(1, nf, nt, pos.shape[-1])[:, :, :nt2]
    new = torch.cat([pos[:, :2], grid.reshape(1, nf * nt2, pos.shape[-1])], dim=1).contiguous()
    model.embeddings.position_embeddings = nn.Parameter(new, requires_grad=pos.requires_grad)
    cfg.max_length = max_length
    return model


def kept_token_index(config_full, max_length: int) -> torch.Tensor:
    """Indices, in the full model's token order, of the tokens a model
    shortened to max_length keeps (the two special tokens first)."""
    nf, nt, nt2 = freq_patches(config_full), time_patches(config_full), time_patches(config_full, max_length)
    patches = torch.arange(nf * nt).reshape(nf, nt)[:, :nt2].reshape(-1) + 2
    return torch.cat([torch.tensor([0, 1]), patches])
