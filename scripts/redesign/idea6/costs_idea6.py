"""Per-clip costs of idea 6's two tiers, written to one JSON file.

EfficientNet-B3 backbone: MACs of every Conv2d and Linear in torchvision's
efficientnet_b3 at the input size the embeddings were made with (300 x 300,
data/esc/esc-50-data_processing.py and data/urban8k/urban8k-data_processing.py:
transforms.Resize((300, 300))), including the 1536 x 1536 Linear those scripts
put in place of the classifier, whose output is the stored embedding. Counted
with forward hooks on a zero image (architecture only; weights do not change
the count). Elementwise work (batch norm, SiLU, the squeeze-excitation gate
multiply, residual adds) is left out, as the AST count leaves out LayerNorm,
GELU and softmax. Bit-ops = MACs x weight bits x activation bits:
  fp32  32 x 32, what the stored embeddings are
  int8  8 x 8, an assumption: no int8 extraction exists, accuracy is the FP32 one

Heads: QWave.costs.cost on the network scripts/benchmark.py builds for each kind
(EfficientNet heads: 1536 inputs; AST heads: 768), per number of classes.

AST backbone: backbone_cost_per_clip of each clean embedding's JSON sidecar
(Stage 2 of idea 5).

    python scripts/redesign/idea6/costs_idea6.py --ast-emb STAGE2/emb/esc50 STAGE2/emb/urban8k --out costs.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from QWave.costs import cost  # noqa: E402
from scripts.benchmark import build_model  # noqa: E402

N_CLASSES = {"esc50": 50, "esc5": 5, "urban": 10}
EFF_KINDS = ["esc", "8", "4", "bitnet"]
AST_HEADS = ["esc", "4"]


def efficientnet_b3_macs(size: int = 300) -> dict:
    from torchvision.models import efficientnet_b3

    m = efficientnet_b3(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, m.classifier[1].in_features)
    m.eval()
    parts = {"conv": 0, "depthwise": 0, "squeeze_excitation": 0, "linear": 0}

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            kh, kw = mod.kernel_size
            macs = out.shape[2] * out.shape[3] * mod.out_channels * (mod.in_channels // mod.groups) * kh * kw
            key = "depthwise" if mod.groups > 1 else ("squeeze_excitation" if out.shape[2] == 1 else "conv")
            parts[key] += macs
        elif isinstance(mod, nn.Linear):
            parts["linear"] += mod.in_features * mod.out_features

    for mm in m.modules():
        if isinstance(mm, (nn.Conv2d, nn.Linear)):
            mm.register_forward_hook(hook)
    with torch.no_grad():
        out = m(torch.zeros(1, 3, size, size))
    assert out.shape == (1, 1536)
    macs = sum(parts.values())
    return {"input": [3, size, size], "macs": float(macs), "parts_macs": {k: float(v) for k, v in parts.items()},
            "params": int(sum(p.numel() for p in m.parameters())),
            "bitops": {"fp32": float(macs * 32 * 32), "int8": float(macs * 8 * 8)}}


def head_costs(in_dim: int, kinds) -> dict:
    cfg = OmegaConf.load(ROOT / "config" / "esc50.yaml")
    out = {}
    for task, n in N_CLASSES.items():
        out[task] = {}
        for k in kinds:
            with contextlib.redirect_stdout(io.StringIO()):
                macs, bitops = cost(build_model(k, in_dim, n, cfg))
            out[task][k] = {"macs": float(macs), "bitops": float(bitops)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ast-emb", nargs="+", required=True, type=Path, help="Stage 2 emb/<dataset> folders")
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    ast = {}
    for d in args.ast_emb:
        for p in sorted(d.glob("*__clean.json")):
            s = json.loads(p.read_text())
            c = s["backbone_cost_per_clip"]
            ast.setdefault(d.name, {})[s["config"]] = {
                "bitops": c["bitops"], "macs": c["macs"], "p_bits": s.get("p_bits"), "n_clips": s["n_clips"],
                "rows_extracted": s["rows_extracted"], "names_crc32": s["names_crc32"], "sidecar": str(p)}
    res = {"efficientnet_b3": efficientnet_b3_macs(args.size),
           "eff_heads": head_costs(1536, EFF_KINDS), "ast_heads": head_costs(768, AST_HEADS), "ast_backbone": ast}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    e = res["efficientnet_b3"]
    print(f"EfficientNet-B3 at {args.size}x{args.size}: {e['macs'] / 1e9:.4f} GMACs, "
          f"{e['bitops']['fp32'] / 1e12:.4f} T bit-ops fp32, {e['bitops']['int8'] / 1e12:.4f} T int8")


if __name__ == "__main__":
    main()
