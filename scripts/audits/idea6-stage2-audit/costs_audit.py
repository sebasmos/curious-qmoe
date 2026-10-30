"""Check 4 (recount): priced bit-ops per clip from first principles, plus the work the claim leaves out.

EfficientNet-B3 at 300 x 300: MACs, conv output elements and conv input elements from my own forward
hooks on torchvision's efficientnet_b3 (random weights; shapes only). Elementwise work (batch norm, SiLU,
sigmoid, gate multiply, residual add) from module hooks. AST: shape formulas for 1214, 602 and 470
tokens. Front ends: my own counts (sparse mel filterbank from torchaudio's filter matrix).
Writes outputs/costs_audit.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, R, S2  # noqa: E402

torch.set_num_threads(2)


def effnet_counts(size=300):
    from torchvision.models import efficientnet_b3
    from torchvision.models.efficientnet import MBConv
    from torchvision.ops.misc import SqueezeExcitation

    m = efficientnet_b3(weights=None).eval()
    m.classifier = nn.Identity()
    c = dict(conv_macs=0, conv_out=0, conv_in=0, dw_macs=0, dw_in=0, pw_in=0, stem_in=0, bn=0, silu=0, sigmoid=0,
             gate=0, residual=0, n_conv=0, pool=0)
    hooks = []

    def conv_hook(mod, inp, out):
        kh, kw = mod.kernel_size
        n_out = out[0].numel()
        c["conv_macs"] += n_out * (mod.in_channels // mod.groups) * kh * kw
        c["conv_out"] += n_out
        c["conv_in"] += inp[0][0].numel()
        c["n_conv"] += 1
        if mod.groups > 1:
            c["dw_macs"] += n_out * kh * kw
            c["dw_in"] += inp[0][0].numel()
        elif (kh, kw) == (1, 1):
            c["pw_in"] += inp[0][0].numel()
        else:
            c["stem_in"] += inp[0][0].numel()

    def count(key):
        def h(mod, inp, out):
            c[key] += out[0].numel()
        return h

    def se_hook(mod, inp, out):
        c["gate"] += out[0].numel()

    def mb_hook(mod, inp, out):
        if mod.use_res_connect:
            c["residual"] += out[0].numel()

    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.BatchNorm2d):
            hooks.append(mod.register_forward_hook(count("bn")))
        elif isinstance(mod, nn.SiLU):
            hooks.append(mod.register_forward_hook(count("silu")))
        elif isinstance(mod, nn.Sigmoid):
            hooks.append(mod.register_forward_hook(count("sigmoid")))
        elif isinstance(mod, SqueezeExcitation):
            hooks.append(mod.register_forward_hook(se_hook))
        elif isinstance(mod, MBConv):
            hooks.append(mod.register_forward_hook(mb_hook))
    with torch.no_grad():
        m(torch.zeros(1, 3, size, size))
    for h in hooks:
        h.remove()
    c["params"] = int(sum(p.numel() for p in m.parameters()))
    return {k: float(v) for k, v in c.items()}


def mel_frontend_macs(n_samples, n_fft=2048, hop=512, n_mels=128, sr=16000, size=300):
    import torchaudio
    fb = torchaudio.functional.melscale_fbanks(n_fft // 2 + 1, 0.0, sr / 2, n_mels, sr, norm="slaney", mel_scale="htk")
    nnz = int((fb != 0).sum())
    frames = 1 + n_samples // hop
    fft = 1.25 * n_fft * math.log2(n_fft)  # 2.5 n log2 n flops for a real FFT, as multiply-adds
    per_frame = n_fft + fft + 2 * (n_fft // 2 + 1) + n_mels  # window, FFT, power, log
    mel = frames * nnz
    norm = 2 * n_mels * frames  # dB offset and min-max scale
    resize = 3 * size * size * 4  # bilinear: 4 taps per output pixel per channel
    return dict(frames=frames, mel_nnz=nnz, macs=frames * per_frame + mel + norm + resize)


def fbank_macs(n_samples, sr=16000, n_mels=128, max_length=1024):
    import torchaudio
    from torchaudio.compliance.kaldi import get_mel_banks
    banks, _ = get_mel_banks(n_mels, 512, float(sr), 20.0, 0.0, 100.0, -500.0, 1.0)
    nnz = int((banks != 0).sum())
    frames = 1 + (n_samples - 400) // 160
    per = 3 * 400 + 1.25 * 512 * 9 + 2 * 257 + n_mels
    return dict(frames=frames, nnz=nnz, macs=frames * (per + nnz) + max_length * n_mels)


# AST shapes
L, D, FF = 12, 768, 3072
BLOCKS = [(3 * D, D), (D, D), (FF, D), (D, FF)]


def ast_tokens(max_length):
    nt = (max_length - 16) // 10 + 1
    return 12 * nt + 2, 12 * nt


def parse(cfg):
    base = cfg.split("_")[0]
    wb = int(base[1])
    g = int(base.split("g")[1]) if "g" in base else None
    return wb, g


def ast_bitops(cfg, max_length, priced=True):
    tok, pat = ast_tokens(max_length)
    lin = L * sum(o * i for o, i in BLOCKS) * tok
    qk = tok * tok * D * L  # Q K^T; P V the same
    patch = D * 256 * pat
    if cfg == "fp32":
        return (lin + 2 * qk + patch) * 1024.0
    wb, g = parse(cfg)
    pbits = 8 if cfg.endswith("_p8") else 16
    resc = tok * L * sum(o * ((i // g) if g else 1) for o, i in BLOCKS)
    return lin * wb * 8 + qk * 64 + qk * pbits * 8 + patch * 1024 + (resc * 1024 if priced else 0)


def ast_elementwise(max_length, quantized=True):
    """Elementwise operations the bit-op count leaves out, per clip: 2 LayerNorms (~4 ops per element),
    GELU (1 per element counted), softmax (3 per score), 2 residual adds, plus, when quantized, one
    quantize per input element of each Linear and of Q, K, V and P."""
    tok, _ = ast_tokens(max_length)
    per_layer = 2 * 4 * tok * D + tok * FF + 3 * 12 * tok * tok + 2 * tok * D
    if quantized:
        per_layer += tok * (D + D + D + FF) + 3 * tok * D + 12 * tok * tok
    return float(per_layer * L)


def main():
    eff = effnet_counts()
    side = {bb: json.loads((R / f"emb/esc50/{bb}__clean.json").read_text())["cost_per_clip"] for bb in ("fp32", "w8a8")}
    out = {"effnet": eff, "effnet_sidecar": side}
    ew = eff["bn"] + eff["silu"] + eff["sigmoid"] + eff["gate"] + eff["residual"]
    out["effnet_bitops"] = {
        "fp32": eff["conv_macs"] * 1024,
        "w8a8_unpriced": eff["conv_macs"] * 64,
        "w8a8_priced": eff["conv_macs"] * 64 + eff["conv_out"] * 1024,
        "elementwise_ops": ew,
        "act_quant_ops": eff["conv_in"] * 2,  # max-reduction compare + quantize multiply per input element
        "w8a8_priced_plus_quant": eff["conv_macs"] * 64 + eff["conv_out"] * 1024 + eff["conv_in"] * 2 * 1024,
        "w8a8_priced_plus_all": eff["conv_macs"] * 64 + eff["conv_out"] * 1024 + (eff["conv_in"] * 2 + ew) * 1024,
        "fp32_plus_all": (eff["conv_macs"] + ew) * 1024,
    }
    out["mel_frontend"] = {"esc50_5s": mel_frontend_macs(80000), "urban_4s": mel_frontend_macs(64000)}
    for k in out["mel_frontend"].values():
        k["bitops"] = k["macs"] * 1024
    lens = {d: np.load(R / f"emb/{d}/lengths__clean.npy") for d in ("esc50", "urban8k")}
    out["fbank"] = {d: float(np.mean([fbank_macs(int(n))["macs"] for n in np.unique(v) for _ in range(1)]))
                    for d, v in lens.items()}
    # mean over clips (weights by clip count)
    for d, v in lens.items():
        u, cnt = np.unique(v, return_counts=True)
        out["fbank"][d] = float(np.sum([fbank_macs(int(n))["macs"] * c for n, c in zip(u, cnt)]) / cnt.sum()) * 1024
        fr = 1 + 0 * u
        out.setdefault("mel_mean_bitops", {})[d] = float(np.sum([mel_frontend_macs(int(n))["macs"] * c
                                                                for n, c in zip(u, cnt)]) / cnt.sum()) * 1024
    cfgs = ["fp32", "w8g64", "w4g128", "n4", "w4", "w3g64", "w5", "w6", "w8"]
    out["ast_priced"] = {f"{c}@{L_}": ast_bitops(c, L_) for c in cfgs for L_ in (1024, 512, 400, 256)}
    out["ast_unpriced"] = {f"{c}@{L_}": ast_bitops(c, L_, False) for c in cfgs for L_ in (1024, 512, 400, 256)}
    out["ast_elementwise_ops"] = {L_: {"quant": ast_elementwise(L_), "fp32": ast_elementwise(L_, False)}
                                  for L_ in (1024, 512, 400, 256)}
    # compare with sidecars (unpriced)
    cmp = {}
    for c in ["fp32", "w8g64", "w4g128", "n4", "w3g64", "w4"]:
        for d, L_ in (("esc50", 512), ("urban8k", 400)):
            s = json.loads((R / f"emb_ast_short/{d}/{c}__clean.json").read_text())
            cmp[f"{d}:{c}@{L_}"] = (ast_bitops(c, L_, False), s["backbone_cost_per_clip"]["bitops"], s["max_length"])
        s = json.loads((S2 / f"emb/esc50/{c}__clean.json").read_text()) if (S2 / f"emb/esc50/{c}__clean.json").exists() else None
        if s:
            cmp[f"esc50:{c}@1024"] = (ast_bitops(c, 1024, False), s["backbone_cost_per_clip"]["bitops"], None)
    out["ast_sidecar_check"] = cmp
    claimed = json.loads((R / "analysis/esc50/costs.json").read_text())
    out["claimed_esc50"] = {"cheap_priced": claimed["cheap"]["priced"], "fbank": claimed["fbank_bitops_mean"],
                            "ast_backbone_priced": claimed["ast_backbone"]["priced"]}
    (OUT / "costs_audit.json").write_text(json.dumps(out, indent=1, default=float))
    e = out["effnet_bitops"]
    print("EffNet MACs", eff["conv_macs"], "sidecar", side["w8a8"]["backbone_macs"], "out", eff["conv_out"],
          side["w8a8"]["backbone_out_elems"])
    print({k: round(v / 1e12, 4) for k, v in e.items()})
    print("mel", out["mel_frontend"], "claimed", side["w8a8"]["frontend_bitops_mean"] / 1e12)
    print("fbank", {k: v / 1e12 for k, v in out["fbank"].items()}, "claimed", claimed["fbank_bitops_mean"] / 1e12)
    for k, v in cmp.items():
        print(k, v[0], v[1], "equal" if abs(v[0] - v[1]) <= 1e-6 * v[1] else "DIFF", v[2])
    print({k: round(v / 1e12, 3) for k, v in out["ast_priced"].items()})
    print("ast elementwise", out["ast_elementwise_ops"])


if __name__ == "__main__":
    main()
