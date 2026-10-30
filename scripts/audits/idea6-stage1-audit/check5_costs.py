"""Check 5: costs of both tiers, recounted independently of costs_idea6.py.

1. EfficientNet-B3 at 300 x 300 with the extractor's 1536 x 1536 Linear: MACs from torch's own
   FlopCounterMode (FLOPs / 2, conv and matmul), output elements of every conv and Linear (the
   number of per-channel rescales an int8 version pays), and the elementwise work the claim leaves
   out (batch norm, SiLU, sigmoid, the squeeze-excitation multiply, residual adds), counted with hooks.
2. AST backbone bit-ops from first principles (shapes of MIT/ast-finetuned-audioset-10-10-0.4593 at
   1024 frames: 1212 patches + 2 tokens), checked against every clean sidecar; view (ii) adds one
   32 x 32 multiply-add per token, output row and weight group (per-channel: one per row).
3. Front ends, counted analytically: the mel image for EfficientNet (STFT 2048 / hop 512 as the
   paper says; sample rate and mel bands unknown, so 22.05 and 44.1 kHz with 128 bands, dense
   filterbank matmul) and the Kaldi fbank for AST (16 kHz, 25 ms / 10 ms, 512-point FFT, 128 bands,
   plus resampling from the file rate).
4. Flag only: AST at 512 frames (ESC-50 clips are 5 s and are padded to 10.24 s) and 400 frames
   (UrbanSound8K clips are at most 4 s), with position embeddings cut to the first columns. Cost only,
   nothing is run for accuracy.
5. Wall clock, FP32, batch 1, CPU, 2 threads: EfficientNet-B3 at 300 x 300 and AST at 1024 frames.

    python check5_costs.py OUT.json
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

torch.set_num_threads(2)
H = Path.home() / "Documents"
import os; RD = Path(os.environ.get("RESULTS_DIR", str(H / "results")))  # results root (archive edit: path only)
S2 = RD / "2026-10-20_2359_redesign-idea5-stage2_code-3c18c38/emb"
I6 = RD / "2026-10-22_0953_redesign-idea6-stage1_code-ec70260"

# ---------------------------------------------------------------- AST shapes
L, D, FF = 12, 768, 3072
BLOCKS = [(3 * D, D), (D, D), (FF, D), (D, FF)]
LIN_W = L * sum(o * i for o, i in BLOCKS)


def ast_shape(frames):
    f_patches = (128 - 16) // 10 + 1                 # 12
    t_patches = (frames - 16) // 10 + 1              # 101 at 1024 frames
    patches = f_patches * t_patches
    return patches, patches + 2


def ast_bitops(cfg, frames=1024, rescale_price=0.0):
    patches, tok = ast_shape(frames)
    lin_macs = LIN_W * tok
    att_half = tok * tok * D * L
    patch_macs = D * 16 * 16 * patches
    if cfg == "fp32":
        return (lin_macs + 2 * att_half + patch_macs) * 1024.0
    base, _, suf = cfg.partition("_")
    pbits = 8 if suf == "p8" else 16
    wb = int(base[1])
    g = int(base.split("g")[1]) if "g" in base else None
    b = lin_macs * wb * 8 + att_half * 64 + att_half * pbits * 8 + patch_macs * 1024
    resc = tok * L * sum(o * (i // g if g else 1) for o, i in BLOCKS)
    return float(b + resc * rescale_price)


def eff_counts():
    from torchvision.models import efficientnet_b3
    m = efficientnet_b3(weights=None)
    m.classifier[1] = nn.Linear(1536, 1536)
    m.eval()
    x = torch.zeros(1, 3, 300, 300)
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        m(x)
    flops = fc.get_total_flops()
    by_op = {str(k): int(v) for k, v in fc.get_flop_counts()["Global"].items()}
    outs = {"conv_linear_outputs": 0, "bn": 0, "silu": 0, "sigmoid": 0, "se_mul": 0, "residual_add": 0}

    def hk(mod, inp, out):
        n = out.numel()
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            outs["conv_linear_outputs"] += n
        elif isinstance(mod, nn.BatchNorm2d):
            outs["bn"] += n
        elif isinstance(mod, nn.SiLU):
            outs["silu"] += n
        elif isinstance(mod, nn.Sigmoid):
            outs["sigmoid"] += n
    from torchvision.ops.misc import SqueezeExcitation
    from torchvision.models.efficientnet import MBConv

    def se_hk(mod, inp, out):
        outs["se_mul"] += out.numel()

    def mb_hk(mod, inp, out):
        if mod.use_res_connect:
            outs["residual_add"] += out.numel()
    for mm in m.modules():
        mm.register_forward_hook(hk)
        if isinstance(mm, SqueezeExcitation):
            mm.register_forward_hook(se_hk)
        if isinstance(mm, MBConv):
            mm.register_forward_hook(mb_hk)
    with torch.no_grad():
        m(x)
    return {"macs_flopcounter": flops / 2, "flops_by_op": by_op, "elementwise_and_outputs": outs,
            "params": int(sum(p.numel() for p in m.parameters()))}


def fft_real_macs(n):
    # real FFT of n points via an n/2 complex radix-2 FFT: (n/4) log2(n/2) butterflies, each one complex
    # multiply (4 real multiply-adds) and 2 complex adds (counted as 2 MACs); plus n/2 post-processing
    # twiddles (4 MACs each)
    return (n / 4) * math.log2(n / 2) * 6 + (n / 2) * 4


def mel_front_end(sr, seconds, n_fft=2048, hop=512, n_mels=128):
    frames = int(sr * seconds) // hop + 1
    per = n_fft + fft_real_macs(n_fft) + 2 * (n_fft // 2 + 1) + n_mels * (n_fft // 2 + 1) + n_mels
    return {"sr": sr, "frames": frames, "macs": frames * per, "bitops_fp32": frames * per * 1024.0}


def fbank_front_end(file_sr, seconds, n_mels=128):
    # resample file_sr -> 16 kHz (torchaudio sinc, lowpass width 6): about 2 * 6 * file_sr / 16000 + 1 taps
    taps = 2 * 6 * max(1.0, file_sr / 16000) + 1
    resample = 16000 * seconds * taps if file_sr != 16000 else 0
    frames = int((16000 * seconds - 400) // 160) + 1
    per = 400 * 3 + fft_real_macs(512) + 2 * 257 + n_mels * 257 + n_mels  # dither, DC, pre-emphasis, window
    norm = 1024 * 128  # (x - mean) / (2 std) on the padded 1024 x 128 input
    macs = resample + frames * per + norm
    return {"file_sr": file_sr, "frames": frames, "macs": macs, "bitops_fp32": macs * 1024.0}


def wall_clock(reps=5):
    from torchvision.models import efficientnet_b3
    res = {}
    m = efficientnet_b3(weights=None)
    m.classifier[1] = nn.Linear(1536, 1536)
    m.eval()
    x = torch.rand(1, 3, 300, 300)
    with torch.no_grad():
        for _ in range(2):
            m(x)
        t = []
        for _ in range(reps):
            t0 = time.perf_counter(); m(x); t.append(time.perf_counter() - t0)
    res["efficientnet_b3_300_s"] = t
    from transformers import ASTForAudioClassification
    a = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593",
                                                  revision="f826b80d28226b62986cc218e5cec390b1096902").eval()
    xa = torch.randn(1, 1024, 128)
    with torch.no_grad():
        for _ in range(1):
            a(xa)
        t = []
        for _ in range(reps):
            t0 = time.perf_counter(); a(xa); t.append(time.perf_counter() - t0)
    res["ast_1024_s"] = t
    return res


def main(out):
    R = {}
    e = eff_counts()
    claimed = json.loads((I6 / "costs.json").read_text())
    R["efficientnet_b3"] = {**e, "claimed_macs": claimed["efficientnet_b3"]["macs"],
                            "rel_diff": e["macs_flopcounter"] / claimed["efficientnet_b3"]["macs"] - 1,
                            "bitops_fp32": e["macs_flopcounter"] * 1024, "bitops_int8": e["macs_flopcounter"] * 64,
                            "int8_per_channel_rescale_bitops": e["elementwise_and_outputs"]["conv_linear_outputs"] * 1024.0}
    print("EfficientNet", json.dumps(R["efficientnet_b3"], indent=1), flush=True)
    # AST vs sidecars
    chk = {}
    for ds in ["esc50", "urban8k"]:
        for j in sorted((S2 / ds).glob("*__clean.json")):
            s = json.loads(j.read_text())
            cfg = s["config"]
            sc = s["backbone_cost_per_clip"]
            if isinstance(sc, str):
                import ast as _a
                sc = _a.literal_eval(sc)
            chk[f"{ds}/{cfg}"] = {"mine_view_i": ast_bitops(cfg), "sidecar": float(sc["bitops"]),
                                  "mine_view_ii": ast_bitops(cfg, rescale_price=1024)}
    R["ast_vs_sidecar"] = chk
    R["ast_max_rel_diff_view_i"] = max(abs(v["mine_view_i"] / v["sidecar"] - 1) for v in chk.values())
    print("AST max rel diff to sidecars", R["ast_max_rel_diff_view_i"], flush=True)
    # flag: shorter AST input
    R["ast_short_input_flag"] = {
        f"{fr}_frames": {"patches_tokens": ast_shape(fr),
                         **{cfg: {"view_i": ast_bitops(cfg, fr), "view_ii": ast_bitops(cfg, fr, 1024)}
                            for cfg in ["w4g128", "w4", "w8g64", "fp32"]}} for fr in [1024, 512, 400]}
    # front ends
    R["front_ends"] = {"mel_efficientnet": [mel_front_end(sr, 5.0) for sr in (22050, 44100)],
                       "fbank_ast_esc50": fbank_front_end(44100, 5.0),
                       "fbank_ast_urban_4s_44k": fbank_front_end(44100, 4.0),
                       "mel_efficientnet_urban_4s": [mel_front_end(sr, 4.0) for sr in (22050, 44100)],
                       # the ESC-50 rendering script in git history (c26abba, esc50_dataset.py): native rate, hop 32
                       "mel_efficientnet_hop32_esc50_5s": mel_front_end(44100, 5.0, hop=32),
                       "mel_efficientnet_hop32_4s_44k": mel_front_end(44100, 4.0, hop=32)}
    print("front ends", json.dumps(R["front_ends"], indent=1), flush=True)
    R["wall_clock_fp32_cpu_2threads"] = wall_clock()
    print("wall clock", R["wall_clock_fp32_cpu_2threads"], flush=True)
    Path(out).write_text(json.dumps(R, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
