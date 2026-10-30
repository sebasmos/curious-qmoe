"""Independent per-clip priced bit-ops for T21 D2 pool rows (CPU only).

AST: analytic count from ViT-B/16 shapes (12 layers, width 768, MLP 3072), my own fbank frame and time-patch rule,
w4g128 priced as 4-bit weights x 8-bit activations, group-128 rescales at 32 x 32, QK^T at 8 x 8, PV at 16 x 8 (16-bit
probabilities), patch embedding and fbank at 32 x 32, FP32 head at 32 x 32.
MN10: MACs and output elements from my own forward hooks on the EfficientAT MN10 module (only the model definition is
loaded from the worktree), W8A8 at 8 x 8, one 32 x 32 rescale per output element, rough front end at 32 x 32.
Usage: python -I audit_costs_d2.py <T21 results> <audit out> <worktree>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R, OUT, WT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
K = 113
L = np.load(R / "emb/d2/mn_pool/lengths__clean.npy").astype(np.int64)
tp_theirs = np.load(R / "emb/d2/ast_pool/time_patches.npy").astype(np.int64)
assert np.array_equal(L, np.load(R / "emb/d2/ast_pool/lengths16k.npy"))
# AST
nfr = np.minimum(1 + (L - 400) // 160, 1024)      # Kaldi fbank 25 ms window, 10 ms hop, snip edges, 16 kHz
t = (nfr - 16) // 10 + 1                           # time patches of a 16-frame patch with stride 10
print("time patches equal theirs:", bool(np.array_equal(t, tp_theirs)), int(t.min()), int(t.max()))
npatch = 12 * t
ntok = npatch + 2
d, h, nl, g = 768, 3072, 12, 128
lin = nl * ntok * (3 * d * d + d * d + d * h + h * d)              # qkv, out proj, fc1, fc2
resc = nl * ntok * (3 * d * (d // g) + d * (d // g) + h * (d // g) + d * (h // g))
att = nl * ntok * ntok * d                                         # QK^T (and the same for PV)
patch = npatch * 16 * 16 * d
fb = nfr * (400 + 512 * 9 * 2.5 + 2 * 257 + 257 * 128 + 128)       # window, real FFT, power, mel, log (rough)
head_ast = 768 * 640 + 640 * 320 + 320 * K
ast_bit = lin * 4 * 8 + resc * 32 * 32 + att * 8 * 8 + att * 16 * 8 + patch * 32 * 32 + fb * 32 * 32 + head_ast * 32 * 32
ast_mac = lin + 2 * att + patch + fb + head_ast
# MN10
import torch
import torch.nn as nn

torch.set_num_threads(2)
sys.path.insert(0, str(WT))
from QWave.efficientat_mn import build_mn  # noqa: E402  model definition only

net = build_mn("mn10").eval()
n32 = np.ceil(2 * L).astype(np.int64)
mfr = 1 + n32 // 320                                                # centred STFT, hop 320 at 32 kHz
memo = {}


def hooks_count(T):
    acc = [0.0, 0.0]

    def hk(mod, i, o):
        if isinstance(mod, nn.Conv2d):
            acc[0] += o.numel() * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
        else:
            acc[0] += mod.in_features * mod.out_features * (o.numel() // mod.out_features)
        acc[1] += o.numel()
    hs = [m.register_forward_hook(hk) for m in net.features.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        net.features(torch.zeros(1, 1, 128, int(T)))
    for x in hs:
        x.remove()
    return acc


for T in np.unique(mfr):
    memo[int(T)] = hooks_count(T)
mn_core = np.array([memo[int(T)][0] for T in mfr])
mn_out = np.array([memo[int(T)][1] for T in mfr])
mn_fe = 25 * n32 + n32 + mfr * (800 + 1024 * 10 * 2.5 + 2 * 513 + 513 * 128 / 4 + 2 * 128)  # resample, pre-emphasis, STFT, mel (rough)
head_mn = 960 * 640 + 640 * 320 + 320 * K
mn_bit = mn_core * 8 * 8 + mn_out * 32 * 32 + mn_fe * 32 * 32 + head_mn * 32 * 32
mn_mac = mn_core + mn_fe + head_mn
np.save(OUT / "cost_ast_per_clip.npy", ast_bit)
np.save(OUT / "cost_mn_per_clip.npy", mn_bit)
mine = pd.DataFrame({"mn10_w8a8_bitops": mn_bit, "ast_w4g128_bitops": ast_bit, "mn10_macs": mn_mac, "ast_macs": ast_mac}).describe()
theirs = pd.read_csv(R / "tables/d2/costs_per_clip_summary.csv", index_col=0)
cmp = pd.concat({"mine": mine, "theirs": theirs, "ratio": mine / theirs}, axis=1)
cmp.to_csv(OUT / "costs_compare.csv")
pd.set_option("display.width", 250)
print(cmp.loc[["mean", "min", "50%", "max"]].T.to_string())
side = json.loads((R / "emb/d2/ast_pool/w4g128__clean.json").read_text())
print("their sidecar AST backbone bit-ops mean", side["backbone_bitops_mean"], "fbank", side["fbank_bitops_mean"])
print("my AST backbone (no fbank, no head) mean", float((ast_bit - fb * 1024 - head_ast * 1024).mean()))
print("head bit-ops mine: mn", head_mn * 1024, "ast", head_ast * 1024)
print("AST cost max/min ratio", float(ast_bit.max() / ast_bit.min()))
