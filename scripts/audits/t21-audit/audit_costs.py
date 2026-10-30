"""Independent per-clip priced bit-op costs for T21 D1 (pool rows). CPU only.
AST: analytic count from the ViT-B/16 shapes (12 layers, d 768, MLP 3072), own frame and patch rule.
MN10: MACs and output elements counted with my own forward hooks on the EfficientAT MN10 module (model definition
loaded from the worktree; no cost function of theirs is called). Front ends: my own rough per-frame estimate.
Usage: python audit_costs.py <results_dir> <out_dir> <worktree>
"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
R, OUT, WT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
L = np.load(R / "emb/d1/mn_pool/lengths__clean.npy").astype(int)
tp_theirs = np.load(R / "emb/d1/ast_pool/time_patches.npy").astype(int)
# ---- AST ----
fr = 1 + (L - 400) // 160                       # Kaldi fbank, snip edges, 25 ms / 10 ms at 16 kHz
fr = np.minimum(fr, 1024)
t = (fr - 16) // 10 + 1                          # time patches, stride 10
print("my time patches equal theirs:", np.array_equal(t, tp_theirs), int(t.min()), int(t.max()))
P_ = 12 * t; N = P_ + 2; D, H, Lyr, G = 768, 3072, 12, 128
lin = Lyr * N * (4 * D * D + 2 * D * H)
resc = Lyr * N * (4 * D * (D // G) + H * (D // G) + D * (H // G))
att = Lyr * N * N * D                            # one of QK^T or PV
patch = P_ * 256 * D
fb_per_frame = 400 * 2 + 2.5 * 512 * 9 / 2 + 2 * 257 + 257 * 128 + 128
ast_fb = fr * fb_per_frame
head_ast = 768 * 640 + 640 * 320 + 320 * 41
ast_bitops = lin * 4 * 8 + resc * 1024 + att * 8 * 8 + att * 16 * 8 + patch * 32 * 32 + ast_fb * 1024 + head_ast * 1024
ast_macs = lin + 2 * att + patch + ast_fb + head_ast
# ---- MN10 ----
import torch, torch.nn as nn
torch.set_num_threads(2)
sys.path.insert(0, str(WT))
from QWave.efficientat_mn import build_mn   # model definition only
m = build_mn("mn10").eval()
n32 = np.ceil(2 * L).astype(int)
frames = 1 + (n32 - 1) // 320                    # centred STFT, hop 320, after pre-emphasis
cache = {}
def count(fr_):
    acc = [0, 0]
    def hk(mod, i, o):
        if isinstance(mod, nn.Conv2d):
            n_out = o.numel(); acc[0] += n_out * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]; acc[1] += n_out
        else:
            acc[0] += mod.in_features * mod.out_features * (o.numel() // mod.out_features); acc[1] += o.numel()
    hs = [mm.register_forward_hook(hk) for mm in m.features.modules() if isinstance(mm, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        m.features(torch.zeros(1, 1, 128, int(fr_)))
    for h in hs: h.remove()
    return acc
for f_ in np.unique(frames):
    cache[int(f_)] = count(f_)
mn_mac = np.array([cache[int(f_)][0] for f_ in frames], float)
mn_out = np.array([cache[int(f_)][1] for f_ in frames], float)
mn_fb = 32 * n32 + n32 + frames * (800 + 2.5 * 1024 * 10 / 2 + 2 * 513 + 513 * 128 / 4 + 2 * 128)   # resample taps ~32, rough
head_mn = 960 * 640 + 640 * 320 + 320 * 41
mn_bitops = mn_mac * 64 + mn_out * 1024 + mn_fb * 1024 + head_mn * 1024
mn_macs = mn_mac + mn_fb + head_mn
np.save(OUT / "cost_mn_per_clip.npy", mn_bitops); np.save(OUT / "cost_ast_per_clip.npy", ast_bitops)
mine = pd.DataFrame({"mn10_w8a8_bitops": mn_bitops, "ast_w4g128_bitops": ast_bitops, "mn10_macs": mn_macs, "ast_macs": ast_macs}).describe()
theirs = pd.read_csv(R / "tables/d1/costs_per_clip_summary.csv", index_col=0)
cmp = pd.concat({"mine": mine, "theirs": theirs, "ratio_mine_over_theirs": mine / theirs}, axis=1)
pd.set_option("display.width", 250)
print(cmp.loc[["mean", "min", "25%", "50%", "75%", "max"]].to_string())
cmp.to_csv(OUT / "costs_compare.csv")
(OUT / "costs_audit.json").write_text(json.dumps({"mn_mean": mn_bitops.mean(), "ast_mean": ast_bitops.mean(),
    "share_frontend_mn": float((mn_fb * 1024).sum() / mn_bitops.sum()), "share_patch_ast": float((patch * 1024).sum() / ast_bitops.sum())}, indent=1))
print(open(OUT / "costs_audit.json").read())
