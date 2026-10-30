"""Check 3: nesting, weight quality, and 8-bit activations, on the real AST weights
(MIT/ast-finetuned-audioset-10-10-0.4593 @ f826b80) and real ESC-50 clips, CPU only.

Part A (weights): my own w8g64 codes for every fused encoder matrix; compare Stage 2's
quantize_weight(W, n3g64) with (i) the rounded child of my codes, (ii) the literal top 3
bit-planes (arithmetic shift), (iii) round-half-up from the top 4 planes. Relative weight
error of several configurations.
Part B (activations): one clip through Stage 2's QuantAST at n3g64 and w8g64 with hooks
that count distinct levels per quantization row of every integer-product input.
Part C (arrays): N clips through QuantAST on CPU for fp32, w8g64, n3g64, w3g64, w4g128 and
the configurations Stage 2 did not run (w3g32, w4g64); relative error to the stored MPS
embeddings and to FP32.

    python check3_nesting.py --stage2 STAGE2_DIR --repo WORKTREE --out OUT [--n-clips 16]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--stage2", type=Path, required=True)
ap.add_argument("--repo", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--n-clips", type=int, default=16)
args = ap.parse_args()
sys.path.insert(0, str(args.repo))
sys.path.insert(0, str(args.repo / "scripts/redesign/idea5"))
torch.set_num_threads(2)

import QWave.backbone_quant as bq  # noqa: E402
from transformers import ASTFeatureExtractor, ASTModel  # noqa: E402

MODEL, REV = "MIT/ast-finetuned-audioset-10-10-0.4593", "f826b80d28226b62986cc218e5cec390b1096902"
model = ASTModel.from_pretrained(MODEL, revision=REV).eval()
res = {}

# ---- Part A ------------------------------------------------------------------
mats = []
for layer in model.encoder.layer:
    att = layer.attention.attention
    mats += [torch.cat([att.query.weight, att.key.weight, att.value.weight]).detach(),
             layer.attention.output.dense.weight.detach(), layer.intermediate.dense.weight.detach(),
             layer.output.dense.weight.detach()]


def my_codes(w, bits, g):
    o, i = w.shape
    wg = w.reshape(o, i // g, g) if g else w.reshape(o, 1, i)
    qmax = 2 ** (bits - 1) - 1
    s = wg.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
    return torch.clamp(torch.round(wg / s), -qmax, qmax), s


n_tot = n_eq = n_trunc_diff = n_hu_diff = n_clamp = 0
max_abs = 0.0
for w in mats:
    c8, s8 = my_codes(w.double(), 8, 64)
    theirs = bq.quantize_weight(w.double(), bq.parse_config("n3g64")[0]).reshape(c8.shape)
    child_round = torch.clamp(torch.round(c8 / 32), -4, 3)
    mine = child_round * s8 * 32
    max_abs = max(max_abs, float((theirs - mine).abs().max()))
    c8i = c8.to(torch.int64)
    child_trunc = torch.div(c8i, 32, rounding_mode="floor")             # top 3 planes of the two's complement code
    child_hu = torch.clamp(torch.div(torch.div(c8i, 16, rounding_mode="floor") + 1, 2, rounding_mode="floor"), -4, 3)
    n_tot += c8.numel()
    n_trunc_diff += int((child_trunc != child_round.to(torch.int64)).sum())
    n_hu_diff += int((child_hu != child_round.to(torch.int64)).sum())
    n_clamp += int((torch.round(c8 / 32) > 3).sum())
res["A_weights"] = {"n_weights": n_tot, "max_abs_diff_stage2_vs_my_rounded_child": max_abs,
                    "frac_child_differs_from_top3_planes": n_trunc_diff / n_tot,
                    "frac_child_differs_from_round_half_up_top4_planes": n_hu_diff / n_tot,
                    "frac_clamped_at_plus3": n_clamp / n_tot}

relerr = {}
for name in ["w8", "w8g128", "w8g64", "w6", "w5", "w4", "n4", "w4g128", "n4g128", "w4g64", "w3g64", "n3g64",
             "w3g32", "w3g128", "n3g128"]:
    spec, _ = bq.parse_config(name)
    num = den = 0.0
    for w in mats:
        q = bq.quantize_weight(w, spec)
        num += float(((q - w) ** 2).sum())
        den += float((w ** 2).sum())
    relerr[name] = (num / den) ** 0.5
res["A_relative_weight_error"] = relerr
print(json.dumps(res, indent=1), flush=True)

# ---- Part B ------------------------------------------------------------------
from audio_shift import index_wavs, load_clip  # noqa: E402

meta = pd.read_csv(Path.home() / "Documents/DATASETS/ESC-50/ast_768/esc-50.csv", usecols=["folder", "name", "label"])
stems = meta["name"].map(lambda s: Path(s).stem).tolist()
audio_dir = Path.home() / "Documents/DATASETS/ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio"
index = index_wavs(audio_dir)
fe = ASTFeatureExtractor.from_pretrained(MODEL, revision=REV)
rng = np.random.default_rng(0)
rows = np.sort(rng.choice(len(stems), args.n_clips, replace=False))
feats = fe([load_clip(index[stems[i]]) for i in rows], sampling_rate=16000, return_tensors="pt")["input_values"]

bq.USE_METAL = False
qast = bq.QuantAST(model)
levels = {"linear_in": 0.0, "qk": 0.0, "v": 0.0, "p": 0.0, "linear_in_q63": 0.0}
orig_linear, orig_matmul = torch.nn.functional.linear, torch.matmul


def off_grid(x, dim, qmax):
    """max over rows of max |x/s - round(x/s)| with s = max|row| / qmax along `dim`: 0 when
    every row sits on a symmetric grid of qmax steps (8-bit: qmax 127)."""
    x = x.detach().double().movedim(dim, -1).reshape(-1, x.shape[dim])
    s = x.abs().amax(1, keepdim=True).clamp_min(1e-30) / qmax
    return float((x / s - torch.round(x / s)).abs().max())


def lin(x, w, b=None):
    levels["linear_in"] = max(levels["linear_in"], off_grid(x, -1, 127))
    levels["linear_in_q63"] = max(levels["linear_in_q63"], off_grid(x, -1, 63))
    return orig_linear(x, w, b)


def mm(a, b):
    if a.dim() == 4 and b.dim() == 4:
        if b.shape[-2] == a.shape[-1] and b.shape[-1] == a.shape[-2]:     # Q K^T
            levels["qk"] = max(levels["qk"], off_grid(a, -1, 127), off_grid(b, -2, 127))
        else:                                                                   # P V
            levels["p"] = max(levels["p"], off_grid(a, -1, 65535))
            levels["v"] = max(levels["v"], off_grid(b, -2, 127))
    return orig_matmul(a, b)


bq.F.linear = lin
torch.matmul = mm
act = {}
for name in ["n3g64", "w8g64"]:
    for k in levels:
        levels[k] = 0
    qast.set_config(*bq.parse_config(name))
    qast(feats[:1])
    act[name] = dict(levels)
bq.F.linear = orig_linear
torch.matmul = orig_matmul
res["B_max_off_grid"] = act
res["B_note"] = ("max distance of any input of an integer product from its 8-bit grid (P: 16-bit grid); "
                 "about 0 means quantized; linear_in_q63 is the same test on a 7-bit grid and should be large")
print(json.dumps(res["B_max_off_grid"]), flush=True)

# ---- Part C ------------------------------------------------------------------
stored = {}
for c in ["fp32", "w8g64", "n3g64", "w3g64", "w4g128", "w8"]:
    stored[c] = np.load(args.stage2 / "emb/esc50" / f"{c}__clean.npy")[rows]
cpu, t0 = {}, time.time()
for c in ["fp32", "w8g64", "n3g64", "w3g64", "w4g128", "w8", "w3g32", "w4g64"]:
    qast.set_config(*bq.parse_config(c))
    cpu[c] = torch.cat([qast(feats[b:b + 4]) for b in range(0, len(feats), 4)]).numpy()
    print(c, f"{time.time() - t0:.0f}s", flush=True)


def rel(a, b):
    return float(np.mean(np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)))


res["C_rows"] = rows.tolist()
res["C_cpu_vs_stored_mps"] = {c: rel(cpu[c], stored[c]) for c in stored}
res["C_rel_err_to_fp32_stored"] = {c: rel(stored[c], stored["fp32"]) for c in stored}
res["C_rel_err_to_fp32_cpu"] = {c: rel(cpu[c], cpu["fp32"]) for c in cpu}
# whole-array relative error to FP32 for the stored configurations (2000 clean clips)
full = {}
f32 = np.load(args.stage2 / "emb/esc50/fp32__clean.npy")
for c in ["w8", "w8g128", "w8g64", "w6", "w5", "w4", "n4", "w4g128", "n4g128", "w3g64", "n3g64", "w3g128", "n3g128"]:
    p = args.stage2 / "emb/esc50" / f"{c}__clean.npy"
    if p.exists():
        full[c] = rel(np.load(p), f32)
res["C_rel_err_to_fp32_all_2000_clean"] = full
csv = pd.read_csv(Path.home() / "Documents/DATASETS/ESC-50/ast_768/esc-50.csv")
X = csv[[str(i) for i in range(768)]].to_numpy(np.float32)
res["C_stage2_fp32_clean_vs_training_csv"] = rel(f32, X)
args.out.mkdir(parents=True, exist_ok=True)
(args.out / "check3_nesting.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k.startswith("C_")}, indent=1))
