"""Checks 3, 4 and 5 on real clips (CPU, 2 threads, a handful of clips).

4: the W8A8 EfficientNet (the claim's QuantEffNet, imported read only from the worktree) with my own forward
   hooks: is every Conv2d weight on a per-output-channel 8-bit grid, and is every Conv2d input on an 8-bit
   grid with one scale per pixel (1 x 1), per channel (depthwise) or per clip (stem)? Recompute stored
   embedding rows from audio (clean, white 0 dB, background 10 dB) on the CPU and compare.
3: the shortened AST: patch embeddings of the kept tokens equal the 1024-frame model's; FP32 embeddings at
   1024 and 512 / 400 frames recomputed from audio and compared with the stored rows.
5: names checksum of the cheap tier, short AST and idea 5 Stage 2 sidecars; NaN pattern of the UrbanSound8K
   short-AST field cells against my field assignment.
Writes outputs/check_models.json.
"""
from __future__ import annotations

import json
import os
import sys
import zlib
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import CELLS, OUT, R, S2, field_of  # noqa: E402

W = Path("/Users/cajas.sebastian/Desktop/repositories/curious-qmoe-redesign-idea6-stage2")
sys.path.insert(0, str(W))
sys.path.insert(0, str(W / "scripts/redesign/idea6/stage2"))
from QWave.effnet_quant import (MelImageSpec, act_scale_dims, build_backbone, efficientnet_b3_pooled,  # noqa: E402
                                mel_images)
from audio_shift import NoiseMaker, index_wavs, load_clip, mix  # noqa: E402

torch.set_num_threads(2)
E = Path.home() / "Documents/DATASETS"
DS = {
    "esc50": dict(csv=E / "ESC-50/ast_768/esc-50.csv", audio=E / "ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio",
                  bg_csv=E / "Urban8k/ast_768/urbansound8k.csv",
                  bg_audio=E / "Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio", L=512),
    "urban8k": dict(csv=E / "Urban8k/ast_768/urbansound8k.csv",
                    audio=E / "Urban8k/audio_raw/urbansound8k_zenodo_1203745/UrbanSound8K/audio",
                    bg_csv=E / "ESC-50/ast_768/esc-50.csv",
                    bg_audio=E / "ESC-50/audio_raw/esc50_master_zip/ESC-50-master/audio", L=400),
}
res = {}


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


# ── check 5: names checksums and field NaN pattern ─────────────────────────
for ds, d in DS.items():
    stems = pd.read_csv(d["csv"], usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist()
    crc = zlib.crc32("\n".join(stems).encode())
    cr = {"csv": crc}
    for p in sorted((R / "emb" / ds).glob("*__clean.json")):
        cr[f"eff:{p.stem}"] = json.loads(p.read_text())["names_crc32"]
    for p in sorted((R / "emb_ast_short" / ds).glob("*__clean.json")):
        cr[f"short:{p.stem}"] = json.loads(p.read_text())["names_crc32"]
    for p in sorted((S2 / "emb" / ds).glob("*__clean.json"))[:3]:
        cr[f"s2:{p.stem}"] = json.loads(p.read_text())["names_crc32"]
    res[f"crc_{ds}"] = {"all_equal": len(set(cr.values())) == 1, "values": cr}
    if ds == "urban8k":
        n = len(stems)
        at, av = field_of(n, 5001), field_of(n, 5002)
        pat = {}
        for c in CELLS:
            f = R / "emb_ast_short/urban8k" / f"n4__{c}.npy"
            X = np.load(f, mmap_mode="r")
            have = np.isfinite(np.asarray(X[:, 0]))
            need = (at == c) | (av == c)
            pat[c] = {"rows_present": int(have.sum()), "rows_needed": int(need.sum()),
                      "present_equals_needed": bool(np.array_equal(have, need)) if have.sum() < n else "full",
                      "needed_subset_of_present": bool(np.all(have[need]))}
        res["urban_short_field_rows"] = pat
        # same-audio: the assignment code of idea 5 Stage 2 is byte-identical (sha256 checked in README of audit)

# ── check 4 and 5: cheap tier from audio ────────────────────────────────────
base = efficientnet_b3_pooled()
fp32 = build_backbone("fp32", base, "cpu")
w8 = build_backbone("w8a8", base, "cpu")
spec = MelImageSpec()

# hooks on the quantized copy: weights and the (already fake-quantized) inputs each Conv2d receives
grid = {"weight_max_dev": 0.0, "weight_max_code": 0.0, "act_max_dev": 0.0, "act_max_code": 0.0, "n_conv": 0,
        "scale_rule": {}, "act_levels_max": 0}
conv_inputs = []


def fwd_hook(mod, inp, out):
    x = inp[0].detach().double()
    dims = act_scale_dims(mod)
    s = x.abs().amax(dim=dims, keepdim=True) / 127
    q = torch.where(s > 0, x / torch.where(s > 0, s, torch.ones_like(s)), torch.zeros_like(x))
    dev = float((q - q.round()).abs().max())
    grid["act_max_dev"] = max(grid["act_max_dev"], dev)
    grid["act_max_code"] = max(grid["act_max_code"], float(q.abs().max()))
    key = "stem" if mod.kernel_size == (3, 3) and mod.groups == 1 else (
        "depthwise" if mod.groups > 1 else "1x1")
    grid["scale_rule"].setdefault(key, set()).add(str(dims))
    # levels actually used per scale slice (at most 255)
    grid["act_levels_max"] = max(grid["act_levels_max"], int(torch.unique(q.round()).numel()))


for mod in w8.net.modules():
    if isinstance(mod, nn.Conv2d):
        w = mod.weight.detach().double()
        s = w.abs().amax(dim=(1, 2, 3), keepdim=True) / 127
        q = w / s
        grid["weight_max_dev"] = max(grid["weight_max_dev"], float((q - q.round()).abs().max()))
        grid["weight_max_code"] = max(grid["weight_max_code"], float(q.abs().max()))
        grid["n_conv"] += 1
        mod.register_forward_hook(fwd_hook)

rng = np.random.default_rng(31)
emb_cmp = []
ast_cmp = []
for ds, d in DS.items():
    stems = pd.read_csv(d["csv"], usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist()
    index = index_wavs(d["audio"])
    rows = sorted(rng.choice(len(stems), 2, replace=False).tolist())
    if ds == "urban8k":
        L = np.load(R / "emb/urban8k/lengths__clean.npy")
        rows.append(int(np.argmax(L)))  # the longest clip (more than 400 fbank frames)
    bg_names = pd.read_csv(d["bg_csv"], usecols=["name"])["name"].map(lambda s: Path(s).stem).tolist()
    bg_index = index_wavs(d["bg_audio"])
    nm = NoiseMaker([b for b in bg_names if b in bg_index], bg_index, False)
    for cond in ("clean", "white__0db", "bg__10db"):
        waves = []
        for i in rows:
            x = load_clip(index[stems[i]])
            if cond != "clean":
                kind, snr = cond.split("__")
                x, _ = mix(x, nm(stems[i], kind, len(x)), float(snr[:-2]))
            waves.append(x)
        imgs = mel_images(waves, spec)
        with torch.no_grad():
            e32 = fp32(imgs).numpy()
            e8 = w8(imgs).numpy()
        for j, i in enumerate(rows):
            s32 = np.load(R / f"emb/{ds}/fp32__{cond}.npy", mmap_mode="r")[i]
            s8 = np.load(R / f"emb/{ds}/w8a8__{cond}.npy", mmap_mode="r")[i]
            emb_cmp.append(dict(ds=ds, cond=cond, row=i, stem=stems[i], fp32_rel_vs_stored=rel(e32[j], s32),
                                w8a8_rel_vs_stored=rel(e8[j], s8), w8a8_rel_vs_fp32=rel(e8[j], e32[j]),
                                stored_w8a8_rel_vs_stored_fp32=rel(np.asarray(s8), np.asarray(s32))))
        # AST at full and short length on the first two rows, clean and white 0 dB only
        if cond == "bg__10db":
            continue
        from transformers import ASTFeatureExtractor, ASTModel
        from QWave.ast_frames import shorten_ast
        mname, rev = "MIT/ast-finetuned-audioset-10-10-0.4593", "f826b80d28226b62986cc218e5cec390b1096902"
        fx_full = ASTFeatureExtractor.from_pretrained(mname, revision=rev)
        fx_s = ASTFeatureExtractor.from_pretrained(mname, revision=rev, max_length=d["L"])
        m_full = ASTModel.from_pretrained(mname, revision=rev).eval()
        m_s = shorten_ast(ASTModel.from_pretrained(mname, revision=rev).eval(), d["L"])
        ww = waves[:2] if ds == "esc50" else [waves[0], waves[-1]]
        rr = rows[:2] if ds == "esc50" else [rows[0], rows[-1]]
        xf = fx_full(ww, sampling_rate=16000, return_tensors="pt")["input_values"]
        xs = fx_s(ww, sampling_rate=16000, return_tensors="pt")["input_values"]
        with torch.no_grad():
            pe_f = m_full.embeddings(xf)
            pe_s = m_s.embeddings(xs)
            nt_f, nt_s = (1024 - 16) // 10 + 1, (d["L"] - 16) // 10 + 1
            kept = torch.cat([torch.tensor([0, 1]), (torch.arange(12 * nt_f).reshape(12, nt_f)[:, :nt_s].reshape(-1) + 2)])
            # patches whose 16 frames lie inside the clip's real frames are identical by construction
            ef = m_full(xf).pooler_output.numpy()
            es = m_s(xs).pooler_output.numpy()
        real_frames = [1 + (len(w) - 400) // 160 for w in ww]
        for j, i in enumerate(rr):
            n_real = min(real_frames[j], d["L"])
            ok_t = [t for t in range(nt_s) if 10 * t + 16 <= d["L"]]
            idx = [0, 1] + [2 + f * nt_s + t for f in range(12) for t in ok_t]
            idx_f = [0, 1] + [2 + f * nt_f + t for f in range(12) for t in ok_t]
            pd_ = float((pe_s[j, idx] - pe_f[j, idx_f]).abs().max())
            s_full = np.load(S2 / f"emb/{ds}/fp32__{cond}.npy", mmap_mode="r")[i]
            s_short = np.load(R / f"emb_ast_short/{ds}/fp32__{cond}.npy", mmap_mode="r")[i]
            ast_cmp.append(dict(ds=ds, cond=cond, row=i, real_frames=real_frames[j], max_length=d["L"],
                                frames_not_covered_by_patches=max(0, real_frames[j] - (10 * (nt_s - 1) + 16)),
                                patch_plus_pos_max_abs_diff_kept_tokens=pd_,
                                full_rel_vs_stored_s2=rel(ef[j], s_full), short_rel_vs_stored=rel(es[j], s_short),
                                short_vs_full_rel=rel(es[j], ef[j]),
                                stored_short_vs_stored_full_rel=rel(np.asarray(s_short), np.asarray(s_full))))
        del m_full, m_s

grid["scale_rule"] = {k: sorted(v) for k, v in grid["scale_rule"].items()}
res["w8a8_grid"] = grid
res["emb_from_audio"] = emb_cmp
res["ast_from_audio"] = ast_cmp
(OUT / "check_models.json").write_text(json.dumps(res, indent=1, default=str))
print(json.dumps(res, indent=1, default=str))
